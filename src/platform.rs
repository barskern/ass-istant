use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use futures::{Stream, StreamExt};
use ollama_rs::generation::chat::ChatMessage as OllamaChatMessage;
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{Instrument, debug, error, info, info_span, trace, warn};

use crate::impersonator::Impersonator;
use crate::utils::human_message_duration;

pub mod discord;
pub mod mattermost;

pub use types::{ChatId, ChatIdRef};

const CMD_PREFIX: &str = "hey slave,";
const AI_PREFIX: &str = "LLM: ";

pub trait Platform {
    fn run(&mut self, cancel: CancellationToken) -> impl Future<Output = Result<()>> + Send;

    fn attach_event_extractor(&self, sender_tx: mpsc::Sender<ChatEvent>);

    fn common_chat_config(&self, chat_id: &ChatIdRef) -> Option<&ChatConfig>;

    fn fetch_chat_histories(
        &self,
    ) -> impl Stream<Item = Result<(ChatId, Vec<OllamaChatMessage>)>> + Send;

    fn send_typing_status(&self, chat_id: &ChatIdRef) -> impl Future<Output = ()> + Send;
    fn send_message(
        &self,
        chat_id: &ChatIdRef,
        message: &str,
    ) -> impl Future<Output = Result<()>> + Send;
}

#[derive(Default)]
struct InflightState {
    cancel: CancellationToken,
}

pub struct Manager<M: Platform> {
    inner: M,
    impersonator: Arc<Impersonator>,
    inflight_responses: Arc<DashMap<ChatId, InflightState>>,
    typing_tasks: TaskTracker,
    processing_tasks: TaskTracker,
}

impl<M: Platform + Clone> Clone for Manager<M> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            impersonator: Arc::clone(&self.impersonator),
            inflight_responses: Arc::clone(&self.inflight_responses),
            typing_tasks: self.typing_tasks.clone(),
            processing_tasks: self.processing_tasks.clone(),
        }
    }
}

impl<M: Platform> Manager<M> {
    pub fn new(inner: M, impersonator: Arc<Impersonator>) -> Self {
        Self {
            inner,
            impersonator,
            inflight_responses: Default::default(),
            typing_tasks: TaskTracker::new(),
            processing_tasks: TaskTracker::new(),
        }
    }
}

impl<M: Platform + Sync + Send + Clone + 'static> Manager<M> {
    pub async fn run(&mut self, cancel: CancellationToken) {
        let tasks = TaskTracker::new();

        tasks.spawn({
            let this = self.to_owned();
            let cancel = cancel.clone();

            cancel
                .run_until_cancelled_owned(async move {
                    this.inner
                        .fetch_chat_histories()
                        .for_each(|res| {
                            let this = this.clone();
                            let impersonator = Arc::clone(&this.impersonator);

                            let chat_id = res
                                .as_ref()
                                .map(|(chat_id, _)| tracing::field::debug(chat_id))
                                .ok();

                            let chat_span = info_span!("chat", chat_id);
                            async move {
                                match res {
                                    Ok((chat_id, raw_messages)) => {
                                        let Some(chat_config) =
                                            this.inner.common_chat_config(chat_id.as_ref())
                                        else {
                                            return;
                                        };

                                        let history = impersonator
                                            .new_blank_history(chat_id.as_ref(), chat_config)
                                            .chain(raw_messages)
                                            .collect();

                                        impersonator
                                            .init_chat_history(chat_id.as_ref(), history)
                                            .await;
                                    }
                                    Err(e) => {
                                        error!("failed to fetch chat history for chat: {e:?}")
                                    }
                                }
                            }
                            .instrument(chat_span)
                        })
                        .await;
                })
                .in_current_span()
        });

        tasks.spawn({
            let this = self.clone();
            let cancel = cancel.clone();
            let clean_history_interval = Duration::from_secs(60 * 60);

            cancel
                .run_until_cancelled_owned(async move {
                    loop {
                        sleep(clean_history_interval).await;
                        this.impersonator.clean_histories().await;
                    }
                })
                .in_current_span()
        });

        // TODO How to log without cloning message?
        //let logger_tx = start_event_logger();

        tasks.spawn({
            let this = self.clone();
            let cancel = cancel.clone();

            let (events_tx, mut events_rx) = mpsc::channel(50);
            this.inner.attach_event_extractor(events_tx);

            cancel
                .run_until_cancelled_owned(async move {
                    while let Some(chat_event) = events_rx.recv().await {
                        this.process_event(chat_event).await;
                        //if let Err(e) = logger_tx.try_send(ev.clone()) {
                        //    warn!("failed to log event: {e:?}");
                        //}
                    }
                })
                .in_current_span()
        });

        if let Err(e) = self.inner.run(cancel.clone()).await {
            error!("failed while running platform: {e:?}");
        }

        let n_inflight_responses = self.inflight_responses.len();
        if n_inflight_responses > 0 {
            info!("had {n_inflight_responses} inflight response(s) on shutdown, cancelling...");
            for elem in self.inflight_responses.iter() {
                elem.cancel.cancel();
            }
            self.inflight_responses.clear();
        }

        tasks.close();
        tasks.wait().await;
        self.typing_tasks.close();
        self.typing_tasks.wait().await;
        self.processing_tasks.close();
        self.processing_tasks.wait().await;
    }

    async fn maybe_respond_to_chat(
        &self,
        chat_id: &ChatIdRef,
        cancel: CancellationToken,
    ) -> Result<()> {
        let Some(chat_config) = self.inner.common_chat_config(chat_id) else {
            return Err(anyhow!("did not find chat config"));
        };

        let start = Instant::now();

        // Just some heuristics to add some realism to the responses
        let write_chars_per_sec = 15.0;
        let read_chars_per_sec = 30.0;
        let max_answer_time = Duration::from_secs(3 * 60);
        let max_deliberation_time = Duration::from_secs(3 * 60);
        let typing_notification_interval = Duration::from_secs(4);
        let time_to_notice_message = Duration::from_secs(5); // Calculate based on time since previous message

        // TODO Fix deliberation_time based on char length latest N messages from user
        let unread_chars = self
            .impersonator
            .length_of_unanswered_messages(chat_id)
            .await
            // Some sensible default of some sort..
            .unwrap_or(30);

        let deliberation_time = (human_message_duration(unread_chars, read_chars_per_sec)
            + time_to_notice_message)
            .min(max_deliberation_time);

        if chat_config.should_detect_natural_end {
            debug!("starting to detect natural end..");
            let maybe_warrants_reply = cancel
                .run_until_cancelled(self.impersonator.warrants_reply(chat_id, chat_config))
                .await;

            let Some(warrants_reply) = maybe_warrants_reply else {
                return Ok(());
            };

            if !warrants_reply
                .inspect_err(|e| warn!("failed to find natural end: {e:?}"))
                .unwrap_or(true)
            {
                debug!("conversation does not warrant reply, stopping");
                return Ok(());
            }
        }

        let typing_cancel = cancel.child_token();
        let typing_guard = typing_cancel.clone().drop_guard();
        self.typing_tasks.spawn({
            let this = self.clone();
            let typing_cancel = typing_cancel.clone();
            let chat_id = chat_id.to_owned();

            typing_cancel
                .run_until_cancelled_owned(async move {
                    sleep(deliberation_time).await;
                    loop {
                        trace!("sending typing notification...");
                        this.inner.send_typing_status(chat_id.as_ref()).await;

                        // TODO Add some noise to make it more realistic?
                        sleep(typing_notification_interval).await;
                    }
                })
                .in_current_span()
        });

        debug!("starting to generate a response...");
        let maybe_chat_response = cancel
            .run_until_cancelled(self.impersonator.generate_a_response(chat_id, chat_config))
            .await;

        let Some(chat_response) = maybe_chat_response else {
            debug!("response was cancelled, not sending anything..");
            return Ok(());
        };

        let chat_response = chat_response.context("generating a response")?;

        let elapsed = start.elapsed();
        let answer_time = (deliberation_time
            + human_message_duration(chat_response.content.len(), write_chars_per_sec))
        .min(max_answer_time);

        let to_sleep = answer_time.saturating_sub(elapsed);
        if !to_sleep.is_zero() {
            debug!(
                "finished in {elapsed:?}, human answer time should be ish {answer_time:?}, so waiting a bit.."
            );
            let Some(_) = cancel.run_until_cancelled(sleep(to_sleep)).await else {
                debug!("cancelled while waiting to send answer, so cancelled sending...");
                return Ok(());
            };
        } else {
            debug!(
                "finished in {elapsed:?}, human answer time should be ish {answer_time:?}, so we were too slow.."
            );
        }

        let text_message = chat_config.postprocess_message(chat_response.content);
        // TODO The following three actions seem racy, ensure they work as intended..
        {
            self.inflight_responses.remove(chat_id);
        }
        // Normally cancel cannot happen after the inflight_state has been removed,
        // but if it does happen, it means that cancel was extracted before we were able to
        // remove it, so stop here.
        if cancel.is_cancelled() {
            return Ok(());
        }

        // Stop sending typing notifications just before we send the real message
        drop(typing_guard);

        self.inner
            .send_message(chat_id, &text_message)
            .await
            .context("failed to send message to chat")?;

        Ok(())
    }

    async fn process_event(&self, event: ChatEvent) {
        debug!("processing event: {event:?}");
        match event {
            ChatEvent::Message {
                chat_id,
                content,
                role,
            } => {
                let Some(chat_config) = self.inner.common_chat_config(chat_id.as_ref()) else {
                    warn!("got message from chat '{chat_id:?}' without config");
                    return;
                };

                self.processing_tasks.spawn({
                    let this = self.clone();
                    let chat_id = chat_id.to_owned();
                    let friend_name = &chat_config.friend_name;
                    let mut content = content.to_owned();

                    let chat_span = info_span!("chat", ?chat_id, friend_name);

                    async move {
                        // We have to fetch chat_config again due to lifetimes
                        let Some(chat_config) = this.inner.common_chat_config(chat_id.as_ref()) else {
                            return;
                        };

                        content.message = chat_config.preprocess_message(content.message);

                        match role {
                            ChatRole::Me => {
                                // We (or user) sent a message that should be added to the history
                                // as an assistant message.
                                debug!(
                                    "adding assistant message to history: {:?}",
                                    &content.message
                                );

                                let chat_message = OllamaChatMessage::assistant(content.message);
                                this.impersonator
                                    .commit_to_history(chat_id.as_ref(), chat_config, [chat_message])
                                    .await;
                            }
                            ChatRole::Other => {
                                // Message was from the other user in the chat, so figure out how to respond

                                debug!(
                                    "adding user message to history: {:?}",
                                    &content.message
                                );

                                if let Some((_, inflight_state)) =
                                    this.inflight_responses.remove(&chat_id)
                                {
                                    // TODO Should we unconditionally cancel?? Verify that this works
                                    // even if we already finished and added to history..
                                    inflight_state.cancel.cancel();
                                }
                                // Push new message into the chat history
                                {
                                    let chat_message = OllamaChatMessage::user(content.message);
                                    this.impersonator
                                        .commit_to_history(chat_id.as_ref(), chat_config, [chat_message])
                                        .await;
                                }

                                let cancel = CancellationToken::new();
                                {
                                    let existing = this.inflight_responses.insert(
                                        chat_id.clone(),
                                        InflightState {
                                            cancel: cancel.clone(),
                                        },
                                    );
                                    if let Some(existing) = existing {
                                        // This should not happen (I hope!)
                                        warn!(
                                            "when inserting inflight state, already existing found, cancelling it.."
                                        );
                                        existing.cancel.cancel();
                                    }
                                }

                                if let Err(e) = this.maybe_respond_to_chat(chat_id.as_ref(), cancel).await {
                                    error!("failed when handling maybe response: {e:?}")
                                }
                            }
                        }
                    }.instrument(chat_span)
                    .in_current_span()
                });
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum ChatEvent {
    Message {
        chat_id: ChatId,
        content: ChatMessage,
        role: ChatRole,
    },
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    from_user_id: String,
    message: String,
}

#[derive(Clone, Copy, Debug)]
pub enum ChatRole {
    Me,
    Other,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct ChatConfig {
    #[serde(default)]
    pub should_prefix: bool,
    pub friend_name: String,
    #[serde(default)]
    pub should_detect_natural_end: bool,
    #[serde(default)]
    pub custom_system_prompt: Option<String>,
}

impl ChatConfig {
    pub fn preprocess_message(&self, message: String) -> String {
        let Self { should_prefix, .. } = self;
        if *should_prefix {
            // In a prefixing chat, we remove AI prefix before sending to the LLM
            message.trim_start_matches(AI_PREFIX).trim().to_string()
        } else {
            // In a non-prefixing chat, we remove the command prefix before sending to the LLM
            message.trim_start_matches(CMD_PREFIX).trim().to_string()
        }
    }

    pub fn postprocess_message(&self, message: String) -> String {
        let Self { should_prefix, .. } = self;
        if *should_prefix {
            // In a prefixing chat, we remove AI prefix before sending to the LLM
            format!("{AI_PREFIX}{message}")
        } else {
            // In a non-prefixing chat, do nothing (for now)
            message
        }
    }

    pub fn determine_role(&self, m: &ChatMessage, my_user_id: &str) -> ChatRole {
        let has_ai_prefix = m.message.starts_with(AI_PREFIX);
        let has_cmd_prefix = m.message.starts_with(CMD_PREFIX);
        let from_me = m.from_user_id == my_user_id;

        let is_me = if self.should_prefix {
            // We are in a prefixed chat, all messages with AI prefix is me,
            // all other messages are others
            has_ai_prefix
        } else {
            // We are in a "normal" chat (non-prefixed), all messages from me without
            // command prefix are me, all other messages are others
            from_me && !has_cmd_prefix
        };

        if is_me { ChatRole::Me } else { ChatRole::Other }
    }
}

mod types {
    use std::borrow::Borrow;

    use anyhow::{Result, anyhow};
    use ref_cast::{RefCastCustom, ref_cast_custom};

    #[derive(RefCastCustom, Hash, PartialEq, Eq, Debug)]
    #[repr(transparent)]
    pub struct ChatIdRef(str);

    impl ChatIdRef {
        #[ref_cast_custom]
        const unsafe fn new_unchecked(value: &str) -> &Self;

        pub fn try_new(value: &str) -> Result<&Self> {
            let Some((ns, sub_id)) = value.split_once(':') else {
                return Err(anyhow!("missing ':' in chat id"));
            };
            if ns.is_empty() {
                return Err(anyhow!("namespace is empty in chat id"));
            }
            if sub_id.is_empty() {
                return Err(anyhow!("sub id is empty in chat id"));
            }

            unsafe { Ok(Self::new_unchecked(value)) }
        }

        pub fn platform_name(&self) -> &str {
            self.0.split_once(':').unwrap().0
        }

        pub fn sub_id(&self) -> &str {
            self.0.split_once(':').unwrap().1
        }
    }

    impl ToOwned for ChatIdRef {
        type Owned = ChatId;

        fn to_owned(&self) -> Self::Owned {
            ChatId(self.0.to_owned())
        }
    }

    impl AsRef<str> for ChatIdRef {
        fn as_ref(&self) -> &str {
            &self.0
        }
    }

    #[derive(Hash, PartialEq, Eq, Clone, Debug)]
    pub struct ChatId(String);

    impl ChatId {
        pub fn new(ns: &str, data: &str) -> Self {
            Self(format!("{ns}:{data}"))
        }
    }

    impl Borrow<ChatIdRef> for ChatId {
        fn borrow(&self) -> &ChatIdRef {
            // SAFETY: A chat id is always a valid reference
            unsafe { ChatIdRef::new_unchecked(&self.0) }
        }
    }

    impl AsRef<ChatIdRef> for ChatId {
        fn as_ref(&self) -> &ChatIdRef {
            // SAFETY: A chat id is always a valid reference
            unsafe { ChatIdRef::new_unchecked(&self.0) }
        }
    }
}
