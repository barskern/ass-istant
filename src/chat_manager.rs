use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use axum::http::Method;
use dashmap::DashMap;
use futures::TryFutureExt;
use mattermost_api::prelude::*;
use ollama_rs::generation::chat::ChatMessage;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::{self, Instant};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{Instrument, debug, error, info, info_span, trace, warn};

use crate::impersonator::Impersonator;
use crate::utils::human_message_duration;

const CMD_PREFIX: &str = "hey slave,";
const AI_PREFIX: &str = "LLM: ";

pub enum ChatRole {
    Me,
    Other,
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    // TODO Fetch from /users/me instead
    my_user_id: String,
    chats: HashMap<String, ChatConfig>,
}

impl Config {
    pub fn determine_role(&self, post: &Post) -> Option<ChatRole> {
        let chat_config = self.chats.get(&post.channel_id)?;

        let has_ai_prefix = post.message.starts_with(AI_PREFIX);
        let has_cmd_prefix = post.message.starts_with(CMD_PREFIX);
        let from_me = post.user_id == self.my_user_id;

        let is_me = if chat_config.should_prefix {
            // We are in a prefixed chat, all messages with AI prefix is me,
            // all other messages are others
            has_ai_prefix
        } else {
            // We are in a "normal" chat (non-prefixed), all messages from me without
            // command prefix are me, all other messages are others
            from_me && !has_cmd_prefix
        };

        if is_me {
            Some(ChatRole::Me)
        } else {
            Some(ChatRole::Other)
        }
    }
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct ChatConfig {
    #[serde(default)]
    pub should_prefix: bool,
    pub friend_name: String,
    #[serde(default)]
    pub extra_props: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub should_detect_natural_end: bool,
}

impl ChatConfig {
    pub fn preprocess_message(&self, message: String) -> String {
        let ChatConfig { should_prefix, .. } = self;
        if *should_prefix {
            // In a prefixing chat, we remove AI prefix before sending to the LLM
            message.trim_start_matches(AI_PREFIX).trim().to_string()
        } else {
            // In a non-prefixing chat, we remove the command prefix before sending to the LLM
            message.trim_start_matches(CMD_PREFIX).trim().to_string()
        }
    }

    pub fn postprocess_message(&self, message: String) -> String {
        let ChatConfig { should_prefix, .. } = self;
        if *should_prefix {
            // In a prefixing chat, we remove AI prefix before sending to the LLM
            format!("{AI_PREFIX}{message}")
        } else {
            // In a non-prefixing chat, do nothing (for now)
            message
        }
    }

    pub fn props(&self) -> serde_json::Value {
        let mut map = self.extra_props.as_ref().cloned().unwrap_or_default();
        map.insert("is_clanker".into(), json!(true));
        serde_json::Value::Object(map)
    }
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
pub struct Post {
    message: String,
    user_id: String,
    channel_id: String,
    #[serde(default)]
    props: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct PostEvent {
    channel_display_name: String,
    sender_name: String,
    #[serde(rename = "post", with = "serde_nested_json")]
    content: Post,
}

#[derive(Serialize, Debug)]
struct NewPostRequest<'a> {
    pub channel_id: &'a str,
    pub message: &'a str,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_id: Option<&'a str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub props: Option<serde_json::Value>,
}

impl PostEvent {
    pub fn chat_id(&self) -> &str {
        &self.content.channel_id
    }
}

#[derive(Default)]
struct InflightState {
    cancel: CancellationToken,
}

#[derive(Clone)]
pub struct Manager {
    // TODO Add rate limiting and retrying to the API here!
    api: Mattermost,

    // TODO We probably actually rather want one impersonator per chat instead,
    // only "usecase" for one impersonator is consistency across chats, but
    // thats probably too complex anyway.
    impersonator: Arc<Impersonator>,
    config: Arc<Config>,
    event_tx: mpsc::Sender<WebsocketEvent>,
    inflight_responses: Arc<DashMap<String, InflightState>>,
}

#[derive(Deserialize, Debug)]
pub struct ChannelPostsResponse {
    order: Vec<String>,
    posts: HashMap<String, Post>,
}

impl Manager {
    pub fn new(
        api: Mattermost,
        config: Config,
        impersonator: Impersonator,
        event_tx: mpsc::Sender<WebsocketEvent>,
    ) -> Self {
        Self {
            api,
            impersonator: Arc::new(impersonator),
            config: Arc::new(config),
            event_tx,
            inflight_responses: Default::default(),
        }
    }

    pub async fn init(&self) {
        let tasks = TaskTracker::new();

        tasks.spawn({
            let this = self.clone();
            async move {
                this.init_chat_histories().await;
            }
        });

        tasks.close();
        tasks.wait().await;
    }

    pub fn background_tasks(&self, cancel: CancellationToken) -> impl Future<Output = ()> + use<> {
        let tasks = TaskTracker::new();

        tasks.spawn({
            let this = self.clone();
            let cancel = cancel.clone();
            let clean_history_interval = Duration::from_secs(60 * 60);

            cancel.run_until_cancelled_owned(async move {
                loop {
                    time::sleep(clean_history_interval).await;
                    this.impersonator.clean_histories().await;
                }
            })
        });

        tasks.spawn({
            let this = self.clone();
            let cancel = cancel.clone();
            async move {
                loop {
                    debug!("starting websocket event handler");
                    match cancel
                        .run_until_cancelled(this.api.clone().connect_to_websocket(this.clone()))
                        .await
                    {
                        Some(Err(e)) => {
                            error!("failed while handling websocket: {e:?}");
                        }
                        Some(Ok(_)) => {
                            info!("websocket gracefully shutdown");
                        }
                        None => break,
                    };

                    // Retry to run websocket connection every minute if failing..
                    time::sleep(Duration::from_secs(60)).await;
                }
            }
        });

        tasks.close();
        async move { tasks.wait().await }
    }

    async fn init_chat_histories(&self) {
        let lookback_duration = Duration::from_secs(7 * 24 * 60 * 60);
        let since_unix_ms = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.saturating_sub(lookback_duration))
            .unwrap_or(Duration::ZERO)
            .as_millis();

        let mut chat_init_tasks = JoinSet::new();
        for (chat_id, chat_config) in self.config.chats.iter() {
            chat_init_tasks.spawn({
                let this = self.clone();
                let chat_id = chat_id.clone();
                let chat_config = chat_config.clone();

                async move {
                    let channel_id = &chat_id;

                    debug!(chat_id, "fetching chat posts...");
                    let Ok(mut res) = this
                        .api
                        .query::<ChannelPostsResponse>(
                            Method::GET.as_str(),
                            &format!("channels/{channel_id}/posts"),
                            Some(&[("since", &since_unix_ms.to_string())]),
                            None,
                        )
                        .await
                        .inspect_err(|e| warn!("failed to download chat posts: {e:?}"))
                    else {
                        return;
                    };

                    let raw_message_history = res
                        .order
                        .iter()
                        .rev()
                        .filter_map(|post_id| res.posts.remove(post_id))
                        .inspect(|m| trace!(?m, "got matter post"))
                        .filter_map(|m| {
                            let message = chat_config.preprocess_message(m.message.clone());
                            this.config.determine_role(&m).map(|r| match r {
                                ChatRole::Me => ChatMessage::assistant(message),
                                ChatRole::Other => ChatMessage::user(message),
                            })
                        });

                    let history = this
                        .impersonator
                        .new_blank_history(&chat_id, &chat_config)
                        .chain(raw_message_history)
                        .collect();

                    this.impersonator.init_chat_history(&chat_id, history).await;
                }
            });
        }
        chat_init_tasks.join_all().await;
    }

    async fn maybe_respond_to_chat(&self, chat_id: &str, cancel: CancellationToken) -> Result<()> {
        let my_user_id = &self.config.my_user_id;
        let Some(chat_config) = self.config.chats.get(chat_id) else {
            return Err(anyhow!("did not find chat config"));
        };

        // TODO Get this from user id (with channel config for DM perhaps?)
        let channel_id = chat_id;

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
            let maybe_natural_end = cancel
                .run_until_cancelled(self.impersonator.at_natural_end(chat_id, chat_config))
                .await;

            let Some(natural_end) = maybe_natural_end else {
                return Ok(());
            };

            if natural_end
                .inspect_err(|e| warn!("failed to query if natural end: {e:?}"))
                .unwrap_or(false)
            {
                debug!("conversation at a natural end, not responding");
                return Ok(());
            }
        }

        let typing_cancel = CancellationToken::new();
        tokio::spawn({
            let this = self.clone();
            let typing_cancel = typing_cancel.clone();
            let cancel = cancel.clone();
            let my_user_id = my_user_id.clone();
            let channel_id = channel_id.to_owned();

            cancel.run_until_cancelled_owned(async move {
                // This makes us wait a bit before we send the typing notification
                time::sleep(deliberation_time).await;

                typing_cancel
                    .run_until_cancelled_owned(async move {
                        loop {
                            trace!("sending typing notification...");
                            let _ = this
                                .api
                                .post::<_, serde_json::Value>(
                                    &format!("users/{my_user_id}/typing"),
                                    None,
                                    &json!({"channel_id": channel_id}),
                                )
                                .inspect_err(|e| {
                                    warn!("failed to send typing status: {e:?}");
                                })
                                .await;

                            time::sleep(typing_notification_interval).await;
                        }
                    })
                    .await;
            })
        });

        debug!("starting to generate a response...");
        let maybe_chat_response = cancel
            .run_until_cancelled(self.impersonator.generate_a_response(chat_id, chat_config))
            .await;

        let Some(chat_response) = maybe_chat_response else {
            debug!("response was cancelled, not sending anything..");
            return Ok(());
        };

        let chat_response = chat_response.context("responding to messages")?;

        let elapsed = start.elapsed();
        let answer_time = (deliberation_time
            + human_message_duration(chat_response.content.len(), write_chars_per_sec))
        .min(max_answer_time);

        let to_sleep = answer_time.saturating_sub(elapsed);
        if !to_sleep.is_zero() {
            debug!(
                "finished in {elapsed:?}, human answer time should be ish {answer_time:?}, so waiting a bit"
            );
            let Some(_) = cancel
                .run_until_cancelled(tokio::time::sleep(to_sleep))
                .await
            else {
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
        typing_cancel.cancel();

        self.api
            .post::<_, serde_json::Value>(
                "posts",
                None,
                &NewPostRequest {
                    channel_id,
                    message: &text_message,
                    root_id: None,
                    props: Some(chat_config.props()),
                },
            )
            .await
            .context("sending chat message")?;

        Ok(())
    }

    fn process_post_event(&self, mut event: PostEvent) {
        if let Some(chat_config) = self.config.chats.get(event.chat_id()) {
            debug!("processing event: {event:?}");
            tokio::spawn({
                let this = self.clone();
                let chat_id = event.chat_id().to_owned();
                let channel_name = &event.channel_display_name;
                let sender_name = &event.sender_name;
                let friend_name = &chat_config.friend_name;
                let chat_span = info_span!("chat", chat_id, channel_name, sender_name, friend_name);

                async move {
                    // We have to fetch chat_config again due to lifetimes
                    let Some(chat_config) = this.config.chats.get(&chat_id) else {
                        return;
                    };

                    let Some(chat_role) = this.config.determine_role(&event.content) else {
                        warn!("failed to determine role of event: {event:?}");
                        return;
                    };

                    event.content.message = chat_config.preprocess_message(event.content.message);

                    match chat_role {
                        ChatRole::Me => {
                            // We (or user) sent a message that should be added to the history
                            // as an assistant message.
                            debug!(
                                "adding assistant message to history: {:?}",
                                &event.content.message
                            );

                            let chat_message = ChatMessage::assistant(event.content.message);
                            this.impersonator
                                .commit_to_history(&chat_id, [chat_message])
                                .await;
                        }
                        ChatRole::Other => {
                            // Message was from the other user in the chat, so figure out how to respond

                            debug!(
                                "adding user message to history: {:?}",
                                &event.content.message
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
                                let chat_message = ChatMessage::user(event.content.message);
                                this.impersonator
                                    .commit_to_history(&chat_id, [chat_message])
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

                            if let Err(e) = this.maybe_respond_to_chat(&chat_id, cancel).await {
                                error!("failed when handling maybe response: {e:?}")
                            }
                        }
                    }
                }.instrument(chat_span)
            });
        } else {
            trace!("skipping event: {event:?}");
        }
    }
}

#[async_trait::async_trait]
impl WebsocketHandler for Manager {
    async fn callback(&self, event: WebsocketEvent) {
        use mattermost_api::socket::WebsocketEventType::*;

        let mut log_full_msg = false;

        match event.event {
            AddedToTeam => {}
            AuthenticationChallenge => {
                warn!("got authentication challenge!");
                log_full_msg = true;
            }
            ChannelConverted => {}
            ChannelCreated => {}
            ChannelDeleted => {}
            ChannelMemberUpdated => {}
            ChannelUpdated => {}
            ChannelViewed => {}
            ConfigChanged => {}
            DeleteTeam => {}
            DirectAdded => {}
            EmojiAdded => {}
            EphemeralMessage => {}
            GroupAdded => {}
            Hello => {
                let server_version = event.data.get("server_version").and_then(|v| v.as_str());
                let server_hostname = event.data.get("server_hostname").and_then(|v| v.as_str());
                let connection_id = event.data.get("connection_id").and_then(|v| v.as_str());
                info!(
                    server_version,
                    server_hostname, connection_id, "got hello from server"
                );
            }
            LeaveTeam => {}
            LicenseChanged => {}
            MemberroleUpdated => {}
            NewUser => {}
            PluginDisabled => {}
            PluginEnabled => {}
            PluginStatusesChanged => {}
            PostDeleted => {}
            PostEdited => {
                log_full_msg = true;
            }
            PostUnread => {
                log_full_msg = true;
            }
            Posted => {
                if let Ok(event) = serde_json::from_value::<PostEvent>(event.data.clone())
                    .inspect_err(|e| warn!("failed to deserialize event data: {e:?}"))
                {
                    self.process_post_event(event);
                }
            }
            PreferenceChanged => {}
            PreferencesChanged => {}
            PreferencesDeleted => {}
            ReactionAdded => {
                log_full_msg = true;
            }
            ReactionRemoved => {
                log_full_msg = true;
            }
            Response => {}
            RoleUpdated => {}
            StatusChange => {}
            Typing => {}
            UpdateTeam => {}
            UserAdded => {}
            UserRemoved => {}
            UserRoleUpdated => {}
            UserUpdated => {}
            DialogOpened => {}
            ThreadUpdated => {}
            ThreadFollowChanged => {}
            ThreadReadChanged => {}
            _ => {}
        };

        if log_full_msg {
            if let Ok(ps) = serde_json::to_string_pretty(&event) {
                debug!("got event: {}", ps);
            } else {
                warn!("failed to serialize when logging event");
            }
        }

        if let Err(e) = self.event_tx.try_send(event) {
            warn!("failed to log event: {e:?}");
        }
    }
}
