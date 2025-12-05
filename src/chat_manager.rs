use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use axum::http::Method;
use dashmap::DashMap;
use futures::TryFutureExt;
use mattermost_api::prelude::*;
use ollama_rs::generation::chat::ChatMessage;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::{self, Instant};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{debug, info, trace, warn};

use crate::impersonator::Impersonator;
use crate::utils::human_message_duration;

const CMD_PREFIX: &str = "hey slave,";
const AI_PREFIX: &str = "LLM: ";

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    // TODO Fetch from /users/me instead
    my_user_id: String,
    chats: HashMap<String, ChatConfig>,
}

#[derive(serde::Deserialize, Clone, Debug)]
struct ChatConfig {
    #[serde(default)]
    should_prefix: bool,
    friend_name: String,
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
}

#[derive(Deserialize, Debug)]
struct PostContent {
    message: String,
    user_id: String,
    channel_id: String,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct PostMessage {
    channel_display_name: String,
    sender_name: String,
    #[serde(rename = "post", with = "serde_nested_json")]
    content: PostContent,
}

impl PostMessage {
    pub fn chat_id(&self) -> &str {
        &self.content.channel_id
    }
}

#[derive(Default)]
struct InflightState {
    cancel: CancellationToken,
    messages: Vec<ChatMessage>,
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
pub struct PostForChannel {
    user_id: String,
    message: String,
}

#[derive(Deserialize, Debug)]
pub struct PostsForChannel {
    order: Vec<String>,
    posts: HashMap<String, PostForChannel>,
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

    async fn init_chat_histories(&self) {
        let lookback_duration = Duration::from_secs(2 * 24 * 60 * 60);
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
                    let ChatConfig { friend_name, .. } = &chat_config;
                    let channel_id = &chat_id;

                    debug!(chat_id, "fetching chat posts...");
                    let Ok(mut res) = this
                        .api
                        .query::<PostsForChannel>(
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
                        .map(|m| {
                            let has_ai_prefix = m.message.starts_with(AI_PREFIX);
                            let has_cmd_prefix = m.message.starts_with(CMD_PREFIX);
                            let from_me = m.user_id == this.config.my_user_id;

                            let message = chat_config.preprocess_message(m.message);
                            if (from_me && !chat_config.should_prefix && !has_cmd_prefix)
                                || (has_ai_prefix && chat_config.should_prefix)
                            {
                                ChatMessage::assistant(message)
                            } else {
                                ChatMessage::user(message)
                            }
                        });

                    let history = this
                        .impersonator
                        .new_blank_history(&chat_id, friend_name)
                        .chain(raw_message_history)
                        .collect();

                    this.impersonator.init_chat_history(&chat_id, history).await;
                }
            });
        }
        chat_init_tasks.join_all().await;
    }

    async fn respond_to_post_message(
        &self,
        post_message: PostMessage,
        messages: Vec<ChatMessage>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let chat_id = post_message.chat_id().to_owned();
        let PostMessage {
            channel_display_name: _,
            sender_name: _,
            content:
                PostContent {
                    message: latest_message,
                    user_id: _,
                    channel_id,
                },
        } = post_message;
        let my_user_id = &self.config.my_user_id;

        let Some(ChatConfig {
            should_prefix,
            friend_name,
        }) = self.config.chats.get(&chat_id)
        else {
            return Err(anyhow!(
                "did not find channel config for message: {latest_message:?}"
            ));
        };

        // TODO Get this from user id (with channel config for DM perhaps?)
        let channel_id = &channel_id;

        let start = Instant::now();

        let latest_message = latest_message.trim().to_string();

        // Just some heuristic to add some realism to the responses
        let write_chars_per_sec = 15.0;
        let read_chars_per_sec = 30.0;

        let deliberation_time =
            human_message_duration(&latest_message, read_chars_per_sec) + Duration::from_secs(5);
        let typing_notification_interval = Duration::from_secs(2);

        let typing_cancel = CancellationToken::new();
        tokio::spawn({
            let this = self.clone();
            let typing_cancel = typing_cancel.clone();
            let cancel = cancel.clone();
            let my_user_id = my_user_id.clone();
            let channel_id = channel_id.clone();

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

        let maybe_chat_response = cancel
            .run_until_cancelled(self.impersonator.respond_to_messages(
                &chat_id,
                friend_name,
                messages.clone(),
            ))
            .await;

        let Some(chat_response) = maybe_chat_response else {
            debug!("response was cancelled, not sending anything..");
            return Ok(());
        };

        let chat_response = chat_response.context("responding to messages")?;

        let elapsed = start.elapsed();
        let answer_time =
            deliberation_time + human_message_duration(&chat_response.content, write_chars_per_sec);

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

        let text_message = if *should_prefix {
            format!("{AI_PREFIX}{}", &chat_response.content)
        } else {
            chat_response.content.clone()
        };

        // TODO The following three actions seem racy, ensure they work as intended..
        {
            self.inflight_responses.remove(&chat_id);
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
                &json!({"channel_id": channel_id, "message": text_message}),
            )
            .await?;

        // Only commit on successful send
        let mut all_messages = messages;
        all_messages.push(chat_response);
        self.impersonator
            .commit_to_history(&chat_id, all_messages)
            .await;

        Ok(())
    }

    fn should_handle_post(&self, post_message: &PostMessage) -> bool {
        let Some(chat_config) = self.config.chats.get(post_message.chat_id()) else {
            return false;
        };
        if chat_config.should_prefix {
            // In prefixing chats, answer all messages NOT starting with the AI prefix
            !post_message.content.message.starts_with(AI_PREFIX)
        } else {
            // In non-prefixing chats, handle all messages (messages from "us" will only be
            // added to the history, so that the "owner" and inject wanted messages)
            true
        }
    }

    fn process_post_message(&self, mut message: PostMessage) {
        if self.should_handle_post(&message) {
            debug!("processing message: {message:?}");
            tokio::spawn({
                let this = self.clone();
                let chat_id = message.chat_id().to_owned();

                async move {
                    let Some(chat_config) = this.config.chats.get(&chat_id) else {
                        return;
                    };
                    let has_cmd_prefix = message.content.message.starts_with(CMD_PREFIX);
                    message.content.message =
                        chat_config.preprocess_message(message.content.message);

                    let is_from_me = message.content.user_id == this.config.my_user_id;
                    if is_from_me && !chat_config.should_prefix && !has_cmd_prefix {
                        // User injected a message into a non-prefixing chat, so we simply add it into the history as
                        // a assistant message!

                        debug!("injecting assistant message: {:?}", message.content.message);
                        let injected_message = ChatMessage::assistant(message.content.message);
                        this.impersonator
                            .commit_to_history(&chat_id, [injected_message])
                            .await;

                        return;
                    }

                    let chat_message = ChatMessage::user(message.content.message.clone());
                    let messages = {
                        if let Some((_, inflight_state)) = this.inflight_responses.remove(&chat_id)
                        {
                            // TODO Should we unconditionally cancel?? Verify that this works
                            // even if we already finished and added to history..
                            inflight_state.cancel.cancel();

                            let mut messages = inflight_state.messages;
                            messages.push(chat_message);
                            messages
                        } else {
                            vec![chat_message]
                        }
                    };

                    let cancel = CancellationToken::new();
                    {
                        let existing = this.inflight_responses.insert(
                            chat_id,
                            InflightState {
                                cancel: cancel.clone(),
                                messages: messages.clone(),
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

                    let _ = this
                        .respond_to_post_message(message, messages, cancel)
                        .inspect_err(|e| warn!("failed to respond to post message: {e:?}"))
                        .await;
                }
            });
        } else {
            debug!("skipping message: {message:?}");
        }
    }
}

#[async_trait::async_trait]
impl WebsocketHandler for Manager {
    async fn callback(&self, message: WebsocketEvent) {
        use mattermost_api::socket::WebsocketEventType::*;

        let mut log_full_msg = false;

        match message.event {
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
                let server_version = message.data.get("server_version").and_then(|v| v.as_str());
                let server_hostname = message.data.get("server_hostname").and_then(|v| v.as_str());
                let connection_id = message.data.get("connection_id").and_then(|v| v.as_str());
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
                if let Ok(message) = serde_json::from_value::<PostMessage>(message.data.clone())
                    .inspect_err(|e| warn!("failed to deserialize message data: {e:?}"))
                {
                    self.process_post_message(message);
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
            if let Ok(ps) = serde_json::to_string_pretty(&message) {
                debug!("got message: {}", ps);
            } else {
                warn!("failed to serialized when logging message");
            }
        }

        if let Err(e) = self.event_tx.try_send(message) {
            warn!("failed to log event: {e:?}");
        }
    }
}
