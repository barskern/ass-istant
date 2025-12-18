use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use axum::http::Method;
use futures::stream::FuturesUnordered;
use futures::{Stream, TryFutureExt};
use mattermost_api::prelude::*;
use ollama_rs::generation::chat::ChatMessage as OllamaChatMessage;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio::time;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, info, trace, warn};
use url::Url;

use crate::oauth;
use crate::platform::{
    ChatConfig as CommonChatConfig, ChatEvent, ChatId, ChatIdRef, ChatMessage, ChatRole, Platform,
};

const PLATFORM_NAME: &str = "matter";

pub fn new_chat_id(channel_id: &str) -> ChatId {
    ChatId::new(PLATFORM_NAME, channel_id)
}

pub async fn init(config: Config, cancel: CancellationToken) -> Result<Manager> {
    let background_tasks = TaskTracker::new();

    let mut auth_manager = oauth::Manager::new(PLATFORM_NAME.into(), config.oauth);
    let token_handle = auth_manager.token_handle();
    background_tasks.spawn({
        let cancel = cancel.clone();
        async move { auth_manager.run(cancel).await }
    });

    let Some(access_token) = cancel
        .run_until_cancelled(token_handle.access_token())
        .await
    else {
        return Err(anyhow!("cancelled while waiting for access_token"));
    };

    let auth = AuthenticationData::from_access_token(access_token.secret());

    let mut api = Mattermost::new(config.instance_url, auth)
        .context("failed to initialize mattermost api")?;

    api.store_session_token()
        .await
        .context("failed to store session token")?;

    Ok(Manager {
        api,
        config: Arc::new(config.chats),
        sender_tx: Arc::new(OnceLock::new()),
        background_tasks,
    })
}

#[derive(Clone)]
pub struct Manager {
    api: Mattermost,
    config: Arc<ChatsConfig>,
    sender_tx: Arc<OnceLock<mpsc::Sender<ChatEvent>>>,
    background_tasks: TaskTracker,
}

impl Platform for Manager {
    async fn run(&mut self, cancel: CancellationToken) -> Result<()> {
        loop {
            debug!("starting websocket event handler");
            match cancel
                .run_until_cancelled(self.api.clone().connect_to_websocket(self.clone()))
                .await
            {
                Some(Err(e)) => {
                    error!("failed while handling websocket: {e:?}");
                }
                Some(Ok(_)) => {
                    info!("websocket gracefully shutdown");
                }
                // Cancelled!
                None => break,
            };

            // Retry to run websocket connection every minute if failing..
            time::sleep(Duration::from_secs(60)).await;
        }

        self.background_tasks.close();
        self.background_tasks.wait().await;

        Ok(())
    }

    fn attach_event_extractor(&self, sender_tx: mpsc::Sender<ChatEvent>) {
        if let Err(e) = self.sender_tx.set(sender_tx) {
            warn!("duplicate initialization of event extractor: {e:?}");
        }
    }

    fn fetch_chat_histories(
        &self,
    ) -> impl Stream<Item = Result<(ChatId, Vec<OllamaChatMessage>)>> + Send {
        let lookback_duration = Duration::from_secs(7 * 24 * 60 * 60);
        let since_unix_ms = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.saturating_sub(lookback_duration))
            .unwrap_or(Duration::ZERO)
            .as_millis();

        let all_histories: FuturesUnordered<_> = self
            .config
            .channels
            .iter()
            .map(|(channel_id, chat_config)| {
                let this = self.to_owned();
                let channel_id = channel_id.clone();
                let chat_config = chat_config.clone();

                async move {
                    debug!(?channel_id, "fetching channel posts...");
                    let mut res = this
                        .api
                        .query::<ChannelPostsResponse>(
                            Method::GET.as_str(),
                            &format!("channels/{channel_id}/posts"),
                            Some(&[("since", &since_unix_ms.to_string())]),
                            None,
                        )
                        .await
                        .context("failed to download chat posts")?;

                    let raw_message_history: Vec<_> = res
                        .order
                        .iter()
                        .rev()
                        .filter_map(|post_id| res.posts.remove(post_id))
                        .inspect(|m| trace!(?m, "got matter post"))
                        .filter_map(|m| {
                            let message = chat_config.common.preprocess_message(m.message.clone());
                            this.config.determine_role(&m).map(|r| match r {
                                ChatRole::Me => OllamaChatMessage::assistant(message),
                                ChatRole::Other => OllamaChatMessage::user(message),
                            })
                        })
                        .collect();

                    anyhow::Ok((new_chat_id(&channel_id), raw_message_history))
                }
            })
            .collect();

        all_histories
    }

    fn common_chat_config(&self, chat_id: &ChatIdRef) -> Option<&CommonChatConfig> {
        let channel_id = chat_id.sub_id();
        self.config.channels.get(channel_id).map(|c| &c.common)
    }

    async fn send_typing_status(&self, chat_id: &ChatIdRef) {
        let my_user_id = &self.config.my_user_id;
        let channel_id = chat_id.sub_id();

        let _ = self
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
    }

    async fn send_message(&self, chat_id: &ChatIdRef, message: &str) -> Result<()> {
        let channel_id = chat_id.sub_id();
        let Some(channel_config) = self.config.channels.get(channel_id) else {
            return Err(anyhow!("chat '{chat_id:?}' had no config.."));
        };

        let channel_id = chat_id.sub_id();
        self.api
            .post::<_, serde_json::Value>(
                "posts",
                None,
                &NewPostRequest {
                    channel_id,
                    message,
                    root_id: None,
                    props: Some(channel_config.props()),
                },
            )
            .await
            .context("sending chat message")?;

        Ok(())
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
                if let Ok(post_event) = serde_json::from_value::<PostEvent>(event.data.clone())
                    .inspect_err(|e| {
                        warn!("failed to deserialize posted event: {e:?}");
                    })
                    && let Some(role) = self.config.determine_role(&post_event.content)
                {
                    let Post {
                        message,
                        user_id,
                        channel_id,
                        ..
                    } = post_event.content;

                    let chat_event = ChatEvent::Message {
                        chat_id: new_chat_id(&channel_id),
                        content: ChatMessage {
                            from_user_id: user_id,
                            message,
                        },
                        role,
                    };

                    if let Some(sender_tx) = self.sender_tx.get()
                        && let Err(e) = sender_tx.send(chat_event).await
                    {
                        error!("failed to send websocket event: {e:?}");
                    }
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
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    instance_url: Url,
    oauth: crate::oauth::Config,
    #[serde(flatten)]
    chats: ChatsConfig,
}

#[derive(serde::Deserialize, Debug)]
struct ChatsConfig {
    // TODO Fetch from /users/me instead
    my_user_id: String,
    channels: HashMap<String, ChannelConfig>,
}

impl ChatsConfig {
    pub fn determine_role(&self, post: &Post) -> Option<ChatRole> {
        let chat_config = self.channels.get(&post.channel_id)?;
        Some(
            chat_config
                .common
                .determine_role(&post.to_owned().into(), &self.my_user_id),
        )
    }
}

#[derive(serde::Deserialize, Clone, Debug)]
struct ChannelConfig {
    #[serde(flatten)]
    common: CommonChatConfig,
    #[serde(default)]
    extra_props: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ChannelConfig {
    pub fn props(&self) -> serde_json::Value {
        let mut map = self.extra_props.as_ref().cloned().unwrap_or_default();
        map.insert("is_clanker".into(), json!(true));
        serde_json::Value::Object(map)
    }
}

#[derive(Deserialize, Clone, Debug)]
#[allow(unused)]
struct Post {
    message: String,
    user_id: String,
    channel_id: String,
    #[serde(default)]
    props: Option<serde_json::Value>,
}

impl From<Post> for ChatMessage {
    fn from(value: Post) -> Self {
        ChatMessage {
            from_user_id: value.user_id,
            message: value.message,
        }
    }
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
    channel_id: &'a str,
    message: &'a str,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    root_id: Option<&'a str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    props: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct ChannelPostsResponse {
    order: Vec<String>,
    posts: HashMap<String, Post>,
}
