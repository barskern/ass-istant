use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{Context as _, Result, anyhow};
use futures::Stream;
use futures::stream::FuturesUnordered;
use oauth2::Scope;
use ollama_rs::generation::chat::ChatMessage as OllamaChatMessage;
use serenity::all::{ChannelId, Context, CreateMessage, MessagePagination};
use serenity::http::Http as DiscordHttp;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{Instrument, debug, error, info, trace, warn};

use crate::oauth;
use crate::platform::{
    ChatConfig as CommonChatConfig, ChatEvent, ChatId, ChatIdRef, ChatMessage, ChatRole, Platform,
};

pub const PLATFORM_NAME: &str = "discord";
// See https://discord.com/developers/docs/topics/oauth2#shared-resources-oauth2-scopes
const REQUIRED_SCOPES: &[&str] = &[
    "identify",
    "messages.read",
    //"dm_channels.messages.read",
    //"dm_channels.messages.write",
];

pub fn new_chat_id(channel_id: &str) -> ChatId {
    ChatId::new(PLATFORM_NAME, channel_id)
}

pub async fn init(config: Config, cancel: CancellationToken) -> Result<Manager> {
    let background_tasks = TaskTracker::new();

    let mut oauth_config = config.oauth;
    oauth_config
        .scopes
        .extend(REQUIRED_SCOPES.iter().map(|&s| Scope::new(s.into())));

    let mut auth_manager = oauth::Manager::new(PLATFORM_NAME.into(), oauth_config);
    let token_handle = auth_manager.token_handle();
    background_tasks.spawn({
        let cancel = cancel.clone();
        async move { auth_manager.run(cancel).await }.in_current_span()
    });

    let Some(access_token) = cancel
        .run_until_cancelled(token_handle.access_token())
        .await
    else {
        return Err(anyhow!("cancelled while waiting for access_token"));
    };

    let oauth_http = DiscordHttp::new(access_token.secret());
    let bot_http = DiscordHttp::new(&config.chats.bot_token);

    let handler = Arc::new(Handler {
        oauth_http,
        bot_http,
        config: config.chats,
        sender_tx: OnceLock::new(),
    });

    Ok(Manager {
        handler,
        background_tasks,
    })
}

#[derive(Clone)]
pub struct Manager {
    handler: Arc<Handler>,
    background_tasks: TaskTracker,
}

impl Platform for Manager {
    async fn run(&mut self, cancel: CancellationToken) -> Result<()> {
        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        let mut client = cancel
            .run_until_cancelled(
                Client::builder(&self.handler.config.bot_token, intents)
                    .event_handler_arc(Arc::clone(&self.handler))
                    .into_future(),
            )
            .await
            .context("cancelled while setting up client")?
            .context("failed to setup client")?;

        loop {
            match cancel.run_until_cancelled(client.start()).await {
                Some(Err(e)) => {
                    error!("failed while handling discord client: {e:?}");
                }
                Some(Ok(_)) => {
                    info!("discord client gracefully shutdown");
                }
                None => break,
            };

            // Retry to run websocket connection every minute if failing..
            sleep(Duration::from_secs(60)).await;
        }

        info!("awaiting background tasks for graceful shutdown..");
        self.background_tasks.close();
        self.background_tasks.wait().await;

        Ok(())
    }

    fn attach_event_extractor(&self, sender_tx: mpsc::Sender<ChatEvent>) {
        if let Err(e) = self.handler.sender_tx.set(sender_tx) {
            warn!("duplicate initialization of event extractor: {e:?}");
        }
    }

    fn fetch_chat_histories(
        &self,
    ) -> impl Stream<Item = Result<(ChatId, Vec<OllamaChatMessage>)>> + Send {
        let lookback_duration = Duration::from_secs(7 * 24 * 60 * 60);
        let lookback_timestamp = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.saturating_sub(lookback_duration))
            .unwrap_or(Duration::ZERO)
            .as_secs()
            .try_into()
            .unwrap_or(i64::MAX);

        let all_histories: FuturesUnordered<_> = self
            .handler
            .config
            .channels
            .iter()
            .map(|(channel_id, chat_config)| {
                let handler = Arc::clone(&self.handler);
                let chat_config = chat_config.clone();

                async move {
                    debug!(?channel_id, "fetching channel messages...");

                    let mut raw_messages = Vec::with_capacity(32);
                    let mut pagination = None;
                    let mut oldest_timestamp = None;
                    let mut max_iter = 500u32;

                    while max_iter > 0
                        && oldest_timestamp
                            .map(|oldest_timestamp| oldest_timestamp > lookback_timestamp)
                            .unwrap_or(true)
                    {
                        debug!(
                            ?pagination,
                            ?oldest_timestamp,
                            max_iter,
                            "fetching more messages..."
                        );

                        let new_raw_messages = handler
                            .bot_http
                            .get_messages(*channel_id, pagination, Some(50))
                            .await
                            .context("failed to get messages")?;

                        let Some(oldest_message) = new_raw_messages
                            .iter()
                            .min_by_key(|m| m.timestamp.timestamp())
                        else {
                            break;
                        };

                        pagination = Some(MessagePagination::Before(oldest_message.id));
                        oldest_timestamp = Some(oldest_message.timestamp.timestamp());
                        max_iter = max_iter.saturating_sub(1);

                        raw_messages.extend(new_raw_messages);
                    }

                    raw_messages.retain(|m| lookback_timestamp < m.timestamp.timestamp());
                    raw_messages.sort_unstable_by(|a, b| a.timestamp.cmp(&b.timestamp));

                    debug!("finished fetching messages!");

                    let processed_messages = raw_messages
                        .into_iter()
                        .inspect(|m| trace!(?m, "got discord message"))
                        .filter_map(|m| {
                            let message = chat_config.common.preprocess_message(m.content.clone());
                            handler.config.determine_role(&m).map(|r| match r {
                                ChatRole::Me => OllamaChatMessage::assistant(message),
                                ChatRole::Other => OllamaChatMessage::user(message),
                            })
                        })
                        .collect();

                    anyhow::Ok((new_chat_id(&channel_id.to_string()), processed_messages))
                }
            })
            .collect();

        all_histories
    }

    fn common_chat_config(&self, chat_id: &ChatIdRef) -> Option<&CommonChatConfig> {
        let Ok(channel_id) = chat_id.sub_id().parse() else {
            return None;
        };

        self.handler
            .config
            .channels
            .get(&channel_id)
            .map(|c| &c.common)
    }

    async fn send_typing_status(&self, chat_id: &ChatIdRef) {
        let Ok(channel_id) = chat_id.sub_id().parse() else {
            return;
        };

        if let Err(e) = self.handler.bot_http.broadcast_typing(channel_id).await {
            error!("failed to send typing broadcast message: {e:?}");
        }
    }

    async fn send_message(&self, chat_id: &ChatIdRef, message: &str) -> Result<()> {
        let Ok(channel_id) = chat_id.sub_id().parse() else {
            return Err(anyhow!("invalid channel id in chat id"));
        };

        let message = CreateMessage::new().content(message);
        self.handler
            .bot_http
            .send_message(channel_id, vec![], &message)
            .await
            .map(|_| ())
            .context("failed to send message")
    }
}

#[allow(unused)]
pub struct Handler {
    oauth_http: DiscordHttp,
    bot_http: DiscordHttp,
    config: ChatsConfig,
    sender_tx: OnceLock<mpsc::Sender<ChatEvent>>,
}

#[serenity::async_trait]
impl EventHandler for Handler {
    async fn message(&self, _context: Context, msg: Message) {
        let Some(role) = self.config.determine_role(&msg) else {
            return;
        };

        debug!(
            channel_id = ?msg.channel_id,
            user_id = ?msg.author.id,
            content = ?msg.content,
            type_ = ?msg.kind,
            role = ?role,
            "got message!"
        );
        trace!("got a message: {msg:?}");

        // TODO We probably want to filter more here!

        let chat_event = ChatEvent::Message {
            chat_id: new_chat_id(&msg.channel_id.to_string()),
            content: ChatMessage {
                from_user_id: msg.author.id.to_string(),
                message: msg.content,
            },
            role,
        };

        if let Some(sender_tx) = self.sender_tx.get()
            && let Err(e) = sender_tx.send(chat_event).await
        {
            error!("failed to send websocket event: {e:?}");
        }
    }

    async fn ready(&self, _: Context, ready: Ready) {
        info!("{} is connected!", ready.user.name);
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    oauth: crate::oauth::Config,
    #[serde(flatten)]
    chats: ChatsConfig,
}

#[derive(serde::Deserialize, Debug)]
struct ChatsConfig {
    // TODO Fetch from /users/@me instead
    my_user_id: String,
    bot_token: String,
    channels: HashMap<ChannelId, ChannelConfig>,
}

impl From<Message> for ChatMessage {
    fn from(msg: Message) -> Self {
        Self {
            from_user_id: msg.author.id.to_string(),
            message: msg.content,
        }
    }
}

impl ChatsConfig {
    pub fn determine_role(&self, msg: &Message) -> Option<ChatRole> {
        let chat_config = self.channels.get(&msg.channel_id)?;
        Some(
            chat_config
                .common
                .determine_role(&msg.to_owned().into(), &self.my_user_id),
        )
    }
}

#[derive(serde::Deserialize, Clone, Debug)]
struct ChannelConfig {
    #[serde(flatten)]
    common: CommonChatConfig,
}
