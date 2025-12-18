use std::collections::HashMap;
use std::iter;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use ollama_rs::Ollama;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use ollama_rs::generation::parameters::{KeepAlive, TimeUnit};
use ollama_rs::models::ModelOptions;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tracing::{debug, instrument, trace, warn};

type SharedHistory = Arc<Mutex<ChatHistory>>;

use crate::platform::{ChatConfig, ChatId, ChatIdRef};

pub struct Impersonator {
    client: Ollama,
    model_options: ModelOptions,
    config: Config,

    // Double mutex by design, as we want to allow modification to the map
    // while histories are "extracted" from it, however we only want one
    // chat request modifying a single history at a time.
    histories: Mutex<HashMap<ChatId, SharedHistory>>,
}

impl Impersonator {
    pub fn new(client: Ollama, config: Config) -> Self {
        let model_options = ModelOptions::default();

        Self {
            client,
            config,
            model_options,
            histories: Default::default(),
        }
    }

    pub async fn init_chat_history(&self, chat_id: &ChatIdRef, messages: Vec<ChatMessage>) {
        let mut histories = self.histories.lock().await;
        if histories.contains_key(chat_id) {
            warn!("tried to init already existing chat history, skipping..");
            return;
        }

        let Config {
            max_chars_in_history,
            at_least_n_messages_in_history,
            ..
        } = self.config;
        let mut history = ChatHistory::new(messages);
        history.clean(max_chars_in_history, at_least_n_messages_in_history);

        debug!("init history to: {:?}", &history);
        histories.insert(chat_id.to_owned(), Arc::new(Mutex::new(history)));
    }

    pub fn new_blank_history(
        &self,
        _chat_id: &ChatIdRef,
        chat_config: &ChatConfig,
    ) -> impl Iterator<Item = ChatMessage> {
        [ChatMessage::system(
            chat_config
                .custom_system_prompt
                .as_ref()
                .cloned()
                .unwrap_or_else(|| {
                    self.config
                        .system_prompt_direct_message
                        .replace("{friend_name}", &chat_config.friend_name)
                }),
        )]
        .into_iter()
    }

    pub async fn length_of_unanswered_messages(&self, chat_id: &ChatIdRef) -> Option<usize> {
        let history = {
            let histories = self.histories.lock().await;
            histories.get(chat_id).map(Arc::clone)?
        };
        let history = history.lock().await;
        let n_chars = history
            .messages
            .iter()
            .rev()
            .take_while(|m| m.role == MessageRole::User)
            .map(|m| m.content.len())
            .sum();

        Some(n_chars)
    }

    pub async fn commit_to_history(
        &self,
        chat_id: &ChatIdRef,
        chat_config: &ChatConfig,
        messages: impl IntoIterator<Item = ChatMessage>,
    ) {
        let history = self.get_or_init_history(chat_id, chat_config).await;

        let mut history = history.lock().await;
        history.messages.extend(messages);

        if history.messages.len() > 1 {
            let first = &history.messages[1];
            let last = &history.messages[history.messages.len() - 1];
            debug!(
                first_role = ?first.role,
                first_content = first.content,
                last_role = ?last.role,
                last_content = last.content,
                "first and last history entries are now"
            );
        } else {
            debug!("history is currently empty");
        }
    }

    /// Ensure that histories are not too big, remove old cruft
    pub async fn clean_histories(&self) {
        debug!("cleaning history!");

        let Config {
            max_chars_in_history,
            at_least_n_messages_in_history,
            ..
        } = self.config;

        let shared_histories: Vec<_> = {
            let histories = self.histories.lock().await;
            histories.values().map(Arc::clone).collect()
        };

        for shared_history in shared_histories {
            {
                let mut history = shared_history.lock().await;
                history.clean(max_chars_in_history, at_least_n_messages_in_history);
            }
        }
    }

    /// Generate a chat response from the model based on the current chat history
    ///
    /// NOTE! Does NOT commit anything to the chat history!
    pub async fn generate_a_response(
        &self,
        chat_id: &ChatIdRef,
        chat_config: &ChatConfig,
    ) -> Result<ChatMessage> {
        let history = self.get_or_init_history(chat_id, chat_config).await;

        let messages = {
            let history = history.lock().await;
            // TODO Prevent an expensive clone perhaps?
            // Maybe ChatMessageRequest could use Arc<str> or &str instead?
            history.messages.clone()
        };

        let request = ChatMessageRequest::new(self.config.model_name.clone(), messages)
            .options(self.model_options.clone())
            .keep_alive(KeepAlive::Until {
                time: 5,
                unit: TimeUnit::Minutes,
            });

        let start_ollama = Instant::now();
        let mut res = self
            .client
            .send_chat_messages(request)
            .await
            .context("sending chat message to ollama")?;
        let ollama_duration = start_ollama.elapsed();
        debug!(
            "ollama spent {}s to generate an answer",
            ollama_duration.as_secs()
        );

        if !res.done {
            warn!("response wasn't done for some reason: {res:?}");
        }

        if !res.message.tool_calls.is_empty() {
            warn!(
                "model expected to call tools, but not implemented: {:?}",
                res.message.tool_calls
            );
        }

        if !self.config.skip_humanize {
            let start_humanize = Instant::now();
            res.message.content = self
                .humanize_message(res.message.content.trim())
                .await
                .inspect_err(|e| warn!("failed to humanize message: {e:?}"))
                .unwrap_or_else(|_| res.message.content.trim().to_string());
            let humanize_duration = start_humanize.elapsed();
            debug!(
                "humanizer spent {}s to humanize",
                humanize_duration.as_secs()
            );
        }

        Ok(res.message)
    }

    pub async fn warrants_reply(
        &self,
        chat_id: &ChatIdRef,
        _chat_config: &ChatConfig,
    ) -> Result<bool> {
        let history = {
            let histories = self.histories.lock().await;
            match histories.get(chat_id).map(Arc::clone) {
                Some(v) => v,
                None => return Ok(true),
            }
        };

        let messages_to_eval = {
            let history = history.lock().await;
            let mut max_assistant_msgs = 2u32;

            let messages_to_keep = history
                .messages
                .iter()
                .rev()
                .position(|m| match m.role {
                    MessageRole::User => false,
                    MessageRole::Assistant => {
                        max_assistant_msgs = max_assistant_msgs.saturating_sub(1);
                        max_assistant_msgs == 0
                    }
                    _ => true,
                })
                .unwrap_or(history.messages.len());

            let keep_last_messages_index = history
                .messages
                .len()
                .saturating_sub(messages_to_keep.min(history.messages.len() - 1));

            if keep_last_messages_index < 1 {
                return Err(anyhow!("got invalid keep last messages index"));
            }

            if let Some(last_messages) = history.messages.get(keep_last_messages_index..) {
                iter::once(ChatMessage::system(
                    self.config.system_prompt_replier.clone(),
                ))
                // TODO Prevent an expensive clone perhaps?
                // Maybe ChatMessageRequest could use Arc<str> or &str instead?
                .chain(last_messages.iter().cloned())
                .collect()
            } else {
                return Err(anyhow!("failed to get last messages for reply query"));
            }
        };

        debug!("asking for natural end of following messages: {messages_to_eval:?}");
        let request = ChatMessageRequest::new(self.config.model_name.clone(), messages_to_eval)
            .options(self.model_options.clone())
            .keep_alive(KeepAlive::Until {
                time: 5,
                unit: TimeUnit::Minutes,
            });

        let res = self
            .client
            .send_chat_messages(request)
            .await
            .context("sending chat message to ollama")?;

        debug!(
            msg = res.message.content.trim(),
            "response from natural end detector"
        );

        if res.message.content.contains("false") || res.message.content.contains("FALSE") {
            Ok(false)
        } else if res.message.content.contains("true") || res.message.content.contains("TRUE") {
            Ok(true)
        } else {
            Err(anyhow!(
                "response from replier detector did not return either true or false, but rather: {}",
                res.message.content
            ))
        }
    }

    async fn humanize_message(&self, message: &str) -> Result<String> {
        debug!(msg = message, "message before humanization");

        let messages = vec![
            ChatMessage::system(self.config.system_prompt_humanizer.clone()),
            ChatMessage::user(message.into()),
        ];

        let request = ChatMessageRequest::new(self.config.model_name.clone(), messages)
            .options(self.model_options.clone())
            .keep_alive(KeepAlive::Until {
                time: 5,
                unit: TimeUnit::Minutes,
            });

        let res = self
            .client
            .send_chat_messages(request)
            .await
            .context("sending chat message to ollama")?;

        debug!(
            msg = res.message.content.trim(),
            "message after humanization"
        );

        Ok(res.message.content.trim().to_string())
    }

    async fn get_or_init_history(
        &self,
        chat_id: &ChatIdRef,
        chat_config: &ChatConfig,
    ) -> Arc<Mutex<ChatHistory>> {
        let mut histories = self.histories.lock().await;
        match histories.get(chat_id).map(Arc::clone) {
            Some(v) => v,
            None => {
                // Hopefully we don't hit this path, as we rather want the initialized
                // histories from previous chats (see `init_chat_history`).
                let new = Arc::new(Mutex::new(ChatHistory::new(
                    self.new_blank_history(chat_id, chat_config).collect(),
                )));
                histories.insert(chat_id.to_owned(), Arc::clone(&new));
                new
            }
        }
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    pub system_prompt_direct_message: String,
    #[serde(default = "default_prompt_humanizer")]
    pub system_prompt_humanizer: String,
    #[serde(default = "default_prompt_replier")]
    pub system_prompt_replier: String,
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_max_chars_in_history")]
    pub max_chars_in_history: usize,
    #[serde(default = "default_at_least_n_messages_in_history")]
    pub at_least_n_messages_in_history: usize,
    #[serde(default)]
    pub skip_humanize: bool,
}

fn default_prompt_humanizer() -> String {
    r#"
You are working as a reformatter, turning generated sentences into a more human vibe.
You try your best to make the sentence sound natural, vocal and realistic.
You may rewrite the sentences to give better flow to the content, especially removing punctuation, quotes and symbols.
You may change leading characters to lowercase, add typos and skip trailing periods to make the messages more realistic.
You do not change the capitalization of abbreviations, and you do NOT change banter, play on words and specific phrasings.
You will always receive a single message, and you will reply with ONLY the reformatted message.
"#.into()
}

fn default_prompt_replier() -> String {
    r#"
You are an interpreter and conversation specialist, specializing in deciding whether to reply or not in a chat.
The following conversation might be between multiple people.
You are to determine if the conversation is at a natural end for your part, or if it warrants a further reply or question.
You will reply with ONLY true if the conversation should continue or ONLY false if the conversation is at a natural end.
"#.into()
}

fn default_model_name() -> String {
    "gemma3:4b".into()
}

fn default_max_chars_in_history() -> usize {
    8000
}

fn default_at_least_n_messages_in_history() -> usize {
    8
}

#[derive(Debug)]
struct ChatHistory {
    messages: Vec<ChatMessage>,
}

impl ChatHistory {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self { messages }
    }

    #[instrument(level = "trace", skip(self))]
    pub fn clean(&mut self, max_chars_in_history: usize, at_least_n_messages: usize) {
        let messages_to_keep = self
            .messages
            .iter()
            .rev()
            .position({
                let mut chars_seen = 0;
                move |m| {
                    chars_seen += m.content.len();
                    chars_seen >= max_chars_in_history
                }
            })
            .unwrap_or(self.messages.len());

        if messages_to_keep < at_least_n_messages {
            trace!(messages_to_keep, "using at least n messages limit");
        } else {
            trace!(messages_to_keep, "using max char limit");
        }

        let keep_from_index = self
            .messages
            .len()
            .saturating_sub(messages_to_keep.max(at_least_n_messages));

        if keep_from_index > 1 {
            trace!("tossing {} messages from history", keep_from_index - 1);
            // We keep the first system prompt, but throw all other messages until our keep from index
            self.messages.drain(1..keep_from_index);
        }
    }
}
