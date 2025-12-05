use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use ollama_rs::Ollama;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::parameters::{KeepAlive, TimeUnit};
use ollama_rs::models::ModelOptions;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tracing::{debug, warn};

type SharedHistory = Arc<Mutex<Vec<ChatMessage>>>;

pub struct Impersonator {
    client: Ollama,
    model_options: ModelOptions,
    config: Config,

    // Double mutex by design, as we want to allow modification to the map
    // while histories are "extracted" from it, however we only want one
    // chat request modifying a single history at a time.
    histories: Mutex<HashMap<String, SharedHistory>>,
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

    pub async fn init_chat_history(&self, chat_id: &str, history: Vec<ChatMessage>) {
        let mut histories = self.histories.lock().await;
        if histories.contains_key(chat_id) {
            warn!("tried to init already existing chat history, skipping..");
            return;
        }

        debug!("init history to: {:?}", &history);
        histories.insert(chat_id.to_owned(), Arc::new(Mutex::new(history)));
    }

    pub fn new_blank_history(
        &self,
        _chat_id: &str,
        message_from_name: &str,
    ) -> impl Iterator<Item = ChatMessage> {
        [ChatMessage::system(
            self.config
                .system_prompt_direct_message
                .replace("{name}", message_from_name),
        )]
        .into_iter()
    }

    pub async fn commit_to_history(
        &self,
        chat_id: &str,
        messages: impl IntoIterator<Item = ChatMessage>,
    ) {
        let histories = self.histories.lock().await;
        let Some(history) = histories.get(chat_id).map(Arc::clone) else {
            return;
        };

        // NOTE! Do NOT hold locks across awaits!
        drop(histories);

        let mut history = history.lock().await;
        history.extend(messages);

        debug!("history is now: {history:?}");
    }

    pub async fn clean_history(&self) {
        // TODO Ensure that history is not too big, remove old cruft
    }

    // NOTE! Does NOT commit response to history!
    pub async fn respond_to_messages(
        &self,
        chat_id: &str,
        message_from_name: &str,
        messages: impl IntoIterator<Item = ChatMessage>,
    ) -> Result<ChatMessage> {
        let history = {
            let mut histories = self.histories.lock().await;

            match histories.get(chat_id) {
                Some(v) => Arc::clone(v),
                None => {
                    // Hopefully we don't hit this path, as we rather want the initialized
                    // histories from previous chats (see `Handler::init_chat_histories`).
                    let new = Arc::new(Mutex::new(
                        self.new_blank_history(chat_id, message_from_name).collect(),
                    ));
                    histories.insert(chat_id.into(), Arc::clone(&new));
                    new
                }
            }
        };

        let messages = {
            let history = history.lock().await;

            history
                .iter()
                .cloned()
                .chain(messages.into_iter())
                .collect()
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

        Ok(res.message)
    }

    async fn humanize_message(&self, message: &str) -> Result<String> {
        debug!("message before humanization: {message}");

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

        debug!("message after humanization: {}", res.message.content);

        Ok(res.message.content.trim().to_string())
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    pub system_prompt_direct_message: String,
    #[serde(default = "default_prompt_humanizer")]
    pub system_prompt_humanizer: String,
    #[serde(default = "default_model_name")]
    pub model_name: String,
}

fn default_prompt_humanizer() -> String {
    r#"
You are working as a reformatter, turning generated sentences into a more human vibe.
You try your best to make the sentence sound natural, vocal and realistic.
You may rewrite the sentences to give better flow to the content, especially removing punctuation, quotes and symbols.
You may change leading characters to lowercase, add typos and skip trailing periods to make the messages more realistic.
You will always receive a single message, and you will reply with ONLY the reformatted message.
"#.into()
}

fn default_model_name() -> String {
    "gemma3:12b".into()
}
