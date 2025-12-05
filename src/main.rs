use std::env;

use anyhow::{Context, Result};
use mattermost_api::prelude::*;
use ollama_rs::Ollama;
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

use crate::chat_manager::Manager;
use crate::config::Config;
use crate::impersonator::Impersonator;

pub mod chat_manager;
pub mod config;
pub mod event_logger;
pub mod impersonator;
pub mod oauth;
pub mod utils;

#[tokio::main]
async fn main() -> Result<()> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .pretty()
        .init();

    info!(concat!(
        "starting ",
        env!("CARGO_PKG_NAME"),
        " v",
        env!("CARGO_PKG_VERSION")
    ));

    let config = Config::try_new().context("failed to init config")?;

    info!(?config, "parsed configuration");

    let auth_manager = oauth::Manager::new(config.oauth);
    tokio::spawn(auth_manager.background_task());

    let access_token = auth_manager.access_token().await;
    let auth = AuthenticationData::from_access_token(access_token.secret());

    let mut api = Mattermost::new(config.instance_url, auth)
        .context("failed to initialize mattermost api")?;
    api.store_session_token()
        .await
        .context("failed to store session token")?;

    let event_tx = event_logger::start_event_logger();
    let impersonator = Impersonator::new(Ollama::default(), config.impersonator);

    let handler = Manager::new(api.clone(), config.chat, impersonator, event_tx);
    handler.init_chat_histories().await;

    api.connect_to_websocket(handler).await.unwrap();

    Ok(())
}
