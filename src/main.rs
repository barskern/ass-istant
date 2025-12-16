use std::env;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use ollama_rs::Ollama;
use tokio::{signal, time};
use tokio_util::sync::CancellationToken;
use tracing::level_filters::LevelFilter;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::config::Config;
use crate::impersonator::Impersonator;

pub mod config;
pub mod event_logger;
pub mod impersonator;
pub mod oauth;
pub mod platform;
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

    let cancel = CancellationToken::new();
    tokio::spawn(cancel_on_shutdown(cancel.clone()));

    let mattermost = platform::mattermost::init(config.mattermost)
        .await
        .context("failed to init mattermost platform")?;

    let impersonator = Arc::new(Impersonator::new(Ollama::default(), config.impersonator));

    let platform = platform::Manager::new(mattermost, impersonator);
    platform.init(cancel.clone()).await;
    let background_tasks = platform.background_tasks(cancel.clone());

    // Wait for shutdown
    cancel.cancelled().await;

    if let Err(e) = time::timeout(Duration::from_secs(3), background_tasks).await {
        error!("timed out while waiting for background tasks to shutdown: {e:?}");
    }

    Ok(())
}

async fn cancel_on_shutdown(cancel: CancellationToken) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("got shutdown signal, stopping!");

    cancel.cancel();
}
