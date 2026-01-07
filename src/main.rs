use std::env;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use ollama_rs::Ollama;
use tokio::{signal, time};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::level_filters::LevelFilter;
use tracing::{Instrument, error, info, info_span};
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

    let impersonator = Arc::new(Impersonator::new(Ollama::default(), config.impersonator));
    let platforms = TaskTracker::new();

    if let Some(matter_config) = config.mattermost {
        platforms.spawn({
            let cancel = cancel.clone();
            let impersonator = Arc::clone(&impersonator);
            let platform_span = info_span!("platform", name = "mattermost");
            async move {
                let Ok(mattermost) = platform::mattermost::init(matter_config, cancel.clone())
                    .await
                    .inspect_err(|e| error!("failed to init mattermost platform: {e:?}"))
                else {
                    return;
                };

                let mut platform = platform::Manager::new(mattermost, impersonator);
                platform.run(cancel.clone()).await
            }
            .instrument(platform_span)
        });
    }

    if let Some(discord_config) = config.discord {
        platforms.spawn({
            let cancel = cancel.clone();
            let impersonator = Arc::clone(&impersonator);
            let platform_span = info_span!("platform", name = "discord");
            async move {
                let Ok(discord) = platform::discord::init(discord_config, cancel.clone())
                    .await
                    .inspect_err(|e| error!("failed to init discord platform: {e:?}"))
                else {
                    return;
                };

                let mut platform = platform::Manager::new(discord, impersonator);
                platform.run(cancel.clone()).await
            }
            .instrument(platform_span)
        });
    }

    platforms.close();

    // Wait until shutdown is requested
    cancel.cancelled().await;

    info!("waiting for platforms to gracefully shutdown...");
    if let Err(e) = time::timeout(Duration::from_secs(3), platforms.wait()).await {
        error!("timed out while waiting for platforms to shutdown: {e:?}");
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
