use anyhow::{Context, Result};
use std::{env, path::PathBuf};
use tracing::{trace, warn};

use crate::{
    impersonator,
    platform::{discord, mattermost},
};

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    pub impersonator: impersonator::Config,
    #[serde(default)]
    pub mattermost: Option<mattermost::Config>,
    #[serde(default)]
    pub discord: Option<discord::Config>,
}

impl Config {
    pub fn try_new() -> Result<Config> {
        let config = config::Config::builder()
            .add_source(config::File::with_name("ass-istant"))
            .build()
            .context("failed to build configuration")?;

        let mut config: Config = config
            .try_deserialize()
            .context("failed to deserialize config")?;

        // Ensure all chats mentioned in a persona, is initialized in the config for each
        // platform aswell, so it is setup and managed.
        for chat_id in config.impersonator.all_configured_chats() {
            // TODO Maybe make a common trait for all platform configs to make this more dynamic?
            match chat_id.platform_name() {
                mattermost::PLATFORM_NAME => {
                    if let Some(mattermost) = &mut config.mattermost {
                        mattermost.ensure_chat_configured(chat_id);
                    } else {
                        warn!(
                            "chat id '{chat_id}' references non-configured platform '{}'",
                            mattermost::PLATFORM_NAME
                        );
                    }
                }
                discord::PLATFORM_NAME => {
                    if let Some(discord) = &mut config.discord {
                        discord.ensure_chat_configured(chat_id);
                    } else {
                        warn!(
                            "chat id '{chat_id}' references non-configured platform '{}'",
                            discord::PLATFORM_NAME
                        );
                    }
                }
                _ => {
                    warn!("chat id '{chat_id}' had unknown namespace, ignoring");
                }
            }
        }

        Ok(config)
    }
}

pub fn default_config_dir() -> PathBuf {
    path_from_env("XDG_CONFIG_HOME")
        .or_else(|| env::home_dir().map(|h| h.join(".config/")))
        .map(|cfg_dir| cfg_dir.join(env!("CARGO_PKG_NAME")))
        .unwrap_or_else(|| {
            concat!("/opt/", env!("CARGO_PKG_NAME"), "/etc/")
                .parse()
                .unwrap()
        })
}

pub fn default_cache_dir() -> PathBuf {
    path_from_env("XDG_CACHE_HOME")
        .or_else(|| env::home_dir().map(|h| h.join(".cache/")))
        .map(|cdir| cdir.join(env!("CARGO_PKG_NAME")))
        .unwrap_or_else(|| env::temp_dir().join(concat!(env!("CARGO_PKG_NAME"), "/")))
}

fn path_from_env(name: &str) -> Option<PathBuf> {
    let maybe_path = env::var_os(name).map(PathBuf::from);
    if maybe_path.is_none() {
        trace!(var_name = name, "did not find env var '{}'", name);
    }
    maybe_path
}
