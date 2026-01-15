use anyhow::{Context, Result};
use tracing::warn;

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
