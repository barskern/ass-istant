use anyhow::{Context, Result};
use url::Url;

use crate::{chat_manager, impersonator, oauth};

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    pub instance_url: Url,
    pub oauth: oauth::Config,
    pub impersonator: impersonator::Config,
    pub chat: chat_manager::Config,
}

impl Config {
    pub fn try_new() -> Result<Config> {
        let config = config::Config::builder()
            .add_source(config::File::with_name("ass-istant"))
            .build()
            .context("failed to build configuration")?;

        config
            .try_deserialize()
            .context("failed to deserialize config")
    }
}
