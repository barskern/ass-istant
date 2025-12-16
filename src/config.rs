use anyhow::{Context, Result};

use crate::{impersonator, platform::mattermost};

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    pub impersonator: impersonator::Config,
    pub mattermost: mattermost::Config,
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
