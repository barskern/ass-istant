use anyhow::{Context, Result};

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

        config
            .try_deserialize()
            .context("failed to deserialize config")
    }
}
