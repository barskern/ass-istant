# Ass-istant

A shitty assistant to handle the dreaded small-talk for you in Mattermost. Use
at your own discretion.

## Quick Guide

1. Copy `ass-istant.sample.toml` to `ass-istant.toml` and fill it in with your
   Mattermost OAuth credentials, Mattermost URLs, system prompts and channel
   configuration.
2. `just run`
3. Follow the authentication instructions from the logs, i.e. authenticate with
   your Mattermost instance.
4. Profit from shitty interactions, in which you spend no effort.

## Runtime Requirements

- [Ollama](https://ollama.com/) running as a system daemon.
- [Mattermost OAuth Application Client ID and Secret](https://developers.mattermost.com/integrate/apps/authentication/oauth2/).

## Build Requirements

- [The Rust toolchain](https://rust-lang.org/tools/install/).
- [Just](https://github.com/casey/just) or running `cargo` commands directly
    (see Justfile).
