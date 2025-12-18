use std::collections::HashSet;
use std::io::BufWriter;
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Form, Router, routing};
use chrono::Utc;
use oauth2::basic::{BasicClient, BasicTokenResponse};
use oauth2::{AccessToken, EndpointNotSet, EndpointSet, Scope, TokenResponse, reqwest};
use oauth2::{
    AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken, PkceCodeChallenge, RedirectUrl,
    TokenUrl,
};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, info, info_span, trace, warn};
use url::Url;

use crate::utils::waitable_lock::WaitableLock;

#[derive(Clone, Debug)]
pub struct TokenHandle {
    token: Arc<WaitableLock<TokenWrapper>>,
}

impl TokenHandle {
    pub async fn access_token(&self) -> AccessToken {
        self.token
            .fetch_or_wait()
            .await
            // TODO Is this unwrap merited??
            .unwrap()
            .response
            .access_token()
            .clone()
    }
}

#[derive(Clone, Debug)]
pub struct Manager {
    name: Arc<str>,
    config: Arc<Config>,
    token: Arc<WaitableLock<TokenWrapper>>,
    oauth_client:
        BasicClient<EndpointSet, EndpointNotSet, EndpointNotSet, EndpointNotSet, EndpointSet>,
    http_client: reqwest::Client,
    cache_write_tasks: TaskTracker,
}

impl Manager {
    pub fn new(name: String, config: Config) -> Self {
        let redirect_url =
            RedirectUrl::from_url(config.external_url.join(&callback_path(&name)).unwrap());

        // Create an OAuth2 client by specifying the client ID, client secret, authorization URL and
        // token URL.
        let oauth_client = BasicClient::new(config.client_id.clone())
            .set_client_secret(config.client_secret.clone())
            .set_auth_uri(config.auth_url.clone())
            .set_token_uri(config.token_url.clone())
            .set_redirect_uri(redirect_url);

        let http_client = reqwest::ClientBuilder::new()
            // Following redirects opens the client up to SSRF vulnerabilities.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("Client should build");

        Self {
            name: name.into(),
            config: Arc::new(config),
            token: Default::default(),
            oauth_client,
            http_client,
            cache_write_tasks: TaskTracker::new(),
        }
    }

    pub fn token_handle(&self) -> TokenHandle {
        TokenHandle {
            token: Arc::clone(&self.token),
        }
    }

    pub async fn run(&mut self, cancel: CancellationToken) {
        // TODO Verify that this is cancel safe
        cancel
            .run_until_cancelled_owned(async {
                match self.read_from_cache().await {
                    Ok(cached_token) => {
                        let grace_expires_at =
                            cached_token.expires_at() - self.config.token_grace_period;

                        let until_refresh = (grace_expires_at - Utc::now())
                            .to_std()
                            .unwrap_or(Duration::ZERO);

                        debug!(?cached_token, ?until_refresh, "cached token info");

                        if !until_refresh.is_zero() {
                            self.token.write(cached_token).await;
                            tokio::time::sleep(until_refresh).await;
                        } else {
                            info!("cached token was fully expired or in grace period");
                        }
                    }
                    Err(e) => {
                        warn!("failed to fetch token from cache: {e:?}");
                    }
                }

                loop {
                    match self.update_token().await {
                        Ok(new_token) => {
                            let grace_expires_at =
                                new_token.expires_at() - self.config.token_grace_period;

                            let until_refresh = (grace_expires_at - Utc::now())
                                .to_std()
                                .unwrap_or(Duration::ZERO);

                            if until_refresh.is_zero() {
                                error!(
                                    "new token is not valid for more than configured grace period!"
                                );
                                tokio::time::sleep(self.config.token_refresh_error_backoff).await;
                            } else {
                                self.token.write(new_token.clone()).await;
                                self.cache_write_tasks.spawn({
                                    let this = self.clone();
                                    let new_token = new_token.clone();
                                    async move {
                                        if let Err(e) = this.write_to_cache(new_token).await {
                                            warn!("failed to write token to cache: {e:?}");
                                        }
                                    }
                                });
                                tokio::time::sleep(until_refresh).await;
                            }
                        }
                        Err(e) => {
                            error!("failed to update token: {e:?}");
                            tokio::time::sleep(self.config.token_refresh_error_backoff).await;
                        }
                    }
                }
            })
            .await;

        self.cache_write_tasks.close();
        self.cache_write_tasks.wait().await;
    }

    async fn update_token(&self) -> Result<TokenWrapper> {
        if let Some(current_token) = self.token.fetch().await
            && let Ok(refreshed_token) =
                self.refresh_token_flow(current_token)
                    .await
                    .inspect_err(|e| {
                        warn!("failed to refresh token: {e:?}");
                    })
        {
            return Ok(refreshed_token);
        };

        info!("running initial auth flow!");
        self.initial_auth_flow().await
    }

    async fn read_from_cache(&self) -> Result<TokenWrapper> {
        let mut token_cache_path = self.config.cache_dir.join(&*self.name);
        token_cache_path.set_extension("json");

        let contents = tokio::fs::read(&token_cache_path)
            .await
            .context("failed to open token cache file for reading")?;

        serde_json::from_slice(&contents).context("failed to parse token cache")
    }

    /// # Cancel Safety
    ///
    /// If cancelled, either the token was successfully written to file, or it
    /// has not been written at all.
    async fn write_to_cache(&self, token: TokenWrapper) -> Result<()> {
        let cache_dir = self.config.cache_dir.clone();

        let mut token_cache_path = cache_dir.join(&*self.name);
        token_cache_path.set_extension("json");

        let mut token_cache_lock = token_cache_path.clone();
        token_cache_lock.set_extension("lock");

        // NOTE! Important that we write the temporary file to the same
        // file system as the destination, or `rename` might not work!
        let mut tmp_token_cache_path = token_cache_path.clone();
        tmp_token_cache_path.set_extension("tmp.json");

        task::spawn_blocking(move || {
            let span = info_span!("write_to_cache", ?token_cache_path);
            let _ = span.enter();

            if let Err(e) = std::fs::create_dir_all(&cache_dir) {
                warn!("failed to create cache directories: {e:?}");
            }

            // This should hopefully be resilient to both multiple iterations
            // of this program running on the same computer, but also for multiple
            // tasks in the same instance trying to update the same token cache.
            let lockfile = std::fs::File::create_new(&token_cache_lock)
                .inspect_err(|e| trace!("could not create new token lockfile: {e:?}"))
                .or_else(|_| std::fs::File::open(&token_cache_lock))
                .context("failed to open/create token lockfile")?;

            // This ensures that only a single write_to_cache task can
            // write to the temporary file at a time, even across processes
            lockfile
                .lock()
                .context("failed to acquire token cache lock")?;

            {
                let file_buf = std::fs::File::create(&tmp_token_cache_path)
                    .map(BufWriter::new)
                    .context("failed to open cache file for writing")?;
                serde_json::to_writer(file_buf, &token).context("failed to serialize json")?;
            }

            std::fs::rename(&tmp_token_cache_path, &token_cache_path)
                .context("failed to rename temp token file")?;

            Ok(())
        })
        .await
        .context("failed to spawn blocking task")?
    }

    async fn refresh_token_flow(&self, token: TokenWrapper) -> Result<TokenWrapper> {
        let Some(refresh_token) = token.response.refresh_token() else {
            return Err(anyhow!("token missing refresh token"));
        };

        let requested_at = Utc::now();
        self.oauth_client
            .exchange_refresh_token(refresh_token)
            .request_async(&self.http_client)
            .await
            .map(|response| TokenWrapper {
                requested_at,
                response,
            })
            .context("failed to request refresh token")
    }

    async fn initial_auth_flow(&self) -> Result<TokenWrapper> {
        // Generate a PKCE challenge.
        let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

        // Generate the full authorization URL.
        let (auth_url, csrf_token) = self
            .oauth_client
            .authorize_url(CsrfToken::new_random)
            .add_scopes(self.config.scopes.iter().cloned())
            .set_pkce_challenge(pkce_challenge)
            .url();

        // This is the URL you should redirect the user to, in order to trigger the authorization
        // process.
        // TODO Make this more elegant!
        info!(%auth_url, "Browse to the auth url to authenticate the bot");

        // Setup a temporary axum server to wait for the redirect and auth code
        let auth_code = {
            #[derive(serde::Deserialize, Debug)]
            #[serde(untagged)]
            enum CallbackParams {
                Valid {
                    code: AuthorizationCode,
                    state: CsrfToken,
                },
                Error {
                    error: String,
                    error_description: String,
                    state: CsrfToken,
                },
            }

            impl CallbackParams {
                fn state(&self) -> &CsrfToken {
                    match self {
                        CallbackParams::Valid { state, .. }
                        | CallbackParams::Error { state, .. } => state,
                    }
                }
            }

            let (sender, mut receiver) = mpsc::channel(10);
            let router = Router::new().route(
                &callback_path(&self.name),
                routing::get(move |Form(params): Form<CallbackParams>| async move {
                    if params.state().secret() != csrf_token.secret() {
                        return (StatusCode::UNAUTHORIZED, "Invalid callback state")
                            .into_response();
                    }

                    match params {
                        CallbackParams::Valid { code, .. } => {
                            let _ = sender.send(code).await;
                            (StatusCode::OK, "Bot is authorized!").into_response()
                        }
                        CallbackParams::Error {
                            error,
                            error_description,
                            ..
                        } => {
                            error!("got error from oauth callback: {error} {error_description}");
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!(
                                    "Got error from oauth callback: **{error}** {error_description}"
                                ),
                            )
                                .into_response()
                        }
                    }
                }),
            );
            let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 8080))
                .await
                .context("failed to setup tcp listener")?;

            let cancel = CancellationToken::new();
            let _cancel_guard = cancel.clone().drop_guard();
            tokio::spawn(async move {
                let _ = axum::serve(listener, router)
                    .with_graceful_shutdown(cancel.cancelled_owned())
                    .await;
                debug!("stopped oauth callback http server");
            });

            receiver
                .recv()
                .await
                .ok_or(anyhow!("callback sender closed unexpectedly"))?
        };

        let requested_at = Utc::now();
        let token_result = self
            .oauth_client
            .exchange_code(auth_code)
            // TODO Why do we have to add this?
            .add_extra_param("client_id", self.config.client_id.as_str())
            .add_extra_param("client_secret", self.config.client_secret.secret())
            .set_pkce_verifier(pkce_verifier)
            .request_async(&self.http_client)
            .await;

        match token_result {
            Ok(response) => {
                debug!(?response, "got new token response");

                Ok(TokenWrapper {
                    requested_at,
                    response,
                })
            }
            Err(e) => {
                match &e {
                    oauth2::RequestTokenError::ServerResponse(_e) => {}
                    oauth2::RequestTokenError::Request(_e) => {}
                    oauth2::RequestTokenError::Parse(error, items) => {
                        let lossy_str = String::from_utf8_lossy(items);
                        error!("parsing error was '{error}' on data: {lossy_str}");
                    }
                    oauth2::RequestTokenError::Other(_e) => {}
                };

                Err(anyhow!("{:?}", e))
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct TokenWrapper {
    requested_at: chrono::DateTime<Utc>,
    response: BasicTokenResponse,
}

impl TokenWrapper {
    fn expires_at(&self) -> chrono::DateTime<Utc> {
        self.requested_at + self.response.expires_in().unwrap_or(Duration::ZERO)
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    #[serde(default = "default_token_grace_period")]
    pub token_grace_period: Duration,
    #[serde(default = "default_token_refresh_error_backoff")]
    pub token_refresh_error_backoff: Duration,
    #[serde(default = "default_external_url")]
    pub external_url: Url,
    #[serde(default)]
    pub scopes: HashSet<Scope>,

    pub auth_url: AuthUrl,
    pub token_url: TokenUrl,
    pub client_id: ClientId,
    pub client_secret: ClientSecret,
}

fn default_external_url() -> Url {
    Url::parse("http://localhost:8080").unwrap()
}

fn default_cache_dir() -> PathBuf {
    crate::utils::PROJECT_DIRS
        .cache_dir()
        .join("token_responses")
}

fn default_token_grace_period() -> Duration {
    Duration::from_secs(60)
}

fn default_token_refresh_error_backoff() -> Duration {
    Duration::from_secs(30)
}

fn callback_path(name: &str) -> String {
    format!("/{name}/oauth/callback")
}
