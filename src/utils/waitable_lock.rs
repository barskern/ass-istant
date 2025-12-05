use tokio::sync::{Notify, RwLock};

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("still empty after being notified")]
    EmptyAfterNotify,
}

/// A lock which can be awaited until there is content to be read
///
/// Useful for having some state that is refreshed or populated by a
/// different task than the one consuming it. In this project, it is
/// used for keeping authentication credentials so they can be
/// transparently refreshed and updated under the hood, while being
/// easy to use.
#[derive(Debug)]
pub struct WaitableLock<T> {
    value: RwLock<Option<T>>,
    notify: Notify,
}

impl<T> std::default::Default for WaitableLock<T> {
    fn default() -> Self {
        Self {
            value: Default::default(),
            notify: Default::default(),
        }
    }
}

impl<T: Clone> WaitableLock<T> {
    /// Read the value within the lock if any
    pub async fn fetch(&self) -> Option<T> {
        self.value.read().await.as_ref().cloned()
    }

    /// Read the value within the lock, if it is not set, await until it is written
    pub async fn fetch_or_wait(&self) -> Result<T, Error> {
        // NOTE! It is important that we hold on to the read guard until we have
        // registered our notify future, or else it's possible that we can read a
        // None, and then before we have the chance to call `notified`, the write
        // can happen and the notification can pass us by!

        let guard = self.value.read().await;
        match guard.as_ref() {
            Some(inner) => Ok(inner.clone()),
            None => {
                // NOTE! We make the future before dropping the read guard!
                let notified = self.notify.notified();

                drop(guard);

                notified.await;

                self.value
                    .read()
                    .await
                    .as_ref()
                    .cloned()
                    .ok_or(Error::EmptyAfterNotify)
            }
        }
    }
}

impl<T> WaitableLock<T> {
    /// Write a new value into the lock and notify all waiting tasks
    ///
    /// Returns the old value if any
    pub async fn write(&self, value: T) -> Option<T> {
        let old = { self.value.write().await.replace(value) };
        self.notify.notify_waiters();
        old
    }

    /// Clear the current inner value
    ///
    /// Returns the old value if any
    pub async fn clear(&self) -> Option<T> {
        self.value.write().await.take()
    }
}

#[cfg(test)]
mod tests {
    use futures::future::{join, join_all};
    use pretty_assertions::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::{join, time::sleep};

    use super::*;

    #[tokio::test]
    async fn simple_test() {
        let store = Arc::new(WaitableLock::<u64>::default());

        let handle_one = tokio::spawn({
            let one = store.clone();
            async move { assert_matches!(one.fetch_or_wait().await, Ok(1)) }
        });

        let handle_two = tokio::spawn({
            let two = store.clone();
            async move { assert_matches!(two.fetch_or_wait().await, Ok(1)) }
        });

        sleep(Duration::from_millis(10)).await;

        store.write(1).await;

        let (res_one, res_two) = join!(handle_one, handle_two);

        assert_matches!(res_one, Ok(_));
        assert_matches!(res_two, Ok(_));
    }

    #[tokio::test]
    async fn stress_test() {
        let store = Arc::new(WaitableLock::<u64>::default());

        let write_handle = tokio::spawn({
            let h = store.clone();
            async move {
                // A tiny sleep to let the spawns below hopefully start
                sleep(Duration::from_nanos(1)).await;
                h.write(1).await;
            }
        });

        let handles: Vec<_> = (1..1000)
            .map(|_| {
                tokio::spawn({
                    let h = store.clone();
                    async move { assert_matches!(h.fetch_or_wait().await, Ok(1)) }
                })
            })
            .collect();

        let (res_write, res_all) = join(write_handle, join_all(handles)).await;
        assert_matches!(res_write, Ok(_));

        for res_read in res_all.into_iter() {
            assert_matches!(res_read, Ok(_));
        }
    }
}
