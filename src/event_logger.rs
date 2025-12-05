use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Write},
    path::{Path, PathBuf},
    thread,
};

use anyhow::{Context, Error};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Serializer;
use tokio::sync::mpsc;
use tracing::{error, warn};

type Result<T, E = Error> = std::result::Result<T, E>;

pub fn start_event_logger<T: Serialize + Send + 'static>() -> mpsc::Sender<T> {
    let (event_tx, mut event_rx) = mpsc::channel::<T>(50);

    thread::spawn(move || {
        let mut event_logger = EventLogger::new();
        while let Some(evt) = event_rx.blocking_recv() {
            if let Err(e) = event_logger.log(evt) {
                warn!("failed to log event: {e:?}");
            }
        }
        if let Err(e) = event_logger.flush() {
            error!("failed to flush event logger: {e:?}");
        }
    });

    event_tx
}

pub struct EventLogger {
    name: String,
    unwritten_bytes: Vec<u8>,
}

impl EventLogger {
    pub fn new() -> Self {
        let now = std::time::SystemTime::now();
        let dur = now.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        Self {
            name: format!("event-log-{dur}"),
            unwritten_bytes: Default::default(),
        }
    }

    pub fn log<E: Serialize>(&mut self, event: E) -> Result<()> {
        {
            let mut serializer = Serializer::new(&mut self.unwritten_bytes);
            event
                .serialize(&mut serializer)
                .context("failed to serialize event to bytes")?;
        }

        if !self.unwritten_bytes.is_empty() {
            // Always push a separator between events
            self.unwritten_bytes.push(b'\n');
        }

        // Prevent buffering from growing too big
        if self.unwritten_bytes.len() > 5000 {
            self.flush()?;
        }

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.unwritten_bytes.is_empty() {
            return Ok(());
        }

        // TODO Write in subfolder?
        let mut filepath = PathBuf::new();
        filepath.set_file_name(&self.name);
        filepath.set_extension("json");

        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(filepath)
                .context("opening/creating event logging file")?;

            file.write_all(&self.unwritten_bytes)
                .context("writing to event logging file")?;

            file.flush().context("flushing event logging file")?;
        }

        self.unwritten_bytes.clear();
        Ok(())
    }
}

pub fn read_events<T: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> Result<impl Iterator<Item = Result<T, serde_json::Error>>> {
    let file = File::open(path.as_ref())
        .map(BufReader::new)
        .context("opening event file")?;
    Ok(serde_json::Deserializer::from_reader(file).into_iter())
}

impl Drop for EventLogger {
    fn drop(&mut self) {
        // We don't have anything to do but ignore the error here..
        let _ = self.flush();
    }
}
