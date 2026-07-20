use gz_features::{
    ENCODING_VERSION, FeatureCollator, FeatureRow, FeatureSchema, RowTargets, decode_feature_row,
    encode_feature_schema_config, encode_training_targets, validate_feature_row_header,
};
use gz_replay::{ReplayError, ReplayStore, SampleConfig, SampleKind};
use std::io::{ErrorKind, Read, Write};
use std::num::{NonZeroU64, NonZeroUsize};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

pub const SAMPLE_PROTOCOL_VERSION: u32 = 10;

const HELLO_ACK_FIXED_LEN: usize = 176;

const MAX_FRAME: usize = 256 * 1024 * 1024;
const FRAME_HELLO: u8 = 1;
const FRAME_HELLO_ACK: u8 = 2;
const FRAME_SAMPLE: u8 = 3;
const FRAME_SAMPLE_RESULT: u8 = 4;
const FRAME_ERROR: u8 = 5;

const ERROR_PROTOCOL: u32 = 1;
const ERROR_ENCODING: u32 = 2;
const ERROR_EMPTY_STORE: u32 = 3;
const ERROR_BAD_REQUEST: u32 = 4;
const ERROR_MISSING_FEATURES: u32 = 5;

#[derive(Clone, Debug)]
pub struct ReplayServeConfig {
    pub replay_dir: PathBuf,
    pub socket: PathBuf,
    pub max_batch: usize,
}

impl ReplayServeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.as_os_str().is_empty() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.socket.as_os_str().is_empty() {
            return Err("missing required --socket".to_owned());
        }
        if self.max_batch == 0 {
            return Err("--max-batch must be greater than zero".to_owned());
        }
        Ok(())
    }
}

pub fn run(config: ReplayServeConfig) -> Result<(), String> {
    let server = ReplaySampleServer::bind(config)?;
    loop {
        if let Err(error) = server.accept_concurrent() {
            eprintln!("replay sample accept failed: {error}");
        }
    }
}

pub fn run_one(config: ReplayServeConfig) -> Result<(), String> {
    ReplaySampleServer::bind(config)?.accept_one()
}

/// Serve loop over a store shared with a live producer (the in-process
/// sample service of `graphzero selfplay --serve-socket`). Appends are
/// serialized inside the store while committed rows remain concurrently
/// sampleable, so one process still owns the RocksDB writer.
pub fn run_shared(
    store: std::sync::Arc<ReplayStore>,
    socket: PathBuf,
    max_batch: usize,
) -> Result<(), String> {
    let server = ReplaySampleServer::bind_shared(store, socket, max_batch)?;
    loop {
        if let Err(error) = server.accept_concurrent() {
            eprintln!("replay sample accept failed: {error}");
        }
    }
}

struct ReplaySampleServer {
    listener: UnixListener,
    store: std::sync::Arc<ReplayStore>,
    schema: FeatureSchema,
    max_batch: NonZeroUsize,
}

struct ReplaySampleSession {
    store: std::sync::Arc<ReplayStore>,
    collator: FeatureCollator,
    max_batch: NonZeroUsize,
}

impl ReplaySampleServer {
    fn bind(config: ReplayServeConfig) -> Result<Self, String> {
        config.validate()?;
        let store = ReplayStore::open(&config.replay_dir).map_err(|error| error.to_string())?;
        Self::bind_shared(std::sync::Arc::new(store), config.socket, config.max_batch)
    }

    fn bind_shared(
        store: std::sync::Arc<ReplayStore>,
        socket: PathBuf,
        max_batch: usize,
    ) -> Result<Self, String> {
        let schema_config = store
            .feature_schema()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "store was not produced by featurized selfplay".to_owned())?;
        let schema = FeatureSchema::new(schema_config).map_err(|error| error.to_string())?;
        let max_batch = NonZeroUsize::new(max_batch)
            .ok_or_else(|| "--max-batch must be greater than zero".to_owned())?;

        if socket.exists() {
            std::fs::remove_file(&socket).map_err(|error| error.to_string())?;
        }
        let listener = UnixListener::bind(&socket).map_err(|error| error.to_string())?;

        Ok(Self {
            listener,
            store,
            schema,
            max_batch,
        })
    }

    fn accept_stream(&self) -> Result<UnixStream, String> {
        let (stream, _) = self.listener.accept().map_err(|error| error.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(300)))
            .map_err(|error| error.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(300)))
            .map_err(|error| error.to_string())?;
        Ok(stream)
    }

    fn session(&self) -> ReplaySampleSession {
        ReplaySampleSession {
            store: std::sync::Arc::clone(&self.store),
            collator: FeatureCollator::new(self.schema.clone(), self.max_batch),
            max_batch: self.max_batch,
        }
    }

    fn accept_one(&self) -> Result<(), String> {
        let mut stream = self.accept_stream()?;
        self.session().handle_client(&mut stream)
    }

    fn accept_concurrent(&self) -> Result<(), String> {
        let mut stream = self.accept_stream()?;
        let mut session = self.session();
        std::thread::Builder::new()
            .name("replay-sample-client".to_owned())
            .spawn(move || {
                if let Err(error) = session.handle_client(&mut stream) {
                    // A client disappearing or timing out ends only its own
                    // session; the listener keeps accepting other trainers.
                    eprintln!("replay sample connection ended: {error}");
                }
            })
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

impl ReplaySampleSession {
    fn handle_client(&mut self, stream: &mut UnixStream) -> Result<(), String> {
        let mut read_buf = Vec::new();
        let mut write_buf = Vec::new();
        let mut batch_buf = Vec::new();
        let mut target_buf = Vec::new();

        let Some((frame_type, payload)) =
            read_frame(stream, &mut read_buf).map_err(|error| error.to_string())?
        else {
            return Ok(());
        };
        if frame_type != FRAME_HELLO {
            send_error(stream, &mut write_buf, ERROR_PROTOCOL, "expected HELLO")?;
            return Ok(());
        }
        if let Err(error) = self.handle_hello(payload, stream, &mut write_buf) {
            send_error(stream, &mut write_buf, error.0, error.1)?;
            return Ok(());
        }

        while let Some((frame_type, payload)) =
            read_frame(stream, &mut read_buf).map_err(|error| error.to_string())?
        {
            // A repeated HELLO re-acks with fresh produced_rows so a
            // long-lived trainer connection can watch production advance.
            if frame_type == FRAME_HELLO {
                if let Err(error) = self.handle_hello(payload, stream, &mut write_buf) {
                    send_error(stream, &mut write_buf, error.0, error.1)?;
                    return Ok(());
                }
                continue;
            }
            if frame_type != FRAME_SAMPLE {
                send_error(stream, &mut write_buf, ERROR_PROTOCOL, "expected SAMPLE")?;
                return Ok(());
            }
            match self.handle_sample(payload, &mut batch_buf, &mut target_buf) {
                Ok(()) => {
                    let gzfb_len = (batch_buf.len() as u32).to_le_bytes();
                    write_frame(
                        stream,
                        &mut write_buf,
                        FRAME_SAMPLE_RESULT,
                        &[&gzfb_len, &batch_buf, &target_buf],
                    )
                    .map_err(|error| error.to_string())?;
                }
                Err(error) => {
                    send_error(stream, &mut write_buf, error.0, error.1)?;
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn handle_hello(
        &self,
        payload: &[u8],
        stream: &mut UnixStream,
        write_buf: &mut Vec<u8>,
    ) -> Result<(), (u32, &'static str)> {
        if payload.len() != 8 {
            return Err((ERROR_PROTOCOL, "bad HELLO length"));
        }
        let protocol_version = u32::from_le_bytes(payload[0..4].try_into().expect("len checked"));
        let encoding_version = u32::from_le_bytes(payload[4..8].try_into().expect("len checked"));
        if protocol_version != SAMPLE_PROTOCOL_VERSION {
            return Err((ERROR_PROTOCOL, "protocol version mismatch"));
        }
        if encoding_version != ENCODING_VERSION {
            return Err((ERROR_ENCODING, "encoding version mismatch"));
        }

        let mut schema_config = Vec::new();
        encode_feature_schema_config(self.collator.schema().config(), &mut schema_config)
            .map_err(|_| (ERROR_ENCODING, "failed to encode schema config"))?;
        let (episodes, episodes_stopped) = self.store.episode_counters();
        // Unseeded EMAs surface as zeros; consumers gate on the episode count.
        let (cost_ema, len_ema, stop_ema) = self.store.outcome_emas().unwrap_or((0.0, 0.0, 0.0));
        // Unseeded surfaces as -1.0: 0.0 is a legitimate all-loss rate.
        let win_ema = self.store.win_rate_ema().unwrap_or(-1.0);
        let latency_ema = self.store.episode_latency_ema().unwrap_or(-1.0);
        let (value_sign_early_ema, value_sign_late_ema) = self.store.value_sign_accuracy_emas();
        let value_sign_early_ema = value_sign_early_ema.unwrap_or(-1.0);
        let value_sign_late_ema = value_sign_late_ema.unwrap_or(-1.0);
        let best_cost = self.store.best_cost().unwrap_or(0.0);
        let symmetric = self.store.symmetric_selfplay_metrics();
        let root_info = self
            .store
            .root_info()
            .map_err(|_| (ERROR_ENCODING, "corrupt root info"))?;
        let counters = self.store.counters();
        let mut payload = Vec::with_capacity(HELLO_ACK_FIXED_LEN + schema_config.len());
        payload.extend_from_slice(&SAMPLE_PROTOCOL_VERSION.to_le_bytes());
        payload.extend_from_slice(self.collator.schema().hash().as_bytes());
        payload.extend_from_slice(&(self.max_batch.get() as u32).to_le_bytes());
        payload.extend_from_slice(&counters.produced_rows.to_le_bytes());
        payload.extend_from_slice(&episodes.to_le_bytes());
        payload.extend_from_slice(&episodes_stopped.to_le_bytes());
        payload.extend_from_slice(&(cost_ema as f32).to_le_bytes());
        payload.extend_from_slice(&(len_ema as f32).to_le_bytes());
        payload.extend_from_slice(&(stop_ema as f32).to_le_bytes());
        payload.extend_from_slice(&(win_ema as f32).to_le_bytes());
        payload.extend_from_slice(&(latency_ema as f32).to_le_bytes());
        payload.extend_from_slice(&(best_cost as f32).to_le_bytes());
        payload.extend_from_slice(&u32::from(root_info.is_some()).to_le_bytes());
        let root = root_info.unwrap_or(gz_replay::ReplayRootInfo {
            cost: 0.0,
            node_count: 0,
            edge_count: 0,
            candidate_count: 0,
        });
        payload.extend_from_slice(&root.cost.to_le_bytes());
        payload.extend_from_slice(&root.node_count.to_le_bytes());
        payload.extend_from_slice(&root.edge_count.to_le_bytes());
        payload.extend_from_slice(&root.candidate_count.to_le_bytes());
        payload.extend_from_slice(&counters.produced_policy_rows.to_le_bytes());
        payload.extend_from_slice(&counters.produced_value_rows.to_le_bytes());
        payload.extend_from_slice(&(value_sign_early_ema as f32).to_le_bytes());
        payload.extend_from_slice(&(value_sign_late_ema as f32).to_le_bytes());
        payload.extend_from_slice(&u32::from(symmetric.is_some()).to_le_bytes());
        let symmetric = symmetric.map_or([0.0; 10], |metrics| {
            [
                metrics.p1_win_rate_ema,
                metrics.p2_win_rate_ema,
                metrics.draw_rate_ema,
                metrics.p1_terminal_cost_ema,
                metrics.p2_terminal_cost_ema,
                metrics.terminal_cost_margin_ema,
                metrics.terminal_cost_best,
                metrics.p1_episode_len_ema,
                metrics.p2_episode_len_ema,
                metrics.episode_len_margin_ema,
            ]
        });
        for value in symmetric {
            payload.extend_from_slice(&(value as f32).to_le_bytes());
        }
        payload.extend_from_slice(&schema_config);
        write_frame(stream, write_buf, FRAME_HELLO_ACK, &[&payload])
            .map_err(|_| (ERROR_PROTOCOL, "failed to write HELLO_ACK"))
    }

    fn handle_sample(
        &mut self,
        payload: &[u8],
        batch_buf: &mut Vec<u8>,
        target_buf: &mut Vec<u8>,
    ) -> Result<(), (u32, &'static str)> {
        if payload.len() != 24 {
            return Err((ERROR_PROTOCOL, "bad SAMPLE length"));
        }
        let batch = u32::from_le_bytes(payload[0..4].try_into().expect("len checked")) as usize;
        let kind = match u32::from_le_bytes(payload[4..8].try_into().expect("len checked")) {
            0 => SampleKind::Any,
            1 => SampleKind::Policy,
            2 => SampleKind::Value,
            _ => return Err((ERROR_BAD_REQUEST, "invalid SAMPLE kind")),
        };
        let window = u64::from_le_bytes(payload[8..16].try_into().expect("len checked"));
        let seed = u64::from_le_bytes(payload[16..24].try_into().expect("len checked"));
        if batch == 0 || batch > self.max_batch.get() || window == 0 {
            return Err((ERROR_BAD_REQUEST, "invalid SAMPLE request"));
        }

        let rows = self
            .store
            .sample_rows_kind(
                SampleConfig {
                    batch: NonZeroUsize::new(batch).expect("batch checked"),
                    window_rows: NonZeroU64::new(window).expect("window checked"),
                    seed,
                },
                kind,
            )
            .map_err(sample_error)?;
        let mut feature_rows = Vec::<FeatureRow>::with_capacity(rows.len());
        let mut targets = Vec::<RowTargets>::with_capacity(rows.len());
        let schema_hash = self.collator.schema().hash();

        for (_, row) in rows {
            let bytes = row
                .feature_row
                .ok_or((ERROR_MISSING_FEATURES, "row is missing feature payload"))?;
            validate_feature_row_header(&bytes, &schema_hash)
                .map_err(|_| (ERROR_ENCODING, "feature row schema mismatch"))?;
            let feature_row =
                decode_feature_row(&bytes).map_err(|_| (ERROR_ENCODING, "bad feature row"))?;
            let reward = row
                .reward_target
                .ok_or((ERROR_ENCODING, "missing reward target"))?;
            targets.push(RowTargets {
                policy: row.policy_target,
                value: row.value_target,
                horizon_value: row.horizon_value_targets,
                reward,
            });
            feature_rows.push(feature_row);
        }

        self.collator
            .collate_into(&feature_rows, batch_buf)
            .map_err(|_| (ERROR_ENCODING, "feature collation failed"))?;
        encode_training_targets(
            &targets,
            self.max_batch.get(),
            self.collator.schema().config().max_actions as usize,
            target_buf,
        )
        .map_err(|_| (ERROR_ENCODING, "target encoding failed"))?;

        Ok(())
    }
}

fn sample_error(error: ReplayError) -> (u32, &'static str) {
    match error {
        ReplayError::Empty => (ERROR_EMPTY_STORE, "replay store is empty"),
        ReplayError::DataModeMismatch => (ERROR_BAD_REQUEST, "replay data mode mismatch"),
        _ => (ERROR_BAD_REQUEST, "sampling failed"),
    }
}

fn read_frame<'a>(
    stream: &mut UnixStream,
    buf: &'a mut Vec<u8>,
) -> std::io::Result<Option<(u8, &'a [u8])>> {
    let mut len = [0u8; 4];
    match stream.read_exact(&mut len) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    let body_len = u32::from_le_bytes(len) as usize;
    if body_len == 0 || body_len > MAX_FRAME {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "bad frame size",
        ));
    }

    if buf.len() < body_len {
        buf.resize(body_len, 0);
    }
    stream.read_exact(&mut buf[..body_len])?;
    Ok(Some((buf[0], &buf[1..body_len])))
}

fn write_frame(
    stream: &mut UnixStream,
    buf: &mut Vec<u8>,
    frame_type: u8,
    parts: &[&[u8]],
) -> std::io::Result<()> {
    let body_len = parts
        .iter()
        .try_fold(1usize, |total, part| total.checked_add(part.len()))
        .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "frame length overflow"))?;
    if body_len > MAX_FRAME {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "frame exceeds maximum size",
        ));
    }
    let frame_len = 4 + body_len;
    if buf.len() < frame_len {
        buf.resize(frame_len, 0);
    }
    buf[0..4].copy_from_slice(&(body_len as u32).to_le_bytes());
    buf[4] = frame_type;
    let mut cursor = 5;
    for part in parts {
        let end = cursor + part.len();
        buf[cursor..end].copy_from_slice(part);
        cursor = end;
    }
    stream.write_all(&buf[..frame_len])
}

fn send_error(
    stream: &mut UnixStream,
    write_buf: &mut Vec<u8>,
    code: u32,
    message: &'static str,
) -> Result<(), String> {
    let message = truncate_message(message);
    let mut payload = Vec::with_capacity(6 + message.len());
    payload.extend_from_slice(&code.to_le_bytes());
    payload.extend_from_slice(&(message.len() as u16).to_le_bytes());
    payload.extend_from_slice(message.as_bytes());
    write_frame(stream, write_buf, FRAME_ERROR, &[&payload]).map_err(|error| error.to_string())
}

fn truncate_message(message: &'static str) -> &'static str {
    const MAX: usize = 512;
    if message.len() <= MAX {
        message
    } else {
        &message[..MAX]
    }
}
