use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_META, CF_ROW_INDEX, CF_ROWS, META_COMPLETED_GAMES, META_CONSUMED_ROWS,
    META_DATA_MODE, META_DELETED_FLOOR, META_EPISODES_STOPPED, META_FEATURE_SCHEMA,
    META_NEXT_EPISODE_SEQ, META_PRODUCED_POLICY_ROWS, META_PRODUCED_ROWS, META_RETAINED_FLOOR,
    META_ROOT_INFO, META_SCHEMA_VERSION, META_TERMINAL_COST_BEST, META_TERMINAL_COST_EMA,
    SCHEMA_VERSION, decode_episode_from_row_key, decode_step_from_row_key, decode_u32, decode_u64,
    decode_u64_key, encode_u32, encode_u64, episode_key, row_index_key, row_key,
};
use crate::records::{
    ReplayEpisodeId, ReplayEpisodeRecord, ReplayRootInfo, ReplayRow, StoredReplayRow,
    validate_episode,
};
use crate::sample::{ReplayRng, SampleConfig, SampleKind};
use gz_features::{FeatureSchema, FeatureSchemaConfig};
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamilyDescriptor, DB, DBCompressionType, IteratorMode, Options,
    WriteBatch,
};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub struct ReplayStore {
    db: Arc<DB>,
    write_lock: Mutex<()>,
    consumed_lock: Mutex<()>,
    data_mode: Mutex<Option<ReplayDataMode>>,
    next_episode_seq: AtomicU64,
    completed_games: AtomicU64,
    episodes_stopped: AtomicU64,
    produced_rows: AtomicU64,
    produced_policy_rows: AtomicU64,
    consumed_rows: AtomicU64,
    /// Rows below this sequence may be gone; sampling clamps to it.
    retained_floor: AtomicU64,
    retain_rows: Option<u64>,
    /// Episode-weighted EMAs over recent appends (decay 0.99), stored as
    /// f64 bits; zero bits = unseeded. Terminal cost persists across reopen.
    cost_ema_bits: AtomicU64,
    len_ema_bits: AtomicU64,
    stop_ema_bits: AtomicU64,
    /// EMA of value_target > 0 over labeled appends only: the
    /// episode-weighted learner win rate.
    win_ema_bits: AtomicU64,
    /// EMA of admission-to-completion wall seconds, fed by lanes at
    /// episode completion: the async lag's queueing term.
    latency_ema_bits: AtomicU64,
    best_cost_bits: AtomicU64,
}

const OUTCOME_EMA_DECAY: f64 = 0.99;

fn next_ema_bits(previous: u64, value: f64) -> u64 {
    let next = if previous == 0 {
        value
    } else {
        OUTCOME_EMA_DECAY * f64::from_bits(previous) + (1.0 - OUTCOME_EMA_DECAY) * value
    };
    next.to_bits()
}

fn next_best_cost_bits(previous: u64, cost: f64) -> u64 {
    if previous == 0 || cost < f64::from_bits(previous) {
        cost.to_bits()
    } else {
        previous
    }
}

fn sample_kind_matches(kind: SampleKind, row: &ReplayRow) -> bool {
    match kind {
        SampleKind::Any => true,
        SampleKind::Policy => row.policy_target.iter().any(|target| *target > 0.0),
        SampleKind::Value => row.value_target.is_some(),
    }
}

fn policy_row_count(rows: &[ReplayRow]) -> ReplayResult<u64> {
    u64::try_from(
        rows.iter()
            .filter(|row| sample_kind_matches(SampleKind::Policy, row))
            .count(),
    )
    .map_err(|_| ReplayError::InvalidRecord)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCounters {
    pub produced_rows: u64,
    pub produced_policy_rows: u64,
    pub consumed_rows: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayDataMode {
    Standard,
    SampledTree,
    Graded {
        sampled_tree: bool,
        reward_scale_bits: u32,
    },
}

impl ReplayDataMode {
    pub fn graded(sampled_tree: bool, reward_scale: f32) -> ReplayResult<Self> {
        if !reward_scale.is_finite() || reward_scale <= 0.0 {
            return Err(ReplayError::InvalidRecord);
        }
        Ok(Self::Graded {
            sampled_tree,
            reward_scale_bits: reward_scale.to_bits(),
        })
    }

    #[must_use]
    pub const fn is_graded(self) -> bool {
        matches!(self, Self::Graded { .. })
    }

    fn bytes(self) -> Vec<u8> {
        match self {
            Self::Standard => b"standard-v1".to_vec(),
            Self::SampledTree => b"sampled-tree-v1".to_vec(),
            Self::Graded {
                sampled_tree,
                reward_scale_bits,
            } => {
                let mut bytes = b"graded-v1".to_vec();
                bytes.push(u8::from(sampled_tree));
                bytes.extend_from_slice(&reward_scale_bits.to_le_bytes());
                bytes
            }
        }
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        match bytes {
            b"standard-v1" => Some(Self::Standard),
            b"sampled-tree-v1" => Some(Self::SampledTree),
            _ if bytes.len() == 14 && bytes.starts_with(b"graded-v1") && bytes[9] <= 1 => {
                let reward_scale_bits = u32::from_le_bytes(bytes[10..14].try_into().ok()?);
                let reward_scale = f32::from_bits(reward_scale_bits);
                (reward_scale.is_finite() && reward_scale > 0.0).then_some(Self::Graded {
                    sampled_tree: bytes[9] != 0,
                    reward_scale_bits,
                })
            }
            _ => None,
        }
    }
}

impl ReplayStore {
    pub fn open(path: &Path) -> ReplayResult<Self> {
        Self::open_with_retention(path, None)
    }

    /// `retain_rows` bounds the store: once produced rows exceed the bound
    /// by 25%, whole episodes below `produced - retain_rows` are
    /// range-deleted and the sampling window clamps to the new floor.
    pub fn open_with_retention(path: &Path, retain_rows: Option<u64>) -> ReplayResult<Self> {
        let db = Arc::new(open_db(path)?);
        ensure_schema(&db)?;

        let next_episode_seq = recover_next_episode_seq(&db)?;
        let produced_rows = recover_next_row_seq(&db)?;
        let produced_policy_rows =
            read_meta_u64(&db, META_PRODUCED_POLICY_ROWS)?.unwrap_or(produced_rows);
        let consumed_rows = read_meta_u64(&db, META_CONSUMED_ROWS)?.unwrap_or(0);
        let completed_games = read_meta_u64(&db, META_COMPLETED_GAMES)?.unwrap_or(next_episode_seq);
        let episodes_stopped = read_meta_u64(&db, META_EPISODES_STOPPED)?.unwrap_or(0);
        let retained_floor = read_meta_u64(&db, META_RETAINED_FLOOR)?.unwrap_or(0);
        let cost_ema_bits = read_meta_u64(&db, META_TERMINAL_COST_EMA)?.unwrap_or(0);
        let best_cost_bits = read_meta_u64(&db, META_TERMINAL_COST_BEST)?.unwrap_or(0);
        let data_mode = read_data_mode(&db)?;
        write_meta_u64(&db, META_NEXT_EPISODE_SEQ, next_episode_seq)?;
        write_meta_u64(&db, META_PRODUCED_ROWS, produced_rows)?;
        write_meta_u64(&db, META_PRODUCED_POLICY_ROWS, produced_policy_rows)?;
        write_meta_u64(&db, META_COMPLETED_GAMES, completed_games)?;

        Ok(Self {
            db,
            write_lock: Mutex::new(()),
            consumed_lock: Mutex::new(()),
            data_mode: Mutex::new(data_mode),
            next_episode_seq: AtomicU64::new(next_episode_seq),
            completed_games: AtomicU64::new(completed_games),
            episodes_stopped: AtomicU64::new(episodes_stopped),
            produced_rows: AtomicU64::new(produced_rows),
            produced_policy_rows: AtomicU64::new(produced_policy_rows),
            consumed_rows: AtomicU64::new(consumed_rows),
            retained_floor: AtomicU64::new(retained_floor),
            retain_rows,
            cost_ema_bits: AtomicU64::new(cost_ema_bits),
            win_ema_bits: AtomicU64::new(0),
            latency_ema_bits: AtomicU64::new(0),
            len_ema_bits: AtomicU64::new(0),
            stop_ema_bits: AtomicU64::new(0),
            best_cost_bits: AtomicU64::new(best_cost_bits),
        })
    }

    pub fn append_episode(
        &self,
        record: &ReplayEpisodeRecord,
        rows: &[ReplayRow],
    ) -> ReplayResult<ReplayEpisodeId> {
        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let feature_schema = read_feature_schema(&self.db)?;
        let feature_schema_hash = feature_schema
            .as_ref()
            .map(|config| FeatureSchema::new(config.clone()).map(|schema| schema.hash()))
            .transpose()
            .map_err(|_| ReplayError::InvalidRecord)?;
        validate_episode(record, rows, feature_schema_hash, self.data_mode()?)?;

        let episode_seq = self.next_episode_seq.load(Ordering::Acquire);
        let row_seq = self.produced_rows.load(Ordering::Acquire);
        let next_episode_seq = episode_seq
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("episode id overflow"))?;
        let produced_rows = row_seq
            .checked_add(rows.len() as u64)
            .ok_or(ReplayError::InvalidRecord)?;
        let produced_policy_rows = self
            .produced_policy_rows
            .load(Ordering::Acquire)
            .checked_add(policy_row_count(rows)?)
            .ok_or(ReplayError::InvalidRecord)?;
        let id = ReplayEpisodeId::new(episode_seq);
        let completed_games = self
            .completed_games
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("completed game counter overflow"))?;
        let cost = f64::from(-record.outcome.learner_reward);
        let cost_ema_bits = next_ema_bits(self.cost_ema_bits.load(Ordering::Acquire), cost);
        let best_cost_bits = next_best_cost_bits(self.best_cost_bits.load(Ordering::Acquire), cost);

        let episodes = self.cf(CF_EPISODES)?;
        let row_cf = self.cf(CF_ROWS)?;
        let row_index = self.cf(CF_ROW_INDEX)?;
        let meta = self.cf(CF_META)?;
        let mut batch = WriteBatch::default();

        batch.put_cf(
            &episodes,
            episode_key(episode_seq),
            postcard::to_allocvec(record)?,
        );

        for (offset, row) in rows.iter().enumerate() {
            let key = row_key(episode_seq, row.step_index);
            batch.put_cf(
                &row_cf,
                key,
                postcard::to_allocvec(&StoredReplayRow::from_row(row)?)?,
            );
            batch.put_cf(
                &row_index,
                row_index_key(row_seq + offset as u64),
                key.as_slice(),
            );
        }

        batch.put_cf(&meta, META_NEXT_EPISODE_SEQ, encode_u64(next_episode_seq));
        batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(produced_rows));
        batch.put_cf(
            &meta,
            META_PRODUCED_POLICY_ROWS,
            encode_u64(produced_policy_rows),
        );
        batch.put_cf(&meta, META_COMPLETED_GAMES, encode_u64(completed_games));
        batch.put_cf(&meta, META_TERMINAL_COST_EMA, encode_u64(cost_ema_bits));
        batch.put_cf(&meta, META_TERMINAL_COST_BEST, encode_u64(best_cost_bits));
        let episodes_stopped =
            self.episodes_stopped.load(Ordering::Acquire) + u64::from(record.outcome.stopped);
        batch.put_cf(&meta, META_EPISODES_STOPPED, encode_u64(episodes_stopped));
        self.db.write(batch)?;
        self.episodes_stopped
            .store(episodes_stopped, Ordering::Release);
        self.next_episode_seq
            .store(next_episode_seq, Ordering::Release);
        self.completed_games
            .store(completed_games, Ordering::Release);
        self.produced_rows.store(produced_rows, Ordering::Release);
        self.produced_policy_rows
            .store(produced_policy_rows, Ordering::Release);
        self.cost_ema_bits.store(cost_ema_bits, Ordering::Release);
        self.best_cost_bits.store(best_cost_bits, Ordering::Release);
        self.enforce_retention(produced_rows)?;
        self.update_ema(&self.len_ema_bits, rows.len() as f64);
        self.update_ema(
            &self.stop_ema_bits,
            f64::from(u8::from(record.outcome.stopped)),
        );
        if let Some(value_target) = record.outcome.value_target {
            // Stored biased by +1.0: an honest all-loss EMA of 0.0 would
            // collide with the zero-bits unseeded sentinel.
            self.update_ema(
                &self.win_ema_bits,
                f64::from(u8::from(value_target > 0.0)) + 1.0,
            );
        }
        Ok(id)
    }

    /// Atomically appends both perspectives of one competitive game. The
    /// primary record is the learner for episode-level telemetry; row and
    /// storage counters include both records, while game counters advance once.
    pub fn append_episode_pair(
        &self,
        primary: (&ReplayEpisodeRecord, &[ReplayRow]),
        secondary: (&ReplayEpisodeRecord, &[ReplayRow]),
    ) -> ReplayResult<(ReplayEpisodeId, ReplayEpisodeId)> {
        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let feature_schema = read_feature_schema(&self.db)?;
        let feature_schema_hash = feature_schema
            .as_ref()
            .map(|config| FeatureSchema::new(config.clone()).map(|schema| schema.hash()))
            .transpose()
            .map_err(|_| ReplayError::InvalidRecord)?;
        let data_mode = self.data_mode()?;
        validate_episode(primary.0, primary.1, feature_schema_hash, data_mode)?;
        validate_episode(secondary.0, secondary.1, feature_schema_hash, data_mode)?;

        let first_seq = self.next_episode_seq.load(Ordering::Acquire);
        let second_seq = first_seq
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("episode id overflow"))?;
        let next_episode_seq = first_seq
            .checked_add(2)
            .ok_or_else(|| ReplayError::storage("episode id overflow"))?;
        let row_seq = self.produced_rows.load(Ordering::Acquire);
        let primary_rows =
            u64::try_from(primary.1.len()).map_err(|_| ReplayError::InvalidRecord)?;
        let total_rows = primary_rows
            .checked_add(u64::try_from(secondary.1.len()).map_err(|_| ReplayError::InvalidRecord)?)
            .ok_or(ReplayError::InvalidRecord)?;
        let produced_rows = row_seq
            .checked_add(total_rows)
            .ok_or(ReplayError::InvalidRecord)?;
        let produced_policy_rows = self
            .produced_policy_rows
            .load(Ordering::Acquire)
            .checked_add(
                policy_row_count(primary.1)?
                    .checked_add(policy_row_count(secondary.1)?)
                    .ok_or(ReplayError::InvalidRecord)?,
            )
            .ok_or(ReplayError::InvalidRecord)?;
        let completed_games = self
            .completed_games
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("completed game counter overflow"))?;
        let cost = f64::from(-primary.0.outcome.learner_reward);
        let cost_ema_bits = next_ema_bits(self.cost_ema_bits.load(Ordering::Acquire), cost);
        let best_cost_bits = next_best_cost_bits(self.best_cost_bits.load(Ordering::Acquire), cost);

        let episodes = self.cf(CF_EPISODES)?;
        let row_cf = self.cf(CF_ROWS)?;
        let row_index = self.cf(CF_ROW_INDEX)?;
        let meta = self.cf(CF_META)?;
        let mut batch = WriteBatch::default();
        for (episode_seq, row_base, (record, rows)) in [
            (first_seq, row_seq, primary),
            (second_seq, row_seq + primary_rows, secondary),
        ] {
            batch.put_cf(
                &episodes,
                episode_key(episode_seq),
                postcard::to_allocvec(record)?,
            );
            for (offset, row) in rows.iter().enumerate() {
                let key = row_key(episode_seq, row.step_index);
                batch.put_cf(
                    &row_cf,
                    key,
                    postcard::to_allocvec(&StoredReplayRow::from_row(row)?)?,
                );
                batch.put_cf(
                    &row_index,
                    row_index_key(row_base + offset as u64),
                    key.as_slice(),
                );
            }
        }

        let episodes_stopped =
            self.episodes_stopped.load(Ordering::Acquire) + u64::from(primary.0.outcome.stopped);
        batch.put_cf(&meta, META_NEXT_EPISODE_SEQ, encode_u64(next_episode_seq));
        batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(produced_rows));
        batch.put_cf(
            &meta,
            META_PRODUCED_POLICY_ROWS,
            encode_u64(produced_policy_rows),
        );
        batch.put_cf(&meta, META_COMPLETED_GAMES, encode_u64(completed_games));
        batch.put_cf(&meta, META_EPISODES_STOPPED, encode_u64(episodes_stopped));
        batch.put_cf(&meta, META_TERMINAL_COST_EMA, encode_u64(cost_ema_bits));
        batch.put_cf(&meta, META_TERMINAL_COST_BEST, encode_u64(best_cost_bits));
        self.db.write(batch)?;

        self.next_episode_seq
            .store(next_episode_seq, Ordering::Release);
        self.completed_games
            .store(completed_games, Ordering::Release);
        self.episodes_stopped
            .store(episodes_stopped, Ordering::Release);
        self.produced_rows.store(produced_rows, Ordering::Release);
        self.produced_policy_rows
            .store(produced_policy_rows, Ordering::Release);
        self.cost_ema_bits.store(cost_ema_bits, Ordering::Release);
        self.best_cost_bits.store(best_cost_bits, Ordering::Release);
        self.enforce_retention(produced_rows)?;

        self.update_ema(&self.len_ema_bits, primary.1.len() as f64);
        self.update_ema(
            &self.stop_ema_bits,
            f64::from(u8::from(primary.0.outcome.stopped)),
        );
        if let Some(value_target) = primary.0.outcome.value_target {
            self.update_ema(
                &self.win_ema_bits,
                f64::from(u8::from(value_target > 0.0)) + 1.0,
            );
        }
        Ok((
            ReplayEpisodeId::new(first_seq),
            ReplayEpisodeId::new(second_seq),
        ))
    }

    /// Runs under the append write lock. Two floors make this safe against
    /// lock-free samplers: keys are only deleted below the floor published
    /// on the PREVIOUS cycle, and any in-flight sampler already clamped to
    /// at least that floor before picking row sequences.
    fn enforce_retention(&self, produced_rows: u64) -> ReplayResult<()> {
        let Some(retain) = self.retain_rows else {
            return Ok(());
        };
        let floor = self.retained_floor.load(Ordering::Acquire);
        if produced_rows.saturating_sub(floor) <= retain + retain / 4 {
            return Ok(());
        }

        let row_index = self.cf(CF_ROW_INDEX)?;
        let target = produced_rows - retain;
        let target_key = self
            .db
            .get_cf(&row_index, row_index_key(target))?
            .ok_or_else(|| ReplayError::storage("missing row index entry at retention target"))?;
        let step = decode_step_from_row_key(&target_key)
            .ok_or_else(|| ReplayError::storage("corrupt row key at retention target"))?;
        // Align the floor to the cutoff episode's first row so episodes are
        // deleted whole.
        let new_floor = target - u64::from(step);
        if new_floor <= floor {
            return Ok(());
        }

        let deleted = read_meta_u64(&self.db, META_DELETED_FLOOR)?.unwrap_or(0);
        let deleted_episode = if deleted == 0 {
            0
        } else {
            let key = self
                .db
                .get_cf(&row_index, row_index_key(deleted))?
                .ok_or_else(|| ReplayError::storage("missing row index entry at deleted floor"))?;
            decode_episode_from_row_key(&key)
                .ok_or_else(|| ReplayError::storage("corrupt row key at deleted floor"))?
        };
        let floor_episode = if floor == 0 {
            0
        } else {
            let key = self
                .db
                .get_cf(&row_index, row_index_key(floor))?
                .ok_or_else(|| ReplayError::storage("missing row index entry at retained floor"))?;
            decode_episode_from_row_key(&key)
                .ok_or_else(|| ReplayError::storage("corrupt row key at retained floor"))?
        };

        let rows = self.cf(CF_ROWS)?;
        let episodes = self.cf(CF_EPISODES)?;
        let mut batch = WriteBatch::default();
        batch.delete_range_cf(&row_index, row_index_key(deleted), row_index_key(floor));
        batch.delete_range_cf(
            &rows,
            row_key(deleted_episode, 0),
            row_key(floor_episode, 0),
        );
        batch.delete_range_cf(
            &episodes,
            episode_key(deleted_episode),
            episode_key(floor_episode),
        );
        let meta = self.cf(CF_META)?;
        batch.put_cf(&meta, META_DELETED_FLOOR, encode_u64(floor));
        batch.put_cf(&meta, META_RETAINED_FLOOR, encode_u64(new_floor));
        self.db.write(batch)?;
        self.retained_floor.store(new_floor, Ordering::Release);

        Ok(())
    }

    pub fn ensure_feature_schema(&self, config: &FeatureSchemaConfig) -> ReplayResult<()> {
        FeatureSchema::new(config.clone()).map_err(|_| ReplayError::InvalidRecord)?;

        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let Some(stored) = read_feature_schema(&self.db)? else {
            let meta = self.cf(CF_META)?;
            self.db
                .put_cf(
                    &meta,
                    META_FEATURE_SCHEMA,
                    postcard::to_allocvec(&StoredFeatureSchemaConfig::from(config))?,
                )
                .map_err(ReplayError::from)?;
            return Ok(());
        };

        if &stored == config {
            Ok(())
        } else {
            Err(ReplayError::InvalidRecord)
        }
    }

    pub fn ensure_data_mode(&self, mode: ReplayDataMode) -> ReplayResult<()> {
        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let meta = self.cf(CF_META)?;
        let encoded = mode.bytes();
        match self.db.get_cf(&meta, META_DATA_MODE)? {
            Some(stored) if stored.as_slice() == encoded => {}
            Some(_) => return Err(ReplayError::DataModeMismatch),
            None => {
                if self.next_episode_seq.load(Ordering::Acquire) > 0
                    && mode != ReplayDataMode::Standard
                {
                    return Err(ReplayError::DataModeMismatch);
                }
                self.db
                    .put_cf(&meta, META_DATA_MODE, encoded)
                    .map_err(ReplayError::from)?;
            }
        }
        *self.data_mode.lock().map_err(ReplayError::storage)? = Some(mode);
        Ok(())
    }

    pub fn feature_schema(&self) -> ReplayResult<Option<FeatureSchemaConfig>> {
        read_feature_schema(&self.db)
    }

    pub fn episode(&self, id: ReplayEpisodeId) -> ReplayResult<Option<ReplayEpisodeRecord>> {
        let episodes = self.cf(CF_EPISODES)?;
        self.db
            .get_cf(&episodes, episode_key(id.get()))?
            .map(|bytes| postcard::from_bytes(&bytes).map_err(ReplayError::from))
            .transpose()
    }

    pub fn sample_rows(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        self.sample_rows_kind(config, SampleKind::Any)
    }

    pub fn sample_rows_kind(
        &self,
        config: SampleConfig,
        kind: SampleKind,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        // Lock-free against producers: appends commit their WriteBatch before
        // publishing produced_rows, so every sampled row_seq is fully visible.
        let produced = self.produced_rows.load(Ordering::Acquire);
        let floor = self.retained_floor.load(Ordering::Acquire);

        if produced == floor || produced == 0 {
            return Err(ReplayError::Empty);
        }

        let window = config.window_rows.get().min(produced - floor);
        let start = produced - window;
        let mut rng = ReplayRng::new(config.seed);
        let batch = config.batch.get();

        if kind != SampleKind::Any {
            let mut out = Vec::with_capacity(batch);
            for _ in 0..16 {
                if out.len() >= batch {
                    break;
                }
                let remaining = batch - out.len();
                let draw_count = remaining.saturating_mul(2).max(16);
                let sequences = (0..draw_count)
                    .map(|_| start + rng.next_bounded(window))
                    .collect::<Vec<_>>();
                out.extend(
                    self.read_row_sequences(&sequences)?
                        .into_iter()
                        .filter(|(_, row)| sample_kind_matches(kind, row))
                        .take(remaining),
                );
            }

            if out.len() < batch {
                // The normal rejection path stays lock-free. A sparse stream
                // falls back to a full-window scan, which may span multiple
                // retention cycles, so pin appends and recompute the floor for
                // this rare path.
                let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
                let produced = self.produced_rows.load(Ordering::Acquire);
                let floor = self.retained_floor.load(Ordering::Acquire);
                let window = config.window_rows.get().min(produced - floor);
                let start = produced - window;
                let mut eligible = Vec::new();
                let mut offset = 0;
                while offset < window {
                    let count = (window - offset).min(512);
                    let sequences = (0..count)
                        .map(|index| start + offset + index)
                        .collect::<Vec<_>>();
                    eligible.extend(
                        self.read_row_sequences(&sequences)?
                            .into_iter()
                            .filter(|(_, row)| sample_kind_matches(kind, row)),
                    );
                    offset += count;
                }
                if eligible.is_empty() {
                    return Err(ReplayError::Empty);
                }
                while out.len() < batch {
                    let index = rng.next_bounded(eligible.len() as u64) as usize;
                    out.push(eligible[index].clone());
                }
            }

            self.record_consumed(config.batch.get() as u64)?;
            return Ok(out);
        }

        // Two batched MultiGet phases instead of 2 x batch sequential
        // point gets: index keys resolve to row keys, row keys resolve to
        // rows. Results return in request order, so sampling stays
        // bit-identical to the sequential loop for a given seed.
        let sequences = (0..batch)
            .map(|_| start + rng.next_bounded(window))
            .collect::<Vec<_>>();
        let out = self.read_row_sequences(&sequences)?;
        self.record_consumed(config.batch.get() as u64)?;
        Ok(out)
    }

    fn read_row_sequences(
        &self,
        sequences: &[u64],
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        let row_index = self.cf(CF_ROW_INDEX)?;
        let rows = self.cf(CF_ROWS)?;
        let index_keys = sequences
            .iter()
            .copied()
            .map(row_index_key)
            .collect::<Vec<_>>();
        let mut row_keys = Vec::with_capacity(sequences.len());
        let mut episode_seqs = Vec::with_capacity(sequences.len());
        for result in self.db.batched_multi_get_cf(&row_index, &index_keys, false) {
            let row_key = result?.ok_or_else(|| ReplayError::storage("missing row index entry"))?;
            let episode_seq = decode_episode_from_row_key(&row_key)
                .ok_or_else(|| ReplayError::storage("corrupt row key"))?;
            episode_seqs.push(episode_seq);
            row_keys.push(row_key);
        }

        let mut out = Vec::with_capacity(sequences.len());
        for (episode_seq, result) in episode_seqs.into_iter().zip(self.db.batched_multi_get_cf(
            &rows,
            row_keys.iter(),
            false,
        )) {
            let row = result?.ok_or_else(|| ReplayError::storage("missing replay row"))?;
            out.push((
                ReplayEpisodeId::new(episode_seq),
                postcard::from_bytes::<StoredReplayRow>(&row)?.into_row(),
            ));
        }
        Ok(out)
    }

    fn record_consumed(&self, count: u64) -> ReplayResult<()> {
        // Concurrent sample sessions may finish out of order. Serialize the
        // metadata update so a late write cannot persist an older counter.
        let _guard = self.consumed_lock.lock().map_err(ReplayError::storage)?;
        let consumed_rows = self
            .consumed_rows
            .load(Ordering::Acquire)
            .checked_add(count)
            .ok_or_else(|| ReplayError::storage("consumed row counter overflow"))?;
        write_meta_u64(&self.db, META_CONSUMED_ROWS, consumed_rows)?;
        self.consumed_rows.store(consumed_rows, Ordering::Release);
        Ok(())
    }

    fn update_ema(&self, bits: &AtomicU64, value: f64) {
        let previous = bits.load(Ordering::Acquire);
        bits.store(next_ema_bits(previous, value), Ordering::Release);
    }

    /// Episode-weighted EMAs over recent appends:
    /// (terminal cost, episode length, stop rate). None until seeded.
    #[must_use]
    pub fn outcome_emas(&self) -> Option<(f64, f64, f64)> {
        let cost = self.cost_ema_bits.load(Ordering::Acquire);
        if cost == 0 {
            return None;
        }
        Some((
            f64::from_bits(cost),
            f64::from_bits(self.len_ema_bits.load(Ordering::Acquire)),
            f64::from_bits(self.stop_ema_bits.load(Ordering::Acquire)),
        ))
    }

    /// Observe one episode's admission-to-completion wall time. Called
    /// by lanes at completion (dropped episodes included: their latency
    /// is real even when their rows are not stored).
    pub fn observe_episode_latency(&self, seconds: f64) {
        if seconds > 0.0 && seconds.is_finite() {
            self.update_ema(&self.latency_ema_bits, seconds);
        }
    }

    /// EMA of episode wall-clock latency in seconds. None until seeded.
    #[must_use]
    pub fn episode_latency_ema(&self) -> Option<f64> {
        let bits = self.latency_ema_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    /// Episode-weighted EMA of "learner beat its reference" over labeled
    /// appends. None until a labeled episode lands.
    #[must_use]
    pub fn win_rate_ema(&self) -> Option<f64> {
        let bits = self.win_ema_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits) - 1.0)
    }

    /// Lowest terminal cost of any appended episode. None until seeded.
    #[must_use]
    pub fn best_cost(&self) -> Option<f64> {
        let bits = self.best_cost_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    /// Static root facts for single-graph runs; survives reopen.
    pub fn set_root_info(&self, info: &ReplayRootInfo) -> ReplayResult<()> {
        let meta = self.cf(CF_META)?;
        self.db
            .put_cf(&meta, META_ROOT_INFO, postcard::to_allocvec(info)?)
            .map_err(ReplayError::from)
    }

    pub fn root_info(&self) -> ReplayResult<Option<ReplayRootInfo>> {
        let meta = self.cf(CF_META)?;
        self.db
            .get_cf(&meta, META_ROOT_INFO)?
            .map(|bytes| postcard::from_bytes(&bytes).map_err(ReplayError::from))
            .transpose()
    }

    /// (completed games, primary learner episodes that selected STOP).
    #[must_use]
    pub fn episode_counters(&self) -> (u64, u64) {
        (
            self.completed_games.load(Ordering::Acquire),
            self.episodes_stopped.load(Ordering::Acquire),
        )
    }

    #[must_use]
    pub fn counters(&self) -> ReplayCounters {
        ReplayCounters {
            produced_rows: self.produced_rows.load(Ordering::Acquire),
            produced_policy_rows: self.produced_policy_rows.load(Ordering::Acquire),
            consumed_rows: self.consumed_rows.load(Ordering::Acquire),
        }
    }

    fn cf(&self, name: &'static str) -> ReplayResult<&rocksdb::ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| ReplayError::storage(format!("missing column family {name}")))
    }

    fn data_mode(&self) -> ReplayResult<ReplayDataMode> {
        Ok(self
            .data_mode
            .lock()
            .map_err(ReplayError::storage)?
            .unwrap_or(ReplayDataMode::Standard))
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct StoredFeatureSchemaConfig {
    name: String,
    node_vocab_size: u16,
    node_attr_dim: u16,
    edge_type_count: u8,
    action_kind_vocab_size: u32,
    max_nodes: u32,
    max_edges: u32,
    max_actions: u32,
    max_subjects: u32,
    opponent_reward_scale: f32,
    expander_degree: u8,
    expander_seed: u64,
}

impl From<&FeatureSchemaConfig> for StoredFeatureSchemaConfig {
    fn from(config: &FeatureSchemaConfig) -> Self {
        Self {
            name: config.name.clone(),
            node_vocab_size: config.node_vocab_size,
            node_attr_dim: config.node_attr_dim,
            edge_type_count: config.edge_type_count,
            action_kind_vocab_size: config.action_kind_vocab_size,
            max_nodes: config.max_nodes,
            max_edges: config.max_edges,
            max_actions: config.max_actions,
            max_subjects: config.max_subjects,
            opponent_reward_scale: config.opponent_reward_scale,
            expander_degree: config.expander_degree,
            expander_seed: config.expander_seed,
        }
    }
}

impl From<StoredFeatureSchemaConfig> for FeatureSchemaConfig {
    fn from(config: StoredFeatureSchemaConfig) -> Self {
        Self {
            name: config.name,
            node_vocab_size: config.node_vocab_size,
            node_attr_dim: config.node_attr_dim,
            edge_type_count: config.edge_type_count,
            action_kind_vocab_size: config.action_kind_vocab_size,
            max_nodes: config.max_nodes,
            max_edges: config.max_edges,
            max_actions: config.max_actions,
            max_subjects: config.max_subjects,
            opponent_reward_scale: config.opponent_reward_scale,
            expander_degree: config.expander_degree,
            expander_seed: config.expander_seed,
        }
    }
}

fn open_db(path: &Path) -> ReplayResult<DB> {
    let mut options = Options::default();
    options.create_if_missing(true);
    options.create_missing_column_families(true);
    // Selfplay writes tens of MB/s of large rows continuously; defaults
    // (64 MB memtables, 2 background jobs, 8 MB cache) accumulate
    // compaction debt until reads and appends stall mid-run.
    options.increase_parallelism(8);
    options.set_max_background_jobs(8);

    // Sized to hold the full sample window in RAM: 50K rows at ~90 KB is
    // ~4.5 GB, and a window that misses cache costs the trainer ~150 ms
    // per 256-row sample under write load (the measured segment-5 stall).
    let cache = Cache::new_lru_cache(16 * 1024 * 1024 * 1024);
    let mut block = BlockBasedOptions::default();
    block.set_block_cache(&cache);

    let mut value_cf = Options::default();
    value_cf.set_write_buffer_size(256 * 1024 * 1024);
    value_cf.set_target_file_size_base(128 * 1024 * 1024);
    value_cf.set_level_compaction_dynamic_level_bytes(true);
    value_cf.set_compression_type(DBCompressionType::Lz4);
    value_cf.set_block_based_table_factory(&block);

    let mut index_cf = Options::default();
    index_cf.set_write_buffer_size(64 * 1024 * 1024);
    index_cf.set_level_compaction_dynamic_level_bytes(true);
    index_cf.set_compression_type(DBCompressionType::Lz4);
    index_cf.set_block_based_table_factory(&block);

    let descriptors = [
        ColumnFamilyDescriptor::new(CF_META, Options::default()),
        ColumnFamilyDescriptor::new(CF_EPISODES, index_cf.clone()),
        ColumnFamilyDescriptor::new(CF_ROWS, value_cf),
        ColumnFamilyDescriptor::new(CF_ROW_INDEX, index_cf),
    ];

    DB::open_cf_descriptors(&options, path, descriptors).map_err(ReplayError::from)
}

fn ensure_schema(db: &DB) -> ReplayResult<()> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    match db.get_cf(&meta, META_SCHEMA_VERSION)? {
        Some(bytes) => {
            if decode_u32(&bytes) == Some(SCHEMA_VERSION) {
                Ok(())
            } else {
                Err(ReplayError::SchemaMismatch)
            }
        }
        None => {
            let mut batch = WriteBatch::default();
            batch.put_cf(&meta, META_SCHEMA_VERSION, encode_u32(SCHEMA_VERSION));
            batch.put_cf(&meta, META_NEXT_EPISODE_SEQ, encode_u64(0));
            batch.put_cf(&meta, META_COMPLETED_GAMES, encode_u64(0));
            batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(0));
            batch.put_cf(&meta, META_PRODUCED_POLICY_ROWS, encode_u64(0));
            batch.put_cf(&meta, META_CONSUMED_ROWS, encode_u64(0));
            db.write(batch).map_err(ReplayError::from)
        }
    }
}

fn recover_next_episode_seq(db: &DB) -> ReplayResult<u64> {
    let episodes = db
        .cf_handle(CF_EPISODES)
        .ok_or_else(|| ReplayError::storage("missing episodes column family"))?;
    let mut iter = db.iterator_cf(&episodes, IteratorMode::End);

    match iter.next().transpose()? {
        Some((key, _)) => decode_u64_key(&key)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| ReplayError::storage("corrupt episode key")),
        None => Ok(0),
    }
}

fn recover_next_row_seq(db: &DB) -> ReplayResult<u64> {
    let row_index = db
        .cf_handle(CF_ROW_INDEX)
        .ok_or_else(|| ReplayError::storage("missing row_index column family"))?;
    let mut iter = db.iterator_cf(&row_index, IteratorMode::End);

    match iter.next().transpose()? {
        Some((key, _)) => decode_u64_key(&key)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| ReplayError::storage("corrupt row index key")),
        None => Ok(0),
    }
}

fn read_meta_u64(db: &DB, key: &[u8]) -> ReplayResult<Option<u64>> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    db.get_cf(&meta, key)?
        .map(|bytes| decode_u64(&bytes).ok_or_else(|| ReplayError::storage("corrupt meta u64")))
        .transpose()
}

fn write_meta_u64(db: &DB, key: &[u8], value: u64) -> ReplayResult<()> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    db.put_cf(&meta, key, encode_u64(value))
        .map_err(ReplayError::from)
}

fn read_feature_schema(db: &DB) -> ReplayResult<Option<FeatureSchemaConfig>> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    db.get_cf(&meta, META_FEATURE_SCHEMA)?
        .map(|bytes| {
            postcard::from_bytes::<StoredFeatureSchemaConfig>(&bytes)
                .map(FeatureSchemaConfig::from)
                .map_err(ReplayError::from)
        })
        .transpose()
}

fn read_data_mode(db: &DB) -> ReplayResult<Option<ReplayDataMode>> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    db.get_cf(&meta, META_DATA_MODE)?
        .map(|bytes| ReplayDataMode::from_bytes(&bytes).ok_or(ReplayError::DataModeMismatch))
        .transpose()
}
