use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_META, CF_ROW_INDEX, CF_ROWS, META_CONSUMED_ROWS, META_NEXT_EPISODE_SEQ,
    META_PRODUCED_ROWS, META_SCHEMA_VERSION, SCHEMA_VERSION, decode_episode_from_row_key,
    decode_u32, decode_u64, decode_u64_key, encode_u32, encode_u64, episode_key, row_index_key,
    row_key,
};
use crate::records::{ReplayEpisodeId, ReplayEpisodeRecord, ReplayRow, validate_episode};
use crate::sample::{ReplayRng, SampleConfig};
use rocksdb::{ColumnFamilyDescriptor, DB, IteratorMode, Options, WriteBatch};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub struct ReplayStore {
    db: Arc<DB>,
    write_lock: Mutex<()>,
    produced_rows: AtomicU64,
    consumed_rows: AtomicU64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCounters {
    pub produced_rows: u64,
    pub consumed_rows: u64,
}

impl ReplayStore {
    pub fn open(path: &Path) -> ReplayResult<Self> {
        let db = Arc::new(open_db(path)?);
        ensure_schema(&db)?;

        let next_episode_seq = recover_next_episode_seq(&db)?;
        let produced_rows = recover_next_row_seq(&db)?;
        let consumed_rows = read_meta_u64(&db, META_CONSUMED_ROWS)?.unwrap_or(0);
        write_meta_u64(&db, META_NEXT_EPISODE_SEQ, next_episode_seq)?;
        write_meta_u64(&db, META_PRODUCED_ROWS, produced_rows)?;

        Ok(Self {
            db,
            write_lock: Mutex::new(()),
            produced_rows: AtomicU64::new(produced_rows),
            consumed_rows: AtomicU64::new(consumed_rows),
        })
    }

    pub fn append_episode(
        &self,
        record: &ReplayEpisodeRecord,
        rows: &[ReplayRow],
    ) -> ReplayResult<ReplayEpisodeId> {
        validate_episode(record, rows)?;

        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let episode_seq = recover_next_episode_seq(&self.db)?;
        let row_seq = recover_next_row_seq(&self.db)?;
        let next_episode_seq = episode_seq
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("episode id overflow"))?;
        let produced_rows = row_seq
            .checked_add(rows.len() as u64)
            .ok_or(ReplayError::InvalidRecord)?;
        let id = ReplayEpisodeId::new(episode_seq);

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
            batch.put_cf(&row_cf, key, postcard::to_allocvec(row)?);
            batch.put_cf(
                &row_index,
                row_index_key(row_seq + offset as u64),
                key.as_slice(),
            );
        }

        batch.put_cf(&meta, META_NEXT_EPISODE_SEQ, encode_u64(next_episode_seq));
        batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(produced_rows));
        self.db.write(batch)?;
        self.produced_rows.store(produced_rows, Ordering::Release);

        Ok(id)
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
        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let produced = recover_next_row_seq(&self.db)?;

        if produced == 0 {
            return Err(ReplayError::Empty);
        }

        let window = config.window_rows.get().min(produced);
        let start = produced - window;
        let mut rng = ReplayRng::new(config.seed);
        let row_index = self.cf(CF_ROW_INDEX)?;
        let rows = self.cf(CF_ROWS)?;
        let mut out = Vec::with_capacity(config.batch.get());

        for _ in 0..config.batch.get() {
            let row_seq = start + rng.next_bounded(window);
            let row_key = self
                .db
                .get_cf(&row_index, row_index_key(row_seq))?
                .ok_or_else(|| ReplayError::storage("missing row index entry"))?;
            let episode_seq = decode_episode_from_row_key(&row_key)
                .ok_or_else(|| ReplayError::storage("corrupt row key"))?;
            let row = self
                .db
                .get_cf(&rows, &row_key)?
                .ok_or_else(|| ReplayError::storage("missing replay row"))?;

            out.push((
                ReplayEpisodeId::new(episode_seq),
                postcard::from_bytes(&row)?,
            ));
        }

        let consumed_rows = self
            .consumed_rows
            .load(Ordering::Acquire)
            .checked_add(config.batch.get() as u64)
            .ok_or_else(|| ReplayError::storage("consumed row counter overflow"))?;
        write_meta_u64(&self.db, META_CONSUMED_ROWS, consumed_rows)?;
        self.consumed_rows.store(consumed_rows, Ordering::Release);
        self.produced_rows.store(produced, Ordering::Release);

        Ok(out)
    }

    #[must_use]
    pub fn counters(&self) -> ReplayCounters {
        ReplayCounters {
            produced_rows: self.produced_rows.load(Ordering::Acquire),
            consumed_rows: self.consumed_rows.load(Ordering::Acquire),
        }
    }

    fn cf(&self, name: &'static str) -> ReplayResult<&rocksdb::ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| ReplayError::storage(format!("missing column family {name}")))
    }
}

fn open_db(path: &Path) -> ReplayResult<DB> {
    let mut options = Options::default();
    options.create_if_missing(true);
    options.create_missing_column_families(true);

    let descriptors = [
        ColumnFamilyDescriptor::new(CF_META, Options::default()),
        ColumnFamilyDescriptor::new(CF_EPISODES, Options::default()),
        ColumnFamilyDescriptor::new(CF_ROWS, Options::default()),
        ColumnFamilyDescriptor::new(CF_ROW_INDEX, Options::default()),
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
            batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(0));
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
