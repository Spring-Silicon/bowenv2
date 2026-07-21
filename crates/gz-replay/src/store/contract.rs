use super::{ReplayDataMode, ReplayStore};
use crate::database::{read_feature_schema, stored_engine_identity, stored_feature_schema};
use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_META, META_DATA_MODE, META_ENGINE_IDENTITY, META_FEATURE_SCHEMA,
};
use crate::records::{ReplayEpisodeRecord, validate_episode_engine_identity};
use gz_engine::EngineIdentity;
use gz_features::{FeatureSchema, FeatureSchemaConfig};
use rocksdb::{IteratorMode, WriteBatch};
use std::sync::atomic::Ordering;

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayContract {
    pub data_mode: ReplayDataMode,
    pub feature_schema: Option<FeatureSchemaConfig>,
    pub engine_identity: EngineIdentity,
}

impl ReplayContract {
    #[must_use]
    pub fn featurized(
        data_mode: ReplayDataMode,
        feature_schema: FeatureSchemaConfig,
        engine_identity: EngineIdentity,
    ) -> Self {
        Self {
            data_mode,
            feature_schema: Some(feature_schema),
            engine_identity,
        }
    }

    #[must_use]
    pub const fn unfeaturized(data_mode: ReplayDataMode, engine_identity: EngineIdentity) -> Self {
        Self {
            data_mode,
            feature_schema: None,
            engine_identity,
        }
    }
}

impl ReplayStore {
    /// Validate and bind all replay semantics in one RocksDB write.
    /// Existing unbound stores are migrated only after retained episodes pass
    /// identity checks; a feature schema cannot be added after rows exist.
    pub fn ensure_contract(&self, contract: &ReplayContract) -> ReplayResult<()> {
        if let Some(config) = &contract.feature_schema {
            FeatureSchema::new(config.clone()).map_err(|_| ReplayError::InvalidRecord)?;
        }

        let _write = self.write_lock.lock().map_err(ReplayError::storage)?;
        let mut data_mode = self.data_mode.lock().map_err(ReplayError::storage)?;
        let mut engine_identity = self.engine_identity.lock().map_err(ReplayError::storage)?;
        let stored_schema = read_feature_schema(&self.db)?;

        match (&stored_schema, &contract.feature_schema) {
            (Some(stored), Some(expected)) if stored == expected => {}
            (None, Some(_)) if self.produced_rows.load(Ordering::Acquire) == 0 => {}
            (None, None) => {}
            _ => return Err(ReplayError::InvalidRecord),
        }

        if let Some(stored) = *data_mode {
            if stored != contract.data_mode {
                return Err(ReplayError::DataModeMismatch);
            }
        } else if self.next_episode_seq.load(Ordering::Acquire) > 0
            && contract.data_mode != ReplayDataMode::Standard
        {
            return Err(ReplayError::DataModeMismatch);
        }

        if let Some(stored) = *engine_identity {
            if stored != contract.engine_identity {
                return Err(ReplayError::EngineIdentityMismatch);
            }
        } else {
            let episodes = self.cf(CF_EPISODES)?;
            for item in self.db.iterator_cf(&episodes, IteratorMode::Start) {
                let (_, bytes) = item?;
                let record: ReplayEpisodeRecord = postcard::from_bytes(&bytes)?;
                validate_episode_engine_identity(&record, contract.engine_identity)?;
            }
        }

        let meta = self.cf(CF_META)?;
        let mut batch = WriteBatch::default();
        if data_mode.is_none() {
            batch.put_cf(&meta, META_DATA_MODE, contract.data_mode.bytes());
        }
        if engine_identity.is_none() {
            batch.put_cf(
                &meta,
                META_ENGINE_IDENTITY,
                stored_engine_identity(contract.engine_identity),
            );
        }
        if stored_schema.is_none()
            && let Some(config) = &contract.feature_schema
        {
            batch.put_cf(&meta, META_FEATURE_SCHEMA, stored_feature_schema(config)?);
        }
        self.db.write(batch)?;

        *data_mode = Some(contract.data_mode);
        *engine_identity = Some(contract.engine_identity);
        Ok(())
    }
}
