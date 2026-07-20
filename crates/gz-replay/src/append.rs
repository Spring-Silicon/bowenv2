use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_POLICY_ROW_INDEX, CF_ROW_INDEX, CF_ROWS, CF_VALUE_ROW_INDEX, episode_key,
    row_index_key, row_key,
};
use crate::records::{ReplayEpisodeRecord, ReplayRow, StoredReplayRow};
use rocksdb::{DB, WriteBatch};

#[derive(Clone, Copy)]
pub(crate) struct AppendSequences {
    pub next_episode: u64,
    pub next_row: u64,
    pub next_policy_row: u64,
    pub next_value_row: u64,
}

pub(crate) struct EpisodeAppend<'a> {
    pub record: &'a ReplayEpisodeRecord,
    pub rows: &'a [ReplayRow],
}

pub(crate) fn stage_episodes(
    db: &DB,
    batch: &mut WriteBatch,
    episodes: &[EpisodeAppend<'_>],
    mut sequences: AppendSequences,
) -> ReplayResult<AppendSequences> {
    let episodes_cf = cf(db, CF_EPISODES)?;
    let rows_cf = cf(db, CF_ROWS)?;
    let row_index = cf(db, CF_ROW_INDEX)?;
    let policy_row_index = cf(db, CF_POLICY_ROW_INDEX)?;
    let value_row_index = cf(db, CF_VALUE_ROW_INDEX)?;

    for episode in episodes {
        batch.put_cf(
            &episodes_cf,
            episode_key(sequences.next_episode),
            postcard::to_allocvec(episode.record)?,
        );
        for row in episode.rows {
            let key = row_key(sequences.next_episode, row.step_index);
            batch.put_cf(
                &rows_cf,
                key,
                postcard::to_allocvec(&StoredReplayRow::from_row(row)?)?,
            );
            batch.put_cf(
                &row_index,
                row_index_key(sequences.next_row),
                key.as_slice(),
            );
            sequences.next_row = increment(sequences.next_row, "row sequence overflow")?;

            if row.policy_target.iter().any(|target| *target > 0.0) {
                batch.put_cf(
                    &policy_row_index,
                    row_index_key(sequences.next_policy_row),
                    key.as_slice(),
                );
                sequences.next_policy_row =
                    increment(sequences.next_policy_row, "policy row sequence overflow")?;
            }
            if row.value_target.is_some() {
                batch.put_cf(
                    &value_row_index,
                    row_index_key(sequences.next_value_row),
                    key.as_slice(),
                );
                sequences.next_value_row =
                    increment(sequences.next_value_row, "value row sequence overflow")?;
            }
        }
        sequences.next_episode = increment(sequences.next_episode, "episode id overflow")?;
    }

    Ok(sequences)
}

fn increment(value: u64, message: &'static str) -> ReplayResult<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| ReplayError::storage(message))
}

fn cf<'a>(db: &'a DB, name: &'static str) -> ReplayResult<&'a rocksdb::ColumnFamily> {
    db.cf_handle(name)
        .ok_or_else(|| ReplayError::storage(format!("missing column family {name}")))
}
