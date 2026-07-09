use crate::{
    CompletedEpisodeArtifact, MeasurerError, ProjectedReference, ProjectionMode, episode_reward,
    project_episode,
};
use gz_engine::{GraphHash, ModelVersion};
use gz_replay::{ReplayError, ReplayStore};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredEpisode {
    pub lane: usize,
    pub episode_id: u64,
    pub artifact: CompletedEpisodeArtifact,
    pub reference: Option<ProjectedReference>,
    pub mode: ProjectionMode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasurerAdmission {
    pub learner_reward: Option<f32>,
    pub status: MeasurerAdmissionStatus,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeasurerAdmissionStatus {
    Appended { row_count: u64 },
    Dropped { reason: MeasurerError },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeasurerRunSummary {
    pub lanes: Vec<MeasurerLaneSummary>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub replay_rows: u64,
    pub measure_ledger: MeasureLedgerSnapshot,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeasurerLaneSummary {
    pub lane: usize,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub replay_rows: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeasureLedgerSnapshot {
    pub finals: u64,
    pub distinct_finals: u64,
    pub repeat_finals: u64,
    pub distinct_by_version: Vec<(ModelVersion, u64)>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeasurerStats {
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub finals: u64,
    pub distinct_finals: u64,
}

pub struct ReplayMeasurer<'a> {
    store: &'a ReplayStore,
    length_tiebreak: bool,
    summary: MeasurerRunSummary,
    ledger: MeasureLedger,
}

impl<'a> ReplayMeasurer<'a> {
    #[must_use]
    pub fn new(store: &'a ReplayStore, length_tiebreak: bool) -> Self {
        Self {
            store,
            length_tiebreak,
            summary: MeasurerRunSummary::default(),
            ledger: MeasureLedger::default(),
        }
    }

    pub fn admit(&mut self, episode: MeasuredEpisode) -> Result<MeasurerAdmission, ReplayError> {
        let learner_reward = episode_reward(&episode.artifact);
        if learner_reward.is_some() {
            self.ledger.observe(&episode.artifact);
        }

        let (status, replay_rows) = match project_episode(
            &episode.artifact,
            episode.reference.as_ref(),
            self.length_tiebreak,
            episode.episode_id,
            episode.mode,
        ) {
            Ok((record, rows)) => {
                let row_count = rows.len() as u64;
                self.store.append_episode(&record, &rows)?;
                (MeasurerAdmissionStatus::Appended { row_count }, row_count)
            }
            Err(reason) => (MeasurerAdmissionStatus::Dropped { reason }, 0),
        };

        match status {
            MeasurerAdmissionStatus::Appended { row_count } => {
                self.summary.episodes_appended += 1;
                self.summary.replay_rows += row_count;
                let lane = lane_summary(&mut self.summary.lanes, episode.lane);
                lane.episodes_appended += 1;
                lane.replay_rows += replay_rows;
            }
            MeasurerAdmissionStatus::Dropped { .. } => {
                self.summary.episodes_dropped += 1;
                lane_summary(&mut self.summary.lanes, episode.lane).episodes_dropped += 1;
            }
        }

        Ok(MeasurerAdmission {
            learner_reward,
            status,
        })
    }

    /// Cumulative admission and ledger counters for periodic
    /// heartbeats; cheap, borrow-only (finish() still owns the full
    /// per-version snapshot).
    #[must_use]
    pub fn stats(&self) -> MeasurerStats {
        MeasurerStats {
            episodes_appended: self.summary.episodes_appended,
            episodes_dropped: self.summary.episodes_dropped,
            finals: self.ledger.finals,
            distinct_finals: self.ledger.seen.len() as u64,
        }
    }

    #[must_use]
    pub fn finish(mut self) -> MeasurerRunSummary {
        self.summary.measure_ledger = self.ledger.snapshot();
        self.summary
    }
}

fn lane_summary(lanes: &mut Vec<MeasurerLaneSummary>, lane: usize) -> &mut MeasurerLaneSummary {
    while lanes.len() <= lane {
        let next = lanes.len();
        lanes.push(MeasurerLaneSummary {
            lane: next,
            ..MeasurerLaneSummary::default()
        });
    }
    &mut lanes[lane]
}

#[derive(Debug, Default)]
struct MeasureLedger {
    finals: u64,
    seen: HashSet<GraphHash>,
    distinct_by_version: BTreeMap<ModelVersion, u64>,
}

impl MeasureLedger {
    fn observe(&mut self, artifact: &CompletedEpisodeArtifact) {
        self.finals += 1;
        if !self.seen.insert(artifact.final_graph.graph.graph_hash) {
            return;
        }

        if let Some(version) = artifact.steps.last().and_then(|step| step.model_version) {
            *self.distinct_by_version.entry(version).or_insert(0) += 1;
        }
    }

    fn snapshot(self) -> MeasureLedgerSnapshot {
        let distinct_finals = self.seen.len() as u64;
        MeasureLedgerSnapshot {
            finals: self.finals,
            distinct_finals,
            repeat_finals: self.finals.saturating_sub(distinct_finals),
            distinct_by_version: self.distinct_by_version.into_iter().collect(),
        }
    }
}
