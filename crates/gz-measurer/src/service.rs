use crate::ValueTargetConfig;
use crate::{
    CompletedEpisodeArtifact, MeasurerError, ProjectedReference, ProjectionMode, episode_reward,
    project_episode_with_value_target,
};
use gz_engine::{GraphHash, ModelVersion};
use gz_replay::{ReplayError, ReplayStore};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredEpisode {
    pub lane: usize,
    pub episode_id: u64,
    pub artifact: CompletedEpisodeArtifact,
    pub root_reward: f32,
    pub reference: Option<ProjectedReference>,
    pub mode: ProjectionMode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredCompetitiveGame {
    pub lane: usize,
    pub game_id: u64,
    pub learner_is_p1: bool,
    pub root_reward: f32,
    pub p1_artifact: CompletedEpisodeArtifact,
    pub p1_reference: ProjectedReference,
    pub p2_artifact: CompletedEpisodeArtifact,
    pub p2_reference: ProjectedReference,
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
    value_target: ValueTargetConfig,
    summary: MeasurerRunSummary,
    ledger: MeasureLedger,
}

impl<'a> ReplayMeasurer<'a> {
    #[must_use]
    pub fn new(store: &'a ReplayStore, length_tiebreak: bool) -> Self {
        Self::with_value_target(store, length_tiebreak, ValueTargetConfig::Sign)
    }

    #[must_use]
    pub fn with_value_target(
        store: &'a ReplayStore,
        length_tiebreak: bool,
        value_target: ValueTargetConfig,
    ) -> Self {
        Self {
            store,
            length_tiebreak,
            value_target,
            summary: MeasurerRunSummary::default(),
            ledger: MeasureLedger::default(),
        }
    }

    pub fn admit(&mut self, episode: MeasuredEpisode) -> Result<MeasurerAdmission, ReplayError> {
        let learner_reward = episode_reward(&episode.artifact);
        if learner_reward.is_some() {
            self.ledger.observe(&episode.artifact);
        }

        let (status, replay_rows) = match project_episode_with_value_target(
            &episode.artifact,
            episode.reference.as_ref(),
            self.length_tiebreak,
            episode.episode_id,
            episode.mode,
            episode.root_reward,
            self.value_target,
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

    pub fn admit_competitive(
        &mut self,
        game: MeasuredCompetitiveGame,
    ) -> Result<MeasurerAdmission, ReplayError> {
        let p1_reward = episode_reward(&game.p1_artifact);
        let p2_reward = episode_reward(&game.p2_artifact);
        let learner_reward = if game.learner_is_p1 {
            p1_reward
        } else {
            p2_reward
        };
        if p1_reward.is_some() {
            self.ledger.observe(&game.p1_artifact);
        }
        if p2_reward.is_some() {
            self.ledger.observe(&game.p2_artifact);
        }

        let projected = project_competitive_game(&game, self.length_tiebreak, self.value_target);
        let (status, replay_rows) = match projected {
            Ok((p1, p2)) => {
                let row_count = (p1.1.len() + p2.1.len()) as u64;
                if game.learner_is_p1 {
                    self.store
                        .append_episode_pair((&p1.0, &p1.1), (&p2.0, &p2.1))?;
                } else {
                    self.store
                        .append_episode_pair((&p2.0, &p2.1), (&p1.0, &p1.1))?;
                }
                (MeasurerAdmissionStatus::Appended { row_count }, row_count)
            }
            Err(reason) => (MeasurerAdmissionStatus::Dropped { reason }, 0),
        };

        match status {
            MeasurerAdmissionStatus::Appended { row_count } => {
                self.summary.episodes_appended += 1;
                self.summary.replay_rows += row_count;
                let lane = lane_summary(&mut self.summary.lanes, game.lane);
                lane.episodes_appended += 1;
                lane.replay_rows += replay_rows;
            }
            MeasurerAdmissionStatus::Dropped { .. } => {
                self.summary.episodes_dropped += 1;
                lane_summary(&mut self.summary.lanes, game.lane).episodes_dropped += 1;
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

type ProjectedEpisode = (gz_replay::ReplayEpisodeRecord, Vec<gz_replay::ReplayRow>);

fn project_competitive_game(
    game: &MeasuredCompetitiveGame,
    length_tiebreak: bool,
    value_target_config: ValueTargetConfig,
) -> Result<(ProjectedEpisode, ProjectedEpisode), MeasurerError> {
    let p1_reward = episode_reward(&game.p1_artifact).ok_or(MeasurerError::Unmeasured)?;
    let p2_reward = episode_reward(&game.p2_artifact).ok_or(MeasurerError::Unmeasured)?;
    let p1_target = if matches!(value_target_config, ValueTargetConfig::Sign)
        && p1_reward == p2_reward
        && !length_tiebreak
    {
        1.0
    } else {
        crate::outcome_target(
            value_target_config,
            p1_reward,
            p2_reward,
            game.root_reward,
            game.p1_artifact.steps.len(),
            Some(game.p2_artifact.steps.len()),
            length_tiebreak,
            game.game_id,
        )
    };
    let mut p1 = project_episode_with_value_target(
        &game.p1_artifact,
        Some(&game.p1_reference),
        false,
        game.game_id,
        ProjectionMode::RequireReference,
        game.root_reward,
        value_target_config,
    )?;
    let mut p2 = project_episode_with_value_target(
        &game.p2_artifact,
        Some(&game.p2_reference),
        false,
        game.game_id,
        ProjectionMode::RequireReference,
        game.root_reward,
        value_target_config,
    )?;
    set_value_target(&mut p1, p1_target);
    set_value_target(&mut p2, -p1_target);
    Ok((p1, p2))
}

fn set_value_target(projected: &mut ProjectedEpisode, target: f32) {
    projected.0.outcome.value_target = Some(target);
    for row in &mut projected.1 {
        row.value_target = Some(target);
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
