use gz_engine::{MeasureSummary, ModelVersion, PortableSearchActionRef, SearchConfigHash};
use gz_replay::{
    ReplayEpisodeRecord, ReplayOutcome, ReplayReference, ReplayReferenceKind, ReplayRow,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedEpisodeArtifact {
    pub root: gz_engine::ReplayGraphContext,
    pub final_graph: gz_engine::ReplayGraphContext,
    pub final_measure: MeasureSummary,
    pub stop_selected: bool,
    pub search_config_hash: SearchConfigHash,
    pub steps: Vec<CompletedEpisodeStep>,
    pub feature_rows: Option<Vec<Vec<u8>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedEpisodeStep {
    pub before: gz_engine::ReplayGraphContext,
    pub after: gz_engine::ReplayGraphContext,
    pub selected_action: PortableSearchActionRef,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedReference {
    pub kind: ReplayReferenceKind,
    pub final_reward: f32,
    pub final_graph: Option<gz_engine::ReplayGraphContext>,
    pub ref_id: Option<u64>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
    pub step_count: usize,
}

impl From<&crate::ReferenceSnapshot> for ProjectedReference {
    fn from(snapshot: &crate::ReferenceSnapshot) -> Self {
        Self {
            kind: snapshot.kind,
            final_reward: snapshot.final_reward,
            final_graph: snapshot.final_graph,
            ref_id: Some(snapshot.ref_id),
            search_config_hash: Some(snapshot.search_config_hash),
            model_version: Some(snapshot.version),
            step_count: snapshot.steps.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionMode {
    AllowUnlabeled,
    RequireReference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasurerError {
    Unmeasured,
    MissingReference,
    FeatureRowCountMismatch,
    StepCountOverflow,
    InvalidValueTargetConfig,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ValueTargetConfig {
    #[default]
    Sign,
    Graded {
        reward_scale: f32,
    },
}

impl ValueTargetConfig {
    #[must_use]
    pub const fn graded(reward_scale: f32) -> Self {
        Self::Graded { reward_scale }
    }

    #[must_use]
    pub fn is_valid(self) -> bool {
        match self {
            Self::Sign => true,
            Self::Graded { reward_scale } => reward_scale.is_finite() && reward_scale > 0.0,
        }
    }
}

pub fn project_episode(
    artifact: &CompletedEpisodeArtifact,
    reference: Option<&ProjectedReference>,
    length_tiebreak: bool,
    episode_id: u64,
    mode: ProjectionMode,
) -> Result<(ReplayEpisodeRecord, Vec<ReplayRow>), MeasurerError> {
    project_episode_with_value_target(
        artifact,
        reference,
        length_tiebreak,
        episode_id,
        mode,
        0.0,
        ValueTargetConfig::Sign,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn project_episode_with_value_target(
    artifact: &CompletedEpisodeArtifact,
    reference: Option<&ProjectedReference>,
    length_tiebreak: bool,
    episode_id: u64,
    mode: ProjectionMode,
    root_reward: f32,
    value_target_config: ValueTargetConfig,
) -> Result<(ReplayEpisodeRecord, Vec<ReplayRow>), MeasurerError> {
    if !value_target_config.is_valid()
        || matches!(value_target_config, ValueTargetConfig::Graded { .. })
            && !root_reward.is_finite()
    {
        return Err(MeasurerError::InvalidValueTargetConfig);
    }
    let learner_reward = episode_reward(artifact).ok_or(MeasurerError::Unmeasured)?;
    if matches!(mode, ProjectionMode::RequireReference) && reference.is_none() {
        return Err(MeasurerError::MissingReference);
    }
    if let Some(feature_rows) = &artifact.feature_rows
        && feature_rows.len() != artifact.steps.len()
    {
        return Err(MeasurerError::FeatureRowCountMismatch);
    }

    let value_target = reference.map(|reference| {
        let reference_len =
            (length_tiebreak && reference.step_count > 0).then(|| reference.step_count - 1);
        outcome_target(
            value_target_config,
            learner_reward,
            reference.final_reward,
            root_reward,
            artifact.steps.len(),
            reference_len,
            length_tiebreak,
            episode_id,
        )
    });
    let replay_reference = reference.map(|reference| ReplayReference {
        kind: reference.kind,
        reward: reference.final_reward,
        final_graph: reference.final_graph,
        trajectory_id: reference.ref_id,
        search_config_hash: reference.search_config_hash,
        model_version: reference.model_version,
    });

    let mut action_history = Vec::<PortableSearchActionRef>::new();
    let mut search_steps = Vec::with_capacity(artifact.steps.len());
    let mut rows = Vec::with_capacity(artifact.steps.len());
    for (index, step) in artifact.steps.iter().enumerate() {
        let step_index = u32::try_from(index).map_err(|_| MeasurerError::StepCountOverflow)?;
        let step_ref = gz_engine::SearchStepRef::new(step.before, step.selected_action, step.after)
            .map_err(|_| MeasurerError::StepCountOverflow)?;
        search_steps.push(step_ref);
        rows.push(ReplayRow {
            step_index,
            root: artifact.root,
            state: step.before,
            action_history: action_history.clone(),
            legal_actions: step.legal_actions.clone(),
            policy_target: step.policy_target.clone(),
            selected_action: step.selected_action,
            value_target,
            reward_target: Some(learner_reward),
            final_measure: artifact.final_measure.clone(),
            model_version: step.model_version,
            search_config_hash: artifact.search_config_hash,
            feature_row: artifact
                .feature_rows
                .as_ref()
                .map(|feature_rows| feature_rows[index].clone()),
        });
        action_history.push(step.selected_action);
    }

    let record = ReplayEpisodeRecord {
        root: artifact.root,
        final_graph: artifact.final_graph,
        steps: search_steps,
        final_measure: artifact.final_measure.clone(),
        outcome: ReplayOutcome {
            value_target,
            learner_reward,
            reference: replay_reference,
            stopped: artifact.stop_selected,
        },
        search_config_hash: artifact.search_config_hash,
        row_count: rows.len() as u32,
    };

    Ok((record, rows))
}

pub fn episode_reward(artifact: &CompletedEpisodeArtifact) -> Option<f32> {
    if !artifact.final_measure.measured || !artifact.final_measure.valid {
        return None;
    }

    match artifact.final_measure.scalar_reward {
        Some(reward) if reward.is_finite() => Some(reward),
        _ => None,
    }
}

pub fn sign_target(
    learner: f32,
    reference: f32,
    learner_len: usize,
    reference_len: Option<usize>,
    episode_id: u64,
) -> f32 {
    if learner > reference {
        return 1.0;
    }
    if learner < reference {
        return -1.0;
    }
    if let Some(reference_len) = reference_len {
        if learner_len < reference_len {
            return 1.0;
        }
        if learner_len > reference_len {
            return -1.0;
        }
    }

    const TIE_SALT: u64 = 0x7469_655f_6272_6561; // "tie_brea"
    if episode_noise_seed(episode_id ^ TIE_SALT) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn outcome_target(
    config: ValueTargetConfig,
    learner: f32,
    reference: f32,
    root_reward: f32,
    learner_len: usize,
    reference_len: Option<usize>,
    length_tiebreak: bool,
    episode_id: u64,
) -> f32 {
    match config {
        ValueTargetConfig::Sign => {
            sign_target(learner, reference, learner_len, reference_len, episode_id)
        }
        ValueTargetConfig::Graded { reward_scale } => {
            let denominator = root_reward.abs().max(1.0);
            let mut margin = (learner - reference) / denominator;
            if learner == reference
                && length_tiebreak
                && let Some(reference_len) = reference_len
            {
                margin += (reference_len as f32 - learner_len as f32) / denominator;
            }
            (margin / reward_scale).tanh()
        }
    }
}

// splitmix64 finalizer, bit-identical to gz-orchestrator's
// root::episode_noise_seed: stored tie coins must not change across
// the projection move (crates cannot share it -- dependency direction).
fn episode_noise_seed(episode_id: u64) -> u64 {
    let mut value = episode_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod coin_tests {
    use super::sign_target;

    /// Frozen against the original implementation (splitmix64 with
    /// wrapping_add, salted "tie_brea"): stored labels from before the
    /// projection moved crates must reproduce byte-identically.
    #[test]
    fn tie_coin_values_are_frozen() {
        assert_eq!(sign_target(0.0, 0.0, 1, None, 7), -1.0);
        assert_eq!(sign_target(0.0, 0.0, 1, None, 42), -1.0);
        assert_eq!(sign_target(0.0, 0.0, 1, None, 1000), 1.0);
    }
}
