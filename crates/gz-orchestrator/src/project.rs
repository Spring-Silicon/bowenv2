use crate::reference::Reference;
use gz_measurer::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, ProjectedReference, ProjectionMode,
};
use gz_replay::{ReplayEpisodeRecord, ReplayRow};
use gz_search::{GumbelEpisode, GumbelStopReason};

pub fn project_episode<G, C>(
    episode: &GumbelEpisode<G, C>,
    reference: Option<&Reference>,
    feature_rows: Option<&[Vec<u8>]>,
    length_tiebreak: bool,
    episode_id: u64,
) -> Option<(ReplayEpisodeRecord, Vec<ReplayRow>)> {
    let artifact = artifact_from_episode(episode, feature_rows);
    let reference = reference.map(projected_reference);
    gz_measurer::project_episode(
        &artifact,
        reference.as_ref(),
        length_tiebreak,
        episode_id,
        ProjectionMode::AllowUnlabeled,
    )
    .ok()
}

/// The learner reward an episode would project with, if eligible.
/// Lets callers that drop an episode from the store still feed the
/// reference provider's reward statistics.
pub fn episode_reward<G, C>(episode: &GumbelEpisode<G, C>) -> Option<f32> {
    gz_measurer::episode_reward(&artifact_from_episode(episode, None))
}

pub(crate) fn artifact_from_episode<G, C>(
    episode: &GumbelEpisode<G, C>,
    feature_rows: Option<&[Vec<u8>]>,
) -> CompletedEpisodeArtifact {
    CompletedEpisodeArtifact {
        root: episode.root_context,
        final_graph: episode.final_context,
        final_measure: gz_engine::MeasureSummary::from(&episode.final_measure),
        stop_selected: matches!(episode.stop_reason, GumbelStopReason::SelectedStop),
        search_config_hash: episode.search_config_hash,
        steps: episode
            .steps
            .iter()
            .map(|step| CompletedEpisodeStep {
                before: step.step_ref.before,
                after: step.step_ref.after,
                selected_action: step.selected_action,
                legal_actions: step.legal_actions.clone(),
                policy_target: step.policy_target.clone(),
                model_version: Some(step.model_version),
            })
            .collect(),
        feature_rows: feature_rows.map(<[Vec<u8>]>::to_vec),
    }
}

pub(crate) fn projected_reference(reference: &Reference) -> ProjectedReference {
    ProjectedReference {
        kind: reference.kind,
        final_reward: reference.final_reward,
        final_graph: reference.final_graph,
        ref_id: reference.ref_id,
        search_config_hash: reference.search_config_hash,
        model_version: reference.model_version,
        step_count: reference.steps.len(),
    }
}
