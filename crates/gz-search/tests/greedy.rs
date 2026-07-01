mod common;

use common::{TestEngine, candidate_ref, context, measure_options, stop_ref};
use gz_engine::{CandidateOptions, MeasureConfigHash, MeasureOptions};
use gz_search::{
    GreedyEpisode, GreedySearch, GreedySearchConfig, GreedyStopReason, SearchAction,
    greedy_search_config_hash,
};

fn search(max_steps: usize, max_candidates: Option<usize>) -> GreedySearch {
    GreedySearch::new(GreedySearchConfig {
        max_steps,
        candidate_options: CandidateOptions {
            max_candidates,
            deterministic_order: true,
        },
        measure_options: measure_options(),
    })
}

fn first_step(episode: &GreedyEpisode<u8, u8>) -> &gz_search::SearchStep<u8, u8> {
    episode.steps.first().unwrap()
}

#[test]
fn zero_step_run_measures_root_and_returns_max_steps() {
    let mut engine = TestEngine::new().reward(0, 3.0);
    let episode = search(0, None).run_from_root(&mut engine).unwrap();

    assert_eq!(episode.stop_reason, GreedyStopReason::MaxSteps);
    assert_eq!(episode.final_graph, 0);
    assert_eq!(episode.final_measure.scalar_reward, Some(3.0));
    assert!(episode.steps.is_empty());
}

#[test]
fn stop_is_appended_after_limited_engine_candidates() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1, 2, 3])
        .reward(0, 0.0)
        .reward(1, 3.0)
        .reward(2, 2.0);
    let episode = search(1, Some(2)).run_from_root(&mut engine).unwrap();
    let step = first_step(&episode);

    assert_eq!(step.engine_candidate_count, 2);
    assert_eq!(step.action_count, 3);
    assert_eq!(step.selected_action, candidate_ref(0, 1));
}

#[test]
fn no_engine_candidates_selects_stop() {
    let mut engine = TestEngine::new().reward(0, 1.0);
    let episode = search(3, None).run_from_root(&mut engine).unwrap();
    let step = first_step(&episode);

    assert_eq!(episode.stop_reason, GreedyStopReason::SelectedStop);
    assert_eq!(step.action, SearchAction::Stop);
    assert_eq!(step.selected_action, stop_ref(0));
}

#[test]
fn unscored_current_graph_returns_without_stop_step() {
    let mut engine = TestEngine::new().unscored(0);
    let episode = search(3, None).run_from_root(&mut engine).unwrap();

    assert_eq!(episode.stop_reason, GreedyStopReason::UnscoredCurrentGraph);
    assert!(episode.steps.is_empty());
}

#[test]
fn rejected_candidates_select_stop() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1, 2])
        .rejected(0, 1)
        .rejected(0, 2)
        .reward(0, 1.0);
    let episode = search(3, None).run_from_root(&mut engine).unwrap();

    assert_eq!(episode.stop_reason, GreedyStopReason::SelectedStop);
    assert_eq!(first_step(&episode).action, SearchAction::Stop);
}

#[test]
fn unscored_successors_select_stop() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1])
        .apply(0, 1, 1)
        .reward(0, 1.0)
        .unscored(1);
    let episode = search(3, None).run_from_root(&mut engine).unwrap();

    assert_eq!(episode.stop_reason, GreedyStopReason::SelectedStop);
    assert_eq!(first_step(&episode).action, SearchAction::Stop);
}

#[test]
fn best_strict_improvement_is_selected() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1, 2])
        .reward(0, 1.0)
        .reward(1, 5.0)
        .reward(2, 4.0);
    let episode = search(1, None).run_from_root(&mut engine).unwrap();
    let step = first_step(&episode);

    assert_eq!(step.action, SearchAction::Candidate(1));
    assert_eq!(step.after, 1);
    assert_eq!(episode.final_graph, 1);
}

#[test]
fn no_strict_improvement_selects_stop() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1, 2])
        .reward(0, 3.0)
        .reward(1, 3.0)
        .reward(2, 2.0);
    let episode = search(3, None).run_from_root(&mut engine).unwrap();
    let step = first_step(&episode);

    assert_eq!(episode.stop_reason, GreedyStopReason::SelectedStop);
    assert_eq!(step.action, SearchAction::Stop);
    assert_eq!(step.after, step.before);
}

#[test]
fn ties_use_static_prior_then_candidate_hash() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1, 2])
        .prior(1, 0.1)
        .prior(2, 0.9)
        .reward(0, 1.0)
        .reward(1, 5.0)
        .reward(2, 5.0);
    let episode = search(1, None).run_from_root(&mut engine).unwrap();

    assert_eq!(first_step(&episode).action, SearchAction::Candidate(2));

    let mut engine = TestEngine::new()
        .candidates(0, vec![2, 1])
        .prior(1, 0.5)
        .prior(2, 0.5)
        .reward(0, 1.0)
        .reward(1, 5.0)
        .reward(2, 5.0);
    let episode = search(1, None).run_from_root(&mut engine).unwrap();

    assert_eq!(first_step(&episode).action, SearchAction::Candidate(1));
}

#[test]
fn stop_step_keeps_graph_and_does_not_apply() {
    let mut engine = TestEngine::new().reward(0, 1.0);
    let episode = search(3, None).run_from_root(&mut engine).unwrap();
    let step = first_step(&episode);

    assert_eq!(engine.apply_calls, Vec::<(u8, u8)>::new());
    assert_eq!(step.before, 0);
    assert_eq!(step.after, 0);
    assert_eq!(step.step_ref.before, context(0));
    assert_eq!(step.step_ref.after, context(0));
}

#[test]
fn selected_action_ref_matches_candidate_info() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![2, 1])
        .reward(0, 0.0)
        .reward(1, 1.0)
        .reward(2, 2.0);
    let episode = search(1, None).run_from_root(&mut engine).unwrap();

    assert_eq!(first_step(&episode).selected_action, candidate_ref(0, 2));
    assert_eq!(first_step(&episode).selected_rank, 0);
}

#[test]
fn search_config_hash_changes_when_path_config_changes() {
    let base = search(1, None).search_config_hash();
    let max_steps = search(2, None).search_config_hash();
    let max_candidates = search(1, Some(1)).search_config_hash();
    let measure_samples = greedy_search_config_hash(
        1,
        CandidateOptions::default(),
        MeasureOptions::new(MeasureConfigHash::from_bytes([9; 32]), 2, None, true).unwrap(),
    );

    assert_ne!(base, max_steps);
    assert_ne!(base, max_candidates);
    assert_ne!(base, measure_samples);
}
