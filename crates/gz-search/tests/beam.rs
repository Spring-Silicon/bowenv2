mod common;

use common::{TestEngine, measure_options};
use gz_engine::{CandidateOptions, MeasureConfigHash, MeasureOptions};
use gz_search::{
    BeamSearch, BeamSearchConfig, BeamStopReason, GreedySearch, GreedySearchConfig, SearchAction,
    beam_search_config_hash,
};
use std::num::NonZeroUsize;

fn beam(max_depth: usize, beam_width: usize, max_candidates: Option<usize>) -> BeamSearch {
    BeamSearch::new(BeamSearchConfig {
        max_depth,
        beam_width: NonZeroUsize::new(beam_width).unwrap(),
        candidate_options: CandidateOptions {
            max_candidates,
            deterministic_order: true,
        },
        measure_options: measure_options(),
    })
}

fn greedy(max_steps: usize) -> GreedySearch {
    GreedySearch::new(GreedySearchConfig {
        max_steps,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(),
    })
}

fn parity_engine() -> TestEngine {
    TestEngine::new()
        .candidates(0, vec![1, 2])
        .candidates(2, vec![3])
        .reward(0, 0.0)
        .reward(1, 1.0)
        .reward(2, 4.0)
        .reward(3, 5.0)
}

#[test]
fn width_one_matches_greedy_path() {
    let mut greedy_engine = parity_engine();
    let greedy_episode = greedy(2).run_from_root(&mut greedy_engine).unwrap();

    let mut beam_engine = parity_engine();
    let beam_episode = beam(2, 1, None).run_from_root(&mut beam_engine).unwrap();

    assert_eq!(beam_episode.final_graph, greedy_episode.final_graph);
    assert_eq!(
        beam_episode
            .steps
            .iter()
            .map(|step| step.action)
            .collect::<Vec<_>>(),
        greedy_episode
            .steps
            .iter()
            .map(|step| step.action)
            .collect::<Vec<_>>()
    );
}

#[test]
fn wider_beam_keeps_lower_immediate_reward_that_later_wins() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1, 2])
        .candidates(1, vec![3])
        .reward(0, -2.0)
        .reward(1, -1.0)
        .reward(2, 1.0)
        .reward(3, 10.0);
    let episode = beam(2, 2, None).run_from_root(&mut engine).unwrap();

    assert_eq!(episode.stop_reason, BeamStopReason::MaxDepth);
    assert_eq!(episode.final_graph, 3);
    assert_eq!(
        episode
            .steps
            .iter()
            .map(|step| step.action)
            .collect::<Vec<_>>(),
        vec![SearchAction::Candidate(1), SearchAction::Candidate(3)]
    );
}

#[test]
fn stopped_entries_are_carried_without_expansion() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1])
        .reward(0, 5.0)
        .reward(1, 6.0);
    let episode = beam(3, 2, None).run_from_root(&mut engine).unwrap();

    assert_eq!(engine.apply_calls, vec![(0, 1)]);
    assert_eq!(episode.stop_reason, BeamStopReason::SelectedStop);
    assert_eq!(episode.final_graph, 1);
    assert_eq!(episode.steps[1].action, SearchAction::Stop);
    assert_eq!(episode.layers.len(), 3);

    let first_layer = &episode.layers[1].entries;
    assert_eq!(first_layer.len(), 2);
    assert!(
        first_layer
            .iter()
            .any(|entry| entry.graph == 1 && !entry.stopped)
    );
    assert!(
        first_layer
            .iter()
            .any(|entry| entry.graph == 0 && entry.stopped)
    );

    let second_layer = &episode.layers[2].entries;
    assert_eq!(second_layer.len(), 2);
    assert!(
        second_layer
            .iter()
            .any(|entry| entry.graph == 0 && entry.stopped && entry.carried)
    );
    assert!(
        second_layer
            .iter()
            .any(|entry| entry.graph == 1 && entry.stopped && !entry.carried)
    );
}

#[test]
fn unscored_root_returns_without_steps() {
    let mut engine = TestEngine::new().unscored(0);
    let episode = beam(3, 2, None).run_from_root(&mut engine).unwrap();

    assert_eq!(episode.stop_reason, BeamStopReason::UnscoredRoot);
    assert!(episode.steps.is_empty());
    assert!(episode.layers.is_empty());
}

#[test]
fn search_config_hash_changes_when_beam_config_changes() {
    let base = beam(1, 2, None).search_config_hash();
    let max_depth = beam(2, 2, None).search_config_hash();
    let beam_width = beam(1, 3, None).search_config_hash();
    let measure_samples = beam_search_config_hash(
        1,
        2,
        CandidateOptions::default(),
        MeasureOptions::new(MeasureConfigHash::from_bytes([9; 32]), 2, None, true).unwrap(),
    );

    assert_ne!(base, max_depth);
    assert_ne!(base, beam_width);
    assert_ne!(base, measure_samples);
}
