mod common;

use common::{TestEngine, candidate_ref, measure_options, stop_ref};
use gz_engine::{CandidateOptions, MeasureConfigHash, MeasureOptions};
use gz_search::{
    RandomEpisode, RandomSearch, RandomSearchConfig, RandomStopReason, SearchAction,
    random_search_config_hash,
};

fn random(max_steps: usize, seed: u64, max_candidates: Option<usize>) -> RandomSearch {
    RandomSearch::new(RandomSearchConfig {
        max_steps,
        seed,
        candidate_options: CandidateOptions {
            max_candidates,
            deterministic_order: true,
        },
        measure_options: measure_options(),
    })
}

fn first_step(episode: &RandomEpisode<u8, u8>) -> &gz_search::SearchStep<u8, u8> {
    episode.steps.first().unwrap()
}

#[test]
fn no_engine_candidates_selects_stop() {
    let mut engine = TestEngine::new().reward(0, 1.0);
    let episode = random(3, 0, None).run_from_root(&mut engine).unwrap();
    let step = first_step(&episode);

    assert_eq!(episode.stop_reason, RandomStopReason::SelectedStop);
    assert_eq!(step.action, SearchAction::Stop);
    assert_eq!(step.selected_action, stop_ref(0));
    assert_eq!(step.selected_rank, 0);
    assert_eq!(step.action_count, 1);
}

#[test]
fn selected_candidate_is_applied_and_measured() {
    let mut engine = TestEngine::new()
        .candidates(0, vec![1])
        .apply(0, 1, 1)
        .reward(0, 0.0)
        .reward(1, 3.0);
    let episode = random(1, 2, None).run_from_root(&mut engine).unwrap();
    let step = first_step(&episode);

    assert_eq!(episode.stop_reason, RandomStopReason::MaxSteps);
    assert_eq!(episode.final_graph, 1);
    assert_eq!(step.action, SearchAction::Candidate(1));
    assert_eq!(step.selected_action, candidate_ref(0, 1));
    assert_eq!(step.selected_rank, 0);
    assert_eq!(step.selected_measure.scalar_reward, Some(3.0));
}

#[test]
fn unscored_current_graph_returns_without_stop_step() {
    let mut engine = TestEngine::new().unscored(0);
    let episode = random(3, 0, None).run_from_root(&mut engine).unwrap();

    assert_eq!(episode.stop_reason, RandomStopReason::UnscoredCurrentGraph);
    assert!(episode.steps.is_empty());
}

#[test]
fn same_seed_repeats_path() {
    let mut left_engine = TestEngine::new()
        .candidates(0, vec![1, 2, 3])
        .reward(0, 0.0)
        .reward(1, 1.0)
        .reward(2, 2.0)
        .reward(3, 3.0);
    let left = random(1, 2, None).run_from_root(&mut left_engine).unwrap();

    let mut right_engine = TestEngine::new()
        .candidates(0, vec![1, 2, 3])
        .reward(0, 0.0)
        .reward(1, 1.0)
        .reward(2, 2.0)
        .reward(3, 3.0);
    let right = random(1, 2, None).run_from_root(&mut right_engine).unwrap();

    assert_eq!(left.final_graph, right.final_graph);
    assert_eq!(left.steps[0].action, right.steps[0].action);
    assert_eq!(left.steps[0].selected_rank, right.steps[0].selected_rank);
}

#[test]
fn search_config_hash_changes_when_random_config_changes() {
    let base = random(1, 2, None).search_config_hash();
    let max_steps = random(2, 2, None).search_config_hash();
    let seed = random(1, 3, None).search_config_hash();
    let max_candidates = random(1, 2, Some(1)).search_config_hash();
    let measure_samples = random_search_config_hash(
        1,
        2,
        CandidateOptions::default(),
        MeasureOptions::new(MeasureConfigHash::from_bytes([9; 32]), 2, None, true).unwrap(),
    );

    assert_ne!(base, max_steps);
    assert_ne!(base, seed);
    assert_ne!(base, max_candidates);
    assert_ne!(base, measure_samples);
}
