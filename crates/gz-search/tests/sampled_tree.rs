#[allow(dead_code)]
mod common;

use common::{TestEngine, measure_options};
use gz_engine::{CandidateOptions, GraphEngine, ModelVersion};
use gz_eval::EvalOutput;
use gz_search::{
    EngineIdentity, EvalModel, ExpandResult, ExpandedCandidate, GumbelEpisode,
    GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig, GumbelPlayer, GumbelValueMode,
    SampledTreeEpisodeTask, SampledTreeRootTask, SearchPoll, SearchWork, SearchWorkResult,
};
use std::collections::HashSet;
use std::num::NonZeroUsize;

fn config(max_steps: usize, simulations: usize) -> GumbelMctsConfig {
    GumbelMctsConfig {
        max_steps,
        simulations: NonZeroUsize::new(simulations).unwrap(),
        max_considered_actions: NonZeroUsize::MIN,
        seed: 0,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        value_mode: GumbelValueMode::Competitive,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(),
    }
}

fn expand_result(
    engine: &mut TestEngine,
    graph: u8,
    options: CandidateOptions,
) -> ExpandResult<u8> {
    let mut candidates = Vec::new();
    engine.candidates(graph, options, &mut candidates).unwrap();
    ExpandResult {
        graph_hash: engine.hash(graph).unwrap(),
        candidates: candidates
            .into_iter()
            .map(|candidate| {
                let info = engine.candidate_info(graph, candidate).unwrap();
                ExpandedCandidate {
                    candidate,
                    candidate_hash: info.candidate_hash,
                    kind: info.kind,
                    tags: info.tags,
                    static_prior: info.static_prior,
                }
            })
            .collect(),
    }
}

struct DrivenEpisode {
    episode: GumbelEpisode<u8, u8>,
    current_pair_evals: usize,
    incumbent_evals: usize,
    released_graphs: usize,
    released_candidates: usize,
    apply_calls: usize,
    expanded_candidates: usize,
}

fn drive(
    mut engine: TestEngine,
    noise_seed: u64,
    max_steps: usize,
    current_stops: bool,
    simulations: usize,
    mask_stop: bool,
) -> DrivenEpisode {
    let mut search_config = config(max_steps, simulations);
    search_config.mask_stop = mask_stop;
    let search = GumbelMcts::new(search_config);
    let mut task = SampledTreeEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        GumbelEpisodeContext {
            noise_seed,
            opponent: None,
        },
    );
    let mut current_pair_evals = 0;
    let mut incumbent_evals = 0;
    let mut released_graphs = 0;
    let mut released_candidates = 0;
    let mut expanded_candidates = 0;

    loop {
        match task.poll().unwrap() {
            SearchPoll::Work(work) => {
                let token = work.token();
                let result = match work {
                    SearchWork::Expand(work) => {
                        let result = expand_result(&mut engine, work.graph, work.options);
                        expanded_candidates += result.candidates.len();
                        SearchWorkResult::Expand(result)
                    }
                    SearchWork::Apply(work) => SearchWorkResult::Apply(
                        GraphEngine::apply(&mut engine, work.graph, work.candidate).unwrap(),
                    ),
                    SearchWork::Measure(work) => {
                        SearchWorkResult::Measure(engine.measure(work.graph, work.options).unwrap())
                    }
                    SearchWork::Eval(work) => {
                        let stop = work.request.actions.len() - 1;
                        let mut logits = vec![-20.0; work.request.actions.len()];
                        let version = match work.model {
                            EvalModel::Current => {
                                assert!(work.opponent.is_some());
                                current_pair_evals += 1;
                                if current_stops {
                                    logits[stop] = 20.0;
                                } else if stop > 0 {
                                    logits[0] = 20.0;
                                } else {
                                    logits[stop] = 20.0;
                                }
                                ModelVersion::from_bytes([7; 16])
                            }
                            EvalModel::Incumbent => {
                                assert!(work.opponent.is_none());
                                incumbent_evals += 1;
                                if stop > 0 {
                                    logits[(stop - 1).min(1)] = 20.0;
                                } else {
                                    logits[stop] = 20.0;
                                }
                                ModelVersion::from_bytes([8; 16])
                            }
                            EvalModel::Episode => panic!("sampled-tree emitted episode routing"),
                        };
                        SearchWorkResult::Eval(EvalOutput {
                            model_version: version,
                            policy_logits: logits,
                            value: 0.0,
                        })
                    }
                    _ => panic!("unsupported sampled-tree work"),
                };
                task.resume(token, result).unwrap();
                let handles = task.take_releasable();
                released_graphs += handles.graphs.len();
                released_candidates += handles.candidates.len();
            }
            SearchPoll::Blocked => panic!("sampled-tree blocked without pending work"),
            SearchPoll::Done(episode) => {
                let apply_calls = engine.apply_calls.len();
                return DrivenEpisode {
                    episode,
                    current_pair_evals,
                    incumbent_evals,
                    released_graphs,
                    released_candidates,
                    apply_calls,
                    expanded_candidates,
                };
            }
        }
    }
}

fn one_step_engine() -> TestEngine {
    TestEngine::new()
        .candidates(0, [1, 2])
        .apply(0, 1, 1)
        .apply(0, 2, 2)
        .reward(1, 5.0)
        .reward(2, 3.0)
}

struct DrivenRoot {
    simulations: usize,
    eval_count: usize,
    current_evals: usize,
    incumbent_evals: usize,
    early_released_graphs: usize,
    released_graphs: usize,
    released_candidates: usize,
    apply_calls: usize,
    expanded_candidates: usize,
    selected_stop: bool,
}

fn drive_root(mut engine: TestEngine, max_considered: usize, simulations: usize) -> DrivenRoot {
    let mut search_config = config(1, simulations);
    search_config.max_considered_actions = NonZeroUsize::new(max_considered).unwrap();
    let search = GumbelMcts::new(search_config);
    let mut task = SampledTreeRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        GumbelPlayer::One,
        0,
        0,
        None,
        0,
        0,
        false,
        0,
        HashSet::new(),
    );
    let mut current_evals = 0;
    let mut incumbent_evals = 0;
    let mut early_released_graphs = 0;
    let mut released_graphs = 0;
    let mut released_candidates = 0;
    let mut expanded_candidates = 0;

    loop {
        match task.poll().unwrap() {
            SearchPoll::Work(work) => {
                let handles = task.take_releasable();
                early_released_graphs += handles.graphs.len();
                released_graphs += handles.graphs.len();
                released_candidates += handles.candidates.len();
                let token = work.token();
                let result = match work {
                    SearchWork::Expand(work) => {
                        let result = expand_result(&mut engine, work.graph, work.options);
                        expanded_candidates += result.candidates.len();
                        SearchWorkResult::Expand(result)
                    }
                    SearchWork::Apply(work) => SearchWorkResult::Apply(
                        GraphEngine::apply(&mut engine, work.graph, work.candidate).unwrap(),
                    ),
                    SearchWork::Measure(work) => {
                        SearchWorkResult::Measure(engine.measure(work.graph, work.options).unwrap())
                    }
                    SearchWork::Eval(work) => {
                        let stop = work.request.actions.len() - 1;
                        let mut logits = vec![0.0; work.request.actions.len()];
                        let version = match work.model {
                            EvalModel::Current => {
                                current_evals += 1;
                                logits[stop] = -20.0;
                                ModelVersion::from_bytes([7; 16])
                            }
                            EvalModel::Incumbent => {
                                incumbent_evals += 1;
                                logits[stop] = -20.0;
                                ModelVersion::from_bytes([8; 16])
                            }
                            EvalModel::Episode => panic!("sampled-tree emitted episode routing"),
                        };
                        SearchWorkResult::Eval(EvalOutput {
                            model_version: version,
                            policy_logits: logits,
                            value: 0.0,
                        })
                    }
                    _ => panic!("unsupported sampled-tree work"),
                };
                task.resume(token, result).unwrap();
                let handles = task.take_releasable();
                early_released_graphs += handles.graphs.len();
                released_graphs += handles.graphs.len();
                released_candidates += handles.candidates.len();
            }
            SearchPoll::Blocked => panic!("sampled-tree blocked without pending work"),
            SearchPoll::Done(result) => {
                let handles = task.take_releasable();
                released_graphs += handles.graphs.len();
                released_candidates += handles.candidates.len();
                return DrivenRoot {
                    simulations: result.stats.simulations,
                    eval_count: result.stats.eval_count,
                    current_evals,
                    incumbent_evals,
                    early_released_graphs,
                    released_graphs,
                    released_candidates,
                    apply_calls: engine.apply_calls.len(),
                    expanded_candidates,
                    selected_stop: result.selected_stop,
                };
            }
        }
    }
}

#[test]
fn sampled_tree_shares_incumbent_policy_across_root_branches() {
    let driven = drive_root(one_step_engine(), 2, 2);

    assert_eq!(driven.simulations, 2);
    assert_eq!(driven.current_evals, 1);
    assert_eq!(driven.incumbent_evals, 1);
    assert_eq!(driven.eval_count, 2);
}

#[test]
fn sampled_tree_releases_halving_losers_before_root_completion() {
    let engine = TestEngine::new()
        .candidates(0, [1, 2, 3, 4])
        .apply(0, 1, 1)
        .apply(0, 2, 2)
        .apply(0, 3, 3)
        .apply(0, 4, 4)
        .reward(1, 5.0)
        .reward(2, 4.0)
        .reward(3, 3.0)
        .reward(4, 2.0);
    let driven = drive_root(engine, 4, 8);

    assert_eq!(driven.simulations, 8);
    assert_eq!(driven.early_released_graphs, 4);
    assert!(!driven.selected_stop);
    assert_eq!(driven.released_graphs + 1, driven.apply_calls);
    assert_eq!(driven.released_candidates, driven.expanded_candidates);
}

#[test]
fn sampled_tree_exercises_both_roles_and_separates_policy_streams() {
    let mut seen = [false; 2];
    for seed in 0..64 {
        let driven = drive(one_step_engine(), seed, 1, false, 1, false);
        let trace = driven.episode.competitive.as_deref().unwrap();
        seen[match trace.learner_player {
            GumbelPlayer::One => 0,
            GumbelPlayer::Two => 1,
        }] = true;

        assert_eq!(driven.episode.steps.len(), 1);
        assert_eq!(trace.opponent_steps.len(), 1);
        let learner_step = &driven.episode.steps[0];
        assert_eq!(learner_step.legal_actions.len(), 3);
        assert_eq!(
            learner_step.selected_action,
            learner_step.legal_actions[learner_step.selected_rank]
        );
        assert!(learner_step.selected_candidate.is_some());
        assert!(learner_step.policy_target.iter().sum::<f32>() > 0.99);
        assert!(
            trace.opponent_steps[0]
                .policy_target
                .iter()
                .all(|target| *target == 0.0)
        );
        assert_eq!(
            driven.episode.steps[0].model_version,
            ModelVersion::from_bytes([7; 16])
        );
        assert_eq!(
            trace.opponent_steps[0].model_version,
            ModelVersion::from_bytes([8; 16])
        );
        assert!(driven.current_pair_evals > 0);
        assert!(driven.incumbent_evals > 0);
        assert_eq!(
            driven.released_graphs + driven.episode.created_graphs.len(),
            driven.apply_calls
        );
        assert_eq!(
            driven.released_candidates + driven.episode.created_candidates.len(),
            driven.expanded_candidates
        );
        if seen == [true, true] {
            break;
        }
    }
    assert_eq!(seen, [true, true]);
}

#[test]
fn sampled_tree_stop_freezes_only_the_learner_actor() {
    let engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(2, [3])
        .apply(0, 2, 2)
        .apply(2, 3, 3)
        .reward(0, 4.0)
        .reward(3, 2.0);
    let driven = drive(engine, 0, 2, true, 1, false);
    let trace = driven.episode.competitive.as_deref().unwrap();

    assert_eq!(driven.episode.steps.len(), 1);
    assert!(matches!(
        driven.episode.steps[0].action,
        gz_search::SearchAction::Stop
    ));
    assert_eq!(trace.opponent_steps.len(), 2);
}

#[test]
fn sampled_tree_rejected_incumbent_action_resumes_the_same_chance_node() {
    let engine = one_step_engine().rejected(0, 2);
    let driven = drive(engine, 0, 1, false, 2, false);

    assert_eq!(driven.episode.root_stats[0].simulations, 2);
    assert_eq!(driven.episode.steps.len(), 1);
    assert_eq!(
        driven
            .episode
            .competitive
            .as_deref()
            .unwrap()
            .opponent_steps
            .len(),
        1
    );
}

#[test]
fn sampled_tree_all_rejected_rewrites_restore_stop() {
    let engine = TestEngine::new()
        .candidates(0, [1])
        .rejected(0, 1)
        .reward(0, 4.0);
    let driven = drive(engine, 0, 1, false, 1, true);

    assert!(matches!(
        driven.episode.steps[0].action,
        gz_search::SearchAction::Stop
    ));
    assert!(matches!(
        driven
            .episode
            .competitive
            .as_deref()
            .unwrap()
            .opponent_steps[0]
            .action,
        gz_search::SearchAction::Stop
    ));
}

#[test]
fn sampled_tree_terminal_ties_are_wins_only_for_player_one() {
    let mut seen = [false; 2];
    for seed in 0..64 {
        let engine = one_step_engine().reward(1, 4.0).reward(2, 4.0);
        let driven = drive(engine, seed, 1, false, 1, false);
        let player = driven
            .episode
            .competitive
            .as_deref()
            .unwrap()
            .learner_player;
        let (index, expected) = match player {
            GumbelPlayer::One => (0, 1.0),
            GumbelPlayer::Two => (1, -1.0),
        };
        seen[index] = true;
        assert_eq!(driven.episode.steps[0].root_q_max, expected);
        if seen == [true, true] {
            break;
        }
    }
    assert_eq!(seen, [true, true]);
}

#[test]
#[should_panic(expected = "sampled-tree does not support tree reuse")]
fn sampled_tree_rejects_tree_reuse_at_the_task_boundary() {
    let engine = one_step_engine();
    let mut tree_config = config(1, 1);
    tree_config.tree_reuse = true;
    let search = GumbelMcts::new(tree_config);
    let _: SampledTreeEpisodeTask<u8, u8> = SampledTreeEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0u8,
        GumbelEpisodeContext::default(),
    );
}
