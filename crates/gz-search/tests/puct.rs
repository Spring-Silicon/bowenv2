mod common;

use common::{TestEngine, measure_options};
use gz_engine::{
    CandidateOptions, EngineResult, GraphEngine, MeasureConfigHash, MeasureOptions, ModelVersion,
};
use gz_eval::{EngineEvalRequest, EngineEvaluator, EvalOutput, EvalRequest, EvalResult, Evaluator};
use gz_search::{
    PuctEpisodeContext, PuctMcts, PuctMctsConfig, PuctSearchContext, PuctStopReason, SearchAction,
};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

#[derive(Clone)]
struct EvalRow {
    logits: Vec<f32>,
    value: f32,
}

#[derive(Default)]
struct RecordedEvaluator {
    rows: BTreeMap<u8, EvalRow>,
}

impl RecordedEvaluator {
    fn row(mut self, graph: u8, logits: impl Into<Vec<f32>>, value: f32) -> Self {
        self.rows.insert(
            graph,
            EvalRow {
                logits: logits.into(),
                value,
            },
        );
        self
    }
}

impl Evaluator for RecordedEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        out.clear();
        for request in requests {
            request.validate_ref()?;
            let graph = request.context.graph.graph_hash.as_bytes()[0];
            let row = self.rows.get(&graph).cloned().unwrap_or(EvalRow {
                logits: vec![0.0; request.action_count()],
                value: 0.0,
            });
            out.push(EvalOutput {
                model_version: ModelVersion::from_bytes([7; 16]),
                policy_logits: row.logits,
                value: row.value,
            });
        }
        Ok(())
    }
}

#[derive(Default)]
struct MeasureEvaluator {
    leaf_measures: usize,
}

impl EngineEvaluator<TestEngine> for MeasureEvaluator {
    fn evaluate(
        &mut self,
        engine: &mut TestEngine,
        input: EngineEvalRequest<'_, TestEngine>,
    ) -> EngineResult<EvalOutput> {
        input.request.validate_ref().unwrap();
        self.leaf_measures += 1;
        let value = engine
            .measure(input.graph, input.measure_options)?
            .scalar_reward
            .unwrap();
        Ok(EvalOutput {
            model_version: ModelVersion::from_bytes([8; 16]),
            policy_logits: vec![0.0; input.request.action_count()],
            value,
        })
    }
}

fn config(max_steps: usize, simulations: usize) -> PuctMctsConfig {
    PuctMctsConfig {
        max_steps,
        simulations: NonZeroUsize::new(simulations).unwrap(),
        c_puct: 1.0,
        seed: 0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(),
    }
}

#[test]
fn equal_logits_produce_uniform_prior_and_visit_target() {
    let mut engine = TestEngine::new().candidates(0, [1, 2]);
    let mut evaluator = RecordedEvaluator::default().row(0, [0.0, 0.0, 0.0], 0.0);
    let result = PuctMcts::new(config(1, 3))
        .search_root(&mut engine, &mut evaluator, 0, PuctSearchContext::default())
        .unwrap();

    assert_eq!(result.stats.simulations, 3);
    assert_eq!(result.policy_target, vec![1.0 / 3.0; 3]);
    assert_eq!(result.considered_action_indices, vec![0, 1, 2]);
    assert!(matches!(result.selected_action, SearchAction::Candidate(1)));
}

#[test]
fn higher_prior_wins_when_q_and_visits_are_equal() {
    let mut engine = TestEngine::new().candidates(0, [1, 2]);
    let mut evaluator = RecordedEvaluator::default().row(0, [-5.0, 5.0, -10.0], 0.0);
    let result = PuctMcts::new(config(1, 1))
        .search_root(&mut engine, &mut evaluator, 0, PuctSearchContext::default())
        .unwrap();

    assert_eq!(result.policy_target, vec![0.0, 1.0, 0.0]);
    assert!(matches!(result.selected_action, SearchAction::Candidate(2)));
}

#[test]
fn higher_q_dominates_with_zero_exploration() {
    let mut engine = TestEngine::new().candidates(0, [1, 2]);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [0.0, 0.0, 0.0], 0.0)
        .row(1, [0.0], -1.0)
        .row(2, [0.0], 2.0);
    let mut search_config = config(1, 3);
    search_config.c_puct = 0.0;
    let result = PuctMcts::new(search_config)
        .search_root(&mut engine, &mut evaluator, 0, PuctSearchContext::default())
        .unwrap();

    assert_eq!(result.policy_target, vec![1.0 / 3.0, 2.0 / 3.0, 0.0]);
    assert!(matches!(result.selected_action, SearchAction::Candidate(2)));
    assert_eq!(result.root_q_max, 2.0);
}

#[test]
fn rejected_rewrite_restores_stop_without_consuming_budget() {
    let mut engine = TestEngine::new().candidates(0, [1]).rejected(0, 1);
    let mut evaluator = RecordedEvaluator::default().row(0, [10.0, -10.0], 3.0);
    let mut search_config = config(1, 2);
    search_config.mask_stop = true;
    let result = PuctMcts::new(search_config)
        .search_root(&mut engine, &mut evaluator, 0, PuctSearchContext::default())
        .unwrap();

    assert_eq!(result.stats.simulations, 2);
    assert_eq!(result.policy_target, vec![0.0, 1.0]);
    assert!(matches!(result.selected_action, SearchAction::Stop));
    assert_eq!(engine.apply_calls, vec![(0, 1)]);
}

#[test]
fn measurement_backed_leaf_values_do_not_replace_final_measurement() {
    let mut engine = TestEngine::new()
        .candidates(0, [1])
        .candidates(2, [])
        .apply(0, 1, 2)
        .reward(0, 0.0)
        .reward(2, 5.0);
    let mut evaluator = MeasureEvaluator::default();
    let episode = PuctMcts::new(config(1, 1))
        .run(
            &mut engine,
            &mut evaluator,
            0,
            PuctEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.final_graph, 2);
    assert_eq!(episode.final_measure.scalar_reward, Some(5.0));
    assert_eq!(evaluator.leaf_measures, 2);
    assert_eq!(engine.measure_calls, vec![0, 2, 2]);
}

#[test]
fn zero_step_episode_skips_eval_and_measures_root() {
    let mut engine = TestEngine::new().reward(0, 4.0);
    let mut evaluator = MeasureEvaluator::default();
    let episode = PuctMcts::new(config(0, 1))
        .run(
            &mut engine,
            &mut evaluator,
            0,
            PuctEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.stop_reason, PuctStopReason::MaxSteps);
    assert!(episode.steps.is_empty());
    assert_eq!(evaluator.leaf_measures, 0);
    assert_eq!(engine.measure_calls, vec![0]);
}

#[test]
fn serial_episode_releases_dead_handles_before_returning_retained_handles() {
    let mut engine = TestEngine::new()
        .candidates(0, [1, 2])
        .candidates(20, [])
        .apply(0, 2, 20)
        .reward(20, 20.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [-10.0, 10.0, -10.0], 0.0)
        .row(20, [0.0], 20.0);
    let episode = PuctMcts::new(config(1, 1))
        .run(
            &mut engine,
            &mut evaluator,
            0,
            PuctEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(engine.released_graphs, Vec::<u8>::new());
    assert_eq!(engine.released_candidates, vec![1, 2]);
    assert_eq!(episode.created_graphs, vec![20]);
    assert_eq!(episode.created_candidates, Vec::<u8>::new());
}

#[test]
fn serial_episode_releases_owned_handles_after_invalid_eval() {
    let mut engine = TestEngine::new().candidates(0, [1, 2]);
    let mut evaluator = RecordedEvaluator::default().row(0, [0.0], 0.0);
    let error = PuctMcts::new(config(1, 1))
        .run(
            &mut engine,
            &mut evaluator,
            0,
            PuctEpisodeContext::default(),
        )
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "internal engine error: code 2: eval failed: expected 3 policy logits, got 1"
    );
    assert_eq!(engine.released_graphs, Vec::<u8>::new());
    assert_eq!(engine.released_candidates, vec![1, 2]);
}

#[test]
fn tree_reuse_carries_ledgers_but_targets_fresh_visits() {
    let mut engine = TestEngine::new()
        .candidates(0, [1])
        .candidates(10, [2])
        .candidates(20, [])
        .apply(0, 1, 10)
        .apply(10, 2, 20)
        .reward(20, 20.0);
    let mut evaluator = RecordedEvaluator::default()
        .row(0, [10.0, -10.0], 0.0)
        .row(10, [10.0, -10.0], 1.0)
        .row(20, [0.0], 2.0);
    let mut search_config = config(2, 4);
    search_config.tree_reuse = true;
    search_config.mask_stop = true;
    let episode = PuctMcts::new(search_config)
        .run(
            &mut engine,
            &mut evaluator,
            0,
            PuctEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(episode.final_graph, 20);
    assert_eq!(episode.root_stats.len(), 2);
    assert!(episode.root_stats[1].carried_nodes > 0);
    assert!(episode.root_stats[1].carried_root_visits > 0);
    assert_eq!(episode.root_stats[1].simulations, 4);
    for step in &episode.steps {
        assert!((step.policy_target.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    }
}

#[test]
fn temperature_sampling_repeats_for_same_seed_and_varies_across_seeds() {
    fn selected(noise_seed: u64) -> usize {
        let mut engine = TestEngine::new().candidates(0, [1, 2]);
        let mut evaluator = RecordedEvaluator::default().row(0, [0.0, 0.0, 0.0], 0.0);
        PuctMcts::new(config(1, 3))
            .search_root(
                &mut engine,
                &mut evaluator,
                0,
                PuctSearchContext {
                    selection_temperature: 1.0,
                    noise_seed,
                    ..PuctSearchContext::default()
                },
            )
            .unwrap()
            .selected_action_index
    }

    assert_eq!(selected(7), selected(7));
    let choices = (1..=32)
        .map(selected)
        .collect::<std::collections::HashSet<_>>();
    assert!(choices.len() > 1);
}

#[test]
fn search_config_hash_covers_behavior_but_not_position_export() {
    let base = config(3, 4);
    let base_hash = PuctMcts::new(base).search_config_hash();
    let variants = [
        PuctMctsConfig {
            max_steps: 4,
            ..base
        },
        PuctMctsConfig {
            simulations: NonZeroUsize::new(5).unwrap(),
            ..base
        },
        PuctMctsConfig {
            c_puct: 2.0,
            ..base
        },
        PuctMctsConfig { seed: 1, ..base },
        PuctMctsConfig {
            temperature_moves: 1,
            ..base
        },
        PuctMctsConfig {
            tree_reuse: true,
            ..base
        },
        PuctMctsConfig {
            mask_stop: true,
            ..base
        },
        PuctMctsConfig {
            no_backtrack: true,
            ..base
        },
        PuctMctsConfig {
            candidate_options: CandidateOptions {
                max_candidates: Some(2),
                ..base.candidate_options
            },
            ..base
        },
        PuctMctsConfig {
            measure_options: MeasureOptions::new(
                MeasureConfigHash::from_bytes([10; 32]),
                2,
                Some(100),
                false,
            )
            .unwrap(),
            ..base
        },
    ];
    for variant in variants {
        assert_ne!(PuctMcts::new(variant).search_config_hash(), base_hash);
    }

    assert_eq!(
        PuctMcts::new(PuctMctsConfig {
            export_position: false,
            ..base
        })
        .search_config_hash(),
        base_hash
    );
}
