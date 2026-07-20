use gz_engine::{CandidateOptions, EngineResult, GraphEngine, GraphHash};
use gz_engine_whittle::{
    WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor,
    WhittleFeatureExtractorConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
    WhittleGraphId, WhittleRoot, rule_name,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::{EvaluatorProcess, EvaluatorProcessConfig, FeatureEvalBackend, Hello};
use gz_features::{
    ActionFeature, FeatureCollator, FeatureExtractor, FeatureRow, FeatureSchema,
    OpponentStateFeatures, PositionFeatures, STOP_ACTION_KIND_TOKEN, decode_feature_row,
};
use gz_measurer::ValueTargetConfig;
use gz_orchestrator::reference::{Reference, ReferenceProvider};
use gz_orchestrator::{
    FeaturizedRuntime, ReplayRuntime, RootSource, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayEpisodeId, ReplayStore, SampleConfig, SampleKind};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::collections::{BTreeMap, HashMap};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAX_CANDIDATES: usize = 1023;
const MAX_BATCH: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DiagnosticConfig::from_args()?;
    match config.mode {
        Mode::Greedy => run_greedy_match(&config),
        Mode::SymmetricValueDeterministic | Mode::SymmetricValueTraining => {
            run_symmetric_value_diagnostics(&config)
        }
        Mode::DeterministicMcts
        | Mode::TrainingMcts
        | Mode::ValueDeterministic
        | Mode::ValueTraining => run_mcts_match(&config),
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Greedy,
    DeterministicMcts,
    TrainingMcts,
    ValueDeterministic,
    ValueTraining,
    SymmetricValueDeterministic,
    SymmetricValueTraining,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "greedy" => Ok(Self::Greedy),
            "deterministic-mcts" => Ok(Self::DeterministicMcts),
            "training-mcts" => Ok(Self::TrainingMcts),
            "value-deterministic" => Ok(Self::ValueDeterministic),
            "value-training" => Ok(Self::ValueTraining),
            "symmetric-value-deterministic" => Ok(Self::SymmetricValueDeterministic),
            "symmetric-value-training" => Ok(Self::SymmetricValueTraining),
            _ => Err(format!(
                "unknown mode {value:?}; expected greedy, deterministic-mcts, training-mcts, value-deterministic, value-training, symmetric-value-deterministic, or symmetric-value-training"
            )
            .into()),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::DeterministicMcts => "deterministic-mcts",
            Self::TrainingMcts => "training-mcts",
            Self::ValueDeterministic => "value-deterministic",
            Self::ValueTraining => "value-training",
            Self::SymmetricValueDeterministic => "symmetric-value-deterministic",
            Self::SymmetricValueTraining => "symmetric-value-training",
        }
    }

    const fn noise(self) -> (f32, f32) {
        match self {
            Self::Greedy
            | Self::DeterministicMcts
            | Self::ValueDeterministic
            | Self::SymmetricValueDeterministic => (0.0, -1.0),
            Self::TrainingMcts | Self::ValueTraining | Self::SymmetricValueTraining => (1.0, 0.5),
        }
    }

    const fn value_diagnostics(self) -> bool {
        matches!(
            self,
            Self::ValueDeterministic
                | Self::ValueTraining
                | Self::SymmetricValueDeterministic
                | Self::SymmetricValueTraining
        )
    }

    const fn symmetric_value(self) -> bool {
        matches!(
            self,
            Self::SymmetricValueDeterministic | Self::SymmetricValueTraining
        )
    }
}

struct DiagnosticConfig {
    mode: Mode,
    checkpoint_dir: PathBuf,
    compare_checkpoint_dir: Option<PathBuf>,
    device: String,
    current_pointer: String,
    opponent_pointer: String,
    compare_pointer: String,
    cases: usize,
    arena_seed: u64,
    max_steps: usize,
    simulations: usize,
    considered: usize,
    c_scale: f32,
    c_visit: f32,
}

impl DiagnosticConfig {
    fn from_args() -> Result<Self, Box<dyn std::error::Error>> {
        let mode = std::env::args().nth(1).ok_or(usage())?;
        let checkpoint_dir = std::env::args().nth(2).ok_or(usage())?;
        let compare_checkpoint_dir = std::env::args()
            .nth(13)
            .map(|path| absolute(Path::new(&path)))
            .transpose()?;
        let config = Self {
            mode: Mode::parse(&mode)?,
            checkpoint_dir: absolute(Path::new(&checkpoint_dir))?,
            compare_checkpoint_dir,
            device: arg_or(3, "cuda:0"),
            current_pointer: arg_or(4, "latest.json"),
            opponent_pointer: arg_or(5, "best.json"),
            compare_pointer: arg_or(14, "latest.json"),
            cases: parse_arg_or(6, 128)?,
            arena_seed: parse_arg_or(7, 910_000_001)?,
            max_steps: parse_arg_or(8, 96)?,
            simulations: parse_arg_or(9, 128)?,
            considered: parse_arg_or(10, 32)?,
            c_scale: parse_arg_or(11, 1.0)?,
            c_visit: parse_arg_or(12, 50.0)?,
        };
        if config.cases == 0 || config.cases > MAX_BATCH {
            return Err(format!("CASES must be in 1..={MAX_BATCH}").into());
        }
        if config.max_steps == 0 || config.simulations == 0 || config.considered == 0 {
            return Err("MAX_STEPS, SIMULATIONS, and CONSIDERED must be nonzero".into());
        }
        if !config.c_scale.is_finite() || config.c_scale < 0.0 {
            return Err("C_SCALE must be finite and non-negative".into());
        }
        if !config.c_visit.is_finite() || config.c_visit < 0.0 {
            return Err("C_VISIT must be finite and non-negative".into());
        }
        Ok(config)
    }
}

fn usage() -> &'static str {
    "usage: diagnose_checkpoint_match MODE CHECKPOINT_DIR [DEVICE] [CURRENT_POINTER] [OPPONENT_POINTER] [CASES] [ARENA_SEED] [MAX_STEPS] [SIMULATIONS] [CONSIDERED] [C_SCALE] [C_VISIT] [COMPARE_CHECKPOINT_DIR] [COMPARE_POINTER]"
}

fn run_greedy_match(config: &DiagnosticConfig) -> Result<(), Box<dyn std::error::Error>> {
    let current = greedy_rollouts(config, &config.current_pointer, "greedy-current")?;
    let opponent = greedy_rollouts(config, &config.opponent_pointer, "greedy-opponent")?;
    let results = current
        .rows
        .iter()
        .zip(&opponent.rows)
        .map(|(current, opponent)| {
            if current.index != opponent.index
                || current.root_hash != opponent.root_hash
                || current.root_cost != opponent.root_cost
            {
                return Err("greedy rollout root mismatch".into());
            }
            Ok(MatchResult {
                index: current.index,
                root_hash: current.root_hash,
                root_cost: current.root_cost,
                current_cost: current.final_cost,
                opponent_cost: opponent.final_cost,
                current_steps: current.steps,
                opponent_steps: opponent.steps,
                current_stopped: current.stopped,
                opponent_stopped: opponent.stopped,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    print_results(
        config,
        current.model_version,
        opponent.model_version,
        &results,
    );
    Ok(())
}

fn greedy_rollouts(
    config: &DiagnosticConfig,
    pointer: &str,
    role: &str,
) -> Result<GreedyRun, Box<dyn std::error::Error>> {
    let mut rollouts = generated_policy_rollouts(config)?;
    let schema = rollouts
        .first()
        .ok_or("diagnostic root set is empty")?
        .extractor
        .schema()
        .clone();
    let first = rollouts.first().ok_or("diagnostic root set is empty")?;
    let mut process = spawn_evaluator(role, config, pointer)?;
    let hello = Hello::new(
        schema.hash(),
        MAX_BATCH as u32,
        first.engine.engine_id(),
        first.engine.engine_version(),
        first.engine.action_set_hash(),
    );
    let mut backend = process.connect(&hello)?;
    let model_version = backend.model_version();
    let mut collator = FeatureCollator::new(schema, nonzero(MAX_BATCH)?);
    let candidate_options = CandidateOptions {
        max_candidates: Some(MAX_CANDIDATES),
        deterministic_order: true,
    };
    let mut rows = Vec::with_capacity(rollouts.len());
    let mut active = Vec::with_capacity(rollouts.len());
    let mut candidate_batches = Vec::<Vec<WhittleCandidateId>>::with_capacity(rollouts.len());
    let mut batch_bytes = Vec::new();

    for step in 0..config.max_steps {
        rows.clear();
        active.clear();
        candidate_batches.clear();
        for (index, rollout) in rollouts.iter_mut().enumerate() {
            if rollout.stopped {
                continue;
            }
            let mut candidates = Vec::new();
            rollout
                .engine
                .candidates(rollout.graph, candidate_options, &mut candidates)?;
            let row = rollout.extractor.extract(
                &rollout.engine,
                rollout.graph,
                &candidates,
                PositionFeatures {
                    root_step: step as u32,
                    leaf_depth: 0,
                    budget_fraction: (config.max_steps - step) as f32 / config.max_steps as f32,
                    budget_step: 1.0 / config.max_steps as f32,
                    opponent_reward: 0.0,
                    opponent_present: false,
                },
            )?;
            active.push(index);
            candidate_batches.push(candidates);
            rows.push(row);
        }
        if rows.is_empty() {
            break;
        }

        let action_counts = rows
            .iter()
            .map(|row| row.actions.len() as u32)
            .collect::<Vec<_>>();
        collator.collate_into(&rows, &mut batch_bytes)?;
        let outputs = backend.eval(&batch_bytes, &action_counts)?;
        for ((rollout_index, candidates), output) in active
            .iter()
            .copied()
            .zip(candidate_batches.drain(..))
            .zip(outputs.rows)
        {
            let rollout = &mut rollouts[rollout_index];
            let stop = candidates.len();
            let mut ranking = (0..output.policy_logits.len()).collect::<Vec<_>>();
            ranking.sort_by(|&left, &right| {
                output.policy_logits[right]
                    .total_cmp(&output.policy_logits[left])
                    .then_with(|| left.cmp(&right))
            });

            let mut advanced = false;
            for action in ranking {
                if action == stop {
                    rollout.engine.release(&[], &candidates)?;
                    rollout.steps += 1;
                    rollout.stopped = true;
                    advanced = true;
                    break;
                }
                let applied = rollout.engine.apply(rollout.graph, candidates[action])?;
                if applied.rejected.is_some() {
                    rollout.engine.release(&[applied.after], &[])?;
                    continue;
                }
                rollout.engine.release(&[rollout.graph], &candidates)?;
                rollout.graph = applied.after;
                rollout.steps += 1;
                advanced = true;
                break;
            }
            if !advanced {
                rollout.engine.release(&[], &candidates)?;
                return Err("greedy policy had no applicable action or STOP".into());
            }
        }
    }

    drop(backend);
    wait_for_process_exit(&mut process)?;
    let mut results = Vec::with_capacity(rollouts.len());
    for mut rollout in rollouts {
        let final_cost = measure_cost(&mut rollout.engine, rollout.graph)?;
        results.push(GreedyResult {
            index: rollout.index,
            root_hash: rollout.root_hash,
            root_cost: rollout.root_cost,
            final_cost,
            steps: rollout.steps,
            stopped: rollout.stopped,
        });
        rollout.engine.release(&[rollout.graph], &[])?;
    }
    Ok(GreedyRun {
        model_version,
        rows: results,
    })
}

fn run_mcts_match(config: &DiagnosticConfig) -> Result<(), Box<dyn std::error::Error>> {
    let generated = generated_mcts_roots(config)?;
    let root_info = generated
        .info
        .iter()
        .map(|info| (info.root_hash, *info))
        .collect::<HashMap<_, _>>();
    if root_info.len() != config.cases {
        return Err("generated diagnostic roots contain duplicate graph hashes".into());
    }
    let schema = generated
        .extractors
        .first()
        .ok_or("diagnostic root set is empty")?
        .schema()
        .clone();
    let first_engine = generated
        .engines
        .first()
        .ok_or("diagnostic root set is empty")?;
    let mut current_process = spawn_evaluator("mcts-current", config, &config.current_pointer)?;
    let hello = Hello::new(
        schema.hash(),
        MAX_BATCH as u32,
        first_engine.engine_id(),
        first_engine.engine_version(),
        first_engine.action_set_hash(),
    );
    let current_backend = current_process.connect(&hello)?;
    let current_version = current_backend.model_version();
    let mut opponent_process = spawn_evaluator("mcts-opponent", config, &config.opponent_pointer)?;
    let opponent_backend = opponent_process.connect(&hello)?;
    let opponent_version = opponent_backend.model_version();
    let (gumbel_scale, gumbel_noise_overlap) = config.mode.noise();
    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps: config.max_steps,
        simulations: nonzero(config.simulations)?,
        max_considered_actions: nonzero(config.considered)?,
        seed: 42,
        gumbel_scale,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        temperature_moves: 0,
        gumbel_noise_overlap,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: true,
        value_mode: gz_search::GumbelValueMode::Competitive,
        candidate_options: CandidateOptions {
            max_candidates: Some(MAX_CANDIDATES),
            deterministic_order: true,
        },
        measure_options: first_engine.measure_options(),
    });
    let placeholder = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 42,
        ..RandomValueEvaluatorConfig::default()
    })?;
    let orchestrator = ThreadedGumbelOrchestrator::new(
        generated.engines,
        placeholder,
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: nonzero(1)?,
            max_batch: nonzero(MAX_BATCH)?,
            flush_after: Duration::from_millis(3),
            admission_stagger: Duration::ZERO,
            admission_smoothing: None,
        },
    );
    let replay_dir = temporary_replay_dir(config.mode.name())?;
    let store = ReplayStore::open(&replay_dir)?;
    let providers = (0..config.cases).map(|_| EvalSampledTreeProvider).collect();
    let run = orchestrator.run_featurized_with_replay(
        generated.roots,
        GumbelEpisodeContext::default(),
        FeaturizedRuntime {
            extractors: generated.extractors,
            backends: vec![current_backend],
            reference_backends: vec![opponent_backend],
            challenger_backends: vec![],
        },
        ReplayRuntime {
            store: &store,
            providers,
            backpressure: None,
            length_tiebreak: true,
            value_target: ValueTargetConfig::Sign,
        },
    )?;
    wait_for_process_exit(&mut current_process)?;
    wait_for_process_exit(&mut opponent_process)?;
    if run.episodes_dropped != 0 || run.episodes_appended as usize != config.cases {
        return Err(format!(
            "diagnostic completion mismatch: appended={} dropped={} expected={}",
            run.episodes_appended, run.episodes_dropped, config.cases
        )
        .into());
    }

    let mut results = Vec::with_capacity(config.cases);
    let mut total_value_rows = 0_u64;
    for game in 0..config.cases {
        let current = store
            .episode(ReplayEpisodeId::new((game * 2) as u64))?
            .ok_or_else(|| format!("missing current replay record for game {game}"))?;
        let opponent = store
            .episode(ReplayEpisodeId::new((game * 2 + 1) as u64))?
            .ok_or_else(|| format!("missing opponent replay record for game {game}"))?;
        if current.root.graph.graph_hash != opponent.root.graph.graph_hash {
            return Err(format!("game {game} root mismatch between replay perspectives").into());
        }
        total_value_rows = total_value_rows
            .checked_add(u64::from(current.row_count) + u64::from(opponent.row_count))
            .ok_or("diagnostic value row count overflow")?;
        let info = root_info
            .get(&current.root.graph.graph_hash)
            .ok_or_else(|| format!("game {game} has an unknown generated root"))?;
        results.push(MatchResult {
            index: info.index,
            root_hash: info.root_hash,
            root_cost: info.root_cost,
            current_cost: -current.outcome.learner_reward,
            opponent_cost: -opponent.outcome.learner_reward,
            current_steps: current.steps.len(),
            opponent_steps: opponent.steps.len(),
            current_stopped: current.outcome.stopped,
            opponent_stopped: opponent.outcome.stopped,
        });
    }
    results.sort_by_key(|result| result.index);
    print_results(config, current_version, opponent_version, &results);
    if config.mode.value_diagnostics() {
        let examples = collect_value_examples(&store, total_value_rows, &schema)?;
        print_value_diagnostics(config, &schema, &hello, current_version, &examples)?;
    }
    drop(store);
    std::fs::remove_dir_all(replay_dir)?;
    Ok(())
}

fn run_symmetric_value_diagnostics(
    config: &DiagnosticConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let generated = generated_mcts_roots(config)?;
    let schema = generated
        .extractors
        .first()
        .ok_or("diagnostic root set is empty")?
        .schema()
        .clone();
    let first_engine = generated
        .engines
        .first()
        .ok_or("diagnostic root set is empty")?;
    let mut process = spawn_evaluator("symmetric-current", config, &config.current_pointer)?;
    let hello = Hello::new(
        schema.hash(),
        MAX_BATCH as u32,
        first_engine.engine_id(),
        first_engine.engine_version(),
        first_engine.action_set_hash(),
    );
    let backend = process.connect(&hello)?;
    let model_version = backend.model_version();
    let (gumbel_scale, gumbel_noise_overlap) = config.mode.noise();
    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps: config.max_steps,
        simulations: nonzero(config.simulations)?,
        max_considered_actions: nonzero(config.considered)?,
        seed: 42,
        gumbel_scale,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        temperature_moves: 0,
        gumbel_noise_overlap,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: true,
        value_mode: gz_search::GumbelValueMode::SymmetricSelfplay,
        candidate_options: CandidateOptions {
            max_candidates: Some(MAX_CANDIDATES),
            deterministic_order: true,
        },
        measure_options: first_engine.measure_options(),
    })
    .with_symmetric_wave_batching(true);
    let placeholder = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 42,
        ..RandomValueEvaluatorConfig::default()
    })?;
    let orchestrator = ThreadedGumbelOrchestrator::new(
        generated.engines,
        placeholder,
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: nonzero(1)?,
            max_batch: nonzero(MAX_BATCH)?,
            flush_after: Duration::from_millis(3),
            admission_stagger: Duration::ZERO,
            admission_smoothing: None,
        },
    );
    let replay_dir = temporary_replay_dir(config.mode.name())?;
    let store = ReplayStore::open(&replay_dir)?;
    let providers = (0..config.cases).map(|_| EvalNoReferenceProvider).collect();
    let run = orchestrator.run_featurized_with_replay(
        generated.roots,
        GumbelEpisodeContext::default(),
        FeaturizedRuntime {
            extractors: generated.extractors,
            backends: vec![backend],
            reference_backends: vec![],
            challenger_backends: vec![],
        },
        ReplayRuntime {
            store: &store,
            providers,
            backpressure: None,
            length_tiebreak: true,
            value_target: ValueTargetConfig::Sign,
        },
    )?;
    wait_for_process_exit(&mut process)?;
    if run.episodes_dropped != 0 || run.episodes_appended as usize != config.cases {
        return Err(format!(
            "symmetric diagnostic completion mismatch: appended={} dropped={} expected={}",
            run.episodes_appended, run.episodes_dropped, config.cases
        )
        .into());
    }

    let mut total_value_rows = 0_u64;
    let mut p1_cost = 0.0_f64;
    let mut p2_cost = 0.0_f64;
    let mut p1_wins = 0_usize;
    let mut p2_wins = 0_usize;
    let mut draws = 0_usize;
    for game in 0..config.cases {
        let p1 = store
            .episode(ReplayEpisodeId::new((game * 2) as u64))?
            .ok_or_else(|| format!("missing P1 replay record for game {game}"))?;
        let p2 = store
            .episode(ReplayEpisodeId::new((game * 2 + 1) as u64))?
            .ok_or_else(|| format!("missing P2 replay record for game {game}"))?;
        if p1.root.graph.graph_hash != p2.root.graph.graph_hash {
            return Err(format!("game {game} root mismatch between replay perspectives").into());
        }
        total_value_rows = total_value_rows
            .checked_add(u64::from(p1.row_count) + u64::from(p2.row_count))
            .ok_or("diagnostic value row count overflow")?;
        p1_cost += f64::from(-p1.outcome.learner_reward);
        p2_cost += f64::from(-p2.outcome.learner_reward);
        match p1.outcome.value_target {
            Some(target) if target > 0.0 => p1_wins += 1,
            Some(target) if target < 0.0 => p2_wins += 1,
            Some(_) => draws += 1,
            None => return Err(format!("game {game} has no symmetric value target").into()),
        }
    }
    let game_count = config.cases as f64;
    let (recorded_early, recorded_late) = store.value_sign_accuracy_emas();
    println!(
        "symmetric_value_settings mode={} cases={} arena_seed={} max_steps={} considered={} simulations={} gumbel_scale={} overlap={} c_scale={} c_visit={} tree_reuse=false wave_batching=true",
        config.mode.name(),
        config.cases,
        config.arena_seed,
        config.max_steps,
        config.considered,
        config.simulations,
        gumbel_scale,
        gumbel_noise_overlap,
        config.c_scale,
        config.c_visit,
    );
    println!(
        "symmetric_value_outcomes model_version={model_version} p1_wins={p1_wins} p2_wins={p2_wins} draws={draws} mean_p1_cost={:.6} mean_p2_cost={:.6} rows={total_value_rows} recorded_search_value_accuracy_early_ema={} recorded_search_value_accuracy_late_ema={}",
        p1_cost / game_count,
        p2_cost / game_count,
        optional_metric(recorded_early),
        optional_metric(recorded_late),
    );
    let examples = collect_value_examples(&store, total_value_rows, &schema)?;
    print_value_diagnostics(config, &schema, &hello, model_version, &examples)?;
    drop(store);
    std::fs::remove_dir_all(replay_dir)?;
    Ok(())
}

fn collect_value_examples(
    store: &ReplayStore,
    expected_rows: u64,
    schema: &FeatureSchema,
) -> Result<Vec<ValueExample>, Box<dyn std::error::Error>> {
    let window_rows = NonZeroU64::new(expected_rows).ok_or("value diagnostic has no rows")?;
    let draw_count = usize::try_from(expected_rows.min(4096))?;
    let mut examples = BTreeMap::new();

    for round in 0..128_u64 {
        let sampled = store.sample_rows_kind(
            SampleConfig {
                batch: nonzero(draw_count)?,
                window_rows,
                seed: 0x7661_6c75_655f_6469 ^ round.wrapping_mul(0x9e37_79b9_7f4a_7c15),
            },
            SampleKind::Value,
        )?;
        for (episode_id, replay_row) in sampled {
            let key = (episode_id.get(), replay_row.step_index);
            let std::collections::btree_map::Entry::Vacant(entry) = examples.entry(key) else {
                continue;
            };
            let target = replay_row
                .value_target
                .ok_or("sampled value row is missing its target")?;
            let selected_index = replay_row
                .legal_actions
                .iter()
                .position(|action| *action == replay_row.selected_action)
                .ok_or("sampled policy row is missing its selected action")?;
            let bytes = replay_row
                .feature_row
                .as_deref()
                .ok_or("sampled value row is missing features")?;
            let row = decode_feature_row(bytes)?;
            row.validate(schema)?;
            if row.opponent.is_none() {
                return Err("pair-value diagnostic row is missing opponent state".into());
            }
            if replay_row.policy_target.len() != row.actions.len()
                || replay_row.legal_actions.len() != row.actions.len()
            {
                return Err("sampled policy row action count mismatch".into());
            }
            entry.insert(ValueExample {
                episode_id: episode_id.get(),
                target,
                policy_target: replay_row.policy_target,
                selected_index,
                row,
            });
        }
        if examples.len() as u64 == expected_rows {
            break;
        }
    }

    if examples.len() as u64 != expected_rows {
        return Err(format!(
            "value diagnostic row coverage mismatch: collected={} expected={expected_rows}",
            examples.len()
        )
        .into());
    }
    Ok(examples.into_values().collect())
}

fn print_value_diagnostics(
    config: &DiagnosticConfig,
    schema: &FeatureSchema,
    hello: &Hello,
    expected_version: gz_engine::ModelVersion,
    examples: &[ValueExample],
) -> Result<(), Box<dyn std::error::Error>> {
    let original_rows = examples
        .iter()
        .map(|example| example.row.clone())
        .collect::<Vec<_>>();
    let swapped_rows = original_rows
        .iter()
        .map(swapped_pair_row)
        .collect::<Result<Vec<_>, _>>()?;

    print_checkpoint_diagnostics(
        "primary",
        "value-analysis-primary",
        config,
        schema,
        hello,
        &config.checkpoint_dir,
        &config.current_pointer,
        Some(expected_version),
        examples,
        &original_rows,
        &swapped_rows,
    )?;
    if let Some(checkpoint_dir) = &config.compare_checkpoint_dir {
        print_checkpoint_diagnostics(
            "comparison",
            "value-analysis-comparison",
            config,
            schema,
            hello,
            checkpoint_dir,
            &config.compare_pointer,
            None,
            examples,
            &original_rows,
            &swapped_rows,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_checkpoint_diagnostics(
    label: &str,
    role: &str,
    config: &DiagnosticConfig,
    schema: &FeatureSchema,
    hello: &Hello,
    checkpoint_dir: &Path,
    pointer: &str,
    expected_version: Option<gz_engine::ModelVersion>,
    examples: &[ValueExample],
    original_rows: &[FeatureRow],
    swapped_rows: &[FeatureRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut process = spawn_evaluator_at(role, config, checkpoint_dir, pointer)?;
    let mut backend = process.connect(hello)?;
    let model_version = backend.model_version();
    if expected_version.is_some_and(|expected| model_version != expected) {
        return Err("value diagnostic evaluator loaded the wrong checkpoint".into());
    }
    let original_outputs = eval_feature_outputs(&mut backend, schema, original_rows)?;
    let swapped_values = eval_feature_values(&mut backend, schema, swapped_rows)?;
    drop(backend);
    wait_for_process_exit(&mut process)?;

    println!(
        "analysis_model label={label} checkpoint_dir={} pointer={pointer} model_version={model_version}",
        checkpoint_dir.display(),
    );
    print_policy_diagnostics(examples, &original_outputs)?;
    let original_values = original_outputs
        .iter()
        .map(|output| output.value)
        .collect::<Vec<_>>();

    let original = examples
        .iter()
        .zip(original_values)
        .map(|(example, prediction)| ValueObservation {
            episode_id: example.episode_id,
            root_step: example.row.position.root_step,
            target: example.target,
            prediction,
        })
        .collect::<Vec<_>>();
    let swapped = examples
        .iter()
        .zip(swapped_values)
        .map(|(example, prediction)| ValueObservation {
            episode_id: example.episode_id,
            root_step: example.row.position.root_step,
            target: -example.target,
            prediction,
        })
        .collect::<Vec<_>>();

    println!(
        "value_dataset rows={} episodes={} coverage_percent=100.000000 model_version={model_version}",
        original.len(),
        config.cases * 2,
    );
    print_value_metric("all", &original);
    print_value_metric("role_swapped", &swapped);
    let episode_means = mean_by_episode(&original)?;
    print_value_metric("episode_mean", &episode_means);
    print_paired_episode_discrimination(&episode_means)?;
    print_value_metric(
        "root_step_0",
        &original
            .iter()
            .copied()
            .filter(|row| row.root_step == 0)
            .collect::<Vec<_>>(),
    );
    for (name, start, end) in [
        ("root_step_1_15", 1, 16),
        ("root_step_16_31", 16, 32),
        ("root_step_32_63", 32, 64),
        ("root_step_64_plus", 64, u32::MAX),
    ] {
        let rows = original
            .iter()
            .copied()
            .filter(|row| row.root_step >= start && row.root_step < end)
            .collect::<Vec<_>>();
        print_value_metric(name, &rows);
    }
    print_calibration(&original);
    print_antisymmetry(&original, &swapped);
    Ok(())
}

fn swapped_pair_row(row: &FeatureRow) -> Result<FeatureRow, Box<dyn std::error::Error>> {
    let opponent = row
        .opponent
        .as_ref()
        .ok_or("cannot swap a value row without opponent state")?;
    let mut own_position = opponent.position;
    own_position.opponent_reward = 0.0;
    own_position.opponent_present = true;
    let mut opponent_position = row.position;
    opponent_position.opponent_reward = 0.0;
    opponent_position.opponent_present = false;
    Ok(FeatureRow {
        node_count: opponent.node_count,
        node_tokens: opponent.node_tokens.clone(),
        node_attrs: opponent.node_attrs.clone(),
        edges: opponent.edges.clone(),
        actions: vec![ActionFeature {
            kind_token: STOP_ACTION_KIND_TOKEN,
            static_prior: 0.0,
            subjects: Vec::new(),
        }],
        position: own_position,
        opponent: Some(OpponentStateFeatures {
            node_count: row.node_count,
            node_tokens: row.node_tokens.clone(),
            node_attrs: row.node_attrs.clone(),
            edges: row.edges.clone(),
            position: opponent_position,
        }),
    })
}

fn eval_feature_values<B: FeatureEvalBackend>(
    backend: &mut B,
    schema: &FeatureSchema,
    rows: &[FeatureRow],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(eval_feature_outputs(backend, schema, rows)?
        .into_iter()
        .map(|output| output.value)
        .collect())
}

fn eval_feature_outputs<B: FeatureEvalBackend>(
    backend: &mut B,
    schema: &FeatureSchema,
    rows: &[FeatureRow],
) -> Result<Vec<FeatureEvaluation>, Box<dyn std::error::Error>> {
    let mut collator = FeatureCollator::new(schema.clone(), nonzero(MAX_BATCH)?);
    let mut batch_bytes = Vec::new();
    let mut evaluated = Vec::with_capacity(rows.len());
    for batch in rows.chunks(MAX_BATCH) {
        let action_counts = batch
            .iter()
            .map(|row| u32::try_from(row.actions.len()))
            .collect::<Result<Vec<_>, _>>()?;
        collator.collate_into(batch, &mut batch_bytes)?;
        let outputs = backend.eval(&batch_bytes, &action_counts)?;
        if outputs.rows.len() != batch.len() {
            return Err("value evaluator returned the wrong row count".into());
        }
        for output in outputs.rows {
            if !output.value.is_finite()
                || output.policy_logits.iter().any(|value| !value.is_finite())
            {
                return Err("evaluator returned a non-finite output".into());
            }
            evaluated.push(FeatureEvaluation {
                value: output.value,
                policy_logits: output.policy_logits,
            });
        }
    }
    Ok(evaluated)
}

struct FeatureEvaluation {
    value: f32,
    policy_logits: Vec<f32>,
}

const POLICY_TOP_K: [usize; 5] = [1, 5, 8, 16, 32];

struct PolicyObservation {
    root_step: u32,
    action_count: usize,
    cross_entropy: f64,
    kl: f64,
    model_entropy: f64,
    target_entropy: f64,
    model_top_probability: f64,
    target_at_model_top: f64,
    target_top_probability: f64,
    model_at_target_top: f64,
    top1_match: bool,
    topk_recall: [bool; POLICY_TOP_K.len()],
    target_top_rank: usize,
    selected_rank: usize,
    selected_probability: f64,
    selected_is_model_top: bool,
    selected_is_target_top: bool,
    stop_model_probability: f64,
    stop_target_probability: f64,
    stop_is_model_top: bool,
    stop_is_target_top: bool,
    stop_selected: bool,
    logit_range: f64,
    family_top_match: bool,
    target_family_model_mass: f64,
    target_family_target_mass: f64,
    target_family_size: usize,
    target_within_family_rank: usize,
    target_within_family_top1: bool,
    kind_model_mass: Vec<f64>,
    kind_target_mass: Vec<f64>,
}

fn print_policy_diagnostics(
    examples: &[ValueExample],
    outputs: &[FeatureEvaluation],
) -> Result<(), Box<dyn std::error::Error>> {
    if examples.len() != outputs.len() {
        return Err("policy diagnostic output count mismatch".into());
    }
    let kind_count = examples
        .iter()
        .flat_map(|example| {
            example
                .row
                .actions
                .iter()
                .map(|action| action.kind_token as usize)
        })
        .max()
        .map_or(0, |kind| kind + 1);
    let observations = examples
        .iter()
        .zip(outputs)
        .map(|(example, output)| policy_observation(example, output, kind_count))
        .collect::<Result<Vec<_>, _>>()?;
    let all = (0..observations.len()).collect::<Vec<_>>();
    print_policy_scope("all", &observations, &all);
    for (name, start, end) in [
        ("root_step_0", 0, 1),
        ("root_step_1_15", 1, 16),
        ("root_step_16_31", 16, 32),
        ("root_step_32_63", 32, 64),
        ("root_step_64_plus", 64, u32::MAX),
    ] {
        let indices = observations
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                (row.root_step >= start && row.root_step < end).then_some(index)
            })
            .collect::<Vec<_>>();
        print_policy_scope(name, &observations, &indices);
    }
    print_policy_confidence(&observations);
    print_policy_kinds(&observations, kind_count);
    Ok(())
}

fn policy_observation(
    example: &ValueExample,
    output: &FeatureEvaluation,
    kind_count: usize,
) -> Result<PolicyObservation, Box<dyn std::error::Error>> {
    let action_count = example.row.actions.len();
    if output.policy_logits.len() != action_count
        || example.policy_target.len() != action_count
        || example.selected_index >= action_count
    {
        return Err("policy diagnostic action count mismatch".into());
    }
    let model = policy_softmax(&output.policy_logits);
    let target_sum = example
        .policy_target
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>();
    if !target_sum.is_finite()
        || target_sum <= 0.0
        || example
            .policy_target
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("policy diagnostic has an invalid target".into());
    }
    let target = example
        .policy_target
        .iter()
        .map(|value| f64::from(*value) / target_sum)
        .collect::<Vec<_>>();
    let model_top = policy_argmax(&model);
    let target_top = policy_argmax(&target);
    let model_ranking = policy_ranking(&model);
    let target_top_rank = model_ranking
        .iter()
        .position(|index| *index == target_top)
        .expect("ranking contains every action")
        + 1;
    let selected_rank = model_ranking
        .iter()
        .position(|index| *index == example.selected_index)
        .expect("ranking contains every action")
        + 1;
    let stop_indices = example
        .row
        .actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| {
            (action.kind_token == STOP_ACTION_KIND_TOKEN).then_some(index)
        })
        .collect::<Vec<_>>();
    if stop_indices.len() != 1 {
        return Err("policy diagnostic expected exactly one STOP action".into());
    }
    let stop = stop_indices[0];
    let mut kind_model_mass = vec![0.0; kind_count];
    let mut kind_target_mass = vec![0.0; kind_count];
    for (index, action) in example.row.actions.iter().enumerate() {
        let kind = action.kind_token as usize;
        kind_model_mass[kind] += model[index];
        kind_target_mass[kind] += target[index];
    }
    let model_top_kind = policy_argmax(&kind_model_mass);
    let target_top_kind = policy_argmax(&kind_target_mass);
    let target_action_kind = example.row.actions[target_top].kind_token as usize;
    let mut target_family_actions = example
        .row
        .actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| {
            (action.kind_token as usize == target_action_kind).then_some(index)
        })
        .collect::<Vec<_>>();
    target_family_actions.sort_by(|left, right| {
        model[*right]
            .total_cmp(&model[*left])
            .then_with(|| left.cmp(right))
    });
    let target_within_family_rank = target_family_actions
        .iter()
        .position(|index| *index == target_top)
        .expect("target action belongs to its family")
        + 1;
    let cross_entropy = target
        .iter()
        .zip(&model)
        .map(|(target, model)| -target * model.max(1.0e-300).ln())
        .sum::<f64>();
    let model_entropy = policy_entropy(&model);
    let target_entropy = policy_entropy(&target);
    let min_logit = output
        .policy_logits
        .iter()
        .copied()
        .reduce(f32::min)
        .unwrap_or(0.0);
    let max_logit = output
        .policy_logits
        .iter()
        .copied()
        .reduce(f32::max)
        .unwrap_or(0.0);
    Ok(PolicyObservation {
        root_step: example.row.position.root_step,
        action_count,
        cross_entropy,
        kl: cross_entropy - target_entropy,
        model_entropy,
        target_entropy,
        model_top_probability: model[model_top],
        target_at_model_top: target[model_top],
        target_top_probability: target[target_top],
        model_at_target_top: model[target_top],
        top1_match: model_top == target_top,
        topk_recall: POLICY_TOP_K.map(|k| target_top_rank <= k),
        target_top_rank,
        selected_rank,
        selected_probability: model[example.selected_index],
        selected_is_model_top: example.selected_index == model_top,
        selected_is_target_top: example.selected_index == target_top,
        stop_model_probability: model[stop],
        stop_target_probability: target[stop],
        stop_is_model_top: stop == model_top,
        stop_is_target_top: stop == target_top,
        stop_selected: stop == example.selected_index,
        logit_range: f64::from(max_logit - min_logit),
        family_top_match: model_top_kind == target_top_kind,
        target_family_model_mass: kind_model_mass[target_action_kind],
        target_family_target_mass: kind_target_mass[target_action_kind],
        target_family_size: target_family_actions.len(),
        target_within_family_rank,
        target_within_family_top1: target_within_family_rank == 1,
        kind_model_mass,
        kind_target_mass,
    })
}

fn print_policy_scope(scope: &str, rows: &[PolicyObservation], indices: &[usize]) {
    if indices.is_empty() {
        println!("policy_metrics scope={scope} rows=0");
        return;
    }
    let count = indices.len() as f64;
    let mean = |value: fn(&PolicyObservation) -> f64| {
        indices
            .iter()
            .map(|index| value(&rows[*index]))
            .sum::<f64>()
            / count
    };
    let fraction = |value: fn(&PolicyObservation) -> bool| {
        indices.iter().filter(|index| value(&rows[**index])).count() as f64 / count
    };
    let mut recall = [0.0_f64; POLICY_TOP_K.len()];
    for (slot, _) in POLICY_TOP_K.iter().enumerate() {
        recall[slot] = indices
            .iter()
            .filter(|index| rows[**index].topk_recall[slot])
            .count() as f64
            / count;
    }
    let stop_brier = indices
        .iter()
        .map(|index| {
            let row = &rows[*index];
            (row.stop_model_probability - row.stop_target_probability).powi(2)
        })
        .sum::<f64>()
        / count;
    println!(
        "policy_metrics scope={scope} rows={} mean_actions={:.3} cross_entropy={:.6} kl={:.6} model_entropy={:.6} model_effective_actions={:.3} target_entropy={:.6} target_effective_actions={:.3} model_top_probability={:.6} target_at_model_top={:.6} target_top_probability={:.6} model_at_target_top={:.6} top1_match={:.6} top5_recall={:.6} top8_recall={:.6} top16_recall={:.6} top32_recall={:.6} mean_target_top_rank={:.3} selected_model_probability={:.6} selected_top1={:.6} selected_is_target_top={:.6} mean_selected_rank={:.3} stop_model_probability={:.6} stop_target_probability={:.6} stop_model_top1={:.6} stop_target_top1={:.6} stop_selected={:.6} stop_brier={stop_brier:.6} family_top1_match={:.6} target_family_model_mass={:.6} target_family_target_mass={:.6} target_family_size={:.3} target_within_family_top1={:.6} mean_target_within_family_rank={:.3} mean_logit_range={:.6}",
        indices.len(),
        mean(|row| row.action_count as f64),
        mean(|row| row.cross_entropy),
        mean(|row| row.kl),
        mean(|row| row.model_entropy),
        mean(|row| row.model_entropy.exp()),
        mean(|row| row.target_entropy),
        mean(|row| row.target_entropy.exp()),
        mean(|row| row.model_top_probability),
        mean(|row| row.target_at_model_top),
        mean(|row| row.target_top_probability),
        mean(|row| row.model_at_target_top),
        fraction(|row| row.top1_match),
        recall[1],
        recall[2],
        recall[3],
        recall[4],
        mean(|row| row.target_top_rank as f64),
        mean(|row| row.selected_probability),
        fraction(|row| row.selected_is_model_top),
        fraction(|row| row.selected_is_target_top),
        mean(|row| row.selected_rank as f64),
        mean(|row| row.stop_model_probability),
        mean(|row| row.stop_target_probability),
        fraction(|row| row.stop_is_model_top),
        fraction(|row| row.stop_is_target_top),
        fraction(|row| row.stop_selected),
        fraction(|row| row.family_top_match),
        mean(|row| row.target_family_model_mass),
        mean(|row| row.target_family_target_mass),
        mean(|row| row.target_family_size as f64),
        fraction(|row| row.target_within_family_top1),
        mean(|row| row.target_within_family_rank as f64),
        mean(|row| row.logit_range),
    );
}

fn print_policy_confidence(rows: &[PolicyObservation]) {
    let mut count = [0_usize; 10];
    let mut confidence = [0.0_f64; 10];
    let mut target_mass = [0.0_f64; 10];
    let mut top_match = [0_usize; 10];
    for row in rows {
        let bin = ((row.model_top_probability * 10.0) as usize).min(9);
        count[bin] += 1;
        confidence[bin] += row.model_top_probability;
        target_mass[bin] += row.target_at_model_top;
        top_match[bin] += usize::from(row.top1_match);
    }
    for bin in 0..10 {
        if count[bin] == 0 {
            continue;
        }
        println!(
            "policy_confidence bin={bin} lower={:.1} upper={:.1} rows={} model_confidence={:.6} target_mass={:.6} target_top1_match={:.6}",
            bin as f64 / 10.0,
            (bin + 1) as f64 / 10.0,
            count[bin],
            confidence[bin] / count[bin] as f64,
            target_mass[bin] / count[bin] as f64,
            top_match[bin] as f64 / count[bin] as f64,
        );
    }
}

fn print_policy_kinds(rows: &[PolicyObservation], kind_count: usize) {
    let mut model = vec![0.0_f64; kind_count];
    let mut target = vec![0.0_f64; kind_count];
    for row in rows {
        for kind in 0..kind_count {
            model[kind] += row.kind_model_mass[kind];
            target[kind] += row.kind_target_mass[kind];
        }
    }
    let count = rows.len().max(1) as f64;
    let mut ranking = (0..kind_count).collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        (model[*right] + target[*right])
            .total_cmp(&(model[*left] + target[*left]))
            .then_with(|| left.cmp(right))
    });
    for kind in ranking.into_iter().take(12) {
        if model[kind] == 0.0 && target[kind] == 0.0 {
            continue;
        }
        println!(
            "policy_kind kind={} name={} model_mass={:.6} target_mass={:.6} delta={:+.6}",
            kind,
            policy_kind_name(kind),
            model[kind] / count,
            target[kind] / count,
            (model[kind] - target[kind]) / count,
        );
    }
}

fn policy_kind_name(kind: usize) -> String {
    if kind == STOP_ACTION_KIND_TOKEN as usize {
        return "STOP".to_owned();
    }
    if kind >= 2 {
        return rule_name((kind - 2) as u16).to_owned();
    }
    "PADDING".to_owned()
}

fn policy_softmax(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().reduce(f32::max).unwrap_or(0.0);
    let mut probabilities = logits
        .iter()
        .map(|logit| f64::from(*logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

fn policy_entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn policy_argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map_or(0, |(index, _)| index)
}

fn policy_ranking(values: &[f64]) -> Vec<usize> {
    let mut ranking = (0..values.len()).collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    ranking
}

fn mean_by_episode(
    rows: &[ValueObservation],
) -> Result<Vec<ValueObservation>, Box<dyn std::error::Error>> {
    let mut grouped = BTreeMap::<u64, Vec<&ValueObservation>>::new();
    for row in rows {
        grouped.entry(row.episode_id).or_default().push(row);
    }
    grouped
        .into_iter()
        .map(|(episode_id, rows)| {
            let target = rows[0].target;
            if rows.iter().any(|row| row.target != target) {
                return Err(format!("episode {episode_id} has inconsistent value targets").into());
            }
            Ok(ValueObservation {
                episode_id,
                root_step: 0,
                target,
                prediction: rows.iter().map(|row| row.prediction).sum::<f32>() / rows.len() as f32,
            })
        })
        .collect()
}

fn print_paired_episode_discrimination(
    rows: &[ValueObservation],
) -> Result<(), Box<dyn std::error::Error>> {
    let by_episode = rows
        .iter()
        .map(|row| (row.episode_id, row))
        .collect::<BTreeMap<_, _>>();
    let mut correct = 0_usize;
    let mut tied = 0_usize;
    let mut total = 0_usize;
    let mut margin_sum = 0.0_f64;
    for game in 0..rows.len() / 2 {
        let p1_id = (game * 2) as u64;
        let p2_id = p1_id + 1;
        let p1 = by_episode
            .get(&p1_id)
            .ok_or_else(|| format!("missing episode mean for P1 game {game}"))?;
        let p2 = by_episode
            .get(&p2_id)
            .ok_or_else(|| format!("missing episode mean for P2 game {game}"))?;
        if p2.target != -p1.target {
            return Err(format!("game {game} has inconsistent paired targets").into());
        }
        if p1.target == 0.0 {
            continue;
        }
        let margin = if p1.target > 0.0 {
            p1.prediction - p2.prediction
        } else {
            p2.prediction - p1.prediction
        };
        total += 1;
        correct += usize::from(margin > 0.0);
        tied += usize::from(margin == 0.0);
        margin_sum += f64::from(margin);
    }
    println!(
        "value_pair_discrimination games={total} correct={correct} tied={tied} accuracy={:.6} mean_winner_minus_loser={:.6}",
        correct as f64 / total.max(1) as f64,
        margin_sum / total.max(1) as f64,
    );
    Ok(())
}

fn print_value_metric(scope: &str, rows: &[ValueObservation]) {
    if rows.is_empty() {
        println!("value_metrics scope={scope} rows=0");
        return;
    }
    let summary = value_metric_summary(rows);
    println!(
        "value_metrics scope={scope} rows={} positive={} negative={} neutral={} accuracy={:.6} balanced_accuracy={:.6} positive_accuracy={:.6} negative_accuracy={:.6} auc={:.6} mse={:.6} brier={:.6} ece10={:.6} mean_margin={:.6} prediction_mean={:.6} prediction_std={:.6} positive_prediction_mean={:.6} negative_prediction_mean={:.6} q05={:.6} q50={:.6} q95={:.6} saturated_0p9={:.6} saturated_0p99={:.6}",
        summary.rows,
        summary.positive,
        summary.negative,
        summary.neutral,
        summary.accuracy,
        summary.balanced_accuracy,
        summary.positive_accuracy,
        summary.negative_accuracy,
        summary.auc,
        summary.mse,
        summary.brier,
        summary.ece,
        summary.mean_margin,
        summary.prediction_mean,
        summary.prediction_std,
        summary.positive_prediction_mean,
        summary.negative_prediction_mean,
        summary.q05,
        summary.q50,
        summary.q95,
        summary.saturated_0p9,
        summary.saturated_0p99,
    );
}

fn value_metric_summary(rows: &[ValueObservation]) -> ValueMetricSummary {
    let count = rows.len() as f64;
    let positive = rows.iter().filter(|row| row.target > 0.0).count();
    let negative = rows.iter().filter(|row| row.target < 0.0).count();
    let neutral = rows.len() - positive - negative;
    let classified = positive + negative;
    let positive_correct = rows
        .iter()
        .filter(|row| row.target > 0.0 && row.prediction >= 0.0)
        .count();
    let negative_correct = rows
        .iter()
        .filter(|row| row.target < 0.0 && row.prediction < 0.0)
        .count();
    let positive_accuracy = positive_correct as f64 / positive.max(1) as f64;
    let negative_accuracy = negative_correct as f64 / negative.max(1) as f64;
    let accuracy = (positive_correct + negative_correct) as f64 / classified.max(1) as f64;
    let prediction_mean = rows
        .iter()
        .map(|row| f64::from(row.prediction))
        .sum::<f64>()
        / count;
    let prediction_variance = rows
        .iter()
        .map(|row| {
            let delta = f64::from(row.prediction) - prediction_mean;
            delta * delta
        })
        .sum::<f64>()
        / count;
    let mse = rows
        .iter()
        .map(|row| {
            let error = f64::from(row.prediction - row.target);
            error * error
        })
        .sum::<f64>()
        / count;
    let mean_margin = rows
        .iter()
        .map(|row| f64::from(row.target * row.prediction))
        .sum::<f64>()
        / count;
    let positive_prediction_mean = rows
        .iter()
        .filter(|row| row.target > 0.0)
        .map(|row| f64::from(row.prediction))
        .sum::<f64>()
        / positive.max(1) as f64;
    let negative_prediction_mean = rows
        .iter()
        .filter(|row| row.target < 0.0)
        .map(|row| f64::from(row.prediction))
        .sum::<f64>()
        / negative.max(1) as f64;
    let mut predictions = rows.iter().map(|row| row.prediction).collect::<Vec<_>>();
    predictions.sort_by(f32::total_cmp);
    let saturated_0p9 = rows
        .iter()
        .filter(|row| row.prediction.abs() >= 0.9)
        .count() as f64
        / count;
    let saturated_0p99 = rows
        .iter()
        .filter(|row| row.prediction.abs() >= 0.99)
        .count() as f64
        / count;
    ValueMetricSummary {
        rows: rows.len(),
        positive,
        negative,
        neutral,
        accuracy,
        balanced_accuracy: if positive == 0 || negative == 0 {
            f64::NAN
        } else {
            0.5 * (positive_accuracy + negative_accuracy)
        },
        positive_accuracy,
        negative_accuracy,
        auc: value_auc(rows),
        mse,
        brier: 0.25 * mse,
        ece: value_ece(rows),
        mean_margin,
        prediction_mean,
        prediction_std: prediction_variance.sqrt(),
        positive_prediction_mean,
        negative_prediction_mean,
        q05: value_quantile(&predictions, 0.05),
        q50: value_quantile(&predictions, 0.50),
        q95: value_quantile(&predictions, 0.95),
        saturated_0p9,
        saturated_0p99,
    }
}

fn value_auc(rows: &[ValueObservation]) -> f64 {
    let mut ranked = rows
        .iter()
        .filter(|row| row.target != 0.0)
        .map(|row| (row.prediction, row.target > 0.0))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.0.total_cmp(&right.0));
    let positives = ranked.iter().filter(|(_, positive)| *positive).count();
    let negatives = ranked.len() - positives;
    if positives == 0 || negatives == 0 {
        return f64::NAN;
    }
    let mut positive_rank_sum = 0.0;
    let mut start = 0;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len() && ranked[end].0.total_cmp(&ranked[start].0).is_eq() {
            end += 1;
        }
        let average_rank = (start + 1 + end) as f64 * 0.5;
        let group_positives = ranked[start..end]
            .iter()
            .filter(|(_, positive)| *positive)
            .count();
        positive_rank_sum += average_rank * group_positives as f64;
        start = end;
    }
    (positive_rank_sum - (positives * (positives + 1) / 2) as f64) / (positives * negatives) as f64
}

fn value_ece(rows: &[ValueObservation]) -> f64 {
    let mut counts = [0_usize; 10];
    let mut confidence = [0.0_f64; 10];
    let mut outcomes = [0.0_f64; 10];
    let classified = rows.iter().filter(|row| row.target != 0.0).count();
    if classified == 0 {
        return f64::NAN;
    }
    for row in rows.iter().filter(|row| row.target != 0.0) {
        let probability = (0.5 * (f64::from(row.prediction) + 1.0)).clamp(0.0, 1.0);
        let bin = ((probability * 10.0) as usize).min(9);
        counts[bin] += 1;
        confidence[bin] += probability;
        outcomes[bin] += if row.target > 0.0 { 1.0 } else { 0.0 };
    }
    counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(bin, count)| {
            let count = *count as f64;
            count / classified as f64 * (confidence[bin] / count - outcomes[bin] / count).abs()
        })
        .sum()
}

fn value_quantile(sorted: &[f32], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    f64::from(sorted[index])
}

fn print_calibration(rows: &[ValueObservation]) {
    let mut counts = [0_usize; 10];
    let mut confidence = [0.0_f64; 10];
    let mut outcomes = [0.0_f64; 10];
    for row in rows.iter().filter(|row| row.target != 0.0) {
        let probability = (0.5 * (f64::from(row.prediction) + 1.0)).clamp(0.0, 1.0);
        let bin = ((probability * 10.0) as usize).min(9);
        counts[bin] += 1;
        confidence[bin] += probability;
        outcomes[bin] += if row.target > 0.0 { 1.0 } else { 0.0 };
    }
    for bin in 0..10 {
        if counts[bin] == 0 {
            continue;
        }
        println!(
            "value_calibration bin={} lower={:.1} upper={:.1} rows={} predicted_positive={:.6} observed_positive={:.6}",
            bin,
            bin as f64 / 10.0,
            (bin + 1) as f64 / 10.0,
            counts[bin],
            confidence[bin] / counts[bin] as f64,
            outcomes[bin] / counts[bin] as f64,
        );
    }
}

fn print_antisymmetry(original: &[ValueObservation], swapped: &[ValueObservation]) {
    let mut residuals = original
        .iter()
        .zip(swapped)
        .map(|(left, right)| (left.prediction + right.prediction).abs())
        .collect::<Vec<_>>();
    residuals.sort_by(f32::total_cmp);
    let opposite_sign = original
        .iter()
        .zip(swapped)
        .filter(|(left, right)| (left.prediction >= 0.0) != (right.prediction >= 0.0))
        .count();
    let mean =
        residuals.iter().map(|value| f64::from(*value)).sum::<f64>() / residuals.len() as f64;
    println!(
        "value_antisymmetry rows={} mean_abs_sum={mean:.6} p95_abs_sum={:.6} max_abs_sum={:.6} opposite_sign_fraction={:.6}",
        residuals.len(),
        value_quantile(&residuals, 0.95),
        residuals.last().copied().map_or(0.0, f64::from),
        opposite_sign as f64 / residuals.len() as f64,
    );
}

fn generated_policy_rollouts(
    config: &DiagnosticConfig,
) -> Result<Vec<PolicyRollout>, Box<dyn std::error::Error>> {
    let mut rollouts = Vec::with_capacity(config.cases);
    for index in 0..config.cases {
        let (mut engine, graph) = generated_root(config.arena_seed, index)?;
        let root_cost = measure_cost(&mut engine, graph)?;
        let root_hash = engine.hash(graph)?;
        let extractor = feature_extractor(&engine);
        rollouts.push(PolicyRollout {
            index,
            engine,
            extractor,
            graph,
            root_hash,
            root_cost,
            steps: 0,
            stopped: false,
        });
    }
    Ok(rollouts)
}

fn generated_mcts_roots(
    config: &DiagnosticConfig,
) -> Result<GeneratedMctsRoots, Box<dyn std::error::Error>> {
    let mut engines = Vec::with_capacity(config.cases);
    let mut extractors = Vec::with_capacity(config.cases);
    let mut roots = Vec::with_capacity(config.cases);
    let mut info = Vec::with_capacity(config.cases);
    for index in 0..config.cases {
        let (mut engine, graph) = generated_root(config.arena_seed, index)?;
        info.push(RootInfo {
            index,
            root_hash: engine.hash(graph)?,
            root_cost: measure_cost(&mut engine, graph)?,
        });
        extractors.push(feature_extractor(&engine));
        roots.push(OneRoot { root: Some(graph) });
        engines.push(engine);
    }
    Ok(GeneratedMctsRoots {
        engines,
        extractors,
        roots,
        info,
    })
}

fn generated_root(
    arena_seed: u64,
    index: usize,
) -> Result<(WhittleEngine, WhittleGraphId), Box<dyn std::error::Error>> {
    let generator_config = WhittleGraphGeneratorConfig::default();
    let mut engine = WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator_config.arity,
            capacity: generator_config.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    })?;
    let mut generator =
        WhittleGraphGenerator::from_seed(generator_config, arena_graph_seed(arena_seed, index));
    let graph = generator.sample_root_into(&mut engine)?;
    Ok((engine, graph))
}

fn feature_extractor(engine: &WhittleEngine) -> WhittleFeatureExtractor {
    WhittleFeatureExtractor::with_config(
        engine,
        WhittleFeatureExtractorConfig {
            max_actions: (MAX_CANDIDATES + 1) as u32,
            ..WhittleFeatureExtractorConfig::default()
        },
    )
}

// Must remain bit-identical to the generated gated-policy arena roots.
fn arena_graph_seed(seed: u64, index: usize) -> u64 {
    let mut value =
        seed ^ 0x6172_656e_615f_6772 ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn spawn_evaluator(
    role: &str,
    config: &DiagnosticConfig,
    pointer: &str,
) -> Result<EvaluatorProcess, Box<dyn std::error::Error>> {
    spawn_evaluator_at(role, config, &config.checkpoint_dir, pointer)
}

fn spawn_evaluator_at(
    role: &str,
    config: &DiagnosticConfig,
    checkpoint_dir: &Path,
    pointer: &str,
) -> Result<EvaluatorProcess, Box<dyn std::error::Error>> {
    let socket_path = std::env::temp_dir().join(format!(
        "gz-checkpoint-diagnostic-{role}-{}.sock",
        std::process::id()
    ));
    let mut extra_args = vec![
        "--backend".to_owned(),
        "torch".to_owned(),
        "--checkpoint-dir".to_owned(),
        checkpoint_dir.display().to_string(),
        "--checkpoint-pointer".to_owned(),
        pointer.to_owned(),
        "--device".to_owned(),
        config.device.clone(),
        "--max-batch".to_owned(),
        MAX_BATCH.to_string(),
        "--poll-interval".to_owned(),
        "0".to_owned(),
    ];
    if config.mode.symmetric_value() {
        extra_args.extend([
            "--require-state-input".to_owned(),
            "joint-board".to_owned(),
            "--require-value-input".to_owned(),
            "single".to_owned(),
        ]);
    }
    Ok(EvaluatorProcess::spawn(EvaluatorProcessConfig {
        working_dir: PathBuf::from("python"),
        socket_path,
        ready_timeout: Duration::from_secs(60),
        io_timeout: Duration::from_secs(600),
        extra_args,
        ..EvaluatorProcessConfig::default()
    })?)
}

fn print_results(
    config: &DiagnosticConfig,
    current_version: gz_engine::ModelVersion,
    opponent_version: gz_engine::ModelVersion,
    results: &[MatchResult],
) {
    let count = results.len() as f32;
    let mean_root = results.iter().map(|row| row.root_cost).sum::<f32>() / count;
    let mean_current = results.iter().map(|row| row.current_cost).sum::<f32>() / count;
    let mean_opponent = results.iter().map(|row| row.opponent_cost).sum::<f32>() / count;
    let mean_current_steps =
        results.iter().map(|row| row.current_steps).sum::<usize>() as f32 / count;
    let mean_opponent_steps =
        results.iter().map(|row| row.opponent_steps).sum::<usize>() as f32 / count;
    let cost = result_counts(results, false);
    let tiebreak = result_counts(results, true);
    let score = 100.0 * (tiebreak.wins as f32 + 0.5 * tiebreak.ties as f32) / count;
    let current_stopped = results.iter().filter(|row| row.current_stopped).count();
    let opponent_stopped = results.iter().filter(|row| row.opponent_stopped).count();
    let (gumbel_scale, overlap) = config.mode.noise();
    let current_no_backtrack = !matches!(config.mode, Mode::Greedy);

    println!(
        "settings mode={} cases={} arena_seed={} max_steps={} considered={} simulations={} gumbel_scale={} overlap={} c_scale={} c_visit={} tree_reuse=false current_no_backtrack={} opponent_no_backtrack=false stop_allowed=true length_tiebreak=true",
        config.mode.name(),
        results.len(),
        config.arena_seed,
        config.max_steps,
        config.considered,
        config.simulations,
        gumbel_scale,
        if overlap < 0.0 {
            "disabled".to_owned()
        } else {
            overlap.to_string()
        },
        config.c_scale,
        config.c_visit,
        current_no_backtrack,
    );
    println!(
        "models current_pointer={} current_version={} opponent_pointer={} opponent_version={}",
        config.current_pointer, current_version, config.opponent_pointer, opponent_version,
    );
    println!(
        "summary score_percent={score:.6} tiebreak_wins={} tiebreak_losses={} tiebreak_ties={} cost_wins={} cost_losses={} cost_ties={} mean_root={mean_root:.6} mean_current={mean_current:.6} mean_opponent={mean_opponent:.6} mean_cost_advantage={:.6} mean_current_reduction={:.6} mean_opponent_reduction={:.6} mean_current_steps={mean_current_steps:.6} mean_opponent_steps={mean_opponent_steps:.6} current_stopped={current_stopped}/{} opponent_stopped={opponent_stopped}/{}",
        tiebreak.wins,
        tiebreak.losses,
        tiebreak.ties,
        cost.wins,
        cost.losses,
        cost.ties,
        mean_opponent - mean_current,
        mean_root - mean_current,
        mean_root - mean_opponent,
        results.len(),
        results.len(),
    );
    println!("index root current opponent current_steps opponent_steps result root_hash");
    for result in results {
        println!(
            "{} {:.0} {:.0} {:.0} {} {} {} {}",
            result.index,
            result.root_cost,
            result.current_cost,
            result.opponent_cost,
            result.current_steps,
            result.opponent_steps,
            comparison(result, true).name(),
            result.root_hash,
        );
    }
}

fn result_counts(results: &[MatchResult], length_tiebreak: bool) -> ResultCounts {
    let mut counts = ResultCounts::default();
    for result in results {
        match comparison(result, length_tiebreak) {
            Comparison::Win => counts.wins += 1,
            Comparison::Loss => counts.losses += 1,
            Comparison::Tie => counts.ties += 1,
        }
    }
    counts
}

fn comparison(result: &MatchResult, length_tiebreak: bool) -> Comparison {
    if result.current_cost < result.opponent_cost {
        return Comparison::Win;
    }
    if result.current_cost > result.opponent_cost {
        return Comparison::Loss;
    }
    if length_tiebreak {
        if result.current_steps < result.opponent_steps {
            return Comparison::Win;
        }
        if result.current_steps > result.opponent_steps {
            return Comparison::Loss;
        }
    }
    Comparison::Tie
}

#[derive(Clone, Copy)]
enum Comparison {
    Win,
    Loss,
    Tie,
}

impl Comparison {
    const fn name(self) -> &'static str {
        match self {
            Self::Win => "win",
            Self::Loss => "loss",
            Self::Tie => "tie",
        }
    }
}

#[derive(Default)]
struct ResultCounts {
    wins: usize,
    losses: usize,
    ties: usize,
}

#[derive(Clone, Copy)]
struct RootInfo {
    index: usize,
    root_hash: GraphHash,
    root_cost: f32,
}

struct GeneratedMctsRoots {
    engines: Vec<WhittleEngine>,
    extractors: Vec<WhittleFeatureExtractor>,
    roots: Vec<OneRoot>,
    info: Vec<RootInfo>,
}

struct PolicyRollout {
    index: usize,
    engine: WhittleEngine,
    extractor: WhittleFeatureExtractor,
    graph: WhittleGraphId,
    root_hash: GraphHash,
    root_cost: f32,
    steps: usize,
    stopped: bool,
}

struct GreedyRun {
    model_version: gz_engine::ModelVersion,
    rows: Vec<GreedyResult>,
}

struct GreedyResult {
    index: usize,
    root_hash: GraphHash,
    root_cost: f32,
    final_cost: f32,
    steps: usize,
    stopped: bool,
}

struct MatchResult {
    index: usize,
    root_hash: GraphHash,
    root_cost: f32,
    current_cost: f32,
    opponent_cost: f32,
    current_steps: usize,
    opponent_steps: usize,
    current_stopped: bool,
    opponent_stopped: bool,
}

struct ValueExample {
    episode_id: u64,
    target: f32,
    policy_target: Vec<f32>,
    selected_index: usize,
    row: FeatureRow,
}

#[derive(Clone, Copy)]
struct ValueObservation {
    episode_id: u64,
    root_step: u32,
    target: f32,
    prediction: f32,
}

struct ValueMetricSummary {
    rows: usize,
    positive: usize,
    negative: usize,
    neutral: usize,
    accuracy: f64,
    balanced_accuracy: f64,
    positive_accuracy: f64,
    negative_accuracy: f64,
    auc: f64,
    mse: f64,
    brier: f64,
    ece: f64,
    mean_margin: f64,
    prediction_mean: f64,
    prediction_std: f64,
    positive_prediction_mean: f64,
    negative_prediction_mean: f64,
    q05: f64,
    q50: f64,
    q95: f64,
    saturated_0p9: f64,
    saturated_0p99: f64,
}

struct OneRoot {
    root: Option<WhittleGraphId>,
}

impl RootSource<WhittleEngine> for OneRoot {
    fn next_root(&mut self, _engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        Ok(self.root.take())
    }

    fn episode_roots_are_owned(&self) -> bool {
        true
    }
}

struct EvalSampledTreeProvider;

impl ReferenceProvider<WhittleEngine> for EvalSampledTreeProvider {
    fn reference(
        &mut self,
        _engine: &mut WhittleEngine,
        _root: WhittleGraphId,
    ) -> EngineResult<Option<Reference>> {
        Ok(None)
    }

    fn sampled_tree_mode(&self) -> bool {
        true
    }
}

struct EvalNoReferenceProvider;

impl ReferenceProvider<WhittleEngine> for EvalNoReferenceProvider {
    fn reference(
        &mut self,
        _engine: &mut WhittleEngine,
        _root: WhittleGraphId,
    ) -> EngineResult<Option<Reference>> {
        Ok(None)
    }
}

fn measure_cost(
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
) -> Result<f32, Box<dyn std::error::Error>> {
    Ok(-engine
        .measure(graph, engine.measure_options())?
        .scalar_reward
        .ok_or("graph was not measured")?)
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        match process.try_wait()? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("Python evaluator exited with {status}").into()),
            None if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            None => return Err("Python evaluator did not exit".into()),
        }
    }
}

fn temporary_replay_dir(mode: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "gz-checkpoint-diagnostic-{mode}-{}-{nonce}",
        std::process::id()
    )))
}

fn nonzero(value: usize) -> Result<NonZeroUsize, Box<dyn std::error::Error>> {
    NonZeroUsize::new(value).ok_or_else(|| "value must be nonzero".into())
}

fn optional_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("{value:.6}"))
}

fn absolute(path: &Path) -> Result<PathBuf, std::io::Error> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn arg_or(index: usize, default: &str) -> String {
    std::env::args()
        .nth(index)
        .unwrap_or_else(|| default.to_owned())
}

fn parse_arg_or<T>(index: usize, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    std::env::args()
        .nth(index)
        .map(|value| value.parse::<T>().map_err(Into::into))
        .unwrap_or(Ok(default))
}
