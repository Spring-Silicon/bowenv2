use gz_engine::{
    CandidateOptions, EngineResult, GraphEngine, ModelVersion, PortableSearchActionRef,
    ReplayGraphContext,
};
use gz_engine_whittle::{
    WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor,
    WhittleFeatureExtractorConfig, WhittleGraphId, WhittleRoot,
};
use gz_eval::EvalOutput;
use gz_eval_service::{EvaluatorProcess, EvaluatorProcessConfig, FeatureEvalBackend, Hello};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, FeatureSchema, OpponentStateFeatures,
    PositionFeatures,
};
use gz_search::{
    EngineIdentity, EvalWork, ExpandResult, ExpandedCandidate, GumbelEpisodeContext, GumbelMcts,
    GumbelMctsConfig, GumbelPlayer, GumbelValueMode, SearchPoll, SearchWork, SearchWorkResult,
    SymmetricEpisode, SymmetricSelfplayEpisodeTask, WorkToken,
};
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MAX_CANDIDATES: usize = 1023;
const MAX_STEPS: usize = 96;
const MAX_BATCH: usize = 128;
const DRIVER_THREADS: usize = 32;
const SEARCH_SEED: u64 = 42;
const GUMBEL_SCALE: f32 = 1.0;
const GUMBEL_NOISE_OVERLAP: f32 = 0.5;
const C_VISIT: f32 = 50.0;
const C_SCALE: f32 = 1.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_args()?;
    let cases = load_cases(&config.artifact_dir, config.case_limit)?;
    let probe = new_engine(&cases[0])?;
    let schema = feature_extractor(&probe).schema().clone();
    let hello = Hello::new(
        schema.hash(),
        MAX_BATCH as u32,
        probe.engine_id(),
        probe.engine_version(),
        probe.action_set_hash(),
    );
    let search = search(&probe, config.simulations, config.considered);
    drop(probe);

    let mut process = spawn_evaluator(&config)?;
    let mut backend = process.connect(&hello)?;
    let model_version = backend.model_version();
    let mut batch_stats = BatchStats::default();

    println!(
        "settings cases={} continuations={} depths={} max_steps={MAX_STEPS} considered={} simulations={} gumbel_scale={GUMBEL_SCALE} overlap={GUMBEL_NOISE_OVERLAP} c_scale={C_SCALE} tree_reuse=true wave_batching=true intermediate_tree_state=fresh search_seed={SEARCH_SEED} noise_seed={}",
        cases.len(),
        config.continuations,
        config
            .depths
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
        config.considered,
        config.simulations,
        config.noise_seed,
    );
    println!(
        "model checkpoint={} pointer={} model_version={model_version}",
        config.checkpoint_dir.display(),
        config.checkpoint_pointer,
    );

    let baseline_started = Instant::now();
    let baseline_trials = cases
        .iter()
        .map(|case| {
            Trial::new(
                case,
                case.index,
                0,
                &Prefix::default(),
                &search,
                splitmix64(config.noise_seed ^ case.index as u64 ^ 0x6261_7365_6c69_6e65),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut baselines = run_trials(
        baseline_trials,
        &mut backend,
        &schema,
        model_version,
        &mut batch_stats,
    )?;
    baselines.sort_by_key(|result| result.case_index);
    let baseline_elapsed = baseline_started.elapsed();
    eprintln!(
        "phase=baseline completed={} elapsed_seconds={:.3}",
        baselines.len(),
        baseline_elapsed.as_secs_f64()
    );

    let states = capture_states(&cases, &baselines, &config.depths)?;
    if states.is_empty() {
        return Err("no active symmetric states were captured at the requested depths".into());
    }
    let state_rows = states
        .iter()
        .map(|state| build_state_row(&cases[state.case_offset], &state.prefix))
        .collect::<Result<Vec<_>, _>>()?;
    let predictions = evaluate_rows(
        &mut backend,
        &schema,
        &state_rows,
        model_version,
        &mut batch_stats,
    )?;
    if predictions.len() != states.len() {
        return Err("state prediction count mismatch".into());
    }

    let total_trials = states
        .len()
        .checked_mul(config.continuations)
        .ok_or("continuation trial count overflow")?;
    let mut outcomes = vec![Vec::with_capacity(config.continuations); states.len()];
    let specs = (0..states.len())
        .flat_map(|state| (0..config.continuations).map(move |replicate| (state, replicate)))
        .collect::<Vec<_>>();
    let continuations_started = Instant::now();
    let mut completed = 0;
    for chunk in specs.chunks(MAX_BATCH) {
        let trials = chunk
            .iter()
            .map(|&(state_index, replicate)| {
                let state = &states[state_index];
                Trial::new(
                    &cases[state.case_offset],
                    state.id,
                    replicate,
                    &state.prefix,
                    &search,
                    continuation_seed(config.noise_seed, state.id, replicate),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for result in run_trials(
            trials,
            &mut backend,
            &schema,
            model_version,
            &mut batch_stats,
        )? {
            outcomes[result.state_id].push(result.outcome);
        }
        completed += chunk.len();
        eprintln!(
            "phase=continuations completed={completed}/{total_trials} elapsed_seconds={:.3}",
            continuations_started.elapsed().as_secs_f64()
        );
    }

    let estimates = states
        .iter()
        .zip(predictions)
        .map(|(state, prediction)| {
            StateEstimate::new(state, prediction, &outcomes[state.id], config.continuations)
        })
        .collect::<Result<Vec<_>, _>>()?;
    print_estimates(&estimates);
    print_summary("all", &estimates);
    print_summary(
        "root_step_0",
        &estimates
            .iter()
            .filter(|row| row.depth == 0)
            .cloned()
            .collect::<Vec<_>>(),
    );
    print_summary(
        "root_step_1_7",
        &estimates
            .iter()
            .filter(|row| (1..=7).contains(&row.depth))
            .cloned()
            .collect::<Vec<_>>(),
    );
    print_summary(
        "root_step_8_15",
        &estimates
            .iter()
            .filter(|row| (8..=15).contains(&row.depth))
            .cloned()
            .collect::<Vec<_>>(),
    );
    println!(
        "work captured_states={} continuation_trials={} eval_rows={} eval_batches={} mean_eval_batch={:.3} baseline_seconds={:.3} continuation_seconds={:.3}",
        states.len(),
        total_trials,
        batch_stats.rows,
        batch_stats.batches,
        batch_stats.mean_batch(),
        baseline_elapsed.as_secs_f64(),
        continuations_started.elapsed().as_secs_f64(),
    );

    drop(backend);
    wait_for_process_exit(&mut process)?;
    Ok(())
}

struct Config {
    artifact_dir: PathBuf,
    checkpoint_dir: PathBuf,
    device: String,
    checkpoint_pointer: String,
    continuations: usize,
    case_limit: usize,
    depths: Vec<usize>,
    simulations: usize,
    considered: usize,
    noise_seed: u64,
}

impl Config {
    fn from_args() -> Result<Self, Box<dyn std::error::Error>> {
        let usage = "usage: diagnose_symmetric_value_expectation ARTIFACT_DIR CHECKPOINT_DIR [DEVICE] [CHECKPOINT_POINTER] [CONTINUATIONS] [CASE_LIMIT] [DEPTHS] [SIMULATIONS] [CONSIDERED] [NOISE_SEED]";
        let artifact_dir = absolute(&PathBuf::from(std::env::args().nth(1).ok_or(usage)?))?;
        let checkpoint_dir = absolute(&PathBuf::from(std::env::args().nth(2).ok_or(usage)?))?;
        let depths = parse_depths(&arg_or(7, "0,3,6,9,12"))?;
        let config = Self {
            artifact_dir,
            checkpoint_dir,
            device: arg_or(3, "cuda:0"),
            checkpoint_pointer: arg_or(4, "latest.json"),
            continuations: parse_arg_or(5, 32)?,
            case_limit: parse_arg_or(6, 20)?,
            depths,
            simulations: parse_arg_or(8, 48)?,
            considered: parse_arg_or(9, 8)?,
            noise_seed: parse_arg_or(10, 920_000_001_u64)?,
        };
        if config.continuations < 2 {
            return Err("CONTINUATIONS must be at least two".into());
        }
        if config.case_limit == 0
            || config.simulations == 0
            || config.considered == 0
            || config.depths.is_empty()
        {
            return Err("CASE_LIMIT, DEPTHS, SIMULATIONS, and CONSIDERED must be nonzero".into());
        }
        if config.depths.iter().any(|depth| *depth >= MAX_STEPS) {
            return Err(format!("DEPTHS must be smaller than {MAX_STEPS}").into());
        }
        Ok(config)
    }
}

#[derive(Clone)]
struct EvalCase {
    index: usize,
    bytes: Vec<u8>,
}

#[derive(Clone, Default)]
struct Prefix {
    actions: [Vec<PortableSearchActionRef>; 2],
}

impl Prefix {
    fn rewrites(&self) -> [usize; 2] {
        [
            rewrite_count(&self.actions[0]),
            rewrite_count(&self.actions[1]),
        ]
    }
}

struct CapturedState {
    id: usize,
    case_offset: usize,
    case_index: usize,
    depth: usize,
    prefix: Prefix,
}

struct Trial {
    case_index: usize,
    state_id: usize,
    replicate: usize,
    prefix: Prefix,
    prefix_rewrites: [usize; 2],
    engine: WhittleEngine,
    extractor: WhittleFeatureExtractor,
    initial_graphs: Vec<WhittleGraphId>,
    task: Option<SymmetricSelfplayEpisodeTask<WhittleGraphId, WhittleCandidateId>>,
}

impl Trial {
    fn new(
        case: &EvalCase,
        state_id: usize,
        replicate: usize,
        prefix: &Prefix,
        search: &GumbelMcts,
        noise_seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engine = new_engine(case)?;
        let identity = EngineIdentity::from_engine(&engine);
        let (actors, initial_graphs) = reconstruct_pair(&mut engine, identity, prefix)?;
        let prefix_rewrites = prefix.rewrites();
        let task = SymmetricSelfplayEpisodeTask::from_position(
            search,
            identity,
            [actors[0].graph, actors[1].graph],
            [actors[0].context, actors[1].context],
            prefix_rewrites,
            [false, false],
            [false, false],
            GumbelPlayer::One,
            [actors[0].visited.clone(), actors[1].visited.clone()],
            GumbelEpisodeContext {
                noise_seed,
                opponent: None,
            },
        );
        let extractor = feature_extractor(&engine);
        Ok(Self {
            case_index: case.index,
            state_id,
            replicate,
            prefix: prefix.clone(),
            prefix_rewrites,
            engine,
            extractor,
            initial_graphs,
            task: Some(task),
        })
    }
}

struct ReconstructedActor {
    graph: WhittleGraphId,
    context: Option<ReplayGraphContext>,
    visited: HashSet<ReplayGraphContext>,
    owned: bool,
}

struct TrialResult {
    case_index: usize,
    state_id: usize,
    replicate: usize,
    actions: [Vec<PortableSearchActionRef>; 2],
    outcome: f32,
}

#[derive(Clone)]
struct StateEstimate {
    case_index: usize,
    depth: usize,
    prediction: f32,
    mean: f64,
    variance: f64,
    sample_variance: f64,
    standard_error: f64,
    wins: usize,
    losses: usize,
    draws: usize,
    outcomes: Vec<f32>,
}

impl StateEstimate {
    fn new(
        state: &CapturedState,
        prediction: f32,
        outcomes: &[f32],
        expected: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if outcomes.len() != expected {
            return Err(format!(
                "state {} continuation count mismatch: actual={} expected={expected}",
                state.id,
                outcomes.len()
            )
            .into());
        }
        let count = outcomes.len() as f64;
        let mean = outcomes.iter().map(|value| f64::from(*value)).sum::<f64>() / count;
        let second_moment = outcomes
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            / count;
        let variance = (second_moment - mean * mean).max(0.0);
        let sample_variance = variance * count / (count - 1.0);
        Ok(Self {
            case_index: state.case_index,
            depth: state.depth,
            prediction,
            mean,
            variance,
            sample_variance,
            standard_error: (sample_variance / count).sqrt(),
            wins: outcomes.iter().filter(|value| **value > 0.0).count(),
            losses: outcomes.iter().filter(|value| **value < 0.0).count(),
            draws: outcomes.iter().filter(|value| **value == 0.0).count(),
            outcomes: outcomes.to_vec(),
        })
    }
}

#[derive(Default)]
struct BatchStats {
    rows: usize,
    batches: usize,
}

impl BatchStats {
    fn mean_batch(&self) -> f64 {
        if self.batches == 0 {
            0.0
        } else {
            self.rows as f64 / self.batches as f64
        }
    }
}

#[derive(Clone, Copy)]
struct PendingEval {
    trial: usize,
    token: WorkToken,
}

struct DrivenTrials {
    pending: Vec<(PendingEval, FeatureRow)>,
    results: Vec<TrialResult>,
    progress: bool,
}

fn run_trials<B: FeatureEvalBackend>(
    mut trials: Vec<Trial>,
    backend: &mut B,
    schema: &FeatureSchema,
    model_version: ModelVersion,
    batch_stats: &mut BatchStats,
) -> Result<Vec<TrialResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::with_capacity(trials.len());
    let mut remaining = trials.len();
    let mut collator = FeatureCollator::new(schema.clone(), nonzero(MAX_BATCH)?);
    let mut batch_bytes = Vec::new();

    while remaining > 0 {
        let thread_count = DRIVER_THREADS.min(trials.len()).max(1);
        let chunk_size = trials.len().div_ceil(thread_count);
        let driven = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut start = 0;
            for chunk in trials.chunks_mut(chunk_size) {
                let chunk_start = start;
                start += chunk.len();
                handles.push(scope.spawn(move || drive_trials(chunk_start, chunk)));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| "symmetric continuation worker panicked".to_owned())?
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let mut pending = Vec::new();
        let mut rows = Vec::new();
        let mut progress = false;
        for mut driven in driven {
            progress |= driven.progress;
            remaining -= driven.results.len();
            results.append(&mut driven.results);
            for (request, row) in driven.pending {
                pending.push(request);
                rows.push(row);
            }
        }

        if pending.is_empty() {
            if remaining > 0 && !progress {
                return Err("symmetric continuation driver made no progress".into());
            }
            continue;
        }
        for (pending_batch, row_batch) in pending.chunks(MAX_BATCH).zip(rows.chunks(MAX_BATCH)) {
            let action_counts = row_batch
                .iter()
                .map(|row| u32::try_from(row.actions.len()))
                .collect::<Result<Vec<_>, _>>()?;
            collator.collate_into(row_batch, &mut batch_bytes)?;
            let outputs = backend.eval(&batch_bytes, &action_counts)?;
            if outputs.model_version != model_version || outputs.rows.len() != pending_batch.len() {
                return Err("continuation evaluator returned mismatched outputs".into());
            }
            batch_stats.rows += pending_batch.len();
            batch_stats.batches += 1;
            for (pending, output) in pending_batch.iter().zip(outputs.rows) {
                resume_trial(
                    &mut trials[pending.trial],
                    pending.token,
                    SearchWorkResult::Eval(EvalOutput {
                        model_version,
                        policy_logits: output.policy_logits,
                        value: output.value,
                    }),
                )?;
            }
        }
    }
    Ok(results)
}

fn drive_trials(start: usize, trials: &mut [Trial]) -> Result<DrivenTrials, String> {
    let mut driven = DrivenTrials {
        pending: Vec::new(),
        results: Vec::new(),
        progress: false,
    };
    for (offset, trial) in trials.iter_mut().enumerate() {
        if trial.task.is_none() {
            continue;
        }
        loop {
            let poll = trial
                .task
                .as_mut()
                .expect("active trial has a task")
                .poll()
                .map_err(|error| error.to_string())?;
            release_releasable(trial).map_err(|error| error.to_string())?;
            match poll {
                SearchPoll::Work(work) => {
                    driven.progress = true;
                    let token = work.token();
                    match work {
                        SearchWork::Expand(work) => {
                            let result = expand(&mut trial.engine, work)
                                .map_err(|error| error.to_string())?;
                            resume_trial(trial, token, SearchWorkResult::Expand(result))
                                .map_err(|error| error.to_string())?;
                        }
                        SearchWork::Apply(work) => {
                            let result = trial
                                .engine
                                .apply(work.graph, work.candidate)
                                .map_err(|error| error.to_string())?;
                            resume_trial(trial, token, SearchWorkResult::Apply(result))
                                .map_err(|error| error.to_string())?;
                        }
                        SearchWork::Measure(work) => {
                            let result = trial
                                .engine
                                .measure(work.graph, work.options)
                                .map_err(|error| error.to_string())?;
                            resume_trial(trial, token, SearchWorkResult::Measure(result))
                                .map_err(|error| error.to_string())?;
                        }
                        SearchWork::Eval(work) => {
                            let row = feature_row_for_work(
                                &mut trial.engine,
                                &mut trial.extractor,
                                &work,
                            )
                            .map_err(|error| error.to_string())?;
                            driven.pending.push((
                                PendingEval {
                                    trial: start + offset,
                                    token,
                                },
                                row,
                            ));
                        }
                        _ => return Err("unsupported symmetric search work".to_owned()),
                    }
                }
                SearchPoll::Blocked => break,
                SearchPoll::Done(episode) => {
                    driven.progress = true;
                    let result = finish_trial(trial, episode).map_err(|error| error.to_string())?;
                    trial.task = None;
                    driven.results.push(result);
                    break;
                }
            }
        }
    }
    Ok(driven)
}

fn resume_trial(
    trial: &mut Trial,
    token: WorkToken,
    result: SearchWorkResult<WhittleGraphId, WhittleCandidateId>,
) -> EngineResult<()> {
    trial
        .task
        .as_mut()
        .expect("active trial has a task")
        .resume(token, result)?;
    release_releasable(trial)
}

fn release_releasable(trial: &mut Trial) -> EngineResult<()> {
    let handles = trial
        .task
        .as_mut()
        .expect("active trial has a task")
        .take_releasable();
    trial.engine.release(&handles.graphs, &handles.candidates)
}

fn finish_trial(
    trial: &mut Trial,
    episode: SymmetricEpisode<WhittleGraphId, WhittleCandidateId>,
) -> Result<TrialResult, Box<dyn std::error::Error>> {
    let p1_reward = measured_reward(&episode.p1.final_measure)?;
    let p2_reward = measured_reward(&episode.p2.final_measure)?;
    let p1_rewrites = trial.prefix_rewrites[0] + rewrite_steps(&episode.p1.steps);
    let p2_rewrites = trial.prefix_rewrites[1] + rewrite_steps(&episode.p2.steps);
    let outcome = symmetric_outcome(p1_reward, p2_reward, p1_rewrites, p2_rewrites);
    let mut actions = trial.prefix.actions.clone();
    actions[0].extend(episode.p1.steps.iter().map(|step| step.selected_action));
    actions[1].extend(episode.p2.steps.iter().map(|step| step.selected_action));
    let mut graphs = episode.created_graphs;
    graphs.append(&mut trial.initial_graphs);
    trial.engine.release(&graphs, &episode.created_candidates)?;
    Ok(TrialResult {
        case_index: trial.case_index,
        state_id: trial.state_id,
        replicate: trial.replicate,
        actions,
        outcome,
    })
}

fn measured_reward<G>(
    measure: &gz_engine::MeasureResult<G>,
) -> Result<f32, Box<dyn std::error::Error>> {
    if !measure.measured || !measure.valid {
        return Err("continuation terminal graph was not validly measured".into());
    }
    measure
        .scalar_reward
        .filter(|reward| reward.is_finite())
        .ok_or_else(|| "continuation terminal graph has no finite reward".into())
}

fn symmetric_outcome(
    p1_reward: f32,
    p2_reward: f32,
    p1_rewrites: usize,
    p2_rewrites: usize,
) -> f32 {
    if p1_reward > p2_reward || p1_reward == p2_reward && p1_rewrites < p2_rewrites {
        1.0
    } else if p1_reward < p2_reward || p1_reward == p2_reward && p1_rewrites > p2_rewrites {
        -1.0
    } else {
        0.0
    }
}

fn expand(
    engine: &mut WhittleEngine,
    work: gz_search::ExpandWork<WhittleGraphId>,
) -> EngineResult<ExpandResult<WhittleCandidateId>> {
    let mut candidates = Vec::new();
    engine.candidates(work.graph, work.options, &mut candidates)?;
    let graph_hash = engine.hash(work.graph)?;
    let candidates = candidates
        .into_iter()
        .map(|candidate| {
            let info = engine.candidate_info(work.graph, candidate)?;
            Ok(ExpandedCandidate {
                candidate,
                candidate_hash: info.candidate_hash,
                kind: info.kind,
                tags: info.tags,
                static_prior: info.static_prior,
            })
        })
        .collect::<EngineResult<Vec<_>>>()?;
    Ok(ExpandResult {
        graph_hash,
        candidates,
    })
}

fn feature_row_for_work(
    engine: &mut WhittleEngine,
    extractor: &mut WhittleFeatureExtractor,
    work: &EvalWork<WhittleGraphId, WhittleCandidateId>,
) -> Result<FeatureRow, Box<dyn std::error::Error>> {
    let scale = extractor.schema().config().opponent_reward_scale;
    let mut row = extractor.extract(
        engine,
        work.graph,
        &work.candidates,
        position_features(work.request.position, scale, true),
    )?;
    let opponent = work
        .opponent
        .as_deref()
        .ok_or("symmetric evaluation is missing its opponent graph")?;
    let opponent_row = extractor.extract(
        engine,
        opponent.graph,
        &[],
        position_features(opponent.position, scale, false),
    )?;
    row.opponent = Some(opponent_state(opponent_row));
    Ok(row)
}

fn position_features(
    position: gz_eval::EvalPositionContext,
    opponent_reward_scale: f32,
    dynamic_opponent: bool,
) -> PositionFeatures {
    PositionFeatures {
        root_step: position.root_step,
        leaf_depth: position.leaf_depth,
        budget_fraction: position.budget_fraction,
        budget_step: position.budget_step,
        opponent_reward: position.opponent.map_or(0.0, |opponent| {
            opponent.final_reward / opponent_reward_scale
        }),
        opponent_present: position.opponent.is_some() || dynamic_opponent,
    }
}

fn opponent_state(row: FeatureRow) -> OpponentStateFeatures {
    OpponentStateFeatures {
        node_count: row.node_count,
        node_tokens: row.node_tokens,
        node_attrs: row.node_attrs,
        edges: row.edges,
        position: row.position,
    }
}

fn reconstruct_pair(
    engine: &mut WhittleEngine,
    identity: EngineIdentity,
    prefix: &Prefix,
) -> Result<([ReconstructedActor; 2], Vec<WhittleGraphId>), Box<dyn std::error::Error>> {
    let p1 = reconstruct_actor(engine, identity, &prefix.actions[0])?;
    let p2 = reconstruct_actor(engine, identity, &prefix.actions[1])?;
    let mut owned = Vec::with_capacity(2);
    if p1.owned {
        owned.push(p1.graph);
    }
    if p2.owned {
        owned.push(p2.graph);
    }
    Ok(([p1, p2], owned))
}

fn reconstruct_actor(
    engine: &mut WhittleEngine,
    identity: EngineIdentity,
    actions: &[PortableSearchActionRef],
) -> Result<ReconstructedActor, Box<dyn std::error::Error>> {
    let mut graph = engine.root();
    let mut owned = false;
    let mut visited = HashSet::with_capacity(actions.len());
    for action in actions {
        let PortableSearchActionRef::Candidate(expected) = action else {
            return Err("captured prefix contains STOP".into());
        };
        let context = identity.context(engine.hash(graph)?);
        if expected.context != context {
            return Err("captured prefix graph context mismatch".into());
        }
        let mut candidates = Vec::new();
        engine.candidates(graph, candidate_options(), &mut candidates)?;
        let mut selected = None;
        for candidate in candidates.iter().copied() {
            if engine.candidate_info(graph, candidate)?.candidate_hash == expected.candidate_hash {
                selected = Some(candidate);
                break;
            }
        }
        let Some(selected) = selected else {
            engine.release(&[], &candidates)?;
            return Err("captured prefix candidate is not legal after reconstruction".into());
        };
        let applied = engine.apply(graph, selected)?;
        engine.release(&[], &candidates)?;
        if owned {
            engine.release(&[graph], &[])?;
        }
        visited.insert(context);
        graph = applied.after;
        owned = true;
    }
    let context = if actions.is_empty() {
        None
    } else {
        Some(identity.context(engine.hash(graph)?))
    };
    Ok(ReconstructedActor {
        graph,
        context,
        visited,
        owned,
    })
}

fn capture_states(
    cases: &[EvalCase],
    baselines: &[TrialResult],
    depths: &[usize],
) -> Result<Vec<CapturedState>, Box<dyn std::error::Error>> {
    if cases.len() != baselines.len() {
        return Err("baseline result count mismatch".into());
    }
    let mut states = Vec::new();
    for (case_offset, (case, baseline)) in cases.iter().zip(baselines).enumerate() {
        if baseline.case_index != case.index || baseline.replicate != 0 {
            return Err("baseline result identity mismatch".into());
        }
        for &depth in depths {
            if baseline.actions[0].len() <= depth
                || baseline.actions[1].len() <= depth
                || baseline.actions[0][..depth]
                    .iter()
                    .chain(&baseline.actions[1][..depth])
                    .any(|action| matches!(action, PortableSearchActionRef::Stop { .. }))
            {
                continue;
            }
            states.push(CapturedState {
                id: states.len(),
                case_offset,
                case_index: case.index,
                depth,
                prefix: Prefix {
                    actions: [
                        baseline.actions[0][..depth].to_vec(),
                        baseline.actions[1][..depth].to_vec(),
                    ],
                },
            });
        }
    }
    println!(
        "capture requested_states={} captured_states={} skipped_states={}",
        cases.len() * depths.len(),
        states.len(),
        cases.len() * depths.len() - states.len(),
    );
    Ok(states)
}

fn build_state_row(
    case: &EvalCase,
    prefix: &Prefix,
) -> Result<FeatureRow, Box<dyn std::error::Error>> {
    let mut engine = new_engine(case)?;
    let identity = EngineIdentity::from_engine(&engine);
    let (actors, owned) = reconstruct_pair(&mut engine, identity, prefix)?;
    let rewrites = prefix.rewrites();
    let mut extractor = feature_extractor(&engine);
    let mut candidates = Vec::new();
    engine.candidates(actors[0].graph, candidate_options(), &mut candidates)?;
    let mut row = extractor.extract(
        &engine,
        actors[0].graph,
        &candidates,
        replay_position(rewrites[0], true),
    )?;
    let opponent = extractor.extract(
        &engine,
        actors[1].graph,
        &[],
        replay_position(rewrites[1], false),
    )?;
    row.opponent = Some(opponent_state(opponent));
    row.validate(extractor.schema())?;
    engine.release(&owned, &candidates)?;
    Ok(row)
}

fn replay_position(step: usize, opponent_present: bool) -> PositionFeatures {
    PositionFeatures {
        root_step: step as u32,
        leaf_depth: 0,
        budget_fraction: MAX_STEPS.saturating_sub(step) as f32 / MAX_STEPS as f32,
        budget_step: 1.0 / MAX_STEPS as f32,
        opponent_reward: 0.0,
        opponent_present,
    }
}

fn evaluate_rows<B: FeatureEvalBackend>(
    backend: &mut B,
    schema: &FeatureSchema,
    rows: &[FeatureRow],
    model_version: ModelVersion,
    batch_stats: &mut BatchStats,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut collator = FeatureCollator::new(schema.clone(), nonzero(MAX_BATCH)?);
    let mut batch_bytes = Vec::new();
    let mut values = Vec::with_capacity(rows.len());
    for batch in rows.chunks(MAX_BATCH) {
        let action_counts = batch
            .iter()
            .map(|row| u32::try_from(row.actions.len()))
            .collect::<Result<Vec<_>, _>>()?;
        collator.collate_into(batch, &mut batch_bytes)?;
        let outputs = backend.eval(&batch_bytes, &action_counts)?;
        if outputs.model_version != model_version || outputs.rows.len() != batch.len() {
            return Err("state evaluator returned mismatched outputs".into());
        }
        batch_stats.rows += batch.len();
        batch_stats.batches += 1;
        values.extend(outputs.rows.into_iter().map(|row| row.value));
    }
    Ok(values)
}

fn print_estimates(estimates: &[StateEstimate]) {
    println!(
        "states case depth prediction empirical_mean standard_error variance wins losses draws"
    );
    for row in estimates {
        println!(
            "state {} {} {:.6} {:.6} {:.6} {:.6} {} {} {}",
            row.case_index,
            row.depth,
            row.prediction,
            row.mean,
            row.standard_error,
            row.variance,
            row.wins,
            row.losses,
            row.draws,
        );
    }
}

fn print_summary(scope: &str, rows: &[StateEstimate]) {
    if rows.is_empty() {
        println!("expectation scope={scope} states=0");
        return;
    }
    let count = rows.len() as f64;
    let raw_mse = rows
        .iter()
        .map(|row| (f64::from(row.prediction) - row.mean).powi(2))
        .sum::<f64>()
        / count;
    let estimation_variance = rows
        .iter()
        .map(|row| row.sample_variance / row.outcomes.len() as f64)
        .sum::<f64>()
        / count;
    let corrected_mse = raw_mse - estimation_variance;
    let mae = rows
        .iter()
        .map(|row| (f64::from(row.prediction) - row.mean).abs())
        .sum::<f64>()
        / count;
    let zero_mse = rows.iter().map(|row| row.mean.powi(2)).sum::<f64>() / count;
    let label_count = rows.iter().map(|row| row.outcomes.len()).sum::<usize>() as f64;
    let label_mse = rows
        .iter()
        .flat_map(|row| {
            row.outcomes
                .iter()
                .map(move |outcome| (f64::from(row.prediction) - f64::from(*outcome)).powi(2))
        })
        .sum::<f64>()
        / label_count;
    let intrinsic_variance = rows.iter().map(|row| row.variance).sum::<f64>() / count;
    let mean_abs_expectation = rows.iter().map(|row| row.mean.abs()).sum::<f64>() / count;
    let ambiguous = rows
        .iter()
        .filter(|row| row.mean.abs() <= 1.96 * row.standard_error)
        .count();
    let mean_prediction = rows
        .iter()
        .map(|row| f64::from(row.prediction))
        .sum::<f64>()
        / count;
    let mean_expectation = rows.iter().map(|row| row.mean).sum::<f64>() / count;
    let correlation = correlation(rows);
    let zero_skill = if zero_mse > 0.0 {
        1.0 - raw_mse / zero_mse
    } else {
        f64::NAN
    };
    println!(
        "expectation scope={scope} states={} labels={} mean_prediction={mean_prediction:.6} mean_empirical={mean_expectation:.6} mean_abs_empirical={mean_abs_expectation:.6} rmse_to_empirical={:.6} noise_corrected_mse={corrected_mse:.6} mae_to_empirical={mae:.6} zero_baseline_mse={zero_mse:.6} zero_baseline_skill={zero_skill:.6} correlation={correlation:.6} intrinsic_label_variance={intrinsic_variance:.6} label_mse={label_mse:.6} decomposition_delta={:.9} ci95_includes_zero={ambiguous}/{}",
        rows.len(),
        label_count as usize,
        raw_mse.sqrt(),
        label_mse - raw_mse - intrinsic_variance,
        rows.len(),
    );
}

fn correlation(rows: &[StateEstimate]) -> f64 {
    let count = rows.len() as f64;
    let mean_prediction = rows
        .iter()
        .map(|row| f64::from(row.prediction))
        .sum::<f64>()
        / count;
    let mean_outcome = rows.iter().map(|row| row.mean).sum::<f64>() / count;
    let covariance = rows
        .iter()
        .map(|row| (f64::from(row.prediction) - mean_prediction) * (row.mean - mean_outcome))
        .sum::<f64>();
    let prediction_sum_squares = rows
        .iter()
        .map(|row| (f64::from(row.prediction) - mean_prediction).powi(2))
        .sum::<f64>();
    let outcome_sum_squares = rows
        .iter()
        .map(|row| (row.mean - mean_outcome).powi(2))
        .sum::<f64>();
    let denominator = (prediction_sum_squares * outcome_sum_squares).sqrt();
    if denominator == 0.0 {
        f64::NAN
    } else {
        covariance / denominator
    }
}

fn search(engine: &WhittleEngine, simulations: usize, considered: usize) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: MAX_STEPS,
        simulations: NonZeroUsize::new(simulations).expect("validated simulations"),
        max_considered_actions: NonZeroUsize::new(considered).expect("validated considered"),
        seed: SEARCH_SEED,
        gumbel_scale: GUMBEL_SCALE,
        c_visit: C_VISIT,
        c_scale: C_SCALE,
        temperature_moves: 0,
        gumbel_noise_overlap: GUMBEL_NOISE_OVERLAP,
        tree_reuse: true,
        export_position: true,
        mask_stop: false,
        no_backtrack: true,
        value_mode: GumbelValueMode::SymmetricSelfplay,
        candidate_options: candidate_options(),
        measure_options: engine.measure_options(),
    })
    .with_symmetric_wave_batching(true)
}

fn candidate_options() -> CandidateOptions {
    CandidateOptions {
        max_candidates: Some(MAX_CANDIDATES),
        deterministic_order: true,
    }
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

fn new_engine(case: &EvalCase) -> EngineResult<WhittleEngine> {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(case.bytes.clone()),
        ..WhittleEngineConfig::default()
    })
}

fn load_cases(dir: &Path, limit: usize) -> Result<Vec<EvalCase>, Box<dyn std::error::Error>> {
    let manifest_path = dir.join("manifest.tsv");
    let manifest = std::fs::read_to_string(&manifest_path)?;
    let mut cases = Vec::new();
    for (line_number, line) in manifest.lines().enumerate() {
        if line_number == 0 && line == "index\tcost\tsource_seed\tartifact" {
            continue;
        }
        if cases.len() == limit {
            break;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(format!(
                "{}:{}: expected four tab-separated fields",
                manifest_path.display(),
                line_number + 1
            )
            .into());
        }
        let index = fields[0].parse::<usize>()?;
        let bytes = std::fs::read(dir.join(fields[3]))?;
        if !bytes.starts_with(b"WAV1") {
            return Err(format!("case {index} is not a WAV1 artifact").into());
        }
        cases.push(EvalCase { index, bytes });
    }
    if cases.is_empty() {
        return Err(format!("{} contains no evaluation cases", dir.display()).into());
    }
    Ok(cases)
}

fn spawn_evaluator(config: &Config) -> Result<EvaluatorProcess, Box<dyn std::error::Error>> {
    let socket_path = std::env::temp_dir().join(format!(
        "gz-symmetric-value-expectation-{}.sock",
        std::process::id()
    ));
    Ok(EvaluatorProcess::spawn(EvaluatorProcessConfig {
        working_dir: PathBuf::from("python"),
        socket_path,
        ready_timeout: Duration::from_secs(30),
        io_timeout: Duration::from_secs(300),
        extra_args: vec![
            "--backend".to_owned(),
            "torch".to_owned(),
            "--checkpoint-dir".to_owned(),
            config.checkpoint_dir.display().to_string(),
            "--checkpoint-pointer".to_owned(),
            config.checkpoint_pointer.clone(),
            "--device".to_owned(),
            config.device.clone(),
            "--max-batch".to_owned(),
            MAX_BATCH.to_string(),
            "--poll-interval".to_owned(),
            "0".to_owned(),
            "--require-state-input".to_owned(),
            "joint-board".to_owned(),
            "--require-value-input".to_owned(),
            "single".to_owned(),
        ],
        ..EvaluatorProcessConfig::default()
    })?)
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait()? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("Python evaluator exited with {status}").into()),
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            None => return Err("Python evaluator did not exit".into()),
        }
    }
}

fn rewrite_steps(steps: &[gz_search::GumbelStep<WhittleGraphId, WhittleCandidateId>]) -> usize {
    steps
        .iter()
        .filter(|step| matches!(step.selected_action, PortableSearchActionRef::Candidate(_)))
        .count()
}

fn rewrite_count(actions: &[PortableSearchActionRef]) -> usize {
    actions
        .iter()
        .filter(|action| matches!(action, PortableSearchActionRef::Candidate(_)))
        .count()
}

fn continuation_seed(base: u64, state: usize, replicate: usize) -> u64 {
    splitmix64(
        base ^ (state as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (replicate as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9),
    )
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn parse_depths(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut depths = value
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    depths.sort_unstable();
    depths.dedup();
    Ok(depths)
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

fn nonzero(value: usize) -> Result<NonZeroUsize, Box<dyn std::error::Error>> {
    NonZeroUsize::new(value).ok_or_else(|| "value must be nonzero".into())
}
