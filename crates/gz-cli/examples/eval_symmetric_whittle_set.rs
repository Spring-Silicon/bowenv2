use gz_engine::{CandidateOptions, EngineResult, GraphEngine, GraphHash, PortableSearchActionRef};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleFeatureExtractorConfig,
    WhittleGraphId, WhittleRoot,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::{
    BackendOutputs, EvaluatorProcess, EvaluatorProcessConfig, FeatureEvalBackend, Hello,
    ModelGeneration, PendingBatch, ProcessBackend, ServiceError, ServiceResult,
};
use gz_measurer::ValueTargetConfig;
use gz_orchestrator::reference::{Reference, ReferenceProvider};
use gz_orchestrator::{
    FeaturizedRuntime, ReplayRuntime, RootSource, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayEpisodeId, ReplayStore};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig, GumbelValueMode};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const EXPECTED_CASES: usize = 20;
const MAX_CANDIDATES: usize = 1023;
const MAX_STEPS: usize = 96;
const DEFAULT_MAX_CONSIDERED: usize = 8;
const DEFAULT_SIMULATIONS: usize = 48;
const DEFAULT_MAX_BATCH: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: eval_symmetric_whittle_set ARTIFACT_DIR CHECKPOINT_DIR [DEVICE] [CHECKPOINT_POINTER] [MAX_CONSIDERED] [SIMULATIONS] [GUMBEL_SCALE] [GUMBEL_NOISE_OVERLAP] [SEED] [C_SCALE] [TREE_REUSE] [MAX_BATCH] [VALUE_CHECKPOINT_DIR] [VALUE_CHECKPOINT_POINTER]";
    let artifact_dir = path_arg(1)?.ok_or(usage)?;
    let checkpoint_dir = path_arg(2)?.ok_or(usage)?;
    let device = string_arg(3).unwrap_or_else(|| "cuda:0".to_owned());
    let checkpoint_pointer = string_arg(4).unwrap_or_else(|| "latest.json".to_owned());
    let max_considered = parse_arg(5)?.unwrap_or(DEFAULT_MAX_CONSIDERED);
    let simulations = parse_arg(6)?.unwrap_or(DEFAULT_SIMULATIONS);
    let gumbel_scale = parse_arg(7)?.unwrap_or(0.0_f32);
    let gumbel_noise_overlap = parse_arg(8)?.unwrap_or(-1.0_f32);
    let seed = parse_arg(9)?.unwrap_or(42_u64);
    let c_scale = parse_arg(10)?.unwrap_or(1.0_f32);
    if !c_scale.is_finite() || c_scale < 0.0 {
        return Err("C_SCALE must be finite and non-negative".into());
    }
    let tree_reuse = parse_arg(11)?.unwrap_or(false);
    let max_batch = nonzero(parse_arg(12)?.unwrap_or(DEFAULT_MAX_BATCH))?;
    let value_checkpoint_dir = path_arg(13)?.map(|path| absolute(&path)).transpose()?;
    let value_checkpoint_pointer = string_arg(14).unwrap_or_else(|| "latest.json".to_owned());

    let artifact_dir = absolute(&artifact_dir)?;
    let checkpoint_dir = absolute(&checkpoint_dir)?;
    let cases = load_cases(&artifact_dir)?;
    if cases.len() != EXPECTED_CASES {
        return Err(format!(
            "expected {EXPECTED_CASES} evaluation cases, found {}",
            cases.len()
        )
        .into());
    }

    let mut engines = Vec::with_capacity(cases.len());
    let mut extractors = Vec::with_capacity(cases.len());
    let mut root_sources = Vec::with_capacity(cases.len());
    let mut case_by_hash = HashMap::with_capacity(cases.len());
    for case in &cases {
        let mut engine = WhittleEngine::new(WhittleEngineConfig {
            root: WhittleRoot::Artifact(case.bytes.clone()),
            ..WhittleEngineConfig::default()
        })?;
        let root = engine.root();
        let root_cost = measure_cost(&mut engine, root)?;
        if root_cost != case.expected_cost {
            return Err(format!(
                "case {} root cost mismatch: manifest={} graphzero={root_cost}",
                case.index, case.expected_cost
            )
            .into());
        }
        let root_hash = engine.hash(root)?;
        if case_by_hash.insert(root_hash, case.index).is_some() {
            return Err(format!("duplicate root hash for case {}", case.index).into());
        }
        extractors.push(WhittleFeatureExtractor::with_config(
            &engine,
            WhittleFeatureExtractorConfig {
                max_actions: (MAX_CANDIDATES + 1) as u32,
                ..WhittleFeatureExtractorConfig::default()
            },
        ));
        root_sources.push(OneRoot::default());
        engines.push(engine);
    }

    let schema = extractors
        .first()
        .ok_or("evaluation set is empty")?
        .schema()
        .clone();
    let first_engine = engines.first().ok_or("evaluation set is empty")?;
    let checkpoint_dir_string = checkpoint_dir.display().to_string();
    let mut process = spawn_evaluator(
        "policy",
        &checkpoint_dir_string,
        &device,
        &checkpoint_pointer,
        max_batch.get(),
    )?;
    let hello = Hello::new(
        schema.hash(),
        u32::try_from(max_batch.get())?,
        first_engine.engine_id(),
        first_engine.engine_version(),
        first_engine.action_set_hash(),
    );
    let policy_backend = process.connect(&hello)?;
    let model_version = policy_backend.model_version();
    let mut value_process = None;
    let (backend, value_model_version) = if let Some(value_checkpoint_dir) = &value_checkpoint_dir {
        let mut spawned = spawn_evaluator(
            "value",
            &value_checkpoint_dir.display().to_string(),
            &device,
            &value_checkpoint_pointer,
            max_batch.get(),
        )?;
        let value_backend = spawned.connect(&hello)?;
        let value_model_version = value_backend.model_version();
        value_process = Some(spawned);
        (
            HeadBackend::Split {
                policy: policy_backend,
                value: value_backend,
            },
            value_model_version,
        )
    } else {
        (HeadBackend::Single(policy_backend), model_version)
    };

    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps: MAX_STEPS,
        simulations: nonzero(simulations)?,
        max_considered_actions: nonzero(max_considered)?,
        seed,
        gumbel_scale,
        c_visit: 50.0,
        c_scale,
        temperature_moves: 0,
        gumbel_noise_overlap,
        tree_reuse,
        export_position: true,
        mask_stop: false,
        no_backtrack: true,
        value_mode: GumbelValueMode::SymmetricSelfplay,
        candidate_options: CandidateOptions {
            max_candidates: Some(MAX_CANDIDATES),
            deterministic_order: true,
        },
        measure_options: first_engine.measure_options(),
    })
    .with_symmetric_wave_batching(true);
    let placeholder = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed,
        ..RandomValueEvaluatorConfig::default()
    })?;
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        placeholder,
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: nonzero(1)?,
            max_batch,
            flush_after: Duration::from_millis(3),
            admission_stagger: Duration::ZERO,
            admission_smoothing: None,
        },
    );
    let replay_dir = temporary_replay_dir()?;
    let store = ReplayStore::open(&replay_dir)?;
    let providers = (0..cases.len()).map(|_| EvalNoReferenceProvider).collect();
    let started = Instant::now();
    let run = orchestrator.run_featurized_with_replay(
        root_sources,
        GumbelEpisodeContext::default(),
        FeaturizedRuntime {
            extractors,
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
    let elapsed = started.elapsed();
    wait_for_process_exit(&mut process)?;
    if let Some(process) = &mut value_process {
        wait_for_process_exit(process)?;
    }

    if run.episodes_dropped != 0 || run.episodes_appended as usize != cases.len() {
        return Err(format!(
            "evaluation completion mismatch: appended={} dropped={} expected={}",
            run.episodes_appended,
            run.episodes_dropped,
            cases.len()
        )
        .into());
    }

    let mut results = Vec::with_capacity(cases.len());
    let mut seen = HashSet::with_capacity(cases.len());
    for pair in 0..cases.len() {
        let p1_id = ReplayEpisodeId::new((pair * 2) as u64);
        let p2_id = ReplayEpisodeId::new((pair * 2 + 1) as u64);
        let p1 = store
            .episode(p1_id)?
            .ok_or_else(|| format!("missing P1 replay record {p1_id:?}"))?;
        let p2 = store
            .episode(p2_id)?
            .ok_or_else(|| format!("missing P2 replay record {p2_id:?}"))?;
        if p1.root.graph.graph_hash != p2.root.graph.graph_hash {
            return Err(format!("root mismatch for replay pair {pair}").into());
        }
        let index = *case_by_hash
            .get(&p1.root.graph.graph_hash)
            .ok_or_else(|| format!("unknown replay root {}", p1.root.graph.graph_hash))?;
        if !seen.insert(index) {
            return Err(format!("duplicate result for case {index}").into());
        }
        results.push(EvalResult {
            index,
            root_hash: p1.root.graph.graph_hash,
            root_cost: cases[index].expected_cost,
            p1_cost: -p1.outcome.learner_reward,
            p2_cost: -p2.outcome.learner_reward,
            p1_rewrites: rewrite_count(&p1.steps),
            p2_rewrites: rewrite_count(&p2.steps),
            p1_stopped: p1.outcome.stopped,
            p2_stopped: p2.outcome.stopped,
        });
    }
    results.sort_by_key(|result| result.index);
    println!(
        "heads policy_model_version={model_version} value_model_version={value_model_version}"
    );
    print_results(
        &results,
        model_version,
        &checkpoint_pointer,
        max_considered,
        simulations,
        gumbel_scale,
        gumbel_noise_overlap,
        c_scale,
        tree_reuse,
        max_batch.get(),
        seed,
        elapsed,
        &run.batch_sizes,
    );
    drop(store);
    std::fs::remove_dir_all(replay_dir)?;
    Ok(())
}

enum HeadBackend {
    Single(ProcessBackend),
    Split {
        policy: ProcessBackend,
        value: ProcessBackend,
    },
}

impl FeatureEvalBackend for HeadBackend {
    fn model_generation(&self) -> ModelGeneration {
        match self {
            Self::Single(backend) => backend.model_generation(),
            Self::Split { policy, .. } => policy.model_generation(),
        }
    }

    fn batch_capacity(&self) -> Option<NonZeroUsize> {
        match self {
            Self::Single(backend) => backend.batch_capacity(),
            Self::Split { policy, value } => policy.batch_capacity().min(value.batch_capacity()),
        }
    }

    fn capacity_work(&self, actual_rows: usize, max_batch: usize) -> usize {
        match self {
            Self::Single(backend) => backend.capacity_work(actual_rows, max_batch),
            Self::Split { policy, .. } => policy.capacity_work(actual_rows, max_batch),
        }
    }

    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        match self {
            Self::Single(backend) => backend.eval(batch_bytes, action_counts),
            Self::Split { policy, value } => {
                let mut policy_outputs = policy.eval(batch_bytes, action_counts)?;
                let value_outputs = value.eval(batch_bytes, action_counts)?;
                if policy_outputs.rows.len() != value_outputs.rows.len() {
                    return Err(ServiceError::protocol(
                        "split-head evaluators returned different row counts",
                    ));
                }
                for (policy_row, value_row) in
                    policy_outputs.rows.iter_mut().zip(value_outputs.rows)
                {
                    policy_row.value = value_row.value;
                }
                Ok(policy_outputs)
            }
        }
    }

    fn submit(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<PendingBatch> {
        match self {
            Self::Single(backend) => backend.submit(batch_bytes, action_counts),
            Self::Split { .. } => Ok(PendingBatch::Ready(self.eval(batch_bytes, action_counts)?)),
        }
    }

    fn submit_for_model(
        &mut self,
        model: ModelGeneration,
        batch_bytes: &[u8],
        action_counts: &[u32],
    ) -> ServiceResult<PendingBatch> {
        match self {
            Self::Single(backend) => backend.submit_for_model(model, batch_bytes, action_counts),
            Self::Split { .. } => {
                if model != self.model_generation() {
                    return Err(ServiceError::backend(
                        1,
                        "split-head evaluator was asked for the wrong policy model",
                    ));
                }
                Ok(PendingBatch::Ready(self.eval(batch_bytes, action_counts)?))
            }
        }
    }

    fn receive(&mut self, pending: PendingBatch) -> ServiceResult<BackendOutputs> {
        match self {
            Self::Single(backend) => backend.receive(pending),
            Self::Split { .. } => match pending {
                PendingBatch::Ready(outputs) => Ok(outputs),
                PendingBatch::InFlight { .. } => Err(ServiceError::protocol(
                    "split-head evaluator received an in-flight batch",
                )),
            },
        }
    }
}

#[derive(Default)]
struct OneRoot {
    yielded: bool,
}

impl RootSource<WhittleEngine> for OneRoot {
    fn next_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        if self.yielded {
            return Ok(None);
        }
        self.yielded = true;
        Ok(Some(engine.root()))
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

struct EvalCase {
    index: usize,
    expected_cost: f32,
    bytes: Vec<u8>,
}

struct EvalResult {
    index: usize,
    root_hash: GraphHash,
    root_cost: f32,
    p1_cost: f32,
    p2_cost: f32,
    p1_rewrites: usize,
    p2_rewrites: usize,
    p1_stopped: bool,
    p2_stopped: bool,
}

fn load_cases(dir: &Path) -> Result<Vec<EvalCase>, Box<dyn std::error::Error>> {
    let manifest_path = dir.join("manifest.tsv");
    let manifest = std::fs::read_to_string(&manifest_path)?;
    let mut cases = Vec::new();
    for (line_number, line) in manifest.lines().enumerate() {
        if line_number == 0 && line == "index\tcost\tsource_seed\tartifact" {
            continue;
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
        if index != cases.len() {
            return Err(format!(
                "{}:{}: expected index {}, found {index}",
                manifest_path.display(),
                line_number + 1,
                cases.len()
            )
            .into());
        }
        let expected_cost = fields[1].parse::<f32>()?;
        let _source_seed = fields[2].parse::<u64>()?;
        let bytes = std::fs::read(dir.join(fields[3]))?;
        if !bytes.starts_with(b"WAV1") {
            return Err(format!("case {index} is not a WAV1 artifact").into());
        }
        cases.push(EvalCase {
            index,
            expected_cost,
            bytes,
        });
    }
    Ok(cases)
}

fn spawn_evaluator(
    role: &str,
    checkpoint_dir: &str,
    device: &str,
    pointer: &str,
    max_batch: usize,
) -> Result<EvaluatorProcess, Box<dyn std::error::Error>> {
    let socket_path = std::env::temp_dir().join(format!(
        "gz-whittle-symmetric-eval-{role}-{}.sock",
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
            checkpoint_dir.to_owned(),
            "--checkpoint-pointer".to_owned(),
            pointer.to_owned(),
            "--device".to_owned(),
            device.to_owned(),
            "--max-batch".to_owned(),
            max_batch.to_string(),
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
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            None => return Err("Python evaluator did not exit".into()),
        }
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

fn rewrite_count(steps: &[gz_engine::SearchStepRef]) -> usize {
    steps
        .iter()
        .filter(|step| matches!(step.action, PortableSearchActionRef::Candidate(_)))
        .count()
}

#[allow(clippy::too_many_arguments)]
fn print_results(
    results: &[EvalResult],
    model_version: gz_engine::ModelVersion,
    checkpoint_pointer: &str,
    max_considered: usize,
    simulations: usize,
    gumbel_scale: f32,
    gumbel_noise_overlap: f32,
    c_scale: f32,
    tree_reuse: bool,
    max_batch: usize,
    seed: u64,
    elapsed: Duration,
    batch_sizes: &[usize],
) {
    let case_count = results.len() as f32;
    let seat_count = case_count * 2.0;
    let mean_root = results.iter().map(|row| row.root_cost).sum::<f32>() / case_count;
    let mean_p1 = results.iter().map(|row| row.p1_cost).sum::<f32>() / case_count;
    let mean_p2 = results.iter().map(|row| row.p2_cost).sum::<f32>() / case_count;
    let mean_seat = results
        .iter()
        .map(|row| row.p1_cost + row.p2_cost)
        .sum::<f32>()
        / seat_count;
    let mean_best = results
        .iter()
        .map(|row| row.p1_cost.min(row.p2_cost))
        .sum::<f32>()
        / case_count;
    let mean_reduction_percent = results
        .iter()
        .map(|row| {
            ((row.root_cost - row.p1_cost) / row.root_cost
                + (row.root_cost - row.p2_cost) / row.root_cost)
                * 50.0
        })
        .sum::<f32>()
        / case_count;
    let mean_rewrites = results
        .iter()
        .map(|row| row.p1_rewrites + row.p2_rewrites)
        .sum::<usize>() as f32
        / seat_count;
    let p1_wins = results
        .iter()
        .filter(|row| row.p1_cost < row.p2_cost)
        .count();
    let p2_wins = results
        .iter()
        .filter(|row| row.p2_cost < row.p1_cost)
        .count();
    let ties = results.len() - p1_wins - p2_wins;
    let stops = results
        .iter()
        .map(|row| usize::from(row.p1_stopped) + usize::from(row.p2_stopped))
        .sum::<usize>();
    let eval_rows = batch_sizes.iter().sum::<usize>();
    let mean_batch = if batch_sizes.is_empty() {
        0.0
    } else {
        eval_rows as f32 / batch_sizes.len() as f32
    };
    println!(
        "settings cases={} max_steps={MAX_STEPS} considered={max_considered} simulations={simulations} gumbel_scale={gumbel_scale} overlap={gumbel_noise_overlap} c_scale={c_scale} tree_reuse={tree_reuse} wave_batching=true max_batch={max_batch} seed={seed}",
        results.len()
    );
    println!("model checkpoint_pointer={checkpoint_pointer} model_version={model_version}");
    println!(
        "summary mean_root={mean_root:.3} mean_p1={mean_p1:.3} mean_p2={mean_p2:.3} mean_seat={mean_seat:.3} mean_best_of_two={mean_best:.3} mean_absolute_reduction={:.3} mean_percent_reduction={mean_reduction_percent:.3} p1_wins={p1_wins} p2_wins={p2_wins} ties={ties} stops={stops}/{} mean_rewrites={mean_rewrites:.3} elapsed_seconds={:.3} eval_rows={eval_rows} eval_batches={} mean_batch={mean_batch:.3}",
        mean_root - mean_seat,
        results.len() * 2,
        elapsed.as_secs_f64(),
        batch_sizes.len(),
    );
    println!("index root p1 p2 best p1_rewrites p2_rewrites p1_stop p2_stop root_hash");
    for result in results {
        println!(
            "{} {:.0} {:.0} {:.0} {:.0} {} {} {} {} {}",
            result.index,
            result.root_cost,
            result.p1_cost,
            result.p2_cost,
            result.p1_cost.min(result.p2_cost),
            result.p1_rewrites,
            result.p2_rewrites,
            result.p1_stopped,
            result.p2_stopped,
            result.root_hash,
        );
    }
}

fn temporary_replay_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "gz-whittle-symmetric-eval-replay-{}-{nonce}",
        std::process::id()
    )))
}

fn nonzero(value: usize) -> Result<NonZeroUsize, Box<dyn std::error::Error>> {
    NonZeroUsize::new(value).ok_or_else(|| "value must be nonzero".into())
}

fn absolute(path: &Path) -> Result<PathBuf, std::io::Error> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn path_arg(index: usize) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    Ok(string_arg(index).map(PathBuf::from))
}

fn string_arg(index: usize) -> Option<String> {
    std::env::args().nth(index)
}

fn parse_arg<T>(index: usize) -> Result<Option<T>, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    string_arg(index)
        .map(|value| value.parse::<T>().map_err(Into::into))
        .transpose()
}
