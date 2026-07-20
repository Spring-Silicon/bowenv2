use gz_engine::{CandidateOptions, EngineResult, GraphEngine, GraphHash};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleFeatureExtractorConfig,
    WhittleGraphId, WhittleRoot,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::{EvaluatorProcess, EvaluatorProcessConfig, Hello};
use gz_measurer::ValueTargetConfig;
use gz_orchestrator::reference::{Reference, ReferenceProvider};
use gz_orchestrator::{
    FeaturizedRuntime, ReplayRuntime, RootSource, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayEpisodeId, ReplayStore};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const EXPECTED_CASES: usize = 20;
const MAX_CANDIDATES: usize = 1023;
const MAX_STEPS: usize = 64;
const DEFAULT_SIMULATIONS: usize = 48;
const MAX_CONSIDERED: usize = 8;
const MAX_BATCH: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = path_arg(1)?.ok_or(
        "usage: eval_whittle_set ARTIFACT_DIR CHECKPOINT_DIR [DEVICE] [CURRENT_POINTER] [INCUMBENT_POINTER] [SIMULATIONS]",
    )?;
    let checkpoint_dir = path_arg(2)?.ok_or(
        "usage: eval_whittle_set ARTIFACT_DIR CHECKPOINT_DIR [DEVICE] [CURRENT_POINTER] [INCUMBENT_POINTER] [SIMULATIONS]",
    )?;
    let device = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "cuda:0".to_owned());
    let current_pointer = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "latest.json".to_owned());
    let incumbent_pointer = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "best.json".to_owned());
    let simulations = std::env::args()
        .nth(6)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_SIMULATIONS);

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
    let mut current_process =
        spawn_evaluator("current", &checkpoint_dir_string, &device, &current_pointer)?;
    let hello = Hello::new(
        schema.hash(),
        MAX_BATCH as u32,
        first_engine.engine_id(),
        first_engine.engine_version(),
        first_engine.action_set_hash(),
    );
    let current_backend = current_process.connect(&hello)?;
    let current_version = current_backend.model_version();
    let mut incumbent_process = spawn_evaluator(
        "incumbent",
        &checkpoint_dir_string,
        &device,
        &incumbent_pointer,
    )?;
    let incumbent_backend = incumbent_process.connect(&hello)?;
    let incumbent_version = incumbent_backend.model_version();

    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps: MAX_STEPS,
        simulations: nonzero(simulations)?,
        max_considered_actions: nonzero(MAX_CONSIDERED)?,
        seed: 42,
        gumbel_scale: 0.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        gumbel_noise_overlap: -1.0,
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
        engines,
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
    let replay_dir = temporary_replay_dir()?;
    let store = ReplayStore::open(&replay_dir)?;
    let providers = (0..cases.len()).map(|_| EvalSampledTreeProvider).collect();
    let run = orchestrator.run_featurized_with_replay(
        root_sources,
        GumbelEpisodeContext::default(),
        FeaturizedRuntime {
            extractors,
            backends: vec![current_backend],
            reference_backends: vec![incumbent_backend],
            challenger_backends: vec![],
        },
        ReplayRuntime {
            store: &store,
            providers,
            backpressure: None,
            length_tiebreak: false,
            value_target: ValueTargetConfig::Sign,
        },
    )?;
    wait_for_process_exit(&mut current_process)?;
    wait_for_process_exit(&mut incumbent_process)?;

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
        let replay_id = ReplayEpisodeId::new((pair * 2) as u64);
        let record = store
            .episode(replay_id)?
            .ok_or_else(|| format!("missing learner replay record {replay_id:?}"))?;
        let index = *case_by_hash
            .get(&record.root.graph.graph_hash)
            .ok_or_else(|| format!("unknown replay root {}", record.root.graph.graph_hash))?;
        if !seen.insert(index) {
            return Err(format!("duplicate result for case {index}").into());
        }
        let reference = record
            .outcome
            .reference
            .as_ref()
            .ok_or_else(|| format!("case {index} has no incumbent outcome"))?;
        results.push(EvalResult {
            index,
            root_hash: record.root.graph.graph_hash,
            root_cost: cases[index].expected_cost,
            learner_cost: -record.outcome.learner_reward,
            incumbent_cost: -reference.reward,
            steps: record.steps.len(),
            stopped: record.outcome.stopped,
        });
    }
    results.sort_by_key(|result| result.index);
    print_results(
        &results,
        current_version,
        incumbent_version,
        &current_pointer,
        &incumbent_pointer,
        simulations,
        &run.batch_sizes,
    );
    drop(store);
    std::fs::remove_dir_all(replay_dir)?;
    Ok(())
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

struct EvalCase {
    index: usize,
    expected_cost: f32,
    bytes: Vec<u8>,
}

struct EvalResult {
    index: usize,
    root_hash: GraphHash,
    root_cost: f32,
    learner_cost: f32,
    incumbent_cost: f32,
    steps: usize,
    stopped: bool,
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
) -> Result<EvaluatorProcess, Box<dyn std::error::Error>> {
    let socket_path = std::env::temp_dir().join(format!(
        "gz-whittle-eval-{role}-{}.sock",
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
            MAX_BATCH.to_string(),
            "--poll-interval".to_owned(),
            "0".to_owned(),
        ],
        ..EvaluatorProcessConfig::default()
    })?)
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
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

fn measure_cost(
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
) -> Result<f32, Box<dyn std::error::Error>> {
    Ok(-engine
        .measure(graph, engine.measure_options())?
        .scalar_reward
        .ok_or("graph was not measured")?)
}

fn print_results(
    results: &[EvalResult],
    current_version: gz_engine::ModelVersion,
    incumbent_version: gz_engine::ModelVersion,
    current_pointer: &str,
    incumbent_pointer: &str,
    simulations: usize,
    batch_sizes: &[usize],
) {
    let count = results.len() as f32;
    let mean_root = results.iter().map(|row| row.root_cost).sum::<f32>() / count;
    let mean_learner = results.iter().map(|row| row.learner_cost).sum::<f32>() / count;
    let mean_incumbent = results.iter().map(|row| row.incumbent_cost).sum::<f32>() / count;
    let mean_steps = results.iter().map(|row| row.steps).sum::<usize>() as f32 / count;
    let wins = results
        .iter()
        .filter(|row| row.learner_cost < row.incumbent_cost)
        .count();
    let losses = results
        .iter()
        .filter(|row| row.learner_cost > row.incumbent_cost)
        .count();
    let ties = results.len() - wins - losses;
    let stopped = results.iter().filter(|row| row.stopped).count();
    let eval_rows = batch_sizes.iter().sum::<usize>();
    let mean_batch = if batch_sizes.is_empty() {
        0.0
    } else {
        eval_rows as f32 / batch_sizes.len() as f32
    };
    println!(
        "settings cases={} max_steps={MAX_STEPS} considered={MAX_CONSIDERED} simulations={simulations} gumbel_scale=0 overlap=disabled tree_reuse=false",
        results.len()
    );
    println!(
        "models current_pointer={current_pointer} current_version={current_version} incumbent_pointer={incumbent_pointer} incumbent_version={incumbent_version}"
    );
    println!(
        "summary mean_root={mean_root:.3} mean_current={mean_learner:.3} mean_incumbent={mean_incumbent:.3} mean_reduction={:.3} wins={wins} losses={losses} ties={ties} stopped={stopped}/{} mean_steps={mean_steps:.3} eval_rows={eval_rows} eval_batches={} mean_batch={mean_batch:.3}",
        mean_root - mean_learner,
        results.len(),
        batch_sizes.len(),
    );
    println!("index root current incumbent reduction steps stopped root_hash");
    for result in results {
        println!(
            "{} {:.0} {:.0} {:.0} {:+.0} {} {} {}",
            result.index,
            result.root_cost,
            result.learner_cost,
            result.incumbent_cost,
            result.root_cost - result.learner_cost,
            result.steps,
            result.stopped,
            result.root_hash,
        );
    }
}

fn temporary_replay_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "gz-whittle-eval-replay-{}-{nonce}",
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
    Ok(std::env::args().nth(index).map(PathBuf::from))
}
