use gz_engine::{CandidateOptions, EngineResult, GraphEngine, ModelVersion};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleGraphGenerator,
    WhittleGraphGeneratorConfig, WhittleGraphId, WhittleRoot,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::{
    EvaluatorProcess, EvaluatorProcessConfig, Hello, STUB_MODEL_VERSION, StubBackend,
};
use gz_orchestrator::reference::{
    BeamReferenceProvider, GreedyReferenceProvider, RandomReferenceProvider, Reference,
    ReferenceProvider, RootBaselineProvider,
};
use gz_orchestrator::{
    FeaturizedRuntime, ReplayRuntime, RootSource, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayCounters, ReplayEpisodeId, ReplayStore};
use gz_search::{
    BeamSearch, BeamSearchConfig, GreedySearch, GreedySearchConfig, GumbelEpisodeContext,
    GumbelMcts, GumbelMctsConfig, RandomSearch, RandomSearchConfig,
};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

const WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES: usize = 255;

#[derive(Clone, Debug)]
pub struct SelfplayConfig {
    pub replay_dir: Option<PathBuf>,
    pub episodes: u64,
    pub lanes: usize,
    pub workers_per_lane: usize,
    pub reference: ReferenceMode,
    pub seed: u64,
    pub max_steps: usize,
    pub simulations: usize,
    pub max_batch: usize,
    pub evaluator: EvaluatorMode,
    pub python_dir: Option<PathBuf>,
}

impl Default for SelfplayConfig {
    fn default() -> Self {
        Self {
            replay_dir: None,
            episodes: 16,
            lanes: 2,
            workers_per_lane: 8,
            reference: ReferenceMode::Root,
            seed: 0,
            max_steps: 8,
            simulations: 8,
            max_batch: 16,
            evaluator: EvaluatorMode::Random,
            python_dir: None,
        }
    }
}

impl SelfplayConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.is_none() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.lanes == 0 {
            return Err("--lanes must be greater than zero".to_owned());
        }
        if self.workers_per_lane == 0 {
            return Err("--workers-per-lane must be greater than zero".to_owned());
        }
        if self.max_steps == 0 {
            return Err("--max-steps must be greater than zero".to_owned());
        }
        if self.simulations == 0 {
            return Err("--simulations must be greater than zero".to_owned());
        }
        if self.max_batch == 0 {
            return Err("--max-batch must be greater than zero".to_owned());
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceMode {
    None,
    Root,
    Greedy,
    Beam,
    Random,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluatorMode {
    Random,
    Stub,
    ProcessStub,
}

impl EvaluatorMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Stub => "stub",
            Self::ProcessStub => "process-stub",
        }
    }
}

impl FromStr for EvaluatorMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "random" => Ok(Self::Random),
            "stub" => Ok(Self::Stub),
            "process-stub" => Ok(Self::ProcessStub),
            _ => Err(format!("unknown evaluator: {value}")),
        }
    }
}

impl FromStr for ReferenceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "root" => Ok(Self::Root),
            "greedy" => Ok(Self::Greedy),
            "beam" => Ok(Self::Beam),
            "random" => Ok(Self::Random),
            _ => Err(format!("unknown reference: {value}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelfplaySummary {
    pub evaluator: EvaluatorMode,
    pub model_version: Option<ModelVersion>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub rows_produced: u64,
    pub wins: u64,
    pub losses: u64,
    pub ties: u64,
    pub eval_batch_count: usize,
    pub mean_eval_batch_size: f64,
    pub counters: ReplayCounters,
}

pub fn run(config: SelfplayConfig) -> Result<SelfplaySummary, String> {
    config.validate()?;

    let replay_dir = config
        .replay_dir
        .as_ref()
        .expect("validated replay_dir exists");
    let store = ReplayStore::open(replay_dir).map_err(|error| error.to_string())?;
    let engines = (0..config.lanes)
        .map(|_| WhittleEngine::new(whittle_engine_config()).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let search = search(&engines[0], &config)?;
    let roots = root_sources(&config);
    let providers = engines
        .iter()
        .enumerate()
        .map(|(lane, engine)| provider(engine, &config, lane))
        .collect::<Result<Vec<_>, _>>()?;

    match config.evaluator {
        EvaluatorMode::Random => run_random(config, store, engines, search, roots, providers),
        EvaluatorMode::Stub => run_stub(config, store, engines, search, roots, providers),
        EvaluatorMode::ProcessStub => {
            run_process_stub(config, store, engines, search, roots, providers)
        }
    }
}

fn run_random(
    config: SelfplayConfig,
    store: ReplayStore,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<GeneratedRoots>,
    providers: Vec<CliReferenceProvider>,
) -> Result<SelfplaySummary, String> {
    let evaluator = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: config.seed,
        ..RandomValueEvaluatorConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        evaluator,
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: nonzero(config.workers_per_lane, "workers_per_lane")?,
            max_batch: nonzero(config.max_batch, "max_batch")?,
            flush_after: Duration::from_millis(1),
        },
    );
    let run = orchestrator
        .run_with_replay(
            roots,
            GumbelEpisodeContext::default(),
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .map_err(|error| error.to_string())?;

    summarize(&store, run, EvaluatorMode::Random, None)
}

fn run_stub(
    config: SelfplayConfig,
    store: ReplayStore,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<GeneratedRoots>,
    providers: Vec<CliReferenceProvider>,
) -> Result<SelfplaySummary, String> {
    let extractors = engines
        .iter()
        .map(WhittleFeatureExtractor::new)
        .collect::<Vec<_>>();
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        random_placeholder(&config)?,
        search,
        threaded_config(&config)?,
    );
    let run = orchestrator
        .run_featurized_with_replay(
            roots,
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backend: StubBackend,
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .map_err(|error| error.to_string())?;

    summarize(&store, run, EvaluatorMode::Stub, Some(STUB_MODEL_VERSION))
}

fn run_process_stub(
    config: SelfplayConfig,
    store: ReplayStore,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<GeneratedRoots>,
    providers: Vec<CliReferenceProvider>,
) -> Result<SelfplaySummary, String> {
    let extractors = engines
        .iter()
        .map(WhittleFeatureExtractor::new)
        .collect::<Vec<_>>();
    let mut process = EvaluatorProcess::spawn(EvaluatorProcessConfig {
        working_dir: config
            .python_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("python")),
        socket_path: process_socket_path(),
        ready_timeout: Duration::from_secs(10),
        io_timeout: Duration::from_secs(30),
        ..EvaluatorProcessConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let hello = Hello::new(
        extractors
            .first()
            .ok_or_else(|| "missing feature extractor".to_owned())?
            .schema()
            .hash(),
        config.max_batch as u32,
        engines[0].engine_id(),
        engines[0].engine_version(),
        engines[0].action_set_hash(),
    );
    let backend = process.connect(&hello).map_err(|error| error.to_string())?;
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        random_placeholder(&config)?,
        search,
        threaded_config(&config)?,
    );
    let run = orchestrator
        .run_featurized_with_replay(
            roots,
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backend,
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
            },
        )
        .map_err(|error| error.to_string())?;
    wait_for_process_exit(&mut process)?;

    summarize(
        &store,
        run,
        EvaluatorMode::ProcessStub,
        Some(STUB_MODEL_VERSION),
    )
}

fn summarize(
    store: &ReplayStore,
    run: gz_orchestrator::ThreadedReplayRun<WhittleGraphId, gz_engine_whittle::WhittleCandidateId>,
    evaluator: EvaluatorMode,
    model_version: Option<ModelVersion>,
) -> Result<SelfplaySummary, String> {
    let counters = store.counters();
    let (wins, losses, ties) = label_counts(store, run.episodes_appended)?;
    let evals = run.run.batch_sizes.iter().sum::<usize>();
    let mean_eval_batch_size = if run.run.batch_sizes.is_empty() {
        0.0
    } else {
        evals as f64 / run.run.batch_sizes.len() as f64
    };

    Ok(SelfplaySummary {
        evaluator,
        model_version,
        episodes_appended: run.episodes_appended,
        episodes_dropped: run.episodes_dropped,
        rows_produced: counters.produced_rows,
        wins,
        losses,
        ties,
        eval_batch_count: run.run.batch_sizes.len(),
        mean_eval_batch_size,
        counters,
    })
}

fn threaded_config(config: &SelfplayConfig) -> Result<ThreadedOrchestratorConfig, String> {
    Ok(ThreadedOrchestratorConfig {
        workers_per_lane: nonzero(config.workers_per_lane, "workers_per_lane")?,
        max_batch: nonzero(config.max_batch, "max_batch")?,
        flush_after: Duration::from_millis(1),
    })
}

fn random_placeholder(config: &SelfplayConfig) -> Result<RandomValueEvaluator, String> {
    RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: config.seed,
        ..RandomValueEvaluatorConfig::default()
    })
    .map_err(|error| error.to_string())
}

fn process_socket_path() -> PathBuf {
    std::env::temp_dir().join(format!("gz-process-stub-{}.sock", std::process::id()))
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait().map_err(|error| error.to_string())? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("Python evaluator exited with {status}")),
            None if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            None => return Err("Python evaluator did not exit".to_owned()),
        }
    }
}

fn search(engine: &WhittleEngine, config: &SelfplayConfig) -> Result<GumbelMcts, String> {
    Ok(GumbelMcts::new(GumbelMctsConfig {
        max_steps: config.max_steps,
        simulations: nonzero(config.simulations, "simulations")?,
        max_considered_actions: NonZeroUsize::new(16).unwrap(),
        seed: config.seed,
        gumbel_scale: 0.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        candidate_options: match config.evaluator {
            EvaluatorMode::Random => CandidateOptions::default(),
            EvaluatorMode::Stub | EvaluatorMode::ProcessStub => feature_candidate_options(),
        },
        measure_options: engine.measure_options(),
    }))
}

fn feature_candidate_options() -> CandidateOptions {
    CandidateOptions {
        max_candidates: Some(WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES),
        deterministic_order: true,
    }
}

fn provider(
    engine: &WhittleEngine,
    config: &SelfplayConfig,
    lane: usize,
) -> Result<CliReferenceProvider, String> {
    let measure_options = engine.measure_options();
    let provider = match config.reference {
        ReferenceMode::None => CliReferenceProvider::None,
        ReferenceMode::Root => {
            CliReferenceProvider::Root(RootBaselineProvider::new(measure_options))
        }
        ReferenceMode::Greedy => CliReferenceProvider::Greedy(GreedyReferenceProvider::new(
            GreedySearch::new(GreedySearchConfig {
                max_steps: config.max_steps,
                candidate_options: CandidateOptions::default(),
                measure_options,
            }),
        )),
        ReferenceMode::Beam => CliReferenceProvider::Beam(BeamReferenceProvider::new(
            BeamSearch::new(BeamSearchConfig {
                max_depth: config.max_steps,
                beam_width: NonZeroUsize::new(4).unwrap(),
                candidate_options: CandidateOptions::default(),
                measure_options,
            }),
        )),
        ReferenceMode::Random => CliReferenceProvider::Random(RandomReferenceProvider::new(
            RandomSearch::new(RandomSearchConfig {
                max_steps: config.max_steps,
                seed: config.seed ^ ((lane as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                candidate_options: CandidateOptions::default(),
                measure_options,
            }),
        )),
    };

    Ok(provider)
}

fn root_sources(config: &SelfplayConfig) -> Vec<GeneratedRoots> {
    let base = config.episodes / config.lanes as u64;
    let extra = config.episodes % config.lanes as u64;

    (0..config.lanes)
        .map(|lane| {
            let count = base + u64::from((lane as u64) < extra);
            GeneratedRoots {
                remaining: count,
                generator: WhittleGraphGenerator::from_seed(
                    whittle_generator_config(),
                    config.seed ^ ((lane as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03)),
                ),
            }
        })
        .collect()
}

fn whittle_engine_config() -> WhittleEngineConfig {
    let generator = whittle_generator_config();
    WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator.arity,
            capacity: generator.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    }
}

fn whittle_generator_config() -> WhittleGraphGeneratorConfig {
    WhittleGraphGeneratorConfig::default()
}

fn label_counts(store: &ReplayStore, episodes: u64) -> Result<(u64, u64, u64), String> {
    let mut wins = 0;
    let mut losses = 0;
    let mut ties = 0;

    for id in 0..episodes {
        let Some(record) = store
            .episode(ReplayEpisodeId::new(id))
            .map_err(|error| error.to_string())?
        else {
            continue;
        };

        match record.outcome.value_target {
            Some(1.0) => wins += 1,
            Some(-1.0) => losses += 1,
            Some(0.0) => ties += 1,
            _ => {}
        }
    }

    Ok((wins, losses, ties))
}

fn nonzero(value: usize, name: &str) -> Result<NonZeroUsize, String> {
    NonZeroUsize::new(value).ok_or_else(|| format!("{name} must be greater than zero"))
}

struct GeneratedRoots {
    remaining: u64,
    generator: WhittleGraphGenerator,
}

impl RootSource<WhittleEngine> for GeneratedRoots {
    fn next_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        if self.remaining == 0 {
            return Ok(None);
        }

        self.remaining -= 1;
        self.generator
            .sample_into(engine)
            .map(|generated| Some(generated.graph))
    }
}

enum CliReferenceProvider {
    None,
    Root(RootBaselineProvider),
    Greedy(GreedyReferenceProvider),
    Beam(BeamReferenceProvider),
    Random(RandomReferenceProvider),
}

impl ReferenceProvider<WhittleEngine> for CliReferenceProvider {
    fn reference(
        &mut self,
        engine: &mut WhittleEngine,
        root: WhittleGraphId,
    ) -> EngineResult<Option<Reference<WhittleGraphId>>> {
        match self {
            Self::None => Ok(None),
            Self::Root(provider) => provider.reference(engine, root),
            Self::Greedy(provider) => provider.reference(engine, root),
            Self::Beam(provider) => provider.reference(engine, root),
            Self::Random(provider) => provider.reference(engine, root),
        }
    }
}
