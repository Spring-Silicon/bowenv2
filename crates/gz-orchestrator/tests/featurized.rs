use gz_engine::{CandidateOptions, EngineResult, GraphEngine, ModelVersion};
use gz_engine_whittle::{
    WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleGraphId,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::{
    BackendOutputs, FeatureEvalBackend, ModelGeneration, PendingBatch, STUB_MODEL_VERSION,
    ServiceError, ServiceResult, StubBackend,
};
use gz_features::{
    FeatureBatchView, FeatureExtractor, FeatureResult, FeatureRow, FeatureSchema,
    FeatureSchemaConfig, PositionFeatures, decode_feature_row,
};
use gz_measurer::ValueTargetConfig;
use gz_orchestrator::reference::{
    ArenaRolloutClaim, EpisodeRolloutClaim, PolicyModel, Reference, ReferenceProvider,
    RolloutOutcome, RootBaselineProvider,
};
use gz_orchestrator::{
    CountedRoots, FeaturizedRuntime, ReplayRuntime, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayReferenceKind, ReplayStore, SampleConfig, SampleKind};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig, PolicyRolloutConfig};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Roots = CountedRoots<fn(&mut WhittleEngine) -> EngineResult<WhittleGraphId>>;

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gz-orchestrator-featurized-test-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();

        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn root_factory(engine: &mut WhittleEngine) -> EngineResult<WhittleGraphId> {
    Ok(engine.root())
}

fn roots(count: u64) -> Roots {
    CountedRoots::new(count, root_factory)
}

fn policy_rollout_config(search: &GumbelMcts) -> PolicyRolloutConfig {
    let config = search.config();
    PolicyRolloutConfig {
        max_steps: config.max_steps,
        seed: config.seed,
        export_position: config.export_position,
        mask_stop: config.mask_stop,
        no_backtrack: config.no_backtrack,
        candidate_options: config.candidate_options,
        measure_options: config.measure_options,
    }
}

struct FixedRoots {
    remaining: u64,
}

impl gz_orchestrator::RootSource<WhittleEngine> for FixedRoots {
    fn next_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(engine.root()))
    }

    fn fixed_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        Ok(Some(engine.root()))
    }
}

fn engines(count: usize) -> Vec<WhittleEngine> {
    (0..count)
        .map(|_| WhittleEngine::new(WhittleEngineConfig::default()).unwrap())
        .collect()
}

fn extractors(engines: &[WhittleEngine]) -> Vec<WhittleFeatureExtractor> {
    engines.iter().map(WhittleFeatureExtractor::new).collect()
}

fn search(engine: &WhittleEngine) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 2,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 11,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        value_mode: gz_search::GumbelValueMode::Competitive,
        candidate_options: CandidateOptions::default(),
        measure_options: engine.measure_options(),
    })
}

fn symmetric_search(engine: &WhittleEngine) -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 1,
        simulations: NonZeroUsize::new(2).unwrap(),
        max_considered_actions: NonZeroUsize::new(4).unwrap(),
        seed: 11,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: true,
        no_backtrack: false,
        value_mode: gz_search::GumbelValueMode::SymmetricSelfplay,
        candidate_options: CandidateOptions::default(),
        measure_options: engine.measure_options(),
    })
}

struct StopBackend;

impl FeatureEvalBackend for StopBackend {
    fn model_generation(&self) -> ModelGeneration {
        ModelGeneration::initial(STUB_MODEL_VERSION)
    }

    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        let mut outputs = StubBackend.eval(batch_bytes, action_counts)?;
        for row in &mut outputs.rows {
            row.policy_logits.fill(-100.0);
            *row.policy_logits.last_mut().expect("STOP action exists") = 100.0;
        }
        Ok(outputs)
    }
}

fn config(workers_per_lane: usize) -> ThreadedOrchestratorConfig {
    ThreadedOrchestratorConfig {
        workers_per_lane: NonZeroUsize::new(workers_per_lane).unwrap(),
        max_batch: NonZeroUsize::new(8).unwrap(),
        flush_after: Duration::from_millis(20),
        admission_stagger: Duration::ZERO,
        admission_smoothing: None,
    }
}

fn evaluator() -> RandomValueEvaluator {
    RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: 0,
        ..RandomValueEvaluatorConfig::default()
    })
    .unwrap()
}

#[test]
fn featurized_selfplay_is_deterministic() {
    let left = run_stub(2, 2, 3);
    let right = run_stub(2, 2, 3);

    assert_eq!(left, right);
    assert_eq!(left.lanes.len(), 2);
    assert!(left.lanes.iter().all(|lane| lane.episodes.len() == 3));
    assert!(!left.batch_sizes.is_empty());
}

#[test]
fn featurized_replay_appends_rows() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(2);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let feature_config = extractors[0].schema().config().clone();
    let providers = engines
        .iter()
        .map(|engine| RootBaselineProvider::new(engine.measure_options()))
        .collect::<Vec<_>>();
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2));
    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(2), roots(2)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_dropped, 0);
    assert_eq!(run.episodes_appended, 4);
    assert!(store.counters().produced_rows > 0);
    assert_eq!(
        store.feature_schema().unwrap(),
        Some(feature_config.clone())
    );

    let sample = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(store.counters().produced_rows as usize).unwrap(),
            window_rows: std::num::NonZeroU64::new(store.counters().produced_rows).unwrap(),
            seed: 0,
        })
        .unwrap();
    for (episode_id, row) in sample {
        let record = store.episode(episode_id).unwrap().unwrap();
        let reference = record.outcome.reference.as_ref().unwrap();
        let feature_row = decode_feature_row(row.feature_row.as_ref().unwrap()).unwrap();
        assert_eq!(feature_row.actions.len(), row.legal_actions.len());
        assert!(feature_row.position.opponent_present);
        assert_eq!(
            feature_row.position.opponent_reward,
            reference.reward / feature_config.opponent_reward_scale
        );
    }
}

#[test]
fn symmetric_selfplay_appends_both_players_without_stop_targets() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = symmetric_search(&engines[0]);
    let extractors = extractors(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));
    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![NoReferenceProvider],
                backpressure: None,
                length_tiebreak: true,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    assert_eq!(
        store.data_mode().unwrap(),
        gz_replay::ReplayDataMode::SymmetricSelfplay
    );
    assert_eq!(store.counters().produced_rows, 2);
    let rows = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(2).unwrap(),
            window_rows: std::num::NonZeroU64::new(2).unwrap(),
            seed: 0,
        })
        .unwrap();
    for (episode_id, row) in rows {
        let record = store.episode(episode_id).unwrap().unwrap();
        let feature_row = decode_feature_row(row.feature_row.as_ref().unwrap()).unwrap();
        assert_eq!(feature_row.actions.len(), row.legal_actions.len() + 1);
        assert_eq!(feature_row.actions.last().unwrap().kind_token, 1);
        assert!(feature_row.opponent.is_some());
        assert_eq!(row.value_target, Some(0.0));
        assert!(record.outcome.reference.is_none());
        assert!(!record.outcome.stopped);
    }
}

#[test]
fn symmetric_selfplay_trains_stop_and_keeps_rewrite_tiebreak_counts() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let mut search_config = symmetric_search(&engines[0]).config();
    search_config.max_steps = 4;
    search_config.mask_stop = false;
    search_config.tree_reuse = true;
    let search = GumbelMcts::new(search_config);
    let extractors = extractors(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StopBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![NoReferenceProvider],
                backpressure: None,
                length_tiebreak: true,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    assert_eq!(
        store.data_mode().unwrap(),
        gz_replay::ReplayDataMode::SymmetricSelfplayStop
    );
    assert_eq!(store.counters().produced_rows, 2);
    let rows = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(2).unwrap(),
            window_rows: std::num::NonZeroU64::new(2).unwrap(),
            seed: 0,
        })
        .unwrap();
    for (episode_id, row) in rows {
        let record = store.episode(episode_id).unwrap().unwrap();
        let feature_row = decode_feature_row(row.feature_row.as_ref().unwrap()).unwrap();
        assert!(record.outcome.stopped);
        assert!(matches!(
            record.steps.last().unwrap().action,
            gz_engine::PortableSearchActionRef::Stop { .. }
        ));
        assert_eq!(feature_row.actions.len(), row.legal_actions.len());
        assert_eq!(feature_row.actions.last().unwrap().kind_token, 1);
        assert!(matches!(
            row.legal_actions.last(),
            Some(gz_engine::PortableSearchActionRef::Stop { .. })
        ));
        assert_eq!(row.policy_target.len(), row.legal_actions.len());
        assert_eq!(row.value_target, Some(0.0));
        if episode_id.get() == 1 {
            let opponent = &feature_row.opponent.as_ref().unwrap().position;
            assert_eq!(opponent.root_step, 0);
            assert_eq!(opponent.budget_fraction, 1.0);
            assert!(opponent.budget_step.is_sign_negative());
        }
    }
    let metrics = store.symmetric_selfplay_metrics().unwrap();
    assert_eq!(metrics.p1_episode_len_ema, 0.0);
    assert_eq!(metrics.p2_episode_len_ema, 0.0);
}

#[test]
fn symmetric_wave_batching_submits_multiple_evals_from_one_worker() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let mut search_config = symmetric_search(&engines[0]).config();
    search_config.max_steps = 2;
    search_config.simulations = NonZeroUsize::new(8).unwrap();
    search_config.max_considered_actions = NonZeroUsize::new(2).unwrap();
    search_config.tree_reuse = true;
    let search = GumbelMcts::new(search_config).with_symmetric_wave_batching(true);
    let extractors = extractors(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![NoReferenceProvider],
                backpressure: None,
                length_tiebreak: true,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    assert!(run.batch_sizes.iter().copied().max().unwrap_or(0) > 1);
}

struct IdentifiedRootProvider {
    inner: RootBaselineProvider,
    ref_id: u64,
}

impl<E: GraphEngine> ReferenceProvider<E> for IdentifiedRootProvider {
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let mut reference = self.inner.reference(engine, root)?;
        if let Some(reference) = &mut reference {
            reference.ref_id = Some(self.ref_id);
        }
        Ok(reference)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let mut reference = self.inner.reference_with_features(
            engine,
            root,
            extractor,
            candidate_options,
            export_position,
        )?;
        if let Some(reference) = &mut reference {
            reference.ref_id = Some(self.ref_id);
        }
        Ok(reference)
    }
}

#[derive(Clone)]
struct CapturingBackend {
    opponent_refs: Arc<Mutex<Vec<(u64, u32)>>>,
}

impl FeatureEvalBackend for CapturingBackend {
    fn model_generation(&self) -> ModelGeneration {
        ModelGeneration::initial(STUB_MODEL_VERSION)
    }

    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        let view = FeatureBatchView::parse(batch_bytes)
            .map_err(|error| ServiceError::protocol(error.to_string()))?;
        self.opponent_refs.lock().unwrap().extend(
            view.opponent_trajectory_id
                .iter()
                .copied()
                .zip(view.opponent_row.iter().copied())
                .take(view.row_count as usize),
        );
        StubBackend.eval(batch_bytes, action_counts)
    }
}

#[test]
fn featurized_eval_batches_carry_stable_opponent_refs() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let provider = IdentifiedRootProvider {
        inner: RootBaselineProvider::new(engines[0].measure_options()),
        ref_id: 41,
    };
    let opponent_refs = Arc::new(Mutex::new(Vec::new()));
    let backend = CapturingBackend {
        opponent_refs: Arc::clone(&opponent_refs),
    };
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![backend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![provider],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    let opponent_refs = opponent_refs.lock().unwrap();
    assert!(!opponent_refs.is_empty());
    assert!(opponent_refs.iter().all(|reference| *reference == (41, 0)));
}

struct SampledTrajectoryProvider {
    next_ref_id: Arc<AtomicU64>,
    finished: Arc<AtomicU64>,
    observed_latest: Arc<Mutex<Vec<Option<ModelVersion>>>>,
}

impl<E: GraphEngine> ReferenceProvider<E> for SampledTrajectoryProvider {
    fn reference(&mut self, _engine: &mut E, _root: E::Graph) -> EngineResult<Option<Reference>> {
        panic!("sampled trajectory must not use the legacy reference path")
    }

    fn sampled_trajectory_mode(&self) -> bool {
        true
    }

    fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        self.observed_latest.lock().unwrap().push(latest);
        false
    }

    fn finish_sampled_trajectory(&mut self, outcome: Option<RolloutOutcome>) -> Option<Reference> {
        let outcome = outcome?;
        self.finished.fetch_add(1, Ordering::Relaxed);
        Some(Reference {
            ref_id: Some(self.next_ref_id.fetch_add(1, Ordering::Relaxed)),
            kind: ReplayReferenceKind::GatedPolicy,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: Some(outcome.search_config_hash),
            model_version: outcome.model_version,
        })
    }
}

#[derive(Clone)]
struct SwappingVersionBackend {
    calls: Arc<AtomicU64>,
    requested: Arc<Mutex<Vec<ModelGeneration>>>,
    released: Arc<Mutex<Vec<ModelGeneration>>>,
}

#[derive(Clone)]
struct VersionBackend {
    version: ModelVersion,
    rows: Arc<AtomicU64>,
}

impl FeatureEvalBackend for VersionBackend {
    fn model_generation(&self) -> ModelGeneration {
        ModelGeneration::initial(self.version)
    }

    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        let mut outputs = StubBackend.eval(batch_bytes, action_counts)?;
        self.rows
            .fetch_add(action_counts.len() as u64, Ordering::Relaxed);
        outputs.model_version = self.version;
        outputs.active_generation = self.model_generation();
        Ok(outputs)
    }
}

struct GeneratedArenaProvider {
    incumbent: ModelVersion,
    challenger: ModelVersion,
    arena_claimed: bool,
    arena_ready: bool,
    arena_root_reward: Option<f32>,
}

struct BatchedArenaProvider {
    incumbent: ModelVersion,
    challenger: ModelVersion,
    arena_size: usize,
    arena_claimed: usize,
    arena_finished: usize,
    claim_window: usize,
}

#[derive(Clone)]
struct BatchRecordingBackend {
    version: ModelVersion,
    batches: Arc<Mutex<Vec<usize>>>,
}

impl FeatureEvalBackend for BatchRecordingBackend {
    fn model_generation(&self) -> ModelGeneration {
        ModelGeneration::initial(self.version)
    }

    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        self.batches.lock().unwrap().push(action_counts.len());
        let mut outputs = StubBackend.eval(batch_bytes, action_counts)?;
        outputs.model_version = self.version;
        outputs.active_generation = self.model_generation();
        Ok(outputs)
    }
}

struct TrajectoryPoolProvider {
    incumbent: ModelVersion,
    claimed: bool,
    reference: Option<Reference>,
}

struct SampledTreeProvider;

impl<E: GraphEngine> ReferenceProvider<E> for SampledTreeProvider {
    fn reference(&mut self, _engine: &mut E, _root: E::Graph) -> EngineResult<Option<Reference>> {
        Ok(None)
    }

    fn sampled_tree_mode(&self) -> bool {
        true
    }
}

impl ReferenceProvider<WhittleEngine> for TrajectoryPoolProvider {
    fn reference(
        &mut self,
        _engine: &mut WhittleEngine,
        _root: WhittleGraphId,
    ) -> EngineResult<Option<Reference>> {
        Ok(self.reference.clone())
    }

    fn claim_sample_rollout(&mut self, _latest: Option<ModelVersion>) -> Option<ModelVersion> {
        if self.claimed || self.reference.is_some() {
            return None;
        }
        self.claimed = true;
        Some(self.incumbent)
    }

    fn finish_sample_rollout(&mut self, version: ModelVersion, outcome: Option<RolloutOutcome>) {
        let outcome = outcome.expect("trajectory-pool rollout must be measured");
        assert_eq!(version, self.incumbent);
        assert_eq!(outcome.model_version, Some(self.incumbent));
        self.reference = Some(Reference {
            ref_id: Some(73),
            kind: ReplayReferenceKind::GatedPolicy,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: Some(outcome.search_config_hash),
            model_version: Some(self.incumbent),
        });
    }

    fn admission_ready(&self) -> bool {
        self.reference.is_some()
    }
}

impl ReferenceProvider<WhittleEngine> for GeneratedArenaProvider {
    fn reference(
        &mut self,
        _engine: &mut WhittleEngine,
        _root: WhittleGraphId,
    ) -> EngineResult<Option<Reference>> {
        panic!("generated arena references must come from per-root policy rollouts")
    }

    fn claim_arena_rollout(
        &mut self,
        _latest: Option<ModelVersion>,
        _lane: usize,
        _lanes: usize,
    ) -> Option<ArenaRolloutClaim> {
        if self.arena_claimed || self.arena_ready {
            return None;
        }
        self.arena_claimed = true;
        Some(ArenaRolloutClaim {
            index: 0,
            version: self.challenger,
            model: PolicyModel::Challenger,
        })
    }

    fn arena_root(
        &mut self,
        engine: &mut WhittleEngine,
        index: usize,
    ) -> EngineResult<Option<WhittleGraphId>> {
        assert_eq!(index, 0);
        let root = engine.root();
        self.arena_root_reward = engine
            .measure(root, engine.measure_options())?
            .scalar_reward;
        Ok(Some(root))
    }

    fn finish_arena_rollout(
        &mut self,
        claim: ArenaRolloutClaim,
        score: Option<f32>,
        outcome: Option<RolloutOutcome>,
    ) {
        let outcome = outcome.expect("arena rollout must be measured");
        assert_eq!(claim.version, self.challenger);
        assert_eq!(outcome.model_version, Some(self.challenger));
        let root_reward = self.arena_root_reward.unwrap();
        let expected = (outcome.final_reward - root_reward) / root_reward.abs().max(1.0);
        assert!((score.unwrap() - expected).abs() < 1e-6);
        self.arena_ready = true;
    }

    fn per_root_policy_mode(&self) -> bool {
        true
    }

    fn claim_per_root_policy(
        &mut self,
        _latest: Option<ModelVersion>,
    ) -> Option<EpisodeRolloutClaim> {
        self.arena_ready.then_some(EpisodeRolloutClaim {
            version: self.incumbent,
            model: PolicyModel::Incumbent,
        })
    }

    fn finish_per_root_policy(
        &mut self,
        claim: EpisodeRolloutClaim,
        outcome: Option<RolloutOutcome>,
    ) -> Option<Reference> {
        let outcome = outcome?;
        if outcome.model_version != Some(claim.version) {
            return None;
        }
        Some(Reference {
            ref_id: Some(91),
            kind: ReplayReferenceKind::GatedPolicy,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: Some(outcome.search_config_hash),
            model_version: Some(claim.version),
        })
    }

    fn admission_ready(&self) -> bool {
        self.arena_ready
    }
}

impl ReferenceProvider<WhittleEngine> for BatchedArenaProvider {
    fn reference(
        &mut self,
        _engine: &mut WhittleEngine,
        _root: WhittleGraphId,
    ) -> EngineResult<Option<Reference>> {
        panic!("batched arena references must come from per-root policy rollouts")
    }

    fn claim_arena_rollout(
        &mut self,
        _latest: Option<ModelVersion>,
        _lane: usize,
        _lanes: usize,
    ) -> Option<ArenaRolloutClaim> {
        if self.arena_claimed >= self.arena_size
            || self.arena_claimed - self.arena_finished >= self.claim_window
        {
            return None;
        }
        let index = self.arena_claimed;
        self.arena_claimed += 1;
        Some(ArenaRolloutClaim {
            index,
            version: self.challenger,
            model: PolicyModel::Challenger,
        })
    }

    fn arena_parallelism(&self) -> usize {
        self.arena_size
    }

    fn arena_root(
        &mut self,
        engine: &mut WhittleEngine,
        index: usize,
    ) -> EngineResult<Option<WhittleGraphId>> {
        assert!(index < self.arena_size);
        Ok(Some(engine.root()))
    }

    fn finish_arena_rollout(
        &mut self,
        claim: ArenaRolloutClaim,
        score: Option<f32>,
        outcome: Option<RolloutOutcome>,
    ) {
        assert_eq!(claim.version, self.challenger);
        assert!(score.is_some());
        assert_eq!(outcome.unwrap().model_version, Some(self.challenger));
        self.arena_finished += 1;
    }

    fn per_root_policy_mode(&self) -> bool {
        true
    }

    fn claim_per_root_policy(
        &mut self,
        _latest: Option<ModelVersion>,
    ) -> Option<EpisodeRolloutClaim> {
        (self.arena_finished == self.arena_size).then_some(EpisodeRolloutClaim {
            version: self.incumbent,
            model: PolicyModel::Incumbent,
        })
    }

    fn finish_per_root_policy(
        &mut self,
        claim: EpisodeRolloutClaim,
        outcome: Option<RolloutOutcome>,
    ) -> Option<Reference> {
        let outcome = outcome?;
        (outcome.model_version == Some(claim.version)).then(|| Reference {
            ref_id: Some(92),
            kind: ReplayReferenceKind::GatedPolicy,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: Some(outcome.search_config_hash),
            model_version: Some(claim.version),
        })
    }

    fn admission_ready(&self) -> bool {
        self.arena_finished == self.arena_size
    }
}

#[test]
fn generated_arena_routes_incumbent_challenger_and_current_separately() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let mut search_config = search(&engines[0]).config();
    search_config.mask_stop = true;
    let search = GumbelMcts::new(search_config);
    let extractors = extractors(&engines);
    let current = ModelVersion::from_bytes([1; 16]);
    let incumbent = ModelVersion::from_bytes([2; 16]);
    let challenger = ModelVersion::from_bytes([3; 16]);
    let current_rows = Arc::new(AtomicU64::new(0));
    let incumbent_rows = Arc::new(AtomicU64::new(0));
    let challenger_rows = Arc::new(AtomicU64::new(0));
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2));

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![VersionBackend {
                    version: current,
                    rows: Arc::clone(&current_rows),
                }],
                reference_backends: vec![VersionBackend {
                    version: incumbent,
                    rows: Arc::clone(&incumbent_rows),
                }],
                challenger_backends: vec![VersionBackend {
                    version: challenger,
                    rows: Arc::clone(&challenger_rows),
                }],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![GeneratedArenaProvider {
                    incumbent,
                    challenger,
                    arena_claimed: false,
                    arena_ready: false,
                    arena_root_reward: None,
                }],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    assert!(current_rows.load(Ordering::Relaxed) > 0);
    assert!(incumbent_rows.load(Ordering::Relaxed) > 0);
    assert!(challenger_rows.load(Ordering::Relaxed) > 0);
    let episode = store
        .episode(gz_replay::ReplayEpisodeId::new(0))
        .unwrap()
        .unwrap();
    assert_eq!(
        episode.outcome.reference.unwrap().model_version,
        Some(incumbent)
    );
    let rows = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(episode.row_count as usize).unwrap(),
            window_rows: std::num::NonZeroU64::new(episode.row_count.into()).unwrap(),
            seed: 0,
        })
        .unwrap();
    assert!(
        rows.iter()
            .all(|(_, row)| row.model_version == Some(current))
    );
}

#[test]
fn generated_arena_batching_reduces_evaluator_launches() {
    let serial = run_generated_arena_batching_case(1);
    let batched = run_generated_arena_batching_case(8);

    assert!(serial.challenger.iter().all(|&batch| batch == 1));
    assert!(batched.challenger.iter().all(|&batch| batch == 8));
    assert_eq!(
        serial.challenger.iter().sum::<usize>(),
        batched.challenger.iter().sum::<usize>()
    );
    assert_eq!(serial.challenger.len(), batched.challenger.len() * 8);
    assert!(serial.current.iter().all(|&batch| batch <= 2));
    assert!(batched.current.iter().all(|&batch| batch <= 2));
}

struct ArenaBatchingRun {
    current: Vec<usize>,
    challenger: Vec<usize>,
}

fn run_generated_arena_batching_case(claim_window: usize) -> ArenaBatchingRun {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let current = ModelVersion::from_bytes([1; 16]);
    let incumbent = ModelVersion::from_bytes([2; 16]);
    let challenger = ModelVersion::from_bytes([3; 16]);
    let current_batches = Arc::new(Mutex::new(Vec::new()));
    let challenger_batches = Arc::new(Mutex::new(Vec::new()));
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2));

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(8)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![BatchRecordingBackend {
                    version: current,
                    batches: Arc::clone(&current_batches),
                }],
                reference_backends: vec![BatchRecordingBackend {
                    version: incumbent,
                    batches: Arc::new(Mutex::new(Vec::new())),
                }],
                challenger_backends: vec![BatchRecordingBackend {
                    version: challenger,
                    batches: Arc::clone(&challenger_batches),
                }],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![BatchedArenaProvider {
                    incumbent,
                    challenger,
                    arena_size: 8,
                    arena_claimed: 0,
                    arena_finished: 0,
                    claim_window,
                }],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 8);
    ArenaBatchingRun {
        current: Arc::try_unwrap(current_batches)
            .unwrap()
            .into_inner()
            .unwrap(),
        challenger: Arc::try_unwrap(challenger_batches)
            .unwrap()
            .into_inner()
            .unwrap(),
    }
}

#[test]
fn trajectory_pool_routes_incumbent_rollout_and_current_learner_separately() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let current = ModelVersion::from_bytes([1; 16]);
    let incumbent = ModelVersion::from_bytes([2; 16]);
    let current_rows = Arc::new(AtomicU64::new(0));
    let incumbent_rows = Arc::new(AtomicU64::new(0));
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let run = orchestrator
        .run_featurized_with_replay(
            vec![FixedRoots { remaining: 1 }],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![VersionBackend {
                    version: current,
                    rows: Arc::clone(&current_rows),
                }],
                reference_backends: vec![VersionBackend {
                    version: incumbent,
                    rows: Arc::clone(&incumbent_rows),
                }],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![TrajectoryPoolProvider {
                    incumbent,
                    claimed: false,
                    reference: None,
                }],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    assert!(current_rows.load(Ordering::Relaxed) > 0);
    assert!(incumbent_rows.load(Ordering::Relaxed) > 0);
    let episode = store
        .episode(gz_replay::ReplayEpisodeId::new(0))
        .unwrap()
        .unwrap();
    assert_eq!(
        episode.outcome.reference.unwrap().model_version,
        Some(incumbent)
    );
}

#[test]
fn sampled_tree_routes_models_and_appends_learner_pairs() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let current = ModelVersion::from_bytes([1; 16]);
    let incumbent = ModelVersion::from_bytes([2; 16]);
    let current_rows = Arc::new(AtomicU64::new(0));
    let incumbent_rows = Arc::new(AtomicU64::new(0));
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![VersionBackend {
                    version: current,
                    rows: Arc::clone(&current_rows),
                }],
                reference_backends: vec![VersionBackend {
                    version: incumbent,
                    rows: Arc::clone(&incumbent_rows),
                }],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![SampledTreeProvider],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    assert_eq!(store.episode_counters().0, 1);
    assert!(current_rows.load(Ordering::Relaxed) > 0);
    assert!(incumbent_rows.load(Ordering::Relaxed) > 0);
    let learner = store
        .episode(gz_replay::ReplayEpisodeId::new(0))
        .unwrap()
        .unwrap();
    assert!(
        store
            .episode(gz_replay::ReplayEpisodeId::new(1))
            .unwrap()
            .is_none()
    );
    assert_eq!(
        learner.outcome.reference.as_ref().unwrap().kind,
        ReplayReferenceKind::GatedPolicy
    );
    assert_eq!(
        learner.outcome.reference.as_ref().unwrap().model_version,
        Some(incumbent)
    );
    assert_eq!(store.counters().produced_rows, u64::from(learner.row_count));
    assert_eq!(
        store.counters().produced_policy_rows,
        u64::from(learner.row_count)
    );
    assert_eq!(
        store.counters().produced_value_rows,
        u64::from(learner.row_count)
    );
    let (early_value_accuracy, late_value_accuracy) = store.value_sign_accuracy_emas();
    assert!(early_value_accuracy.is_some_and(|value| (0.0..=1.0).contains(&value)));
    assert!(late_value_accuracy.is_none());

    let window = std::num::NonZeroU64::new(store.counters().produced_rows).unwrap();
    let policy = store
        .sample_rows_kind(
            SampleConfig {
                batch: NonZeroUsize::new(16).unwrap(),
                window_rows: window,
                seed: 3,
            },
            SampleKind::Policy,
        )
        .unwrap();
    assert!(policy.iter().all(|(id, row)| {
        *id == gz_replay::ReplayEpisodeId::new(0)
            && row.model_version == Some(current)
            && row.policy_target.iter().any(|target| *target > 0.0)
    }));
    let mut opponent_steps = vec![None; learner.row_count as usize];
    for seed in 0..64 {
        let values = store
            .sample_rows_kind(
                SampleConfig {
                    batch: NonZeroUsize::new(16).unwrap(),
                    window_rows: window,
                    seed,
                },
                SampleKind::Value,
            )
            .unwrap();
        for (id, row) in values {
            assert_eq!(id, gz_replay::ReplayEpisodeId::new(0));
            let features = decode_feature_row(row.feature_row.as_ref().unwrap()).unwrap();
            assert!(row.value_target.is_some());
            assert!(features.position.opponent_present);
            let opponent = features.opponent.unwrap();
            opponent_steps[row.step_index as usize] = Some(opponent.position.root_step);
        }
        if opponent_steps.iter().all(Option::is_some) {
            break;
        }
    }
    assert!(opponent_steps.iter().all(Option::is_some));
}

impl FeatureEvalBackend for SwappingVersionBackend {
    fn model_generation(&self) -> ModelGeneration {
        ModelGeneration::initial(ModelVersion::from_bytes([1; 16]))
    }

    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        let mut outputs = StubBackend.eval(batch_bytes, action_counts)?;
        outputs.model_version = self.model_generation().version;
        outputs.active_generation = self.model_generation();
        Ok(outputs)
    }

    fn submit_for_model(
        &mut self,
        model: ModelGeneration,
        batch_bytes: &[u8],
        action_counts: &[u32],
    ) -> ServiceResult<PendingBatch> {
        let mut outputs = StubBackend.eval(batch_bytes, action_counts)?;
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.requested.lock().unwrap().push(model);
        outputs.model_version = model.version;
        outputs.active_generation = ModelGeneration {
            id: 2,
            version: ModelVersion::from_bytes([2; 16]),
        };
        Ok(PendingBatch::Ready(outputs))
    }

    fn release_model_generation(&mut self, model: ModelGeneration) -> ServiceResult<()> {
        self.released.lock().unwrap().push(model);
        Ok(())
    }
}

#[test]
fn sampled_trajectory_runs_one_active_policy_prelude_per_learner_episode() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let finished = Arc::new(AtomicU64::new(0));
    let provider = SampledTrajectoryProvider {
        next_ref_id: Arc::new(AtomicU64::new(1)),
        finished: Arc::clone(&finished),
        observed_latest: Arc::new(Mutex::new(Vec::new())),
    };
    let policy_rollout = policy_rollout_config(&search);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(2))
        .with_policy_rollout(policy_rollout);

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(3)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![provider],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 3);
    assert_eq!(finished.load(Ordering::Relaxed), 3);
    let mut ref_ids = Vec::new();
    for replay_id in 0..run.episodes_appended {
        let episode = store
            .episode(gz_replay::ReplayEpisodeId::new(replay_id))
            .unwrap()
            .unwrap();
        let reference = episode.outcome.reference.unwrap();
        ref_ids.push(reference.trajectory_id.unwrap());
        assert_eq!(reference.model_version, Some(STUB_MODEL_VERSION));
    }
    ref_ids.sort_unstable();
    ref_ids.dedup();
    assert_eq!(ref_ids.len(), 3);
}

#[test]
fn sampled_trajectory_pins_each_episode_and_new_episodes_adopt_swaps() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let mut search_config = search(&engines[0]).config();
    search_config.max_steps = 3;
    search_config.mask_stop = true;
    let search = GumbelMcts::new(search_config);
    let extractors = extractors(&engines);
    let observed_latest = Arc::new(Mutex::new(Vec::new()));
    let provider = SampledTrajectoryProvider {
        next_ref_id: Arc::new(AtomicU64::new(1)),
        finished: Arc::new(AtomicU64::new(0)),
        observed_latest: Arc::clone(&observed_latest),
    };
    let calls = Arc::new(AtomicU64::new(0));
    let requested = Arc::new(Mutex::new(Vec::new()));
    let released = Arc::new(Mutex::new(Vec::new()));
    let backend = SwappingVersionBackend {
        calls: Arc::clone(&calls),
        requested: Arc::clone(&requested),
        released: Arc::clone(&released),
    };
    let policy_rollout = policy_rollout_config(&search);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1))
        .with_policy_rollout(policy_rollout);

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(2)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![backend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![provider],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 2);
    let first_episode = store
        .episode(gz_replay::ReplayEpisodeId::new(0))
        .unwrap()
        .unwrap();
    let first = ModelGeneration::initial(ModelVersion::from_bytes([1; 16]));
    let second = ModelGeneration {
        id: 2,
        version: ModelVersion::from_bytes([2; 16]),
    };
    assert_eq!(
        first_episode.outcome.reference.unwrap().model_version,
        Some(first.version)
    );
    let second_episode = store
        .episode(gz_replay::ReplayEpisodeId::new(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        second_episode.outcome.reference.unwrap().model_version,
        Some(second.version)
    );
    assert!(calls.load(Ordering::Relaxed) > 1);
    let requested = requested.lock().unwrap();
    let second_start = requested
        .iter()
        .position(|model| *model == second)
        .expect("second episode never adopted the new model generation");
    assert!(second_start > 1, "first episode did not retain its lease");
    assert!(
        requested[..second_start]
            .iter()
            .all(|model| *model == first)
    );
    assert!(
        requested[second_start..]
            .iter()
            .all(|model| *model == second)
    );
    let observed_latest = observed_latest.lock().unwrap();
    assert!(observed_latest.contains(&Some(second.version)));
    assert!(!observed_latest.contains(&Some(first.version)));
    assert_eq!(*released.lock().unwrap(), vec![first]);
}

#[test]
fn symmetric_selfplay_pins_both_players_and_next_game_adopts_swap() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let mut search_config = symmetric_search(&engines[0]).config();
    search_config.max_steps = 2;
    let search = GumbelMcts::new(search_config);
    let extractors = extractors(&engines);
    let calls = Arc::new(AtomicU64::new(0));
    let requested = Arc::new(Mutex::new(Vec::new()));
    let released = Arc::new(Mutex::new(Vec::new()));
    let backend = SwappingVersionBackend {
        calls: Arc::clone(&calls),
        requested: Arc::clone(&requested),
        released: Arc::clone(&released),
    };
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(2)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![backend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![NoReferenceProvider],
                backpressure: None,
                length_tiebreak: true,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 2);
    let first = ModelGeneration::initial(ModelVersion::from_bytes([1; 16]));
    let second = ModelGeneration {
        id: 2,
        version: ModelVersion::from_bytes([2; 16]),
    };
    for replay_id in 0..4 {
        let record = store
            .episode(gz_replay::ReplayEpisodeId::new(replay_id))
            .unwrap()
            .unwrap();
        assert!(!record.steps.is_empty());
    }
    let requested = requested.lock().unwrap();
    let second_start = requested
        .iter()
        .position(|model| *model == second)
        .expect("second game never adopted the new model generation");
    assert!(second_start > 1, "first game did not retain its lease");
    assert!(
        requested[..second_start]
            .iter()
            .all(|model| *model == first)
    );
    assert!(
        requested[second_start..]
            .iter()
            .all(|model| *model == second)
    );
    assert_eq!(*released.lock().unwrap(), vec![first]);
}

/// Never supplies a reference and never expects one: rows are stored
/// unlabeled instead of dropped (the reference=none pipeline shape).
struct NoReferenceProvider;

impl<E: GraphEngine> ReferenceProvider<E> for NoReferenceProvider {
    fn reference(
        &mut self,
        _engine: &mut E,
        _root: E::Graph,
    ) -> EngineResult<Option<gz_orchestrator::reference::Reference>> {
        Ok(None)
    }

    fn expects_reference(&self) -> bool {
        false
    }
}

#[test]
fn featurized_replay_unlabeled_rows_have_no_opponent_scalar() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));
    let run = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers: vec![NoReferenceProvider],
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap();

    assert_eq!(run.episodes_appended, 1);
    let sample = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(store.counters().produced_rows as usize).unwrap(),
            window_rows: std::num::NonZeroU64::new(store.counters().produced_rows).unwrap(),
            seed: 0,
        })
        .unwrap();
    for (episode_id, row) in sample {
        let record = store.episode(episode_id).unwrap().unwrap();
        let feature_row = decode_feature_row(row.feature_row.as_ref().unwrap()).unwrap();
        assert!(record.outcome.reference.is_none());
        assert!(!feature_row.position.opponent_present);
        assert_eq!(feature_row.position.opponent_reward, 0.0);
    }
}

#[test]
fn featurized_replay_schema_error_includes_replay_detail() {
    let dir = TestDir::new();
    let store = ReplayStore::open(dir.path()).unwrap();
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let mut stored_config = extractors[0].schema().config().clone();
    stored_config.name = "stored-mismatch".to_owned();
    store.ensure_feature_schema(&stored_config).unwrap();
    let providers = engines
        .iter()
        .map(|engine| RootBaselineProvider::new(engine.measure_options()))
        .collect::<Vec<_>>();
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let error = orchestrator
        .run_featurized_with_replay(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: None,
                length_tiebreak: false,
                value_target: ValueTargetConfig::Sign,
            },
        )
        .unwrap_err();

    assert!(error.to_string().contains("invalid replay record"));
}

#[test]
fn featurized_rejects_lane_and_schema_mismatches() {
    let engine_set = engines(2);
    let gumbel = search(&engine_set[0]);
    let mut extractor_set = extractors(&engine_set);
    extractor_set.pop();
    let orchestrator = ThreadedGumbelOrchestrator::new(engine_set, evaluator(), gumbel, config(2));
    let error = orchestrator
        .run_featurized(
            vec![roots(1), roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors: extractor_set,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("lane count mismatch"));

    let engine_set = engines(2);
    let gumbel = search(&engine_set[0]);
    let mut extractor_set = extractors(&engine_set);
    let schema = FeatureSchema::new(FeatureSchemaConfig {
        name: "mismatch".to_owned(),
        ..extractor_set[0].schema().config().clone()
    })
    .unwrap();
    let wrapped = vec![
        WrappedExtractor::matching(extractor_set.remove(0)),
        WrappedExtractor::with_schema(extractor_set.remove(0), schema),
    ];
    let orchestrator = ThreadedGumbelOrchestrator::new(engine_set, evaluator(), gumbel, config(2));
    let error = orchestrator
        .run_featurized(
            vec![roots(1), roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors: wrapped,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("feature schema mismatch"));
}

#[test]
fn featurized_extraction_failure_aborts_run() {
    let engines = engines(1);
    let search = search(&engines[0]);
    let extractors = vec![FailingExtractor {
        inner: WhittleFeatureExtractor::new(&engines[0]),
    }];
    let orchestrator = ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(1));

    let error = orchestrator
        .run_featurized(
            vec![roots(1)],
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
        )
        .unwrap_err();

    assert!(error.to_string().contains("feature extraction failed"));
}

fn run_stub(
    lanes: usize,
    workers_per_lane: usize,
    roots_per_lane: u64,
) -> gz_orchestrator::ThreadedRun<WhittleGraphId, WhittleCandidateId> {
    let engines = engines(lanes);
    let search = search(&engines[0]);
    let extractors = extractors(&engines);
    let orchestrator =
        ThreadedGumbelOrchestrator::new(engines, evaluator(), search, config(workers_per_lane));
    orchestrator
        .run_featurized(
            (0..lanes).map(|_| roots(roots_per_lane)).collect(),
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
        )
        .unwrap()
}

struct WrappedExtractor {
    inner: WhittleFeatureExtractor,
    schema: FeatureSchema,
}

impl WrappedExtractor {
    fn matching(inner: WhittleFeatureExtractor) -> Self {
        let schema = inner.schema().clone();
        Self { inner, schema }
    }

    fn with_schema(inner: WhittleFeatureExtractor, schema: FeatureSchema) -> Self {
        Self { inner, schema }
    }
}

impl FeatureExtractor<WhittleEngine> for WrappedExtractor {
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn extract(
        &mut self,
        engine: &WhittleEngine,
        graph: WhittleGraphId,
        candidates: &[WhittleCandidateId],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow> {
        self.inner.extract(engine, graph, candidates, position)
    }
}

struct FailingExtractor {
    inner: WhittleFeatureExtractor,
}

impl FeatureExtractor<WhittleEngine> for FailingExtractor {
    fn schema(&self) -> &FeatureSchema {
        self.inner.schema()
    }

    fn extract(
        &mut self,
        _engine: &WhittleEngine,
        _graph: WhittleGraphId,
        _candidates: &[WhittleCandidateId],
        _position: PositionFeatures,
    ) -> FeatureResult<FeatureRow> {
        Err(gz_features::FeatureError::InvalidRow("forced failure"))
    }
}
