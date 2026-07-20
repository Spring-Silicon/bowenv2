use gz_engine::{
    CandidateOptions, EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine,
    MeasureOptions, ModelVersion, PortableGraphId, ReplayGraphContext, SearchConfigHash,
};
use gz_features::{FeatureExtractor, FeatureRow, OpponentStateFeatures, PositionFeatures};
use gz_measurer::{ArenaGateRegistry, ReferenceRegistry, ReferenceSnapshot};
pub use gz_measurer::{
    ArenaRolloutClaim, EpisodeRolloutClaim, PolicyModel, ReferenceStep, RolloutOutcome,
};
use gz_replay::ReplayReferenceKind;
use gz_search::{BeamSearch, GreedySearch, RandomSearch, SearchStep};
use std::sync::Arc;

pub trait ReferenceProvider<E: GraphEngine> {
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>>;

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        Self: Sized,
        X: FeatureExtractor<E>,
    {
        let _ = (extractor, candidate_options, export_position);
        self.reference(engine, root)
    }

    /// Called by the replay drivers for every replay-eligible completed
    /// episode with the learner's final measured reward. Default: no-op.
    fn observe(&mut self, learner_reward: f32) {
        let _ = learner_reward;
    }

    /// Rollout-driven providers (the policy opponent) answer true when
    /// the lane should play one opponent episode from the fixed root:
    /// `latest` is the active model version advertised on this lane's eval
    /// replies, and a rollout is due whenever it differs from the version
    /// the current reference was played under. `latest` None means no
    /// eval reply has advertised a version yet; providers that seed their
    /// reference with a cold-start rollout answer true exactly once
    /// there. Default: never.
    fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        let _ = latest;
        false
    }

    /// Atomically claims a rollout admission when one is due. Most
    /// providers are lane-local and can use the legacy rollout_due +
    /// begin_rollout pair. Shared providers override this to collapse
    /// many lanes racing on the same checkpoint to one challenger.
    fn claim_rollout(&mut self, latest: Option<ModelVersion>) -> bool {
        if !self.rollout_due(latest) {
            return false;
        }
        <Self as ReferenceProvider<E>>::begin_rollout(self, latest);
        true
    }

    /// The lane admitted the requested rollout episode. `version` is
    /// None for the cold-start seed rollout (admitted before any eval
    /// reply named a version); the finished episode's own replies name
    /// it via `RolloutOutcome::model_version`.
    fn begin_rollout(&mut self, version: Option<ModelVersion>) {
        let _ = version;
    }

    /// The rollout episode finished. None means it went unmeasured or
    /// invalid; the provider keeps its previous reference and the lane
    /// will retry while the version still differs.
    fn finish_rollout(&mut self, outcome: Option<RolloutOutcome>) {
        let _ = outcome;
    }

    /// Claims one stochastic rollout for the accepted policy's trajectory
    /// pool. The returned version is the checkpoint the rollout must use.
    fn claim_sample_rollout(&mut self, latest: Option<ModelVersion>) -> Option<ModelVersion> {
        let _ = latest;
        None
    }

    /// Finishes a previously claimed stochastic trajectory rollout.
    fn finish_sample_rollout(&mut self, version: ModelVersion, outcome: Option<RolloutOutcome>) {
        let _ = (version, outcome);
    }

    fn sampled_trajectory_mode(&self) -> bool {
        false
    }

    fn sampled_tree_mode(&self) -> bool {
        false
    }

    /// Maximum arena roots driven as one coordinated evaluator wave. Zero
    /// keeps the legacy lane-local rollout admission path.
    fn arena_parallelism(&self) -> usize {
        0
    }

    fn finish_sampled_trajectory(&mut self, outcome: Option<RolloutOutcome>) -> Option<Reference> {
        let _ = outcome;
        None
    }

    /// Claims one fixed-arena rollout. Shared arena providers use this to
    /// distribute root indexes across lanes while collapsing duplicate work.
    fn claim_arena_rollout(
        &mut self,
        latest: Option<ModelVersion>,
        lane: usize,
        lanes: usize,
    ) -> Option<ArenaRolloutClaim> {
        let _ = (latest, lane, lanes);
        None
    }

    /// Returns this lane's engine-local handle for a fixed arena root.
    fn arena_root(&mut self, engine: &mut E, index: usize) -> EngineResult<Option<E::Graph>> {
        let _ = (engine, index);
        Ok(None)
    }

    fn finish_arena_rollout(
        &mut self,
        claim: ArenaRolloutClaim,
        score: Option<f32>,
        outcome: Option<RolloutOutcome>,
    ) {
        let _ = (claim, score, outcome);
    }

    /// Whether every learner root first needs an opponent-policy rollout on
    /// that same root. This is the generated-root arena-gated policy path.
    fn per_root_policy_mode(&self) -> bool {
        false
    }

    fn claim_per_root_policy(
        &mut self,
        latest: Option<ModelVersion>,
    ) -> Option<EpisodeRolloutClaim> {
        let _ = latest;
        None
    }

    fn finish_per_root_policy(
        &mut self,
        claim: EpisodeRolloutClaim,
        outcome: Option<RolloutOutcome>,
    ) -> Option<Reference> {
        let _ = (claim, outcome);
        None
    }

    /// Whether episodes are expected to carry a reference once this
    /// provider is warmed up. When true, episodes that completed before
    /// the first reference existed (the pre-rollout admission wave) are
    /// dropped instead of stored as unlabeled rows: the store then only
    /// ever contains labeled, on-distribution training data.
    fn expects_reference(&self) -> bool {
        true
    }

    /// Whether admissions would receive a reference right now. Lanes
    /// hold learner admission while this is false so the cold-start
    /// wave is played against the seed rollout instead of being dropped
    /// unlabeled. Providers whose reference can only come from played
    /// episodes (self-average) must stay true or nothing ever plays.
    fn admission_ready(&self) -> bool {
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    pub ref_id: Option<u64>,
    pub kind: ReplayReferenceKind,
    pub final_reward: f32,
    pub final_graph: Option<ReplayGraphContext>,
    pub steps: Arc<[ReferenceStep]>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
}

pub struct RootBaselineProvider {
    measure_options: MeasureOptions,
}

impl RootBaselineProvider {
    #[must_use]
    pub const fn new(measure_options: MeasureOptions) -> Self {
        Self { measure_options }
    }
}

impl<E> ReferenceProvider<E> for RootBaselineProvider
where
    E: GraphEngine,
{
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let measure = engine.measure(root, self.measure_options)?;
        let Some(final_reward) = score(measure.measured, measure.valid, measure.scalar_reward)
        else {
            return Ok(None);
        };
        let final_graph = context(engine, measure.graph_hash);

        Ok(Some(Reference {
            ref_id: None,
            kind: ReplayReferenceKind::RootBaseline,
            final_reward,
            final_graph: Some(final_graph),
            steps: vec![ReferenceStep {
                context: final_graph,
                features: None,
            }]
            .into(),
            search_config_hash: None,
            model_version: None,
        }))
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
        let measure = engine.measure(root, self.measure_options)?;
        let Some(final_reward) = score(measure.measured, measure.valid, measure.scalar_reward)
        else {
            return Ok(None);
        };
        let final_graph = context(engine, measure.graph_hash);
        let mut created_candidates = Vec::new();
        let step = feature_reference_step(
            engine,
            extractor,
            root,
            final_graph,
            candidate_options,
            ReferenceFeatureContext {
                index: 0,
                final_reward,
                export_position,
            },
            &mut created_candidates,
        );
        let release = engine.release(&[], &created_candidates);
        let step = step?;
        release?;

        Ok(Some(Reference {
            ref_id: None,
            kind: ReplayReferenceKind::RootBaseline,
            final_reward,
            final_graph: Some(final_graph),
            steps: vec![step].into(),
            search_config_hash: None,
            model_version: None,
        }))
    }
}

pub struct GreedyReferenceProvider {
    search: GreedySearch,
}

impl GreedyReferenceProvider {
    #[must_use]
    pub fn new(search: GreedySearch) -> Self {
        Self { search }
    }
}

impl<E> ReferenceProvider<E> for GreedyReferenceProvider
where
    E: GraphEngine,
{
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode(
            ReplayReferenceKind::Greedy,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        Ok(reference)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        _candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode_with_features(
            engine,
            extractor,
            SearchReferenceProjection {
                kind: ReplayReferenceKind::Greedy,
                final_graph: episode.final_graph,
                final_context: episode.final_context,
                steps: &episode.steps,
                final_reward: score(
                    episode.final_measure.measured,
                    episode.final_measure.valid,
                    episode.final_measure.scalar_reward,
                ),
                search_config_hash: Some(episode.search_config_hash),
                candidate_options: self.search.config().candidate_options,
                export_position,
            },
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        reference
    }
}

pub struct BeamReferenceProvider {
    search: BeamSearch,
}

impl BeamReferenceProvider {
    #[must_use]
    pub fn new(search: BeamSearch) -> Self {
        Self { search }
    }
}

impl<E> ReferenceProvider<E> for BeamReferenceProvider
where
    E: GraphEngine,
{
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode(
            ReplayReferenceKind::Beam,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        Ok(reference)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        _candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode_with_features(
            engine,
            extractor,
            SearchReferenceProjection {
                kind: ReplayReferenceKind::Beam,
                final_graph: episode.final_graph,
                final_context: episode.final_context,
                steps: &episode.steps,
                final_reward: score(
                    episode.final_measure.measured,
                    episode.final_measure.valid,
                    episode.final_measure.scalar_reward,
                ),
                search_config_hash: Some(episode.search_config_hash),
                candidate_options: self.search.config().candidate_options,
                export_position,
            },
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        reference
    }
}

pub struct RandomReferenceProvider {
    search: RandomSearch,
}

impl RandomReferenceProvider {
    #[must_use]
    pub fn new(search: RandomSearch) -> Self {
        Self { search }
    }
}

impl<E> ReferenceProvider<E> for RandomReferenceProvider
where
    E: GraphEngine,
{
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode(
            ReplayReferenceKind::Random,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        Ok(reference)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        _candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode_with_features(
            engine,
            extractor,
            SearchReferenceProjection {
                kind: ReplayReferenceKind::Random,
                final_graph: episode.final_graph,
                final_context: episode.final_context,
                steps: &episode.steps,
                final_reward: score(
                    episode.final_measure.measured,
                    episode.final_measure.valid,
                    episode.final_measure.scalar_reward,
                ),
                search_config_hash: Some(episode.search_config_hash),
                candidate_options: self.search.config().candidate_options,
                export_position,
            },
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        reference
    }
}

fn project_search_episode<G, C>(
    kind: ReplayReferenceKind,
    final_graph: ReplayGraphContext,
    steps: &[SearchStep<G, C>],
    final_reward: Option<f32>,
    search_config_hash: Option<SearchConfigHash>,
) -> Option<Reference> {
    let final_reward = final_reward?;
    let mut reference_steps = Vec::with_capacity(steps.len() + 1);

    match steps.first() {
        Some(step) => reference_steps.push(ReferenceStep {
            context: step.step_ref.before,
            features: None,
        }),
        None => reference_steps.push(ReferenceStep {
            context: final_graph,
            features: None,
        }),
    }

    reference_steps.extend(steps.iter().map(|step| ReferenceStep {
        context: step.step_ref.after,
        features: None,
    }));

    Some(Reference {
        ref_id: None,
        kind,
        final_reward,
        final_graph: Some(final_graph),
        steps: reference_steps.into(),
        search_config_hash,
        model_version: None,
    })
}

struct SearchReferenceProjection<'a, E: GraphEngine> {
    kind: ReplayReferenceKind,
    final_graph: E::Graph,
    final_context: ReplayGraphContext,
    steps: &'a [SearchStep<E::Graph, E::Candidate>],
    final_reward: Option<f32>,
    search_config_hash: Option<SearchConfigHash>,
    candidate_options: CandidateOptions,
    export_position: bool,
}

fn project_search_episode_with_features<E, X>(
    engine: &mut E,
    extractor: &mut X,
    projection: SearchReferenceProjection<'_, E>,
) -> EngineResult<Option<Reference>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let Some(final_reward) = projection.final_reward else {
        return Ok(None);
    };
    let mut feature_candidates = Vec::new();
    let mut reference_steps = Vec::with_capacity(projection.steps.len() + 1);

    match projection.steps.first() {
        Some(step) => reference_steps.push(feature_reference_step(
            engine,
            extractor,
            step.before,
            step.step_ref.before,
            projection.candidate_options,
            ReferenceFeatureContext {
                index: 0,
                final_reward,
                export_position: projection.export_position,
            },
            &mut feature_candidates,
        )),
        None => reference_steps.push(feature_reference_step(
            engine,
            extractor,
            projection.final_graph,
            projection.final_context,
            projection.candidate_options,
            ReferenceFeatureContext {
                index: 0,
                final_reward,
                export_position: projection.export_position,
            },
            &mut feature_candidates,
        )),
    }

    for (index, step) in projection.steps.iter().enumerate() {
        reference_steps.push(feature_reference_step(
            engine,
            extractor,
            step.after,
            step.step_ref.after,
            projection.candidate_options,
            ReferenceFeatureContext {
                index: index + 1,
                final_reward,
                export_position: projection.export_position,
            },
            &mut feature_candidates,
        ));
    }

    let release = engine.release(&[], &feature_candidates);
    let mut steps = Vec::with_capacity(reference_steps.len());
    for step in reference_steps {
        steps.push(step?);
    }
    release?;

    Ok(Some(Reference {
        ref_id: None,
        kind: projection.kind,
        final_reward,
        final_graph: Some(projection.final_context),
        steps: steps.into(),
        search_config_hash: projection.search_config_hash,
        model_version: None,
    }))
}

#[derive(Clone, Copy)]
struct ReferenceFeatureContext {
    index: usize,
    final_reward: f32,
    export_position: bool,
}

fn feature_reference_step<E, X>(
    engine: &mut E,
    extractor: &mut X,
    graph: E::Graph,
    context: ReplayGraphContext,
    candidate_options: CandidateOptions,
    feature_context: ReferenceFeatureContext,
    created_candidates: &mut Vec<E::Candidate>,
) -> EngineResult<ReferenceStep>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    engine.candidates(graph, candidate_options, &mut candidates)?;
    created_candidates.extend(candidates.iter().copied());
    let scale = extractor.schema().config().opponent_reward_scale;
    let (root_step, budget_fraction, budget_step) = if feature_context.export_position {
        (
            u32::try_from(feature_context.index).map_err(|_| internal("root step overflow"))?,
            0.0,
            0.0,
        )
    } else {
        (0, 0.0, 0.0)
    };
    let row = extractor
        .extract(
            engine,
            graph,
            &candidates,
            PositionFeatures {
                root_step,
                leaf_depth: 0,
                budget_fraction,
                budget_step,
                opponent_reward: feature_context.final_reward / scale,
                opponent_present: true,
            },
        )
        .map_err(|_| internal("reference feature extraction failed"))?;

    Ok(ReferenceStep {
        context,
        features: Some(opponent_state(row)),
    })
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

/// Adaptive reference: a reward EMA of the learner's own recent episodes
/// on this provider's lane. Unlabeled until the first observed episode
/// seeds the EMA. Never touches the engine.
pub struct SelfAverageProvider {
    decay: f64,
    ema: Option<f64>,
}

impl SelfAverageProvider {
    #[must_use]
    pub fn new(decay: f32) -> Self {
        assert!(
            decay.is_finite() && decay > 0.0 && decay < 1.0,
            "self-average decay must be in (0, 1)"
        );
        Self {
            decay: f64::from(decay),
            ema: None,
        }
    }
}

impl<E> ReferenceProvider<E> for SelfAverageProvider
where
    E: GraphEngine,
{
    fn reference(&mut self, _engine: &mut E, _root: E::Graph) -> EngineResult<Option<Reference>> {
        let Some(ema) = self.ema else {
            return Ok(None);
        };

        Ok(Some(Reference {
            ref_id: None,
            kind: ReplayReferenceKind::SelfAverage,
            final_reward: ema as f32,
            final_graph: None,
            steps: Vec::new().into(),
            search_config_hash: None,
            model_version: None,
        }))
    }

    fn observe(&mut self, learner_reward: f32) {
        let reward = f64::from(learner_reward);
        self.ema = Some(match self.ema {
            None => reward,
            Some(ema) => self.decay * ema + (1.0 - self.decay) * reward,
        });
    }
}

/// The network itself as the opponent: the terminal reward of a greedy
/// (temperature-0, one-simulation) policy rollout from the fixed root,
/// played once per published checkpoint. The lane drives the rollout
/// through the normal episode machinery and reports back through the
/// rollout hooks; this provider only holds the resulting scalar.
/// Episodes are unlabeled until the first rollout completes.
pub struct PolicyReferenceProvider {
    gate: PolicyGate,
    current: Option<PolicyReference>,
    registry: Option<Arc<ReferenceRegistry>>,
    last_challenged: Option<ModelVersion>,
    pending: Option<PendingChallenge>,
    sampled_trajectory: bool,
    sampled_tree: bool,
    arena_registry: Option<Arc<ArenaGateRegistry>>,
}

/// A challenger rollout in flight: versioned when the lane knew the
/// evaluator's checkpoint at admission, seed when it was admitted cold
/// (before any eval reply) and the version comes from the finished
/// episode itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingChallenge {
    Versioned(ModelVersion),
    Seed,
}

/// How a finished challenger rollout updates the reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyGate {
    /// Every measured rollout replaces the reference: the bar tracks the
    /// newest checkpoint, up or down.
    Latest,
    /// whittlezero's arena gate on the fixed root: a challenger is
    /// accepted only if it strictly beats the incumbent, so the bar is
    /// monotone and rows attribute the incumbent's version. Every
    /// checkpoint is still challenged exactly once.
    Best,
}

#[derive(Clone)]
struct PolicyReference {
    reward: f32,
    version: ModelVersion,
    final_graph: ReplayGraphContext,
    steps: Arc<[ReferenceStep]>,
    search_config_hash: SearchConfigHash,
}

impl PolicyReferenceProvider {
    #[must_use]
    pub const fn new() -> Self {
        Self::with_gate(PolicyGate::Latest)
    }

    #[must_use]
    pub const fn gated() -> Self {
        Self::with_gate(PolicyGate::Best)
    }

    #[must_use]
    pub fn gated_with_registry(registry: Arc<ReferenceRegistry>) -> Self {
        let mut provider = Self::with_gate(PolicyGate::Best);
        provider.registry = Some(registry);
        provider
    }

    #[must_use]
    pub fn sampled_trajectory_with_registry(registry: Arc<ReferenceRegistry>) -> Self {
        let mut provider = Self::new();
        provider.registry = Some(registry);
        provider.sampled_trajectory = true;
        provider
    }

    #[must_use]
    pub fn sampled_tree_with_registry(registry: Arc<ReferenceRegistry>) -> Self {
        let mut provider = Self::with_gate(PolicyGate::Best);
        provider.registry = Some(registry);
        provider.sampled_tree = true;
        provider
    }

    #[must_use]
    pub fn arena_gated(registry: Arc<ArenaGateRegistry>) -> Self {
        let mut provider = Self::with_gate(PolicyGate::Best);
        provider.arena_registry = Some(registry);
        provider
    }

    #[must_use]
    pub fn arena_sampled_tree(registry: Arc<ArenaGateRegistry>) -> Self {
        let mut provider = Self::with_gate(PolicyGate::Best);
        provider.arena_registry = Some(registry);
        provider.sampled_tree = true;
        provider
    }

    const fn with_gate(gate: PolicyGate) -> Self {
        Self {
            gate,
            current: None,
            registry: None,
            last_challenged: None,
            pending: None,
            sampled_trajectory: false,
            sampled_tree: false,
            arena_registry: None,
        }
    }
}

impl Default for PolicyReferenceProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> ReferenceProvider<E> for PolicyReferenceProvider
where
    E: GraphEngine,
{
    fn reference(&mut self, _engine: &mut E, _root: E::Graph) -> EngineResult<Option<Reference>> {
        if self.sampled_tree || self.arena_registry.is_some() {
            return Ok(None);
        }
        if let Some(registry) = &self.registry {
            return Ok(registry.sampled().as_deref().map(reference_from_snapshot));
        }
        let Some(current) = self.current.as_ref() else {
            return Ok(None);
        };

        Ok(Some(Reference {
            ref_id: None,
            kind: match self.gate {
                PolicyGate::Latest => ReplayReferenceKind::Gumbel,
                PolicyGate::Best => ReplayReferenceKind::GatedPolicy,
            },
            final_reward: current.reward,
            final_graph: Some(current.final_graph),
            steps: Arc::clone(&current.steps),
            search_config_hash: Some(current.search_config_hash),
            model_version: Some(current.version),
        }))
    }

    fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        if self.sampled_trajectory || self.arena_registry.is_some() {
            return false;
        }
        if let Some(registry) = &self.registry {
            return registry.rollout_due(latest);
        }
        if self.pending.is_some() {
            return false;
        }
        // Dueness anchors on the last MEASURED challenge, not the
        // incumbent: gated rejections must not retry, unmeasured rollouts
        // must. Latest mode is unchanged by this anchor (it accepts every
        // measured challenge, so last_challenged tracks current.version).
        match latest {
            Some(latest) => self.last_challenged != Some(latest),
            // No version observed yet: the seed rollout is due exactly
            // once, so the lane's first act creates the reference and
            // learner admission never runs unlabeled.
            None => self.last_challenged.is_none(),
        }
    }

    fn claim_rollout(&mut self, latest: Option<ModelVersion>) -> bool {
        if self.sampled_trajectory || self.arena_registry.is_some() {
            return false;
        }
        if let Some(registry) = &self.registry {
            return registry.claim_challenge(latest);
        }
        if !<Self as ReferenceProvider<E>>::rollout_due(self, latest) {
            return false;
        }
        <Self as ReferenceProvider<E>>::begin_rollout(self, latest);
        true
    }

    fn begin_rollout(&mut self, version: Option<ModelVersion>) {
        if let Some(registry) = &self.registry {
            let _ = registry.claim_challenge(version);
            return;
        }
        self.pending = Some(match version {
            Some(version) => PendingChallenge::Versioned(version),
            None => PendingChallenge::Seed,
        });
    }

    fn finish_rollout(&mut self, outcome: Option<RolloutOutcome>) {
        if let Some(registry) = &self.registry {
            if let Some(event) = registry.finish_challenge(outcome) {
                eprintln!(
                    "event=policy_gate accepted={} challenger={} best={} steps={} version={}",
                    event.accepted, event.challenger, event.best, event.steps, event.version,
                );
            }
            return;
        }
        let Some(pending) = self.pending.take() else {
            return;
        };
        // Unmeasured challengers retry: last_challenged stays put.
        let Some(outcome) = outcome else {
            return;
        };
        let version = match pending {
            PendingChallenge::Versioned(version) if outcome.model_version == Some(version) => {
                Some(version)
            }
            PendingChallenge::Versioned(_) => None,
            PendingChallenge::Seed => outcome.model_version,
        };
        // A seed rollout whose replies never named a version counts as
        // unmeasured and retries.
        let Some(version) = version else {
            return;
        };
        self.last_challenged = Some(version);
        let accepted = match self.gate {
            PolicyGate::Latest => true,
            PolicyGate::Best => self
                .current
                .as_ref()
                .is_none_or(|incumbent| outcome.final_reward > incumbent.reward),
        };
        if self.gate == PolicyGate::Best {
            // Machine-parsed by the trainer driver (opponent metrics);
            // field changes must update its parser.
            eprintln!(
                "event=policy_gate accepted={accepted} challenger={} best={} steps={} version={version}",
                outcome.final_reward,
                self.current
                    .as_ref()
                    .map_or(outcome.final_reward, |incumbent| incumbent.reward),
                outcome.steps.len(),
            );
        }
        let challenger = PolicyReference {
            reward: outcome.final_reward,
            version,
            final_graph: outcome.final_graph,
            steps: outcome.steps.into(),
            search_config_hash: outcome.search_config_hash,
        };
        if accepted {
            self.current = Some(challenger);
        }
    }

    fn claim_sample_rollout(&mut self, latest: Option<ModelVersion>) -> Option<ModelVersion> {
        self.registry
            .as_ref()
            .and_then(|registry| registry.claim_sample(latest))
    }

    fn finish_sample_rollout(&mut self, version: ModelVersion, outcome: Option<RolloutOutcome>) {
        let Some(registry) = &self.registry else {
            return;
        };
        if registry.finish_sample(version, outcome) {
            eprintln!(
                "event=reference_trajectory_pool version={version} size={}",
                registry.trajectory_pool_len(),
            );
        }
    }

    fn sampled_trajectory_mode(&self) -> bool {
        self.sampled_trajectory
    }

    fn sampled_tree_mode(&self) -> bool {
        self.sampled_tree
    }

    fn finish_sampled_trajectory(&mut self, outcome: Option<RolloutOutcome>) -> Option<Reference> {
        let outcome = outcome?;
        let ref_id = self.registry.as_ref()?.allocate_reference_id();
        Some(Reference {
            ref_id: Some(ref_id),
            kind: ReplayReferenceKind::Gumbel,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: Some(outcome.search_config_hash),
            model_version: outcome.model_version,
        })
    }

    fn claim_arena_rollout(
        &mut self,
        latest: Option<ModelVersion>,
        lane: usize,
        lanes: usize,
    ) -> Option<ArenaRolloutClaim> {
        let registry = self.arena_registry.as_ref()?;
        if let Some(latest) = latest {
            registry.observe_current(latest);
        }
        registry.claim_arena(lane, lanes)
    }

    fn finish_arena_rollout(
        &mut self,
        claim: ArenaRolloutClaim,
        score: Option<f32>,
        outcome: Option<RolloutOutcome>,
    ) {
        let Some(registry) = &self.arena_registry else {
            return;
        };
        let actual_version = outcome.as_ref().and_then(|outcome| outcome.model_version);
        let steps = outcome.as_ref().map_or(0, |outcome| outcome.steps.len());
        if let Some(event) = registry.finish_arena(claim, actual_version, score, steps) {
            eprintln!(
                "event=arena_gate accepted={} challenger={} best={} margin={} arena_size={} steps={} version={}",
                event.accepted,
                event.challenger_mean,
                event.best_mean,
                event.margin_sum,
                event.arena_size,
                event.steps,
                event.version,
            );
        }
    }

    fn per_root_policy_mode(&self) -> bool {
        self.arena_registry.is_some() && !self.sampled_tree
    }

    fn claim_per_root_policy(
        &mut self,
        latest: Option<ModelVersion>,
    ) -> Option<EpisodeRolloutClaim> {
        let registry = self.arena_registry.as_ref()?;
        if let Some(latest) = latest {
            registry.observe_current(latest);
        }
        registry.claim_episode()
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
        let registry = self.arena_registry.as_ref()?;
        Some(Reference {
            ref_id: Some(registry.allocate_reference_id()),
            kind: ReplayReferenceKind::GatedPolicy,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: Some(outcome.search_config_hash),
            model_version: Some(claim.version),
        })
    }

    fn admission_ready(&self) -> bool {
        if let Some(registry) = &self.arena_registry {
            return registry.admission_ready();
        }
        if self.sampled_trajectory {
            return true;
        }
        self.registry
            .as_ref()
            .map_or(self.current.is_some(), |registry| {
                registry.admission_ready()
            })
    }
}

fn reference_from_snapshot(snapshot: &ReferenceSnapshot) -> Reference {
    Reference {
        ref_id: Some(snapshot.ref_id),
        kind: snapshot.kind,
        final_reward: snapshot.final_reward,
        final_graph: snapshot.final_graph,
        steps: Arc::clone(&snapshot.steps),
        search_config_hash: Some(snapshot.search_config_hash),
        model_version: Some(snapshot.version),
    }
}

fn score(measured: bool, valid: bool, scalar_reward: Option<f32>) -> Option<f32> {
    if !measured || !valid {
        return None;
    }

    match scalar_reward {
        Some(reward) if reward.is_finite() => Some(reward),
        _ => None,
    }
}

fn context<E: GraphEngine>(engine: &E, graph_hash: gz_engine::GraphHash) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(graph_hash, engine.engine_id(), engine.engine_version()),
        engine.action_set_hash(),
    )
}

fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(9_001),
        message: ErrorMessage::new(message).expect("static error message fits"),
    }
}
