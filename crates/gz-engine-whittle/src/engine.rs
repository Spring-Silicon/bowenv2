use crate::graph::{
    GraphBody, GraphError, NO_NODE, OpCode, WhittleCandidateId, WhittleGraph, WhittleGraphId,
    compact_graph, deserialize_wav1, serialize_wav1, serialize_whittle_graph_wav1,
};
use crate::rules::{
    RawCandidate, RuleError, apply_graph, category_weight, enumerate_graph,
    enumerate_graph_limited, inverse_rule_id, rule_name,
};
use gz_engine::{
    ActionSetHash, ApplyMetrics, ApplyResult, BatchGraphEngine, CandidateHash, CandidateInfo,
    CandidateKindId, CandidateMetadata, CandidateOptions, CandidateTags, EngineContractFixture,
    EngineError, EngineId, EngineResult, EngineVersion, ErrorCode, ErrorMessage, GraphArtifact,
    GraphArtifactFormat, GraphEngine, GraphHash, MeasureConfigHash, MeasureMetadata,
    MeasureOptions, MeasureResult, SubjectId,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub type WhittleRng = rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct WhittleEngineConfig {
    pub root: WhittleRoot,
    pub include_reverse_constant_folding: bool,
    pub measure_mode: WhittleMeasureMode,
    pub cache_candidates: bool,
    pub cache_transitions: bool,
}

impl Default for WhittleEngineConfig {
    fn default() -> Self {
        Self {
            root: WhittleRoot::Input {
                arity: 1,
                capacity: 16,
                input_index: 0,
            },
            include_reverse_constant_folding: false,
            measure_mode: WhittleMeasureMode::NegativeCost,
            cache_candidates: true,
            cache_transitions: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum WhittleRoot {
    Input {
        arity: u16,
        capacity: u16,
        input_index: u16,
    },
    Artifact(Vec<u8>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhittleMeasureMode {
    NegativeCost,
}

#[derive(Clone, Debug)]
pub struct WhittleGraphGeneratorConfig {
    pub arity: u16,
    pub capacity: u16,
    pub exception_terms_min: u16,
    pub exception_terms_max: u16,
    pub prewalk_steps_min: u16,
    pub prewalk_steps_max: u16,
}

impl Default for WhittleGraphGeneratorConfig {
    fn default() -> Self {
        Self {
            arity: 6,
            capacity: 256,
            exception_terms_min: 5,
            exception_terms_max: 7,
            prewalk_steps_min: 4,
            prewalk_steps_max: 64,
        }
    }
}

impl WhittleGraphGeneratorConfig {
    pub fn validate(self) -> Result<Self, WhittleGeneratorConfigError> {
        if self.arity == 0 {
            return Err(WhittleGeneratorConfigError::ZeroArity);
        }
        if self.arity > 16 {
            return Err(WhittleGeneratorConfigError::ArityTooLarge {
                max: 16,
                actual: self.arity,
            });
        }
        if self.capacity < self.arity + 1 {
            return Err(WhittleGeneratorConfigError::CapacityTooSmall);
        }
        if self.exception_terms_min > self.exception_terms_max {
            return Err(WhittleGeneratorConfigError::InvalidExceptionTermRange);
        }
        if self.exception_terms_max == 0 {
            return Err(WhittleGeneratorConfigError::ZeroExceptionTerms);
        }
        if self.prewalk_steps_min > self.prewalk_steps_max {
            return Err(WhittleGeneratorConfigError::InvalidPrewalkRange);
        }

        Ok(self)
    }
}

pub struct WhittleGraphGenerator {
    config: WhittleGraphGeneratorConfig,
    rng: WhittleRng,
}

impl WhittleGraphGenerator {
    pub fn from_seed(config: WhittleGraphGeneratorConfig, seed: u64) -> Self {
        use rand::SeedableRng;

        Self {
            config,
            rng: WhittleRng::seed_from_u64(seed),
        }
    }

    pub fn sample_into(
        &mut self,
        engine: &mut WhittleEngine,
    ) -> EngineResult<GeneratedWhittleGraph> {
        self.sample_with_created_graphs(engine)
            .map(|(generated, _)| generated)
    }

    /// Samples the same generated-root distribution as [`Self::sample_into`],
    /// but transfers only the final graph reference to the caller. Seed and
    /// intermediate prewalk references are released before returning.
    pub fn sample_root_into(&mut self, engine: &mut WhittleEngine) -> EngineResult<WhittleGraphId> {
        let (generated, created_graphs) = self.sample_with_created_graphs(engine)?;
        let intermediates = &created_graphs[..created_graphs.len().saturating_sub(1)];
        engine.release(intermediates, &[])?;
        Ok(generated.graph)
    }

    fn sample_with_created_graphs(
        &mut self,
        engine: &mut WhittleEngine,
    ) -> EngineResult<(GeneratedWhittleGraph, Vec<WhittleGraphId>)> {
        let config = self.config.clone().validate().map_err(generator_error)?;
        let seed_body = truth_table_seed(&config, &mut self.rng).map_err(generator_error)?;
        let seed_graph = engine.graphs.insert(seed_body).map_err(internal_graph)?;
        let mut created_graphs = vec![seed_graph];
        let seed = engine.graph(seed_graph)?;
        let start_cost = seed.cost();
        let mut graph = seed_graph;
        let mut body = seed.body();
        let steps = random_u16_inclusive(
            &mut self.rng,
            config.prewalk_steps_min,
            config.prewalk_steps_max,
        );
        let mut applied = 0;
        let mut last_rule = None;

        for _ in 0..steps {
            let blocked_rule = last_rule.and_then(inverse_rule_id);
            let candidates = enumerate_graph(&body, true).map_err(internal_rule)?;
            let Some(candidate) =
                choose_prewalk_candidate(&candidates, blocked_rule, &mut self.rng)
            else {
                break;
            };

            body = apply_graph(&body, candidate).map_err(internal_rule)?;
            graph = engine.graphs.insert(body.clone()).map_err(internal_graph)?;
            created_graphs.push(graph);
            last_rule = Some(candidate.rule_id);
            applied += 1;
        }

        Ok((
            GeneratedWhittleGraph {
                graph,
                seed_graph,
                prewalk_steps_requested: steps,
                prewalk_steps_applied: applied,
                start_cost,
                final_cost: engine.graph(graph)?.cost(),
            },
            created_graphs,
        ))
    }
}

pub struct GeneratedWhittleGraph {
    pub graph: WhittleGraphId,
    pub seed_graph: WhittleGraphId,
    pub prewalk_steps_requested: u16,
    pub prewalk_steps_applied: u16,
    pub start_cost: u32,
    pub final_cost: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaOccupancy {
    pub graphs_live: usize,
    pub graph_refs: u64,
    pub candidates_live: usize,
    pub candidate_refs: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateStorageStats {
    pub handle_slots: usize,
    pub handle_bytes: usize,
    pub handle_slot_size: usize,
    pub batches_live: usize,
    pub records_live: usize,
    pub record_bytes: usize,
    pub record_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhittleGeneratorConfigError {
    ZeroArity,
    ArityTooLarge { max: u16, actual: u16 },
    CapacityTooSmall,
    InvalidExceptionTermRange,
    ZeroExceptionTerms,
    InvalidPrewalkRange,
    SeedGraphExceedsCapacity,
}

impl fmt::Display for WhittleGeneratorConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroArity => f.write_str("arity must be greater than zero"),
            Self::ArityTooLarge { max, actual } => {
                write!(f, "arity must be <= {max}, got {actual}")
            }
            Self::CapacityTooSmall => f.write_str("capacity must fit inputs and output"),
            Self::InvalidExceptionTermRange => {
                f.write_str("exception term minimum must be <= maximum")
            }
            Self::ZeroExceptionTerms => f.write_str("exception term maximum must be positive"),
            Self::InvalidPrewalkRange => f.write_str("prewalk minimum must be <= maximum"),
            Self::SeedGraphExceedsCapacity => f.write_str("seed graph exceeds capacity"),
        }
    }
}

impl std::error::Error for WhittleGeneratorConfigError {}

pub struct WhittleEngine {
    config: WhittleEngineConfig,
    root: WhittleGraphId,
    graphs: GraphArena,
    candidates: CandidateArena,
    caches: WhittleCaches,
    engine_id: EngineId,
    engine_version: EngineVersion,
    action_set_hash: ActionSetHash,
    measure_config_hash: MeasureConfigHash,
}

impl Drop for WhittleEngine {
    fn drop(&mut self) {
        if std::env::var_os("GZ_ARENA_STATS").is_some() {
            let occupancy = self.arena_occupancy();
            let candidate_storage = self.candidate_storage_stats();
            let cache_cand_ids: usize = self.caches.candidates.values().map(|c| c.ids.len()).sum();
            let cache_trans_bodies: usize =
                self.caches.transitions.values().map(HashMap::len).sum();
            eprintln!(
                "arena_stats graphs_live={} graphs_slots={} graph_refs={} cands_live={} cands_slots={} cand_refs={} cand_batches={} cand_records={} cand_record_bytes={} cache_cand_keys={} cache_cand_ids={} cache_trans_keys={} cache_trans_bodies={}",
                occupancy.graphs_live,
                self.graphs.items.len(),
                occupancy.graph_refs,
                occupancy.candidates_live,
                candidate_storage.handle_slots,
                occupancy.candidate_refs,
                candidate_storage.batches_live,
                candidate_storage.records_live,
                candidate_storage.record_bytes,
                self.caches.candidates.len(),
                cache_cand_ids,
                self.caches.transitions.len(),
                cache_trans_bodies,
            );
        }
    }
}

impl WhittleEngine {
    pub fn new(config: WhittleEngineConfig) -> EngineResult<Self> {
        let engine_id = engine_id();
        let engine_version = engine_version();
        let action_set_hash =
            action_set_hash(engine_version, config.include_reverse_constant_folding);
        let measure_config_hash = measure_config_hash(config.measure_mode);
        let mut graphs = GraphArena::new(engine_id, engine_version);
        let root_body = match &config.root {
            WhittleRoot::Input {
                arity,
                capacity,
                input_index,
            } => GraphBody::input(*arity, *capacity, *input_index).map_err(internal_graph)?,
            WhittleRoot::Artifact(bytes) => deserialize_wav1(bytes)
                .and_then(|body| compact_graph(&body))
                .map_err(internal_graph)?,
        };
        let root = graphs.insert(root_body).map_err(internal_graph)?;

        Ok(Self {
            config,
            root,
            graphs,
            candidates: CandidateArena::default(),
            caches: WhittleCaches::default(),
            engine_id,
            engine_version,
            action_set_hash,
            measure_config_hash,
        })
    }

    #[must_use]
    pub fn measure_options(&self) -> MeasureOptions {
        MeasureOptions::new(self.measure_config_hash, 1, None, true)
            .expect("static Whittle measure options are valid")
    }

    #[must_use]
    pub const fn measure_config_hash(&self) -> MeasureConfigHash {
        self.measure_config_hash
    }

    #[must_use]
    pub fn arena_occupancy(&self) -> ArenaOccupancy {
        ArenaOccupancy {
            graphs_live: self.graphs.by_hash.len(),
            graph_refs: self
                .graphs
                .hash_refs
                .values()
                .map(|refs| u64::from(*refs))
                .sum(),
            candidates_live: self.candidates.live_count(),
            candidate_refs: self.candidates.ref_count(),
        }
    }

    #[must_use]
    pub fn candidate_storage_stats(&self) -> CandidateStorageStats {
        self.candidates.storage_stats()
    }

    pub(crate) fn graph(&self, graph: WhittleGraphId) -> EngineResult<&WhittleGraph> {
        self.graphs
            .get(graph)
            .ok_or(EngineError::UnknownGraph { graph_hash: None })
    }

    fn candidate(&self, candidate: WhittleCandidateId) -> EngineResult<WhittleCandidate> {
        self.candidates
            .get(candidate)
            .ok_or(EngineError::UnknownCandidate {
                candidate_hash: None,
            })
    }

    /// Every returned id carries a reference the caller must release. The
    /// limit applies BEFORE slots are created: candidates past it never
    /// enter the arena, so truncation cannot strand references.
    fn enumerate_candidate_ids(
        &mut self,
        graph: WhittleGraphId,
        limit: Option<usize>,
    ) -> EngineResult<Vec<WhittleCandidateId>> {
        let graph_hash = self.hash(graph)?;
        let cache_key = (graph_hash, self.action_set_hash);

        if self.config.cache_candidates
            && let Some(cached) = self.caches.candidates.get(&cache_key)
            && !(cached.truncated && limit.is_none_or(|limit| limit > cached.ids.len()))
        {
            let take = limit.map_or(cached.ids.len(), |limit| limit.min(cached.ids.len()));
            let ids = cached.ids[..take].to_vec();
            for candidate in &ids {
                self.candidates.retain(*candidate)?;
            }
            return Ok(ids);
        }

        let raw = {
            let graph_body = self.graph(graph)?.body();
            enumerate_graph_limited(
                &graph_body,
                self.config.include_reverse_constant_folding,
                limit,
            )
            .map_err(internal_rule)?
        };
        // Equality is conservatively treated as truncation: if the complete
        // set happened to equal the limit, a later larger request only pays an
        // unnecessary re-enumeration and can never observe a stale prefix.
        let truncated = limit.is_some_and(|limit| raw.len() == limit);
        let ids = self.candidates.insert_batch(graph_hash, raw)?;

        if self.config.cache_candidates {
            self.caches.candidates.insert(
                cache_key,
                CachedCandidateIds {
                    ids: ids.clone(),
                    truncated,
                },
            );
        }

        Ok(ids)
    }
}

impl Default for WhittleEngine {
    fn default() -> Self {
        Self::new(WhittleEngineConfig::default()).expect("default Whittle config is valid")
    }
}

impl GraphEngine for WhittleEngine {
    type Graph = WhittleGraphId;
    type Candidate = WhittleCandidateId;

    fn engine_id(&self) -> EngineId {
        self.engine_id
    }

    fn engine_version(&self) -> EngineVersion {
        self.engine_version
    }

    fn action_set_hash(&self) -> ActionSetHash {
        self.action_set_hash
    }

    fn root(&self) -> Self::Graph {
        self.root
    }

    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash> {
        Ok(self.graph(graph)?.hash)
    }

    fn candidates(
        &mut self,
        graph: Self::Graph,
        options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()> {
        out.clear();
        out.extend(self.enumerate_candidate_ids(graph, options.max_candidates)?);
        Ok(())
    }

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo> {
        let graph_hash = self.hash(graph)?;
        let candidate = self.candidate(candidate)?;
        let candidate_hash =
            candidate_hash(candidate.graph_hash, self.action_set_hash, candidate.raw);

        if candidate.graph_hash != graph_hash {
            return Err(EngineError::StaleCandidate {
                expected_graph_hash: candidate.graph_hash,
                actual_graph_hash: graph_hash,
                candidate_hash,
            });
        }

        Ok(CandidateInfo {
            candidate_hash,
            graph_hash,
            action_set_hash: self.action_set_hash,
            kind: CandidateKindId::new(candidate.raw.rule_id.into()),
            display_name: format!(
                "{}@{}",
                rule_name(candidate.raw.rule_id),
                candidate.raw.root
            ),
            static_prior: 0.0,
            tags: CandidateTags::EMPTY,
            subjects: candidate
                .matched_slice()
                .iter()
                .copied()
                .map(u64::from)
                .map(SubjectId::new)
                .collect(),
            metadata: CandidateMetadata {
                bytes: candidate_metadata(&candidate),
            },
        })
    }

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>> {
        let before_hash = self.hash(graph)?;
        let candidate_body = self.candidate(candidate)?;
        let candidate_hash = candidate_hash(
            candidate_body.graph_hash,
            self.action_set_hash,
            candidate_body.raw,
        );

        if candidate_body.graph_hash != before_hash {
            return Err(EngineError::StaleCandidate {
                expected_graph_hash: candidate_body.graph_hash,
                actual_graph_hash: before_hash,
                candidate_hash,
            });
        }

        let cached_after_hash = if self.config.cache_transitions {
            self.caches
                .transitions
                .get(&before_hash)
                .and_then(|inner| inner.get(&candidate_hash))
                .copied()
        } else {
            None
        };

        let after = match cached_after_hash {
            Some(after_hash) => self.graphs.retain_hash(after_hash)?,
            None => None,
        };
        let after = match after {
            Some(after) => after,
            None => {
                let before = self.graph(graph)?.body();
                let after_body = apply_graph(&before, candidate_body.raw).map_err(internal_rule)?;
                let after = self.graphs.insert(after_body).map_err(internal_graph)?;
                if self.config.cache_transitions {
                    let after_hash = self.hash(after)?;
                    self.caches
                        .transitions
                        .entry(before_hash)
                        .or_default()
                        .insert(candidate_hash, after_hash);
                }
                after
            }
        };
        let after_hash = self.hash(after)?;

        Ok(ApplyResult {
            before: graph,
            after,
            before_hash,
            after_hash,
            candidate,
            candidate_hash,
            changed: before_hash != after_hash,
            rejected: None,
            metrics: ApplyMetrics::default(),
        })
    }

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>> {
        let graph_body = self.graph(graph)?;
        let cost = graph_body.cost();

        Ok(MeasureResult {
            graph,
            graph_hash: graph_body.hash,
            config_hash: options.config_hash,
            measured: true,
            valid: true,
            latency: None,
            scalar_reward: Some(-(cost as f32)),
            failure: None,
            metadata: MeasureMetadata {
                bytes: measure_metadata(cost, graph_body.arity, graph_body.capacity),
            },
        })
    }

    fn release(
        &mut self,
        graphs: &[Self::Graph],
        candidates: &[Self::Candidate],
    ) -> EngineResult<()> {
        for candidate in candidates.iter().copied() {
            if let Some(candidate_body) = self.candidates.release(candidate)? {
                self.caches
                    .candidates
                    .remove(&(candidate_body.graph_hash, self.action_set_hash));
                self.caches.transitions.remove(&candidate_body.graph_hash);
            }
        }

        for graph in graphs.iter().copied() {
            // Rewrite cycles can dedup an episode-created graph onto the
            // engine root; the episode's reference is real and releasable.
            // Only dropping the root's LAST reference is a caller bug.
            let (graph_hash, last_ref) = self.graphs.release_protected(graph, self.root)?;
            if last_ref {
                self.caches
                    .candidates
                    .remove(&(graph_hash, self.action_set_hash));
                self.caches.transitions.remove(&graph_hash);
            }
        }

        Ok(())
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact> {
        let graph = self.graph(graph)?;

        Ok(GraphArtifact {
            graph_hash: graph.hash,
            format: GraphArtifactFormat::Binary,
            bytes: serialize_whittle_graph_wav1(graph),
        })
    }
}

impl BatchGraphEngine for WhittleEngine {}

struct GraphArena {
    items: Vec<GraphSlot>,
    free: Vec<u32>,
    by_hash: HashMap<GraphHash, WhittleGraphId>,
    hash_refs: HashMap<GraphHash, u32>,
    engine_id: EngineId,
    engine_version: EngineVersion,
}

impl GraphArena {
    fn new(engine_id: EngineId, engine_version: EngineVersion) -> Self {
        Self {
            items: Vec::new(),
            free: Vec::new(),
            by_hash: HashMap::new(),
            hash_refs: HashMap::new(),
            engine_id,
            engine_version,
        }
    }

    fn insert(&mut self, body: GraphBody) -> Result<WhittleGraphId, GraphError> {
        let compact = compact_graph(&body)?;
        let canonical = serialize_wav1(&compact);
        let hash = graph_hash(self.engine_id, self.engine_version, &canonical);

        if let Some(id) = self.by_hash.get(&hash).copied() {
            let refs = self
                .hash_refs
                .get_mut(&hash)
                .ok_or(GraphError::InvalidInput("missing graph refcount"))?;
            *refs = refs
                .checked_add(1)
                .ok_or(GraphError::InvalidInput("graph refcount overflow"))?;
            return Ok(id);
        }

        let graph = WhittleGraph {
            arity: compact.arity,
            capacity: compact.capacity,
            output_node: compact.output_node,
            op: compact.op.into_boxed_slice(),
            arg0: compact.arg0.into_boxed_slice(),
            arg1: compact.arg1.into_boxed_slice(),
            hash,
        };
        let id = if let Some(index) = self.free.pop() {
            let slot = &mut self.items[index as usize];
            debug_assert!(slot.graph.is_none());
            slot.graph = Some(graph);
            WhittleGraphId::from_slot(index, slot.generation())
        } else {
            let index = self.items.len() as u32;
            self.items.push(GraphSlot::new(graph));
            WhittleGraphId::from_slot(index, 0)
        };
        self.by_hash.insert(hash, id);
        *self.hash_refs.entry(hash).or_insert(0) += 1;
        Ok(id)
    }

    fn get(&self, id: WhittleGraphId) -> Option<&WhittleGraph> {
        self.items.get(id.raw() as usize).and_then(|slot| {
            slot.assert_generation(id.generation(), "stale WhittleGraphId");
            slot.graph.as_ref()
        })
    }

    fn retain_hash(&mut self, hash: GraphHash) -> EngineResult<Option<WhittleGraphId>> {
        let Some(id) = self.by_hash.get(&hash).copied() else {
            return Ok(None);
        };
        let refs = self
            .hash_refs
            .get_mut(&hash)
            .ok_or_else(|| internal_error(5, "missing Whittle graph hash refcount"))?;
        *refs = refs
            .checked_add(1)
            .ok_or_else(|| internal_error(5, "Whittle graph refcount overflow"))?;
        Ok(Some(id))
    }

    fn release_protected(
        &mut self,
        id: WhittleGraphId,
        protected: WhittleGraphId,
    ) -> EngineResult<(GraphHash, bool)> {
        let Some(slot) = self.items.get_mut(id.raw() as usize) else {
            return Err(EngineError::UnknownGraph { graph_hash: None });
        };
        slot.assert_generation(id.generation(), "stale WhittleGraphId");
        let Some(graph) = slot.graph.as_ref() else {
            return Err(EngineError::UnknownGraph { graph_hash: None });
        };
        let hash = graph.hash;
        let refs = self
            .hash_refs
            .get_mut(&hash)
            .ok_or_else(|| internal_error(5, "missing Whittle graph hash refcount"))?;
        if *refs > 1 {
            *refs -= 1;
            return Ok((hash, false));
        }
        if id == protected {
            return Err(internal_error(4, "cannot free the Whittle root graph"));
        }
        slot.graph.take();
        slot.bump_generation();
        self.free.push(id.raw());
        self.hash_refs.remove(&hash);
        self.by_hash.remove(&hash);
        Ok((hash, true))
    }
}

struct GraphSlot {
    graph: Option<WhittleGraph>,
    #[cfg(debug_assertions)]
    generation: u32,
}

impl GraphSlot {
    fn new(graph: WhittleGraph) -> Self {
        Self {
            graph: Some(graph),
            #[cfg(debug_assertions)]
            generation: 0,
        }
    }

    #[cfg(debug_assertions)]
    const fn generation(&self) -> u32 {
        self.generation
    }

    #[cfg(not(debug_assertions))]
    const fn generation(&self) -> u32 {
        0
    }

    #[cfg(debug_assertions)]
    fn assert_generation(&self, expected: u32, message: &'static str) {
        assert_eq!(self.generation, expected, "{message}");
    }

    #[cfg(not(debug_assertions))]
    fn assert_generation(&self, _expected: u32, _message: &'static str) {}

    #[cfg(debug_assertions)]
    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    #[cfg(not(debug_assertions))]
    fn bump_generation(&mut self) {}
}

#[derive(Clone, Copy, Debug)]
struct WhittleCandidate {
    graph_hash: GraphHash,
    raw: RawCandidate,
}

impl WhittleCandidate {
    fn matched_slice(&self) -> &[u32] {
        self.raw.matched_slice()
    }
}

#[derive(Default)]
struct CandidateArena {
    handles: Vec<CandidateSlot>,
    free_handles: Vec<u32>,
    batches: Vec<Option<CandidateBatch>>,
    free_batches: Vec<u32>,
}

impl CandidateArena {
    fn insert_batch(
        &mut self,
        graph_hash: GraphHash,
        raw: Vec<RawCandidate>,
    ) -> EngineResult<Vec<WhittleCandidateId>> {
        if raw.is_empty() {
            return Ok(Vec::new());
        }
        let record_count = u32::try_from(raw.len())
            .map_err(|_| internal_error(7, "Whittle candidate batch exceeds u32"))?;
        let records = raw
            .into_iter()
            .map(|raw| CandidateRecord { raw })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let batch = CandidateBatch {
            graph_hash,
            live_handles: records.len(),
            records,
        };
        let batch_index = if let Some(index) = self.free_batches.pop() {
            let slot = &mut self.batches[index as usize];
            debug_assert!(slot.is_none());
            *slot = Some(batch);
            index
        } else {
            let index = u32::try_from(self.batches.len())
                .map_err(|_| internal_error(7, "Whittle candidate batch arena exceeds u32"))?;
            self.batches.push(Some(batch));
            index
        };

        let mut ids = Vec::with_capacity(record_count as usize);
        for record_index in 0..record_count {
            ids.push(self.insert_handle(CandidateLocation {
                batch: batch_index,
                record: record_index,
            })?);
        }
        Ok(ids)
    }

    fn insert_handle(&mut self, location: CandidateLocation) -> EngineResult<WhittleCandidateId> {
        if let Some(index) = self.free_handles.pop() {
            let slot = &mut self.handles[index as usize];
            debug_assert!(slot.location.is_none());
            slot.location = Some(location);
            slot.refs = 1;
            Ok(WhittleCandidateId::from_slot(index, slot.generation()))
        } else {
            let index = u32::try_from(self.handles.len())
                .map_err(|_| internal_error(7, "Whittle candidate handle arena exceeds u32"))?;
            self.handles.push(CandidateSlot::new(location));
            Ok(WhittleCandidateId::from_slot(index, 0))
        }
    }

    fn retain(&mut self, id: WhittleCandidateId) -> EngineResult<()> {
        let Some(slot) = self.handles.get_mut(id.raw() as usize) else {
            return Err(EngineError::UnknownCandidate {
                candidate_hash: None,
            });
        };
        slot.assert_generation(id.generation(), "stale WhittleCandidateId");
        if slot.location.is_none() {
            return Err(EngineError::UnknownCandidate {
                candidate_hash: None,
            });
        }
        slot.refs = slot
            .refs
            .checked_add(1)
            .ok_or_else(|| internal_error(6, "Whittle candidate refcount overflow"))?;
        Ok(())
    }

    fn get(&self, id: WhittleCandidateId) -> Option<WhittleCandidate> {
        let slot = self.handles.get(id.raw() as usize)?;
        slot.assert_generation(id.generation(), "stale WhittleCandidateId");
        let location = slot.location?;
        let batch = self.batches.get(location.batch as usize)?.as_ref()?;
        let record = batch.records.get(location.record as usize)?;
        Some(WhittleCandidate {
            graph_hash: batch.graph_hash,
            raw: record.raw,
        })
    }

    fn release(&mut self, id: WhittleCandidateId) -> EngineResult<Option<WhittleCandidate>> {
        let candidate = self.get(id).ok_or(EngineError::UnknownCandidate {
            candidate_hash: None,
        })?;
        let slot = self
            .handles
            .get_mut(id.raw() as usize)
            .expect("candidate was validated above");
        if slot.refs > 1 {
            slot.refs -= 1;
            return Ok(None);
        }
        let location = slot.location.take().ok_or(EngineError::UnknownCandidate {
            candidate_hash: None,
        })?;
        slot.refs = 0;
        slot.bump_generation();
        self.free_handles.push(id.raw());

        let batch_slot = self
            .batches
            .get_mut(location.batch as usize)
            .ok_or_else(|| {
                internal_error(7, "Whittle candidate handle references missing batch")
            })?;
        let batch = batch_slot.as_mut().ok_or_else(|| {
            internal_error(7, "Whittle candidate handle references released batch")
        })?;
        batch.live_handles = batch
            .live_handles
            .checked_sub(1)
            .ok_or_else(|| internal_error(7, "Whittle candidate batch refcount underflow"))?;
        if batch.live_handles == 0 {
            *batch_slot = None;
            self.free_batches.push(location.batch);
        }
        Ok(Some(candidate))
    }

    fn live_count(&self) -> usize {
        self.handles.len() - self.free_handles.len()
    }

    fn ref_count(&self) -> u64 {
        self.handles.iter().map(|slot| u64::from(slot.refs)).sum()
    }

    fn storage_stats(&self) -> CandidateStorageStats {
        let records_live = self
            .batches
            .iter()
            .filter_map(Option::as_ref)
            .map(|batch| batch.records.len())
            .sum::<usize>();
        CandidateStorageStats {
            handle_slots: self.handles.len(),
            handle_bytes: self.handles.capacity() * std::mem::size_of::<CandidateSlot>(),
            handle_slot_size: std::mem::size_of::<CandidateSlot>(),
            batches_live: self.batches.iter().filter(|batch| batch.is_some()).count(),
            records_live,
            record_bytes: records_live * std::mem::size_of::<CandidateRecord>(),
            record_size: std::mem::size_of::<CandidateRecord>(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CandidateLocation {
    batch: u32,
    record: u32,
}

struct CandidateBatch {
    graph_hash: GraphHash,
    records: Box<[CandidateRecord]>,
    live_handles: usize,
}

#[derive(Clone, Copy, Debug)]
struct CandidateRecord {
    raw: RawCandidate,
}

#[derive(Default)]
struct CandidateSlot {
    location: Option<CandidateLocation>,
    refs: u32,
    #[cfg(debug_assertions)]
    generation: u32,
}

impl CandidateSlot {
    fn new(location: CandidateLocation) -> Self {
        Self {
            location: Some(location),
            refs: 1,
            #[cfg(debug_assertions)]
            generation: 0,
        }
    }

    #[cfg(debug_assertions)]
    const fn generation(&self) -> u32 {
        self.generation
    }

    #[cfg(not(debug_assertions))]
    const fn generation(&self) -> u32 {
        0
    }

    #[cfg(debug_assertions)]
    fn assert_generation(&self, expected: u32, message: &'static str) {
        assert_eq!(self.generation, expected, "{message}");
    }

    #[cfg(not(debug_assertions))]
    fn assert_generation(&self, _expected: u32, _message: &'static str) {}

    #[cfg(debug_assertions)]
    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    #[cfg(not(debug_assertions))]
    fn bump_generation(&mut self) {}
}

struct CachedCandidateIds {
    ids: Vec<WhittleCandidateId>,
    /// The enumeration was cut at a caller limit; a later call with a
    /// larger (or absent) limit must re-enumerate instead of hitting.
    truncated: bool,
}

#[derive(Default)]
struct WhittleCaches {
    candidates: HashMap<(GraphHash, ActionSetHash), CachedCandidateIds>,
    transitions: HashMap<GraphHash, HashMap<CandidateHash, GraphHash>>,
}

pub struct WhittleContractFixture;

impl EngineContractFixture for WhittleContractFixture {
    type Engine = WhittleEngine;

    fn make_engine(&self) -> Self::Engine {
        WhittleEngine::new(WhittleEngineConfig {
            root: WhittleRoot::Artifact(and_idempotent_artifact()),
            ..WhittleEngineConfig::default()
        })
        .expect("Whittle contract fixture config is valid")
    }

    fn measure_options(&self) -> MeasureOptions {
        self.make_engine().measure_options()
    }

    fn known_path(&self) -> Vec<<Self::Engine as GraphEngine>::Candidate> {
        vec![WhittleCandidateId::from_raw(0)]
    }

    fn unknown_graph(&self) -> Option<<Self::Engine as GraphEngine>::Graph> {
        Some(WhittleGraphId::from_raw(u32::MAX))
    }

    fn unknown_candidate(&self) -> Option<<Self::Engine as GraphEngine>::Candidate> {
        Some(WhittleCandidateId::from_raw(u32::MAX))
    }
}

fn and_idempotent_artifact() -> Vec<u8> {
    serialize_wav1(&GraphBody {
        arity: 1,
        capacity: 16,
        output_node: 2,
        op: vec![OpCode::Input, OpCode::And, OpCode::Output],
        arg0: vec![0, 0, 1],
        arg1: vec![u32::MAX, 0, u32::MAX],
    })
}

fn truth_table_seed(
    config: &WhittleGraphGeneratorConfig,
    rng: &mut WhittleRng,
) -> Result<GraphBody, WhittleGeneratorConfigError> {
    let bit_count = 1u32 << config.arity;
    let max_terms = u32::from(config.exception_terms_max).min(bit_count / 2);
    let min_terms = u32::from(config.exception_terms_min).min(max_terms);
    let term_count = random_u32_inclusive(rng, min_terms, max_terms);
    let selected = sample_assignments(rng, bit_count, term_count);
    let exceptions_are_true = random_bool(rng);

    let mut op = vec![OpCode::Input; config.arity as usize];
    let mut arg0: Vec<_> = (0..u32::from(config.arity)).collect();
    let mut arg1 = vec![NO_NODE; config.arity as usize];
    let mut neg_ref = vec![None; config.arity as usize];
    let mut dnf_root = None;

    for assignment in selected {
        let mut term_root = None;

        for var in 0..u32::from(config.arity) {
            let literal = if ((assignment >> var) & 1) == 1 {
                var
            } else {
                match neg_ref[var as usize] {
                    Some(node) => node,
                    None => {
                        let node =
                            append_node(&mut op, &mut arg0, &mut arg1, OpCode::Not, var, NO_NODE);
                        neg_ref[var as usize] = Some(node);
                        node
                    }
                }
            };

            term_root = Some(match term_root {
                Some(term) => {
                    append_node(&mut op, &mut arg0, &mut arg1, OpCode::And, term, literal)
                }
                None => literal,
            });
        }

        let term = term_root.expect("arity validation guarantees non-empty terms");
        dnf_root = Some(match dnf_root {
            Some(dnf) => append_node(&mut op, &mut arg0, &mut arg1, OpCode::Or, dnf, term),
            None => term,
        });
    }

    let mut root = dnf_root.expect("exception term validation guarantees non-empty DNF");
    if !exceptions_are_true {
        root = append_node(&mut op, &mut arg0, &mut arg1, OpCode::Not, root, NO_NODE);
    }
    let output = append_node(&mut op, &mut arg0, &mut arg1, OpCode::Output, root, NO_NODE);

    if op.len() > usize::from(config.capacity) {
        return Err(WhittleGeneratorConfigError::SeedGraphExceedsCapacity);
    }

    let body = GraphBody::new(config.arity, config.capacity, output, op, arg0, arg1)
        .map_err(|_| WhittleGeneratorConfigError::SeedGraphExceedsCapacity)?;
    compact_graph(&body).map_err(|_| WhittleGeneratorConfigError::SeedGraphExceedsCapacity)
}

fn sample_assignments(rng: &mut WhittleRng, bit_count: u32, term_count: u32) -> Vec<u32> {
    let mut selected = Vec::with_capacity(term_count as usize);
    let mut seen = HashSet::with_capacity(term_count as usize);

    while selected.len() < term_count as usize {
        let assignment = random_u32_exclusive(rng, bit_count);
        if seen.insert(assignment) {
            selected.push(assignment);
        }
    }

    selected
}

fn choose_prewalk_candidate(
    candidates: &[RawCandidate],
    blocked_rule: Option<u16>,
    rng: &mut WhittleRng,
) -> Option<RawCandidate> {
    let total: f64 = candidates
        .iter()
        .filter(|candidate| Some(candidate.rule_id) != blocked_rule)
        .map(|candidate| category_weight(candidate.rule_id))
        .sum();

    if total <= 0.0 {
        return None;
    }

    let mut target = random_f64(rng) * total;

    for candidate in candidates
        .iter()
        .copied()
        .filter(|candidate| Some(candidate.rule_id) != blocked_rule)
    {
        let weight = category_weight(candidate.rule_id);
        if target <= weight {
            return Some(candidate);
        }
        target -= weight;
    }

    candidates
        .iter()
        .copied()
        .rev()
        .find(|candidate| Some(candidate.rule_id) != blocked_rule)
}

fn append_node(
    op: &mut Vec<OpCode>,
    arg0: &mut Vec<u32>,
    arg1: &mut Vec<u32>,
    code: OpCode,
    a: u32,
    b: u32,
) -> u32 {
    op.push(code);
    arg0.push(a);
    arg1.push(b);
    (op.len() - 1) as u32
}

fn random_u16_inclusive(rng: &mut WhittleRng, low: u16, high: u16) -> u16 {
    use rand::RngExt;

    rng.random_range(low..=high)
}

fn random_u32_inclusive(rng: &mut WhittleRng, low: u32, high: u32) -> u32 {
    use rand::RngExt;

    rng.random_range(low..=high)
}

fn random_u32_exclusive(rng: &mut WhittleRng, high: u32) -> u32 {
    use rand::RngExt;

    rng.random_range(0..high)
}

fn random_bool(rng: &mut WhittleRng) -> bool {
    use rand::RngExt;

    rng.random_bool(0.5)
}

fn random_f64(rng: &mut WhittleRng) -> f64 {
    use rand::RngExt;

    rng.random()
}

fn candidate_metadata(candidate: &WhittleCandidate) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1 + 2 + 4 + 1 + usize::from(candidate.raw.match_len) * 4);
    bytes.push(1);
    bytes.extend_from_slice(&candidate.raw.rule_id.to_le_bytes());
    bytes.extend_from_slice(&candidate.raw.root.to_le_bytes());
    bytes.push(candidate.raw.match_len);
    for node in candidate.matched_slice() {
        bytes.extend_from_slice(&node.to_le_bytes());
    }
    bytes
}

fn measure_metadata(cost: u32, arity: u16, capacity: u16) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(9);
    bytes.push(1);
    bytes.extend_from_slice(&cost.to_le_bytes());
    bytes.extend_from_slice(&arity.to_le_bytes());
    bytes.extend_from_slice(&capacity.to_le_bytes());
    bytes
}

fn engine_id() -> EngineId {
    let bytes = hash32(b"gz-engine-whittle", &[]);
    let mut id = [0; 16];
    id.copy_from_slice(&bytes[..16]);
    EngineId::from_bytes(id)
}

fn engine_version() -> EngineVersion {
    let bytes = hash32(b"whittle-rules-v1", &[&44u16.to_le_bytes(), b"WAV1", &[1]]);
    let mut id = [0; 16];
    id.copy_from_slice(&bytes[..16]);
    EngineVersion::from_bytes(id)
}

fn action_set_hash(
    engine_version: EngineVersion,
    include_reverse_constant_folding: bool,
) -> ActionSetHash {
    ActionSetHash::from_bytes(hash32(
        b"whittle-action-set-v1",
        &[
            engine_version.as_bytes().as_slice(),
            &[u8::from(include_reverse_constant_folding)],
        ],
    ))
}

fn graph_hash(engine_id: EngineId, engine_version: EngineVersion, canonical: &[u8]) -> GraphHash {
    GraphHash::from_bytes(hash32(
        b"whittle-graph-v1",
        &[
            engine_id.as_bytes().as_slice(),
            engine_version.as_bytes().as_slice(),
            canonical,
        ],
    ))
}

fn candidate_hash(
    graph_hash: GraphHash,
    action_set_hash: ActionSetHash,
    candidate: RawCandidate,
) -> CandidateHash {
    let mut fixed = Vec::with_capacity(2 + 4 + 1 + usize::from(candidate.match_len) * 4);
    fixed.extend_from_slice(&candidate.rule_id.to_le_bytes());
    fixed.extend_from_slice(&candidate.root.to_le_bytes());
    fixed.push(candidate.match_len);
    for node in candidate.matched_slice() {
        fixed.extend_from_slice(&node.to_le_bytes());
    }

    CandidateHash::from_bytes(hash32(
        b"whittle-candidate-v1",
        &[
            graph_hash.as_bytes().as_slice(),
            action_set_hash.as_bytes().as_slice(),
            &fixed,
        ],
    ))
}

fn measure_config_hash(mode: WhittleMeasureMode) -> MeasureConfigHash {
    let mode = match mode {
        WhittleMeasureMode::NegativeCost => [0],
    };
    MeasureConfigHash::from_bytes(hash32(b"whittle-measure-config-v1", &[&mode]))
}

fn hash32(domain: &[u8], chunks: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);

    for chunk in chunks {
        hasher.update(&(chunk.len() as u64).to_le_bytes());
        hasher.update(chunk);
    }

    *hasher.finalize().as_bytes()
}

fn internal_graph(error: GraphError) -> EngineError {
    internal_error(1, error.to_string())
}

fn internal_rule(error: RuleError) -> EngineError {
    internal_error(2, error.to_string())
}

fn generator_error(error: WhittleGeneratorConfigError) -> EngineError {
    internal_error(3, error.to_string())
}

fn internal_error(code: u32, message: impl Into<String>) -> EngineError {
    let message = ErrorMessage::new(message).unwrap_or_else(|_| {
        ErrorMessage::new("whittle internal error").expect("fallback message is valid")
    });
    EngineError::Internal {
        code: ErrorCode::new(code),
        message,
    }
}
