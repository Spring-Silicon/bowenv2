use crate::{NO_NODE, OpCode, RULE_COUNT, WhittleEngine, WhittleGraphId};
use gz_engine::{GraphEngine, PortableGraphId};
use gz_features::{
    ActionFeature, FeatureEdge, FeatureError, FeatureExtractor, FeatureResult, FeatureRow,
    FeatureSchema, FeatureSchemaConfig, PositionFeatures, STOP_ACTION_KIND_TOKEN,
};
use std::collections::HashMap;

const WHITTLE_MAX_ACTIONS: u32 = 256;
const WHITTLE_MAX_SUBJECTS: u32 = 8;
const WHITTLE_ENGINE_EDGE_TYPES: u8 = 2;
/// Input nodes keep their slot identity as distinct tokens (the engine
/// caps arity at 16). Vocab: 0 pad, 1..=16 input slots, 17/18 const
/// false/true, 19 not, 20 and, 21 or, 22 output. Collapsing inputs and
/// constant polarity into single tokens made semantically opposite
/// rewrites (x AND 0 vs x AND 1) indistinguishable to the model.
const WHITTLE_INPUT_SLOT_TOKENS: u16 = 16;
const WHITTLE_NODE_VOCAB: u16 = 23;
/// Per-node structural attrs the trunks cannot recover on their own
/// (max-pool SAGE is count-blind; attention has no positional encoding):
/// fan-out / (n-1), fan-in / 2, depth-from-output / max-depth.
const WHITTLE_NODE_ATTR_DIM: u16 = 3;
const DEFAULT_EXPANDER_DEGREE: u8 = 5;
const DEFAULT_OPPONENT_REWARD_SCALE: f32 = 256.0;

#[derive(Clone, Debug)]
pub struct WhittleFeatureExtractor {
    schema: FeatureSchema,
    state_cache: HashMap<PortableGraphId, CachedStateFeatures>,
    expander_cache: HashMap<u32, Vec<FeatureEdge>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhittleFeatureExtractorConfig {
    pub expander_degree: u8,
    pub expander_seed: u64,
    /// Action rows per position (engine candidates + STOP). Changing it
    /// changes the feature schema hash: checkpoints and stores only match
    /// runs built with the same value.
    pub max_actions: u32,
}

impl Default for WhittleFeatureExtractorConfig {
    fn default() -> Self {
        Self {
            expander_degree: DEFAULT_EXPANDER_DEGREE,
            expander_seed: 0,
            max_actions: WHITTLE_MAX_ACTIONS,
        }
    }
}

impl WhittleFeatureExtractor {
    #[must_use]
    pub fn new(engine: &WhittleEngine) -> Self {
        Self::with_config(engine, WhittleFeatureExtractorConfig::default())
    }

    #[must_use]
    pub fn with_config(engine: &WhittleEngine, config: WhittleFeatureExtractorConfig) -> Self {
        let root = engine.root();
        let root_graph = engine.graph(root).expect("Whittle root graph exists");
        let capacity = u32::from(root_graph.capacity);
        let edge_type_count = if config.expander_degree == 0 {
            WHITTLE_ENGINE_EDGE_TYPES
        } else {
            WHITTLE_ENGINE_EDGE_TYPES + 1
        };
        let schema = FeatureSchema::new(FeatureSchemaConfig {
            name: "whittle-v2".to_string(),
            node_vocab_size: WHITTLE_NODE_VOCAB,
            node_attr_dim: WHITTLE_NODE_ATTR_DIM,
            edge_type_count,
            action_kind_vocab_size: RULE_COUNT + 2,
            max_nodes: capacity,
            max_edges: capacity * 2 + u32::from(config.expander_degree) * capacity,
            max_actions: config.max_actions,
            max_subjects: WHITTLE_MAX_SUBJECTS,
            opponent_reward_scale: DEFAULT_OPPONENT_REWARD_SCALE,
            expander_degree: config.expander_degree,
            expander_seed: config.expander_seed,
        })
        .expect("Whittle feature schema is valid");

        Self {
            schema,
            state_cache: HashMap::new(),
            expander_cache: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn schema(&self) -> &FeatureSchema {
        &self.schema
    }
}

impl FeatureExtractor<WhittleEngine> for WhittleFeatureExtractor {
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn extract(
        &mut self,
        engine: &WhittleEngine,
        graph: WhittleGraphId,
        candidates: &[<WhittleEngine as GraphEngine>::Candidate],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow> {
        let graph_hash = engine.hash(graph)?;
        let graph_id =
            PortableGraphId::new(graph_hash, engine.engine_id(), engine.engine_version());
        // Unbounded selfplay sees millions of distinct graphs; the cache
        // only pays off within a search, so cap it instead of growing it.
        if self.state_cache.len() >= 8192 {
            self.state_cache.clear();
        }
        let state = if let Some(cached) = self.state_cache.get(&graph_id) {
            cached.clone()
        } else {
            let graph_body = engine.graph(graph)?;
            let state = CachedStateFeatures {
                node_count: graph_body.op.len() as u32,
                node_tokens: graph_body
                    .op
                    .iter()
                    .zip(graph_body.arg0.as_ref())
                    .map(|(op, arg0)| node_token(*op, *arg0))
                    .collect(),
                node_attrs: node_attrs(
                    graph_body.op.as_ref(),
                    graph_body.arg0.as_ref(),
                    graph_body.arg1.as_ref(),
                    graph_body.output_node,
                ),
                edges: graph_edges(
                    graph_body.op.as_ref(),
                    graph_body.arg0.as_ref(),
                    graph_body.arg1.as_ref(),
                ),
            };
            self.state_cache.insert(graph_id, state.clone());
            state
        };

        let mut actions = Vec::with_capacity(candidates.len() + 1);
        for candidate in candidates.iter().copied() {
            let info = engine.candidate_info(graph, candidate)?;
            let mut subjects = Vec::with_capacity(info.subjects.len());
            for subject in info.subjects {
                let subject = u32::try_from(subject.get())
                    .map_err(|_| FeatureError::InvalidRow("Whittle subject id overflow"))?;
                subjects.push(subject);
            }
            actions.push(ActionFeature {
                kind_token: info.kind.get() + 2,
                static_prior: info.static_prior,
                subjects,
            });
        }
        actions.push(ActionFeature {
            kind_token: STOP_ACTION_KIND_TOKEN,
            static_prior: 0.0,
            subjects: Vec::new(),
        });

        let mut edges = state.edges;
        self.append_expander_edges(state.node_count, &mut edges);

        let row = FeatureRow {
            node_count: state.node_count,
            node_tokens: state.node_tokens,
            node_attrs: state.node_attrs,
            edges,
            actions,
            position,
            opponent: None,
        };
        row.validate(&self.schema)?;
        Ok(row)
    }
}

impl WhittleFeatureExtractor {
    fn append_expander_edges(&mut self, node_count: u32, out: &mut Vec<FeatureEdge>) {
        let config = self.schema.config();
        if config.expander_degree == 0 || node_count <= 1 {
            return;
        }

        let edge_type = config.edge_type_count - 1;
        let expander_degree = config.expander_degree;
        let expander_seed = config.expander_seed;
        let template = self.expander_cache.entry(node_count).or_insert_with(|| {
            expander_template(node_count, expander_degree, expander_seed, edge_type)
        });
        out.extend_from_slice(template);
    }
}

#[derive(Clone, Debug)]
struct CachedStateFeatures {
    node_count: u32,
    node_tokens: Vec<u16>,
    node_attrs: Vec<f32>,
    edges: Vec<FeatureEdge>,
}

fn node_token(op: OpCode, arg0: u32) -> u16 {
    match op {
        OpCode::Input => 1 + (arg0 as u16).min(WHITTLE_INPUT_SLOT_TOKENS - 1),
        OpCode::Const => WHITTLE_INPUT_SLOT_TOKENS + 1 + u16::from(arg0 != 0),
        OpCode::Not => WHITTLE_INPUT_SLOT_TOKENS + 3,
        OpCode::And => WHITTLE_INPUT_SLOT_TOKENS + 4,
        OpCode::Or => WHITTLE_INPUT_SLOT_TOKENS + 5,
        OpCode::Output => WHITTLE_INPUT_SLOT_TOKENS + 6,
    }
}

fn operands(op: OpCode, arg0: u32, arg1: u32) -> [Option<u32>; 2] {
    let valid = |arg: u32| (arg != NO_NODE).then_some(arg);
    match op {
        OpCode::Input | OpCode::Const => [None, None],
        OpCode::Not | OpCode::Output => [valid(arg0), None],
        OpCode::And | OpCode::Or => [valid(arg0), valid(arg1)],
    }
}

/// whittlezero's structural node features: normalized fan-out (users),
/// fan-in (operands, at most 2), and BFS depth from the output over
/// operand edges, unreachable nodes clamped to depth 0.
fn node_attrs(op: &[OpCode], arg0: &[u32], arg1: &[u32], output_node: u32) -> Vec<f32> {
    let n = op.len();
    let mut fan_out = vec![0u32; n];
    let mut fan_in = vec![0u32; n];
    for (dst, code) in op.iter().copied().enumerate() {
        for operand in operands(code, arg0[dst], arg1[dst]).into_iter().flatten() {
            if (operand as usize) < n {
                fan_out[operand as usize] += 1;
                fan_in[dst] += 1;
            }
        }
    }

    let mut depth = vec![-1i64; n];
    let mut queue = std::collections::VecDeque::new();
    if (output_node as usize) < n {
        depth[output_node as usize] = 0;
        queue.push_back(output_node as usize);
    }
    while let Some(node) = queue.pop_front() {
        for operand in operands(op[node], arg0[node], arg1[node])
            .into_iter()
            .flatten()
        {
            let operand = operand as usize;
            if operand < n && depth[operand] < 0 {
                depth[operand] = depth[node] + 1;
                queue.push_back(operand);
            }
        }
    }
    let max_depth = depth.iter().copied().max().unwrap_or(0).max(1) as f32;

    let fan_out_scale = (n.saturating_sub(1)).max(1) as f32;
    let mut attrs = Vec::with_capacity(n * WHITTLE_NODE_ATTR_DIM as usize);
    for node in 0..n {
        attrs.push(fan_out[node] as f32 / fan_out_scale);
        attrs.push(fan_in[node] as f32 / 2.0);
        attrs.push(depth[node].max(0) as f32 / max_depth);
    }
    attrs
}

fn graph_edges(op: &[OpCode], arg0: &[u32], arg1: &[u32]) -> Vec<FeatureEdge> {
    let mut edges = Vec::with_capacity(op.len().saturating_mul(2));
    for (dst, code) in op.iter().copied().enumerate() {
        match code {
            OpCode::Input | OpCode::Const => {}
            OpCode::Not | OpCode::Output => push_edge(&mut edges, arg0[dst], dst as u32, 0),
            OpCode::And | OpCode::Or => {
                push_edge(&mut edges, arg0[dst], dst as u32, 0);
                push_edge(&mut edges, arg1[dst], dst as u32, 1);
            }
        }
    }
    edges
}

fn push_edge(edges: &mut Vec<FeatureEdge>, src: u32, dst: u32, edge_type: u8) {
    if src != NO_NODE {
        edges.push(FeatureEdge {
            src,
            dst,
            edge_type,
        });
    }
}

fn expander_template(node_count: u32, degree: u8, seed: u64, edge_type: u8) -> Vec<FeatureEdge> {
    let n = node_count as usize;
    let mut edges = Vec::with_capacity(n.saturating_mul(usize::from(degree)));

    for permutation_index in 0..degree {
        let mut permutation = (0..node_count).collect::<Vec<_>>();
        let mut rng = SplitMix64::new(expander_permutation_seed(
            seed,
            u64::from(permutation_index),
            node_count,
        ));

        for index in (1..n).rev() {
            let other = rng.next_bounded((index + 1) as u64) as usize;
            permutation.swap(index, other);
        }

        for (src, dst) in permutation.iter().copied().enumerate() {
            if src as u32 != dst {
                edges.push(FeatureEdge {
                    src: src as u32,
                    dst,
                    edge_type,
                });
            }
        }
    }

    edges
}

fn expander_permutation_seed(seed: u64, permutation_index: u64, node_count: u32) -> u64 {
    let value = mix64(0x677a_2d65_7870_616e);
    let value = mix64(value ^ seed);
    let value = mix64(value ^ permutation_index);
    mix64(value ^ u64::from(node_count))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    fn next_bounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        if bound == 1 {
            return 0;
        }

        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let value = self.next_u64();
            if value < zone {
                return value % bound;
            }
        }
    }
}
