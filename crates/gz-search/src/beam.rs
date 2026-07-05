use crate::support::{candidate_info, graph_context, graph_context_from_hash, score, step_ref};
use crate::{SearchAction, SearchCandidateSummary, SearchStep, beam_search_config_hash};
use gz_engine::{
    CandidateOptions, EngineResult, GraphEngine, GraphHash, MeasureOptions, MeasureResult,
    MeasureSummary, PortableCandidateRef, PortableSearchActionRef, ReplayGraphContext,
    SearchConfigHash,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::num::NonZeroUsize;

#[derive(Clone, Debug, PartialEq)]
pub struct BeamEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<SearchStep<G, C>>,
    pub layers: Vec<BeamLayer<G>>,
    pub created_graphs: Vec<G>,
    pub created_candidates: Vec<C>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: BeamStopReason,
    pub search_config_hash: SearchConfigHash,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeamLayer<G> {
    pub depth: usize,
    pub entries: Vec<BeamEntrySummary<G>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeamEntrySummary<G> {
    pub graph: G,
    pub context: ReplayGraphContext,
    pub measure: MeasureSummary,
    pub reward: f32,
    pub stopped: bool,
    pub carried: bool,
    pub parent_index: Option<usize>,
    pub selected_action: Option<PortableSearchActionRef>,
    pub selected_rank: Option<usize>,
    pub engine_candidate_count: Option<usize>,
    pub action_count: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BeamSearchConfig {
    pub max_depth: usize,
    pub beam_width: NonZeroUsize,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

pub struct BeamSearch {
    config: BeamSearchConfig,
    search_config_hash: SearchConfigHash,
}

impl BeamSearch {
    #[must_use]
    pub fn new(config: BeamSearchConfig) -> Self {
        let search_config_hash = beam_search_config_hash(
            config.max_depth,
            config.beam_width.get(),
            config.candidate_options,
            config.measure_options,
        );

        Self {
            config,
            search_config_hash,
        }
    }

    #[must_use]
    pub const fn config(&self) -> BeamSearchConfig {
        self.config
    }

    #[must_use]
    pub const fn search_config_hash(&self) -> SearchConfigHash {
        self.search_config_hash
    }

    pub fn run_from_root<E: GraphEngine>(
        &self,
        engine: &mut E,
    ) -> EngineResult<BeamEpisode<E::Graph, E::Candidate>> {
        self.run(engine, engine.root())
    }

    pub fn run<E: GraphEngine>(
        &self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<BeamEpisode<E::Graph, E::Candidate>> {
        let root_context = graph_context(engine, root)?;
        let root_measure = engine.measure(root, self.config.measure_options)?;

        let Some(root_reward) = score(&root_measure) else {
            return Ok(BeamEpisode {
                root,
                final_graph: root,
                root_context,
                final_context: root_context,
                steps: Vec::new(),
                layers: Vec::new(),
                created_graphs: Vec::new(),
                created_candidates: Vec::new(),
                final_measure: root_measure,
                stop_reason: BeamStopReason::UnscoredRoot,
                search_config_hash: self.search_config_hash,
            });
        };

        let mut nodes = vec![BeamNode {
            graph: root,
            context: root_context,
            measure: root_measure,
            parent: None,
            incoming: None,
            layer_index: Some(0),
            stopped: false,
            rank: BeamRank {
                reward: root_reward,
                stopped: false,
                depth: 0,
                static_prior: 0.0,
                action_ref: PortableSearchActionRef::stop(root_context),
            },
        }];
        let mut layers = vec![BeamLayer {
            depth: 0,
            entries: vec![root_entry(
                root,
                root_context,
                &nodes[0].measure,
                root_reward,
            )],
        }];
        let mut active = vec![0];
        let mut scratch = BeamScratch::default();
        let mut created_graphs = Vec::new();
        let mut created_candidates = Vec::new();

        for _ in 0..self.config.max_depth {
            scratch.entries.clear();
            scratch.new_nodes.clear();
            scratch.best_by_graph.clear();
            scratch.top_entries.clear();
            scratch.selected_entries.clear();
            scratch.next_active.clear();
            scratch.next_active.reserve(self.config.beam_width.get());

            for node_id in active.iter().copied() {
                let node = &nodes[node_id];

                if node.stopped {
                    scratch.entries.push(BeamEntry::Existing(node_id));
                    continue;
                }

                expand_node(
                    engine,
                    self.config,
                    node_id,
                    node,
                    &mut scratch,
                    &mut created_graphs,
                    &mut created_candidates,
                )?;
            }

            scratch.best_by_graph.reserve(scratch.entries.len());
            for entry in scratch.entries.drain(..) {
                keep_best_graph_entry(
                    &nodes,
                    &scratch.new_nodes,
                    &mut scratch.best_by_graph,
                    entry,
                );
            }

            scratch.top_entries.reserve(self.config.beam_width.get());
            for entry in scratch.best_by_graph.values().copied() {
                let ranked = RankedEntry {
                    rank: entry.rank(&nodes, &scratch.new_nodes),
                    entry,
                };

                if scratch.top_entries.len() < self.config.beam_width.get() {
                    scratch.top_entries.push(ranked);
                    continue;
                }

                let worst = scratch
                    .top_entries
                    .peek()
                    .expect("non-empty heap after beam_width entries");
                if rank_order(&ranked.rank, &worst.rank).is_gt() {
                    scratch.top_entries.pop();
                    scratch.top_entries.push(ranked);
                }
            }

            scratch
                .selected_entries
                .reserve(self.config.beam_width.get());
            scratch.selected_entries.extend(scratch.top_entries.drain());
            scratch
                .selected_entries
                .sort_by(|left, right| rank_order(&right.rank, &left.rank));

            for ranked in scratch.selected_entries.drain(..) {
                match ranked.entry {
                    BeamEntry::Existing(node_id) => {
                        scratch.next_active.push(NextActive {
                            node_id,
                            parent_index: nodes[node_id].layer_index,
                            carried: true,
                        });
                    }
                    BeamEntry::New(node_id) => {
                        let node = scratch.new_nodes[node_id]
                            .take()
                            .expect("selected pending beam node exists");
                        let parent_index = node.parent.and_then(|parent| nodes[parent].layer_index);
                        nodes.push(node);
                        scratch.next_active.push(NextActive {
                            node_id: nodes.len() - 1,
                            parent_index,
                            carried: false,
                        });
                    }
                }
            }

            let depth = layers.len();
            let entries = scratch
                .next_active
                .iter()
                .map(|entry| summarize_entry(&nodes, entry))
                .collect();

            for (index, entry) in scratch.next_active.iter().enumerate() {
                nodes[entry.node_id].layer_index = Some(index);
            }

            active.clear();
            active.extend(scratch.next_active.iter().map(|entry| entry.node_id));
            layers.push(BeamLayer { depth, entries });

            if active.iter().all(|node_id| nodes[*node_id].stopped) {
                break;
            }
        }

        let best = active
            .into_iter()
            .max_by(|left, right| rank_order(&nodes[*left].rank, &nodes[*right].rank))
            .expect("beam always contains at least the root node");
        let selected = &nodes[best];
        let stop_reason = if selected.stopped {
            BeamStopReason::SelectedStop
        } else {
            BeamStopReason::MaxDepth
        };

        Ok(BeamEpisode {
            root,
            final_graph: selected.graph,
            root_context,
            final_context: selected.context,
            steps: collect_steps(&nodes, best),
            layers,
            created_graphs,
            created_candidates,
            final_measure: selected.measure.clone(),
            stop_reason,
            search_config_hash: self.search_config_hash,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BeamStopReason {
    MaxDepth,
    SelectedStop,
    UnscoredRoot,
}

struct BeamNode<G, C> {
    graph: G,
    context: ReplayGraphContext,
    measure: MeasureResult<G>,
    parent: Option<usize>,
    incoming: Option<SearchStep<G, C>>,
    layer_index: Option<usize>,
    stopped: bool,
    rank: BeamRank,
}

struct NextActive {
    node_id: usize,
    parent_index: Option<usize>,
    carried: bool,
}

struct BeamScratch<G, C> {
    candidates: Vec<C>,
    new_nodes: Vec<Option<BeamNode<G, C>>>,
    entries: Vec<BeamEntry>,
    best_by_graph: HashMap<GraphHash, BeamEntry>,
    top_entries: BinaryHeap<RankedEntry>,
    selected_entries: Vec<RankedEntry>,
    next_active: Vec<NextActive>,
}

impl<G, C> Default for BeamScratch<G, C> {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
            new_nodes: Vec::new(),
            entries: Vec::new(),
            best_by_graph: HashMap::new(),
            top_entries: BinaryHeap::new(),
            selected_entries: Vec::new(),
            next_active: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
enum BeamEntry {
    Existing(usize),
    New(usize),
}

impl BeamEntry {
    fn context<G, C>(
        &self,
        nodes: &[BeamNode<G, C>],
        new_nodes: &[Option<BeamNode<G, C>>],
    ) -> ReplayGraphContext {
        match self {
            Self::Existing(node_id) => nodes[*node_id].context,
            Self::New(node_id) => pending_node(new_nodes, *node_id).context,
        }
    }

    fn rank<G, C>(
        &self,
        nodes: &[BeamNode<G, C>],
        new_nodes: &[Option<BeamNode<G, C>>],
    ) -> BeamRank {
        match self {
            Self::Existing(node_id) => nodes[*node_id].rank,
            Self::New(node_id) => pending_node(new_nodes, *node_id).rank,
        }
    }
}

#[derive(Clone, Copy)]
struct BeamRank {
    reward: f32,
    stopped: bool,
    depth: usize,
    static_prior: f32,
    action_ref: PortableSearchActionRef,
}

#[derive(Clone, Copy)]
struct RankedEntry {
    rank: BeamRank,
    entry: BeamEntry,
}

impl Eq for RankedEntry {}

impl PartialEq for RankedEntry {
    fn eq(&self, other: &Self) -> bool {
        rank_order(&self.rank, &other.rank).is_eq()
    }
}

impl Ord for RankedEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        rank_order(&other.rank, &self.rank)
    }
}

impl PartialOrd for RankedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn expand_node<E: GraphEngine>(
    engine: &mut E,
    config: BeamSearchConfig,
    node_id: usize,
    node: &BeamNode<E::Graph, E::Candidate>,
    scratch: &mut BeamScratch<E::Graph, E::Candidate>,
    created_graphs: &mut Vec<E::Graph>,
    created_candidates: &mut Vec<E::Candidate>,
) -> EngineResult<()> {
    engine.candidates(
        node.graph,
        config.candidate_options,
        &mut scratch.candidates,
    )?;
    created_candidates.extend(scratch.candidates.iter().copied());

    let engine_candidate_count = scratch.candidates.len();
    let stop_ref = PortableSearchActionRef::stop(node.context);
    let stop_rank = engine_candidate_count;
    let action_count = engine_candidate_count + 1;

    scratch.new_nodes.reserve(action_count);
    scratch.entries.reserve(action_count);

    for (selected_rank, candidate) in scratch.candidates.iter().copied().enumerate() {
        let info = candidate_info(engine, node.graph, candidate)?;
        let static_prior = info.static_prior;
        let candidate_ref = PortableCandidateRef::new(node.context, info.candidate_hash);
        let action_ref = PortableSearchActionRef::candidate(candidate_ref);

        let applied = engine.apply(node.graph, candidate)?;
        created_graphs.push(applied.after);
        if applied.rejected.is_some() {
            continue;
        }

        let measure = engine.measure(applied.after, config.measure_options)?;
        let Some(reward) = score(&measure) else {
            continue;
        };

        let after_context = graph_context_from_hash(engine, applied.after_hash);
        let step_ref = step_ref(node.context, action_ref, after_context)?;
        let selected_measure = MeasureSummary::from(&measure);

        scratch.new_nodes.push(Some(BeamNode {
            graph: applied.after,
            context: after_context,
            measure,
            parent: Some(node_id),
            incoming: Some(SearchStep {
                before: node.graph,
                after: applied.after,
                action: SearchAction::Candidate(candidate),
                step_ref,
                selected_action: action_ref,
                selected_candidate: Some(SearchCandidateSummary {
                    kind: info.kind,
                    tags: info.tags,
                    static_prior,
                }),
                selected_measure,
                engine_candidate_count,
                action_count,
                selected_rank,
            }),
            layer_index: None,
            stopped: false,
            rank: BeamRank {
                reward,
                stopped: false,
                depth: node.rank.depth + 1,
                static_prior,
                action_ref,
            },
        }));
        scratch
            .entries
            .push(BeamEntry::New(scratch.new_nodes.len() - 1));
    }

    let step_ref = step_ref(node.context, stop_ref, node.context)?;
    let selected_measure = MeasureSummary::from(&node.measure);

    scratch.new_nodes.push(Some(BeamNode {
        graph: node.graph,
        context: node.context,
        measure: node.measure.clone(),
        parent: Some(node_id),
        incoming: Some(SearchStep {
            before: node.graph,
            after: node.graph,
            action: SearchAction::Stop,
            step_ref,
            selected_action: stop_ref,
            selected_candidate: None,
            selected_measure,
            engine_candidate_count,
            action_count,
            selected_rank: stop_rank,
        }),
        layer_index: None,
        stopped: true,
        rank: BeamRank {
            reward: node.rank.reward,
            stopped: true,
            depth: node.rank.depth + 1,
            static_prior: 0.0,
            action_ref: stop_ref,
        },
    }));
    scratch
        .entries
        .push(BeamEntry::New(scratch.new_nodes.len() - 1));

    Ok(())
}

fn keep_best_graph_entry<G, C>(
    nodes: &[BeamNode<G, C>],
    new_nodes: &[Option<BeamNode<G, C>>],
    best_by_graph: &mut HashMap<GraphHash, BeamEntry>,
    entry: BeamEntry,
) {
    let graph_hash = entry.context(nodes, new_nodes).graph.graph_hash;

    match best_by_graph.get_mut(&graph_hash) {
        Some(best) => {
            let entry_rank = entry.rank(nodes, new_nodes);
            let best_rank = best.rank(nodes, new_nodes);
            if rank_order(&entry_rank, &best_rank).is_gt() {
                *best = entry;
            }
        }
        None => {
            best_by_graph.insert(graph_hash, entry);
        }
    }
}

fn root_entry<G: Copy>(
    graph: G,
    context: ReplayGraphContext,
    measure: &MeasureResult<G>,
    reward: f32,
) -> BeamEntrySummary<G> {
    BeamEntrySummary {
        graph,
        context,
        measure: MeasureSummary::from(measure),
        reward,
        stopped: false,
        carried: false,
        parent_index: None,
        selected_action: None,
        selected_rank: None,
        engine_candidate_count: None,
        action_count: None,
    }
}

fn summarize_entry<G: Copy, C>(
    nodes: &[BeamNode<G, C>],
    entry: &NextActive,
) -> BeamEntrySummary<G> {
    let node = &nodes[entry.node_id];
    let incoming = if entry.carried {
        None
    } else {
        node.incoming.as_ref()
    };

    BeamEntrySummary {
        graph: node.graph,
        context: node.context,
        measure: MeasureSummary::from(&node.measure),
        reward: node.rank.reward,
        stopped: node.stopped,
        carried: entry.carried,
        parent_index: entry.parent_index,
        selected_action: incoming.map(|step| step.selected_action),
        selected_rank: incoming.map(|step| step.selected_rank),
        engine_candidate_count: incoming.map(|step| step.engine_candidate_count),
        action_count: incoming.map(|step| step.action_count),
    }
}

fn pending_node<G, C>(new_nodes: &[Option<BeamNode<G, C>>], node_id: usize) -> &BeamNode<G, C> {
    new_nodes[node_id]
        .as_ref()
        .expect("pending beam node exists")
}

fn collect_steps<G: Clone, C: Clone>(
    nodes: &[BeamNode<G, C>],
    mut node_id: usize,
) -> Vec<SearchStep<G, C>> {
    let mut steps = Vec::new();

    while let Some(parent) = nodes[node_id].parent {
        let step = nodes[node_id]
            .incoming
            .as_ref()
            .expect("non-root beam nodes have incoming steps")
            .clone();
        steps.push(step);
        node_id = parent;
    }

    steps.reverse();
    steps
}

fn rank_order(left: &BeamRank, right: &BeamRank) -> Ordering {
    left.reward
        .total_cmp(&right.reward)
        .then_with(|| left.stopped.cmp(&right.stopped))
        .then_with(|| left.static_prior.total_cmp(&right.static_prior))
        .then_with(|| right.action_ref.cmp(&left.action_ref))
        .then_with(|| right.depth.cmp(&left.depth))
}
