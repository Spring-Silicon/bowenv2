use super::super::schedule::{
    GumbelRng, budget_fraction, considered_actions, considered_visit_sequence, overlap_noise_scale,
    root_seed, sample_root_gumbels, softmax,
};
use super::super::{
    GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelPlayer, GumbelRootStats, GumbelStep,
    GumbelValueMode,
};
use crate::support::{internal, step_ref};
use crate::work::{
    ApplyWork, EngineIdentity, EvalOpponentWork, EvalWork, ExpandResult, ExpandWork, SearchPoll,
    SearchWork, SearchWorkResult, WorkToken,
};
use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    ApplyResult, CandidateHash, EngineResult, PortableCandidateRef, PortableSearchActionRef,
    ReplayGraphContext,
};
use gz_eval::{
    EvalAction, EvalOutput, EvalPositionContext, EvalRequest, eval_error_to_engine_error,
};
use std::collections::HashSet;
use std::hash::Hash;

mod reuse;
mod task;
mod tree;
mod wave;

const PLAYER_SALT: u64 = 0x7379_6d6d_5f70_6c79;

pub enum SymmetricRootResult<G, C> {
    Action(Box<SymmetricRootAction<G, C>>),
    Pass {
        player: GumbelPlayer,
        context: ReplayGraphContext,
    },
}

pub struct SymmetricRootAction<G, C> {
    pub step: GumbelStep<G, C>,
    pub player: GumbelPlayer,
    pub selected_after: G,
    pub selected_after_context: ReplayGraphContext,
    pub stats: GumbelRootStats,
}

pub struct SymmetricSelfplayRootTask<G, C> {
    config: GumbelMctsConfig,
    identity: EngineIdentity,
    noise_seed: u64,
    visited: [HashSet<ReplayGraphContext>; 2],
    root_context: Option<ReplayGraphContext>,
    root_candidates: Vec<RootCandidate>,
    nodes: Vec<Node<G, C>>,
    created_graphs: Vec<G>,
    speculative_graphs: Vec<G>,
    created_candidates: Vec<C>,
    releasable: GumbelHandleBatch<G, C>,
    reused: Option<ReusedTree<G, C>>,
    eval_count: usize,
    carried_nodes: usize,
    carried_root_visits: u32,
    next_token: u64,
    pending: Option<Pending<G, C>>,
    state: State<G, C>,
}

fn candidate_ref(context: ReplayGraphContext, hash: CandidateHash) -> PortableSearchActionRef {
    PortableSearchActionRef::candidate(PortableCandidateRef::new(context, hash))
}

#[derive(Clone, Copy)]
struct Board<G> {
    graphs: [G; 2],
    contexts: [Option<ReplayGraphContext>; 2],
    rewrites: [usize; 2],
    inactive: [bool; 2],
    stopped: [bool; 2],
    player: GumbelPlayer,
}

impl<G: Copy> Board<G> {
    fn current_graph(self) -> G {
        self.graphs[self.player.index()]
    }

    fn active(self, max_steps: usize, player: GumbelPlayer) -> bool {
        !self.inactive[player.index()] && self.rewrites[player.index()] < max_steps
    }

    fn terminal(self, max_steps: usize) -> bool {
        !self.active(max_steps, GumbelPlayer::One) && !self.active(max_steps, GumbelPlayer::Two)
    }
}

#[derive(Clone, Copy)]
struct RootCandidate {
    hash: CandidateHash,
    summary: SearchCandidateSummary,
}

#[derive(Clone, Copy)]
struct CandidateEntry<C> {
    handle: C,
    hash: CandidateHash,
    summary: SearchCandidateSummary,
}

struct Expansion<C> {
    context: ReplayGraphContext,
    candidates: Vec<CandidateEntry<C>>,
    eval_actions: Vec<EvalAction>,
}

#[derive(Clone)]
struct Node<G, C> {
    board: Board<G>,
    context: ReplayGraphContext,
    candidates: Vec<C>,
    candidate_metadata: Vec<RootCandidate>,
    logits: Vec<f32>,
    priors: Vec<f32>,
    value: f32,
    model_version: gz_engine::ModelVersion,
    masked: Vec<bool>,
    edges: Vec<ActionEdge<G>>,
    pass: Option<Branch>,
}

struct ReusedTree<G, C> {
    nodes: Vec<Node<G, C>>,
    created_graphs: Vec<G>,
    created_candidates: Vec<C>,
    carried_nodes: usize,
    carried_root_visits: u32,
}

impl<G, C> Node<G, C> {
    fn action_count(&self) -> usize {
        self.edges.len()
    }

    fn is_stop(&self, action: usize) -> bool {
        action == self.candidates.len() && self.action_count() > self.candidates.len()
    }

    fn action_refs(&self, candidates: &[RootCandidate]) -> Vec<PortableSearchActionRef> {
        let mut actions = candidates
            .iter()
            .map(|candidate| candidate_ref(self.context, candidate.hash))
            .collect::<Vec<_>>();
        if self.action_count() > self.candidates.len() {
            actions.push(PortableSearchActionRef::stop(self.context));
        }
        actions
    }

    fn total_visits(&self) -> u32 {
        self.edges.iter().map(|edge| edge.visits).sum()
    }

    fn max_visits(&self) -> u32 {
        self.edges.iter().map(|edge| edge.visits).max().unwrap_or(0)
    }

    fn search_value(&self) -> f32 {
        let visits = self.total_visits();
        if visits == 0 {
            self.value
        } else {
            self.edges.iter().map(|edge| edge.value_sum).sum::<f32>() / visits as f32
        }
    }
}

#[derive(Clone)]
struct ActionEdge<G> {
    branch: Option<Branch>,
    after_graph: Option<G>,
    after_context: Option<ReplayGraphContext>,
    visits: u32,
    value_sum: f32,
}

impl<G> Default for ActionEdge<G> {
    fn default() -> Self {
        Self {
            branch: None,
            after_graph: None,
            after_context: None,
            visits: 0,
            value_sum: 0.0,
        }
    }
}

impl<G> ActionEdge<G> {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Copy)]
struct Branch {
    target: BranchTarget,
    flip: bool,
}

#[derive(Clone, Copy)]
enum BranchTarget {
    Node(usize),
    Terminal(f32),
}

fn same_board<G: Eq>(left: Board<G>, right: Board<G>) -> bool {
    left.graphs == right.graphs
        && left.contexts == right.contexts
        && left.rewrites == right.rewrites
        && left.inactive == right.inactive
        && left.stopped == right.stopped
        && left.player == right.player
}

fn remap_branch(branch: &mut Branch, remap: &[Option<usize>]) -> EngineResult<()> {
    let BranchTarget::Node(old_index) = branch.target else {
        return Ok(());
    };
    let new_index = remap
        .get(old_index)
        .copied()
        .flatten()
        .ok_or_else(|| internal("symmetric subtree child was not retained"))?;
    branch.target = BranchTarget::Node(new_index);
    Ok(())
}

struct Run {
    base_scores: Vec<f32>,
    considered: Vec<usize>,
    schedule: Vec<u32>,
    root_visit_baseline: Vec<u32>,
    schedule_index: usize,
    simulations: usize,
}

struct WaveRun<G, C> {
    run: Run,
    active: Vec<WaveSimulation<G, C>>,
    root_apply_preflight: bool,
}

impl<G, C> WaveRun<G, C> {
    fn new(run: Run) -> Self {
        Self {
            run,
            active: Vec::new(),
            root_apply_preflight: false,
        }
    }

    fn root_apply_preflight_ready(&self) -> bool {
        self.root_apply_preflight
            && !self.active.is_empty()
            && self.active.iter().all(|simulation| {
                matches!(simulation.state, WaveSimulationState::RootApplyReady(_))
            })
    }

    fn has_pending(&self, token: WorkToken) -> bool {
        self.active
            .iter()
            .any(|simulation| simulation.state.pending_token() == Some(token))
    }

    fn any_pending(&self) -> bool {
        self.active
            .iter()
            .any(|simulation| simulation.state.pending_token().is_some())
    }
}

struct WaveSimulation<G, C> {
    state: WaveSimulationState<G, C>,
}

struct RootApplyReady<G, C> {
    descent: Descent,
    node: usize,
    action: usize,
    applied: ApplyResult<G, C>,
}

#[allow(clippy::large_enum_variant)]
enum WaveSimulationState<G, C> {
    Running(Descent),
    Resolve {
        descent: Descent,
        board: Board<G>,
        attach: Attach,
    },
    ApplyPending {
        token: WorkToken,
        descent: Descent,
        node: usize,
        action: usize,
    },
    RootApplyReady(RootApplyReady<G, C>),
    ExpandPending {
        token: WorkToken,
        descent: Descent,
        board: Board<G>,
        attach: Attach,
    },
    EvalReady {
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
    },
    EvalPending {
        token: WorkToken,
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
        request: EvalRequest,
    },
    EvalComplete {
        descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
        request: EvalRequest,
        output: EvalOutput,
    },
    TerminalEvalPending {
        token: WorkToken,
        descent: Descent,
        attach: Attach,
        request: EvalRequest,
    },
    TerminalEvalComplete {
        descent: Descent,
        attach: Attach,
        request: EvalRequest,
        output: EvalOutput,
    },
    Outcome {
        descent: Descent,
        value: f32,
    },
    Vacant,
}

impl<G, C> WaveSimulationState<G, C> {
    fn pending_token(&self) -> Option<WorkToken> {
        match self {
            Self::ApplyPending { token, .. }
            | Self::ExpandPending { token, .. }
            | Self::EvalPending { token, .. }
            | Self::TerminalEvalPending { token, .. } => Some(*token),
            _ => None,
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum WaveAdvance<G, C> {
    Work(SearchWork<G, C>),
    Progressed,
    Waiting,
}

fn wave_target_state<G, C>(
    mut descent: Descent,
    target: BranchTarget,
) -> WaveSimulationState<G, C> {
    match target {
        BranchTarget::Node(node) => {
            descent.node = node;
            WaveSimulationState::Running(descent)
        }
        BranchTarget::Terminal(value) => WaveSimulationState::Outcome { descent, value },
    }
}

struct Descent {
    node: usize,
    path: Vec<PathStep>,
    seen: [HashSet<ReplayGraphContext>; 2],
    forced: Option<usize>,
}

#[derive(Clone, Copy)]
enum PathStep {
    Decision {
        node: usize,
        action: usize,
        flip: bool,
    },
    Transform {
        flip: bool,
    },
}

#[derive(Clone, Copy)]
struct Attach {
    slot: AttachSlot,
    turns: u8,
}

#[derive(Clone, Copy)]
enum AttachSlot {
    Action { node: usize, action: usize },
    Pass { node: usize },
}

#[allow(clippy::large_enum_variant)]
enum State<G, C> {
    Resolve {
        board: Board<G>,
    },
    Eval {
        board: Board<G>,
        expansion: Expansion<C>,
    },
    WaveRunning(WaveRun<G, C>),
    DoneResult(SymmetricRootResult<G, C>),
    Done,
}

#[allow(clippy::large_enum_variant)]
enum Pending<G, C> {
    Expand {
        token: WorkToken,
        board: Board<G>,
    },
    Eval {
        token: WorkToken,
        board: Board<G>,
        expansion: Expansion<C>,
        request: EvalRequest,
    },
}

impl<G, C> Pending<G, C> {
    fn token(&self) -> WorkToken {
        match self {
            Self::Expand { token, .. } | Self::Eval { token, .. } => *token,
        }
    }
}
