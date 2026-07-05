use super::super::schedule::GumbelRng;
use super::super::tree::Edge;
use crate::SearchCandidateSummary;
use crate::work::{SearchWork, WorkToken};
use gz_engine::{CandidateHash, ReplayGraphContext};
use gz_eval::{EvalAction, EvalRequest};
use std::collections::HashSet;

pub(super) enum RootTaskState<G, C> {
    EmitNodeExpand {
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<G, C>>,
    },
    EmitNodeEval {
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<G, C>>,
    },
    Running(RunState<G, C>),
    Done,
}

pub(super) enum PendingRootWork<G, C> {
    ExpandNode {
        token: WorkToken,
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<G, C>>,
    },
    EvalNode {
        token: WorkToken,
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<G, C>>,
        request: EvalRequest,
    },
    Apply {
        token: WorkToken,
        run: RunState<G, C>,
        action: usize,
    },
    StopEval {
        token: WorkToken,
        run: RunState<G, C>,
        request: EvalRequest,
    },
}

impl<G, C> PendingRootWork<G, C> {
    pub(super) const fn token(&self) -> WorkToken {
        match self {
            Self::ExpandNode { token, .. }
            | Self::EvalNode { token, .. }
            | Self::Apply { token, .. }
            | Self::StopEval { token, .. } => *token,
        }
    }
}

pub(super) struct NodeExpansion<G, C> {
    pub(super) graph: G,
    pub(super) context: ReplayGraphContext,
    pub(super) depth: usize,
    pub(super) candidates: Vec<C>,
    pub(super) eval_actions: Vec<EvalAction>,
    pub(super) candidate_hashes: Vec<CandidateHash>,
    pub(super) summaries: Vec<Option<SearchCandidateSummary>>,
}

pub(super) struct RunState<G, C> {
    pub(super) base_scores: Vec<f32>,
    pub(super) considered: Vec<usize>,
    pub(super) schedule: Vec<u32>,
    pub(super) schedule_index: usize,
    pub(super) simulations: usize,
    pub(super) rng: GumbelRng,
    pub(super) descent: Option<DescentState>,
    pub(super) _marker: std::marker::PhantomData<(G, C)>,
}

impl<G, C> RunState<G, C> {
    pub(super) fn complete_simulation(&mut self) {
        self.simulations += 1;
        self.schedule_index += 1;
        self.descent = None;
    }
}

pub(super) struct DescentState {
    pub(super) node_index: usize,
    pub(super) depth: usize,
    pub(super) path: Vec<Edge>,
    pub(super) seen: HashSet<ReplayGraphContext>,
    pub(super) forced: Option<usize>,
}

#[allow(clippy::large_enum_variant)]
pub(super) enum DescentPoll<G, C> {
    Continue(RunState<G, C>),
    Work(SearchWork<G, C>, PendingRootWork<G, C>),
}
