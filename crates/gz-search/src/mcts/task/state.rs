use super::super::tree::MctsEdge;
use crate::SearchCandidateSummary;
use crate::work::{SearchWork, WorkToken};
use gz_engine::{CandidateHash, ReplayGraphContext};
use gz_eval::{EvalAction, EvalRequest};
use std::collections::HashSet;

pub(super) enum RootTaskState<G, C, R> {
    EmitNodeExpand {
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<R>>,
    },
    EmitNodeEval {
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<R>>,
    },
    Running(RunState<R>),
    Done,
}

// Keeping node expansion inline avoids an allocation on every evaluation handoff.
#[allow(clippy::large_enum_variant)]
pub(super) enum PendingRootWork<G, C, R> {
    ExpandNode {
        token: WorkToken,
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<R>>,
    },
    EvalNode {
        token: WorkToken,
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<R>>,
        request: EvalRequest,
    },
    Apply {
        token: WorkToken,
        run: RunState<R>,
        action: usize,
    },
}

impl<G, C, R> PendingRootWork<G, C, R> {
    pub(super) const fn token(&self) -> WorkToken {
        match self {
            Self::ExpandNode { token, .. }
            | Self::EvalNode { token, .. }
            | Self::Apply { token, .. } => *token,
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

pub(super) struct RunState<R> {
    pub(super) strategy: R,
    pub(super) simulations: usize,
    pub(super) descent: Option<DescentState>,
}

impl<R> RunState<R> {
    pub(super) fn complete_simulation(&mut self) {
        self.simulations += 1;
        self.descent = None;
    }
}

pub(super) struct DescentState {
    pub(super) node_index: usize,
    pub(super) depth: usize,
    pub(super) path: Vec<MctsEdge>,
    pub(super) seen: HashSet<ReplayGraphContext>,
    pub(super) forced: Option<usize>,
}

#[allow(clippy::large_enum_variant)]
pub(super) enum DescentPoll<G, C, R> {
    Continue(RunState<R>),
    Work(SearchWork<G, C>, PendingRootWork<G, C, R>),
}
