//! Portable graph and action reference types.

use crate::{ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PortableGraphId {
    pub graph_hash: GraphHash,
    pub engine_id: EngineId,
    pub engine_version: EngineVersion,
}

impl PortableGraphId {
    #[must_use]
    pub const fn new(
        graph_hash: GraphHash,
        engine_id: EngineId,
        engine_version: EngineVersion,
    ) -> Self {
        Self {
            graph_hash,
            engine_id,
            engine_version,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ReplayGraphContext {
    pub graph: PortableGraphId,
    pub action_set_hash: ActionSetHash,
}

impl ReplayGraphContext {
    #[must_use]
    pub const fn new(graph: PortableGraphId, action_set_hash: ActionSetHash) -> Self {
        Self {
            graph,
            action_set_hash,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PortableCandidateRef {
    pub context: ReplayGraphContext,
    pub candidate_hash: CandidateHash,
}

impl PortableCandidateRef {
    #[must_use]
    pub const fn new(context: ReplayGraphContext, candidate_hash: CandidateHash) -> Self {
        Self {
            context,
            candidate_hash,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum PortableSearchActionRef {
    Candidate(PortableCandidateRef),
    Stop { context: ReplayGraphContext },
}

impl PortableSearchActionRef {
    #[must_use]
    pub const fn candidate(candidate: PortableCandidateRef) -> Self {
        Self::Candidate(candidate)
    }

    #[must_use]
    pub const fn stop(context: ReplayGraphContext) -> Self {
        Self::Stop { context }
    }

    #[must_use]
    pub const fn context(self) -> ReplayGraphContext {
        match self {
            Self::Candidate(candidate) => candidate.context,
            Self::Stop { context } => context,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SearchStepRef {
    pub before: ReplayGraphContext,
    pub action: PortableSearchActionRef,
    pub after: ReplayGraphContext,
}

impl SearchStepRef {
    pub fn new(
        before: ReplayGraphContext,
        action: PortableSearchActionRef,
        after: ReplayGraphContext,
    ) -> Result<Self, SearchStepRefError> {
        let action_context = action.context();

        if action_context != before {
            return Err(SearchStepRefError::ActionContextMismatch {
                before: Box::new(before),
                action_context: Box::new(action_context),
            });
        }

        if matches!(action, PortableSearchActionRef::Stop { .. }) && after != before {
            return Err(SearchStepRefError::StopAfterMismatch {
                before: Box::new(before),
                after: Box::new(after),
            });
        }

        Ok(Self {
            before,
            action,
            after,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SearchStepRefError {
    ActionContextMismatch {
        before: Box<ReplayGraphContext>,
        action_context: Box<ReplayGraphContext>,
    },
    StopAfterMismatch {
        before: Box<ReplayGraphContext>,
        after: Box<ReplayGraphContext>,
    },
}

impl fmt::Display for SearchStepRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActionContextMismatch { .. } => {
                write!(f, "action context does not match step before context")
            }
            Self::StopAfterMismatch { .. } => {
                write!(f, "stop action after context does not match before context")
            }
        }
    }
}

impl std::error::Error for SearchStepRefError {}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SearchStepRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Unchecked {
            before: ReplayGraphContext,
            action: PortableSearchActionRef,
            after: ReplayGraphContext,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self::new(unchecked.before, unchecked.action, unchecked.after)
            .map_err(serde::de::Error::custom)
    }
}
