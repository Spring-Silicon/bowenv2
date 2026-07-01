//! Portable graph and candidate reference types.

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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SearchStepRef {
    pub before: ReplayGraphContext,
    pub candidate: PortableCandidateRef,
    pub after: ReplayGraphContext,
}

impl SearchStepRef {
    pub fn new(
        before: ReplayGraphContext,
        candidate: PortableCandidateRef,
        after: ReplayGraphContext,
    ) -> Result<Self, SearchStepRefError> {
        if candidate.context != before {
            return Err(SearchStepRefError::CandidateContextMismatch {
                before: Box::new(before),
                candidate_context: Box::new(candidate.context),
            });
        }

        Ok(Self {
            before,
            candidate,
            after,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SearchStepRefError {
    CandidateContextMismatch {
        before: Box<ReplayGraphContext>,
        candidate_context: Box<ReplayGraphContext>,
    },
}

impl fmt::Display for SearchStepRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateContextMismatch { .. } => {
                write!(f, "candidate context does not match step before context")
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
            candidate: PortableCandidateRef,
            after: ReplayGraphContext,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self::new(unchecked.before, unchecked.candidate, unchecked.after)
            .map_err(serde::de::Error::custom)
    }
}
