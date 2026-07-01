//! Candidate and artifact metadata types.

use crate::{CandidateHash, ErrorCode, ErrorMessage, GraphHash};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct CandidateInfo {
    pub candidate_hash: CandidateHash,
    pub graph_hash: GraphHash,
    pub action_set_hash: crate::ActionSetHash,
    pub kind: CandidateKindId,
    pub display_name: String,
    pub static_prior: f32,
    pub tags: CandidateTags,
    pub subjects: Vec<SubjectId>,
    pub metadata: CandidateMetadata,
}

impl CandidateInfo {
    pub fn validate(self) -> Result<Self, CandidateInfoError> {
        if !self.static_prior.is_finite() {
            return Err(CandidateInfoError::NonFiniteStaticPrior {
                static_prior: self.static_prior,
            });
        }

        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CandidateInfoError {
    NonFiniteStaticPrior { static_prior: f32 },
}

impl fmt::Display for CandidateInfoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteStaticPrior { static_prior } => {
                write!(
                    f,
                    "candidate static_prior must be finite, got {static_prior}"
                )
            }
        }
    }
}

impl std::error::Error for CandidateInfoError {}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CandidateInfo {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Unchecked {
            candidate_hash: CandidateHash,
            graph_hash: GraphHash,
            action_set_hash: crate::ActionSetHash,
            kind: CandidateKindId,
            display_name: String,
            static_prior: f32,
            tags: CandidateTags,
            subjects: Vec<SubjectId>,
            metadata: CandidateMetadata,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self {
            candidate_hash: unchecked.candidate_hash,
            graph_hash: unchecked.graph_hash,
            action_set_hash: unchecked.action_set_hash,
            kind: unchecked.kind,
            display_name: unchecked.display_name,
            static_prior: unchecked.static_prior,
            tags: unchecked.tags,
            subjects: unchecked.subjects,
            metadata: unchecked.metadata,
        }
        .validate()
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct CandidateKindId(u32);

impl CandidateKindId {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct SubjectId(u64);

impl SubjectId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct CandidateTags(u64);

impl CandidateTags {
    pub const EMPTY: Self = Self(0);

    #[must_use]
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, tag: Self) -> bool {
        (self.0 & tag.0) == tag.0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct CandidateMetadata {
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ApplyJob<G, C> {
    pub graph: G,
    pub candidate: C,
}

impl<G, C> ApplyJob<G, C> {
    #[must_use]
    pub const fn new(graph: G, candidate: C) -> Self {
        Self { graph, candidate }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ApplyResult<G, C> {
    pub before: G,
    pub after: G,
    pub before_hash: GraphHash,
    pub after_hash: GraphHash,
    pub candidate: C,
    pub candidate_hash: CandidateHash,
    pub changed: bool,
    pub rejected: Option<RewriteRejection>,
    pub metrics: ApplyMetrics,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct RewriteRejection {
    pub code: ErrorCode,
    pub message: ErrorMessage,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ApplyMetrics {
    pub elapsed_ms: Option<f32>,
    pub engine_steps: Option<u64>,
}

impl ApplyMetrics {
    pub fn new(
        elapsed_ms: Option<f32>,
        engine_steps: Option<u64>,
    ) -> Result<Self, ApplyValidationError> {
        let metrics = Self {
            elapsed_ms,
            engine_steps,
        };
        metrics.validate()
    }

    pub fn validate(self) -> Result<Self, ApplyValidationError> {
        if let Some(elapsed_ms) = self.elapsed_ms {
            if !elapsed_ms.is_finite() {
                return Err(ApplyValidationError::NonFiniteElapsedMs { elapsed_ms });
            }

            if elapsed_ms < 0.0 {
                return Err(ApplyValidationError::NegativeElapsedMs { elapsed_ms });
            }
        }

        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ApplyValidationError {
    NonFiniteElapsedMs { elapsed_ms: f32 },
    NegativeElapsedMs { elapsed_ms: f32 },
}

impl fmt::Display for ApplyValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteElapsedMs { elapsed_ms } => {
                write!(f, "apply elapsed_ms must be finite, got {elapsed_ms}")
            }
            Self::NegativeElapsedMs { elapsed_ms } => {
                write!(f, "apply elapsed_ms must be non-negative, got {elapsed_ms}")
            }
        }
    }
}

impl std::error::Error for ApplyValidationError {}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ApplyMetrics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Unchecked {
            elapsed_ms: Option<f32>,
            engine_steps: Option<u64>,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self::new(unchecked.elapsed_ms, unchecked.engine_steps).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct GraphArtifact {
    pub graph_hash: GraphHash,
    pub format: GraphArtifactFormat,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum GraphArtifactFormat {
    Text,
    Json,
    Dot,
    Binary,
    AdapterSpecific(u32),
}
