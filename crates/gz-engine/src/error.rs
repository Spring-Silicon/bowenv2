//! Engine error types.

use crate::{CandidateHash, GraphHash};
use std::fmt;

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum EngineError {
    UnknownGraph {
        graph_hash: Option<GraphHash>,
    },
    UnknownCandidate {
        candidate_hash: Option<CandidateHash>,
    },
    StaleCandidate {
        expected_graph_hash: GraphHash,
        actual_graph_hash: GraphHash,
        candidate_hash: CandidateHash,
    },
    Timeout {
        operation: OperationKind,
        limit_ms: u64,
    },
    Internal {
        code: ErrorCode,
        message: ErrorMessage,
    },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownGraph {
                graph_hash: Some(graph_hash),
            } => write!(f, "unknown graph {graph_hash}"),
            Self::UnknownGraph { graph_hash: None } => write!(f, "unknown graph"),
            Self::UnknownCandidate {
                candidate_hash: Some(candidate_hash),
            } => write!(f, "unknown candidate {candidate_hash}"),
            Self::UnknownCandidate {
                candidate_hash: None,
            } => write!(f, "unknown candidate"),
            Self::StaleCandidate {
                expected_graph_hash,
                actual_graph_hash,
                candidate_hash,
            } => write!(
                f,
                "stale candidate {candidate_hash}: expected graph {expected_graph_hash}, got {actual_graph_hash}"
            ),
            Self::Timeout {
                operation,
                limit_ms,
            } => write!(f, "{operation} timed out after {limit_ms} ms"),
            Self::Internal { code, message } => {
                write!(f, "internal engine error: code {code}: {message}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct ErrorCode(u32);

impl ErrorCode {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ErrorMessage(String);

impl ErrorMessage {
    pub const MAX_LEN: usize = 512;

    pub fn new(message: impl Into<String>) -> Result<Self, ErrorMessageTooLong> {
        let message = message.into();
        let len = message.len();

        if len > Self::MAX_LEN {
            return Err(ErrorMessageTooLong {
                max: Self::MAX_LEN,
                actual: len,
            });
        }

        Ok(Self(message))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ErrorMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for ErrorMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ErrorMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let message = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::new(message).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ErrorMessageTooLong {
    pub max: usize,
    pub actual: usize,
}

impl fmt::Display for ErrorMessageTooLong {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error message too long: max {}, got {}",
            self.max, self.actual
        )
    }
}

impl std::error::Error for ErrorMessageTooLong {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum OperationKind {
    Root,
    Hash,
    Candidates,
    CandidateInfo,
    Apply,
    Measure,
    ExportGraph,
}

impl fmt::Display for OperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Root => "root",
            Self::Hash => "hash",
            Self::Candidates => "candidates",
            Self::CandidateInfo => "candidate_info",
            Self::Apply => "apply",
            Self::Measure => "measure",
            Self::ExportGraph => "export_graph",
        })
    }
}
