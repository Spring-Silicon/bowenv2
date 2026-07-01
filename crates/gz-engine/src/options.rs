//! Engine operation option types.

use crate::MeasureConfigHash;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct CandidateOptions {
    pub max_candidates: Option<usize>,
    pub deterministic_order: bool,
}

impl Default for CandidateOptions {
    fn default() -> Self {
        Self {
            max_candidates: None,
            deterministic_order: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MeasureOptions {
    pub config_hash: MeasureConfigHash,
    pub samples: u32,
    pub timeout_ms: Option<u64>,
    pub deterministic: bool,
}

impl MeasureOptions {
    pub fn new(
        config_hash: MeasureConfigHash,
        samples: u32,
        timeout_ms: Option<u64>,
        deterministic: bool,
    ) -> Result<Self, MeasureOptionsError> {
        if samples == 0 {
            return Err(MeasureOptionsError::ZeroSamples);
        }

        if timeout_ms == Some(0) {
            return Err(MeasureOptionsError::ZeroTimeout);
        }

        Ok(Self {
            config_hash,
            samples,
            timeout_ms,
            deterministic,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasureOptionsError {
    ZeroSamples,
    ZeroTimeout,
}

impl fmt::Display for MeasureOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSamples => write!(f, "measure options require at least one sample"),
            Self::ZeroTimeout => write!(f, "measure timeout must be greater than zero"),
        }
    }
}

impl std::error::Error for MeasureOptionsError {}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MeasureOptions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Unchecked {
            config_hash: MeasureConfigHash,
            samples: u32,
            timeout_ms: Option<u64>,
            deterministic: bool,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self::new(
            unchecked.config_hash,
            unchecked.samples,
            unchecked.timeout_ms,
            unchecked.deterministic,
        )
        .map_err(serde::de::Error::custom)
    }
}
