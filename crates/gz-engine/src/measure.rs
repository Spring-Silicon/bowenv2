//! Measurement result and summary types.

use crate::{ErrorCode, ErrorMessage, GraphHash, MeasureConfigHash};
use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MeasureResult<G> {
    pub graph: G,
    pub graph_hash: GraphHash,
    pub config_hash: MeasureConfigHash,
    pub measured: bool,
    pub valid: bool,
    pub latency: Option<LatencyStats>,
    /// Terminal utility, maximized by search.
    pub scalar_reward: Option<f32>,
    pub failure: Option<MeasureFailure>,
    pub metadata: MeasureMetadata,
}

impl<G> MeasureResult<G> {
    pub fn validate(self) -> Result<Self, MeasurementValidationError> {
        let Self {
            graph,
            graph_hash,
            config_hash,
            measured,
            valid,
            latency,
            scalar_reward,
            failure,
            metadata,
        } = self;

        if let Some(scalar_reward) = scalar_reward
            && !scalar_reward.is_finite()
        {
            return Err(MeasurementValidationError::NonFiniteScalarReward { scalar_reward });
        }

        let latency = match latency {
            Some(latency) => Some(latency.validate()?),
            None => None,
        };

        Ok(Self {
            graph,
            graph_hash,
            config_hash,
            measured,
            valid,
            latency,
            scalar_reward,
            failure,
            metadata,
        })
    }
}

#[cfg(feature = "serde")]
impl<'de, G> serde::Deserialize<'de> for MeasureResult<G>
where
    G: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Unchecked<G> {
            graph: G,
            graph_hash: GraphHash,
            config_hash: MeasureConfigHash,
            measured: bool,
            valid: bool,
            latency: Option<LatencyStats>,
            scalar_reward: Option<f32>,
            failure: Option<MeasureFailure>,
            metadata: MeasureMetadata,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self {
            graph: unchecked.graph,
            graph_hash: unchecked.graph_hash,
            config_hash: unchecked.config_hash,
            measured: unchecked.measured,
            valid: unchecked.valid,
            latency: unchecked.latency,
            scalar_reward: unchecked.scalar_reward,
            failure: unchecked.failure,
            metadata: unchecked.metadata,
        }
        .validate()
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MeasureSummary {
    pub graph_hash: GraphHash,
    pub config_hash: MeasureConfigHash,
    pub measured: bool,
    pub valid: bool,
    pub latency: Option<LatencyStats>,
    /// Terminal utility, maximized by search.
    pub scalar_reward: Option<f32>,
    pub failure_code: Option<ErrorCode>,
}

impl<G> From<&MeasureResult<G>> for MeasureSummary {
    fn from(result: &MeasureResult<G>) -> Self {
        Self {
            graph_hash: result.graph_hash,
            config_hash: result.config_hash,
            measured: result.measured,
            valid: result.valid,
            latency: result.latency.clone(),
            scalar_reward: result.scalar_reward,
            failure_code: result.failure.as_ref().map(|failure| failure.code),
        }
    }
}

impl MeasureSummary {
    pub fn validate(self) -> Result<Self, MeasurementValidationError> {
        if let Some(scalar_reward) = self.scalar_reward
            && !scalar_reward.is_finite()
        {
            return Err(MeasurementValidationError::NonFiniteScalarReward { scalar_reward });
        }

        let latency = match self.latency {
            Some(latency) => Some(latency.validate()?),
            None => None,
        };

        Ok(Self { latency, ..self })
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MeasureSummary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Unchecked {
            graph_hash: GraphHash,
            config_hash: MeasureConfigHash,
            measured: bool,
            valid: bool,
            latency: Option<LatencyStats>,
            scalar_reward: Option<f32>,
            failure_code: Option<ErrorCode>,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self {
            graph_hash: unchecked.graph_hash,
            config_hash: unchecked.config_hash,
            measured: unchecked.measured,
            valid: unchecked.valid,
            latency: unchecked.latency,
            scalar_reward: unchecked.scalar_reward,
            failure_code: unchecked.failure_code,
        }
        .validate()
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LatencyStats {
    pub mean_ms: f32,
    pub median_ms: f32,
    pub p95_ms: f32,
    pub samples_ms: Vec<f32>,
}

impl LatencyStats {
    pub fn new(
        mean_ms: f32,
        median_ms: f32,
        p95_ms: f32,
        samples_ms: Vec<f32>,
    ) -> Result<Self, MeasurementValidationError> {
        Self {
            mean_ms,
            median_ms,
            p95_ms,
            samples_ms,
        }
        .validate()
    }

    pub fn from_samples(samples_ms: Vec<f32>) -> Result<Self, MeasurementValidationError> {
        validate_latency_values(&samples_ms)?;

        if samples_ms.is_empty() {
            return Self::new(0.0, 0.0, 0.0, samples_ms);
        }

        let mut sorted = samples_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let mean_ms = samples_ms.iter().sum::<f32>() / samples_ms.len() as f32;
        let median_ms = percentile_nearest_rank(&sorted, 0.50);
        let p95_ms = percentile_nearest_rank(&sorted, 0.95);

        Self::new(mean_ms, median_ms, p95_ms, samples_ms)
    }

    pub fn validate(self) -> Result<Self, MeasurementValidationError> {
        validate_latency_value("mean_ms", self.mean_ms)?;
        validate_latency_value("median_ms", self.median_ms)?;
        validate_latency_value("p95_ms", self.p95_ms)?;
        validate_latency_values(&self.samples_ms)?;
        Ok(self)
    }
}

fn percentile_nearest_rank(sorted: &[f32], percentile: f32) -> f32 {
    let rank = (percentile * sorted.len() as f32).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn validate_latency_values(values: &[f32]) -> Result<(), MeasurementValidationError> {
    for (index, value) in values.iter().copied().enumerate() {
        validate_latency_value("samples_ms", value).map_err(|source| {
            MeasurementValidationError::InvalidLatencySample {
                index,
                value,
                source: Box::new(source),
            }
        })?;
    }

    Ok(())
}

fn validate_latency_value(
    field: &'static str,
    value: f32,
) -> Result<(), MeasurementValidationError> {
    if !value.is_finite() {
        return Err(MeasurementValidationError::NonFiniteLatency { field, value });
    }

    if value < 0.0 {
        return Err(MeasurementValidationError::NegativeLatency { field, value });
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct MeasureFailure {
    pub code: ErrorCode,
    pub message: ErrorMessage,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct MeasureMetadata {
    pub bytes: Vec<u8>,
}

#[derive(Debug, PartialEq)]
pub enum MeasurementValidationError {
    NonFiniteLatency {
        field: &'static str,
        value: f32,
    },
    NegativeLatency {
        field: &'static str,
        value: f32,
    },
    InvalidLatencySample {
        index: usize,
        value: f32,
        source: Box<MeasurementValidationError>,
    },
    NonFiniteScalarReward {
        scalar_reward: f32,
    },
}

impl fmt::Display for MeasurementValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteLatency { field, value } => {
                write!(f, "{field} must be finite, got {value}")
            }
            Self::NegativeLatency { field, value } => {
                write!(f, "{field} must be non-negative, got {value}")
            }
            Self::InvalidLatencySample {
                index,
                value,
                source,
            } => {
                write!(
                    f,
                    "invalid latency sample at index {index} with value {value}: {source}"
                )
            }
            Self::NonFiniteScalarReward { scalar_reward } => {
                write!(f, "scalar_reward must be finite, got {scalar_reward}")
            }
        }
    }
}

impl std::error::Error for MeasurementValidationError {}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for LatencyStats {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Unchecked {
            mean_ms: f32,
            median_ms: f32,
            p95_ms: f32,
            samples_ms: Vec<f32>,
        }

        let unchecked = Unchecked::deserialize(deserializer)?;
        Self::new(
            unchecked.mean_ms,
            unchecked.median_ms,
            unchecked.p95_ms,
            unchecked.samples_ms,
        )
        .map_err(serde::de::Error::custom)
    }
}
