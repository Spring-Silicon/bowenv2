use crate::{EvalError, EvalOutput, EvalRequest, EvalResult, Evaluator, validate_outputs};
use gz_engine::ModelVersion;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomValueEvaluatorConfig {
    pub seed: u64,
    pub value_min: f32,
    pub value_max: f32,
}

impl Default for RandomValueEvaluatorConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            value_min: -1.0,
            value_max: 1.0,
        }
    }
}

impl RandomValueEvaluatorConfig {
    pub fn validate(self) -> EvalResult<Self> {
        if !self.value_min.is_finite()
            || !self.value_max.is_finite()
            || self.value_min > self.value_max
        {
            return Err(EvalError::InvalidValueRange {
                value_min: self.value_min,
                value_max: self.value_max,
            });
        }

        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct RandomValueEvaluator {
    config: RandomValueEvaluatorConfig,
    model_version: ModelVersion,
}

impl RandomValueEvaluator {
    pub fn new(config: RandomValueEvaluatorConfig) -> EvalResult<Self> {
        let config = config.validate()?;
        let model_version = model_version(config);

        Ok(Self {
            config,
            model_version,
        })
    }

    #[must_use]
    pub const fn config(&self) -> RandomValueEvaluatorConfig {
        self.config
    }

    #[must_use]
    pub const fn model_version(&self) -> ModelVersion {
        self.model_version
    }
}

impl Default for RandomValueEvaluator {
    fn default() -> Self {
        Self::new(RandomValueEvaluatorConfig::default()).expect("default config is valid")
    }
}

impl Evaluator for RandomValueEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        out.clear();
        out.reserve(requests.len());

        for request in requests {
            request.validate_ref()?;
            out.push(EvalOutput {
                model_version: self.model_version,
                policy_logits: vec![0.0; request.action_count()],
                value: self.value(request),
            });
        }

        validate_outputs(requests, out)
    }
}

impl RandomValueEvaluator {
    fn value(&self, request: &EvalRequest) -> f32 {
        if self.config.value_min == self.config.value_max {
            return self.config.value_min;
        }

        let mut hasher = blake3::Hasher::new();
        update_chunk(&mut hasher, b"gz-eval-random-value-v1");
        update_u64(&mut hasher, self.config.seed);
        update_chunk(&mut hasher, request.context.graph.graph_hash.as_bytes());
        update_chunk(&mut hasher, request.context.graph.engine_id.as_bytes());
        update_chunk(&mut hasher, request.context.graph.engine_version.as_bytes());
        update_chunk(&mut hasher, request.context.action_set_hash.as_bytes());

        let hash = hasher.finalize();
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&hash.as_bytes()[..8]);
        let unit = u64::from_le_bytes(bytes) as f64 / u64::MAX as f64;
        let min = f64::from(self.config.value_min);
        let max = f64::from(self.config.value_max);

        (min + (max - min) * unit) as f32
    }
}

fn model_version(config: RandomValueEvaluatorConfig) -> ModelVersion {
    let mut hasher = blake3::Hasher::new();
    update_chunk(&mut hasher, b"gz-eval-random-value-model-v1");
    update_u64(&mut hasher, config.seed);
    update_u32(&mut hasher, config.value_min.to_bits());
    update_u32(&mut hasher, config.value_max.to_bits());

    let hash = hasher.finalize();
    let mut bytes = [0; 16];
    bytes.copy_from_slice(&hash.as_bytes()[..16]);
    ModelVersion::from_bytes(bytes)
}

fn update_chunk(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    update_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn update_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn update_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}
