use crate::types::{EvalOutput, EvalRequest};
use gz_engine::ModelVersion;
use std::collections::HashMap;

/// Opt-in memo of network evaluations, keyed by everything the model
/// sees: graph identity, action-set identity, action count, position
/// features, and opponent context. A hit returns the exact bytes a fresh
/// evaluation of the same model version would return (rows are computed
/// independently within a batch), so enabling the cache changes cost,
/// never behavior.
///
/// Model swaps invalidate implicitly: outputs carry their model version,
/// and the first insert from a newer version clears the cache. Entries
/// may serve for at most one eval round-trip after a swap lands (until
/// that first new-version reply) -- the same staleness window as an
/// in-flight batch.
///
/// Eviction is two-generation rotation: inserts fill the current
/// generation; when it reaches half capacity the previous generation is
/// dropped. Hits promote entries into the current generation, so a
/// steadily reused working set survives rotation.
pub struct NnEvalCache {
    half_capacity: usize,
    current: HashMap<EvalCacheKey, EvalOutput>,
    previous: HashMap<EvalCacheKey, EvalOutput>,
    version: Option<ModelVersion>,
    hits: u64,
    misses: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct EvalCacheKey {
    graph: gz_engine::PortableGraphId,
    action_set_hash: gz_engine::ActionSetHash,
    action_count: u32,
    root_step: u32,
    leaf_depth: u32,
    budget_fraction_bits: u32,
    budget_step_bits: u32,
    opponent: Option<(u64, u32)>,
}

impl EvalCacheKey {
    fn new(request: &EvalRequest) -> Self {
        let position = request.position;
        Self {
            graph: request.context.graph,
            action_set_hash: request.context.action_set_hash,
            action_count: request.actions.len() as u32,
            root_step: position.root_step,
            leaf_depth: position.leaf_depth,
            budget_fraction_bits: position.budget_fraction.to_bits(),
            budget_step_bits: position.budget_step.to_bits(),
            opponent: position
                .opponent
                .map(|opponent| (opponent.trajectory_id, opponent.row_count)),
        }
    }
}

impl NnEvalCache {
    /// `capacity` bounds the total retained entries across both
    /// generations.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let half_capacity = (capacity / 2).max(1);
        Self {
            half_capacity,
            current: HashMap::with_capacity(half_capacity),
            previous: HashMap::new(),
            version: None,
            hits: 0,
            misses: 0,
        }
    }

    pub fn lookup(&mut self, request: &EvalRequest) -> Option<EvalOutput> {
        let key = EvalCacheKey::new(request);
        if let Some(output) = self.current.get(&key) {
            self.hits += 1;
            return Some(output.clone());
        }
        if let Some(output) = self.previous.remove(&key) {
            self.hits += 1;
            self.insert_rotating(key, output.clone());
            return Some(output);
        }
        self.misses += 1;
        None
    }

    pub fn insert(&mut self, request: &EvalRequest, output: &EvalOutput) {
        match self.version {
            Some(version) if version == output.model_version => {}
            Some(_) => {
                // A reply from a different version: newer replies flush the
                // cache; stragglers from before the swap are dropped. Both
                // cases key off "not the version we cached for".
                self.current.clear();
                self.previous.clear();
                self.version = Some(output.model_version);
            }
            None => self.version = Some(output.model_version),
        }
        self.insert_rotating(EvalCacheKey::new(request), output.clone());
    }

    fn insert_rotating(&mut self, key: EvalCacheKey, output: EvalOutput) {
        if self.current.len() >= self.half_capacity {
            self.previous = std::mem::take(&mut self.current);
            self.current = HashMap::with_capacity(self.half_capacity);
        }
        self.current.insert(key, output);
    }

    #[must_use]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    #[must_use]
    pub const fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EvalAction, EvalPositionContext};
    use gz_engine::{
        ActionSetHash, EngineId, EngineVersion, GraphHash, PortableGraphId, ReplayGraphContext,
    };

    fn request(graph: u8, root_step: u32) -> EvalRequest {
        let context = ReplayGraphContext::new(
            PortableGraphId::new(
                GraphHash::from_bytes([graph; 32]),
                EngineId::from_bytes([1; 16]),
                EngineVersion::from_bytes([2; 16]),
            ),
            ActionSetHash::from_bytes([3; 32]),
        );
        EvalRequest::with_position(
            context,
            vec![EvalAction::stop(context)],
            EvalPositionContext {
                root_step,
                leaf_depth: 0,
                budget_fraction: 1.0,
                budget_step: 0.5,
                opponent: None,
            },
        )
        .unwrap()
    }

    fn output(version: u8, value: f32) -> EvalOutput {
        EvalOutput {
            model_version: ModelVersion::from_bytes([version; 16]),
            policy_logits: vec![value, -value],
            value,
        }
    }

    #[test]
    fn hits_key_on_graph_and_position() {
        let mut cache = NnEvalCache::new(8);
        cache.insert(&request(1, 0), &output(1, 0.5));

        assert_eq!(cache.lookup(&request(1, 0)), Some(output(1, 0.5)));
        assert_eq!(cache.lookup(&request(1, 1)), None);
        assert_eq!(cache.lookup(&request(2, 0)), None);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 2);
    }

    #[test]
    fn version_change_flushes() {
        let mut cache = NnEvalCache::new(8);
        cache.insert(&request(1, 0), &output(1, 0.5));
        cache.insert(&request(2, 0), &output(2, 0.25));

        assert_eq!(cache.lookup(&request(1, 0)), None);
        assert_eq!(cache.lookup(&request(2, 0)), Some(output(2, 0.25)));

        // A straggler reply from the old version must not repopulate.
        cache.insert(&request(3, 0), &output(1, 0.75));
        assert_eq!(cache.lookup(&request(2, 0)), None);
    }

    #[test]
    fn rotation_keeps_promoted_entries() {
        let mut cache = NnEvalCache::new(4);
        cache.insert(&request(1, 0), &output(1, 0.1));
        cache.insert(&request(2, 0), &output(1, 0.2));
        // Promote graph 1 out of the elder generation.
        assert!(cache.lookup(&request(1, 0)).is_some());
        cache.insert(&request(3, 0), &output(1, 0.3));
        cache.insert(&request(4, 0), &output(1, 0.4));
        cache.insert(&request(5, 0), &output(1, 0.5));

        assert!(cache.lookup(&request(5, 0)).is_some());
    }
}
