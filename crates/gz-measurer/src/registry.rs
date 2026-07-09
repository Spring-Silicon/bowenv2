use gz_engine::{ModelVersion, ReplayGraphContext, SearchConfigHash};
use gz_features::OpponentStateFeatures;
use gz_replay::ReplayReferenceKind;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceStep {
    pub context: ReplayGraphContext,
    pub features: Option<OpponentStateFeatures>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceSnapshot {
    pub ref_id: u64,
    pub kind: ReplayReferenceKind,
    pub version: ModelVersion,
    pub final_reward: f32,
    pub final_graph: Option<ReplayGraphContext>,
    pub steps: Arc<[ReferenceStep]>,
    pub search_config_hash: SearchConfigHash,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RolloutOutcome {
    pub final_reward: f32,
    pub final_graph: ReplayGraphContext,
    pub steps: Vec<ReferenceStep>,
    pub search_config_hash: SearchConfigHash,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PendingChallenge {
    Versioned(ModelVersion),
    Seed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateEvent {
    pub accepted: bool,
    pub challenger: f32,
    pub best: f32,
    pub steps: usize,
    pub version: ModelVersion,
}

#[derive(Debug, Default)]
pub struct ReferenceRegistry {
    state: Mutex<RegistryState>,
}

#[derive(Debug)]
struct RegistryState {
    current: Option<Arc<ReferenceSnapshot>>,
    last_challenged: Option<ModelVersion>,
    pending: Option<PendingChallenge>,
    next_ref_id: u64,
}

impl Default for RegistryState {
    fn default() -> Self {
        Self {
            current: None,
            last_challenged: None,
            pending: None,
            next_ref_id: 1,
        }
    }
}

impl ReferenceRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn current(&self) -> Option<Arc<ReferenceSnapshot>> {
        self.state
            .lock()
            .expect("reference registry mutex poisoned")
            .current
            .clone()
    }

    #[must_use]
    pub fn admission_ready(&self) -> bool {
        self.state
            .lock()
            .expect("reference registry mutex poisoned")
            .current
            .is_some()
    }

    #[must_use]
    pub fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        let state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        if state.pending.is_some() {
            return false;
        }
        match latest {
            Some(latest) => state.last_challenged != Some(latest),
            None => state.last_challenged.is_none(),
        }
    }

    pub fn claim_challenge(&self, latest: Option<ModelVersion>) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        if state.pending.is_some() {
            return false;
        }
        let due = match latest {
            Some(latest) => state.last_challenged != Some(latest),
            None => state.last_challenged.is_none(),
        };
        if !due {
            return false;
        }
        state.pending = Some(match latest {
            Some(version) => PendingChallenge::Versioned(version),
            None => PendingChallenge::Seed,
        });
        true
    }

    pub fn finish_challenge(&self, outcome: Option<RolloutOutcome>) -> Option<GateEvent> {
        let mut state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        let pending = state.pending.take()?;
        let outcome = outcome?;
        let version = match pending {
            PendingChallenge::Versioned(version) => Some(version),
            PendingChallenge::Seed => outcome.model_version,
        }?;

        state.last_challenged = Some(version);
        let best = state
            .current
            .as_ref()
            .map_or(outcome.final_reward, |incumbent| incumbent.final_reward);
        let accepted = state
            .current
            .as_ref()
            .is_none_or(|incumbent| outcome.final_reward > incumbent.final_reward);
        let event = GateEvent {
            accepted,
            challenger: outcome.final_reward,
            best,
            steps: outcome.steps.len(),
            version,
        };
        if accepted {
            let ref_id = state.next_ref_id;
            state.next_ref_id = state.next_ref_id.saturating_add(1);
            state.current = Some(Arc::new(ReferenceSnapshot {
                ref_id,
                kind: ReplayReferenceKind::GatedPolicy,
                version,
                final_reward: outcome.final_reward,
                final_graph: Some(outcome.final_graph),
                steps: outcome.steps.into(),
                search_config_hash: outcome.search_config_hash,
            }));
        }
        Some(event)
    }
}
