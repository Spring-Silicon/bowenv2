use gz_engine::{ModelVersion, ReplayGraphContext, SearchConfigHash};
use gz_features::OpponentStateFeatures;
use gz_replay::ReplayReferenceKind;
use std::collections::{HashMap, HashSet};
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyModel {
    Current,
    Incumbent,
    Challenger,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaRolloutClaim {
    pub index: usize,
    pub version: ModelVersion,
    pub model: PolicyModel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpisodeRolloutClaim {
    pub version: ModelVersion,
    pub model: PolicyModel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArenaGateEvent {
    pub accepted: bool,
    pub challenger_mean: f32,
    pub best_mean: f32,
    pub margin_sum: f32,
    pub steps: usize,
    pub arena_size: usize,
    pub version: ModelVersion,
}

#[derive(Debug)]
pub struct ArenaGateRegistry {
    state: Mutex<ArenaGateState>,
}

#[derive(Debug)]
struct ArenaGateState {
    arena_size: usize,
    gamma: f32,
    seed: u64,
    draws: u64,
    next_ref_id: u64,
    incumbent_version: Option<ModelVersion>,
    incumbent_scores: Vec<Option<f32>>,
    incumbent_claimed: HashSet<usize>,
    current_version: Option<ModelVersion>,
    challenger_version: Option<ModelVersion>,
    challenged: HashSet<ModelVersion>,
    challenger: Option<ArenaChallenger>,
}

#[derive(Debug)]
struct ArenaChallenger {
    version: ModelVersion,
    scores: Vec<Option<f32>>,
    steps: Vec<Option<usize>>,
    claimed: HashSet<usize>,
}

impl ArenaGateRegistry {
    #[must_use]
    pub fn new(arena_size: usize, gamma: f32, seed: u64) -> Self {
        assert!(arena_size > 0);
        assert!(gamma.is_finite() && (0.0..1.0).contains(&gamma));
        Self {
            state: Mutex::new(ArenaGateState {
                arena_size,
                gamma,
                seed,
                draws: 0,
                next_ref_id: 1,
                incumbent_version: None,
                incumbent_scores: vec![None; arena_size],
                incumbent_claimed: HashSet::new(),
                current_version: None,
                challenger_version: None,
                challenged: HashSet::new(),
                challenger: None,
            }),
        }
    }

    pub fn initialize(
        &self,
        incumbent: ModelVersion,
        current: ModelVersion,
        challenger: ModelVersion,
    ) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        if let Some(existing) = state.incumbent_version {
            return existing == incumbent
                && state.current_version == Some(current)
                && state.challenger_version == Some(challenger);
        }
        state.incumbent_version = Some(incumbent);
        state.current_version = Some(current);
        state.challenger_version = Some(challenger);
        true
    }

    pub fn observe_current(&self, version: ModelVersion) {
        let mut state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        state.current_version = Some(version);
    }

    pub fn observe_challenger(&self, version: ModelVersion) {
        let mut state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        state.challenger_version = Some(version);
    }

    #[must_use]
    pub fn claim_arena(&self, lane: usize, lanes: usize) -> Option<ArenaRolloutClaim> {
        assert!(lanes > 0);
        let mut state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        let incumbent_version = state.incumbent_version?;
        let index = {
            let ArenaGateState {
                incumbent_scores,
                incumbent_claimed,
                ..
            } = &mut *state;
            claim_index(incumbent_scores, incumbent_claimed, lane, lanes)
        };
        if let Some(index) = index {
            return Some(ArenaRolloutClaim {
                index,
                version: incumbent_version,
                model: PolicyModel::Incumbent,
            });
        }
        if state.incumbent_scores.iter().any(Option::is_none) {
            return None;
        }

        if state.challenger.is_none() {
            let latest = state.challenger_version?;
            if latest == incumbent_version || state.challenged.contains(&latest) {
                return None;
            }
            state.challenger = Some(ArenaChallenger {
                version: latest,
                scores: vec![None; state.arena_size],
                steps: vec![None; state.arena_size],
                claimed: HashSet::new(),
            });
        }
        let challenger = state.challenger.as_mut().expect("challenger initialized");
        claim_index(&challenger.scores, &mut challenger.claimed, lane, lanes).map(|index| {
            ArenaRolloutClaim {
                index,
                version: challenger.version,
                model: PolicyModel::Challenger,
            }
        })
    }

    pub fn finish_arena(
        &self,
        claim: ArenaRolloutClaim,
        actual_version: Option<ModelVersion>,
        score: Option<f32>,
        steps: usize,
    ) -> Option<ArenaGateEvent> {
        let mut state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        let score = score.filter(|score| score.is_finite());
        match claim.model {
            PolicyModel::Incumbent => {
                if state.incumbent_version != Some(claim.version) || claim.index >= state.arena_size
                {
                    return None;
                }
                state.incumbent_claimed.remove(&claim.index);
                if actual_version == Some(claim.version)
                    && let Some(score) = score
                {
                    state.incumbent_scores[claim.index] = Some(score);
                }
                None
            }
            PolicyModel::Current => None,
            PolicyModel::Challenger => {
                let arena_size = state.arena_size;
                let challenger = state.challenger.as_mut()?;
                if challenger.version != claim.version || claim.index >= arena_size {
                    return None;
                }
                challenger.claimed.remove(&claim.index);
                if actual_version == Some(claim.version)
                    && let Some(score) = score
                {
                    challenger.scores[claim.index] = Some(score);
                    challenger.steps[claim.index] = Some(steps);
                }
                if challenger.scores.iter().any(Option::is_none) {
                    return None;
                }

                let challenger_version = challenger.version;
                let challenger_scores = challenger
                    .scores
                    .iter()
                    .map(|score| score.expect("complete challenger"))
                    .collect::<Vec<_>>();
                let challenger_steps = challenger
                    .steps
                    .iter()
                    .map(|steps| steps.expect("complete challenger"))
                    .sum();
                let best_scores = state
                    .incumbent_scores
                    .iter()
                    .map(|score| score.expect("challenger waits for incumbent"))
                    .collect::<Vec<_>>();
                let margin_sum = challenger_scores
                    .iter()
                    .zip(&best_scores)
                    .map(|(challenger, best)| challenger - best)
                    .sum::<f32>();
                let denominator = arena_size as f32;
                let challenger_mean = challenger_scores.iter().sum::<f32>() / denominator;
                let best_mean = best_scores.iter().sum::<f32>() / denominator;
                let accepted = margin_sum > 0.0;
                state.challenged.insert(challenger_version);
                if accepted {
                    state.incumbent_version = Some(challenger_version);
                    state.incumbent_scores = challenger_scores.into_iter().map(Some).collect();
                    state.incumbent_claimed.clear();
                }
                state.challenger = None;
                Some(ArenaGateEvent {
                    accepted,
                    challenger_mean,
                    best_mean,
                    margin_sum,
                    steps: challenger_steps,
                    arena_size,
                    version: challenger_version,
                })
            }
        }
    }

    #[must_use]
    pub fn claim_episode(&self) -> Option<EpisodeRolloutClaim> {
        const EPISODE_POLICY_SALT: u64 = 0x6570_6973_6f64_655f;
        let mut state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        let incumbent = state.incumbent_version?;
        let latest = state.current_version.unwrap_or(incumbent);
        state.draws = state.draws.wrapping_add(1);
        let use_current = state.gamma > 0.0
            && latest != incumbent
            && random_unit(state.seed, state.draws, EPISODE_POLICY_SALT) < state.gamma;
        Some(if use_current {
            EpisodeRolloutClaim {
                version: latest,
                model: PolicyModel::Current,
            }
        } else {
            EpisodeRolloutClaim {
                version: incumbent,
                model: PolicyModel::Incumbent,
            }
        })
    }

    #[must_use]
    pub fn admission_ready(&self) -> bool {
        let state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        state.incumbent_version.is_some()
    }

    #[must_use]
    pub fn incumbent_version(&self) -> Option<ModelVersion> {
        self.state
            .lock()
            .expect("arena gate registry mutex poisoned")
            .incumbent_version
    }

    pub fn allocate_reference_id(&self) -> u64 {
        let mut state = self
            .state
            .lock()
            .expect("arena gate registry mutex poisoned");
        let ref_id = state.next_ref_id;
        state.next_ref_id = state.next_ref_id.saturating_add(1);
        ref_id
    }
}

#[derive(Debug, Default)]
pub struct ReferenceRegistry {
    state: Mutex<RegistryState>,
}

#[derive(Debug)]
struct RegistryState {
    current: Option<Arc<ReferenceSnapshot>>,
    latest: Option<Arc<ReferenceSnapshot>>,
    trajectory_pool: Vec<Arc<ReferenceSnapshot>>,
    sample_in_flight: HashMap<ModelVersion, usize>,
    challenged: HashSet<ModelVersion>,
    pending: Option<PendingChallenge>,
    next_ref_id: u64,
    gamma: f32,
    trajectory_pool_size: usize,
    seed: u64,
    draws: u64,
    trajectory_draws: u64,
}

impl Default for RegistryState {
    fn default() -> Self {
        Self {
            current: None,
            latest: None,
            trajectory_pool: Vec::new(),
            sample_in_flight: HashMap::new(),
            challenged: HashSet::new(),
            pending: None,
            next_ref_id: 1,
            gamma: 0.0,
            trajectory_pool_size: 0,
            seed: 0,
            draws: 0,
            trajectory_draws: 0,
        }
    }
}

impl ReferenceRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_gamma(gamma: f32, seed: u64) -> Self {
        Self::with_gamma_and_trajectory_pool(gamma, seed, 0)
    }

    #[must_use]
    pub fn with_gamma_and_trajectory_pool(
        gamma: f32,
        seed: u64,
        trajectory_pool_size: usize,
    ) -> Self {
        assert!(gamma.is_finite() && (0.0..1.0).contains(&gamma));
        Self {
            state: Mutex::new(RegistryState {
                gamma,
                trajectory_pool_size,
                seed,
                ..RegistryState::default()
            }),
        }
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
    pub fn latest(&self) -> Option<Arc<ReferenceSnapshot>> {
        self.state
            .lock()
            .expect("reference registry mutex poisoned")
            .latest
            .clone()
    }

    #[must_use]
    pub fn sampled(&self) -> Option<Arc<ReferenceSnapshot>> {
        let mut state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        let current = state.current.clone()?;
        let Some(latest) = state.latest.clone() else {
            return Some(current);
        };
        let selected = if state.gamma == 0.0 || next_unit(&mut state) >= state.gamma {
            current.clone()
        } else {
            latest
        };
        if selected.version != current.version || state.trajectory_pool.is_empty() {
            return Some(selected);
        }

        state.trajectory_draws = state.trajectory_draws.wrapping_add(1);
        let index = sample_index(
            state.seed,
            state.trajectory_draws,
            state.trajectory_pool.len(),
        );
        Some(Arc::clone(&state.trajectory_pool[index]))
    }

    pub fn allocate_reference_id(&self) -> u64 {
        let mut state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        let ref_id = state.next_ref_id;
        state.next_ref_id = state.next_ref_id.saturating_add(1);
        ref_id
    }

    #[must_use]
    pub fn trajectory_pool_len(&self) -> usize {
        self.state
            .lock()
            .expect("reference registry mutex poisoned")
            .trajectory_pool
            .len()
    }

    pub fn claim_sample(&self, _latest: Option<ModelVersion>) -> Option<ModelVersion> {
        let mut state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        if state.trajectory_pool_size == 0 {
            return None;
        }
        let version = state.current.as_ref()?.version;
        // Pool samples run on the dedicated incumbent evaluator. The current
        // evaluator may advance arbitrarily while this accepted version fills.
        // finish_sample verifies the backend actually served the incumbent.
        let in_flight = state.sample_in_flight.get(&version).copied().unwrap_or(0);
        if state.trajectory_pool.len() + in_flight >= state.trajectory_pool_size {
            return None;
        }
        *state.sample_in_flight.entry(version).or_insert(0) += 1;
        Some(version)
    }

    pub fn finish_sample(&self, version: ModelVersion, outcome: Option<RolloutOutcome>) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        let Some(in_flight) = state.sample_in_flight.get_mut(&version) else {
            return false;
        };
        *in_flight -= 1;
        if *in_flight == 0 {
            state.sample_in_flight.remove(&version);
        }

        let Some(outcome) = outcome else {
            return false;
        };
        if outcome.model_version != Some(version)
            || state.current.as_ref().map(|current| current.version) != Some(version)
            || state.trajectory_pool.len() >= state.trajectory_pool_size
        {
            return false;
        }

        let snapshot = Arc::new(ReferenceSnapshot {
            ref_id: state.next_ref_id,
            kind: ReplayReferenceKind::GatedPolicy,
            version,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: outcome.search_config_hash,
        });
        state.next_ref_id = state.next_ref_id.saturating_add(1);
        state.trajectory_pool.push(snapshot);
        true
    }

    #[must_use]
    pub fn admission_ready(&self) -> bool {
        let state = self
            .state
            .lock()
            .expect("reference registry mutex poisoned");
        state.current.is_some()
            && (state.trajectory_pool_size == 0
                || state.trajectory_pool.len() == state.trajectory_pool_size)
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
            Some(latest) => !state.challenged.contains(&latest),
            None => state.challenged.is_empty(),
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
            Some(latest) => !state.challenged.contains(&latest),
            None => state.challenged.is_empty(),
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
            PendingChallenge::Versioned(version) if outcome.model_version == Some(version) => {
                Some(version)
            }
            PendingChallenge::Versioned(_) => None,
            PendingChallenge::Seed => outcome.model_version,
        }?;

        state.challenged.insert(version);
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
        let snapshot = Arc::new(ReferenceSnapshot {
            ref_id: state.next_ref_id,
            kind: ReplayReferenceKind::GatedPolicy,
            version,
            final_reward: outcome.final_reward,
            final_graph: Some(outcome.final_graph),
            steps: outcome.steps.into(),
            search_config_hash: outcome.search_config_hash,
        });
        state.next_ref_id = state.next_ref_id.saturating_add(1);
        state.latest = Some(Arc::clone(&snapshot));
        if accepted {
            state.trajectory_pool.clear();
            state.current = Some(snapshot);
        }
        Some(event)
    }
}

fn next_unit(state: &mut RegistryState) -> f32 {
    state.draws = state.draws.wrapping_add(1);
    let mut value = state.seed ^ state.draws.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 40) as f32 / (1u64 << 24) as f32
}

fn sample_index(seed: u64, draw: u64, len: usize) -> usize {
    const TRAJECTORY_SALT: u64 = 0x7472_616a_5f70_6f6f; // "traj_poo"
    let mut value = seed ^ TRAJECTORY_SALT ^ draw.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value as usize) % len
}

fn claim_index(
    scores: &[Option<f32>],
    claimed: &mut HashSet<usize>,
    lane: usize,
    lanes: usize,
) -> Option<usize> {
    let index = (lane..scores.len())
        .step_by(lanes)
        .find(|index| scores[*index].is_none() && !claimed.contains(index))?;
    claimed.insert(index);
    Some(index)
}

fn random_unit(seed: u64, draw: u64, salt: u64) -> f32 {
    let mut value = seed ^ salt ^ draw.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 40) as f32 / (1u64 << 24) as f32
}
