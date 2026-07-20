use gz_engine::{CandidateOptions, EngineResult, GraphEngine, ModelVersion};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleFeatureExtractorConfig,
    WhittleGraphGenerator, WhittleGraphGeneratorConfig, WhittleGraphId, WhittleRoot,
};
use gz_eval::{RandomValueEvaluator, RandomValueEvaluatorConfig};
use gz_eval_service::{
    EvaluatorProcess, EvaluatorProcessConfig, Hello, STUB_MODEL_VERSION, StubBackend,
};
use gz_features::{FeatureExtractor, FeatureSchemaHash, PositionFeatures};
use gz_measurer::{ArenaGateRegistry, MeasureLedgerSnapshot, ReferenceRegistry, ValueTargetConfig};
use gz_orchestrator::reference::{
    ArenaRolloutClaim, BeamReferenceProvider, EpisodeRolloutClaim, GreedyReferenceProvider,
    PolicyReferenceProvider, RandomReferenceProvider, Reference, ReferenceProvider, RolloutOutcome,
    RootBaselineProvider, SelfAverageProvider,
};
use gz_orchestrator::{
    AdmissionSmoothingConfig, FeaturizedRuntime, ReplayBackpressure, ReplayRuntime, RootSource,
    ThreadedGumbelOrchestrator, ThreadedOrchestratorConfig,
};
use gz_replay::{ReplayCounters, ReplayDataMode, ReplayEpisodeId, ReplayRootInfo, ReplayStore};
use gz_search::{
    BeamSearch, BeamSearchConfig, GreedySearch, GreedySearchConfig, GumbelEpisodeContext,
    GumbelMcts, GumbelMctsConfig, GumbelValueMode, PolicyRolloutConfig, RandomSearch,
    RandomSearchConfig,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

const WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES: usize = 255;

#[derive(Clone, Debug)]
pub struct SelfplayConfig {
    pub replay_dir: Option<PathBuf>,
    pub episodes: u64,
    pub lanes: usize,
    pub workers_per_lane: usize,
    pub training_mode: TrainingMode,
    pub reference: ReferenceMode,
    pub root_mode: RootMode,
    pub reference_ema_decay: f32,
    /// Fraction of admitted references drawn from the latest measured
    /// challenger instead of the gated best. Gated-policy only.
    pub reference_gamma: f32,
    /// Number of stochastic trajectories retained for the accepted policy.
    /// Learner episodes sample one complete trajectory from this pool; zero
    /// preserves the deterministic greedy reference.
    pub reference_trajectory_pool: usize,
    /// Fixed generated-root arena used to gate the historical opponent.
    /// Zero keeps the legacy fixed-root gate.
    pub reference_arena_size: usize,
    pub reference_arena_seed: u64,
    /// Checkpoint pointer followed by the frozen incumbent evaluator.
    pub reference_checkpoint_pointer: Option<PathBuf>,
    /// Checkpoint pointer followed by the pinned arena challenger evaluator.
    pub reference_challenger_checkpoint_pointer: Option<PathBuf>,
    /// How a policy checkpoint supplies the opponent. None preserves the
    /// legacy trajectory/pool behavior for old configs.
    pub policy_opponent_mode: Option<PolicyOpponentMode>,
    /// Optional STOP-masking override for the policy reference rollout.
    /// None inherits the learner's mask_stop setting.
    pub reference_mask_stop: Option<bool>,
    pub seed: u64,
    pub max_steps: usize,
    pub simulations: usize,
    pub max_considered: usize,
    pub gumbel_scale: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    /// Auto-temper root noise to the policy's sharpness (whittlezero's
    /// overlap); negative disables and the fixed gumbel_scale applies.
    pub gumbel_noise_overlap: f32,
    pub tree_reuse: bool,
    pub max_candidates: usize,
    pub max_batch: usize,
    /// Incumbent evaluator batch capacity. None inherits max_batch.
    pub reference_max_batch: Option<usize>,
    /// Arena challenger evaluator batch capacity. None inherits the
    /// incumbent evaluator capacity.
    pub challenger_max_batch: Option<usize>,
    pub evaluator: EvaluatorMode,
    pub python_dir: Option<PathBuf>,
    pub checkpoint_dir: Option<PathBuf>,
    pub eval_device: Option<String>,
    pub eval_poll_interval: Option<f32>,
    pub serve_socket: Option<PathBuf>,
    pub serve_max_batch: usize,
    pub replay_backlog: Option<u64>,
    pub replay_retain: Option<u64>,
    /// Export real position features to evals and rows (default). Off
    /// conditions the model on graph + opponent alone.
    pub position_features: bool,
    /// Mask search actions that revisit the current or a prior episode
    /// root (whittlezero's no_backtrack). Learner search only; reference
    /// rollouts stay plain greedy.
    pub no_backtrack: bool,
    /// Mask STOP out of the learner's search wherever a rewrite exists
    /// (STOP-only nodes keep it); episodes then run to the step budget.
    /// Policy rollouts inherit `mask_stop` from the learner configuration.
    pub mask_stop: bool,
    /// Break equal-reward games by episode length (shorter wins) before
    /// the coin flip: whittlezero's duration tiebreak, discrete form.
    pub length_tiebreak: bool,
    /// Pair-outcome target: hard win/loss signs or WhittleZero's graded
    /// root-normalized reward margin.
    pub value_reward: ValueReward,
    pub value_reward_scale: f32,
    /// Evaluator processes to spawn and stripe lanes across (featurized
    /// evaluators only). Each process parallelizes per-batch host work
    /// on its own interpreter and keeps the GPU kernel queue dense.
    pub eval_processes: usize,
    /// Wall-clock spacing between learner-root admissions on each lane.
    /// Lane phase offsets spread admissions globally and are reapplied
    /// after a closed gate; zero disables pacing.
    pub admission_stagger_ms: u64,
    /// Pace learner-root admissions at measured evaluator capacity.
    pub admission_smoothing: bool,
    /// Evaluate independent symmetric MCTS root branches concurrently.
    pub wave_batching: bool,
}

#[derive(Clone, Debug)]
pub struct ReplayInitConfig {
    pub replay_dir: Option<PathBuf>,
    pub max_candidates: usize,
}

impl Default for ReplayInitConfig {
    fn default() -> Self {
        Self {
            replay_dir: None,
            max_candidates: WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES,
        }
    }
}

impl ReplayInitConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.is_none() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.max_candidates == 0 {
            return Err("--max-candidates must be greater than zero".to_owned());
        }
        feature_max_actions(self.max_candidates)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayInitSummary {
    pub feature_schema_hash: FeatureSchemaHash,
    pub max_actions: u32,
}

impl Default for SelfplayConfig {
    fn default() -> Self {
        Self {
            replay_dir: None,
            episodes: 16,
            lanes: 2,
            workers_per_lane: 8,
            training_mode: TrainingMode::Competitive,
            reference: ReferenceMode::Root,
            root_mode: RootMode::Generated,
            reference_ema_decay: 0.99,
            reference_gamma: 0.0,
            reference_trajectory_pool: 0,
            reference_arena_size: 0,
            reference_arena_seed: 910_000_001,
            reference_checkpoint_pointer: None,
            reference_challenger_checkpoint_pointer: None,
            policy_opponent_mode: None,
            reference_mask_stop: None,
            seed: 0,
            max_steps: 8,
            simulations: 8,
            max_considered: 16,
            gumbel_scale: 0.0,
            c_visit: 50.0,
            c_scale: 1.0,
            gumbel_noise_overlap: -1.0,
            tree_reuse: true,
            max_candidates: WHITTLE_FEATURE_MAX_ENGINE_CANDIDATES,
            max_batch: 16,
            reference_max_batch: None,
            challenger_max_batch: None,
            evaluator: EvaluatorMode::Random,
            python_dir: None,
            checkpoint_dir: None,
            eval_device: None,
            eval_poll_interval: None,
            serve_socket: None,
            serve_max_batch: 512,
            replay_backlog: None,
            replay_retain: None,
            position_features: true,
            no_backtrack: false,
            mask_stop: false,
            length_tiebreak: false,
            value_reward: ValueReward::Sign,
            value_reward_scale: 0.1,
            eval_processes: 1,
            admission_stagger_ms: 0,
            admission_smoothing: false,
            wave_batching: false,
        }
    }
}

impl SelfplayConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.is_none() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.lanes == 0 {
            return Err("--lanes must be greater than zero".to_owned());
        }
        if self.workers_per_lane == 0 {
            return Err("--workers-per-lane must be greater than zero".to_owned());
        }
        if self.max_steps == 0 {
            return Err("--max-steps must be greater than zero".to_owned());
        }
        if !self.c_visit.is_finite() || self.c_visit < 0.0 {
            return Err("--c-visit must be finite and non-negative".to_owned());
        }
        if !self.c_scale.is_finite() || self.c_scale < 0.0 {
            return Err("--c-scale must be finite and non-negative".to_owned());
        }
        if self.simulations == 0 {
            return Err("--simulations must be greater than zero".to_owned());
        }
        if self.max_batch == 0 {
            return Err("--max-batch must be greater than zero".to_owned());
        }
        if self.reference_max_batch == Some(0) {
            return Err("--reference-max-batch must be greater than zero".to_owned());
        }
        if self.challenger_max_batch == Some(0) {
            return Err("--challenger-max-batch must be greater than zero".to_owned());
        }
        if self.max_candidates == 0 {
            return Err("--max-candidates must be greater than zero".to_owned());
        }
        feature_max_actions(self.max_candidates)?;
        if self.max_considered == 0 {
            return Err("--max-considered must be greater than zero".to_owned());
        }
        if !self.gumbel_scale.is_finite() || self.gumbel_scale < 0.0 {
            return Err("--gumbel-scale must be zero or positive".to_owned());
        }
        if !self.gumbel_noise_overlap.is_finite() || self.gumbel_noise_overlap >= 1.0 {
            return Err("--gumbel-noise-overlap must be < 1 (negative disables)".to_owned());
        }
        if !self.reference_ema_decay.is_finite()
            || self.reference_ema_decay < 0.0
            || self.reference_ema_decay >= 1.0
        {
            return Err("--reference-ema-decay must be in [0, 1)".to_owned());
        }
        if self.reference == ReferenceMode::SelfAverage && self.reference_ema_decay == 0.0 {
            return Err(
                "--reference self-average requires --reference-ema-decay in (0, 1)".to_owned(),
            );
        }
        if !self.reference_gamma.is_finite()
            || self.reference_gamma < 0.0
            || self.reference_gamma >= 1.0
        {
            return Err("--reference-gamma must be in [0, 1)".to_owned());
        }
        if self.training_mode == TrainingMode::SingleVanilla {
            if self.reference != ReferenceMode::None
                || self.reference_ema_decay != 0.0
                || self.reference_gamma != 0.0
                || self.reference_trajectory_pool != 0
                || self.reference_arena_size != 0
                || self.reference_checkpoint_pointer.is_some()
                || self.reference_challenger_checkpoint_pointer.is_some()
                || self.policy_opponent_mode.is_some()
                || self.reference_mask_stop.is_some()
                || self.reference_max_batch.is_some()
                || self.challenger_max_batch.is_some()
            {
                return Err(
                    "--training-mode single-vanilla requires all reference and arena settings disabled"
                        .to_owned(),
                );
            }
            if self.tree_reuse {
                return Err("--training-mode single-vanilla requires --tree-reuse false".to_owned());
            }
            if self.mask_stop {
                return Err("--training-mode single-vanilla requires --mask-stop false".to_owned());
            }
            if self.length_tiebreak {
                return Err(
                    "--training-mode single-vanilla requires --length-tiebreak false".to_owned(),
                );
            }
            if self.value_reward != ValueReward::Sign {
                return Err(
                    "--training-mode single-vanilla requires --value-reward sign".to_owned(),
                );
            }
        }
        if self.training_mode == TrainingMode::SymmetricSelfplay {
            if self.reference != ReferenceMode::None
                || self.reference_ema_decay != 0.0
                || self.reference_gamma != 0.0
                || self.reference_trajectory_pool != 0
                || self.reference_arena_size != 0
                || self.reference_checkpoint_pointer.is_some()
                || self.reference_challenger_checkpoint_pointer.is_some()
                || self.policy_opponent_mode.is_some()
                || self.reference_mask_stop.is_some()
                || self.reference_max_batch.is_some()
                || self.challenger_max_batch.is_some()
            {
                return Err(
                    "--training-mode symmetric-selfplay requires all reference and arena settings disabled"
                        .to_owned(),
                );
            }
            if !self.mask_stop && !self.position_features {
                return Err(
                    "STOP-enabled symmetric-selfplay requires --position-features true".to_owned(),
                );
            }
            if !self.length_tiebreak {
                return Err(
                    "--training-mode symmetric-selfplay requires --length-tiebreak true".to_owned(),
                );
            }
            if self.value_reward != ValueReward::Sign {
                return Err(
                    "--training-mode symmetric-selfplay requires --value-reward sign".to_owned(),
                );
            }
            if self.evaluator == EvaluatorMode::Random {
                return Err(
                    "--training-mode symmetric-selfplay requires a featurized evaluator".to_owned(),
                );
            }
        }
        if self.wave_batching && self.training_mode != TrainingMode::SymmetricSelfplay {
            return Err("--wave-batching requires --training-mode symmetric-selfplay".to_owned());
        }
        if let Some(mode) = self.policy_opponent_mode {
            match mode {
                PolicyOpponentMode::SampledTrajectory
                    if self.reference != ReferenceMode::Policy =>
                {
                    return Err(
                        "--policy-opponent-mode sampled-trajectory requires --reference policy"
                            .to_owned(),
                    );
                }
                PolicyOpponentMode::GreedyTrajectory | PolicyOpponentMode::SampledTree
                    if self.reference != ReferenceMode::GatedPolicy =>
                {
                    return Err(
                        "--policy-opponent-mode requires --reference gated-policy".to_owned()
                    );
                }
                _ => {}
            }
            let generated_arena = matches!(
                mode,
                PolicyOpponentMode::GreedyTrajectory | PolicyOpponentMode::SampledTree
            ) && self.root_mode == RootMode::Generated
                && self.reference_arena_size > 0;
            if self.root_mode != RootMode::Fixed && !generated_arena {
                return Err(
                    "--policy-opponent-mode requires --root-mode fixed or a generated-root arena"
                        .to_owned(),
                );
            }
            if self.reference_trajectory_pool > 0 {
                return Err(
                    "--policy-opponent-mode cannot be combined with --reference-trajectory-pool"
                        .to_owned(),
                );
            }
            if mode == PolicyOpponentMode::SampledTrajectory
                && self.evaluator == EvaluatorMode::Random
            {
                return Err(
                    "--policy-opponent-mode sampled-trajectory requires a featurized evaluator"
                        .to_owned(),
                );
            }
            if mode == PolicyOpponentMode::SampledTrajectory && self.reference_gamma != 0.0 {
                return Err(
                    "--policy-opponent-mode sampled-trajectory requires --reference-gamma 0; active-policy rollouts do not select a historical reference"
                        .to_owned(),
                );
            }
            if mode == PolicyOpponentMode::SampledTree {
                if self.reference_gamma != 0.0 {
                    return Err(
                        "--policy-opponent-mode sampled-tree requires --reference-gamma 0"
                            .to_owned(),
                    );
                }
                if self.tree_reuse {
                    return Err(
                        "--policy-opponent-mode sampled-tree requires --tree-reuse false"
                            .to_owned(),
                    );
                }
                if self.evaluator != EvaluatorMode::Torch {
                    return Err(
                        "--policy-opponent-mode sampled-tree requires --evaluator torch".to_owned(),
                    );
                }
                if self.eval_processes != 1 {
                    return Err(
                        "--policy-opponent-mode sampled-tree requires --eval-processes 1"
                            .to_owned(),
                    );
                }
            }
        }
        if self.reference_gamma > 0.0 && self.reference != ReferenceMode::GatedPolicy {
            return Err("--reference-gamma requires --reference gated-policy".to_owned());
        }
        if self.reference_trajectory_pool > 0 && self.reference != ReferenceMode::GatedPolicy {
            return Err("--reference-trajectory-pool requires --reference gated-policy".to_owned());
        }
        if self.reference_trajectory_pool > 0 {
            if !matches!(
                self.evaluator,
                EvaluatorMode::ProcessStub | EvaluatorMode::Torch
            ) {
                return Err(
                    "--reference-trajectory-pool requires --evaluator process-stub|torch"
                        .to_owned(),
                );
            }
            if self.eval_processes != 1 {
                return Err(
                    "--reference-trajectory-pool requires --eval-processes 1; opaque model versions cannot order staggered process swaps"
                        .to_owned(),
                );
            }
        }
        if self.reference_arena_size > 0
            && !(self.reference == ReferenceMode::GatedPolicy
                && self.root_mode == RootMode::Generated
                && matches!(
                    self.policy_opponent_mode,
                    Some(PolicyOpponentMode::GreedyTrajectory | PolicyOpponentMode::SampledTree)
                ))
        {
            return Err(
                "--reference-arena-size requires generated-root gated-policy greedy-trajectory|sampled-tree"
                    .to_owned(),
            );
        }
        if self.reference_arena_size > 0 {
            if self.reference_trajectory_pool > 0 {
                return Err(
                    "--reference-arena-size cannot be combined with --reference-trajectory-pool"
                        .to_owned(),
                );
            }
            if self.evaluator != EvaluatorMode::Torch {
                return Err("generated-root arena gating requires --evaluator torch".to_owned());
            }
            if self.eval_processes != 1 {
                return Err(
                    "generated-root arena gating requires --eval-processes 1; opaque model versions cannot order staggered process swaps"
                        .to_owned(),
                );
            }
        }
        let sampled_tree = self.policy_opponent_mode == Some(PolicyOpponentMode::SampledTree);
        let historical_incumbent =
            self.reference_arena_size > 0 || self.reference_trajectory_pool > 0 || sampled_tree;
        if historical_incumbent && self.reference_checkpoint_pointer.is_none() {
            return Err(
                "historical incumbent evaluation requires --reference-checkpoint-pointer"
                    .to_owned(),
            );
        }
        if self.reference_checkpoint_pointer.is_some() && !historical_incumbent {
            return Err(
                "--reference-checkpoint-pointer requires --reference-arena-size, --reference-trajectory-pool, or sampled-tree"
                    .to_owned(),
            );
        }
        if self.reference_arena_size > 0 && self.reference_challenger_checkpoint_pointer.is_none() {
            return Err(
                "generated-root arena evaluation requires --reference-challenger-checkpoint-pointer"
                    .to_owned(),
            );
        }
        if self.reference_challenger_checkpoint_pointer.is_some() && self.reference_arena_size == 0
        {
            return Err(
                "--reference-challenger-checkpoint-pointer requires --reference-arena-size"
                    .to_owned(),
            );
        }
        if self.reference_mask_stop.is_some()
            && !matches!(
                self.reference,
                ReferenceMode::Policy | ReferenceMode::GatedPolicy
            )
        {
            return Err(
                "--reference-mask-stop requires --reference policy|gated-policy".to_owned(),
            );
        }
        if self.serve_socket.is_some() {
            if self.episodes != 0 {
                return Err("--serve-socket requires --episodes 0 (unbounded)".to_owned());
            }
            if self.evaluator == EvaluatorMode::Random {
                return Err(
                    "--serve-socket requires a featurized evaluator (stub|process-stub|torch)"
                        .to_owned(),
                );
            }
        }
        if matches!(
            self.reference,
            ReferenceMode::Policy | ReferenceMode::GatedPolicy
        ) && self.root_mode != RootMode::Fixed
            && self.reference_arena_size == 0
        {
            return Err("--reference policy|gated-policy requires --root-mode fixed".to_owned());
        }
        if self.evaluator == EvaluatorMode::Torch && self.checkpoint_dir.is_none() {
            return Err("--evaluator torch requires --checkpoint-dir".to_owned());
        }
        if self.evaluator != EvaluatorMode::Torch {
            if self.checkpoint_dir.is_some() {
                return Err("--checkpoint-dir requires --evaluator torch".to_owned());
            }
            if self.eval_device.is_some() {
                return Err("--eval-device requires --evaluator torch".to_owned());
            }
            if self.eval_poll_interval.is_some() {
                return Err("--eval-poll-interval requires --evaluator torch".to_owned());
            }
        }
        if let Some(interval) = self.eval_poll_interval
            && (!interval.is_finite() || interval < 0.0)
        {
            return Err("--eval-poll-interval must be zero (disabled) or positive".to_owned());
        }
        if self.reference_arena_size > 0 && self.eval_poll_interval == Some(0.0) {
            return Err(
                "generated-root arena gating requires a positive --eval-poll-interval".to_owned(),
            );
        }
        if self.episodes == 0 && self.serve_socket.is_none() {
            return Err("--episodes 0 (unbounded) requires --serve-socket".to_owned());
        }
        if self.serve_max_batch == 0 {
            return Err("--serve-max-batch must be greater than zero".to_owned());
        }
        if self.eval_processes == 0 {
            return Err("--eval-processes must be greater than zero".to_owned());
        }
        if self.eval_processes > 1
            && !matches!(
                self.evaluator,
                EvaluatorMode::ProcessStub | EvaluatorMode::Torch
            )
        {
            return Err("--eval-processes requires --evaluator process-stub|torch".to_owned());
        }
        if self.eval_processes > self.lanes {
            return Err("--eval-processes cannot exceed --lanes".to_owned());
        }
        if self.admission_stagger_ms > u64::MAX / 1_000_000 {
            return Err("--admission-stagger-ms is too large".to_owned());
        }
        if self.admission_smoothing && self.admission_stagger_ms != 0 {
            return Err(
                "--admission-smoothing and --admission-stagger-ms are mutually exclusive"
                    .to_owned(),
            );
        }
        if self.replay_backlog == Some(0) {
            return Err("--replay-backlog must be greater than zero".to_owned());
        }
        if self.replay_retain == Some(0) {
            return Err("--replay-retain must be greater than zero".to_owned());
        }
        if !self.value_reward_scale.is_finite() || self.value_reward_scale <= 0.0 {
            return Err("--value-reward-scale must be finite and positive".to_owned());
        }

        Ok(())
    }

    /// Extra command-line arguments passed to the spawned evaluator child.
    pub fn evaluator_extra_args(&self) -> Vec<String> {
        match self.evaluator {
            EvaluatorMode::Random | EvaluatorMode::Stub | EvaluatorMode::ProcessStub => Vec::new(),
            EvaluatorMode::Torch => {
                let checkpoint_dir = self
                    .checkpoint_dir
                    .as_ref()
                    .expect("validated checkpoint_dir exists");
                let device = self.eval_device.as_deref().unwrap_or("cuda:0");
                let mut args = vec![
                    "--backend".to_owned(),
                    "torch".to_owned(),
                    "--checkpoint-dir".to_owned(),
                    checkpoint_dir.display().to_string(),
                    "--device".to_owned(),
                    device.to_owned(),
                ];
                if let Some(interval) = self.eval_poll_interval {
                    args.push("--poll-interval".to_owned());
                    args.push(interval.to_string());
                }
                if self.training_mode == TrainingMode::SymmetricSelfplay {
                    args.extend([
                        "--require-state-input".to_owned(),
                        "joint-board".to_owned(),
                        "--require-value-input".to_owned(),
                        "single".to_owned(),
                    ]);
                }
                args
            }
        }
    }

    pub fn reference_evaluator_extra_args(&self) -> Vec<String> {
        let mut args = self.evaluator_extra_args();
        if self.evaluator == EvaluatorMode::Torch {
            args.push("--policy-only".to_owned());
        }
        if let Some(pointer) = &self.reference_checkpoint_pointer {
            args.push("--checkpoint-pointer".to_owned());
            args.push(pointer.display().to_string());
        }
        args
    }

    pub fn challenger_evaluator_extra_args(&self) -> Vec<String> {
        let mut args = self.evaluator_extra_args();
        if self.evaluator == EvaluatorMode::Torch {
            args.push("--policy-only".to_owned());
        }
        if let Some(pointer) = &self.reference_challenger_checkpoint_pointer {
            args.push("--checkpoint-pointer".to_owned());
            args.push(pointer.display().to_string());
        }
        args
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootMode {
    Generated,
    Fixed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TrainingMode {
    #[default]
    Competitive,
    SingleVanilla,
    SymmetricSelfplay,
}

impl FromStr for TrainingMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "competitive" => Ok(Self::Competitive),
            "single-vanilla" => Ok(Self::SingleVanilla),
            "symmetric-selfplay" => Ok(Self::SymmetricSelfplay),
            _ => Err(format!("unknown training mode: {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueReward {
    Sign,
    Graded,
}

impl FromStr for ValueReward {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "sign" => Ok(Self::Sign),
            "graded" => Ok(Self::Graded),
            _ => Err(format!("unknown value reward: {value}")),
        }
    }
}

impl FromStr for RootMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "generated" => Ok(Self::Generated),
            "fixed" => Ok(Self::Fixed),
            _ => Err(format!("unknown root mode: {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceMode {
    None,
    Root,
    Greedy,
    Beam,
    Random,
    SelfAverage,
    Policy,
    GatedPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyOpponentMode {
    GreedyTrajectory,
    SampledTrajectory,
    SampledTree,
}

impl FromStr for PolicyOpponentMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "greedy-trajectory" => Ok(Self::GreedyTrajectory),
            "sampled-trajectory" => Ok(Self::SampledTrajectory),
            "sampled-tree" => Ok(Self::SampledTree),
            _ => Err(format!("unknown policy opponent mode: {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluatorMode {
    Random,
    Stub,
    ProcessStub,
    Torch,
}

impl EvaluatorMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Stub => "stub",
            Self::ProcessStub => "process-stub",
            Self::Torch => "torch",
        }
    }
}

impl FromStr for EvaluatorMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "random" => Ok(Self::Random),
            "stub" => Ok(Self::Stub),
            "process-stub" => Ok(Self::ProcessStub),
            "torch" => Ok(Self::Torch),
            _ => Err(format!("unknown evaluator: {value}")),
        }
    }
}

impl FromStr for ReferenceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "root" => Ok(Self::Root),
            "greedy" => Ok(Self::Greedy),
            "beam" => Ok(Self::Beam),
            "random" => Ok(Self::Random),
            "self-average" => Ok(Self::SelfAverage),
            "policy" => Ok(Self::Policy),
            "gated-policy" => Ok(Self::GatedPolicy),
            _ => Err(format!("unknown reference: {value}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelfplaySummary {
    pub evaluator: EvaluatorMode,
    pub model_version: Option<ModelVersion>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub rows_produced: u64,
    pub wins: u64,
    pub losses: u64,
    pub ties: u64,
    pub eval_batch_count: usize,
    pub mean_eval_batch_size: f64,
    pub search_contexts: u64,
    pub replay_rows: u64,
    pub reference_steps: u64,
    pub counters: ReplayCounters,
    pub measure_ledger: MeasureLedgerSnapshot,
}

pub fn init_replay(config: ReplayInitConfig) -> Result<ReplayInitSummary, String> {
    config.validate()?;

    let replay_dir = config
        .replay_dir
        .as_ref()
        .expect("validated replay_dir exists");
    let max_actions = feature_max_actions(config.max_candidates)?;
    let store = ReplayStore::open(replay_dir).map_err(|error| error.to_string())?;
    let engine = WhittleEngine::new(whittle_engine_config()).map_err(|error| error.to_string())?;
    let extractor = WhittleFeatureExtractor::with_config(
        &engine,
        WhittleFeatureExtractorConfig {
            max_actions,
            ..WhittleFeatureExtractorConfig::default()
        },
    );
    let schema = extractor.schema();
    store
        .ensure_feature_schema(schema.config())
        .map_err(|error| error.to_string())?;

    Ok(ReplayInitSummary {
        feature_schema_hash: schema.hash(),
        max_actions: schema.config().max_actions,
    })
}

pub fn run(config: SelfplayConfig) -> Result<SelfplaySummary, String> {
    config.validate()?;

    let replay_dir = config
        .replay_dir
        .as_ref()
        .expect("validated replay_dir exists");
    let store = std::sync::Arc::new(
        ReplayStore::open_with_retention(replay_dir, config.replay_retain)
            .map_err(|error| error.to_string())?,
    );
    store
        .ensure_data_mode(replay_data_mode(&config)?)
        .map_err(|error| error.to_string())?;
    let engines = (0..config.lanes)
        .map(|_| WhittleEngine::new(whittle_engine_config()).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let search = search(&engines[0], &config)?;
    let roots = root_sources(&config);
    let arena_registry = (config.reference_arena_size > 0).then(|| {
        Arc::new(ArenaGateRegistry::new(
            config.reference_arena_size,
            config.reference_gamma,
            config.reference_arena_seed,
        ))
    });
    let registry = match (config.reference, config.policy_opponent_mode) {
        (ReferenceMode::GatedPolicy, _) if arena_registry.is_none() => {
            Some(Arc::new(ReferenceRegistry::with_gamma_and_trajectory_pool(
                config.reference_gamma,
                config.seed,
                config.reference_trajectory_pool,
            )))
        }
        (ReferenceMode::Policy, Some(PolicyOpponentMode::SampledTrajectory)) => {
            Some(Arc::new(ReferenceRegistry::new()))
        }
        _ => None,
    };
    let providers = engines
        .iter()
        .enumerate()
        .map(|(lane, engine)| {
            provider(
                engine,
                &config,
                lane,
                registry.as_ref(),
                arena_registry.as_ref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    if config.root_mode == RootMode::Fixed {
        probe_fixed_root(&store, &config)?;
    }

    if let Some(socket) = config.serve_socket.clone() {
        // The featurized run registers the schema itself, but the sample
        // service binds before the run starts and needs it already stored.
        let extractor = feature_extractor(&engines[0], &config);
        store
            .ensure_feature_schema(extractor.schema().config())
            .map_err(|error| error.to_string())?;
        let serve_store = store.clone();
        let serve_max_batch = config.serve_max_batch;
        std::thread::spawn(move || {
            if let Err(error) = crate::serve::run_shared(serve_store, socket, serve_max_batch) {
                // The trainer depends on this service; fail the whole
                // process loudly rather than starving it silently.
                eprintln!("sample service failed: {error}");
                std::process::exit(1);
            }
        });
    }

    match config.evaluator {
        EvaluatorMode::Random => run_random(config, store, engines, search, roots, providers),
        EvaluatorMode::Stub => run_stub(config, store, engines, search, roots, providers),
        EvaluatorMode::ProcessStub | EvaluatorMode::Torch => run_process(
            config,
            store,
            engines,
            search,
            roots,
            providers,
            arena_registry,
        ),
    }
}

fn value_target_config(config: &SelfplayConfig) -> ValueTargetConfig {
    if config.training_mode == TrainingMode::SingleVanilla {
        return ValueTargetConfig::SingleVanilla;
    }
    match config.value_reward {
        ValueReward::Sign => ValueTargetConfig::Sign,
        ValueReward::Graded => ValueTargetConfig::graded(config.value_reward_scale),
    }
}

fn replay_data_mode(config: &SelfplayConfig) -> Result<ReplayDataMode, String> {
    if config.training_mode == TrainingMode::SingleVanilla {
        return Ok(ReplayDataMode::SingleVanilla);
    }
    if config.training_mode == TrainingMode::SymmetricSelfplay {
        return Ok(if config.mask_stop {
            ReplayDataMode::SymmetricSelfplay
        } else {
            ReplayDataMode::SymmetricSelfplayStop
        });
    }
    let sampled_tree = config.policy_opponent_mode == Some(PolicyOpponentMode::SampledTree);
    match config.value_reward {
        ValueReward::Sign if sampled_tree => Ok(ReplayDataMode::SampledTree),
        ValueReward::Sign => Ok(ReplayDataMode::Standard),
        ValueReward::Graded => ReplayDataMode::graded(sampled_tree, config.value_reward_scale)
            .map_err(|error| error.to_string()),
    }
}

fn run_random(
    config: SelfplayConfig,
    store: std::sync::Arc<ReplayStore>,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<CliRoots>,
    providers: Vec<CliReferenceProvider>,
) -> Result<SelfplaySummary, String> {
    let evaluator = RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: config.seed,
        ..RandomValueEvaluatorConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let policy_rollout = policy_rollout_config(&engines[0], &config);
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        evaluator,
        search,
        ThreadedOrchestratorConfig {
            workers_per_lane: nonzero(config.workers_per_lane, "workers_per_lane")?,
            max_batch: nonzero(config.max_batch, "max_batch")?,
            admission_stagger: Duration::from_millis(config.admission_stagger_ms),
            admission_smoothing: admission_smoothing(&config)?,
            // 3ms: at ~10ms model forwards a 1ms lull shipped partial
            // batches (historical fill averaged 29/128); the deeper
            // in-flight pool refills within this window.
            flush_after: Duration::from_millis(3),
        },
    )
    .with_policy_rollout(policy_rollout);
    let run = orchestrator
        .run_with_replay(
            roots,
            GumbelEpisodeContext::default(),
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: replay_backpressure(&config),
                length_tiebreak: config.length_tiebreak,
                value_target: value_target_config(&config),
            },
        )
        .map_err(|error| error.to_string())?;

    summarize(&store, run, EvaluatorMode::Random, None)
}

fn run_stub(
    config: SelfplayConfig,
    store: std::sync::Arc<ReplayStore>,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<CliRoots>,
    providers: Vec<CliReferenceProvider>,
) -> Result<SelfplaySummary, String> {
    let extractors = engines
        .iter()
        .map(|engine| feature_extractor(engine, &config))
        .collect::<Vec<_>>();
    let policy_rollout = policy_rollout_config(&engines[0], &config);
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        random_placeholder(&config)?,
        search,
        threaded_config(&config)?,
    )
    .with_policy_rollout(policy_rollout);
    let run = orchestrator
        .run_featurized_with_replay(
            roots,
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends: vec![StubBackend],
                reference_backends: vec![],
                challenger_backends: vec![],
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: replay_backpressure(&config),
                length_tiebreak: config.length_tiebreak,
                value_target: value_target_config(&config),
            },
        )
        .map_err(|error| error.to_string())?;

    summarize(&store, run, EvaluatorMode::Stub, Some(STUB_MODEL_VERSION))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointPointer {
    version_dir: String,
    model_version: String,
}

struct ArenaPointerWatcher {
    stop: Option<mpsc::Sender<()>>,
    handle: Option<JoinHandle<()>>,
}

impl ArenaPointerWatcher {
    fn start(
        path: PathBuf,
        registry: Arc<ArenaGateRegistry>,
        poll_interval: Duration,
    ) -> Result<Self, String> {
        let initial = read_checkpoint_pointer(&path)?;
        registry.observe_challenger(initial);
        let (stop, stopped) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let mut observed = initial;
            let mut last_error = None;
            loop {
                match stopped.recv_timeout(poll_interval) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
                match read_checkpoint_pointer(&path) {
                    Ok(version) => {
                        last_error = None;
                        if version != observed {
                            observed = version;
                            registry.observe_challenger(version);
                            eprintln!("event=arena_pointer_observed model_version={version}");
                        }
                    }
                    Err(error) => {
                        if last_error.as_deref() != Some(error.as_str()) {
                            eprintln!("event=arena_pointer_rejected error={error}");
                            last_error = Some(error);
                        }
                    }
                }
            }
        });
        Ok(Self {
            stop: Some(stop),
            handle: Some(handle),
        })
    }
}

impl Drop for ArenaPointerWatcher {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_checkpoint_pointer(path: &PathBuf) -> Result<ModelVersion, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let pointer: CheckpointPointer = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    if pointer.version_dir.is_empty() || pointer.version_dir.contains('/') {
        return Err(format!("invalid version_dir in {}", path.display()));
    }
    ModelVersion::try_from_hex(&pointer.model_version)
        .map_err(|error| format!("invalid model_version in {}: {error}", path.display()))
}

fn run_process(
    config: SelfplayConfig,
    store: std::sync::Arc<ReplayStore>,
    engines: Vec<WhittleEngine>,
    search: GumbelMcts,
    roots: Vec<CliRoots>,
    providers: Vec<CliReferenceProvider>,
    arena_registry: Option<Arc<ArenaGateRegistry>>,
) -> Result<SelfplaySummary, String> {
    let extractors = engines
        .iter()
        .map(|engine| feature_extractor(engine, &config))
        .collect::<Vec<_>>();
    let mut processes = Vec::with_capacity(config.eval_processes);
    for index in 0..config.eval_processes {
        processes.push(
            EvaluatorProcess::spawn(EvaluatorProcessConfig {
                working_dir: config
                    .python_dir
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("python")),
                socket_path: process_socket_path(index),
                ready_timeout: Duration::from_secs(10),
                // Generous: a tripped eval timeout kills the whole selfplay
                // run, and warm-up/compile stalls on the evaluator are
                // legitimate.
                io_timeout: Duration::from_secs(300),
                extra_args: config.evaluator_extra_args(),
                ..EvaluatorProcessConfig::default()
            })
            .map_err(|error| error.to_string())?,
        );
    }
    let hello = Hello::new(
        extractors
            .first()
            .ok_or_else(|| "missing feature extractor".to_owned())?
            .schema()
            .hash(),
        config.max_batch as u32,
        engines[0].engine_id(),
        engines[0].engine_version(),
        engines[0].action_set_hash(),
    );
    let reference_hello = Hello {
        batch_capacity: config.reference_max_batch.unwrap_or(config.max_batch) as u32,
        ..hello
    };
    let challenger_hello = Hello {
        batch_capacity: config
            .challenger_max_batch
            .unwrap_or(reference_hello.batch_capacity as usize) as u32,
        ..hello
    };
    let mut backends = Vec::with_capacity(processes.len());
    for process in &mut processes {
        backends.push(process.connect(&hello).map_err(|error| error.to_string())?);
    }
    let model_version = backends[0].model_version();
    let mut reference_processes = Vec::new();
    let mut reference_backends = Vec::new();
    if config.reference_checkpoint_pointer.is_some() {
        reference_processes.reserve(config.eval_processes);
        for index in 0..config.eval_processes {
            reference_processes.push(
                EvaluatorProcess::spawn(EvaluatorProcessConfig {
                    working_dir: config
                        .python_dir
                        .clone()
                        .unwrap_or_else(|| PathBuf::from("python")),
                    socket_path: reference_process_socket_path(index),
                    ready_timeout: Duration::from_secs(10),
                    io_timeout: Duration::from_secs(300),
                    extra_args: config.reference_evaluator_extra_args(),
                    ..EvaluatorProcessConfig::default()
                })
                .map_err(|error| error.to_string())?,
            );
        }
        for process in &mut reference_processes {
            reference_backends.push(
                process
                    .connect(&reference_hello)
                    .map_err(|error| error.to_string())?,
            );
        }
    }
    let mut challenger_processes = Vec::new();
    let mut challenger_backends = Vec::new();
    if config.reference_challenger_checkpoint_pointer.is_some() {
        challenger_processes.reserve(config.eval_processes);
        for index in 0..config.eval_processes {
            challenger_processes.push(
                EvaluatorProcess::spawn(EvaluatorProcessConfig {
                    working_dir: config
                        .python_dir
                        .clone()
                        .unwrap_or_else(|| PathBuf::from("python")),
                    socket_path: challenger_process_socket_path(index),
                    ready_timeout: Duration::from_secs(10),
                    io_timeout: Duration::from_secs(300),
                    extra_args: config.challenger_evaluator_extra_args(),
                    ..EvaluatorProcessConfig::default()
                })
                .map_err(|error| error.to_string())?,
            );
        }
        for process in &mut challenger_processes {
            challenger_backends.push(
                process
                    .connect(&challenger_hello)
                    .map_err(|error| error.to_string())?,
            );
        }
    }
    if let Some(registry) = &arena_registry {
        let incumbent_version = reference_backends[0].model_version();
        let challenger_version = challenger_backends[0].model_version();
        if !registry.initialize(incumbent_version, model_version, challenger_version) {
            return Err("arena registry initialization mismatch".to_owned());
        }
    }
    let _arena_pointer_watcher = match (
        config.reference_challenger_checkpoint_pointer.clone(),
        arena_registry.as_ref(),
    ) {
        (Some(pointer), Some(registry)) => Some(ArenaPointerWatcher::start(
            pointer,
            Arc::clone(registry),
            Duration::from_secs_f32(config.eval_poll_interval.unwrap_or(10.0)),
        )?),
        _ => None,
    };
    let policy_rollout = policy_rollout_config(&engines[0], &config);
    let orchestrator = ThreadedGumbelOrchestrator::new(
        engines,
        random_placeholder(&config)?,
        search,
        threaded_config(&config)?,
    )
    .with_policy_rollout(policy_rollout);
    let run = orchestrator
        .run_featurized_with_replay(
            roots,
            GumbelEpisodeContext::default(),
            FeaturizedRuntime {
                extractors,
                backends,
                reference_backends,
                challenger_backends,
            },
            ReplayRuntime {
                store: &store,
                providers,
                backpressure: replay_backpressure(&config),
                length_tiebreak: config.length_tiebreak,
                value_target: value_target_config(&config),
            },
        )
        .map_err(|error| error.to_string())?;
    for process in &mut processes {
        wait_for_process_exit(process)?;
    }
    for process in &mut reference_processes {
        wait_for_process_exit(process)?;
    }
    for process in &mut challenger_processes {
        wait_for_process_exit(process)?;
    }

    summarize(&store, run, config.evaluator, Some(model_version))
}

fn summarize(
    store: &ReplayStore,
    run: gz_orchestrator::ThreadedReplayRun,
    evaluator: EvaluatorMode,
    model_version: Option<ModelVersion>,
) -> Result<SelfplaySummary, String> {
    let counters = store.counters();
    let (wins, losses, ties) = label_counts(store)?;
    let evals = run.batch_sizes.iter().sum::<usize>();
    let mean_eval_batch_size = if run.batch_sizes.is_empty() {
        0.0
    } else {
        evals as f64 / run.batch_sizes.len() as f64
    };

    Ok(SelfplaySummary {
        evaluator,
        model_version,
        episodes_appended: run.episodes_appended,
        episodes_dropped: run.episodes_dropped,
        rows_produced: counters.produced_rows,
        wins,
        losses,
        ties,
        eval_batch_count: run.batch_sizes.len(),
        mean_eval_batch_size,
        search_contexts: run.search_contexts,
        replay_rows: run.replay_rows,
        reference_steps: run.reference_steps,
        counters,
        measure_ledger: run.measure_ledger,
    })
}

fn replay_backpressure(config: &SelfplayConfig) -> Option<ReplayBackpressure> {
    config.replay_backlog.map(|cap| ReplayBackpressure {
        max_row_backlog: std::num::NonZeroU64::new(cap).expect("validated nonzero"),
        gate_poll: Duration::from_millis(1),
    })
}

fn threaded_config(config: &SelfplayConfig) -> Result<ThreadedOrchestratorConfig, String> {
    Ok(ThreadedOrchestratorConfig {
        workers_per_lane: nonzero(config.workers_per_lane, "workers_per_lane")?,
        max_batch: nonzero(config.max_batch, "max_batch")?,
        admission_stagger: Duration::from_millis(config.admission_stagger_ms),
        admission_smoothing: admission_smoothing(config)?,
        // 3ms: at ~10ms model forwards a 1ms lull shipped partial
        // batches (historical fill averaged 29/128); the deeper
        // in-flight pool refills within this window.
        flush_after: Duration::from_millis(3),
    })
}

fn admission_smoothing(
    config: &SelfplayConfig,
) -> Result<Option<AdmissionSmoothingConfig>, String> {
    if !config.admission_smoothing {
        return Ok(None);
    }
    let max_steps = u64::try_from(config.max_steps)
        .map_err(|_| "max_steps exceeds admission work range".to_owned())?;
    let simulations = u64::try_from(config.simulations)
        .map_err(|_| "simulations exceeds admission work range".to_owned())?;
    let actor_trajectories = if config.training_mode == TrainingMode::SymmetricSelfplay {
        2
    } else {
        1
    };
    let initial_episode_eval_work = max_steps
        .checked_mul(actor_trajectories)
        .ok_or_else(|| "admission work estimate overflow".to_owned())?
        .checked_mul(
            simulations
                .checked_add(1)
                .ok_or_else(|| "admission work estimate overflow".to_owned())?,
        )
        .and_then(NonZeroU64::new)
        .ok_or_else(|| "admission work estimate overflow".to_owned())?;
    Ok(Some(AdmissionSmoothingConfig {
        initial_episode_eval_work,
    }))
}

fn random_placeholder(config: &SelfplayConfig) -> Result<RandomValueEvaluator, String> {
    RandomValueEvaluator::new(RandomValueEvaluatorConfig {
        seed: config.seed,
        ..RandomValueEvaluatorConfig::default()
    })
    .map_err(|error| error.to_string())
}

fn process_socket_path(index: usize) -> PathBuf {
    std::env::temp_dir().join(format!(
        "gz-process-stub-{}-{index}.sock",
        std::process::id()
    ))
}

fn reference_process_socket_path(index: usize) -> PathBuf {
    std::env::temp_dir().join(format!(
        "gz-process-reference-{}-{index}.sock",
        std::process::id()
    ))
}

fn challenger_process_socket_path(index: usize) -> PathBuf {
    std::env::temp_dir().join(format!(
        "gz-process-challenger-{}-{index}.sock",
        std::process::id()
    ))
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait().map_err(|error| error.to_string())? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("Python evaluator exited with {status}")),
            None if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            None => return Err("Python evaluator did not exit".to_owned()),
        }
    }
}

fn search(engine: &WhittleEngine, config: &SelfplayConfig) -> Result<GumbelMcts, String> {
    let search = GumbelMcts::new(GumbelMctsConfig {
        max_steps: config.max_steps,
        simulations: nonzero(config.simulations, "simulations")?,
        max_considered_actions: nonzero(config.max_considered, "max_considered")?,
        seed: config.seed,
        gumbel_scale: config.gumbel_scale,
        gumbel_noise_overlap: config.gumbel_noise_overlap,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        temperature_moves: 0,
        tree_reuse: config.tree_reuse,
        export_position: config.position_features,
        mask_stop: config.mask_stop,
        no_backtrack: config.no_backtrack,
        value_mode: match config.training_mode {
            TrainingMode::Competitive => GumbelValueMode::Competitive,
            TrainingMode::SingleVanilla => GumbelValueMode::SingleVanilla,
            TrainingMode::SymmetricSelfplay => GumbelValueMode::SymmetricSelfplay,
        },
        candidate_options: match config.evaluator {
            EvaluatorMode::Random => CandidateOptions::default(),
            EvaluatorMode::Stub | EvaluatorMode::ProcessStub | EvaluatorMode::Torch => {
                feature_candidate_options(config)
            }
        },
        measure_options: engine.measure_options(),
    })
    .with_symmetric_wave_batching(config.wave_batching);
    Ok(match config.reference_mask_stop {
        Some(mask_stop) => search.with_policy_rollout_mask_stop(mask_stop),
        None => search,
    })
}

fn policy_rollout_config(engine: &WhittleEngine, config: &SelfplayConfig) -> PolicyRolloutConfig {
    PolicyRolloutConfig {
        max_steps: config.max_steps,
        seed: config.seed,
        export_position: config.position_features,
        mask_stop: config.reference_mask_stop.unwrap_or(config.mask_stop),
        no_backtrack: config.no_backtrack,
        candidate_options: match config.evaluator {
            EvaluatorMode::Random => CandidateOptions::default(),
            EvaluatorMode::Stub | EvaluatorMode::ProcessStub | EvaluatorMode::Torch => {
                feature_candidate_options(config)
            }
        },
        measure_options: engine.measure_options(),
    }
}

fn feature_candidate_options(config: &SelfplayConfig) -> CandidateOptions {
    CandidateOptions {
        max_candidates: Some(config.max_candidates),
        deterministic_order: true,
    }
}

/// Feature rows hold one action per engine candidate plus STOP.
fn feature_extractor(engine: &WhittleEngine, config: &SelfplayConfig) -> WhittleFeatureExtractor {
    WhittleFeatureExtractor::with_config(
        engine,
        WhittleFeatureExtractorConfig {
            max_actions: feature_max_actions(config.max_candidates).expect("validated max_actions"),
            ..WhittleFeatureExtractorConfig::default()
        },
    )
}

fn feature_max_actions(max_candidates: usize) -> Result<u32, String> {
    let candidates = u32::try_from(max_candidates)
        .map_err(|_| "--max-candidates exceeds schema action limit".to_owned())?;
    candidates
        .checked_add(1)
        .ok_or_else(|| "--max-candidates exceeds schema action limit".to_owned())
}

fn provider(
    engine: &WhittleEngine,
    config: &SelfplayConfig,
    lane: usize,
    registry: Option<&Arc<ReferenceRegistry>>,
    arena_registry: Option<&Arc<ArenaGateRegistry>>,
) -> Result<CliReferenceProvider, String> {
    let measure_options = engine.measure_options();
    let provider = match config.reference {
        ReferenceMode::None => CliReferenceProvider::None,
        ReferenceMode::Root => {
            CliReferenceProvider::Root(RootBaselineProvider::new(measure_options))
        }
        ReferenceMode::Greedy => CliReferenceProvider::Greedy(GreedyReferenceProvider::new(
            GreedySearch::new(GreedySearchConfig {
                max_steps: config.max_steps,
                candidate_options: CandidateOptions::default(),
                measure_options,
            }),
        )),
        ReferenceMode::Beam => CliReferenceProvider::Beam(BeamReferenceProvider::new(
            BeamSearch::new(BeamSearchConfig {
                max_depth: config.max_steps,
                beam_width: NonZeroUsize::new(4).unwrap(),
                candidate_options: CandidateOptions::default(),
                measure_options,
            }),
        )),
        ReferenceMode::Random => CliReferenceProvider::Random(RandomReferenceProvider::new(
            RandomSearch::new(RandomSearchConfig {
                max_steps: config.max_steps,
                seed: config.seed ^ ((lane as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                candidate_options: CandidateOptions::default(),
                measure_options,
            }),
        )),
        ReferenceMode::SelfAverage => {
            CliReferenceProvider::SelfAverage(SelfAverageProvider::new(config.reference_ema_decay))
        }
        ReferenceMode::Policy => {
            let provider =
                if config.policy_opponent_mode == Some(PolicyOpponentMode::SampledTrajectory) {
                    PolicyReferenceProvider::sampled_trajectory_with_registry(Arc::clone(
                        registry.expect("sampled-trajectory registry exists"),
                    ))
                } else {
                    PolicyReferenceProvider::new()
                };
            CliReferenceProvider::Policy(CliPolicyProvider::new(provider))
        }
        ReferenceMode::GatedPolicy => {
            let provider = match arena_registry {
                Some(registry)
                    if config.policy_opponent_mode == Some(PolicyOpponentMode::SampledTree) =>
                {
                    PolicyReferenceProvider::arena_sampled_tree(Arc::clone(registry))
                }
                Some(registry) => PolicyReferenceProvider::arena_gated(Arc::clone(registry)),
                None if config.policy_opponent_mode == Some(PolicyOpponentMode::SampledTree) => {
                    PolicyReferenceProvider::sampled_tree_with_registry(Arc::clone(
                        registry.expect("gated-policy registry exists"),
                    ))
                }
                None => PolicyReferenceProvider::gated_with_registry(Arc::clone(
                    registry.expect("gated-policy registry exists"),
                )),
            };
            let arena = arena_registry.map(|_| CliArenaRoots {
                size: config.reference_arena_size,
                seed: config.reference_arena_seed,
                roots: HashMap::new(),
            });
            CliReferenceProvider::Policy(CliPolicyProvider { provider, arena })
        }
    };

    Ok(provider)
}

/// Measures and describes the shared root once so the trainer can anchor
/// graph-level metrics (reduction = root cost - terminal cost). Uses a
/// throwaway engine seeded exactly like every lane's fixed source.
fn probe_fixed_root(store: &ReplayStore, config: &SelfplayConfig) -> Result<(), String> {
    let mut engine = WhittleEngine::new(whittle_engine_config()).map_err(|e| e.to_string())?;
    let mut generator = WhittleGraphGenerator::from_seed(whittle_generator_config(), config.seed);
    let root = generator
        .sample_root_into(&mut engine)
        .map_err(|e| e.to_string())?;
    let mut candidates = Vec::new();
    engine
        .candidates(root, feature_candidate_options(config), &mut candidates)
        .map_err(|e| e.to_string())?;
    let mut extractor = feature_extractor(&engine, config);
    let row = extractor
        .extract(
            &engine,
            root,
            &candidates,
            PositionFeatures {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 1.0,
                budget_step: 0.0,
                opponent_reward: 0.0,
                opponent_present: false,
            },
        )
        .map_err(|e| format!("root feature probe failed: {e:?}"))?;
    let measure = engine
        .measure(root, engine.measure_options())
        .map_err(|e| e.to_string())?;
    let cost = -measure
        .scalar_reward
        .ok_or_else(|| "fixed root has no scalar reward".to_owned())?;
    // Expander edges are model wiring, not graph structure.
    let edge_count = row.edges.iter().filter(|edge| edge.edge_type < 2).count() as u32;

    store
        .set_root_info(&ReplayRootInfo {
            cost,
            node_count: row.node_count,
            edge_count,
            candidate_count: candidates.len() as u32,
        })
        .map_err(|e| e.to_string())
}

fn root_sources(config: &SelfplayConfig) -> Vec<CliRoots> {
    let base = config.episodes / config.lanes as u64;
    let extra = config.episodes % config.lanes as u64;

    (0..config.lanes)
        .map(|lane| {
            let count = base + u64::from((lane as u64) < extra);
            let remaining = (config.episodes != 0).then_some(count);
            match config.root_mode {
                RootMode::Generated => CliRoots::Generated(GeneratedRoots {
                    remaining,
                    generator: WhittleGraphGenerator::from_seed(
                        whittle_generator_config(),
                        config.seed ^ ((lane as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03)),
                    ),
                }),
                RootMode::Fixed => CliRoots::Fixed {
                    remaining,
                    generator: WhittleGraphGenerator::from_seed(
                        whittle_generator_config(),
                        config.seed,
                    ),
                    root: None,
                },
            }
        })
        .collect()
}

fn whittle_engine_config() -> WhittleEngineConfig {
    let generator = whittle_generator_config();
    WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator.arity,
            capacity: generator.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    }
}

fn whittle_generator_config() -> WhittleGraphGeneratorConfig {
    WhittleGraphGeneratorConfig::default()
}

fn label_counts(store: &ReplayStore) -> Result<(u64, u64, u64), String> {
    if store
        .data_mode()
        .map_err(|error| error.to_string())?
        .is_single_vanilla()
    {
        return Ok((0, 0, 0));
    }
    let mut wins = 0;
    let mut losses = 0;
    let mut ties = 0;

    for id in 0..store.episode_sequence_end() {
        let Some(record) = store
            .episode(ReplayEpisodeId::new(id))
            .map_err(|error| error.to_string())?
        else {
            continue;
        };

        match record.outcome.value_target {
            Some(1.0) => wins += 1,
            Some(-1.0) => losses += 1,
            Some(0.0) => ties += 1,
            _ => {}
        }
    }

    Ok((wins, losses, ties))
}

fn nonzero(value: usize, name: &str) -> Result<NonZeroUsize, String> {
    NonZeroUsize::new(value).ok_or_else(|| format!("{name} must be greater than zero"))
}

struct GeneratedRoots {
    /// None = unbounded: the run ends only by signal (kill-safe: every
    /// append is one atomic WriteBatch, so a store killed mid-write
    /// reopens intact).
    remaining: Option<u64>,
    generator: WhittleGraphGenerator,
}

enum CliRoots {
    /// A fresh generated root per episode (the default).
    Generated(GeneratedRoots),
    /// One graph, sampled lazily on the first episode and shared by every
    /// episode after it. Lanes seed the generator identically, so all
    /// lanes optimize the same graph -- the single-graph compiler regime.
    /// Episode diversity comes from per-episode Gumbel noise seeds.
    Fixed {
        remaining: Option<u64>,
        generator: WhittleGraphGenerator,
        root: Option<WhittleGraphId>,
    },
}

impl RootSource<WhittleEngine> for CliRoots {
    fn next_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        let remaining = match self {
            Self::Generated(source) => &mut source.remaining,
            Self::Fixed { remaining, .. } => remaining,
        };
        match remaining.as_mut() {
            Some(0) => return Ok(None),
            Some(remaining) => *remaining -= 1,
            None => {}
        }

        match self {
            Self::Generated(source) => source.generator.sample_root_into(engine).map(Some),
            Self::Fixed {
                generator, root, ..
            } => {
                if root.is_none() {
                    *root = Some(generator.sample_root_into(engine)?);
                }
                Ok(*root)
            }
        }
    }

    fn episode_roots_are_owned(&self) -> bool {
        matches!(self, Self::Generated(_))
    }

    /// Opponent rollouts replay the shared root without consuming the
    /// episode budget. Generated mode has no fixed root (policy-opponent
    /// rollouts are a fixed-root feature).
    fn fixed_root(&mut self, engine: &mut WhittleEngine) -> EngineResult<Option<WhittleGraphId>> {
        match self {
            Self::Generated(_) => Ok(None),
            Self::Fixed {
                generator, root, ..
            } => {
                if root.is_none() {
                    *root = Some(generator.sample_root_into(engine)?);
                }
                Ok(*root)
            }
        }
    }
}

// One instance per lane; the policy variant's rollout bookkeeping
// outweighs the others and boxing it buys nothing at this count.
struct CliPolicyProvider {
    provider: PolicyReferenceProvider,
    arena: Option<CliArenaRoots>,
}

impl CliPolicyProvider {
    const fn new(provider: PolicyReferenceProvider) -> Self {
        Self {
            provider,
            arena: None,
        }
    }
}

struct CliArenaRoots {
    size: usize,
    seed: u64,
    roots: HashMap<usize, WhittleGraphId>,
}

impl ReferenceProvider<WhittleEngine> for CliPolicyProvider {
    fn reference(
        &mut self,
        engine: &mut WhittleEngine,
        root: WhittleGraphId,
    ) -> EngineResult<Option<Reference>> {
        self.provider.reference(engine, root)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut WhittleEngine,
        root: WhittleGraphId,
        extractor: &mut X,
        candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<WhittleEngine>,
    {
        self.provider.reference_with_features(
            engine,
            root,
            extractor,
            candidate_options,
            export_position,
        )
    }

    fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        ReferenceProvider::<WhittleEngine>::rollout_due(&self.provider, latest)
    }

    fn claim_rollout(&mut self, latest: Option<ModelVersion>) -> bool {
        ReferenceProvider::<WhittleEngine>::claim_rollout(&mut self.provider, latest)
    }

    fn begin_rollout(&mut self, version: Option<ModelVersion>) {
        ReferenceProvider::<WhittleEngine>::begin_rollout(&mut self.provider, version);
    }

    fn finish_rollout(&mut self, outcome: Option<RolloutOutcome>) {
        ReferenceProvider::<WhittleEngine>::finish_rollout(&mut self.provider, outcome);
    }

    fn claim_sample_rollout(&mut self, latest: Option<ModelVersion>) -> Option<ModelVersion> {
        ReferenceProvider::<WhittleEngine>::claim_sample_rollout(&mut self.provider, latest)
    }

    fn finish_sample_rollout(&mut self, version: ModelVersion, outcome: Option<RolloutOutcome>) {
        ReferenceProvider::<WhittleEngine>::finish_sample_rollout(
            &mut self.provider,
            version,
            outcome,
        );
    }

    fn sampled_trajectory_mode(&self) -> bool {
        ReferenceProvider::<WhittleEngine>::sampled_trajectory_mode(&self.provider)
    }

    fn sampled_tree_mode(&self) -> bool {
        ReferenceProvider::<WhittleEngine>::sampled_tree_mode(&self.provider)
    }

    fn arena_parallelism(&self) -> usize {
        self.arena.as_ref().map_or(0, |arena| arena.size)
    }

    fn finish_sampled_trajectory(&mut self, outcome: Option<RolloutOutcome>) -> Option<Reference> {
        ReferenceProvider::<WhittleEngine>::finish_sampled_trajectory(&mut self.provider, outcome)
    }

    fn claim_arena_rollout(
        &mut self,
        latest: Option<ModelVersion>,
        lane: usize,
        lanes: usize,
    ) -> Option<ArenaRolloutClaim> {
        ReferenceProvider::<WhittleEngine>::claim_arena_rollout(
            &mut self.provider,
            latest,
            lane,
            lanes,
        )
    }

    fn arena_root(
        &mut self,
        engine: &mut WhittleEngine,
        index: usize,
    ) -> EngineResult<Option<WhittleGraphId>> {
        let Some(arena) = &mut self.arena else {
            return Ok(None);
        };
        if index >= arena.size {
            return Ok(None);
        }
        if let Some(root) = arena.roots.get(&index) {
            return Ok(Some(*root));
        }
        let mut generator = WhittleGraphGenerator::from_seed(
            whittle_generator_config(),
            arena_graph_seed(arena.seed, index),
        );
        let root = generator.sample_root_into(engine)?;
        arena.roots.insert(index, root);
        Ok(Some(root))
    }

    fn finish_arena_rollout(
        &mut self,
        claim: ArenaRolloutClaim,
        score: Option<f32>,
        outcome: Option<RolloutOutcome>,
    ) {
        ReferenceProvider::<WhittleEngine>::finish_arena_rollout(
            &mut self.provider,
            claim,
            score,
            outcome,
        );
    }

    fn per_root_policy_mode(&self) -> bool {
        ReferenceProvider::<WhittleEngine>::per_root_policy_mode(&self.provider)
    }

    fn claim_per_root_policy(
        &mut self,
        latest: Option<ModelVersion>,
    ) -> Option<EpisodeRolloutClaim> {
        ReferenceProvider::<WhittleEngine>::claim_per_root_policy(&mut self.provider, latest)
    }

    fn finish_per_root_policy(
        &mut self,
        claim: EpisodeRolloutClaim,
        outcome: Option<RolloutOutcome>,
    ) -> Option<Reference> {
        ReferenceProvider::<WhittleEngine>::finish_per_root_policy(
            &mut self.provider,
            claim,
            outcome,
        )
    }

    fn admission_ready(&self) -> bool {
        ReferenceProvider::<WhittleEngine>::admission_ready(&self.provider)
    }
}

fn arena_graph_seed(seed: u64, index: usize) -> u64 {
    let mut value =
        seed ^ 0x6172_656e_615f_6772 ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[allow(clippy::large_enum_variant)]
enum CliReferenceProvider {
    None,
    Root(RootBaselineProvider),
    Greedy(GreedyReferenceProvider),
    Beam(BeamReferenceProvider),
    Random(RandomReferenceProvider),
    SelfAverage(SelfAverageProvider),
    Policy(CliPolicyProvider),
}

impl ReferenceProvider<WhittleEngine> for CliReferenceProvider {
    fn expects_reference(&self) -> bool {
        !matches!(self, Self::None)
    }

    fn reference(
        &mut self,
        engine: &mut WhittleEngine,
        root: WhittleGraphId,
    ) -> EngineResult<Option<Reference>> {
        match self {
            Self::None => Ok(None),
            Self::Root(provider) => provider.reference(engine, root),
            Self::Greedy(provider) => provider.reference(engine, root),
            Self::Beam(provider) => provider.reference(engine, root),
            Self::Random(provider) => provider.reference(engine, root),
            Self::SelfAverage(provider) => provider.reference(engine, root),
            Self::Policy(provider) => provider.reference(engine, root),
        }
    }

    // Forwarded explicitly: the trait default falls back to reference(),
    // which silently drops per-step opponent features (root references lost
    // their states this way -- scalar present, no state).
    fn reference_with_features<X>(
        &mut self,
        engine: &mut WhittleEngine,
        root: WhittleGraphId,
        extractor: &mut X,
        candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<WhittleEngine>,
    {
        match self {
            Self::None => Ok(None),
            Self::Root(provider) => provider.reference_with_features(
                engine,
                root,
                extractor,
                candidate_options,
                export_position,
            ),
            Self::Greedy(provider) => provider.reference_with_features(
                engine,
                root,
                extractor,
                candidate_options,
                export_position,
            ),
            Self::Beam(provider) => provider.reference_with_features(
                engine,
                root,
                extractor,
                candidate_options,
                export_position,
            ),
            Self::Random(provider) => provider.reference_with_features(
                engine,
                root,
                extractor,
                candidate_options,
                export_position,
            ),
            Self::SelfAverage(provider) => provider.reference_with_features(
                engine,
                root,
                extractor,
                candidate_options,
                export_position,
            ),
            Self::Policy(provider) => provider.reference_with_features(
                engine,
                root,
                extractor,
                candidate_options,
                export_position,
            ),
        }
    }

    // The enum must forward observe explicitly: the trait default is a
    // no-op, which would silently starve the self-average EMA. Any future
    // stateful provider variant must be forwarded here.
    fn observe(&mut self, learner_reward: f32) {
        match self {
            Self::None
            | Self::Root(_)
            | Self::Greedy(_)
            | Self::Beam(_)
            | Self::Random(_)
            | Self::Policy(_) => {}
            Self::SelfAverage(provider) => {
                ReferenceProvider::<WhittleEngine>::observe(provider, learner_reward);
            }
        }
    }

    // The rollout hooks likewise forward explicitly: the defaults are
    // no-ops, which would silently disable the policy opponent.
    fn rollout_due(&self, latest: Option<gz_engine::ModelVersion>) -> bool {
        match self {
            Self::Policy(provider) => {
                ReferenceProvider::<WhittleEngine>::rollout_due(provider, latest)
            }
            _ => false,
        }
    }

    fn claim_rollout(&mut self, latest: Option<gz_engine::ModelVersion>) -> bool {
        match self {
            Self::Policy(provider) => {
                ReferenceProvider::<WhittleEngine>::claim_rollout(provider, latest)
            }
            _ => false,
        }
    }

    fn begin_rollout(&mut self, version: Option<gz_engine::ModelVersion>) {
        if let Self::Policy(provider) = self {
            ReferenceProvider::<WhittleEngine>::begin_rollout(provider, version);
        }
    }

    fn finish_rollout(&mut self, outcome: Option<RolloutOutcome>) {
        if let Self::Policy(provider) = self {
            ReferenceProvider::<WhittleEngine>::finish_rollout(provider, outcome);
        }
    }

    fn claim_sample_rollout(
        &mut self,
        latest: Option<gz_engine::ModelVersion>,
    ) -> Option<gz_engine::ModelVersion> {
        match self {
            Self::Policy(provider) => {
                ReferenceProvider::<WhittleEngine>::claim_sample_rollout(provider, latest)
            }
            _ => None,
        }
    }

    fn finish_sample_rollout(
        &mut self,
        version: gz_engine::ModelVersion,
        outcome: Option<RolloutOutcome>,
    ) {
        if let Self::Policy(provider) = self {
            ReferenceProvider::<WhittleEngine>::finish_sample_rollout(provider, version, outcome);
        }
    }

    fn sampled_trajectory_mode(&self) -> bool {
        match self {
            Self::Policy(provider) => {
                ReferenceProvider::<WhittleEngine>::sampled_trajectory_mode(provider)
            }
            _ => false,
        }
    }

    fn sampled_tree_mode(&self) -> bool {
        match self {
            Self::Policy(provider) => {
                ReferenceProvider::<WhittleEngine>::sampled_tree_mode(provider)
            }
            _ => false,
        }
    }

    fn arena_parallelism(&self) -> usize {
        match self {
            Self::Policy(provider) => {
                ReferenceProvider::<WhittleEngine>::arena_parallelism(provider)
            }
            _ => 0,
        }
    }

    fn finish_sampled_trajectory(&mut self, outcome: Option<RolloutOutcome>) -> Option<Reference> {
        match self {
            Self::Policy(provider) => {
                ReferenceProvider::<WhittleEngine>::finish_sampled_trajectory(provider, outcome)
            }
            _ => None,
        }
    }

    fn claim_arena_rollout(
        &mut self,
        latest: Option<ModelVersion>,
        lane: usize,
        lanes: usize,
    ) -> Option<ArenaRolloutClaim> {
        match self {
            Self::Policy(provider) => provider.claim_arena_rollout(latest, lane, lanes),
            _ => None,
        }
    }

    fn arena_root(
        &mut self,
        engine: &mut WhittleEngine,
        index: usize,
    ) -> EngineResult<Option<WhittleGraphId>> {
        match self {
            Self::Policy(provider) => provider.arena_root(engine, index),
            _ => Ok(None),
        }
    }

    fn finish_arena_rollout(
        &mut self,
        claim: ArenaRolloutClaim,
        score: Option<f32>,
        outcome: Option<RolloutOutcome>,
    ) {
        if let Self::Policy(provider) = self {
            provider.finish_arena_rollout(claim, score, outcome);
        }
    }

    fn per_root_policy_mode(&self) -> bool {
        match self {
            Self::Policy(provider) => provider.per_root_policy_mode(),
            _ => false,
        }
    }

    fn claim_per_root_policy(
        &mut self,
        latest: Option<ModelVersion>,
    ) -> Option<EpisodeRolloutClaim> {
        match self {
            Self::Policy(provider) => provider.claim_per_root_policy(latest),
            _ => None,
        }
    }

    fn finish_per_root_policy(
        &mut self,
        claim: EpisodeRolloutClaim,
        outcome: Option<RolloutOutcome>,
    ) -> Option<Reference> {
        match self {
            Self::Policy(provider) => provider.finish_per_root_policy(claim, outcome),
            _ => None,
        }
    }

    fn admission_ready(&self) -> bool {
        match self {
            Self::Policy(provider) => ReferenceProvider::<WhittleEngine>::admission_ready(provider),
            _ => true,
        }
    }
}
