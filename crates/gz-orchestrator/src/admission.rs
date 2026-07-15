use std::collections::VecDeque;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;

const CAPACITY_SAMPLE_BUSY_NS: u64 = 1_000_000_000;
const MAX_PRESSURE_GAIN: f64 = 1.5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionSmoothingConfig {
    /// Search evaluations expected for a newly admitted episode. Completed
    /// episodes refine this seed, but pacing is available before the first one.
    pub initial_episode_eval_work: NonZeroU64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdmissionDecision {
    pub limit: usize,
    pub bootstrap_grants: usize,
    pub paced_grants: usize,
    pub retry_after: Option<Duration>,
}

pub struct AdaptiveAdmissionSchedule {
    waiting: Vec<usize>,
    queued: Vec<bool>,
    waiting_order: VecDeque<usize>,
    total_waiting: usize,
    next_admission: Duration,
    bootstrap_admitted: bool,
    evaluator_processes: usize,
    target_outstanding: usize,
    last_capacity_work: u64,
    last_capacity_busy_ns: u64,
    allow_short_capacity_sample: bool,
    eval_capacity_ema: Option<f64>,
    outstanding_ema: Option<f64>,
    episode_eval_work_ema: Option<f64>,
}

impl AdaptiveAdmissionSchedule {
    pub fn new(
        lanes: NonZeroUsize,
        total_workers: NonZeroUsize,
        evaluator_processes: NonZeroUsize,
        target_outstanding: NonZeroUsize,
        config: AdmissionSmoothingConfig,
    ) -> Result<Self, &'static str> {
        if !total_workers.get().is_multiple_of(lanes.get()) {
            return Err("workers are not evenly partitioned across lanes");
        }
        let workers_per_lane = total_workers.get() / lanes.get();
        Ok(Self {
            waiting: vec![workers_per_lane; lanes.get()],
            queued: vec![false; lanes.get()],
            waiting_order: VecDeque::with_capacity(lanes.get()),
            total_waiting: total_workers.get(),
            next_admission: Duration::ZERO,
            bootstrap_admitted: false,
            evaluator_processes: evaluator_processes.get(),
            target_outstanding: target_outstanding.get(),
            last_capacity_work: 0,
            last_capacity_busy_ns: 0,
            allow_short_capacity_sample: false,
            eval_capacity_ema: None,
            outstanding_ema: None,
            episode_eval_work_ema: Some(config.initial_episode_eval_work.get() as f64),
        })
    }

    pub fn request(
        &mut self,
        now: Duration,
        lane: usize,
        idle_workers: usize,
        capacity_work: u64,
        capacity_busy_ns: u64,
        outstanding: usize,
    ) -> AdmissionDecision {
        self.observe_eval_capacity(capacity_work, capacity_busy_ns, outstanding);
        self.set_waiting(lane, idle_workers);
        if idle_workers == 0 {
            return AdmissionDecision::default();
        }

        let front = self.front_waiting_lane() == Some(lane);
        let gap = self.admission_gap();
        let bootstrap_grants = usize::from(front && !self.bootstrap_admitted);
        if bootstrap_grants > 0 {
            self.bootstrap_admitted = true;
        }
        let paced_grants = usize::from(
            front && bootstrap_grants == 0 && gap.is_some() && now >= self.next_admission,
        );
        let limit = bootstrap_grants + paced_grants;
        self.take_from_lane(lane, limit);

        if limit > 0
            && let Some(gap) = gap
        {
            // Never accumulate token debt. A delayed caller still releases one
            // worker and starts the next interval from the actual release time.
            self.next_admission = now.saturating_add(gap);
        }

        let retry_after = (self.waiting[lane] > 0).then(|| {
            if front && gap.is_some() {
                self.next_admission
                    .saturating_sub(now)
                    .max(Duration::from_millis(1))
            } else {
                Duration::from_millis(1)
            }
        });
        AdmissionDecision {
            limit,
            bootstrap_grants,
            paced_grants,
            retry_after,
        }
    }

    pub fn observe_episode_work(&mut self, evaluations: u64) {
        if evaluations == 0 {
            return;
        }
        if self.eval_capacity_ema.is_none() {
            self.allow_short_capacity_sample = true;
        }
        self.episode_eval_work_ema = Some(ema(self.episode_eval_work_ema, evaluations as f64, 0.1));
    }

    pub fn restore_unused(&mut self, lane: usize, unused: usize, bootstrap: bool) {
        if unused == 0 {
            return;
        }
        if bootstrap {
            self.bootstrap_admitted = false;
        }
        self.waiting[lane] = self.waiting[lane].saturating_add(unused);
        self.total_waiting = self.total_waiting.saturating_add(unused);
        self.queue_lane(lane);
    }

    pub fn clear_lane(&mut self, lane: usize) {
        self.total_waiting = self.total_waiting.saturating_sub(self.waiting[lane]);
        self.waiting[lane] = 0;
        self.discard_empty_front();
    }

    #[must_use]
    pub const fn total_waiting(&self) -> usize {
        self.total_waiting
    }

    #[must_use]
    pub fn lane_waiting(&self, lane: usize) -> bool {
        self.waiting[lane] > 0
    }

    #[must_use]
    pub const fn eval_capacity_ema(&self) -> Option<f64> {
        self.eval_capacity_ema
    }

    #[must_use]
    pub const fn episode_eval_work_ema(&self) -> Option<f64> {
        self.episode_eval_work_ema
    }

    #[must_use]
    pub fn pressure_gain(&self) -> f64 {
        self.outstanding_ema.map_or(1.0, |outstanding| {
            (self.target_outstanding as f64 / outstanding.max(1.0)).clamp(1.0, MAX_PRESSURE_GAIN)
        })
    }

    #[must_use]
    pub fn admission_gap(&self) -> Option<Duration> {
        let seconds =
            self.episode_eval_work_ema? / (self.eval_capacity_ema? * self.pressure_gain());
        if !seconds.is_finite() || seconds <= 0.0 {
            return None;
        }
        Some(Duration::from_secs_f64(seconds.max(0.000_001)))
    }

    fn observe_eval_capacity(
        &mut self,
        capacity_work: u64,
        capacity_busy_ns: u64,
        outstanding: usize,
    ) {
        if capacity_work < self.last_capacity_work || capacity_busy_ns < self.last_capacity_busy_ns
        {
            self.last_capacity_work = capacity_work;
            self.last_capacity_busy_ns = capacity_busy_ns;
            return;
        }
        let work = capacity_work - self.last_capacity_work;
        let busy_ns = capacity_busy_ns - self.last_capacity_busy_ns;
        if work == 0 || (busy_ns < CAPACITY_SAMPLE_BUSY_NS && !self.allow_short_capacity_sample) {
            return;
        }
        self.last_capacity_work = capacity_work;
        self.last_capacity_busy_ns = capacity_busy_ns;
        self.allow_short_capacity_sample = false;
        let per_process_rate = work as f64 * 1_000_000_000.0 / busy_ns as f64;
        let capacity = per_process_rate * self.evaluator_processes as f64;
        if capacity.is_finite() && capacity > 0.0 {
            self.eval_capacity_ema = Some(ema(self.eval_capacity_ema, capacity, 0.2));
            self.outstanding_ema = Some(ema(self.outstanding_ema, outstanding as f64, 0.2));
        }
    }

    fn set_waiting(&mut self, lane: usize, idle_workers: usize) {
        let previous = self.waiting[lane];
        self.total_waiting = self
            .total_waiting
            .saturating_sub(previous)
            .saturating_add(idle_workers);
        self.waiting[lane] = idle_workers;
        self.queue_lane(lane);
        self.discard_empty_front();
    }

    fn queue_lane(&mut self, lane: usize) {
        if self.waiting[lane] > 0 && !self.queued[lane] {
            self.waiting_order.push_back(lane);
            self.queued[lane] = true;
        }
    }

    fn take_from_lane(&mut self, lane: usize, count: usize) -> usize {
        let taken = self.waiting[lane].min(count);
        self.waiting[lane] -= taken;
        self.total_waiting -= taken;
        if self.waiting_order.front() == Some(&lane) {
            self.waiting_order.pop_front();
            self.queued[lane] = false;
            self.queue_lane(lane);
        }
        self.discard_empty_front();
        taken
    }

    fn front_waiting_lane(&mut self) -> Option<usize> {
        self.discard_empty_front();
        self.waiting_order.front().copied()
    }

    fn discard_empty_front(&mut self) {
        while self
            .waiting_order
            .front()
            .is_some_and(|&lane| self.waiting[lane] == 0)
        {
            if let Some(lane) = self.waiting_order.pop_front() {
                self.queued[lane] = false;
            }
        }
    }
}

fn ema(previous: Option<f64>, sample: f64, alpha: f64) -> f64 {
    previous.map_or(sample, |previous| previous + alpha * (sample - previous))
}
