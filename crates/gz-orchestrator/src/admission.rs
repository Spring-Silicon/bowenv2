use crate::internal;
use gz_engine::EngineResult;
use std::collections::VecDeque;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const CAPACITY_SAMPLE_BUSY_NS: u64 = 1_000_000_000;
const MAX_PRESSURE_GAIN: f64 = 1.5;
pub(super) const EVAL_PIPELINE_DEPTH: usize = 3;

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

#[derive(Default)]
pub(super) struct EvalPressure {
    outstanding: AtomicUsize,
    reserved: AtomicUsize,
    capacity_work: AtomicU64,
    capacity_busy_ns: AtomicU64,
}

impl EvalPressure {
    pub(super) fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }

    fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    fn capacity_totals(&self) -> (u64, u64) {
        // record_capacity publishes busy time before work. Loading work first
        // means observing a new work total also observes its matching duration.
        let work = self.capacity_work.load(Ordering::Acquire);
        let busy_ns = self.capacity_busy_ns.load(Ordering::Acquire);
        (work, busy_ns)
    }

    fn reserve(&self, count: usize) {
        self.reserved.fetch_add(count, Ordering::AcqRel);
    }

    fn cancel_reservations(&self, count: usize) {
        atomic_saturating_sub(&self.reserved, count);
    }

    pub(super) fn submit(&self, reserved: bool) {
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        if reserved {
            atomic_saturating_sub(&self.reserved, 1);
        }
    }

    pub(super) fn cancel_submission(&self) {
        atomic_saturating_sub(&self.outstanding, 1);
    }

    pub(super) fn complete(&self, count: usize) {
        atomic_saturating_sub(&self.outstanding, count);
    }

    pub(super) fn complete_current_batch(
        &self,
        count: usize,
        capacity_work: usize,
        busy: Duration,
    ) {
        self.complete(count);
        let busy_ns = busy.as_nanos().min(u128::from(u64::MAX)) as u64;
        atomic_saturating_add_u64(&self.capacity_busy_ns, busy_ns.max(1));
        atomic_saturating_add_u64(
            &self.capacity_work,
            u64::try_from(capacity_work).unwrap_or(u64::MAX),
        );
    }
}

fn atomic_saturating_sub(value: &AtomicUsize, count: usize) {
    let _ = value.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(count))
    });
}

fn atomic_saturating_add_u64(value: &AtomicU64, count: u64) {
    let _ = value.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_add(count))
    });
}

pub(super) struct SharedAdmissionShaper {
    started: Instant,
    schedule: Mutex<AdaptiveAdmissionSchedule>,
    pressure: Arc<EvalPressure>,
    bootstrap_grants: AtomicUsize,
    paced_grants: AtomicUsize,
    max_waiting: AtomicUsize,
    waiting_lanes: Vec<AtomicBool>,
    next_stats_ms: AtomicUsize,
}

impl SharedAdmissionShaper {
    fn new(
        lanes: usize,
        workers_per_lane: NonZeroUsize,
        evaluator_processes: usize,
        max_batch: NonZeroUsize,
        config: AdmissionSmoothingConfig,
        pressure: Arc<EvalPressure>,
    ) -> EngineResult<Self> {
        let lanes = NonZeroUsize::new(lanes).ok_or_else(|| internal("zero lanes"))?;
        let evaluator_processes = NonZeroUsize::new(evaluator_processes)
            .ok_or_else(|| internal("zero evaluator processes"))?;
        let total_workers = lanes
            .get()
            .checked_mul(workers_per_lane.get())
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| internal("worker count overflow"))?;
        let target_outstanding = max_batch
            .get()
            .checked_mul(EVAL_PIPELINE_DEPTH)
            .and_then(|target| target.checked_mul(evaluator_processes.get()))
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| internal("evaluator pressure target overflow"))?;
        let schedule = AdaptiveAdmissionSchedule::new(
            lanes,
            total_workers,
            evaluator_processes,
            target_outstanding,
            config,
        )
        .map_err(|_| internal("invalid admission smoothing config"))?;
        let lane_count = lanes.get();
        Ok(Self {
            started: Instant::now(),
            schedule: Mutex::new(schedule),
            pressure,
            bootstrap_grants: AtomicUsize::new(0),
            paced_grants: AtomicUsize::new(0),
            max_waiting: AtomicUsize::new(0),
            waiting_lanes: (0..lane_count).map(|_| AtomicBool::new(false)).collect(),
            next_stats_ms: AtomicUsize::new(30_000),
        })
    }

    pub(super) fn request(
        &self,
        lane: usize,
        idle_workers: usize,
    ) -> EngineResult<AdmissionDecision> {
        if idle_workers == 0 && !self.waiting_lanes[lane].load(Ordering::Acquire) {
            return Ok(AdmissionDecision::default());
        }
        let mut schedule = self
            .schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?;
        let (capacity_work, capacity_busy_ns) = self.pressure.capacity_totals();
        let outstanding = self.pressure.outstanding();
        let decision = schedule.request(
            self.started.elapsed(),
            lane,
            idle_workers,
            capacity_work,
            capacity_busy_ns,
            outstanding,
        );
        let waiting = schedule.total_waiting();
        self.waiting_lanes[lane].store(schedule.lane_waiting(lane), Ordering::Release);
        self.pressure.reserve(decision.limit);
        self.bootstrap_grants
            .fetch_add(decision.bootstrap_grants, Ordering::Relaxed);
        self.paced_grants
            .fetch_add(decision.paced_grants, Ordering::Relaxed);
        self.max_waiting.fetch_max(waiting, Ordering::Relaxed);
        self.report_stats(&schedule, waiting)?;
        Ok(decision)
    }

    pub(super) fn observe_episode_work(&self, evaluations: u64) -> EngineResult<()> {
        self.schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?
            .observe_episode_work(evaluations);
        Ok(())
    }

    pub(super) fn finish_admission(
        &self,
        lane: usize,
        decision: AdmissionDecision,
        admitted: usize,
        roots_exhausted: bool,
    ) -> EngineResult<()> {
        let unused = decision.limit.saturating_sub(admitted);
        self.pressure.cancel_reservations(unused);
        let mut schedule = self
            .schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?;
        schedule.restore_unused(lane, unused, decision.bootstrap_grants > 0);
        if roots_exhausted {
            schedule.clear_lane(lane);
        }
        self.waiting_lanes[lane].store(schedule.lane_waiting(lane), Ordering::Release);
        Ok(())
    }

    pub(super) fn clear_lane(&self, lane: usize) -> EngineResult<()> {
        self.schedule
            .lock()
            .map_err(|_| internal("admission shaper lock poisoned"))?
            .clear_lane(lane);
        self.waiting_lanes[lane].store(false, Ordering::Release);
        Ok(())
    }

    fn report_stats(
        &self,
        schedule: &AdaptiveAdmissionSchedule,
        waiting: usize,
    ) -> EngineResult<()> {
        const STATS_INTERVAL_MS: usize = 30_000;
        let elapsed_ms = usize::try_from(self.started.elapsed().as_millis()).unwrap_or(usize::MAX);
        let next = self.next_stats_ms.load(Ordering::Relaxed);
        if elapsed_ms < next
            || self
                .next_stats_ms
                .compare_exchange(
                    next,
                    elapsed_ms.saturating_add(STATS_INTERVAL_MS),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_err()
        {
            return Ok(());
        }
        let eval_capacity_milli = schedule
            .eval_capacity_ema()
            .map_or(0, |capacity| (capacity * 1_000.0).round() as u64);
        let episode_work_milli = schedule
            .episode_eval_work_ema()
            .map_or(0, |work| (work * 1_000.0).round() as u64);
        let gap_us = schedule
            .admission_gap()
            .map_or(0, |gap| gap.as_micros().min(u128::from(u64::MAX)) as u64);
        let pressure_gain_milli = (schedule.pressure_gain() * 1_000.0).round() as u64;
        eprintln!(
            "event=admission_stats outstanding={} reserved={} waiting={} max_waiting={} bootstrap_grants={} paced_grants={} eval_capacity_milli={} episode_work_milli={} pressure_gain_milli={} gap_us={}",
            self.pressure.outstanding(),
            self.pressure.reserved(),
            waiting,
            self.max_waiting.load(Ordering::Relaxed),
            self.bootstrap_grants.load(Ordering::Relaxed),
            self.paced_grants.load(Ordering::Relaxed),
            eval_capacity_milli,
            episode_work_milli,
            pressure_gain_milli,
            gap_us,
        );
        Ok(())
    }
}

pub(super) fn build_admission_shaper(
    lanes: usize,
    evaluator_processes: usize,
    workers_per_lane: NonZeroUsize,
    max_batch: NonZeroUsize,
    admission_stagger: Duration,
    smoothing: Option<AdmissionSmoothingConfig>,
    pressure: Arc<EvalPressure>,
) -> EngineResult<Option<Arc<SharedAdmissionShaper>>> {
    let Some(smoothing) = smoothing else {
        return Ok(None);
    };
    if !admission_stagger.is_zero() {
        return Err(internal(
            "fixed and adaptive admission pacing are mutually exclusive",
        ));
    }
    SharedAdmissionShaper::new(
        lanes,
        workers_per_lane,
        evaluator_processes,
        max_batch,
        smoothing,
        pressure,
    )
    .map(Arc::new)
    .map(Some)
}

pub(super) struct AdmissionPacer {
    stagger: Duration,
    next: Instant,
    resume_offset: Duration,
}

impl AdmissionPacer {
    pub(super) fn new(
        lane: usize,
        lanes: usize,
        workers_per_lane: usize,
        stagger: Duration,
    ) -> Self {
        let now = Instant::now();
        if stagger.is_zero() {
            return Self {
                stagger,
                next: now,
                resume_offset: Duration::ZERO,
            };
        }
        let offset = spread_duration(stagger, lane, lanes);
        eprintln!(
            "event=admission_pacer lane={lane} interval_ms={} first_delay_ms={} cohort_span_ms={}",
            stagger.as_millis(),
            offset.as_millis(),
            stagger.as_millis() * workers_per_lane as u128,
        );
        Self {
            stagger,
            next: now + offset,
            resume_offset: offset,
        }
    }

    pub(super) fn ready(&mut self) -> bool {
        if self.stagger.is_zero() {
            return true;
        }
        let now = Instant::now();
        if now.saturating_duration_since(self.next) >= self.stagger {
            // Do not repay missed admissions in a burst after a closed gate
            // or a fully occupied lane. Reapply the lane's global phase.
            self.next = now + self.resume_offset;
        }
        now >= self.next
    }

    pub(super) fn limit(&self) -> usize {
        if self.stagger.is_zero() {
            usize::MAX
        } else {
            1
        }
    }

    pub(super) fn record(&mut self, admitted: usize) {
        if self.stagger.is_zero() || admitted == 0 {
            return;
        }
        self.next = Instant::now() + self.stagger;
    }

    pub(super) fn sleep_until_ready(&self) -> Option<Duration> {
        if self.stagger.is_zero() {
            return None;
        }
        Some(self.next.saturating_duration_since(Instant::now())).filter(|sleep| !sleep.is_zero())
    }
}

fn spread_duration(duration: Duration, index: usize, count: usize) -> Duration {
    if count <= 1 || index == 0 || duration.is_zero() {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos().saturating_mul(index as u128) / count as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}
