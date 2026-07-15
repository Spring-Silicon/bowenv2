use gz_orchestrator::{AdaptiveAdmissionSchedule, AdmissionSmoothingConfig};
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;

fn schedule() -> AdaptiveAdmissionSchedule {
    AdaptiveAdmissionSchedule::new(
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        AdmissionSmoothingConfig {
            initial_episode_eval_work: NonZeroU64::new(1_000).unwrap(),
        },
    )
    .unwrap()
}

#[test]
fn bootstrap_admission_is_globally_single_token() {
    let mut schedule = schedule();

    let first = schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);
    assert_eq!(schedule.total_waiting(), 99);
    schedule.clear_lane(0);
    let same_instant = schedule.request(Duration::ZERO, 1, 25, 0, 0, 1);

    assert_eq!(first.limit, 1);
    assert_eq!(first.bootstrap_grants, 1);
    assert_eq!(same_instant.limit, 0);
    assert_eq!(schedule.total_waiting(), 75);
}

#[test]
fn no_capacity_model_means_no_admission_after_bootstrap() {
    let mut schedule = schedule();
    schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);

    let decision = schedule.request(Duration::from_millis(1), 1, 25, 0, 0, 1);

    assert_eq!(decision.limit, 0);
    assert_eq!(decision.retry_after, Some(Duration::from_millis(1)));
}

#[test]
fn workers_must_partition_evenly_across_lanes() {
    let result = AdaptiveAdmissionSchedule::new(
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(99).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        AdmissionSmoothingConfig {
            initial_episode_eval_work: NonZeroU64::new(1_000).unwrap(),
        },
    );

    assert!(result.is_err());
}

#[test]
fn virtual_clock_uses_evaluator_service_and_episode_work() {
    let mut schedule = schedule();
    schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);

    let first = schedule.request(Duration::from_secs(10), 0, 24, 10_000, 10_000_000_000, 1);
    let early = schedule.request(
        Duration::from_millis(10_500),
        0,
        23,
        10_500,
        10_500_000_000,
        1,
    );
    let due = schedule.request(Duration::from_secs(11), 0, 23, 11_000, 11_000_000_000, 1);

    assert_eq!(first.paced_grants, 1);
    assert_eq!(early.limit, 0);
    assert_eq!(early.retry_after, Some(Duration::from_millis(500)));
    assert_eq!(due.paced_grants, 1);
    assert_eq!(schedule.eval_capacity_ema(), Some(1_000.0));
    assert_eq!(schedule.episode_eval_work_ema(), Some(1_000.0));
    assert_eq!(schedule.admission_gap(), Some(Duration::from_secs(1)));
}

#[test]
fn evaluator_capacity_aggregates_subsecond_pipeline_samples() {
    let mut schedule = schedule();
    schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);

    schedule.request(Duration::from_millis(100), 0, 24, 1_000, 100_000_000, 1);
    schedule.request(Duration::from_millis(200), 0, 24, 1_100, 200_000_000, 1);

    assert_eq!(schedule.eval_capacity_ema(), None);

    schedule.request(Duration::from_millis(1_100), 0, 24, 2_000, 1_100_000_000, 1);

    let capacity = schedule.eval_capacity_ema().unwrap();
    assert!((capacity - 2_000.0 / 1.1).abs() < 1e-9);
}

#[test]
fn completed_bootstrap_episode_unlocks_a_short_initial_capacity_sample() {
    let mut schedule = schedule();
    schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);
    schedule.request(Duration::from_millis(100), 0, 24, 6, 100_000_000, 1);
    assert_eq!(schedule.eval_capacity_ema(), None);

    schedule.observe_episode_work(6);
    let decision = schedule.request(Duration::from_millis(100), 0, 24, 6, 100_000_000, 1);

    assert_eq!(schedule.eval_capacity_ema(), Some(60.0));
    assert_eq!(decision.paced_grants, 1);
}

#[test]
fn delayed_caller_does_not_accumulate_catch_up_tokens() {
    let mut schedule = schedule();
    schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);
    schedule.request(Duration::from_secs(10), 0, 24, 10_000, 10_000_000_000, 1);

    let delayed = schedule.request(Duration::from_secs(20), 0, 23, 20_000, 20_000_000_000, 1);
    let same_instant = schedule.request(Duration::from_secs(20), 0, 22, 20_000, 20_000_000_000, 1);

    assert_eq!(delayed.limit, 1);
    assert_eq!(same_instant.limit, 0);
    assert_eq!(same_instant.retry_after, Some(Duration::from_secs(1)));
}

#[test]
fn evaluator_pressure_cannot_bypass_the_virtual_clock() {
    let mut schedule = schedule();
    schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);
    schedule.request(Duration::from_secs(10), 0, 24, 10_000, 10_000_000_000, 1);

    let pressure = schedule.request(
        Duration::from_millis(10_500),
        0,
        23,
        10_500,
        10_500_000_000,
        1,
    );
    let old_due = schedule.request(Duration::from_secs(11), 0, 23, 11_000, 11_000_000_000, 1);

    assert_eq!(pressure.limit, 0);
    assert_eq!(old_due.paced_grants, 1);
}

#[test]
fn low_evaluator_pressure_shortens_only_future_clock_intervals() {
    let mut schedule = AdaptiveAdmissionSchedule::new(
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        AdmissionSmoothingConfig {
            initial_episode_eval_work: NonZeroU64::new(1_000).unwrap(),
        },
    )
    .unwrap();
    schedule.request(Duration::ZERO, 0, 25, 0, 0, 0);

    let first = schedule.request(Duration::from_secs(10), 0, 24, 10_000, 10_000_000_000, 50);
    let gap = schedule.admission_gap().unwrap();
    let same_instant = schedule.request(Duration::from_secs(10), 0, 23, 10_000, 10_000_000_000, 50);
    let due = schedule.request(
        Duration::from_secs(10) + gap,
        0,
        23,
        10_000,
        10_000_000_000,
        50,
    );

    assert_eq!(first.paced_grants, 1);
    assert_eq!(schedule.pressure_gain(), 1.5);
    assert_eq!(gap, Duration::from_secs_f64(2.0 / 3.0));
    assert_eq!(same_instant.limit, 0);
    assert_eq!(due.paced_grants, 1);
}

#[test]
fn unused_grants_can_be_restored_or_cleared() {
    let mut schedule = schedule();
    let decision = schedule.request(Duration::ZERO, 0, 25, 0, 0, 1);
    assert_eq!(decision.limit, 1);

    schedule.restore_unused(0, 1, true);
    assert_eq!(schedule.total_waiting(), 100);
    assert_eq!(
        schedule
            .request(Duration::from_millis(1), 0, 25, 0, 0, 1)
            .bootstrap_grants,
        1
    );
    schedule.clear_lane(0);
    assert_eq!(schedule.total_waiting(), 75);
}
