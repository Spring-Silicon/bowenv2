use gz_orchestrator::{
    SelfplayBenchConfig, SelfplayEpisodeStats, SelfplayRunStats, run_selfplay_benchmark,
    run_serial_selfplay_benchmark,
};

#[test]
fn benchmark_times_a_complete_selfplay_run() {
    let report = run_selfplay_benchmark(SelfplayBenchConfig::new(3), |config| {
        Ok(SelfplayRunStats::new(config.episodes, 6))
    })
    .unwrap();

    assert_eq!(report.episodes, 3);
    assert_eq!(report.steps, 6);
}

#[test]
fn serial_benchmark_counts_completed_episodes_and_steps() {
    let mut calls = 0;

    let report = run_serial_selfplay_benchmark(SelfplayBenchConfig::new(3), || {
        calls += 1;
        Ok(SelfplayEpisodeStats::new(2))
    })
    .unwrap();

    assert_eq!(calls, 3);
    assert_eq!(report.episodes, 3);
    assert_eq!(report.steps, 6);
}
