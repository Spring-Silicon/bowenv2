use gz_engine::EngineResult;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelfplayBenchConfig {
    pub episodes: u64,
}

impl SelfplayBenchConfig {
    #[must_use]
    pub const fn new(episodes: u64) -> Self {
        Self { episodes }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelfplayEpisodeStats {
    pub steps: u64,
}

impl SelfplayEpisodeStats {
    #[must_use]
    pub const fn new(steps: u64) -> Self {
        Self { steps }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SelfplayRunStats {
    pub episodes: u64,
    pub steps: u64,
}

impl SelfplayRunStats {
    #[must_use]
    pub const fn new(episodes: u64, steps: u64) -> Self {
        Self { episodes, steps }
    }

    pub fn record_episode(&mut self, episode: SelfplayEpisodeStats) {
        self.episodes += 1;
        self.steps += episode.steps;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfplayBenchReport {
    pub episodes: u64,
    pub steps: u64,
    pub elapsed: Duration,
}

impl SelfplayBenchReport {
    #[must_use]
    pub fn episodes_per_second(self) -> f64 {
        rate(self.episodes, self.elapsed)
    }

    #[must_use]
    pub fn steps_per_second(self) -> f64 {
        rate(self.steps, self.elapsed)
    }
}

pub fn run_selfplay_benchmark<F>(
    config: SelfplayBenchConfig,
    run: F,
) -> EngineResult<SelfplayBenchReport>
where
    F: FnOnce(SelfplayBenchConfig) -> EngineResult<SelfplayRunStats>,
{
    let start = Instant::now();
    let stats = run(config)?;

    Ok(SelfplayBenchReport {
        episodes: stats.episodes,
        steps: stats.steps,
        elapsed: start.elapsed(),
    })
}

pub fn run_serial_selfplay_benchmark<F>(
    config: SelfplayBenchConfig,
    mut run_episode: F,
) -> EngineResult<SelfplayBenchReport>
where
    F: FnMut() -> EngineResult<SelfplayEpisodeStats>,
{
    run_selfplay_benchmark(config, |config| {
        let mut stats = SelfplayRunStats::default();
        for _ in 0..config.episodes {
            stats.record_episode(run_episode()?);
        }
        Ok(stats)
    })
}

fn rate(count: u64, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds == 0.0 {
        0.0
    } else {
        count as f64 / seconds
    }
}
