#![forbid(unsafe_code)]

mod selfplay;

use selfplay::{SelfplayConfig, run};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        eprintln!("{}", usage());
        std::process::exit(2);
    };

    let result = match command.as_str() {
        "selfplay" => parse_selfplay(args.collect()).and_then(run).map(|summary| {
            println!(
                "episodes appended={} dropped={} rows={} labels win/loss/tie={}/{}/{} eval_batches={} mean_batch={:.3} counters produced={} consumed={}",
                summary.episodes_appended,
                summary.episodes_dropped,
                summary.rows_produced,
                summary.wins,
                summary.losses,
                summary.ties,
                summary.eval_batch_count,
                summary.mean_eval_batch_size,
                summary.counters.produced_rows,
                summary.counters.consumed_rows,
            );
        }),
        _ => Err(format!("unknown command: {command}\n{}", usage())),
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn parse_selfplay(args: Vec<String>) -> Result<SelfplayConfig, String> {
    let mut config = SelfplayConfig::default();
    let mut max_batch = None;
    let mut index = 0;

    while index < args.len() {
        let flag = &args[index];
        index += 1;

        let Some(value) = args.get(index) else {
            return Err(format!("missing value for {flag}\n{}", usage()));
        };
        index += 1;

        match flag.as_str() {
            "--replay-dir" => config.replay_dir = Some(PathBuf::from(value)),
            "--episodes" => config.episodes = parse_u64(flag, value)?,
            "--lanes" => config.lanes = parse_usize(flag, value)?,
            "--workers-per-lane" => config.workers_per_lane = parse_usize(flag, value)?,
            "--reference" => config.reference = value.parse()?,
            "--seed" => config.seed = parse_u64(flag, value)?,
            "--max-steps" => config.max_steps = parse_usize(flag, value)?,
            "--simulations" => config.simulations = parse_usize(flag, value)?,
            "--max-batch" => max_batch = Some(parse_usize(flag, value)?),
            _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
        }
    }

    config.max_batch = max_batch.unwrap_or(config.lanes * config.workers_per_lane);
    config.validate()?;
    Ok(config)
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an unsigned integer"))
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects a positive integer"))
}

fn usage() -> &'static str {
    "usage: graphzero selfplay --replay-dir PATH [--episodes N] [--lanes L] [--workers-per-lane W] [--reference root|greedy|beam|random|none] [--seed S] [--max-steps M] [--simulations K] [--max-batch B]"
}
