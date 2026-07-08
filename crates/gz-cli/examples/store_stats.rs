//! Per-episode outcome dump for a replay store: episode id, final cost,
//! length, stopped flag, label. One line per episode, tab-separated, for
//! offline distribution analysis (episode-noise digs, benchmark parity).
//!
//! Usage: cargo run --release -p gz-cli --example store_stats -- <replay-dir> [start] [end]

use gz_replay::{ReplayEpisodeId, ReplayStore};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: store_stats <replay-dir> [start] [end]");
    let store = ReplayStore::open(std::path::Path::new(&dir)).expect("open store");
    let (episodes, _) = store.episode_counters();
    let start: u64 = args.next().map_or(0, |s| s.parse().expect("start"));
    let end: u64 = args.next().map_or(episodes, |s| s.parse().expect("end"));

    println!("episode\tcost\tlen\tstopped\tlabel\treference_reward");
    for id in start..end.min(episodes) {
        let Ok(Some(record)) = store.episode(ReplayEpisodeId::new(id)) else {
            continue;
        };
        println!(
            "{id}\t{:.1}\t{}\t{}\t{}\t{}",
            -record.outcome.learner_reward,
            record.row_count,
            u8::from(record.outcome.stopped),
            record
                .outcome
                .value_target
                .map_or_else(|| "-".to_owned(), |v| format!("{v:+.0}")),
            record
                .outcome
                .reference
                .as_ref()
                .map_or_else(|| "-".to_owned(), |r| format!("{:.1}", -r.reward)),
        );
    }
}
