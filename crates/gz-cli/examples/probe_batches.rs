//! Emits mechanistic-interpretation probe batches for trained checkpoints:
//! the production fixed root (generator seed from argv), every candidate's
//! measured after-cost as ground truth, a value-vs-cost sweep over applied
//! states, and opponent/orientation variants for the pair value head. The
//! Python probe script under the session scratchpad consumes the .gzfb
//! batches plus meta.json.
use gz_cli::selfplay_probe::{ProbeArgs, run_probe};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: probe_batches OUT_DIR [SEED]")?;
    let seed = std::env::args()
        .nth(2)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(42);
    std::fs::create_dir_all(&out_dir)?;
    run_probe(ProbeArgs { out_dir, seed })?;
    Ok(())
}
