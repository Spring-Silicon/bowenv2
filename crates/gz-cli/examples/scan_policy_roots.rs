use gz_engine::{CandidateOptions, GraphEngine, GraphHash};
use gz_engine_whittle::{
    WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor,
    WhittleFeatureExtractorConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
    WhittleGraphId, WhittleRoot,
};
use gz_eval_service::{EvaluatorProcess, EvaluatorProcessConfig, FeatureEvalBackend, Hello};
use gz_features::{FeatureCollator, FeatureExtractor, PositionFeatures};
use gz_search::{GreedySearch, GreedySearchConfig};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

const MAX_CANDIDATES: usize = 1023;
const DEPTH_BUCKETS: [&str; 5] = ["0", "1-7", "8-31", "32-63", "64+"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_dir = path_arg(1)?.ok_or(
        "usage: scan_policy_roots CHECKPOINT_DIR [SEED_START] [COUNT] [MAX_STEPS] [DEVICE] [CHECKPOINT_POINTER]",
    )?;
    let seed_start = u64_arg(2)?.unwrap_or(0);
    let count = usize_arg(3)?.unwrap_or(128);
    let max_steps = usize_arg(4)?.unwrap_or(64);
    let device = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "cuda:0".to_owned());
    let checkpoint_pointer = std::env::args().nth(6);
    let capacity = NonZeroUsize::new(count).ok_or("COUNT must be nonzero")?;
    let count_u32 = u32::try_from(count).map_err(|_| "COUNT exceeds the protocol limit")?;
    let max_steps_u32 = u32::try_from(max_steps).map_err(|_| "MAX_STEPS exceeds u32")?;
    if max_steps_u32 == 0 {
        return Err("MAX_STEPS must be nonzero".into());
    }
    let seed_end = seed_start
        .checked_add(u64::try_from(count).map_err(|_| "COUNT exceeds u64")?)
        .ok_or("seed range overflow")?;

    let mut engine = whittle_engine()?;
    let mut extractor = WhittleFeatureExtractor::with_config(
        &engine,
        WhittleFeatureExtractorConfig {
            max_actions: MAX_CANDIDATES as u32 + 1,
            ..WhittleFeatureExtractorConfig::default()
        },
    );
    let schema = extractor.schema().clone();
    let checkpoint_dir = absolute(&checkpoint_dir)?;
    let socket_path =
        std::env::temp_dir().join(format!("gz-policy-root-scan-{}.sock", std::process::id()));
    let mut evaluator_args = vec![
        "--backend".to_owned(),
        "torch".to_owned(),
        "--checkpoint-dir".to_owned(),
        checkpoint_dir.display().to_string(),
        "--device".to_owned(),
        device,
        "--max-batch".to_owned(),
        count.to_string(),
        "--poll-interval".to_owned(),
        "3600".to_owned(),
    ];
    if let Some(pointer) = checkpoint_pointer {
        evaluator_args.extend(["--checkpoint-pointer".to_owned(), pointer]);
    }
    let mut process = EvaluatorProcess::spawn(EvaluatorProcessConfig {
        working_dir: PathBuf::from("python"),
        socket_path,
        ready_timeout: Duration::from_secs(30),
        io_timeout: Duration::from_secs(300),
        extra_args: evaluator_args,
        ..EvaluatorProcessConfig::default()
    })?;
    let hello = Hello::new(
        schema.hash(),
        count_u32,
        engine.engine_id(),
        engine.engine_version(),
        engine.action_set_hash(),
    );
    let mut backend = process.connect(&hello)?;
    let model_version = backend.model_version();
    let mut collator = FeatureCollator::new(schema, capacity);
    let options = CandidateOptions {
        max_candidates: Some(MAX_CANDIDATES),
        deterministic_order: true,
    };
    let mut rollouts = Vec::with_capacity(count);
    for offset in 0..count {
        let seed = seed_start
            .checked_add(offset as u64)
            .ok_or("seed range overflow")?;
        let graph = generated_root(&mut engine, seed)?;
        let root_cost = measure_cost(&mut engine, graph)?;
        let root_hash = engine.hash(graph)?;
        rollouts.push(PolicyRollout {
            seed,
            root_hash,
            root_cost,
            best_cost: root_cost,
            best_step: 0,
            graph,
            steps: 0,
            stopped: false,
            nonreducing_moves: 0,
        });
    }

    let mut rows = Vec::with_capacity(count);
    let mut active = Vec::with_capacity(count);
    let mut candidate_batches = Vec::<Vec<WhittleCandidateId>>::with_capacity(count);
    let mut batch_bytes = Vec::new();
    let mut moves_by_depth = [0_u64; DEPTH_BUCKETS.len()];
    let mut nonreducing_by_depth = [0_u64; DEPTH_BUCKETS.len()];
    let mut worsening_by_depth = [0_u64; DEPTH_BUCKETS.len()];
    let mut stops_by_depth = [0_u64; DEPTH_BUCKETS.len()];
    for step in 0..max_steps {
        rows.clear();
        active.clear();
        candidate_batches.clear();
        for (index, rollout) in rollouts.iter().enumerate() {
            if rollout.stopped {
                continue;
            }
            let mut candidates = Vec::new();
            engine.candidates(rollout.graph, options, &mut candidates)?;
            let row = extractor.extract(
                &engine,
                rollout.graph,
                &candidates,
                PositionFeatures {
                    root_step: step as u32,
                    leaf_depth: 0,
                    budget_fraction: max_steps.saturating_sub(step) as f32 / max_steps as f32,
                    budget_step: 1.0 / max_steps as f32,
                    opponent_reward: 0.0,
                    opponent_present: false,
                },
            )?;
            active.push(index);
            candidate_batches.push(candidates);
            rows.push(row);
        }
        if rows.is_empty() {
            break;
        }
        let action_counts = rows
            .iter()
            .map(|row| row.actions.len() as u32)
            .collect::<Vec<_>>();
        collator.collate_into(&rows, &mut batch_bytes)?;
        let outputs = backend.eval(&batch_bytes, &action_counts)?;
        for ((rollout_index, candidates), output) in active
            .iter()
            .copied()
            .zip(candidate_batches.drain(..))
            .zip(outputs.rows)
        {
            let selected = argmax_first(&output.policy_logits);
            let rollout = &mut rollouts[rollout_index];
            let depth_bucket = depth_bucket(step);
            if selected == candidates.len() {
                engine.release(&[], &candidates)?;
                rollout.stopped = true;
                stops_by_depth[depth_bucket] += 1;
                continue;
            }
            let before_cost = measure_cost(&mut engine, rollout.graph)?;
            let applied = engine.apply(rollout.graph, candidates[selected])?;
            let after_cost = measure_cost(&mut engine, applied.after)?;
            moves_by_depth[depth_bucket] += 1;
            nonreducing_by_depth[depth_bucket] += u64::from(after_cost >= before_cost);
            worsening_by_depth[depth_bucket] += u64::from(after_cost > before_cost);
            rollout.nonreducing_moves += usize::from(after_cost >= before_cost);
            engine.release(&[rollout.graph], &candidates)?;
            rollout.graph = applied.after;
            rollout.steps += 1;
            if after_cost < rollout.best_cost {
                rollout.best_cost = after_cost;
                rollout.best_step = rollout.steps;
            }
        }
    }

    let mut results = Vec::with_capacity(count);
    for rollout in rollouts {
        let policy_cost = measure_cost(&mut engine, rollout.graph)?;
        let greedy_cost = engine_greedy_cost(rollout.seed, max_steps)?;
        results.push(ScanResult {
            seed: rollout.seed,
            root_hash: rollout.root_hash,
            root_cost: rollout.root_cost,
            best_cost: rollout.best_cost,
            best_step: rollout.best_step,
            policy_cost,
            greedy_cost,
            gap: policy_cost - greedy_cost,
            policy_steps: rollout.steps,
            stopped: rollout.stopped,
            nonreducing_moves: rollout.nonreducing_moves,
        });
        engine.release(&[rollout.graph], &[])?;
    }
    let mean_root_cost = results.iter().map(|row| row.root_cost).sum::<f32>() / count as f32;
    let mean_policy_cost = results.iter().map(|row| row.policy_cost).sum::<f32>() / count as f32;
    let mean_best_cost = results.iter().map(|row| row.best_cost).sum::<f32>() / count as f32;
    let mean_greedy_cost = results.iter().map(|row| row.greedy_cost).sum::<f32>() / count as f32;
    let mean_reduction = results
        .iter()
        .map(|row| (row.root_cost - row.policy_cost) / row.root_cost.max(1.0))
        .sum::<f32>()
        / count as f32;
    let mean_best_reduction = results
        .iter()
        .map(|row| (row.root_cost - row.best_cost) / row.root_cost.max(1.0))
        .sum::<f32>()
        / count as f32;
    let mean_post_best_degradation = results
        .iter()
        .map(|row| row.policy_cost - row.best_cost)
        .sum::<f32>()
        / count as f32;
    let best_seen = results
        .iter()
        .min_by(|left, right| {
            left.best_cost
                .total_cmp(&right.best_cost)
                .then_with(|| left.seed.cmp(&right.seed))
        })
        .map(|row| (row.best_cost, row.seed))
        .expect("count is nonzero");
    let ended_above_best = results
        .iter()
        .filter(|row| row.policy_cost > row.best_cost)
        .count();
    let stopped = results.iter().filter(|row| row.stopped).count();
    let mean_steps =
        results.iter().map(|row| row.policy_steps).sum::<usize>() as f32 / count as f32;
    results.sort_by(|left, right| {
        right
            .gap
            .total_cmp(&left.gap)
            .then_with(|| right.policy_cost.total_cmp(&left.policy_cost))
            .then_with(|| left.seed.cmp(&right.seed))
    });

    println!("model_version={model_version} seeds={seed_start}..{seed_end} max_steps={max_steps}");
    println!(
        "summary mean_root_cost={mean_root_cost:.3} mean_policy_cost={mean_policy_cost:.3} \
         mean_best_cost={mean_best_cost:.3} mean_greedy_cost={mean_greedy_cost:.3} \
         mean_reduction={mean_reduction:.6} mean_best_reduction={mean_best_reduction:.6} \
         mean_post_best_degradation={mean_post_best_degradation:.3} \
         best_seen_cost={:.0} best_seen_seed={} \
         ended_above_best={ended_above_best}/{count} stopped={stopped}/{count} \
         mean_steps={mean_steps:.3}",
        best_seen.0, best_seen.1,
    );
    println!(
        "seed root_cost best_cost best_step policy_cost post_best greedy_cost gap steps stopped nonreducing root_hash"
    );
    for index in 0..DEPTH_BUCKETS.len() {
        let moves = moves_by_depth[index];
        let nonreducing_rate = nonreducing_by_depth[index] as f64 / moves.max(1) as f64;
        let worsening_rate = worsening_by_depth[index] as f64 / moves.max(1) as f64;
        println!(
            "depth bucket={} moves={} nonreducing_rate={nonreducing_rate:.6} worsening_rate={worsening_rate:.6} stops={}",
            DEPTH_BUCKETS[index], moves, stops_by_depth[index],
        );
    }
    for result in results.iter().take(32) {
        println!(
            "{} {:.0} {:.0} {} {:.0} {:+.0} {:.0} {:+.0} {} {} {} {}",
            result.seed,
            result.root_cost,
            result.best_cost,
            result.best_step,
            result.policy_cost,
            result.policy_cost - result.best_cost,
            result.greedy_cost,
            result.gap,
            result.policy_steps,
            result.stopped,
            result.nonreducing_moves,
            result.root_hash,
        );
    }
    Ok(())
}

struct PolicyRollout {
    seed: u64,
    root_hash: GraphHash,
    root_cost: f32,
    best_cost: f32,
    best_step: usize,
    graph: WhittleGraphId,
    steps: usize,
    stopped: bool,
    nonreducing_moves: usize,
}

struct ScanResult {
    seed: u64,
    root_hash: GraphHash,
    root_cost: f32,
    best_cost: f32,
    best_step: usize,
    policy_cost: f32,
    greedy_cost: f32,
    gap: f32,
    policy_steps: usize,
    stopped: bool,
    nonreducing_moves: usize,
}

fn engine_greedy_cost(seed: u64, max_steps: usize) -> Result<f32, Box<dyn std::error::Error>> {
    let mut engine = whittle_engine()?;
    let root = generated_root(&mut engine, seed)?;
    let search = GreedySearch::new(GreedySearchConfig {
        max_steps,
        candidate_options: CandidateOptions {
            max_candidates: Some(MAX_CANDIDATES),
            deterministic_order: true,
        },
        measure_options: engine.measure_options(),
    });
    let episode = search.run(&mut engine, root)?;
    Ok(-episode
        .final_measure
        .scalar_reward
        .ok_or("greedy final graph was not measured")?)
}

fn generated_root(
    engine: &mut WhittleEngine,
    seed: u64,
) -> Result<WhittleGraphId, Box<dyn std::error::Error>> {
    Ok(
        WhittleGraphGenerator::from_seed(WhittleGraphGeneratorConfig::default(), seed)
            .sample_root_into(engine)?,
    )
}

fn whittle_engine() -> Result<WhittleEngine, Box<dyn std::error::Error>> {
    let generator = WhittleGraphGeneratorConfig::default();
    Ok(WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator.arity,
            capacity: generator.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    })?)
}

fn measure_cost(
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
) -> Result<f32, Box<dyn std::error::Error>> {
    Ok(-engine
        .measure(graph, engine.measure_options())?
        .scalar_reward
        .ok_or("graph was not measured")?)
}

fn argmax_first(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn depth_bucket(step: usize) -> usize {
    match step {
        0 => 0,
        1..=7 => 1,
        8..=31 => 2,
        32..=63 => 3,
        _ => 4,
    }
}

fn absolute(path: &Path) -> Result<PathBuf, std::io::Error> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn path_arg(index: usize) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    Ok(std::env::args().nth(index).map(PathBuf::from))
}

fn u64_arg(index: usize) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    std::env::args()
        .nth(index)
        .map(|arg| arg.parse::<u64>().map_err(Into::into))
        .transpose()
}

fn usize_arg(index: usize) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    std::env::args()
        .nth(index)
        .map(|arg| arg.parse::<usize>().map_err(Into::into))
        .transpose()
}
