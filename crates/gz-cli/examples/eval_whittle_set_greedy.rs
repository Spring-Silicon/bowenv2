use gz_engine::{CandidateOptions, GraphEngine, GraphHash};
use gz_engine_whittle::{
    WhittleCandidateId, WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor,
    WhittleFeatureExtractorConfig, WhittleGraphId, WhittleRoot,
};
use gz_eval_service::{EvaluatorProcess, EvaluatorProcessConfig, FeatureEvalBackend, Hello};
use gz_features::{FeatureCollator, FeatureExtractor, PositionFeatures};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

const EXPECTED_CASES: usize = 20;
const MAX_CANDIDATES: usize = 1023;
const MAX_STEPS: usize = 64;
const MAX_BATCH: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = path_arg(1)?.ok_or(
        "usage: eval_whittle_set_greedy ARTIFACT_DIR CHECKPOINT_DIR [DEVICE] [CHECKPOINT_POINTER]",
    )?;
    let checkpoint_dir = path_arg(2)?.ok_or(
        "usage: eval_whittle_set_greedy ARTIFACT_DIR CHECKPOINT_DIR [DEVICE] [CHECKPOINT_POINTER]",
    )?;
    let device = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "cuda:0".to_owned());
    let checkpoint_pointer = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "latest.json".to_owned());
    let artifact_dir = absolute(&artifact_dir)?;
    let checkpoint_dir = absolute(&checkpoint_dir)?;
    let cases = load_cases(&artifact_dir)?;
    if cases.len() != EXPECTED_CASES {
        return Err(format!(
            "expected {EXPECTED_CASES} evaluation cases, found {}",
            cases.len()
        )
        .into());
    }

    let mut rollouts = Vec::with_capacity(cases.len());
    for case in cases {
        let mut engine = WhittleEngine::new(WhittleEngineConfig {
            root: WhittleRoot::Artifact(case.bytes),
            ..WhittleEngineConfig::default()
        })?;
        let graph = engine.root();
        let root_cost = measure_cost(&mut engine, graph)?;
        if root_cost != case.expected_cost {
            return Err(format!(
                "case {} root cost mismatch: manifest={} graphzero={root_cost}",
                case.index, case.expected_cost
            )
            .into());
        }
        let root_hash = engine.hash(graph)?;
        let extractor = WhittleFeatureExtractor::with_config(
            &engine,
            WhittleFeatureExtractorConfig {
                max_actions: (MAX_CANDIDATES + 1) as u32,
                ..WhittleFeatureExtractorConfig::default()
            },
        );
        rollouts.push(PolicyRollout {
            index: case.index,
            engine,
            extractor,
            graph,
            root_hash,
            root_cost,
            steps: 0,
            stopped: false,
        });
    }

    let schema = rollouts
        .first()
        .ok_or("evaluation set is empty")?
        .extractor
        .schema()
        .clone();
    let first = rollouts.first().ok_or("evaluation set is empty")?;
    let checkpoint_dir_string = checkpoint_dir.display().to_string();
    let mut process = spawn_evaluator(&checkpoint_dir_string, &device, &checkpoint_pointer)?;
    let hello = Hello::new(
        schema.hash(),
        MAX_BATCH as u32,
        first.engine.engine_id(),
        first.engine.engine_version(),
        first.engine.action_set_hash(),
    );
    let mut backend = process.connect(&hello)?;
    let model_version = backend.model_version();
    let mut collator = FeatureCollator::new(
        schema,
        NonZeroUsize::new(MAX_BATCH).expect("MAX_BATCH is nonzero"),
    );
    let candidate_options = CandidateOptions {
        max_candidates: Some(MAX_CANDIDATES),
        deterministic_order: true,
    };
    let mut rows = Vec::with_capacity(rollouts.len());
    let mut active = Vec::with_capacity(rollouts.len());
    let mut candidate_batches = Vec::<Vec<WhittleCandidateId>>::with_capacity(rollouts.len());
    let mut batch_bytes = Vec::new();

    for step in 0..MAX_STEPS {
        rows.clear();
        active.clear();
        candidate_batches.clear();
        for (index, rollout) in rollouts.iter_mut().enumerate() {
            if rollout.stopped {
                continue;
            }
            let mut candidates = Vec::new();
            rollout
                .engine
                .candidates(rollout.graph, candidate_options, &mut candidates)?;
            let row = rollout.extractor.extract(
                &rollout.engine,
                rollout.graph,
                &candidates,
                PositionFeatures {
                    root_step: step as u32,
                    leaf_depth: 0,
                    budget_fraction: (MAX_STEPS - step) as f32 / MAX_STEPS as f32,
                    budget_step: 1.0 / MAX_STEPS as f32,
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
            if selected == candidates.len() {
                rollout.engine.release(&[], &candidates)?;
                rollout.stopped = true;
                continue;
            }
            let applied = rollout.engine.apply(rollout.graph, candidates[selected])?;
            if rollout.graph == rollout.engine.root() {
                rollout.engine.release(&[], &candidates)?;
            } else {
                rollout.engine.release(&[rollout.graph], &candidates)?;
            }
            rollout.graph = applied.after;
            rollout.steps += 1;
        }
    }

    let mut results = Vec::with_capacity(rollouts.len());
    for mut rollout in rollouts {
        let final_cost = measure_cost(&mut rollout.engine, rollout.graph)?;
        results.push(GreedyResult {
            index: rollout.index,
            root_hash: rollout.root_hash,
            root_cost: rollout.root_cost,
            final_cost,
            steps: rollout.steps,
            stopped: rollout.stopped,
        });
        if rollout.graph != rollout.engine.root() {
            rollout.engine.release(&[rollout.graph], &[])?;
        }
    }
    drop(backend);
    wait_for_process_exit(&mut process)?;
    results.sort_by_key(|result| result.index);
    print_results(&results, model_version, &checkpoint_pointer);
    Ok(())
}

struct EvalCase {
    index: usize,
    expected_cost: f32,
    bytes: Vec<u8>,
}

struct PolicyRollout {
    index: usize,
    engine: WhittleEngine,
    extractor: WhittleFeatureExtractor,
    graph: WhittleGraphId,
    root_hash: GraphHash,
    root_cost: f32,
    steps: usize,
    stopped: bool,
}

struct GreedyResult {
    index: usize,
    root_hash: GraphHash,
    root_cost: f32,
    final_cost: f32,
    steps: usize,
    stopped: bool,
}

fn load_cases(dir: &Path) -> Result<Vec<EvalCase>, Box<dyn std::error::Error>> {
    let manifest_path = dir.join("manifest.tsv");
    let manifest = std::fs::read_to_string(&manifest_path)?;
    let mut cases = Vec::new();
    for (line_number, line) in manifest.lines().enumerate() {
        if line_number == 0 && line == "index\tcost\tsource_seed\tartifact" {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(format!(
                "{}:{}: expected four tab-separated fields",
                manifest_path.display(),
                line_number + 1
            )
            .into());
        }
        let index = fields[0].parse::<usize>()?;
        if index != cases.len() {
            return Err(format!(
                "{}:{}: expected index {}, found {index}",
                manifest_path.display(),
                line_number + 1,
                cases.len()
            )
            .into());
        }
        let expected_cost = fields[1].parse::<f32>()?;
        let _source_seed = fields[2].parse::<u64>()?;
        let bytes = std::fs::read(dir.join(fields[3]))?;
        if !bytes.starts_with(b"WAV1") {
            return Err(format!("case {index} is not a WAV1 artifact").into());
        }
        cases.push(EvalCase {
            index,
            expected_cost,
            bytes,
        });
    }
    Ok(cases)
}

fn spawn_evaluator(
    checkpoint_dir: &str,
    device: &str,
    pointer: &str,
) -> Result<EvaluatorProcess, Box<dyn std::error::Error>> {
    let socket_path = std::env::temp_dir().join(format!(
        "gz-whittle-eval-greedy-{}.sock",
        std::process::id()
    ));
    Ok(EvaluatorProcess::spawn(EvaluatorProcessConfig {
        working_dir: PathBuf::from("python"),
        socket_path,
        ready_timeout: Duration::from_secs(30),
        io_timeout: Duration::from_secs(300),
        extra_args: vec![
            "--backend".to_owned(),
            "torch".to_owned(),
            "--checkpoint-dir".to_owned(),
            checkpoint_dir.to_owned(),
            "--checkpoint-pointer".to_owned(),
            pointer.to_owned(),
            "--device".to_owned(),
            device.to_owned(),
            "--max-batch".to_owned(),
            MAX_BATCH.to_string(),
            "--poll-interval".to_owned(),
            "0".to_owned(),
        ],
        ..EvaluatorProcessConfig::default()
    })?)
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait()? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("Python evaluator exited with {status}").into()),
            None if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            None => return Err("Python evaluator did not exit".into()),
        }
    }
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

fn print_results(
    results: &[GreedyResult],
    model_version: gz_engine::ModelVersion,
    checkpoint_pointer: &str,
) {
    let count = results.len() as f32;
    let mean_root = results.iter().map(|row| row.root_cost).sum::<f32>() / count;
    let mean_final = results.iter().map(|row| row.final_cost).sum::<f32>() / count;
    let mean_percent = 100.0
        * results
            .iter()
            .map(|row| (row.root_cost - row.final_cost) / row.root_cost)
            .sum::<f32>()
        / count;
    let aggregate_percent = 100.0 * (mean_root - mean_final) / mean_root;
    let mean_steps = results.iter().map(|row| row.steps).sum::<usize>() as f32 / count;
    let stopped = results.iter().filter(|row| row.stopped).count();
    println!(
        "settings cases={} mode=direct-greedy max_steps={MAX_STEPS} max_candidates={MAX_CANDIDATES} stop_allowed=true position_features=true",
        results.len()
    );
    println!("model checkpoint_pointer={checkpoint_pointer} model_version={model_version}");
    println!(
        "summary mean_root={mean_root:.3} mean_final={mean_final:.3} mean_reduction={:.3} mean_percent_reduction={mean_percent:.6} aggregate_percent_reduction={aggregate_percent:.6} stopped={stopped}/{} mean_steps={mean_steps:.3}",
        mean_root - mean_final,
        results.len(),
    );
    println!("index root final reduction percent_reduction steps stopped root_hash");
    for result in results {
        println!(
            "{} {:.0} {:.0} {:+.0} {:.6} {} {} {}",
            result.index,
            result.root_cost,
            result.final_cost,
            result.root_cost - result.final_cost,
            100.0 * (result.root_cost - result.final_cost) / result.root_cost,
            result.steps,
            result.stopped,
            result.root_hash,
        );
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
