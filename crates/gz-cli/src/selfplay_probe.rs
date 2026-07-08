//! Probe-batch emission for checkpoint interpretability: reproduces the
//! production fixed root, measures every root candidate's after-cost as
//! ground truth, and writes feature batches (value-vs-cost sweep,
//! opponent variants, orientation swaps) plus meta.json for the Python
//! probe script. Mirrors the selfplay pipeline's engine, generator, and
//! extractor configuration so rows are in-distribution for checkpoints
//! trained at max_candidates 1023.
use gz_engine::{CandidateOptions, GraphEngine};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleFeatureExtractorConfig,
    WhittleGraphGenerator, WhittleGraphGeneratorConfig, WhittleGraphId, WhittleRoot,
};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, OpponentStateFeatures, PositionFeatures,
};
use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::path::PathBuf;

const MAX_CANDIDATES: usize = 1023;
const OPPONENT_REWARD_SCALE: f32 = 256.0;
const SWEEP_STATES: usize = 32;

pub struct ProbeArgs {
    pub out_dir: PathBuf,
    pub seed: u64,
}

struct Candidate {
    index: usize,
    rule_name: &'static str,
    after: Option<(WhittleGraphId, f32)>,
}

pub fn run_probe(args: ProbeArgs) -> Result<(), String> {
    let generator_config = WhittleGraphGeneratorConfig::default();
    let mut engine = WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator_config.arity,
            capacity: generator_config.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let mut generator = WhittleGraphGenerator::from_seed(generator_config, args.seed);
    let root = generator
        .sample_into(&mut engine)
        .map_err(|error| error.to_string())?
        .graph;
    let root_cost = measure_cost(&mut engine, root)?;

    let mut extractor = WhittleFeatureExtractor::with_config(
        &engine,
        WhittleFeatureExtractorConfig {
            max_actions: MAX_CANDIDATES as u32 + 1,
            ..WhittleFeatureExtractorConfig::default()
        },
    );
    let options = CandidateOptions {
        max_candidates: Some(MAX_CANDIDATES),
        deterministic_order: true,
    };

    let mut ids = Vec::new();
    engine
        .candidates(root, options, &mut ids)
        .map_err(|error| error.to_string())?;

    // Apply and measure every candidate: the ground truth the probes
    // correlate priors and values against.
    let mut candidates = Vec::with_capacity(ids.len());
    for (index, id) in ids.iter().copied().enumerate() {
        let info = engine
            .candidate_info(root, id)
            .map_err(|error| error.to_string())?;
        let rule_name = gz_engine_whittle::rule_name(info.kind.get() as u16);
        let applied = engine.apply(root, id).map_err(|error| error.to_string())?;
        let after = if applied.rejected.is_some() {
            None
        } else {
            Some((applied.after, measure_cost(&mut engine, applied.after)?))
        };
        candidates.push(Candidate {
            index,
            rule_name,
            after,
        });
    }

    let best = applied_extreme(&candidates, |a, b| a < b).ok_or("no applied candidates")?;
    let worst = applied_extreme(&candidates, |a, b| a > b).ok_or("no applied candidates")?;

    // Opponent variants share the probe's reference convention: the
    // opponent state row is the opponent graph at its own step 0.
    let root_opp = opponent_features(&mut extractor, &mut engine, root, root_cost, options)?;
    let best_opp = opponent_features(&mut extractor, &mut engine, best.0, best.1, options)?;
    let worst_opp = opponent_features(&mut extractor, &mut engine, worst.0, worst.1, options)?;

    let mut meta = String::from("{\n");
    let _ = writeln!(meta, "  \"seed\": {},", args.seed);
    let _ = writeln!(meta, "  \"root_cost\": {root_cost},");
    let _ = writeln!(meta, "  \"candidates\": [");
    for (position, candidate) in candidates.iter().enumerate() {
        let delta = candidate.after.map_or("null".to_string(), |(_, cost)| {
            format!("{}", cost - root_cost)
        });
        let comma = if position + 1 == candidates.len() {
            ""
        } else {
            ","
        };
        let _ = writeln!(
            meta,
            "    {{\"index\": {}, \"rule\": \"{}\", \"delta\": {}}}{}",
            candidate.index, candidate.rule_name, delta, comma
        );
    }
    let _ = writeln!(meta, "  ],");

    // Sweep: the root plus applied states spanning the delta spectrum,
    // every row paired against the fixed root-state opponent.
    let mut applied: Vec<(usize, WhittleGraphId, f32)> = candidates
        .iter()
        .filter_map(|candidate| {
            candidate
                .after
                .map(|(graph, cost)| (candidate.index, graph, cost))
        })
        .collect();
    applied.sort_by(|left, right| left.2.total_cmp(&right.2));
    let stride = (applied.len() / SWEEP_STATES).max(1);
    let sweep: Vec<(usize, WhittleGraphId, f32)> =
        applied.iter().copied().step_by(stride).collect();

    let mut sweep_rows = vec![probe_row(
        &mut extractor,
        &mut engine,
        root,
        options,
        &root_opp,
        root_cost,
    )?];
    let _ = writeln!(meta, "  \"sweep\": [");
    let _ = writeln!(
        meta,
        "    {{\"row\": 0, \"cost\": {root_cost}, \"index\": null}},"
    );
    for (position, (index, graph, cost)) in sweep.iter().enumerate() {
        sweep_rows.push(probe_row(
            &mut extractor,
            &mut engine,
            *graph,
            options,
            &root_opp,
            root_cost,
        )?);
        let comma = if position + 1 == sweep.len() { "" } else { "," };
        let _ = writeln!(
            meta,
            "    {{\"row\": {}, \"cost\": {}, \"index\": {}}}{}",
            position + 1,
            cost,
            index,
            comma
        );
    }
    let _ = writeln!(meta, "  ],");

    // Opponent variants: the same root state against no opponent, a
    // worse opponent, itself, and the best opponent. The pair head
    // should rank them monotonically.
    let mut opp_rows = vec![probe_row_no_opponent(
        &mut extractor,
        &mut engine,
        root,
        options,
    )?];
    for opp in [&worst_opp, &root_opp, &best_opp] {
        opp_rows.push(probe_row(
            &mut extractor,
            &mut engine,
            root,
            options,
            opp,
            opp.cost,
        )?);
    }
    let _ = writeln!(
        meta,
        "  \"opponents\": [{{\"row\": 0, \"cost\": null}}, {{\"row\": 1, \"cost\": {}}}, {{\"row\": 2, \"cost\": {}}}, {{\"row\": 3, \"cost\": {}}}],",
        worst_opp.cost, root_cost, best_opp.cost
    );

    // Orientation: (best vs root) and (root vs best); an antisymmetric
    // pair head puts them on opposite sides of zero.
    let orient_rows = vec![
        probe_row(
            &mut extractor,
            &mut engine,
            best.0,
            options,
            &root_opp,
            root_cost,
        )?,
        probe_row(
            &mut extractor,
            &mut engine,
            root,
            options,
            &best_opp,
            best.1,
        )?,
    ];
    let _ = writeln!(
        meta,
        "  \"orientation\": {{\"best_cost\": {}, \"root_cost\": {root_cost}}}",
        best.1
    );
    let _ = writeln!(meta, "}}");

    write_batch(&extractor, &args.out_dir.join("sweep.gzfb"), &sweep_rows)?;
    write_batch(&extractor, &args.out_dir.join("opponents.gzfb"), &opp_rows)?;
    write_batch(
        &extractor,
        &args.out_dir.join("orientation.gzfb"),
        &orient_rows,
    )?;
    std::fs::write(args.out_dir.join("meta.json"), meta).map_err(|error| error.to_string())?;
    println!(
        "root_cost={root_cost} candidates={} applied={} sweep_rows={}",
        candidates.len(),
        applied.len(),
        sweep_rows.len()
    );
    Ok(())
}

struct OpponentRow {
    features: OpponentStateFeatures,
    cost: f32,
}

fn measure_cost(engine: &mut WhittleEngine, graph: WhittleGraphId) -> Result<f32, String> {
    let options = engine.measure_options();
    let measured = engine
        .measure(graph, options)
        .map_err(|error| error.to_string())?;
    let reward = measured.scalar_reward.ok_or("unmeasured graph")?;
    Ok(-reward)
}

fn applied_extreme(
    candidates: &[Candidate],
    better: fn(f32, f32) -> bool,
) -> Option<(WhittleGraphId, f32)> {
    let mut extreme: Option<(WhittleGraphId, f32)> = None;
    for candidate in candidates {
        let Some((graph, cost)) = candidate.after else {
            continue;
        };
        if extreme.is_none_or(|(_, current)| better(cost, current)) {
            extreme = Some((graph, cost));
        }
    }
    extreme
}

fn opponent_features(
    extractor: &mut WhittleFeatureExtractor,
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
    cost: f32,
    options: CandidateOptions,
) -> Result<OpponentRow, String> {
    let row = extract_row(extractor, engine, graph, options, opponent_position(cost))?;
    Ok(OpponentRow {
        features: OpponentStateFeatures {
            node_count: row.node_count,
            node_tokens: row.node_tokens,
            node_attrs: row.node_attrs,
            edges: row.edges,
            position: row.position,
        },
        cost,
    })
}

fn opponent_position(cost: f32) -> PositionFeatures {
    PositionFeatures {
        root_step: 0,
        leaf_depth: 0,
        budget_fraction: 1.0,
        budget_step: 1.0 / 128.0,
        opponent_reward: -cost / OPPONENT_REWARD_SCALE,
        opponent_present: true,
    }
}

fn probe_row(
    extractor: &mut WhittleFeatureExtractor,
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
    options: CandidateOptions,
    opponent: &OpponentRow,
    opponent_cost: f32,
) -> Result<FeatureRow, String> {
    let mut row = extract_row(
        extractor,
        engine,
        graph,
        options,
        opponent_position(opponent_cost),
    )?;
    row.opponent = Some(opponent.features.clone());
    Ok(row)
}

fn probe_row_no_opponent(
    extractor: &mut WhittleFeatureExtractor,
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
    options: CandidateOptions,
) -> Result<FeatureRow, String> {
    extract_row(
        extractor,
        engine,
        graph,
        options,
        PositionFeatures {
            root_step: 0,
            leaf_depth: 0,
            budget_fraction: 1.0,
            budget_step: 1.0 / 128.0,
            opponent_reward: 0.0,
            opponent_present: false,
        },
    )
}

fn extract_row(
    extractor: &mut WhittleFeatureExtractor,
    engine: &mut WhittleEngine,
    graph: WhittleGraphId,
    options: CandidateOptions,
    position: PositionFeatures,
) -> Result<FeatureRow, String> {
    let mut candidates = Vec::new();
    engine
        .candidates(graph, options, &mut candidates)
        .map_err(|error| error.to_string())?;
    extractor
        .extract(engine, graph, &candidates, position)
        .map_err(|error| format!("{error:?}"))
}

fn write_batch(
    extractor: &WhittleFeatureExtractor,
    path: &PathBuf,
    rows: &[FeatureRow],
) -> Result<(), String> {
    let capacity = NonZeroUsize::new(rows.len()).ok_or("empty batch")?;
    let mut collator = FeatureCollator::new(extractor.schema().clone(), capacity);
    let mut bytes = Vec::new();
    collator
        .collate_into(rows, &mut bytes)
        .map_err(|error| format!("{error:?}"))?;
    std::fs::write(path, &bytes).map_err(|error| error.to_string())
}
