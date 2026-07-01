use gz_engine::{CandidateOptions, EngineError, GraphArtifactFormat, GraphEngine, MeasureOptions};
use gz_engine_whittle::{
    RuleId, WhittleEngine, WhittleEngineConfig, WhittleGeneratorConfigError, WhittleGraphGenerator,
    WhittleGraphGeneratorConfig, WhittleRoot,
};

const NO_NODE: u32 = u32::MAX;

fn input_artifact() -> Vec<u8> {
    wav1(1, 16, 1, &[(0, 0, NO_NODE), (5, 0, NO_NODE)])
}

fn and_idempotent_artifact() -> Vec<u8> {
    wav1(1, 16, 2, &[(0, 0, NO_NODE), (2, 0, 0), (5, 1, NO_NODE)])
}

fn wav1(arity: u16, capacity: u16, output_node: u32, nodes: &[(i8, u32, u32)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"WAV1");
    bytes.extend_from_slice(&arity.to_le_bytes());
    bytes.extend_from_slice(&capacity.to_le_bytes());
    bytes.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&output_node.to_le_bytes());

    for (op, arg0, arg1) in nodes {
        bytes.push(*op as u8);
        bytes.extend_from_slice(&arg0.to_le_bytes());
        bytes.extend_from_slice(&arg1.to_le_bytes());
    }

    bytes
}

fn and_engine() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(and_idempotent_artifact()),
        ..WhittleEngineConfig::default()
    })
    .unwrap()
}

#[test]
fn input_graph_exports_legacy_wav1_artifact() {
    let engine = WhittleEngine::default();
    let artifact = engine.export_graph(engine.root()).unwrap();

    assert_eq!(artifact.format, GraphArtifactFormat::Binary);
    assert_eq!(artifact.bytes, input_artifact());
}

#[test]
fn and_idempotent_is_first_candidate_and_reduces_cost() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();

    let first = candidates[0];
    let info = engine.candidate_info(root, first).unwrap();
    assert_eq!(
        info.kind,
        gz_engine::CandidateKindId::new(RuleId::AndIdempotent as u32)
    );
    assert_eq!(info.graph_hash, engine.hash(root).unwrap());
    assert_eq!(info.action_set_hash, engine.action_set_hash());

    let before = engine.measure(root, engine.measure_options()).unwrap();
    let applied = engine.apply(root, first).unwrap();
    let after = engine
        .measure(applied.after, engine.measure_options())
        .unwrap();

    assert_eq!(before.scalar_reward, Some(-3.0));
    assert_eq!(after.scalar_reward, Some(-2.0));
    assert_ne!(applied.before_hash, applied.after_hash);
}

#[test]
fn export_graph_roundtrips_through_artifact_root() {
    let engine = and_engine();
    let artifact = engine.export_graph(engine.root()).unwrap();
    let imported = WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(artifact.bytes.clone()),
        ..WhittleEngineConfig::default()
    })
    .unwrap();
    let imported_artifact = imported.export_graph(imported.root()).unwrap();

    assert_eq!(imported_artifact.bytes, artifact.bytes);
    assert_eq!(
        imported.hash(imported.root()).unwrap(),
        engine.hash(engine.root()).unwrap()
    );
}

#[test]
fn reverse_constant_folding_changes_action_set_hash() {
    let default = WhittleEngine::default();
    let reverse_constants = WhittleEngine::new(WhittleEngineConfig {
        include_reverse_constant_folding: true,
        ..WhittleEngineConfig::default()
    })
    .unwrap();

    assert_ne!(
        default.action_set_hash(),
        reverse_constants.action_set_hash()
    );
}

#[test]
fn measure_returns_negative_cost_and_compact_metadata() {
    let mut engine = WhittleEngine::default();
    let result = engine
        .measure(
            engine.root(),
            MeasureOptions::new(engine.measure_config_hash(), 3, Some(100), false).unwrap(),
        )
        .unwrap();

    assert_eq!(result.scalar_reward, Some(-2.0));
    assert_eq!(result.latency, None);
    assert_eq!(result.metadata.bytes, vec![1, 2, 0, 0, 0, 1, 0, 16, 0]);
}

#[test]
fn bad_artifact_returns_engine_error() {
    let result = WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(b"bad".to_vec()),
        ..WhittleEngineConfig::default()
    });

    assert!(matches!(result, Err(EngineError::Internal { .. })));
}

#[test]
fn generator_config_rejects_invalid_ranges() {
    assert_eq!(
        WhittleGraphGeneratorConfig {
            arity: 0,
            ..WhittleGraphGeneratorConfig::default()
        }
        .validate()
        .unwrap_err(),
        WhittleGeneratorConfigError::ZeroArity
    );

    assert_eq!(
        WhittleGraphGeneratorConfig {
            exception_terms_min: 3,
            exception_terms_max: 2,
            ..WhittleGraphGeneratorConfig::default()
        }
        .validate()
        .unwrap_err(),
        WhittleGeneratorConfigError::InvalidExceptionTermRange
    );
}

#[test]
fn generator_is_deterministic_for_fixed_seed() {
    let config = WhittleGraphGeneratorConfig {
        arity: 3,
        capacity: 96,
        exception_terms_min: 2,
        exception_terms_max: 3,
        prewalk_steps_min: 2,
        prewalk_steps_max: 4,
    };

    let mut left_engine = WhittleEngine::default();
    let mut right_engine = WhittleEngine::default();
    let mut left = WhittleGraphGenerator::from_seed(config.clone(), 7);
    let mut right = WhittleGraphGenerator::from_seed(config, 7);
    let left_graph = left.sample_into(&mut left_engine).unwrap();
    let right_graph = right.sample_into(&mut right_engine).unwrap();

    assert_eq!(
        left_engine.export_graph(left_graph.graph).unwrap().bytes,
        right_engine.export_graph(right_graph.graph).unwrap().bytes
    );
    assert!(left_graph.prewalk_steps_applied <= left_graph.prewalk_steps_requested);
}
