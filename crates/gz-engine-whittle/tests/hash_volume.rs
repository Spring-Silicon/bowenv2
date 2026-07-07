use gz_engine::{CandidateOptions, GraphEngine};
use gz_engine_whittle::{HashVolumeCounters, WhittleEngine, WhittleEngineConfig, WhittleRoot};

const NO_NODE: u32 = u32::MAX;

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

fn engine() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(and_idempotent_artifact()),
        ..WhittleEngineConfig::default()
    })
    .unwrap()
}

#[test]
fn hash_volume_counters_track_insert_hash_and_dedup_counts() {
    let mut engine = engine();
    assert_eq!(
        engine.hash_volume_counters(),
        HashVolumeCounters {
            graph_inserts: 1,
            dedup_hits: 0,
            portable_hashes: 1,
        }
    );

    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let applied = engine.apply(root, candidates[0]).unwrap();
    let counters = engine.hash_volume_counters();

    assert_eq!(counters.graph_inserts, 2);
    assert_eq!(counters.portable_hashes, 2);
    assert_eq!(counters.dedup_hits, u64::from(!applied.changed));

    engine.release(&[applied.after], &candidates).unwrap();
}
