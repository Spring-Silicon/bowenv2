use gz_engine::{
    ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash, PortableCandidateRef,
    PortableGraphId, ReplayGraphContext, SearchStepRef, SearchStepRefError,
};
use std::collections::{BTreeSet, HashSet};

fn graph(byte: u8) -> PortableGraphId {
    PortableGraphId::new(
        GraphHash::from_bytes([byte; 32]),
        EngineId::from_bytes([1; 16]),
        EngineVersion::from_bytes([2; 16]),
    )
}

fn context(byte: u8) -> ReplayGraphContext {
    ReplayGraphContext::new(graph(byte), ActionSetHash::from_bytes([3; 32]))
}

fn candidate(context: ReplayGraphContext, byte: u8) -> PortableCandidateRef {
    PortableCandidateRef::new(context, CandidateHash::from_bytes([byte; 32]))
}

#[test]
fn portable_graph_id_is_stable_map_key() {
    let id = graph(9);

    let mut btree = BTreeSet::new();
    btree.insert(id);
    assert!(btree.contains(&id));

    let mut hash_set = HashSet::new();
    hash_set.insert(id);
    assert!(hash_set.contains(&id));
}

#[test]
fn candidate_ref_includes_graph_action_context() {
    let context = context(1);
    let candidate = candidate(context, 4);

    assert_eq!(candidate.context, context);
    assert_eq!(candidate.candidate_hash, CandidateHash::from_bytes([4; 32]));
}

#[test]
fn search_step_rejects_candidate_context_mismatch() {
    let before = context(1);
    let after = context(2);
    let wrong_context = context(3);
    let candidate = candidate(wrong_context, 4);

    assert_eq!(
        SearchStepRef::new(before, candidate, after).unwrap_err(),
        SearchStepRefError::CandidateContextMismatch {
            before: Box::new(before),
            candidate_context: Box::new(wrong_context),
        }
    );
}

#[test]
fn search_step_accepts_matching_candidate_context() {
    let before = context(1);
    let after = context(2);
    let candidate = candidate(before, 4);
    let step = SearchStepRef::new(before, candidate, after).unwrap();

    assert_eq!(step.before, before);
    assert_eq!(step.candidate, candidate);
    assert_eq!(step.after, after);
}
