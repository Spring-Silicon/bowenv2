use gz_engine::{
    ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash, PortableCandidateRef,
    PortableGraphId, PortableSearchActionRef, ReplayGraphContext, SearchStepRef,
    SearchStepRefError,
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

fn candidate_action(context: ReplayGraphContext, byte: u8) -> PortableSearchActionRef {
    PortableSearchActionRef::candidate(candidate(context, byte))
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
fn action_ref_exposes_context() {
    let context = context(1);
    let candidate = candidate_action(context, 4);
    let stop = PortableSearchActionRef::stop(context);

    assert_eq!(candidate.context(), context);
    assert_eq!(stop.context(), context);
}

#[test]
fn search_step_rejects_action_context_mismatch() {
    let before = context(1);
    let after = context(2);
    let wrong_context = context(3);
    let action = candidate_action(wrong_context, 4);

    assert_eq!(
        SearchStepRef::new(before, action, after).unwrap_err(),
        SearchStepRefError::ActionContextMismatch {
            before: Box::new(before),
            action_context: Box::new(wrong_context),
        }
    );
}

#[test]
fn search_step_accepts_matching_candidate_context() {
    let before = context(1);
    let after = context(2);
    let action = candidate_action(before, 4);
    let step = SearchStepRef::new(before, action, after).unwrap();

    assert_eq!(step.before, before);
    assert_eq!(step.action, action);
    assert_eq!(step.after, after);
}

#[test]
fn search_step_rejects_stop_after_mismatch() {
    let before = context(1);
    let after = context(2);
    let action = PortableSearchActionRef::stop(before);

    assert_eq!(
        SearchStepRef::new(before, action, after).unwrap_err(),
        SearchStepRefError::StopAfterMismatch {
            before: Box::new(before),
            after: Box::new(after),
        }
    );
}

#[test]
fn search_step_accepts_stop_at_same_context() {
    let before = context(1);
    let action = PortableSearchActionRef::stop(before);
    let step = SearchStepRef::new(before, action, before).unwrap();

    assert_eq!(step.before, before);
    assert_eq!(step.action, action);
    assert_eq!(step.after, before);
}
