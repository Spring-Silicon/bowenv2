//! Reusable contract checks for engine adapters.

use crate::{
    BatchGraphEngine, CandidateHash, CandidateOptions, EngineError, GraphEngine, MeasureOptions,
};
use std::fmt;

pub trait EngineContractFixture {
    type Engine: BatchGraphEngine;

    fn make_engine(&self) -> Self::Engine;
    fn measure_options(&self) -> MeasureOptions;
    fn known_path(&self) -> Vec<<Self::Engine as GraphEngine>::Candidate>;
    fn unknown_graph(&self) -> Option<<Self::Engine as GraphEngine>::Graph>;
    fn unknown_candidate(&self) -> Option<<Self::Engine as GraphEngine>::Candidate>;
}

pub fn run_engine_contract<F: EngineContractFixture>(fixture: &F) -> Result<(), ContractError> {
    check_smoke(fixture)?;
    check_determinism(fixture)?;
    check_batch_equivalence(fixture)?;
    check_negative_cases(fixture)
}

fn check_smoke<F: EngineContractFixture>(fixture: &F) -> Result<(), ContractError> {
    let mut engine = fixture.make_engine();
    let _engine_id = engine.engine_id();
    let _engine_version = engine.engine_version();
    let action_set_hash = engine.action_set_hash();
    let root = engine.root();
    let root_hash = engine.hash(root)?;
    let mut candidates = Vec::new();

    engine.candidates(root, CandidateOptions::default(), &mut candidates)?;

    for candidate in candidates.iter().copied() {
        let info = engine.candidate_info(root, candidate)?.validate()?;
        ensure(
            info.graph_hash == root_hash,
            "candidate_info graph_hash must match hash(root)",
        )?;
        ensure(
            info.action_set_hash == action_set_hash,
            "candidate_info action_set_hash must match engine action_set_hash",
        )?;

        let applied = engine.apply(root, candidate)?;
        ensure(
            applied.before_hash == root_hash,
            "apply before_hash must match hash(root)",
        )?;
        ensure(
            applied.candidate_hash == info.candidate_hash,
            "apply candidate_hash must match candidate_info",
        )?;
        engine.candidates(applied.after, CandidateOptions::default(), &mut Vec::new())?;
    }

    let artifact = engine.export_graph(root)?;
    ensure(
        artifact.graph_hash == root_hash,
        "export_graph graph_hash must match hash(root)",
    )?;

    let measured = engine
        .measure(root, fixture.measure_options())?
        .validate()?;
    ensure(
        measured.graph_hash == root_hash,
        "measure graph_hash must match hash(root)",
    )?;

    let mut graph = root;
    for candidate in fixture.known_path() {
        let result = engine.apply(graph, candidate)?;
        graph = result.after;
    }
    let _ = engine
        .measure(graph, fixture.measure_options())?
        .validate()?;

    Ok(())
}

fn check_determinism<F: EngineContractFixture>(fixture: &F) -> Result<(), ContractError> {
    let mut left = fixture.make_engine();
    let mut right = fixture.make_engine();
    let left_root = left.root();
    let right_root = right.root();
    let left_hash = left.hash(left_root)?;
    let right_hash = right.hash(right_root)?;

    ensure(
        left_hash == right_hash,
        "root hash must match across engine instances",
    )?;

    let default_options = CandidateOptions::default();
    let limited_options = CandidateOptions {
        max_candidates: Some(1),
        deterministic_order: true,
    };
    let action_set_hash = left.action_set_hash();
    let mut scratch = Vec::new();
    left.candidates(left_root, default_options, &mut scratch)?;
    let first_left_hashes = candidate_hashes_from(&left, left_root, &scratch)?;
    left.candidates(left_root, limited_options, &mut scratch)?;
    ensure(
        scratch.len() <= 1,
        "candidates must clear the output buffer before writing",
    )?;
    ensure(
        left.action_set_hash() == action_set_hash,
        "action_set_hash must be stable across CandidateOptions changes",
    )?;

    let left_candidates = candidate_hashes(&mut left, left_root, default_options)?;
    ensure(
        left_candidates == first_left_hashes,
        "candidate hashes must preserve order across repeated enumeration",
    )?;
    let right_candidates = candidate_hashes(&mut right, right_root, default_options)?;
    ensure(
        left_candidates == right_candidates,
        "candidate hashes must match across engine instances",
    )?;

    let mut limited_first = fixture.make_engine();
    let limited_root = limited_first.root();
    limited_first.candidates(limited_root, limited_options, &mut Vec::new())?;
    let after_limited = candidate_hashes(&mut limited_first, limited_root, default_options)?;
    ensure(
        after_limited == right_candidates,
        "a limited request must not truncate later candidate enumeration",
    )?;

    if let Some(candidate) = fixture.known_path().first().copied() {
        let left_after = left.apply(left_root, candidate)?.after_hash;
        let right_after = right.apply(right_root, candidate)?.after_hash;
        ensure(
            left_after == right_after,
            "known path first apply hash must match across engine instances",
        )?;
    }

    Ok(())
}

fn check_batch_equivalence<F: EngineContractFixture>(fixture: &F) -> Result<(), ContractError> {
    let mut engine = fixture.make_engine();
    let root = engine.root();
    let options = CandidateOptions::default();
    let mut single_candidates = Vec::new();
    engine.candidates(root, options, &mut single_candidates)?;

    let single_hashes = candidate_hashes_from(&engine, root, &single_candidates)?;
    let batch_candidates = engine.candidates_batch(&[root], options);
    ensure(
        batch_candidates.len() == 1,
        "candidates_batch length mismatch",
    )?;
    let batch_candidates = batch_candidates.into_iter().next().unwrap()?;
    let batch_hashes = candidate_hashes_from(&engine, root, &batch_candidates)?;
    ensure(
        batch_hashes == single_hashes,
        "candidates_batch candidate hashes must equal single candidates call",
    )?;

    if let Some(candidate) = single_candidates.first().copied() {
        let single_apply = engine.apply(root, candidate)?;
        let batch_apply = engine.apply_batch(&[crate::ApplyJob::new(root, candidate)]);
        ensure(batch_apply.len() == 1, "apply_batch length mismatch")?;
        ensure(
            batch_apply.into_iter().next().unwrap()?.after_hash == single_apply.after_hash,
            "apply_batch after_hash must equal single apply call",
        )?;
    }

    let single_measure = engine.measure(root, fixture.measure_options())?;
    let batch_measure = engine.measure_batch(&[root], fixture.measure_options());
    ensure(batch_measure.len() == 1, "measure_batch length mismatch")?;
    ensure(
        batch_measure.into_iter().next().unwrap()?.graph_hash == single_measure.graph_hash,
        "measure_batch graph_hash must equal single measure call",
    )?;

    Ok(())
}

fn check_negative_cases<F: EngineContractFixture>(fixture: &F) -> Result<(), ContractError> {
    if let Some(graph) = fixture.unknown_graph() {
        let engine = fixture.make_engine();
        ensure(
            matches!(engine.hash(graph), Err(EngineError::UnknownGraph { .. })),
            "unknown graph must return EngineError::UnknownGraph",
        )?;
    }

    if let Some(candidate) = fixture.unknown_candidate() {
        let engine = fixture.make_engine();
        let root = engine.root();
        ensure(
            matches!(
                engine.candidate_info(root, candidate),
                Err(EngineError::UnknownCandidate { .. })
            ),
            "unknown candidate must return EngineError::UnknownCandidate",
        )?;
    }

    Ok(())
}

fn candidate_hashes<E: GraphEngine>(
    engine: &mut E,
    graph: E::Graph,
    options: CandidateOptions,
) -> Result<Vec<CandidateHash>, ContractError> {
    let mut candidates = Vec::new();
    engine.candidates(graph, options, &mut candidates)?;

    candidate_hashes_from(engine, graph, &candidates)
}

fn candidate_hashes_from<E: GraphEngine>(
    engine: &E,
    graph: E::Graph,
    candidates: &[E::Candidate],
) -> Result<Vec<CandidateHash>, ContractError> {
    candidates
        .iter()
        .copied()
        .map(|candidate| {
            Ok(engine
                .candidate_info(graph, candidate)?
                .validate()?
                .candidate_hash)
        })
        .collect()
}

fn ensure(condition: bool, message: &'static str) -> Result<(), ContractError> {
    if condition {
        Ok(())
    } else {
        Err(ContractError::AssertionFailed(message))
    }
}

#[derive(Debug)]
pub enum ContractError {
    Engine(EngineError),
    CandidateInfo(crate::CandidateInfoError),
    Measurement(crate::MeasurementValidationError),
    AssertionFailed(&'static str),
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(error) => write!(f, "{error}"),
            Self::CandidateInfo(error) => write!(f, "{error}"),
            Self::Measurement(error) => write!(f, "{error}"),
            Self::AssertionFailed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ContractError {}

impl From<EngineError> for ContractError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

impl From<crate::CandidateInfoError> for ContractError {
    fn from(error: crate::CandidateInfoError) -> Self {
        Self::CandidateInfo(error)
    }
}

impl From<crate::MeasurementValidationError> for ContractError {
    fn from(error: crate::MeasurementValidationError) -> Self {
        Self::Measurement(error)
    }
}
