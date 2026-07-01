use gz_engine::{
    CandidateOptions, GraphEngine, PortableCandidateRef, PortableGraphId, ReplayGraphContext,
};
use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig, WhittleRoot};
use gz_eval::{EngineEvalRequest, EngineEvaluator, EvalAction, EvalRequest};
use gz_eval_whittle::WhittleMeasureEvaluator;

const NO_NODE: u32 = u32::MAX;

#[test]
fn measure_evaluator_values_current_graph_and_scores_candidates_in_order() {
    let mut engine = and_engine();
    let graph = engine.root();
    let measure_options = engine.measure_options();
    let mut candidates = Vec::new();
    engine
        .candidates(graph, CandidateOptions::default(), &mut candidates)
        .unwrap();

    let request = eval_request(&engine, graph, &candidates);
    let mut evaluator = WhittleMeasureEvaluator::new();
    let output = evaluator
        .evaluate(
            &mut engine,
            EngineEvalRequest {
                graph,
                candidates: &candidates,
                request: &request,
                measure_options,
            },
        )
        .unwrap();

    output.validate_for(&request).unwrap();
    assert_eq!(output.model_version, evaluator.model_version());
    assert_eq!(output.value, -3.0);
    assert_eq!(output.policy_logits.last(), Some(&0.5));

    let expected = expected_candidate_logits(&mut engine, graph, &candidates);
    assert_eq!(
        &output.policy_logits[..candidates.len()],
        expected.as_slice()
    );
    assert_eq!(output.policy_logits[0], 1.0);
}

fn expected_candidate_logits(
    engine: &mut WhittleEngine,
    graph: gz_engine_whittle::WhittleGraphId,
    candidates: &[gz_engine_whittle::WhittleCandidateId],
) -> Vec<f32> {
    let options = engine.measure_options();
    let before = engine
        .measure(graph, options)
        .unwrap()
        .scalar_reward
        .unwrap();

    candidates
        .iter()
        .copied()
        .map(|candidate| {
            let after = engine
                .apply(graph, candidate)
                .and_then(|applied| engine.measure(applied.after, options))
                .unwrap()
                .scalar_reward
                .unwrap();

            if after > before {
                1.0
            } else if after < before {
                0.0
            } else {
                0.5
            }
        })
        .collect()
}

fn eval_request(
    engine: &WhittleEngine,
    graph: gz_engine_whittle::WhittleGraphId,
    candidates: &[gz_engine_whittle::WhittleCandidateId],
) -> EvalRequest {
    let context = graph_context(engine, graph);
    let mut actions = Vec::with_capacity(candidates.len() + 1);

    for candidate in candidates {
        let info = engine.candidate_info(graph, *candidate).unwrap();
        let candidate_ref = PortableCandidateRef::new(context, info.candidate_hash);
        actions.push(EvalAction::candidate(
            candidate_ref,
            info.kind,
            info.tags,
            info.static_prior,
        ));
    }

    actions.push(EvalAction::stop(context));
    EvalRequest::new(context, actions).unwrap()
}

fn graph_context(
    engine: &WhittleEngine,
    graph: gz_engine_whittle::WhittleGraphId,
) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(
            engine.hash(graph).unwrap(),
            engine.engine_id(),
            engine.engine_version(),
        ),
        engine.action_set_hash(),
    )
}

fn and_engine() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(and_idempotent_artifact()),
        ..WhittleEngineConfig::default()
    })
    .unwrap()
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
