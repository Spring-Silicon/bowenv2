mod common;

use common::{TestEngine, measure_options};
use gz_engine::{CandidateOptions, GraphEngine, ModelVersion};
use gz_eval::EvalOutput;
use gz_search::{
    EngineIdentity, ExpandResult, ExpandedCandidate, PuctEpisodeContext, PuctEpisodeTask, PuctMcts,
    PuctMctsConfig, PuctRootTask, PuctSearchContext, SearchPoll, SearchWork, SearchWorkResult,
    WorkToken,
};
use std::num::NonZeroUsize;

fn config(max_steps: usize) -> PuctMctsConfig {
    PuctMctsConfig {
        max_steps,
        simulations: NonZeroUsize::MIN,
        c_puct: 1.0,
        seed: 0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(),
    }
}

fn expand_result(
    engine: &mut TestEngine,
    graph: u8,
    options: CandidateOptions,
) -> ExpandResult<u8> {
    let mut candidates = Vec::new();
    engine.candidates(graph, options, &mut candidates).unwrap();
    let candidates = candidates
        .into_iter()
        .map(|candidate| {
            let info = engine.candidate_info(graph, candidate).unwrap();
            ExpandedCandidate {
                candidate,
                candidate_hash: info.candidate_hash,
                kind: info.kind,
                tags: info.tags,
                static_prior: info.static_prior,
            }
        })
        .collect();
    ExpandResult {
        graph_hash: engine.hash(graph).unwrap(),
        candidates,
    }
}

fn output(actions: usize, value: f32) -> EvalOutput {
    EvalOutput {
        model_version: ModelVersion::from_bytes([7; 16]),
        policy_logits: vec![0.0; actions],
        value,
    }
}

#[test]
fn root_task_emits_shared_expand_eval_apply_protocol() {
    let mut engine = TestEngine::new().candidates(0, [1]);
    let search = PuctMcts::new(config(1));
    let mut task = PuctRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        PuctSearchContext::default(),
    );

    let expand = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Expand(work)) => work,
        other => panic!("expected expand, got {other:?}"),
    };
    assert!(matches!(task.poll().unwrap(), SearchPoll::Blocked));
    task.resume(
        expand.token,
        SearchWorkResult::Expand(expand_result(&mut engine, expand.graph, expand.options)),
    )
    .unwrap();

    let eval = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Eval(work)) => work,
        other => panic!("expected eval, got {other:?}"),
    };
    assert_eq!(eval.candidates, vec![1]);
    assert_eq!(eval.request.action_count(), 2);
    task.resume(eval.token, SearchWorkResult::Eval(output(2, 0.0)))
        .unwrap();

    assert!(matches!(
        task.poll().unwrap(),
        SearchPoll::Work(SearchWork::Apply(_))
    ));
}

#[test]
fn root_task_rejects_unknown_token_without_losing_pending_work() {
    let mut engine = TestEngine::new();
    let search = PuctMcts::new(config(1));
    let mut task = PuctRootTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        PuctSearchContext::default(),
    );
    let expand = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Expand(work)) => work,
        other => panic!("expected expand, got {other:?}"),
    };
    let result = SearchWorkResult::Expand(expand_result(&mut engine, 0, expand.options));
    let error = task
        .resume(WorkToken::new(expand.token.value() + 1), result)
        .unwrap_err();

    assert!(error.to_string().contains("unknown work token"));
    assert!(matches!(task.poll().unwrap(), SearchPoll::Blocked));
}

#[test]
fn episode_task_emits_terminal_measure_after_stop() {
    let mut engine = TestEngine::new().candidates(0, []).reward(0, 2.0);
    let search = PuctMcts::new(config(1));
    let mut task = PuctEpisodeTask::new(
        &search,
        EngineIdentity::from_engine(&engine),
        0,
        PuctEpisodeContext::default(),
    );

    let expand = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Expand(work)) => work,
        other => panic!("expected expand, got {other:?}"),
    };
    task.resume(
        expand.token,
        SearchWorkResult::Expand(expand_result(&mut engine, expand.graph, expand.options)),
    )
    .unwrap();
    let eval = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Eval(work)) => work,
        other => panic!("expected eval, got {other:?}"),
    };
    task.resume(eval.token, SearchWorkResult::Eval(output(1, 2.0)))
        .unwrap();

    let measure = match task.poll().unwrap() {
        SearchPoll::Work(SearchWork::Measure(work)) => work,
        other => panic!("expected measure, got {other:?}"),
    };
    assert_eq!(measure.graph, 0);
}
