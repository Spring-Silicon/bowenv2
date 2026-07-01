use crate::{EpisodeId, WorkerId};
use gz_engine::{EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine};
use gz_eval::{EngineEvalRequest, EngineEvaluator};
use gz_search::{
    EngineIdentity, ExpandedCandidate, GumbelEpisode, GumbelEpisodeContext, GumbelEpisodeTask,
    GumbelMcts, SearchPoll, SearchWork, SearchWorkResult,
};

pub struct SerialGumbelOrchestrator<E, V> {
    worker_id: WorkerId,
    next_episode_id: u64,
    engine: E,
    evaluator: V,
    search: GumbelMcts,
}

pub struct SerialEpisode<G, C> {
    pub worker_id: WorkerId,
    pub episode_id: EpisodeId,
    pub episode: GumbelEpisode<G, C>,
}

impl<E, V> SerialGumbelOrchestrator<E, V>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    pub fn new(worker_id: WorkerId, engine: E, evaluator: V, search: GumbelMcts) -> Self {
        Self {
            worker_id,
            next_episode_id: 0,
            engine,
            evaluator,
            search,
        }
    }

    pub fn run_from_root(
        &mut self,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>> {
        let root = self.engine.root();
        self.run(root, context)
    }

    pub fn run(
        &mut self,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<SerialEpisode<E::Graph, E::Candidate>> {
        let identity = EngineIdentity::from_engine(&self.engine);
        let mut task = GumbelEpisodeTask::new(&self.search, identity, root, context);

        loop {
            match task.poll()? {
                SearchPoll::Work(work) => {
                    let token = work.token();
                    let result = service_work(&mut self.engine, &mut self.evaluator, work)?;
                    task.resume(token, result)?;
                }
                SearchPoll::Blocked => return Err(internal("serial driver blocked")),
                SearchPoll::Done(episode) => {
                    let episode_id = EpisodeId::new(self.next_episode_id);
                    self.next_episode_id += 1;
                    return Ok(SerialEpisode {
                        worker_id: self.worker_id,
                        episode_id,
                        episode,
                    });
                }
            }
        }
    }
}

fn service_work<E, V>(
    engine: &mut E,
    evaluator: &mut V,
    work: SearchWork<E::Graph, E::Candidate>,
) -> EngineResult<SearchWorkResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    match work {
        SearchWork::Expand(work) => {
            let mut candidates = Vec::new();
            engine.candidates(work.graph, work.options, &mut candidates)?;
            let graph_hash = engine.hash(work.graph)?;
            let candidates = candidates
                .into_iter()
                .map(|candidate| {
                    engine
                        .candidate_info(work.graph, candidate)?
                        .validate()
                        .map_err(|_| internal("invalid candidate info"))
                        .map(|info| ExpandedCandidate {
                            candidate,
                            candidate_hash: info.candidate_hash,
                            kind: info.kind,
                            tags: info.tags,
                            static_prior: info.static_prior,
                        })
                })
                .collect::<EngineResult<Vec<_>>>()?;

            Ok(SearchWorkResult::Expand(gz_search::ExpandResult {
                graph_hash,
                candidates,
            }))
        }
        SearchWork::Apply(work) => engine
            .apply(work.graph, work.candidate)
            .map(SearchWorkResult::Apply),
        SearchWork::Measure(work) => engine
            .measure(work.graph, work.options)
            .map(SearchWorkResult::Measure),
        SearchWork::Eval(work) => {
            let output = evaluator.evaluate(
                engine,
                EngineEvalRequest {
                    graph: work.graph,
                    candidates: &work.candidates,
                    request: &work.request,
                    measure_options: work.measure_options,
                },
            )?;
            Ok(SearchWorkResult::Eval(output))
        }
        _ => Err(internal("unsupported search work")),
    }
}

fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(1),
        message: ErrorMessage::new(message).expect("internal orchestrator message is short"),
    }
}
