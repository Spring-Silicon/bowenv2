use crate::service::{internal, service_work};
use crate::{EpisodeId, WorkerId};
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::EngineEvaluator;
use gz_search::{
    EngineIdentity, GumbelEpisode, GumbelEpisodeContext, GumbelEpisodeTask, GumbelHandleBatch,
    GumbelMcts, SearchPoll,
};

pub struct SerialGumbelOrchestrator<E, V> {
    worker_id: WorkerId,
    next_episode_id: u64,
    engine: E,
    evaluator: V,
    search: GumbelMcts,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrchestratedEpisode<G, C> {
    pub worker_id: WorkerId,
    pub episode_id: EpisodeId,
    pub evaluations: u64,
    pub episode: GumbelEpisode<G, C>,
}

pub type SerialEpisode<G, C> = OrchestratedEpisode<G, C>;

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
        let context = GumbelEpisodeContext {
            noise_seed: crate::root::episode_noise_seed(self.next_episode_id),
            ..context
        };
        let mut task = GumbelEpisodeTask::new(&self.search, identity, root, context);
        let mut evaluations = 0;

        loop {
            let poll = match task.poll() {
                Ok(poll) => poll,
                Err(error) => {
                    self.release_handles(task.take_all_handles())?;
                    return Err(error);
                }
            };
            match poll {
                SearchPoll::Work(work) => {
                    self.release_handles(task.take_releasable())?;
                    evaluations += u64::from(matches!(&work, gz_search::SearchWork::Eval(_)));
                    let token = work.token();
                    let result = match service_work(&mut self.engine, &mut self.evaluator, work) {
                        Ok(result) => result,
                        Err(error) => {
                            self.release_handles(task.take_all_handles())?;
                            return Err(error);
                        }
                    };
                    if let Err(error) = task.resume(token, result) {
                        self.release_handles(task.take_all_handles())?;
                        return Err(error);
                    }
                    self.release_handles(task.take_releasable())?;
                }
                SearchPoll::Blocked => {
                    self.release_handles(task.take_all_handles())?;
                    return Err(internal("serial driver blocked"));
                }
                SearchPoll::Done(episode) => {
                    let episode_id = EpisodeId::new(self.next_episode_id);
                    self.next_episode_id += 1;
                    self.engine
                        .release(&episode.created_graphs, &episode.created_candidates)?;
                    return Ok(OrchestratedEpisode {
                        worker_id: self.worker_id,
                        episode_id,
                        evaluations,
                        episode,
                    });
                }
            }
        }
    }

    fn release_handles(
        &mut self,
        handles: GumbelHandleBatch<E::Graph, E::Candidate>,
    ) -> EngineResult<()> {
        if handles.is_empty() {
            return Ok(());
        }
        self.engine.release(&handles.graphs, &handles.candidates)
    }
}
