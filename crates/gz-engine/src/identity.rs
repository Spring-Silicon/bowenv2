//! Portable engine identity shared by search, replay, and serving.

use crate::{
    ActionSetHash, EngineId, EngineVersion, GraphEngine, GraphHash, PortableGraphId,
    ReplayGraphContext,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct EngineIdentity {
    pub engine_id: EngineId,
    pub engine_version: EngineVersion,
    pub action_set_hash: ActionSetHash,
}

impl EngineIdentity {
    #[must_use]
    pub fn from_engine<E: GraphEngine>(engine: &E) -> Self {
        Self {
            engine_id: engine.engine_id(),
            engine_version: engine.engine_version(),
            action_set_hash: engine.action_set_hash(),
        }
    }

    #[must_use]
    pub const fn from_context(context: ReplayGraphContext) -> Self {
        Self {
            engine_id: context.graph.engine_id,
            engine_version: context.graph.engine_version,
            action_set_hash: context.action_set_hash,
        }
    }

    #[must_use]
    pub const fn context(self, graph_hash: GraphHash) -> ReplayGraphContext {
        ReplayGraphContext::new(
            PortableGraphId::new(graph_hash, self.engine_id, self.engine_version),
            self.action_set_hash,
        )
    }
}
