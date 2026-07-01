#[allow(dead_code)]
mod common;

use common::{TestEngine, context};
use gz_engine::GraphEngine;
use gz_search::EngineIdentity;

#[test]
fn engine_identity_context_matches_engine_context() {
    let engine = TestEngine::new();
    let identity = EngineIdentity::from_engine(&engine);
    let graph_hash = engine.hash(7).unwrap();

    assert_eq!(identity.context(graph_hash), context(7));
}
