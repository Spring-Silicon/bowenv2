use gz_engine::GraphEngine;
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
    WhittleRoot,
};

#[test]
fn sampled_root_transfers_only_the_final_graph_reference() {
    let generator_config = WhittleGraphGeneratorConfig::default();
    let mut engine = WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Input {
            arity: generator_config.arity,
            capacity: generator_config.capacity,
            input_index: 0,
        },
        ..WhittleEngineConfig::default()
    })
    .unwrap();
    let mut generator = WhittleGraphGenerator::from_seed(generator_config, 42);

    let generated = generator.sample_root_into(&mut engine).unwrap();
    let occupied = engine.arena_occupancy();

    assert_eq!(occupied.graph_refs, 2);
    assert!(occupied.graphs_live <= 2);

    engine.release(&[generated], &[]).unwrap();
    let released = engine.arena_occupancy();
    assert_eq!(released.graph_refs, 1);
    assert_eq!(released.graphs_live, 1);
}
