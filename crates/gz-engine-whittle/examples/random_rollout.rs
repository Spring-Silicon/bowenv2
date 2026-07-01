use gz_engine::{CandidateOptions, GraphEngine, PortableSearchActionRef};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
    rule_name,
};
use gz_search::{RandomSearch, RandomSearchConfig, SearchAction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph_seed = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<u64>())
        .transpose()?
        .unwrap_or(42);
    let rollout_seed = std::env::args()
        .nth(2)
        .map(|arg| arg.parse::<u64>())
        .transpose()?
        .unwrap_or(2);
    let max_steps = std::env::args()
        .nth(3)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(64);

    let mut engine = WhittleEngine::new(WhittleEngineConfig::default())?;
    let generator_config = WhittleGraphGeneratorConfig {
        arity: 6,
        ..WhittleGraphGeneratorConfig::default()
    };
    let mut generator = WhittleGraphGenerator::from_seed(generator_config, graph_seed);
    let generated = generator.sample_into(&mut engine)?;

    let search = RandomSearch::new(RandomSearchConfig {
        max_steps,
        seed: rollout_seed,
        candidate_options: CandidateOptions::default(),
        measure_options: engine.measure_options(),
    });
    let episode = search.run(&mut engine, generated.graph)?;

    let start_measure = engine.measure(generated.graph, engine.measure_options())?;

    println!("graph_seed={graph_seed}");
    println!("rollout_seed={rollout_seed}");
    println!("max_steps={max_steps}");
    println!("generated_graph={}", generated.graph.raw());
    println!("seed_graph={}", generated.seed_graph.raw());
    println!(
        "prewalk_requested={} prewalk_applied={}",
        generated.prewalk_steps_requested, generated.prewalk_steps_applied
    );
    println!(
        "generated_start_cost={} generated_final_cost={}",
        generated.start_cost, generated.final_cost
    );
    println!("root_hash={}", episode.root_context.graph.graph_hash);
    println!("root_reward={:?}", start_measure.scalar_reward);
    println!("steps={}", episode.steps.len());
    println!("stop_reason={:?}", episode.stop_reason);
    println!("final_graph={}", episode.final_graph.raw());
    println!("final_hash={}", episode.final_context.graph.graph_hash);
    println!("final_reward={:?}", episode.final_measure.scalar_reward);

    for (index, step) in episode.steps.iter().enumerate() {
        match step.action {
            SearchAction::Candidate(candidate) => {
                let rule = step
                    .selected_candidate
                    .map(|summary| rule_name(summary.kind.get() as u16))
                    .unwrap_or("Unknown");

                println!(
                    "step={index} action=candidate rule={rule} candidate={} rank={} candidates={} actions={} reward={:?} after={}",
                    candidate.raw(),
                    step.selected_rank,
                    step.engine_candidate_count,
                    step.action_count,
                    step.selected_measure.scalar_reward,
                    step.after.raw()
                );
            }
            SearchAction::Stop => {
                println!(
                    "step={index} action=STOP rank={} candidates={} actions={} reward={:?}",
                    step.selected_rank,
                    step.engine_candidate_count,
                    step.action_count,
                    step.selected_measure.scalar_reward
                );
            }
        }

        if let PortableSearchActionRef::Candidate(candidate) = step.selected_action {
            println!("step={index} candidate_hash={}", candidate.candidate_hash);
        }
    }

    Ok(())
}
