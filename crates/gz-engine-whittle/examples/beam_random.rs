use gz_engine::{CandidateOptions, GraphEngine, PortableSearchActionRef};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleGraphGenerator, WhittleGraphGeneratorConfig,
};
use gz_search::{BeamSearch, BeamSearchConfig, SearchAction};
use std::num::NonZeroUsize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<u64>())
        .transpose()?
        .unwrap_or(42);
    let max_depth = std::env::args()
        .nth(2)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(256);
    let beam_width = std::env::args()
        .nth(3)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    let beam_width = NonZeroUsize::new(beam_width).ok_or("beam_width must be nonzero")?;

    let mut engine = WhittleEngine::new(WhittleEngineConfig::default())?;
    let generator_config = WhittleGraphGeneratorConfig {
        arity: 6,
        ..WhittleGraphGeneratorConfig::default()
    };
    let mut generator = WhittleGraphGenerator::from_seed(generator_config, seed);
    let generated = generator.sample_into(&mut engine)?;

    let search = BeamSearch::new(BeamSearchConfig {
        max_depth,
        beam_width,
        candidate_options: CandidateOptions::default(),
        measure_options: engine.measure_options(),
    });
    let episode = search.run(&mut engine, generated.graph)?;

    let start_measure = engine.measure(generated.graph, engine.measure_options())?;

    println!("command_seed={seed}");
    println!("max_depth={max_depth}");
    println!("beam_width={beam_width}");
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
                println!(
                    "step={index} action=candidate candidate={} rank={} candidates={} actions={} reward={:?} after={}",
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
