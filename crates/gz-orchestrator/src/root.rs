use gz_engine::{EngineResult, GraphEngine};

pub trait RootSource<E: GraphEngine> {
    fn next_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>>;

    /// Whether each graph returned by `next_root` transfers one engine
    /// reference that must be released with the completed episode.
    fn episode_roots_are_owned(&self) -> bool {
        false
    }
}

impl<E, F> RootSource<E> for F
where
    E: GraphEngine,
    F: FnMut(&mut E) -> EngineResult<Option<E::Graph>>,
{
    fn next_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>> {
        self(engine)
    }
}

/// Derives an episode's Gumbel noise seed from its id (which encodes lane
/// and per-lane order), so episodes sharing a root explore differently
/// while staying deterministic across drivers.
#[must_use]
pub fn episode_noise_seed(episode_id: u64) -> u64 {
    // splitmix64 finalizer.
    let mut z = episode_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
