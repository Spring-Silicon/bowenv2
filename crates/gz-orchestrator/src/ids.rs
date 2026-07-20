#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EpisodeId(u64);

impl EpisodeId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}
