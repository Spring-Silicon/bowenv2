use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Clone, Copy, Debug)]
pub struct SampleConfig {
    pub batch: NonZeroUsize,
    pub window_rows: NonZeroU64,
    pub seed: u64,
}

pub(crate) struct ReplayRng {
    state: u64,
}

impl ReplayRng {
    pub(crate) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_bounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);

        if bound == 1 {
            return 0;
        }

        let zone = u64::MAX - (u64::MAX % bound);

        loop {
            let value = self.next_u64();
            if value < zone {
                return value % bound;
            }
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
