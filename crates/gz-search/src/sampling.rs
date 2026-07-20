pub(crate) fn root_seed(seed: u64, root_step: u32) -> u64 {
    seed ^ 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(u64::from(root_step) + 1)
}

pub(crate) struct SearchRng {
    state: u64,
}

impl SearchRng {
    const STEP: u64 = 0x9e37_79b9_7f4a_7c15;

    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn unit(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        let unit = (value as f32 + 0.5) / (1_u32 << 24) as f32;
        unit.clamp(1.0e-7, 1.0 - 1.0e-7)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::STEP);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
