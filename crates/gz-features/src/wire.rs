//! Narrow wire scalars shared by the batch and row encodings.

/// Round-to-nearest-even truncation to bfloat16 bits.
#[must_use]
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

#[must_use]
pub const fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
