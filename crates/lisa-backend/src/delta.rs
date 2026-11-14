//! Geometry of a fused causal-convolution/delta-rule kernel. No checkpoint names
//! or model topology are part of this operation contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaDecode {
    pub key_heads: i32,
    pub value_heads: i32,
    pub key_width: i32,
    pub value_width: i32,
    pub convolution: i32,
    pub channels: i32,
    pub projection: i32,
    pub beta_offset: i32,
    pub decay_offset: i32,
    pub gate_offset: i32,
    pub epsilon: u32,
    pub swish: bool,
}
impl DeltaDecode {
    pub fn validate(&self) -> bool {
        let s = self;
        let eps = f32::from_bits(s.epsilon);
        s.key_heads > 0
            && s.value_heads > 0
            && s.value_heads % s.key_heads == 0
            && s.key_width == 128
            && s.value_width == 128
            && s.convolution > 1
            && i64::from(s.channels)
                == (2 * i64::from(s.key_heads) + i64::from(s.value_heads)) * 128
            && s.projection > 0
            && s.projection >= s.channels
            && [s.beta_offset, s.decay_offset, s.gate_offset]
                .iter()
                .all(|&v| v >= 0)
            && i64::from(s.beta_offset) + i64::from(s.value_heads) <= i64::from(s.projection)
            && i64::from(s.decay_offset) + i64::from(s.value_heads) <= i64::from(s.projection)
            && i64::from(s.gate_offset) + i64::from(s.value_heads) * i64::from(s.value_width)
                <= i64::from(s.projection)
            && eps.is_finite()
            && eps >= 0.
    }
}

/// Causal depthwise convolution, normalized query/key and delta gates. Packed
/// gates are selected by offsets; separate gates are explicit kernel operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaPrep {
    pub key_heads: i32,
    pub value_heads: i32,
    pub key_width: i32,
    pub value_width: i32,
    pub convolution: i32,
    pub channels: i32,
    pub projection: i32,
    pub beta_offset: i32,
    pub decay_offset: i32,
    pub separate: bool,
}
impl DeltaPrep {
    pub fn validate(&self) -> bool {
        self.key_heads > 0
            && self.value_heads > 0
            && self.value_heads % self.key_heads == 0
            && self.key_width == 128
            && self.value_width == 128
            && self.convolution > 1
            && i64::from(self.channels)
                == (2 * i64::from(self.key_heads) + i64::from(self.value_heads)) * 128
            && self.projection >= self.channels
            && (self.separate
                || [self.beta_offset, self.decay_offset].iter().all(|&offset| {
                    offset >= self.channels
                        && i64::from(offset) + i64::from(self.value_heads)
                            <= i64::from(self.projection)
                }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaScan {
    Scalar,
    TwoRows,
    FourRows,
    Verify { epsilon: u32 },
}
impl DeltaScan {
    pub fn gated(self) -> bool {
        matches!(self, Self::Verify { .. })
    }
}
