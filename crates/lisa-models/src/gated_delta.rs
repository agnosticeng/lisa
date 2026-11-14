//! Recurrent-state laws. Compact histories preserve sequential delta arithmetic;
//! model plans select base/final states and define accepted-prefix restoration.
use crate::{
    graph::{Builder, Graph, Value},
    program::Build,
};
use lisa_backend::{Kernel, TensorOperation as T};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayGeometry {
    pub batch: i32,
    pub tokens: i32,
    pub key_heads: i32,
    pub value_heads: i32,
    pub key_width: i32,
    pub value_width: i32,
}
impl ReplayGeometry {
    pub fn accepts_types(&self, types: [Option<lisa_backend::ScalarType>; 6]) -> bool {
        use lisa_backend::ScalarType::{BF16, F32};
        self.supported()
            && types
                == [
                    Some(BF16),
                    Some(BF16),
                    Some(BF16),
                    Some(F32),
                    Some(F32),
                    Some(F32),
                ]
    }
    /// Admission for the existing audited compact-history schedule.
    pub fn supported(&self) -> bool {
        self.batch == 1
            && self.tokens > 1
            && self.key_heads > 0
            && self.value_heads > 0
            && self.value_heads % self.key_heads == 0
            && self.key_width == 128
            && self.value_width == 128
    }
    pub fn capture(&self) -> anyhow::Result<Graph> {
        anyhow::ensure!(self.supported(), "unsupported compact delta history");
        let mut b = Builder::new(6, 0, 0);
        let base = b.one(
            Kernel::Tensor(T::Reshape(vec![
                self.batch,
                self.value_heads,
                self.value_width,
                self.key_width,
            ])),
            vec![Value::Input(5)],
        );
        let out = b.call(
            Kernel::DeltaReplayCapture,
            vec![
                Value::Input(0),
                Value::Input(1),
                Value::Input(2),
                Value::Input(3),
                Value::Input(4),
                base,
            ],
        );
        // Activation/final outputs followed by the five immutable history bindings.
        Ok(b.finish(vec![
            out[0],
            out[1],
            base,
            out[1],
            Value::Input(1),
            Value::Input(3),
            out[2],
        ]))
    }
    /// Inputs: base, final, keys, decay, delta. Boundary zero and the complete
    /// window reuse the corresponding immutable state without a replay dispatch.
    pub fn restore(&self, prefix: usize) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            self.supported() && prefix <= self.tokens as usize,
            "delta replay exceeds captured window"
        );
        let mut b = Builder::new(5, 0, 0);
        let out = if prefix == 0 {
            Value::Input(0)
        } else if prefix == self.tokens as usize {
            Value::Input(1)
        } else {
            b.one(
                Kernel::DeltaReplayRestore { prefix },
                vec![
                    Value::Input(0),
                    Value::Input(2),
                    Value::Input(3),
                    Value::Input(4),
                ],
            )
        };
        Ok(b.finish(vec![out]))
    }
}
/// Snapshot capture layout is [B*S,Hv,Dv,Dk]. The existing serving law restores
/// a single-stream prefix as a rank-three state, retaining its original offset.
pub fn snapshot_state(prefix: usize, shape: &[i32]) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        shape.len() == 4 && prefix > 0 && prefix <= shape[0] as usize,
        "invalid GDN capture boundary"
    );
    let schema = Default::default();
    let mut p = Build::new(1, &schema);
    let out = p.slice(Value::Input(0), 0, prefix as i32 - 1, prefix as i32);
    let out = p.reshape(out, &shape[1..]);
    let out = p.contiguous(out);
    Ok(p.finish(vec![out])?.graph)
}
/// Restore convolution histories after an accepted prefix. PLE uses its own
/// history width. The engine installs all returned handles together.
pub fn convolution_prefix(prefix: usize, shape: &[i32], history: usize) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        shape.len() == 3
            && shape.iter().all(|&d| d > 0)
            && prefix
                .checked_add(history)
                .is_some_and(|end| end <= shape[1] as usize),
        "invalid convolution capture boundary"
    );
    let schema = Default::default();
    let mut p = Build::new(1, &schema);
    let out = p.slice(
        Value::Input(0),
        1,
        i32::try_from(prefix)?,
        i32::try_from(prefix + history)?,
    );
    let out = p.contiguous(out);
    Ok(p.finish(vec![out])?.graph)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_boundaries_choose_immutable_state_without_kernels() {
        let g = ReplayGeometry {
            batch: 1,
            tokens: 7,
            key_heads: 16,
            value_heads: 48,
            key_width: 128,
            value_width: 128,
        };
        for n in 0..=7 {
            let p = g.restore(n).unwrap();
            p.validate().unwrap();
            assert_eq!(p.calls.len(), usize::from(n > 0 && n < 7));
        }
        assert!(g.restore(8).is_err());
        assert!(snapshot_state(0, &[7, 48, 128, 128]).is_err());
        assert!(convolution_prefix(7, &[1, 9, 10240], 3).is_err());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub kernel: lisa_backend::DeltaDecode,
    pub batch: i32,
    pub convolution_present: bool,
    pub state_present: bool,
    /// Packed input/output projection group sizes and bit widths.
    pub projections: [(i32, i32); 2],
}
impl DecodePlan {
    /// Weights: input packed triple, convolution, negative-exp decay, bias,
    /// normalization scale, output packed triple. Inputs follow x, optional
    /// convolution state, optional SSM state. Outputs are activation/conv/SSM.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        use lisa_backend::ScalarType;
        let s = &self.kernel;
        anyhow::ensure!(
            s.validate() && self.batch > 0,
            "invalid fused delta decode geometry"
        );
        let mut b = Builder::new(
            1 + usize::from(self.convolution_present) + usize::from(self.state_present),
            10,
            0,
        );
        let linear = |b: &mut Builder, x, weight, quant: (i32, i32)| {
            b.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![
                    x,
                    Value::Weight(weight),
                    Value::Weight(weight + 1),
                    Value::Weight(weight + 2),
                ],
            )
        };
        let projection = linear(&mut b, Value::Input(0), 0, self.projections[0]);
        let mut next = 1;
        let conv = if self.convolution_present {
            let v = Value::Input(next);
            next += 1;
            v
        } else {
            b.one(
                Kernel::Zeros {
                    shape: vec![self.batch, s.convolution - 1, s.channels],
                    dtype: ScalarType::BF16,
                },
                vec![],
            )
        };
        let state = if self.state_present {
            Value::Input(next)
        } else {
            b.one(
                Kernel::Zeros {
                    shape: vec![self.batch, s.value_heads, s.value_width, s.key_width],
                    dtype: ScalarType::F32,
                },
                vec![],
            )
        };
        let state = b.one(
            Kernel::Tensor(T::Reshape(vec![
                self.batch,
                s.value_heads,
                s.value_width,
                s.key_width,
            ])),
            vec![state],
        );
        let conv_weight = b.one(
            Kernel::Tensor(T::Reshape(vec![s.channels, s.convolution])),
            vec![Value::Weight(3)],
        );
        let out = b.call(
            Kernel::DeltaDecode(*s),
            vec![
                projection,
                conv,
                conv_weight,
                Value::Weight(4),
                Value::Weight(5),
                state,
                Value::Weight(6),
            ],
        );
        let gated = b.one(
            Kernel::Tensor(T::Reshape(vec![
                self.batch,
                1,
                s.value_heads * s.value_width,
            ])),
            vec![out[0]],
        );
        let activation = linear(&mut b, gated, 7, self.projections[1]);
        Ok(b.finish(vec![activation, out[2], out[1]]))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepPlan {
    pub kernel: lisa_backend::DeltaPrep,
    pub batch: i32,
    pub tokens: i32,
    pub gate_offset: i32,
    pub capture: bool,
    pub convolution_present: bool,
    /// Packed, qkv, output gate, beta, decay projection encodings.
    pub projections: [(i32, i32); 5],
}
impl PrepPlan {
    pub fn separate(batch: i32, tokens: i32, capture: bool) -> bool {
        !capture && batch == 1 && tokens > 8
    }
    /// Bind all five projection triples, then convolution, decay, bias.
    /// Outputs: q/k/v/decay/beta, convolution, z, optional convolution history.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let s = self.kernel;
        anyhow::ensure!(
            s.validate()
                && self.batch > 0
                && self.tokens > 0
                && s.separate == Self::separate(self.batch, self.tokens, self.capture),
            "invalid delta preparation plan"
        );
        anyhow::ensure!(
            self.gate_offset >= 0
                && i64::from(self.gate_offset)
                    + i64::from(s.value_heads) * i64::from(s.value_width)
                    <= i64::from(s.projection),
            "invalid output gate range"
        );
        let mut b = Builder::new(1 + usize::from(self.convolution_present), 18, 0);
        let linear = |b: &mut Builder, i: usize| {
            let quant = self.projections[i];
            b.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![
                    Value::Input(0),
                    Value::Weight(i * 3),
                    Value::Weight(i * 3 + 1),
                    Value::Weight(i * 3 + 2),
                ],
            )
        };
        // Preserve projection and allocation order from the original execution.
        let (proj, z, gates) = if s.separate {
            let proj = linear(&mut b, 1);
            let z = linear(&mut b, 2);
            let beta = linear(&mut b, 3);
            let decay = linear(&mut b, 4);
            let z = b.one(
                Kernel::Tensor(T::Reshape(vec![
                    self.batch,
                    self.tokens,
                    s.value_heads,
                    s.value_width,
                ])),
                vec![z],
            );
            (proj, z, Some((beta, decay)))
        } else {
            let proj = linear(&mut b, 0);
            let z = b.one(
                Kernel::Tensor(T::Slice {
                    axis: 2,
                    start: self.gate_offset,
                    end: self.gate_offset + s.value_heads * s.value_width,
                }),
                vec![proj],
            );
            (proj, z, None)
        };
        let conv = if self.convolution_present {
            Value::Input(1)
        } else {
            b.one(
                Kernel::Zeros {
                    shape: vec![self.batch, s.convolution - 1, s.channels],
                    dtype: lisa_backend::ScalarType::BF16,
                },
                vec![],
            )
        };
        let mut args = vec![
            proj,
            conv,
            Value::Weight(15),
            Value::Weight(16),
            Value::Weight(17),
        ];
        if let Some((beta, decay)) = gates {
            args.extend([beta, decay]);
        }
        let out = b.call(Kernel::DeltaPrep(s), args);
        let mut result = out[..6].to_vec();
        result.push(z);
        if !s.separate {
            result.push(out[6]);
        }
        Ok(b.finish(result))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureKind {
    None,
    Snapshots,
    Replay,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrencePlan {
    pub geometry: ReplayGeometry,
    pub capture: bool,
    pub replay: bool,
    pub cache: bool,
    pub swish: bool,
    pub verify_fusion: bool,
    pub epsilon: u32,
    pub state_present: bool,
    pub reuse: bool,
    pub projection: (i32, i32),
}
impl RecurrencePlan {
    pub fn capture_kind(&self) -> CaptureKind {
        if !self.capture {
            CaptureKind::None
        } else if self.replay && self.cache && self.geometry.supported() {
            CaptureKind::Replay
        } else {
            CaptureKind::Snapshots
        }
    }
    pub fn mode(&self) -> lisa_backend::DeltaScan {
        use lisa_backend::DeltaScan;
        if self.capture
            && !self.replay
            && self.swish
            && self.verify_fusion
            && (2..=16).contains(&self.geometry.tokens)
        {
            DeltaScan::Verify {
                epsilon: self.epsilon,
            }
        } else if self.geometry.tokens > 8 && !self.capture {
            DeltaScan::FourRows
        } else if self.geometry.tokens > 1 {
            DeltaScan::TwoRows
        } else {
            DeltaScan::Scalar
        }
    }
    pub fn accepts_workspace(&self, y: Option<&[i32]>, state: Option<&[i32]>) -> bool {
        let g = self.geometry;
        self.capture_kind() == CaptureKind::Snapshots
            && matches!(
                self.mode(),
                lisa_backend::DeltaScan::TwoRows | lisa_backend::DeltaScan::Verify { .. }
            )
            && y == Some(&[g.batch, g.tokens, g.value_heads, g.value_width][..])
            && state
                == Some(
                    &[
                        g.batch * g.tokens,
                        g.value_heads,
                        g.value_width,
                        g.key_width,
                    ][..],
                )
    }
    /// Inputs q/k/v/decay/beta/z, optional initial state, optional reusable
    /// output/history storage. Weights: RMS scale, output projection triple.
    /// Outputs activation, recurrence output, captured/final state, committed
    /// state, then optional replay bindings (base, final, keys, decay, deltas).
    pub fn graph(&self) -> anyhow::Result<Graph> {
        use lisa_backend::ScalarType;
        let g = self.geometry;
        let mode = self.mode();
        let kind = self.capture_kind();
        anyhow::ensure!(
            g.batch > 0
                && g.tokens > 0
                && g.key_heads > 0
                && g.value_heads > 0
                && g.value_heads % g.key_heads == 0
                && g.key_width == 128
                && g.value_width == 128,
            "unsupported delta recurrence geometry"
        );
        anyhow::ensure!(
            !self.reuse
                || (kind == CaptureKind::Snapshots
                    && matches!(
                        mode,
                        lisa_backend::DeltaScan::TwoRows | lisa_backend::DeltaScan::Verify { .. }
                    )),
            "unsupported delta output reuse"
        );
        let mut b = Builder::new(
            6 + usize::from(self.state_present) + 2 * usize::from(self.reuse),
            4,
            0,
        );
        let state = if self.state_present {
            Value::Input(6)
        } else {
            b.one(
                Kernel::Zeros {
                    shape: vec![g.batch, g.value_heads, g.value_width, g.key_width],
                    dtype: ScalarType::F32,
                },
                vec![],
            )
        };
        let mut args = (0..5).map(Value::Input).collect::<Vec<_>>();
        args.push(state);
        let (out, history) = if kind == CaptureKind::Replay {
            let base = b.one(
                Kernel::Tensor(T::Reshape(vec![
                    g.batch,
                    g.value_heads,
                    g.value_width,
                    g.key_width,
                ])),
                vec![state],
            );
            args[5] = base;
            let out = b.call(Kernel::DeltaReplayCapture, args);
            (
                vec![out[0], out[1]],
                vec![base, out[1], Value::Input(1), Value::Input(3), out[2]],
            )
        } else {
            if mode.gated() {
                args.extend([Value::Input(5), Value::Weight(0)]);
            }
            if self.reuse {
                let i = 6 + usize::from(self.state_present);
                b.writable_input(i);
                b.writable_input(i + 1);
                args.extend([Value::Input(i), Value::Input(i + 1)]);
            }
            (
                b.call(
                    Kernel::DeltaScan {
                        mode,
                        capture: self.capture,
                        reuse: self.reuse,
                    },
                    args,
                ),
                vec![],
            )
        };
        // Preserve the single-stream snapshot indexing law. The engine does
        // not derive a prefix or interpret the captured tensor's layout.
        let final_state = if kind == CaptureKind::Snapshots && self.cache {
            let last = b.one(
                Kernel::Tensor(T::Slice {
                    axis: 0,
                    start: g.tokens - 1,
                    end: g.tokens,
                }),
                vec![out[1]],
            );
            let last = b.one(
                Kernel::Tensor(T::Reshape(vec![g.value_heads, g.value_width, g.key_width])),
                vec![last],
            );
            b.one(Kernel::Tensor(T::Contiguous), vec![last])
        } else {
            out[1]
        };
        let normed = if mode.gated() && kind != CaptureKind::Replay {
            out[0]
        } else {
            b.one(
                Kernel::GatedRms {
                    heads: g.value_heads,
                    width: g.value_width,
                    epsilon: self.epsilon,
                    swish: self.swish,
                },
                vec![out[0], Value::Input(5), Value::Weight(0)],
            )
        };
        let normed = b.one(
            Kernel::Tensor(T::Reshape(vec![
                g.batch,
                g.tokens,
                g.value_heads * g.value_width,
            ])),
            vec![normed],
        );
        let activation = b.one(
            Kernel::AffineLinear {
                group_size: self.projection.0,
                bits: self.projection.1,
            },
            vec![normed, Value::Weight(1), Value::Weight(2), Value::Weight(3)],
        );
        let mut result = vec![activation, out[0], out[1], final_state];
        result.extend(history);
        Ok(b.finish(result))
    }
}

/// Initial projection layout and exact bind-time transforms. Concatenation is
/// deliberately pairwise in the original order. Split projections become views
/// into the joined storage, so initialization does not retain duplicate weights.
#[derive(Clone, Copy, Debug)]
pub struct ProjectionLayout {
    pub offsets: [i32; 5],
}
impl ProjectionLayout {
    pub fn new(widths: [i32; 4]) -> anyhow::Result<Self> {
        anyhow::ensure!(widths.iter().all(|&v| v > 0), "empty delta projection");
        let mut offsets = [0i32; 5];
        for (i, n) in widths.into_iter().enumerate() {
            offsets[i + 1] = offsets[i]
                .checked_add(n)
                .ok_or_else(|| anyhow::anyhow!("delta projection overflow"))?;
        }
        Ok(Self { offsets })
    }
    /// Inputs: qkv/z/beta/decay projection triples, convolution, A_log.
    /// Outputs: joined triple, four view triples, convolution, -exp(A_log.f32).
    pub fn initialize(&self, convolution_shape: &[i32]) -> Graph {
        let mut b = Builder::new(14, 0, 0);
        let conv = if convolution_shape.len() == 3
            && convolution_shape[1] == 1
            && convolution_shape[2] > 1
        {
            b.one(
                Kernel::Tensor(T::Transpose(vec![0, 2, 1])),
                vec![Value::Input(12)],
            )
        } else {
            Value::Input(12)
        };
        let joined: Vec<_> = (0..3)
            .map(|column| {
                let mut part = Value::Input(column);
                for row in 1..4 {
                    part = b.one(
                        Kernel::Tensor(T::Concatenate { axis: 0, inputs: 2 }),
                        vec![part, Value::Input(row * 3 + column)],
                    );
                }
                part
            })
            .collect();
        let mut output = joined.clone();
        for i in 0..4 {
            for &part in &joined {
                output.push(b.one(
                    Kernel::Tensor(T::Slice {
                        axis: 0,
                        start: self.offsets[i],
                        end: self.offsets[i + 1],
                    }),
                    vec![part],
                ));
            }
        }
        let alog = b.one(
            Kernel::Tensor(T::Cast(lisa_backend::ScalarType::F32)),
            vec![Value::Input(13)],
        );
        let exp = b.one(
            Kernel::Tensor(T::Unary(lisa_backend::Unary::Exp)),
            vec![alog],
        );
        let neg = b.one(
            Kernel::Tensor(T::Unary(lisa_backend::Unary::Negate)),
            vec![exp],
        );
        output.extend([conv, neg]);
        b.finish(output)
    }
}

/// Checkpoint names and geometry are model contracts, supplied to an engine
/// reader without teaching the format or device about gated-delta layers.
pub struct BindingSchema {
    pub convolution: String,
    pub decay: String,
    pub bias: String,
    pub norm: String,
}
impl BindingSchema {
    pub const INPUTS: [&str; 4] = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"];
    pub const OUTPUT: &str = "out_proj";
    pub fn at(prefix: &str) -> Self {
        Self {
            convolution: format!("{prefix}.conv1d.weight"),
            decay: format!("{prefix}.A_log"),
            bias: format!("{prefix}.dt_bias"),
            norm: format!("{prefix}.norm"),
        }
    }
    pub fn channels(
        key_heads: usize,
        value_heads: usize,
        key_width: usize,
        value_width: usize,
    ) -> anyhow::Result<usize> {
        anyhow::ensure!(
            key_heads > 0
                && value_heads > 0
                && value_heads % key_heads == 0
                && key_width > 0
                && value_width > 0,
            "invalid delta head geometry"
        );
        key_heads
            .checked_mul(key_width)
            .and_then(|k| k.checked_mul(2))
            .zip(value_heads.checked_mul(value_width))
            .and_then(|(k, v)| k.checked_add(v))
            .ok_or_else(|| anyhow::anyhow!("delta channel overflow"))
    }
}

/// Complete shared gated-delta invocation. All stage selection and sequencing
/// live here; the engine binds one graph and stores its declared outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub decode: DecodePlan,
    pub tokens: i32,
    pub capture: bool,
    pub replay: bool,
    pub cache: bool,
    pub verify_fusion: bool,
    pub reuse: bool,
    pub input_projections: [(i32, i32); 5],
}
impl Plan {
    pub fn recurrence(&self) -> RecurrencePlan {
        let s = self.decode.kernel;
        RecurrencePlan {
            geometry: ReplayGeometry {
                batch: self.decode.batch,
                tokens: self.tokens,
                key_heads: s.key_heads,
                value_heads: s.value_heads,
                key_width: s.key_width,
                value_width: s.value_width,
            },
            capture: self.capture,
            replay: self.replay,
            cache: self.cache,
            swish: s.swish,
            verify_fusion: self.verify_fusion,
            epsilon: s.epsilon,
            state_present: self.decode.state_present,
            reuse: self.reuse,
            projection: self.decode.projections[1],
        }
    }
    pub fn capture_kind(&self) -> CaptureKind {
        self.recurrence().capture_kind()
    }
    pub fn is_decode(&self) -> bool {
        !self.capture && self.tokens == 1 && self.decode.kernel.validate()
    }
    /// Inputs: x, optional convolution, optional SSM, optional output/tape.
    /// Weights: five input projection triples, convolution/decay/bias/norm,
    /// output projection triple. The first three outputs always commit the
    /// activation, convolution and SSM. Capture adds its representation below.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let mut b = Builder::new(
            1 + usize::from(self.decode.convolution_present)
                + usize::from(self.decode.state_present)
                + 2 * usize::from(self.reuse),
            22,
            0,
        );
        let weights = (0..22).map(Value::Weight).collect::<Vec<_>>();
        let conv = 1;
        let state = 1 + usize::from(self.decode.convolution_present);
        let scratch = state + usize::from(self.decode.state_present);
        let outputs = if self.is_decode() {
            anyhow::ensure!(!self.reuse, "decode does not use capture storage");
            let mut inputs = vec![Value::Input(0)];
            if self.decode.convolution_present {
                inputs.push(Value::Input(conv));
            }
            if self.decode.state_present {
                inputs.push(Value::Input(state));
            }
            let w = [0, 1, 2, 15, 16, 17, 18, 19, 20, 21].map(Value::Weight);
            b.append(self.decode.graph()?, &inputs, &w, &[])
                .map_err(anyhow::Error::msg)?
        } else {
            let s = self.decode.kernel;
            let prep = PrepPlan {
                kernel: lisa_backend::DeltaPrep {
                    key_heads: s.key_heads,
                    value_heads: s.value_heads,
                    key_width: s.key_width,
                    value_width: s.value_width,
                    convolution: s.convolution,
                    channels: s.channels,
                    projection: s.projection,
                    beta_offset: s.beta_offset,
                    decay_offset: s.decay_offset,
                    separate: PrepPlan::separate(self.decode.batch, self.tokens, self.capture),
                },
                batch: self.decode.batch,
                tokens: self.tokens,
                gate_offset: s.gate_offset,
                capture: self.capture,
                convolution_present: self.decode.convolution_present,
                projections: self.input_projections,
            };
            let mut inputs = vec![Value::Input(0)];
            if self.decode.convolution_present {
                inputs.push(Value::Input(conv));
            }
            let prepared = b
                .append(prep.graph()?, &inputs, &weights[..18], &[])
                .map_err(anyhow::Error::msg)?;
            let mut inputs = prepared[..5].to_vec();
            inputs.push(prepared[6]);
            if self.decode.state_present {
                inputs.push(Value::Input(state));
            }
            if self.reuse {
                inputs.extend([Value::Input(scratch), Value::Input(scratch + 1)]);
            }
            let scan = b
                .append(self.recurrence().graph()?, &inputs, &weights[18..22], &[])
                .map_err(anyhow::Error::msg)?;
            let mut out = vec![scan[0], prepared[5], scan[3]];
            match self.capture_kind() {
                CaptureKind::None => {}
                CaptureKind::Snapshots => out.extend([scan[1], scan[2], prepared[7]]),
                CaptureKind::Replay => {
                    out.push(prepared[7]);
                    out.extend_from_slice(&scan[4..]);
                }
            }
            out
        };
        let graph = b.finish(outputs);
        graph.validate().map_err(anyhow::Error::msg)?;
        Ok(graph)
    }
}
