//! Per-layer embedding projection, gating and resumable short convolution.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use anyhow::{Result, ensure};
use lisa_backend::{Kernel, ResidualBf16 as R, ScalarType as D, TensorOperation as T};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub hidden: i32,
    pub streams: i32,
    pub dilation: i32,
    pub state_rows: i32,
    pub epsilon: u32,
    pub key: (i32, i32),
    pub value: (i32, i32),
}
pub fn embedding(batch: i32, rows: i32, group_size: i32, bits: i32) -> Graph {
    let mut p = crate::graph::Builder::new(3, 0, 0);
    let v = p.one(
        Kernel::AffineDequantize { group_size, bits },
        vec![V::Input(0), V::Input(1), V::Input(2)],
    );
    let v = p.one(Kernel::Tensor(T::Reshape(vec![batch, rows, -1])), vec![v]);
    p.finish(vec![v])
}
pub fn prepare_convolution(shape: &[i32]) -> Graph {
    let mut p = crate::graph::Builder::new(1, 0, 0);
    let v = if shape.len() == 3 && shape[1] == 1 && shape[2] > 1 {
        p.one(
            Kernel::Tensor(T::Transpose(vec![0, 2, 1])),
            vec![V::Input(0)],
        )
    } else {
        V::Input(0)
    };
    p.finish(vec![v])
}
impl Spec {
    /// Projections retain the decode reduction tree for every captured row.
    /// The gating reductions and dilated convolution already operate on
    /// independent rows with fixed accumulation order.
    pub fn graph_captured(self, shape: &[i32], has_state: bool) -> Result<Graph> {
        Ok(self.graph(shape, has_state)?.with_serial_affine(false))
    }

    /// Inputs: hidden, embedding, key triple, value triple, norm_key,
    /// norm_query, norm_conv, convolution weight, optional previous state.
    /// Outputs: result, cast embedding, gated stream, next state, full history.
    pub fn graph(self, shape: &[i32], has_state: bool) -> Result<Graph> {
        ensure!(
            shape.len() == 3
                && shape.iter().all(|&n| n > 0)
                && self.hidden >= 128
                && self.hidden <= 4096
                && self.hidden % 128 == 0
                && self.streams > 0
                && self.hidden.checked_mul(self.streams) == Some(shape[2])
                && self.dilation > 0
                && self.state_rows >= 0,
            "invalid embedding convolution geometry"
        );
        let [b, s, w] = <[i32; 3]>::try_from(shape).unwrap();
        let full_rows = s
            .checked_add(self.state_rows)
            .ok_or_else(|| anyhow::anyhow!("embedding history overflow"))?;
        let schema = Default::default();
        let mut p = Build::new(if has_state { 13 } else { 12 }, &schema);
        let embedded = p.cast(V::Input(1), D::BF16);
        let epsilon = p.scalar(f32::from_bits(self.epsilon), D::F32);
        let epsilon = p.cast(epsilon, D::BF16);
        let epsilon = p.reshape(epsilon, &[1]);
        let linear = |p: &mut Build<'_>, base, quant: (i32, i32)| {
            p.graph.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![
                    embedded,
                    V::Input(base),
                    V::Input(base + 1),
                    V::Input(base + 2),
                ],
            )
        };
        let key = linear(&mut p, 2, self.key);
        let value = linear(&mut p, 5, self.value);
        let prod = p.graph.one(
            Kernel::ResidualBf16(R::NormalizedProduct {
                hidden: self.hidden,
                streams: self.streams,
            }),
            vec![key, V::Input(0), V::Input(8), V::Input(9), epsilon],
        );
        let prod = p.reshape(prod, &[b, s, self.streams, self.hidden]);
        let sum = p.op(
            T::Sum {
                axis: -1,
                keep_dims: false,
            },
            vec![prod],
        );
        let divisor = p.scalar((self.hidden as f32).sqrt(), D::F32);
        let divisor = p.cast(divisor, D::BF16);
        let divisor = p.reshape(divisor, &[1]);
        let floor = p.scalar(1e-6, D::F32);
        let floor = p.cast(floor, D::BF16);
        let floor = p.reshape(floor, &[1]);
        let gated = p.graph.call(
            Kernel::ResidualBf16(R::GatedNorm {
                hidden: self.hidden,
                streams: self.streams,
            }),
            vec![sum, value, V::Input(10), divisor, floor, epsilon],
        );
        let state = if has_state {
            V::Input(12)
        } else {
            p.graph.one(
                Kernel::Zeros {
                    shape: vec![b, self.state_rows, w],
                    dtype: D::BF16,
                },
                vec![],
            )
        };
        let full = p.concat(vec![state, gated[1]], 1);
        let tail = p.slice(full, 1, s, full_rows);
        let tail = p.contiguous(tail);
        let out = p.graph.one(
            Kernel::ResidualBf16(R::DilatedConvAdd {
                dilation: self.dilation,
            }),
            vec![full, V::Input(11), gated[0]],
        );
        Ok(p.finish(vec![out, embedded, gated[0], tail, full])?.graph)
    }
}

pub struct Names {
    pub convolution: String,
    pub key: &'static str,
    pub value: &'static str,
    pub norm_key: String,
    pub norm_query: String,
    pub norm_convolution: String,
}
pub fn names(prefix: &str) -> Names {
    Names {
        convolution: format!("{prefix}.conv1d.weight"),
        key: "key_proj",
        value: "value_proj",
        norm_key: format!("{prefix}.norm_key"),
        norm_query: format!("{prefix}.norm_query"),
        norm_convolution: format!("{prefix}.norm_conv"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_changes_only_projection_law_and_keeps_wide_history() {
        let spec = Spec {
            hidden: 128,
            streams: 4,
            dilation: 3,
            state_rows: 9,
            epsilon: 1e-6f32.to_bits(),
            key: (32, 4),
            value: (32, 4),
        };
        for rows in [1, 2, 7] {
            for state in [false, true] {
                let serial = spec.graph(&[1, rows, 512], state).unwrap();
                let captured = spec.graph_captured(&[1, rows, 512], state).unwrap();
                captured.validate().unwrap();
                assert_eq!(serial.outputs, captured.outputs);
                assert_eq!(serial.calls.len(), captured.calls.len());
                let mut projections = 0;
                for (original, capture) in serial.calls.iter().zip(&captured.calls) {
                    assert_eq!(original.inputs, capture.inputs);
                    if let Kernel::AffineLinear { group_size, bits } = original.kernel {
                        assert_eq!(
                            capture.kernel,
                            Kernel::AffineSerialLinear {
                                group_size,
                                bits,
                                allow_verify_splitk: false,
                            }
                        );
                        projections += 1;
                    } else {
                        assert_eq!(original.kernel, capture.kernel);
                    }
                }
                assert_eq!(projections, 2);
            }
        }
    }
}
