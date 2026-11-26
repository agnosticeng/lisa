//! Qwen4 gated residual computation and checkpoint row packing.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use anyhow::{Result, ensure};
use lisa_backend::{
    Binary as B, Kernel, ResidualBf16 as R, ScalarType as D, TensorOperation as T, Unary as U,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub streams: i32,
    pub hidden: i32,
    pub rank: i32,
    pub down: (i32, i32),
    pub up: (i32, i32),
    pub inject: Option<(i32, i32)>,
    pub packed: bool,
    pub epsilon: u32,
    pub force_norm_operations: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Mixed,
    Mix(bool),
    WithInject,
    Norm,
}
pub struct Names {
    pub norm: String,
    pub down: &'static str,
    pub up: &'static str,
    pub inject: Option<&'static str>,
}
pub fn names(prefix: &str, inject: bool) -> Names {
    Names {
        norm: format!("{prefix}.hc_norm"),
        down: "input_mix_weight_down",
        up: "input_mix_weight_up",
        inject: inject.then_some("block_inject_weight"),
    }
}
pub fn decode_order(hidden: usize, streams: usize, output: usize) -> Option<Vec<i32>> {
    if streams != 4
        || !hidden.is_multiple_of(2)
        || hidden.checked_mul(streams) != Some(output)
        || output > i32::MAX as usize
    {
        return None;
    }
    Some(
        (0..hidden)
            .step_by(2)
            .flat_map(|d| {
                (0..streams)
                    .flat_map(move |s| [(s * hidden + d) as i32, (s * hidden + d + 1) as i32])
            })
            .collect(),
    )
}
pub fn pack_rows(order: &[i32]) -> Graph {
    let mut p = crate::graph::Builder::new(3, 0, 0);
    let indices = p.constant_i32(order, &[order.len() as i32]);
    let out = (0..3)
        .map(|i| {
            p.one(
                Kernel::Tensor(T::Take { axis: 0 }),
                vec![V::Input(i), indices],
            )
        })
        .collect::<Vec<_>>();
    p.finish(out)
}
pub fn norm_scale(streams: i32) -> Graph {
    let schema = Default::default();
    let mut p = Build::new(1, &schema);
    let q = p.scalar(1. / streams as f32, D::BF16);
    let y = p.binary(B::Multiply, V::Input(0), q);
    p.finish(vec![y]).unwrap().graph
}
pub fn inject(shape: &[i32]) -> Result<Graph> {
    ensure!(shape.len() >= 2, "invalid residual shape");
    let schema = Default::default();
    let mut p = Build::new(3, &schema);
    let mut ys = shape[..shape.len() - 1].to_vec();
    ys.extend([1, shape[shape.len() - 1]]);
    let y = p.reshape(V::Input(1), &ys);
    let mut ins = shape[..shape.len() - 1].to_vec();
    ins.extend([-1, 1]);
    let i = p.reshape(V::Input(2), &ins);
    let spread = p.binary(B::Multiply, y, i);
    let mut flat = shape[..shape.len() - 1].to_vec();
    flat.push(-1);
    let spread = p.reshape(spread, &flat);
    let out = p.binary(B::Add, V::Input(0), spread);
    Ok(p.finish(vec![out])?.graph)
}
impl Spec {
    /// Captured rows use the actual one-row mixer, including its split-K down
    /// and packed up paths. Those kernels do not expose an independent row
    /// grid, so this localized graph unrolls only the mixer, not the decoder.
    pub fn graph_captured(self, shape: &[i32], mode: Mode) -> Result<Graph> {
        ensure!(
            shape.len() == 3 && shape.iter().all(|&n| n > 0),
            "invalid captured residual shape"
        );
        let [batch, rows, width] = <[i32; 3]>::try_from(shape).unwrap();
        let count = batch
            .checked_mul(rows)
            .ok_or_else(|| anyhow::anyhow!("residual row overflow"))?;
        if count == 1 {
            return self.graph(shape, mode);
        }
        let scalar = self.graph(&[1, 1, width], mode)?;
        let widths = match mode {
            Mode::Norm => vec![width],
            Mode::Mixed => vec![self.hidden],
            Mode::Mix(_) => vec![self.hidden, self.streams],
            Mode::WithInject => vec![self.hidden, width, self.streams],
        };
        let mut graph = crate::graph::Builder::new(14, 0, 0);
        let flat = graph.one(
            Kernel::Tensor(T::Reshape(vec![count, width])),
            vec![V::Input(0)],
        );
        let mut outputs = vec![Vec::new(); widths.len()];
        for row in 0..count {
            let x = graph.one(
                Kernel::Tensor(T::Slice {
                    axis: 0,
                    start: row,
                    end: row + 1,
                }),
                vec![flat],
            );
            let x = graph.one(Kernel::Tensor(T::Contiguous), vec![x]);
            let x = graph.one(Kernel::Tensor(T::Reshape(vec![1, 1, width])), vec![x]);
            let mut inputs = vec![x];
            inputs.extend((1..14).map(V::Input));
            for (values, value) in outputs.iter_mut().zip(
                graph
                    .append(scalar.clone(), &inputs, &[], &[])
                    .map_err(anyhow::Error::msg)?,
            ) {
                values.push(value);
            }
        }
        let outputs: Vec<_> = outputs
            .into_iter()
            .zip(widths)
            .map(|(values, width)| {
                let joined = graph.one(
                    Kernel::Tensor(T::Concatenate {
                        axis: 1,
                        inputs: values.len(),
                    }),
                    values,
                );
                graph.one(
                    Kernel::Tensor(T::Reshape(vec![batch, rows, width])),
                    vec![joined],
                )
            })
            .collect();
        Ok(graph.finish(outputs))
    }

    /// Inputs: state, norm scale, down triple, up triple, inject triple,
    /// decode-packed up triple. Unused triples can alias their source weights.
    pub fn graph(self, shape: &[i32], mode: Mode) -> Result<Graph> {
        ensure!(
            shape.len() == 3
                && shape.iter().all(|&n| n > 0)
                && self.hidden > 0
                && self.streams > 0
                && self.hidden.checked_mul(self.streams) == Some(shape[2]),
            "invalid gated residual shape"
        );
        let b = shape[0];
        let s = shape[1];
        let h = self.hidden;
        let hc = self.streams;
        let rows = b
            .checked_mul(s)
            .ok_or_else(|| anyhow::anyhow!("residual row overflow"))?;
        ensure!(
            !matches!(mode, Mode::WithInject | Mode::Mix(true)) || self.inject.is_some(),
            "missing residual inject projection"
        );
        let schema = Default::default();
        let mut p = Build::new(14, &schema);
        let mut normed = V::Input(0);
        if matches!(mode, Mode::WithInject | Mode::Norm) {
            let scale = p
                .graph
                .append(norm_scale(hc), &[V::Input(1)], &[], &[])
                .map_err(anyhow::Error::msg)?[0];
            normed = p
                .graph
                .append(
                    crate::normalization::grouped_rms(
                        shape,
                        D::BF16,
                        f32::from_bits(self.epsilon),
                        h as usize,
                        true,
                        self.force_norm_operations,
                    )?,
                    &[normed, scale],
                    &[],
                    &[],
                )
                .map_err(anyhow::Error::msg)?[0];
            if mode == Mode::Norm {
                return Ok(p.finish(vec![normed])?.graph);
            }
        }
        let linear = |p: &mut Build<'_>, x, base, quant: (i32, i32)| {
            p.graph.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![x, V::Input(base), V::Input(base + 1), V::Input(base + 2)],
            )
        };
        if mode == Mode::Mixed {
            let lo = linear(&mut p, normed, 2, self.down);
            let divisor = p.scalar(hc as f32, D::BF16);
            let lo = p.binary(B::Divide, lo, divisor);
            let act = p.unary(U::Silu, lo);
            let w = linear(&mut p, act, 5, self.up);
            let w = p.unary(U::Sigmoid, w);
            let w = p.reshape(w, &[b, s, hc, h]);
            let n = p.reshape(normed, &[b, s, hc, h]);
            let product = p.binary(B::Multiply, w, n);
            let out = p.op(
                T::Mean {
                    axis: -2,
                    keep_dims: false,
                },
                vec![product],
            );
            return Ok(p.finish(vec![out])?.graph);
        }
        let has_inject = match mode {
            Mode::Mix(i) => i,
            Mode::WithInject => true,
            _ => unreachable!(),
        };
        // Preserve the native narrow-window pair, including S=1 packed up rows.
        let fused = matches!(mode, Mode::Mix(_))
            && b == 1
            && s <= 8
            && hc == 4
            && h % 2 == 0
            && self.rank % 32 == 0
            && self.down == (32, 4)
            && self.up == (32, 4)
            && self.inject.is_none_or(|q| q == (32, 4));
        let (input, inject) = if fused {
            let ib = if self.inject.is_some() { 8 } else { 2 };
            let down = p.graph.call(
                Kernel::ResidualBf16(R::DownInject {
                    width: hc * h,
                    rank: self.rank,
                    streams: hc,
                    rows: s,
                    inject: has_inject,
                }),
                vec![
                    normed,
                    V::Input(2),
                    V::Input(3),
                    V::Input(4),
                    V::Input(ib),
                    V::Input(ib + 1),
                    V::Input(ib + 2),
                ],
            );
            let packed = s == 1 && self.packed;
            let ub = if packed { 11 } else { 5 };
            let up = p.graph.call(
                Kernel::ResidualBf16(R::UpMix {
                    hidden: h,
                    streams: hc,
                    rows: s,
                    inject: has_inject,
                    packed,
                }),
                vec![
                    down[1],
                    normed,
                    V::Input(ub),
                    V::Input(ub + 1),
                    V::Input(ub + 2),
                    down[2],
                ],
            );
            (up[0], up[1])
        } else {
            let inj = if let Some(q) = self.inject {
                linear(&mut p, normed, 8, q)
            } else {
                normed
            };
            let lo = linear(&mut p, normed, 2, self.down);
            let act = p.graph.one(
                Kernel::ResidualBf16(R::SiluHead { width: self.rank }),
                vec![lo],
            );
            let act = p.reshape(act, &[b, s, self.rank]);
            let w = linear(&mut p, act, 5, self.up);
            let out = p.graph.call(
                Kernel::ResidualBf16(R::Mix {
                    hidden: h,
                    streams: hc,
                    rows,
                    inject: has_inject,
                }),
                vec![w, normed, inj],
            );
            (out[0], out[1])
        };
        let input = p.reshape(input, &[b, s, h]);
        let inject = p.reshape(inject, &[b, s, hc]);
        Ok(p.finish(if mode == Mode::WithInject {
            vec![input, V::Input(0), inject]
        } else {
            vec![input, inject]
        })?
        .graph)
    }
}

/// Carried hyper-stream injection and ordered normalization. Scale is already
/// divided by the stream count, as in the original decoder schedule.
pub fn inject_norm(
    shape: &[i32],
    hidden: i32,
    streams: i32,
    epsilon: u32,
    inject: bool,
) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        shape.len() == 3
            && shape.iter().all(|&d| d > 0)
            && hidden.checked_mul(streams) == Some(shape[2]),
        "invalid residual normalization geometry"
    );
    let rows = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| anyhow::anyhow!("residual row count overflow"))?;
    let mut g = crate::graph::Builder::new(4, 0, 0);
    let out = g.call(
        lisa_backend::Kernel::ResidualBf16(lisa_backend::ResidualBf16::InjectNorm {
            hidden,
            streams,
            rows,
            inject,
            tile: false,
            epsilon,
        }),
        vec![
            crate::graph::Value::Input(0),
            crate::graph::Value::Input(1),
            crate::graph::Value::Input(2),
            crate::graph::Value::Input(3),
        ],
    );
    let out = out
        .into_iter()
        .map(|x| {
            g.one(
                lisa_backend::Kernel::Tensor(lisa_backend::TensorOperation::Reshape(
                    shape.to_vec(),
                )),
                vec![x],
            )
        })
        .collect::<Vec<_>>();
    Ok(g.finish(out))
}
/// Final carried injection followed by the inject-less stream mixer.
pub fn finish(spec: Spec, shape: &[i32], epsilon: u32, inject: bool) -> Result<Graph> {
    finish_with_capture_law(spec, shape, epsilon, inject, false)
}
pub fn finish_captured(spec: Spec, shape: &[i32], epsilon: u32, inject: bool) -> Result<Graph> {
    finish_with_capture_law(spec, shape, epsilon, inject, true)
}
fn finish_with_capture_law(
    spec: Spec,
    shape: &[i32],
    epsilon: u32,
    inject: bool,
    captured: bool,
) -> Result<Graph> {
    let mut g = crate::graph::Builder::new(16, 0, 0);
    let scale = g
        .append(norm_scale(spec.streams), &[V::Input(3)], &[], &[])
        .map_err(anyhow::Error::msg)?[0];
    let normalized = g
        .append(
            inject_norm(shape, spec.hidden, spec.streams, epsilon, inject)?,
            &[V::Input(0), V::Input(1), V::Input(2), scale],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)?;
    let mut inputs = vec![normalized[1]];
    inputs.extend((3..16).map(V::Input));
    let mixer = if captured {
        spec.graph_captured(shape, Mode::Mix(false))?
    } else {
        spec.graph(shape, Mode::Mix(false))?
    };
    let mixed = g
        .append(mixer, &inputs, &[], &[])
        .map_err(anyhow::Error::msg)?;
    Ok(g.finish(vec![mixed[0], normalized[0]]))
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    #[test]
    fn captured_mixer_retains_each_packed_single_row_contract() {
        let spec = Spec {
            streams: 4,
            hidden: 2560,
            rank: 320,
            down: (32, 4),
            up: (32, 4),
            inject: Some((32, 4)),
            packed: true,
            epsilon: 1e-6f32.to_bits(),
            force_norm_operations: false,
        };
        for (batch, rows) in [(1, 1), (1, 7), (2, 2)] {
            let graph = spec
                .graph_captured(&[batch, rows, 10240], Mode::Mix(true))
                .unwrap();
            graph.validate().unwrap();
            let mut down = 0;
            let mut up = 0;
            for call in &graph.calls {
                match call.kernel {
                    Kernel::ResidualBf16(R::DownInject { rows, .. }) => {
                        assert_eq!(rows, 1);
                        down += 1;
                    }
                    Kernel::ResidualBf16(R::UpMix { rows, packed, .. }) => {
                        assert_eq!(rows, 1);
                        assert!(packed);
                        up += 1;
                    }
                    _ => {}
                }
            }
            assert_eq!(down, batch * rows);
            assert_eq!(up, batch * rows);
        }
    }
}
