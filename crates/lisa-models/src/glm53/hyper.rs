//! DS4 GLM four-stream split/collapse/expansion and its projection shape gates.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use anyhow::{Result, ensure};
use lisa_backend::{Kernel, ScalarType as D, SparseF32 as K, TensorOperation as T};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub streams: usize,
    pub hidden: usize,
    pub epsilon: u32,
    pub rms_epsilon: u32,
    pub iterations: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Mix,
    Split,
    Normed(u32),
    Expand,
}
pub fn requirements(prefix: &str) -> [String; 3] {
    [
        format!("{prefix}_fn"),
        format!("{prefix}_scale"),
        format!("{prefix}_base"),
    ]
}
pub fn promote_parameter() -> Graph {
    let mut p = crate::graph::Builder::new(1, 0, 0);
    let y = p.one(Kernel::Tensor(T::Cast(D::F32)), vec![V::Input(0)]);
    p.finish(vec![y])
}
impl Spec {
    pub fn validate_matrix(self, shape: &[i32]) -> Result<()> {
        let width = self
            .streams
            .checked_mul(self.hidden)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| anyhow::anyhow!("residual matrix width overflow"))?;
        ensure!(shape == [24, width], "invalid residual matrix geometry");
        Ok(())
    }
    pub fn normed_mode(self, shape: &[i32], epsilon: f32) -> Option<Mode> {
        if shape.len() == 4 && (shape[2] != 4 || shape[3] != 4096) {
            None
        } else {
            Some(Mode::Normed(epsilon.to_bits()))
        }
    }

    pub fn graph(self, shape: &[i32], dtype: D, mode: Mode) -> Result<Graph> {
        ensure!(
            shape.len() == 4
                && shape.iter().all(|&n| n > 0)
                && self.streams == 4
                && shape[2] == 4
                && shape[3] as usize == self.hidden,
            "unsupported GLM residual geometry"
        );
        ensure!(
            self.iterations > 0 && self.iterations <= i32::MAX as usize,
            "invalid Sinkhorn iteration count"
        );
        let b = shape[0];
        let t = shape[1];
        let h = shape[3];
        let rows = b
            .checked_mul(t)
            .ok_or_else(|| anyhow::anyhow!("residual row overflow"))?;
        let schema = Default::default();
        // Mix/split inputs: state, matrix, scale, base, optional norm weight.
        // Expand inputs: y, residual, post, combination matrix.
        let mut p = Build::new(
            if matches!(mode, Mode::Normed(_)) {
                5
            } else {
                4
            },
            &schema,
        );
        if mode == Mode::Expand {
            let a = (0..4)
                .map(|i| p.contiguous(V::Input(i)))
                .collect::<Vec<_>>();
            let y = p.graph.one(Kernel::SparseF32(K::Expand4), a);
            return Ok(p.finish(vec![y])?.graph);
        }
        let x = p.cast(V::Input(0), D::F32);
        let x = p.reshape(x, &[rows, 4 * h]);
        let mixes = if rows <= 8 && h == 4096 && dtype == D::BF16 {
            p.graph.one(
                Kernel::SparseF32(K::MixRmsBf16Linear {
                    epsilon: self.rms_epsilon,
                }),
                vec![V::Input(1), x],
            )
        } else {
            let norm = p.graph.one(
                Kernel::SparseF32(K::MixRms {
                    epsilon: self.rms_epsilon,
                }),
                vec![x],
            );
            if dtype == D::BF16 {
                p.graph.one(
                    Kernel::SparseF32(K::Bf16Linear { wide: rows > 8 }),
                    vec![V::Input(1), norm],
                )
            } else {
                let w = p.cast(V::Input(1), D::F32);
                let w = p.transpose(w, &[1, 0]);
                p.op(T::Matmul, vec![norm, w])
            }
        };
        if mode == Mode::Mix {
            return Ok(p.finish(vec![mixes])?.graph);
        }
        if let Mode::Normed(norm_epsilon) = mode {
            ensure!(
                h == 4096,
                "fused residual normalization requires width 4096"
            );
            let out = p.graph.call(
                Kernel::SparseF32(K::SplitNorm4 {
                    epsilon: self.epsilon,
                    norm_epsilon,
                    iterations: self.iterations,
                }),
                vec![mixes, V::Input(2), V::Input(3), V::Input(0), V::Input(4)],
            );
            let collapsed = p.reshape(out[3], &[b, t, h]);
            let norm = p.reshape(out[4], &[b, t, h]);
            return Ok(p
                .finish(vec![collapsed, norm, out[1], out[2], mixes, out[0]])?
                .graph);
        }
        let split = p.graph.one(
            Kernel::SparseF32(K::SplitSinkhorn4 {
                epsilon: self.epsilon,
                iterations: self.iterations,
            }),
            vec![mixes, V::Input(2), V::Input(3)],
        );
        let mut col = |start, end, shape: &[i32]| {
            let v = p.slice(split, -1, start, end);
            let v = p.contiguous(v);
            p.reshape(v, shape)
        };
        let pre = col(0, 4, &[b, t, 4]);
        let post = col(4, 8, &[b, t, 4]);
        let comb = col(8, 24, &[b, t, 4, 4]);
        let pre = p.contiguous(pre);
        let state = p.contiguous(V::Input(0));
        let y = p
            .graph
            .one(Kernel::SparseF32(K::WeightedSum4), vec![state, pre]);
        Ok(p.finish(vec![y, post, comb])?.graph)
    }
}
