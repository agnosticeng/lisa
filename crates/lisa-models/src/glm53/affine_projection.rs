//! The reference affine projection crossover and exact fallback chains.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{AffineSchedule as A, Kernel as K, ScalarType as D, SparseF32 as S};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub shape: Vec<i32>,
    pub weight: Vec<i32>,
    pub input_dtype: D,
    pub weight_dtype: D,
    pub scales_dtype: D,
    pub biases_dtype: D,
    pub bits: i32,
    pub group_size: i32,
    pub native: bool,
}
impl Spec {
    pub fn schedule(&self) -> anyhow::Result<A> {
        anyhow::ensure!(
            !self.shape.is_empty()
                && self.shape.iter().all(|&d| d > 0)
                && self.weight.len() == 2
                && self.weight.iter().all(|&d| d > 0),
            "invalid affine projection geometry"
        );
        let k = *self.shape.last().unwrap();
        let rows = self.shape[..self.shape.len() - 1]
            .iter()
            .try_fold(1i32, |a, &b| a.checked_mul(b))
            .ok_or_else(|| anyhow::anyhow!("affine row overflow"))?;
        if self.native
            && self.bits == 8
            && self.group_size == 64
            && self.input_dtype == D::F32
            && matches!(self.scales_dtype, D::F32 | D::BF16)
            && self.biases_dtype == self.scales_dtype
            && self.weight_dtype == D::U32
            && k % 64 == 0
            && i64::from(self.weight[1]) * 4 == i64::from(k)
        {
            Ok(if rows > 16 {
                A::Matrix
            } else if rows == 1 || k % 128 == 0 {
                A::Tiny
            } else {
                A::Vector
            })
        } else {
            Ok(
                if self.shape.len() == 3 && self.shape[0] > 1 && self.shape[1] == 1 {
                    A::IndependentRows
                } else {
                    A::Generic
                },
            )
        }
    }
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let schedule = self.schedule()?;
        let schema = Default::default();
        let mut p = Build::new(4, &schema);
        let rows = self.shape[..self.shape.len() - 1]
            .iter()
            .product::<i32>()
            .max(1);
        let k = *self.shape.last().unwrap();
        let mut x = V::Input(0);
        let mut scales = V::Input(2);
        let mut biases = V::Input(3);
        if schedule != A::IndependentRows {
            x = p.reshape(x, &[rows, k]);
        }
        if matches!(schedule, A::Tiny | A::Matrix | A::Vector) {
            x = p.contiguous(x);
        }
        if matches!(schedule, A::Tiny | A::Matrix) {
            scales = p.contiguous(scales);
            biases = p.contiguous(biases);
        }
        let y = p.graph.one(
            K::SparseF32(S::AffineProjection {
                group_size: self.group_size,
                bits: self.bits,
                schedule,
            }),
            vec![x, V::Input(1), scales, biases],
        );
        let y = if schedule == A::IndependentRows {
            y
        } else {
            let mut out = self.shape.clone();
            *out.last_mut().unwrap() = self.weight[0];
            p.reshape(y, &out)
        };
        Ok(p.finish(vec![y])?.graph)
    }
}
