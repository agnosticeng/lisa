#![cfg(test)]
use crate::bindings::glm53_math_oracle::linear_projection;
use crate::bindings::glm53_projection::Projection;
use crate::bindings::glm53_projection::ProjectionExecution;
use lisa_metal::kernels::sparse::kernels as kda_kernels;
use lisa_metal::{Array, Dtype};
fn narrow(a: &Array, axis: usize, start: usize, len: usize) -> lisa_metal::error::Result<Array> {
    Ok(Array::new(a.t.narrow(axis, start, len)?))
}
pub struct Source {
    pub q_w: Projection,
    pub k_w: Projection,
    pub v_w: Projection,
    pub f_a_w: Projection,
    pub f_b_w: Projection,
    pub g_a_w: Projection,
    pub g_b_w: Projection,
    pub b_w: Projection,
}
impl Source {
    /// Apply a plain `[out,in]` projection using the reference's BF16 GEMV/GEMM split.
    pub fn linear(&self, x: &Array, weight: &Projection) -> anyhow::Result<Array> {
        match weight {
            Projection::Plain(w) => linear_projection(x, w),
            Projection::Packed(_) => weight.forward(x),
        }
    }

    /// Fuse the two one-to-eight-row dense KDA gate projections. This keeps
    /// each output's reference GEMV reduction order while sharing a Metal dispatch.
    pub fn gate_projections(&self, fa: &Array, ga: &Array) -> anyhow::Result<(Array, Array)> {
        if fa.rank() == 3
            && ga.shape() == fa.shape()
            && fa.dtype() == Dtype::Float32
            && ga.dtype() == Dtype::Float32
            && self.f_b_w.dtype() == Dtype::Bfloat16
            && self.g_b_w.dtype() == Dtype::Bfloat16
        {
            let (b, t, input_dim) = (fa.dim(0) as usize, fa.dim(1) as usize, fa.dim(2) as usize);
            let rows = b.saturating_mul(t);
            if (1..=8).contains(&rows)
                && self.f_b_w.dim(1) as usize == input_dim
                && self.g_b_w.dim(1) as usize == input_dim
                && self.f_b_w.dim(0) == self.g_b_w.dim(0)
            {
                let stream = lisa_metal::Stream::thread_local_or_default();
                let fa_rows = fa.reshape(&[rows as i32, input_dim as i32])?;
                let ga_rows = ga.reshape(&[rows as i32, input_dim as i32])?;
                if let Some((gate, output_gate)) = kda_kernels::bf16_pair_gemv(
                    &self.f_b_w,
                    &fa_rows,
                    &self.g_b_w,
                    &ga_rows,
                    &stream,
                ) {
                    let shape = [b as i32, t as i32, self.f_b_w.dim(0)];
                    return Ok((gate.reshape(&shape)?, output_gate.reshape(&shape)?));
                }
            }
        }
        Ok((self.linear(fa, &self.f_b_w)?, self.linear(ga, &self.g_b_w)?))
    }

    fn qkv(&self, x: &Array) -> anyhow::Result<(Array, Array, Array)> {
        if x.dim(0) == 1
            && x.dim(1) == 1
            && self.q_w.dtype() == Dtype::Bfloat16
            && self.k_w.dtype() == Dtype::Bfloat16
            && self.v_w.dtype() == Dtype::Bfloat16
        {
            let input = x.reshape(&[1, x.dim(2)])?;
            let stream = lisa_metal::Stream::thread_local_or_default();
            if let Some((q, k, v)) =
                kda_kernels::bf16_qkv(&self.q_w, &self.k_w, &self.v_w, &input, &stream)
            {
                let shape = [1, 1, self.q_w.dim(0)];
                return Ok((q.reshape(&shape)?, k.reshape(&shape)?, v.reshape(&shape)?));
            }
        }
        Ok((
            self.linear(x, &self.q_w)?,
            self.linear(x, &self.k_w)?,
            self.linear(x, &self.v_w)?,
        ))
    }

    /// Project all six KDA inputs in one Metal dispatch for one to eight
    /// single-token BF16 decode rows. The fused kernel preserves the separate
    /// the reference-style GEMV row reduction; larger batches retain the staged route.
    pub fn input_projections(
        &self,
        x: &Array,
    ) -> anyhow::Result<(Array, Array, Array, Array, Array, Array)> {
        let rows = x.dim(0) as usize;
        if (1..=8).contains(&rows)
            && x.dim(1) == 1
            && [
                &self.q_w,
                &self.k_w,
                &self.v_w,
                &self.f_a_w,
                &self.g_a_w,
                &self.b_w,
            ]
            .iter()
            .all(|w| w.dtype() == Dtype::Bfloat16)
        {
            let _trace = lisa_metal::trace::span("glm53.kda.bf16_fused_input_projections");
            let input = x.reshape(&[rows as i32, x.dim(2)])?;
            let stream = lisa_metal::Stream::thread_local_or_default();
            if let Some((packed, fa, ga)) = kda_kernels::bf16_kda_project_with_nsg(
                &self.q_w,
                &self.k_w,
                &self.v_w,
                &self.f_a_w,
                &self.g_a_w,
                &self.b_w,
                &input,
                crate::core::quant::glm53_kda_nsg_override(),
                &stream,
            ) {
                let qkv_dim = self.q_w.dim(0) as usize;
                let low_dim = self.f_a_w.dim(0) as usize;
                let heads = self.b_w.dim(0) as usize;
                let shape = [rows as i32, 1, qkv_dim as i32];
                let q = narrow(&packed, 1, 0, qkv_dim)?.reshape(&shape)?;
                let k = narrow(&packed, 1, qkv_dim, qkv_dim)?.reshape(&shape)?;
                let v = narrow(&packed, 1, 2 * qkv_dim, qkv_dim)?.reshape(&shape)?;
                let beta = narrow(&packed, 1, 3 * qkv_dim, heads)?.reshape(&[
                    rows as i32,
                    1,
                    heads as i32,
                ])?;
                return Ok((
                    q,
                    k,
                    v,
                    fa.reshape(&[rows as i32, 1, low_dim as i32])?,
                    ga.reshape(&[rows as i32, 1, low_dim as i32])?,
                    beta,
                ));
            }
        }
        let (q, k, v) = self.qkv(x)?;
        let fa = self.linear(x, &self.f_a_w)?;
        let ga = self.linear(x, &self.g_a_w)?;
        let beta = self.linear(x, &self.b_w)?;
        Ok((q, k, v, fa, ga, beta))
    }
}
