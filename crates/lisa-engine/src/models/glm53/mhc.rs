//! mHC hyper-connections for `glm5_next`, following ds4's F32 inference graph.
//!
//! The residual stream is
//! `[B, T, C, H]` with `C = hc_mult = 4`. Per half (attn / ffn), a
//! HyperConnection collapses the stream into a module input and expands the
//! module output back into a new stream:
//!
//! ```text
//! mixes = rms(state.flatten(-2)) @ fn.T            # [B, T, 24]  (F32)
//! pre   = sigmoid(mixes[..0:4]  * scale0 + base0) + eps
//! post  = 2 * sigmoid(mixes[..4:8] * scale1 + base4)
//! comb  = softmax(reshape(mixes[..8:24], [C,C]) * scale2 + base8) + eps
//! comb  = col-normalize; then sinkhorn (iters-1 x: /rowsum, /colsum)
//! xc    = sum_c pre[c] * state[c]                  # collapse  [B, T, H]
//! new_c = post[c] * y + sum_p comb[p, c] * state_p # expand    [B, T, C, H]
//! ```

use crate::core::loader::TensorSource;

use lisa_mlx::{Array, Dtype, ops};

pub struct Hc {
    fn_w: Array,          // [24, C*H] source dtype; BF16 decode uses ds4 row GEMV
    base: Array,          // [24] f32
    scale: Array,         // [3] f32
    scale_vals: [f32; 3], // cached at load — per-call item reads cost a GPU sync each
    hc_mult: usize,
    hidden: usize,
    eps: f32,
    sinkhorn_iters: usize,
    rms_eps: f32,
}

pub fn fscalar(a: &Array, v: f32) -> lisa_mlx::error::Result<Array> {
    Ok(Array::from_f32_as(v, a.dtype()))
}

/// Concatenate owned arrays along `axis` (the wrapper op borrows).
pub fn concat(arrays: Vec<Array>, axis: i32) -> lisa_mlx::error::Result<Array> {
    let refs: Vec<&Array> = arrays.iter().collect();
    ops::concatenate(&refs, axis)
}

/// Concatenate borrowed arrays along `axis`.
pub fn concat_refs(arrays: &[&Array], axis: i32) -> lisa_mlx::error::Result<Array> {
    ops::concatenate(arrays, axis)
}

/// Narrow view `axis` in `[start, start+len)` (a view, via the inner array).
pub fn narrow(a: &Array, axis: usize, start: usize, len: usize) -> lisa_mlx::error::Result<Array> {
    Ok(Array::new(a.t.narrow(axis, start, len)?))
}

/// Batched-safe matmul: flatten all leading dims to rows, contract, reshape
/// back (mlx-c matmul is only exact on 2-D here, same as quantized_matmul).
pub fn mm(a: &Array, w: &Array) -> lisa_mlx::error::Result<Array> {
    let sh = a.shape();
    let rank = a.rank();
    let rows: i32 = sh[..rank - 1].iter().product::<i32>().max(1);
    let k = sh[rank - 1];
    let flat = a.reshape(&[rows, k])?;
    let out = flat.matmul(w)?;
    Ok(out.reshape(&[&sh[..rank - 1], &[w.shape()[1]]].concat())?)
}

/// Sum over `axis` keeping a size-1 dim.
pub fn keep_sum(a: &Array, axis: i32) -> lisa_mlx::error::Result<Array> {
    a.sum_axis(axis, true)
}

/// Mean over `axis` keeping a size-1 dim.
pub fn keep_mean(a: &Array, axis: i32) -> lisa_mlx::error::Result<Array> {
    a.mean_axis(axis, true)
}

impl Hc {
    /// `prefix` = `model.layers.N.hc_attn` or `...hc_ffn` (bare tensors:
    /// `{prefix}_fn`, `{prefix}_base`, `{prefix}_scale`).
    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        hc_mult: usize,
        hidden: usize,
        hc_eps: f32,
        sinkhorn_iters: usize,
        rms_eps: f32,
    ) -> anyhow::Result<Self> {
        let fn_w = src.get(&format!("{prefix}_fn"))?;
        let base = src
            .get(&format!("{prefix}_base"))?
            .as_dtype(Dtype::Float32)?;
        let scale = src
            .get(&format!("{prefix}_scale"))?
            .as_dtype(Dtype::Float32)?;
        debug_assert_eq!(fn_w.shape()[0], 24);
        debug_assert_eq!(fn_w.shape()[1], (hc_mult * hidden) as i32);
        let s0 = narrow(&scale, 0, 0, 1)?.item_cast::<f32>();
        let s1 = narrow(&scale, 0, 1, 1)?.item_cast::<f32>();
        let s2 = narrow(&scale, 0, 2, 1)?.item_cast::<f32>();
        Ok(Self {
            fn_w,
            base,
            scale,
            scale_vals: [s0, s1, s2],
            hc_mult,
            hidden,
            eps: hc_eps,
            sinkhorn_iters,
            rms_eps,
        })
    }

    fn mix_rows(&self, state: &Array) -> anyhow::Result<(usize, usize, usize, usize, Array)> {
        let (b, t, c, h) = (
            state.dim(0) as usize,
            state.dim(1) as usize,
            state.dim(2) as usize,
            state.dim(3) as usize,
        );
        debug_assert_eq!(c, self.hc_mult);
        debug_assert_eq!(h, self.hidden);
        let flat = state
            .as_dtype(Dtype::Float32)?
            .reshape(&[(b * t) as i32, (c * h) as i32])?;
        // RMS over the flattened stream (single norm over C*H), matching the
        // reference `rms_norm(y.flatten(-2), eps)` before the mHC projection.
        let rows = b * t;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        if rows > 0
            && rows <= 8
            && let Some(mixes) = lisa_mlx::models::glm53::kernels::glm53_hc_mix_rms_bf16_gemv(
                &self.fn_w,
                &flat,
                self.rms_eps,
                &stream,
            )
        {
            return Ok((b, t, c, h, mixes));
        }
        let normed =
            lisa_mlx::models::glm53::kernels::glm53_hc_mix_rms(&flat, self.rms_eps, &stream)
                .ok_or_else(|| anyhow::anyhow!("GLM mHC input RMSNorm Metal kernel failed"))?;
        if rows > 0 && rows <= 8 && self.fn_w.dtype() == Dtype::Bfloat16 {
            let mixes =
                lisa_mlx::models::glm53::kernels::glm53_bf16_gemv(&self.fn_w, &normed, &stream)
                    .ok_or_else(|| anyhow::anyhow!("GLM mHC BF16 GEMV failed for {rows} rows"))?;
            return Ok((b, t, c, h, mixes));
        }
        if rows > 8 && self.fn_w.dtype() == Dtype::Bfloat16 {
            let mixes =
                lisa_mlx::models::glm53::kernels::glm53_bf16_gemm(&self.fn_w, &normed, &stream)
                    .ok_or_else(|| anyhow::anyhow!("GLM mHC BF16 GEMM failed for {rows} rows"))?;
            return Ok((b, t, c, h, mixes));
        }
        let w = self
            .fn_w
            .as_dtype(Dtype::Float32)?
            .transpose_axes(&[1, 0])?;
        let mixes = normed.matmul(&w)?;
        Ok((b, t, c, h, mixes))
    }

    /// Fused ds4 HC=4 split, collapse, and module-input RMSNorm. The GLM
    /// architecture requires this Metal path; unsupported other shapes use
    /// the separate operation path.
    pub fn split_normed(
        &self,
        state: &Array,
        norm_weight: &Array,
        norm_eps: f32,
    ) -> anyhow::Result<Option<(Array, Array, Array, Array)>> {
        Ok(self
            .split_normed_with_mixes(state, norm_weight, norm_eps)?
            .map(|(collapsed, normed, post, comb, _mixes, _split)| (collapsed, normed, post, comb)))
    }

    /// The capture variant also returns the 24-wide mHC projection so parity
    /// traces can compare DS4's split/Sinkhorn from the exact same activation.
    pub fn split_normed_with_mixes(
        &self,
        state: &Array,
        norm_weight: &Array,
        norm_eps: f32,
    ) -> anyhow::Result<Option<(Array, Array, Array, Array, Array, Array)>> {
        let (b, t, c, h, mixes) = self.mix_rows(state)?;
        if c != 4 || h != 4096 {
            return Ok(None);
        }
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let (split, post, comb, collapsed, normed) =
            lisa_mlx::models::glm53::kernels::hc_split_weighted_sum_norm4(
                &mixes,
                &self.scale,
                &self.base,
                state,
                norm_weight,
                self.eps,
                norm_eps,
                self.sinkhorn_iters,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("GLM HC=4 split/collapse/norm Metal kernel failed"))?;
        let collapsed = collapsed.reshape(&[b as i32, t as i32, h as i32])?;
        let normed = normed.reshape(&[b as i32, t as i32, h as i32])?;
        Ok(Some((collapsed, normed, post, comb, mixes, split)))
    }

    /// Collapse the stream: returns `(xc [B,T,H] f32, post [B,T,C] f32,
    /// comb [B,T,C,C] f32)`.
    pub fn split(&self, state: &Array) -> anyhow::Result<(Array, Array, Array)> {
        let (b, t, c, h, mixes) = self.mix_rows(state)?;
        let dbg = std::env::var("LISA_GLM53_DEBUG").is_ok();
        macro_rules! dstep {
            ($n:expr, $e:expr) => {{
                let r: lisa_mlx::error::Result<Array> = (|| Ok($e))();
                match r {
                    Ok(a) => {
                        if dbg {
                            eprintln!("[dbg-mhc] {} {:?}", $n, a.shape());
                        }
                        a
                    }
                    Err(e) => {
                        eprintln!("[dbg-mhc] FAILED at {} : {}", $n, e);
                        return Err(anyhow::anyhow!("{e}"));
                    }
                }
            }};
        }

        let (s0, s1, s2) = (self.scale_vals[0], self.scale_vals[1], self.scale_vals[2]);
        let n = b * t;
        let col = |a: &Array, lo: usize, len: usize| -> lisa_mlx::error::Result<Array> {
            narrow(&a, 1, lo, len)?.reshape(&[(n) as i32, (len) as i32])
        };

        // The fused split/Sinkhorn kernel computes `pre`, `post`, and `comb`
        // together. Return its result directly; the op path below is fallback.
        {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            if let Some(split) = lisa_mlx::models::glm53::kernels::hc_split_sinkhorn(
                &mixes,
                &self.scale,
                &self.base,
                self.eps,
                self.sinkhorn_iters,
                &stream,
            ) {
                // pre = split[.., 0:4], post = split[.., 4:8],
                // comb = split[.., 8:24] (row-major [source][destination]).
                let colx = |lo: i32, len: i32| -> anyhow::Result<Array> {
                    lisa_mlx::ops::indexing::IndexOp::index(
                        &split,
                        (lisa_mlx::ops::indexing::Ellipsis, lo..lo + len),
                    )
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("{e}"))
                };
                let pre =
                    colx(0, self.hc_mult as i32)?.reshape(&[(b) as i32, (t) as i32, (c) as i32])?;
                let post = colx(self.hc_mult as i32, self.hc_mult as i32)?.reshape(&[
                    (b) as i32,
                    (t) as i32,
                    (c) as i32,
                ])?;
                let comb = colx(
                    2 * self.hc_mult as i32,
                    (self.hc_mult * self.hc_mult) as i32,
                )?
                .reshape(&[(b) as i32, (t) as i32, (c) as i32, (c) as i32])?;
                // Collapse: xc = sum_c pre[c] * state[c] (hc_weighted_sum).
                let pre_bt = pre.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;
                let state_c = state.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;
                let xc: Array = {
                    match lisa_mlx::models::glm53::kernels::hc_weighted_sum(
                        &state_c, &pre_bt, &stream,
                    ) {
                        Some(x) => x,
                        None => {
                            let pre4 = pre_bt.expand_dims(3)?;
                            let prod = state_c
                                .as_dtype(Dtype::Float32)?
                                .multiply(&pre4)?
                                .contiguous()?;
                            let mut acc = narrow(&prod, 2, 0, 1)?.contiguous()?;
                            for st in 1..c {
                                acc = acc.add(&narrow(&prod, 2, st, 1)?.contiguous()?)?;
                            }
                            acc.reshape(&[(b) as i32, (t) as i32, (h) as i32])?
                        }
                    }
                };
                return Ok((xc, post, comb));
            }
        }
        let pre = dstep!("pre-col", col(&mixes, 0, self.hc_mult)?)
            .multiply(&fscalar(&mixes, s0)?)?
            .add(&narrow(&self.base, 0, 0, self.hc_mult)?.as_dtype(Dtype::Float32)?)?;
        let pre = pre.sigmoid()?.add(&fscalar(&pre, self.eps)?)?; // [N, C]
        let post = col(&mixes, self.hc_mult, self.hc_mult)?
            .multiply(&fscalar(&mixes, s1)?)?
            .add(&narrow(&self.base, 0, self.hc_mult, self.hc_mult)?.as_dtype(Dtype::Float32)?)?;
        let post = post.sigmoid()?.multiply(&fscalar(&post, 2.0)?)?; // [N, C]

        // comb: mixes[..8:24] -> [N, C, C], scale + base, softmax(-1), +eps,
        // col-normalize, then sinkhorn iterations.
        let mix_c = col(&mixes, 2 * self.hc_mult, self.hc_mult * self.hc_mult)?.reshape(&[
            (n) as i32,
            (self.hc_mult) as i32,
            (self.hc_mult) as i32,
        ])?;
        let base_c = narrow(&self.base, 0, 2 * self.hc_mult, self.hc_mult * self.hc_mult)?
            .as_dtype(Dtype::Float32)?
            .reshape(&[(self.hc_mult) as i32, (self.hc_mult) as i32])?;
        let bc = base_c.expand_dims(0)?.broadcast_to(&[
            (n) as i32,
            (self.hc_mult) as i32,
            (self.hc_mult) as i32,
        ])?;
        let mut comb = mix_c.multiply(&fscalar(&mix_c, s2)?)?.add(&bc)?;
        comb = comb.softmax_axis(-1)?;
        comb = comb.add(&fscalar(&comb, self.eps)?)?;
        // comb is [N, C, C]: last axis = 2, second-to-last = 1.
        comb = normalize_axis(&comb, 2, self.eps)?; // / sum(-1)
        comb = normalize_axis(&comb, 1, self.eps)?; // / sum(-2)
        for _ in 0..self.sinkhorn_iters.saturating_sub(1) {
            comb = normalize_axis(&comb, 2, self.eps)?;
            comb = normalize_axis(&comb, 1, self.eps)?;
        }

        // Collapse: xc = sum_c pre[c] * state[c]. Prefer the fused weighted
        // sum kernel; the op chain below remains the fallback.
        let pre_bt = pre
            .reshape(&[(b) as i32, (t) as i32, (c) as i32])?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let state_c = state.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;
        let xc: Array = {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            match lisa_mlx::models::glm53::kernels::hc_weighted_sum(&state_c, &pre_bt, &stream) {
                Some(x) => x,
                None => {
                    let pre4 = pre_bt.expand_dims(3)?; // [B,T,C,1]
                    let prod = state_c
                        .as_dtype(Dtype::Float32)?
                        .multiply(&pre4)?
                        .contiguous()?; // [B,T,C,H]
                    // sum over the stream axis (2): loop the 4 streams (wrapper-safe).
                    let mut acc = narrow(&prod, 2, 0, 1)?.contiguous()?;
                    for st in 1..c {
                        acc = acc.add(&narrow(&prod, 2, st, 1)?.contiguous()?)?;
                    }
                    acc.reshape(&[(b) as i32, (t) as i32, (h) as i32])?
                }
            }
        };

        Ok((xc, post, comb))
    }

    /// Expand: `new[c,d] = post[c]*y[d] + sum_p comb[p,c]*state[p,d]`.
    /// Prefer the fused kernel and keep the op path as fallback.
    pub fn expand(
        &self,
        y: &Array,        // [B,T,H] f32
        residual: &Array, // [B,T,C,H] f32
        post: &Array,     // [B,T,C] f32
        comb: &Array,     // [B,T,C,C] f32
    ) -> anyhow::Result<Array> {
        let c = self.hc_mult;
        let h = y.dim((y.rank() - 1) as i32);
        {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let yc = y.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;
            let rc = residual.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;
            let pc = post.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;
            let cc = comb.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;
            if let Some(fused) =
                lisa_mlx::models::glm53::kernels::hc_expand4(&yc, &rc, &pc, &cc, &stream)
            {
                return Ok(fused);
            }
        }
        anyhow::ensure!(c != 4 || h != 4096, "GLM HC=4 expand Metal kernel failed");
        let (b, t, h) = (y.dim(0), y.dim(1), y.dim(2));
        let c = self.hc_mult;
        let yf = y.as_dtype(Dtype::Float32)?;
        let rf = residual.as_dtype(Dtype::Float32)?;
        // post * y, broadcast over C.
        let mut new = post
            .expand_dims(3)? // [B,T,C,1]
            .multiply(&yf.expand_dims(2)?.broadcast_to(&[
                (b) as i32,
                (t) as i32,
                (c) as i32,
                (h) as i32,
            ])?)?; // [B,T,C,H]
        // sum_p comb[p, c] * residual[p] — p is the first comb axis. Slice
        // that source row and transpose it into the output-stream broadcast.
        for p in 0..c {
            let coef = narrow(&comb, 2, p, 1)?.transpose_axes(&[0, 1, 3, 2])?;
            let res_p = narrow(&rf, 2, p, 1)?; // [B,T,1,H]
            new = new.add(&coef.multiply(&res_p)?)?; // [B,T,C,H]
        }
        Ok(new)
    }
}

/// Normalize along the SECOND-TO-LAST axis (comb [N, C, C]: sum over p).
/// Implemented via transpose -> last-axis sum -> transpose back (the wrapper
/// only reliably reduces the last axis).
fn normalize_axis(a: &Array, axis: usize, eps: f32) -> anyhow::Result<Array> {
    match axis {
        2 => {
            // last axis: direct.
            let a = a.contiguous()?;
            let s = a.sum_axis(2, true)?.add(&fscalar(&a, eps)?)?;
            Ok(a.divide(&s)?)
        }
        1 => {
            let at = a.transpose_axes(&[0, 2, 1])?.contiguous()?; // [N, c, p]
            let s = at.sum_axis(2, true)?.add(&fscalar(&at, eps)?)?; // [N, c, 1]
            let atn = at.divide(&s)?; // [N, c, p]
            Ok(atn.transpose_axes(&[0, 2, 1])?) // back to [N, p, c]
        }
        _ => anyhow::bail!("normalize_axis: unsupported axis {axis}"),
    }
}

#[cfg(test)]
#[path = "mhc_tests.rs"]
mod tests;
