//! `lisa_mlx::ops`-shaped free functions.
use super::*;

/// mlx exposes indexing under `ops::indexing`.
pub use super::indexing;

pub fn exp(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().exp()
}
pub fn log(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().log()
}
pub fn sqrt(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().sqrt()
}
pub fn rsqrt(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().rsqrt()
}
pub fn sin(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().sin()
}
pub fn cos(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().cos()
}
pub fn log1p(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().log1p()
}
pub fn argsort(a: &Array) -> Result<Array> {
    Ok(Array::new(crate::jit::argsort_last(a.t.device(), &a.t)?))
}
pub fn tile(a: &Array, reps: &[i32]) -> Result<Array> {
    let r: Vec<usize> = reps.iter().map(|&v| v as usize).collect();
    a.tile(&r)
}

/// mlx `ops::dequantize` (affine): `w = q * scale + bias` per group.
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    // GPU path (MLX `affine_dequantize`): the host implementation is a full
    // GPU->CPU->GPU round trip and the embedding forward calls this per
    // token.
    return Ok(Array::new(crate::jit::affine_dequantize(
        w.t.device(),
        &w.t,
        &scales.t,
        &biases.t,
        group_size,
        bits,
    )?));
    #[allow(unreachable_code)]
    let gs = group_size as usize;
    let bits = bits as usize;
    let dims: Vec<usize> = w.shape().iter().map(|&d| d as usize).collect();
    let kq = *dims.last().unwrap();
    let per_word = 32 / bits;
    let k = kq * per_word;
    let rows = w.size() / kq;
    let groups = k / gs;
    let q = w.t.cast(Dtype::Uint32)?.to_vec::<u32>()?;
    let sc: Vec<f32> = scales.t.cast(Dtype::Float32)?.to_vec::<f32>()?;
    let bi: Vec<f32> = biases.t.cast(Dtype::Float32)?.to_vec::<f32>()?;
    let mut out = vec![0f32; rows * k];
    let mask = (1u32 << bits) - 1;
    for r in 0..rows {
        for c in 0..k {
            let word = q[r * kq + c / per_word];
            let qv = (word >> ((c % per_word) * bits)) & mask;
            let g = c / gs;
            out[r * k + c] = qv as f32 * sc[r * groups + g] + bi[r * groups + g];
        }
    }
    let mut shape = dims.clone();
    shape.pop();
    shape.push(k);
    let rt = Stream::thread_local_or_default().runtime().clone();
    let t = crate::array::Array::from_slice_dt(&rt, &out, &shape, Dtype::Float32)?;
    // MLX returns the scales' dtype (the original weight type), not the packed type.
    Ok(Array::new(t.to_dtype(scales.t.dtype())?))
}

/// mlx `ops::conv1d`. Only the depthwise (`groups == C_in == C_out`,
/// one input channel per group) case the GDN fallback uses is implemented.
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: Option<i32>,
    padding: Option<(i32, i32)>,
    dilation: Option<i32>,
    groups: Option<i32>,
) -> Result<Array> {
    let ish = input.shape().to_vec();
    let wsh = weight.shape().to_vec();
    let (n, l, cin) = (ish[0] as usize, ish[1] as usize, ish[2] as usize);
    let (cout, k) = (wsh[0] as usize, wsh[1] as usize);
    let g = groups.unwrap_or(1) as usize;
    let s = stride.unwrap_or(1) as usize;
    let dil = dilation.unwrap_or(1) as usize;
    let (pl, pu) = padding.unwrap_or((0, 0));
    if g != cin || cout != cin || wsh[2] != 1 {
        crate::bail!("conv1d: only depthwise (groups == C_in == C_out) is implemented");
    }
    let lp = l + pl as usize + pu as usize;
    let lout = if lp >= (k - 1) * dil + 1 {
        (lp - ((k - 1) * dil + 1)) / s + 1
    } else {
        0
    };
    let rt = Stream::thread_local_or_default().runtime().clone();
    let mut acc = crate::array::Array::zeros(
        &rt,
        &[n as usize, lout as usize, cin as usize],
        Dtype::Float32,
    )?;
    let xf = input.t.to_dtype(Dtype::Float32)?;
    let wf = weight.t.to_dtype(Dtype::Float32)?;
    for t in 0..k {
        let src_start = pl as usize + t * dil;
        let last = src_start + (lout.saturating_sub(1)) * s + 1;
        if lout == 0 || last > lp {
            continue;
        }
        let slice = xf.narrow(1, src_start, lout)?.contiguous()?;
        let wc = wf.narrow(1, t, 1)?.reshape(&[1, 1, cin as usize])?;
        acc = acc.broadcast_add(&slice.broadcast_mul(&wc)?)?;
    }
    Ok(Array::new(acc.to_dtype(input.t.dtype())?))
}
pub fn abs(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().abs()
}
pub fn sign(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().sign()
}
pub fn sigmoid(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().sigmoid()
}
pub fn multiply(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().multiply(b.as_ref())
}
pub fn add(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().add(b.as_ref())
}
pub fn divide(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().divide(b.as_ref())
}
pub fn maximum(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().maximum(b.as_ref())
}
pub fn minimum(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().minimum(b.as_ref())
}
pub fn softmax_axis(a: &Array, axis: i32, _precise: bool) -> Result<Array> {
    a.softmax_axis(axis)
}
pub fn matmul(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().matmul(b.as_ref())
}
pub fn logaddexp(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().logaddexp(b.as_ref())
}
pub fn is_nan(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().is_nan()
}
pub fn floor_divide(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().floor_divide(b.as_ref())
}
pub fn r#where(cond: &Array, a: &Array, b: &Array) -> Result<Array> {
    Array::where_(cond, a, b)
}
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    let rank = arrays.first().map(|a| a.t.rank()).unwrap_or(1);
    let inner: Vec<crate::array::Array> = arrays.iter().map(|a| a.t.clone()).collect();
    Ok(Array::new(crate::array::Array::cat(
        &inner,
        norm_axis(rank, axis),
    )?))
}
pub fn sum(a: &Array, axis: Option<&[i32]>) -> Result<Array> {
    a.sum(axis)
}
pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array> {
    a.broadcast_to(shape)
}

/// mlx `ops::repeat_axis` (owned array, `T` only for turbofish parity).
pub fn repeat_axis<T>(array: Array, count: i32, axis: i32) -> Result<Array> {
    let _ = std::marker::PhantomData::<T>;
    array.repeat_axis(count.max(0) as usize, axis)
}

/// mlx `ops::stack`.
pub fn stack(arrays: &[impl AsRef<Array>], axis: i32) -> Result<Array> {
    let rank = arrays.first().map(|a| a.as_ref().t.rank()).unwrap_or(0) + 1;
    let inner: Vec<crate::array::Array> = arrays.iter().map(|a| a.as_ref().t.clone()).collect();
    Ok(Array::new(crate::array::Array::stack(
        &inner,
        norm_axis(rank, axis),
    )?))
}

/// mlx `ops::full`: broadcast `values` to `shape`.
pub fn full<T>(shape: &[i32], values: impl AsRef<Array>) -> Result<Array> {
    let _ = std::marker::PhantomData::<T>;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let v = values.as_ref();
    Ok(Array::new(v.t.broadcast_as(&dims)?.contiguous()?))
}

/// mlx `ops::quantize` (affine). Returns `(wq, scales, biases)` with `wq`
/// packed `bits`-per-element into `u32` along the last axis, and per-group
/// `scales`/`biases` in the input dtype, matching mlx's affine scheme:
/// `scale = (max - min) / (2^bits - 1)`, `bias = min`.
pub fn quantize(
    w: &Array,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
) -> Result<(Array, Array, Array)> {
    let gs = group_size.into().unwrap_or(64) as usize;
    let bits = bits.into().unwrap_or(4) as usize;
    let dims: Vec<usize> = w.shape().iter().map(|&d| d as usize).collect();
    let k = *dims.last().unwrap();
    if k % gs != 0 || gs % (32 / bits) != 0 {
        crate::bail!("quantize: group size {gs} incompatible with K={k}, bits={bits}");
    }
    let vals = w.t.cast(Dtype::Float32)?.to_vec::<f32>()?;
    let rows = vals.len() / k;
    let groups = k / gs;
    let maxq = ((1u32 << bits) - 1) as f32;

    let mut wq = vec![0u32; rows * k * bits / 32];
    let mut sc = vec![0f32; rows * groups];
    let mut bi = vec![0f32; rows * groups];
    let per_word = 32 / bits;
    const EPS: f32 = 1e-7;
    for r in 0..rows {
        for g in 0..groups {
            let base = r * k + g * gs;
            // MLX `affine_quantize` (quantized.h): a signed scale and the
            // outer edge as bias, not (max-min)/bins with bias=min.
            let mut w_min = f32::MAX;
            let mut w_max = 0f32;
            for i in 0..gs {
                let v = vals[base + i];
                w_min = w_min.min(v);
                w_max = w_max.max(v);
            }
            let mut scale = ((w_max - w_min) / maxq).max(EPS);
            let side = w_min.abs() > w_max.abs();
            if !side {
                scale = -scale;
            }
            let edge = if side { w_min } else { w_max };
            let q0 = (edge / scale).round();
            let at_zero = q0 == 0.0;
            if !at_zero {
                scale = edge / q0;
            }
            let bias = if at_zero { 0.0 } else { edge };
            sc[r * groups + g] = scale;
            bi[r * groups + g] = bias;
            for i in 0..gs {
                let q = (((vals[base + i] - bias) / scale).round()).min(maxq) as i32;
                let word = base / per_word + i / per_word;
                let shift = (i % per_word) * bits;
                wq[word] |= ((q as u32) & ((1 << bits) - 1)) << shift;
            }
        }
    }
    let rt = Stream::thread_local_or_default().runtime().clone();
    let mut wq_shape: Vec<usize> = dims.clone();
    wq_shape.pop();
    wq_shape.push(k * bits / 32);
    let wq = crate::array::Array::from_slice_dt(&rt, &wq, &wq_shape, Dtype::Uint32)?;
    let mut ss_shape: Vec<usize> = dims.clone();
    ss_shape.pop();
    ss_shape.push(groups);
    let scales = crate::array::Array::from_slice_dt(&rt, &sc, &ss_shape, Dtype::Float32)?;
    let biases = crate::array::Array::from_slice_dt(&rt, &bi, &ss_shape, Dtype::Float32)?;
    let wdt = w.t.dtype();
    Ok((
        Array::new(wq),
        Array::new(scales.to_dtype(wdt)?),
        Array::new(biases.to_dtype(wdt)?),
    ))
}
pub fn zeros<T: ZeroElem>(shape: &[i32]) -> Result<Array> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(Array::new(T::zeros(
        &dims,
        &Stream::thread_local_or_default().runtime().clone(),
    )?))
}
/// mlx `ops::logical_not`.
pub fn logical_not(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().logical_not()
}
/// Only the last axis is supported (the ported MLX sort kernel is last-axis).
pub fn argpartition_axis(a: &Array, kth: i32, axis: i32) -> Result<Array> {
    let last = a.rank().saturating_sub(1) as i32;
    if axis != -1 && axis != last {
        crate::bail!("argpartition_axis: only last axis supported");
    }
    Ok(Array::new(crate::jit::argpartition_axis(
        a.t.device(),
        &a.t,
        kth,
    )?))
}

// Tail-ULP contract scope (AGENTS.md §9, MTP verify rows ONLY): when set,
// `quantized_matmul` routes the M 2..=7 split-K verify lane
// (`affine_verify_qmm_splitk`, maxdiff <= 1 bf16 ULP + argmax equality vs
// the qmv_wide chain) INSTEAD of qmv_wide. Thread-local because the scope
// must cover exactly the enqueue window of one verify forward + head on the
// calling thread — never the serial (M=1), prefill (large M), or B>1 batch
// paths, which keep their bit-exact dispatch.
thread_local! {
    static VERIFY_SPLITK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII guard for [`VERIFY_SPLITK`]: enables the split-K verify lane on this
/// thread until dropped.
pub struct VerifySplitkGuard {
    _priv: (),
}

impl Drop for VerifySplitkGuard {
    fn drop(&mut self) {
        VERIFY_SPLITK.with(|f| f.set(false));
    }
}

/// Enter the MTP-verify tail-ULP scope on this thread (AGENTS.md §9). Call
/// around the verify forward + head enqueue ONLY; the returned guard restores
/// the stock dispatch.
pub fn enter_verify_splitk_scope() -> VerifySplitkGuard {
    VERIFY_SPLITK.with(|f| f.set(true));
    VerifySplitkGuard { _priv: () }
}

fn verify_splitk_enabled() -> bool {
    VERIFY_SPLITK.with(std::cell::Cell::get)
}

/// mlx `quantized_matmul` (transpose=true, affine). The input is flattened
/// to 2-D `[M, K]`: `M == 1` takes the qmv kernel, `M > 1` the split-K qmm.
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    if !transpose {
        crate::bail!("quantized_matmul: only transpose=true supported");
    }
    let bi = biases.ok_or_else(|| {
        crate::error::Error::Msg("quantized_matmul: affine biases required".into())
    })?;
    let rank = x.rank();
    let k = x.dim(rank as i32 - 1) as usize;
    let m = x.size() / k;
    let n = w.dim(0) as usize;
    // mlx: out_type = promote_types(x, scales) and every input is cast to it.
    let dt = promote_dtype(x.t.dtype(), scales.t.dtype());
    let x2 = x.t.to_dtype(dt)?.reshape(&[m, k])?.contiguous()?;
    let sc = scales.t.to_dtype(dt)?.contiguous()?;
    let b2 = bi.t.to_dtype(dt)?.contiguous()?;
    if !matches!(dt, Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32) {
        crate::bail!("quantized_matmul: unsupported compute dtype {dt:?}");
    }
    let out = if m == 1 {
        // Serial split-K qmv (`jit::affine_qmv_splitk`, the reference's
        // split-K accumulation at MROWS=1) is FALSIFIED in situ: goldens
        // pass 310/310 with the lane wired, but the serial step is a wash
        // (paired interleaved 27B kv1024 audits, base/new/new/base:
        // 36.66/37.21 vs 36.75/36.20 ms median) and kp=4 on the wide-N
        // projections regresses hard (85-88 ms). Kernel-level opt-in, like
        // the msg/nax precedents — the stock serial dispatch stays qmv_fast.
        crate::jit::affine_qmv_fast(x.t.device(), &x2, &w.t, &sc, &b2, group_size, bits)?
    } else if m < crate::jit::qmv_batch_limit(k, n) {
        // MLX uses qmv_wide for 2 <= M < vector_limit (affine, gen>=15).
        // Tail-ULP contract exception (AGENTS.md §9, MTP verify rows only):
        // inside the verify_splitk scope, the M 2..=8 4-bit shapes take the
        // split-K verify tile instead (maxdiff <= 1 bf16 ULP + argmax
        // equality vs this qmv_wide chain, pinned by the dedicated test; M=8
        // is the widest MROWS that does not spill — specs/08 m16 campaign).
        // Outside the scope nothing changes.
        if verify_splitk_enabled() {
            match crate::jit::affine_verify_qmm_splitk(
                x.t.device(),
                &x2,
                &w.t,
                &sc,
                &b2,
                group_size,
                bits,
            ) {
                Ok(o) => o,
                // Ineligible shape inside the scope: fall through to stock.
                // Measured falsifications for the M >= 9 middle (specs/08
                // item 2): the split-K tile spills at MROWS >= 9 (isolated
                // 1500-4500 us/dis vs 180 at M=8); a two-tile pass conflicts
                // on the double weight stream (>= 1550 us/dis at every M);
                // qmm_nax flips an argmax at M=11 (contract-illegal); the
                // MPP m16 tile port does not execute under lisa's
                // newLibraryWithSource pipeline path (the minimal MPP probe
                // hangs — the additive-binary linkage the reference's
                // fast-kernel path uses is not wired here). Kept as
                // kernel-level opt-in (jit::affine_verify_qmm_nax_m16), like
                // the msg tile precedent.
                Err(_) => crate::jit::qmv_wide(x.t.device(), &x2, &w.t, &sc, &b2, group_size, bits)?,
            }
        } else {
            crate::jit::qmv_wide(x.t.device(), &x2, &w.t, &sc, &b2, group_size, bits)?
        }
    } else {
        // Split-K when the tiling supports it (A/B-verified); qmm_nax is
        // MLX's other non-split path when split_k would collapse to 1.
        match crate::jit::affine_qmm_splitk(x.t.device(), &x2, &w.t, &sc, &b2, group_size, bits) {
            Ok(o) => o,
            Err(_) if k % 64 == 0 && x.t.device().nax() => {
                // `qmm_nax` is the MPP fallback; without it, surface the
                // split-K error rather than compiling a NAX kernel.
                crate::jit::qmm_nax(x.t.device(), &x2, &w.t, &sc, &b2, group_size, bits)?
            }
            Err(e) => return Err(e),
        }
    };
    let mut shape: Vec<usize> = x.shape()[..rank - 1].iter().map(|&d| d as usize).collect();
    shape.push(n);
    Ok(Array::new(out.reshape_dims(shape)?))
}

/// mlx `gather_qmm` (affine, transpose). Only the right-sorted path the
/// engine's MoE uses is ported: `lhs_indices = None`, `sorted_indices =
/// true`. An `[rows, 1, K]` lhs yields `M = 1, B = rows`, which is exactly
/// the branch that maps to `gather_qmm_rhs_nax`.
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
) -> Result<Array> {
    if !transpose {
        crate::bail!("gather_qmm: only transpose=true supported");
    }
    if lhs_indices.is_some() {
        crate::bail!("gather_qmm: lhs_indices path not ported");
    }
    if !sorted_indices {
        crate::bail!("gather_qmm: only sorted_indices=true supported");
    }
    let rhs = rhs_indices
        .ok_or_else(|| crate::error::Error::Msg("gather_qmm: rhs_indices required".into()))?;
    let bi = biases
        .ok_or_else(|| crate::error::Error::Msg("gather_qmm: affine biases required".into()))?;
    let rank = x.rank();
    let n = w.dim(w.rank() as i32 - 2) as usize;
    let mut out_shape: Vec<usize> = x.shape()[..rank - 2].iter().map(|&d| d as usize).collect();
    out_shape.push(x.dim(rank as i32 - 2) as usize);
    out_shape.push(n);
    if !x.t.device().nax() {
        // Non-NAX: the rows are sorted by expert, so run the split-K
        // quantized GEMM once per contiguous expert group — the same math
        // without the MPP gather kernel.
        let ids = rhs.as_dtype(Dtype::Uint32)?;
        let idv = ids.as_slice::<u32>();
        let rows = x.dim(0) as usize;
        let mut pieces: Vec<Array> = Vec::new();
        let mut s = 0usize;
        while s < rows {
            let eid = idv[s];
            let mut e = s + 1;
            while e < rows && idv[e] == eid {
                e += 1;
            }
            let sel: Vec<i32> = (s as i32..e as i32).collect();
            let sel = Array::from_slice(&sel, &[(e - s) as i32]);
            let one = Array::from_slice(&[eid as i32], &[1]);
            let xr = x.take_axis(&sel, 0)?;
            let we = w.take_axis(&one, 0)?.reshape(&[w.dim(1), w.dim(2)])?;
            let se = scales
                .take_axis(&one, 0)?
                .reshape(&[scales.dim(1), scales.dim(2)])?;
            let be = bi.take_axis(&one, 0)?.reshape(&[bi.dim(1), bi.dim(2)])?;
            pieces.push(quantized_matmul(
                &xr,
                &we,
                &se,
                Some(&be),
                true,
                group_size,
                bits,
            )?);
            s = e;
        }
        let refs: Vec<&Array> = pieces.iter().collect();
        let cat = Array::concatenate(&refs, 0)?;
        let shape: Vec<i32> = out_shape.iter().map(|&d| d as i32).collect();
        return Ok(cat.reshape(&shape)?);
    }
    Ok(Array::new(crate::jit::gather_qmm_rhs_nax(
        x.t.device(),
        &x.t,
        &w.t,
        &scales.t,
        &bi.t,
        &rhs.t,
        out_shape,
        group_size,
        bits,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quantize;

    /// THE TAIL-ULP CONTRACT (AGENTS.md §9, MTP verify rows only): the
    /// split-K verify tile vs the qmv_wide chain it replaces —
    ///   1. maxdiff <= 1 bf16 ULP on EVERY output element, and
    ///   2. argmax equality over every verify row (the draft-acceptance
    ///      axis: verify argmax flips would change committed tokens),
    /// on the verify shapes S in {3,5,7} x the head dims 5120->17408 and the
    /// lm_head shape 5120->151936, over 3 independent draws per cell. Any
    /// case over budget fails with its max ULP printed. The head shape rides
    /// the msg lane (N >= 100k) inside the scope; the trunk shape rides
    /// splitk — both are pinned against the same qmv_wide reference chain.
    #[test]
    fn verify_qmm_tile_env_override_seam() {
        // The pure parser: in-range bn:kp accepted, malformed/out-of-range
        // refused (the runtime OnceLock resolves the env once per process —
        // not mutably testable in-process, hence the pure seam).
        use crate::jit::parse_tile_override as p;
        assert_eq!(p("1:4"), Some((1, 4)));
        assert_eq!(p("2:2"), Some((2, 2)));
        assert_eq!(p("8:32"), Some((8, 32)));
        assert_eq!(p("0:2"), None);
        assert_eq!(p("9:2"), None);
        assert_eq!(p("2:33"), None);
        assert_eq!(p("bogus"), None);
        assert_eq!(p("2"), None);
        assert_eq!(p("2:2:2"), None);
        // Unset / malformed env keeps the measured constants on the gate.
        assert_eq!(
            crate::jit::verify_qmm_splitk_lane(7, 5120, 17408, 4),
            Some((2, 2))
        );
    }

    #[test]
    fn verify_qmm_splitk_tailulp_vs_qmv_wide() {
        for (k, n) in [(5120usize, 17408usize), (5120usize, 151936usize)] {
            let s_list: &[usize] = if n >= 100_000 { &[3, 5, 7] } else { &[3, 5, 7, 8] };
            for s in s_list.iter().copied() {
                for draw in 0..3u64 {
                    let mut seed = 1000u64.wrapping_mul(draw + 1).wrapping_add(s as u64);
                    let mut rnd = move || {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 40) as f32 / 8_388_608.0) - 1.0
                    };
                    let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 2.0 - 1.0).collect();
                    let (wq, scales, biases) =
                        quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
                    let wq = wq.contiguous().unwrap();
                    // bf16 scales/biases so promote_dtype keeps the whole
                    // chain (and the output) bf16 — the runtime verify class.
                    let scales = scales
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let biases = biases
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let xv: Vec<f32> = (0..s * k).map(|_| rnd() * 4.0 - 2.0).collect();
                    let x = Array::from_slice(&xv, &[s as i32, k as i32])
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    // Reference: the stock dispatch chain (qmv_wide), taken
                    // OUTSIDE the scope.
                    let y_ref =
                        quantized_matmul(&x, &wq, &scales, Some(&biases), true, 64, 4).unwrap();
                    let _ = y_ref.eval();
                    // Candidate: the scoped dispatch (splitk on the trunk
                    // shape) — or the msg tile DIRECTLY on the lm_head shape
                    // (the msg lane is kernel-level opt-in only after the
                    // in-situ falsification; the contract pin stays).
                    let y_new = if n >= 100_000 {
                        Array::new(
                            crate::jit::affine_verify_qmm_msg(
                                x.t.device(),
                                &x.t,
                                &wq.t,
                                &scales.t,
                                &biases.t,
                                64,
                                4,
                            )
                            .unwrap(),
                        )
                    } else {
                        let _guard = enter_verify_splitk_scope();
                        quantized_matmul(&x, &wq, &scales, Some(&biases), true, 64, 4).unwrap()
                    };
                    let _ = y_new.eval();
                    let a = y_ref.as_slice::<u16>().to_vec();
                    let b = y_new.as_slice::<u16>().to_vec();
                    assert_eq!(a.len(), s * n);
                    assert_eq!(b.len(), s * n);
                    // bf16 ULP distance via the sign-magnitude -> biased-int map.
                    let key = |u: u16| -> i32 {
                        let mag = (u & 0x7fff) as i32;
                        if u & 0x8000 != 0 { -mag } else { mag }
                    };
                    let mut max_ulp = 0i32;
                    let mut argmax_bad = 0usize;
                    for r in 0..s {
                        let (mut max_r, mut ia, mut ib) = (0i32, 0usize, 0usize);
                        let (mut va, mut vb) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                        for c in 0..n {
                            let (pa, pb) = (a[r * n + c], b[r * n + c]);
                            max_r = max_r.max((key(pa) - key(pb)).abs());
                            // bf16 bits -> f32.
                            let fa = f32::from_bits((pa as u32) << 16);
                            let fb = f32::from_bits((pb as u32) << 16);
                            if fa > va {
                                va = fa;
                                ia = c;
                            }
                            if fb > vb {
                                vb = fb;
                                ib = c;
                            }
                        }
                        if ia != ib {
                            argmax_bad += 1;
                        }
                        max_ulp = max_ulp.max(max_r);
                    }
                    assert!(
                        max_ulp <= 1,
                        "TAIL-ULP FAIL k={k} n={n} s={s} draw={draw}: maxdiff {max_ulp} ULP > 1"
                    );
                    assert!(
                        argmax_bad == 0,
                        "ARGMAX FAIL k={k} n={n} s={s} draw={draw}: {argmax_bad} rows flipped"
                    );
                    println!(
                        "tail-ULP k={k} n={n} s={s} draw={draw}: maxdiff {max_ulp} ULP, argmax equal ({s} rows)"
                    );
                }
            }
        }
        // lm_head class: N = 151936 >= 100000 routes to the msg lane — the
        // splitk lane must still DECLINE there, and the msg lane must take it.
        assert_eq!(
            crate::jit::verify_qmm_splitk_lane(7, 5120, 151936, 4),
            None,
            "lm_head (N>=100k) must stay outside the splitk lane (msg lane takes it)"
        );
        assert_eq!(
            crate::jit::verify_qmm_msg_lane(7, 5120, 151936, 4),
            Some(2),
            "lm_head (N>=100k) must ride the msg lane at M=7 (BN=2)"
        );
        assert_eq!(
            crate::jit::verify_qmm_msg_lane(5, 5120, 151936, 4),
            Some(4),
            "msg BN: 4 through M=6"
        );
        assert_eq!(crate::jit::verify_qmm_msg_lane(7, 5120, 17408, 4), None);
    }

    /// The m16 lane gates (specs/08): M=8 rides the split-K tile; M 9..=16
    /// stays on the stock chain — every wider candidate measured falsified
    /// (splitk spills, two-tile conflicts, qmm_nax argmax-illegal, MPP tile
    /// inert under newLibraryWithSource), so there is NO scoped dispatch to
    /// pin beyond M=8 (pinned by the splitk test above, s=8).
    #[test]
    fn verify_qmm_m16_lane_gates() {
        assert_eq!(crate::jit::verify_qmm_splitk_lane(8, 5120, 17408, 4), Some((2, 2)));
        assert_eq!(crate::jit::verify_qmm_splitk_lane(9, 5120, 17408, 4), None);
        assert_eq!(crate::jit::verify_qmm_nax_m16_lane(9, 5120, 17408, 4, 64), true);
        // M=8 is eligible for BOTH (splitk wins by dispatch order); the gate
        // itself accepts 8..=16.
        assert_eq!(crate::jit::verify_qmm_nax_m16_lane(8, 5120, 17408, 4, 64), true);
        assert_eq!(crate::jit::verify_qmm_nax_m16_lane(16, 5120, 17408, 4, 64), true);
        assert_eq!(crate::jit::verify_qmm_nax_m16_lane(9, 5120, 17408, 4, 32), true);
        assert_eq!(crate::jit::verify_qmm_nax_m16_lane(9, 5104, 17408, 4, 64), false, "K % 128");
        assert_eq!(crate::jit::verify_qmm_nax_m16_lane(9, 5120, 151936, 4, 64), false, "huge-N");
    }

    /// TAIL-ULP PINS for the splitk variant axis (VLOAD uint4 loads /
    /// Math::Fast compile): each variant vs the stock qmv_wide chain on the
    /// trunk verify shape, S in {3,7}, 2 draws — maxdiff <= 1 bf16 ULP +
    /// argmax equality, same contract as the stock splitk pin above.
    #[test]
    fn verify_qmm_splitk_variants_tailulp_vs_qmv_wide() {
        let (k, n) = (5120usize, 17408usize);
        for (vload, fast_math) in [(true, false), (false, true), (true, true)] {
            for s in [3usize, 7] {
                for draw in 0..2u64 {
                    let mut seed = 7000u64
                        .wrapping_mul(draw + 1)
                        .wrapping_add(s as u64)
                        .wrapping_add(vload as u64)
                        .wrapping_add(fast_math as u64 * 2);
                    let mut rnd = move || {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 40) as f32 / 8_388_608.0) - 1.0
                    };
                    let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 2.0 - 1.0).collect();
                    let (wq, scales, biases) =
                        quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
                    let wq = wq.contiguous().unwrap();
                    let scales = scales
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let biases = biases
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let xv: Vec<f32> = (0..s * k).map(|_| rnd() * 4.0 - 2.0).collect();
                    let x = Array::from_slice(&xv, &[s as i32, k as i32])
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let y_ref =
                        quantized_matmul(&x, &wq, &scales, Some(&biases), true, 64, 4).unwrap();
                    let _ = y_ref.eval();
                    let y_new = Array::new(
                        crate::jit::affine_verify_qmm_splitk_tile_ex(
                            x.t.device(),
                            &x.t,
                            &wq.t,
                            &scales.t,
                            &biases.t,
                            64,
                            4,
                            2,
                            2,
                            1,
                            vload,
                            fast_math,
                        )
                        .unwrap(),
                    );
                    let _ = y_new.eval();
                    let a = y_ref.as_slice::<u16>().to_vec();
                    let b = y_new.as_slice::<u16>().to_vec();
                    let key = |u: u16| -> i32 {
                        let mag = (u & 0x7fff) as i32;
                        if u & 0x8000 != 0 { -mag } else { mag }
                    };
                    let mut max_ulp = 0i32;
                    let mut argmax_bad = 0usize;
                    for r in 0..s {
                        let (mut max_r, mut ia, mut ib) = (0i32, 0usize, 0usize);
                        let (mut va, mut vb) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                        for c in 0..n {
                            let (pa, pb) = (a[r * n + c], b[r * n + c]);
                            max_r = max_r.max((key(pa) - key(pb)).abs());
                            let fa = f32::from_bits((pa as u32) << 16);
                            let fb = f32::from_bits((pb as u32) << 16);
                            if fa > va {
                                va = fa;
                                ia = c;
                            }
                            if fb > vb {
                                vb = fb;
                                ib = c;
                            }
                        }
                        if ia != ib {
                            argmax_bad += 1;
                        }
                        max_ulp = max_ulp.max(max_r);
                    }
                    assert!(
                        max_ulp <= 1 && argmax_bad == 0,
                        "TAIL-ULP FAIL vl={vload} fm={fast_math} s={s} draw={draw}: maxdiff {max_ulp} ULP, {argmax_bad} argmax flips"
                    );
                    println!(
                        "tail-ULP vl={vload} fm={fast_math} s={s} draw={draw}: maxdiff {max_ulp} ULP, argmax equal"
                    );
                }
            }
        }
    }

    /// Q|GATE MERGE (specs/16 phase 2): the decode attention arm replaces the
    /// per-head [q|gate] interleave + two `.contiguous()` copies with ONE qmv
    /// over row-gathered [Q|G] weights and two raw slice views. Two exactness
    /// pins: (1) qmv over the gathered packed weights must equal the gather of
    /// the interleaved qmv, word-for-word (row independence of the GEMV);
    /// (2) the fused kernels reading the SLICE VIEWS (buffer-offset bindings)
    /// must read the same words as the contiguous copies.
    #[test]
    fn qgate_rowgather_qmv_bitexact_vs_interleaved() {
        let k = 5120usize;
        let qn = 16384usize; // heads*hd half-width
        let n = 2 * qn;
        let hd = 256usize;
        let mut seed = 97531u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        // Interleaved weight rows: per head [q rows | gate rows] (hd-wide).
        let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 2.0 - 1.0).collect();
        let (wq, scales, biases) =
            quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
        let wq = wq.contiguous().unwrap();
        let scales = scales
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
            .contiguous()
            .unwrap();
        let biases = biases
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
            .contiguous()
            .unwrap();
        // Gather index: all q rows (head*2*hd + j), then all gate rows.
        let mut idx: Vec<u32> = Vec::with_capacity(n);
        for h in 0..(qn / hd) {
            for j in 0..hd {
                idx.push((h * 2 * hd + j) as u32);
            }
        }
        for h in 0..(qn / hd) {
            for j in 0..hd {
                idx.push((h * 2 * hd + hd + j) as u32);
            }
        }
        let idx = Array::from_slice(&idx, &[n as i32]);
        let wq_g = wq.take_axis(&idx, 0).unwrap().contiguous().unwrap();
        let sc_g = scales.take_axis(&idx, 0).unwrap().contiguous().unwrap();
        let bi_g = biases.take_axis(&idx, 0).unwrap().contiguous().unwrap();
        // Two draws -> >= 65k compared words.
        for draw in 0..2u64 {
            let xv: Vec<f32> = (0..k).map(|_| rnd() * 4.0 - 2.0).collect();
            let x = Array::from_slice(&xv, &[1i32, k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let y_ref =
                quantized_matmul(&x, &wq, &scales, Some(&biases), true, 64, 4).unwrap();
            // y_ref is [1, N]: flatten before gathering output ROWS.
            let y_gath_ref = y_ref
                .reshape(&[n as i32])
                .unwrap()
                .take_axis(&idx, 0)
                .unwrap()
                .contiguous()
                .unwrap();
            let y_merged =
                quantized_matmul(&x, &wq_g, &sc_g, Some(&bi_g), true, 64, 4).unwrap();
            let a = y_merged.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            let b = y_gath_ref
                .as_dtype(Dtype::Float32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let bad: Vec<usize> = (0..n).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
            assert!(
                bad.is_empty(),
                "draw {draw}: {} mismatched words, first at {}",
                bad.len(),
                bad.first().copied().unwrap_or(0)
            );
        }
        println!("qgate_rowgather: 2 draws x {n} words bit-exact");
    }

    /// The merged decode arm hands SLICE VIEWS of the packed [Q|G] row to the
    /// fused kernels (bindings honor the buffer offset). Pin that the kernels
    /// read exactly the words the contiguous copies would have produced:
    /// sigmoid_mul on the offset gate view and qk_norm_rope on the offset-0
    /// q view, both vs the contiguous-input results, word-for-word.
    #[test]
    fn qgate_offset_slice_views_bitexact_vs_contiguous() {
        use crate::ops::indexing::IndexOp;
        let stream = Stream::thread_local_or_default();
        let mut seed = 24680u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        // --- sigmoid_mul on the offset gate view (2 x 32768 words) ---
        for draw in 0..2u64 {
            let m = 32768usize;
            let yv: Vec<f32> = (0..2 * m).map(|_| rnd() * 40.0 - 20.0).collect();
            let y = Array::from_slice(&yv, &[1i32, (2 * m) as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let av: Vec<f32> = (0..m).map(|_| rnd() * 4.0 - 2.0).collect();
            let a = Array::from_slice(&av, &[1i32, m as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let gate_view = y.index((.., (m as i32)..(2 * m as i32))).reshape(&[1, 1, 1, m as i32]).unwrap();
            let gate_copy = gate_view.contiguous().unwrap();
            let a4 = a.reshape(&[1, 1, 1, m as i32]).unwrap();
            let f_view = crate::models::qwen35::kernels::sigmoid_mul(&gate_view, &a4, &stream)
                .expect("kernel");
            let f_copy = crate::models::qwen35::kernels::sigmoid_mul(&gate_copy, &a4, &stream)
                .expect("kernel");
            let v = f_view.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            let c = f_copy.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            let bad: Vec<usize> = (0..m).filter(|&i| v[i].to_bits() != c[i].to_bits()).collect();
            assert!(
                bad.is_empty(),
                "sigmoid_mul draw {draw}: {} mismatched words",
                bad.len()
            );
        }
        // --- qk_norm_rope on the offset-0 q view (real attn shape) ---
        let (hq, hk, hd) = (24i32, 4i32, 256i32);
        let qn = (hq * hd) as usize;
        let eps = 1e-6f32;
        let mk = |nn: usize, r: &mut dyn FnMut() -> f32| {
            let v: Vec<f32> = (0..nn).map(|_| r()).collect();
            Array::from_slice(&v, &[nn as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let y = mk(2 * qn, &mut rnd); // packed [Q|G] row
        let krow = mk((hk * hd) as usize, &mut rnd);
        let qw = mk(qn, &mut rnd);
        let kw = mk((hk * hd) as usize, &mut rnd);
        let cosv = mk(64, &mut rnd);
        let sinv = mk(64, &mut rnd);
        let q_view = y
            .index(0..(qn as i32))
            .reshape(&[1i32, qn as i32])
            .unwrap();
        let q_copy = q_view.contiguous().unwrap();
        let (oq_v, ok_v) = crate::models::qwen35::kernels::qk_norm_rope(
            &q_view, &krow, &qw, &kw, &cosv, &sinv, eps, hq, hk, &stream,
        )
        .expect("kernel");
        let (oq_c, ok_c) = crate::models::qwen35::kernels::qk_norm_rope(
            &q_copy, &krow, &qw, &kw, &cosv, &sinv, eps, hq, hk, &stream,
        )
        .expect("kernel");
        let vv = oq_v.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
        let cc = oq_c.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
        let bad: Vec<usize> = (0..vv.len()).filter(|&i| vv[i].to_bits() != cc[i].to_bits()).collect();
        assert!(bad.is_empty(), "oq: {} mismatched words", bad.len());
        let vv = ok_v.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
        let cc = ok_c.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
        let bad: Vec<usize> = (0..vv.len()).filter(|&i| vv[i].to_bits() != cc[i].to_bits()).collect();
        assert!(bad.is_empty(), "ok: {} mismatched words", bad.len());
        println!("qgate_offset_views: sigmoid_mul 2x32768 + qk_norm_rope bit-exact");
    }

    /// NORME-DANS-QMV (specs/16 phase 1): the mega-kernel must be
    /// word-for-word identical to the chain it replaces
    /// (`fast::fused_add_rms_norm` + `quantized_matmul` M=1) on the real
    /// decode shapes — the 128 rms sites: post-norm → gate|up (N=34816),
    /// GDN in_proj_all (N≈11k) and the narrower attention projections.
    /// Compares y, sum AND normed bit-for-bit, in both write_norm arms, on
    /// the looped-rms reduction (axis 5120 > 4096).
    #[test]
    fn qmv_addnorm_bitexact_vs_fused_chain() {
        let eps = 1e-6f32;
        for (k, n) in [
            (5120usize, 34816usize), // post-norm -> gate|up
            (5120, 11264),           // GDN in_proj_all-class width
            (5120, 8192),            // attention-class width
            (5120, 151936),          // lm_head-class width (documented: unfused)
        ] {
            let mut seed = 4242u64 + k as u64 * 31 + n as u64;
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 2.0 - 1.0).collect();
            let (wq, scales, biases) =
                quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
            let wq = wq.contiguous().unwrap();
            let scales = scales
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let biases = biases
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let nwv: Vec<f32> = (0..k).map(|_| 0.5 + rnd()).collect();
            let nw = Array::from_slice(&nwv, &[k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            // Two draws per shape (>= 65k compared values across the sweep).
            for draw in 0..2u64 {
                let xv: Vec<f32> = (0..k).map(|_| rnd() * 4.0 - 2.0).collect();
                let rv: Vec<f32> = (0..k).map(|_| rnd() * 4.0 - 2.0).collect();
                let x = Array::from_slice(&xv, &[1i32, k as i32])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let r = Array::from_slice(&rv, &[1i32, k as i32])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                // Reference chain: fused_add_rms_norm THEN the stock qmv
                // (also arms the looped-rms tgs record the prologue reads).
                let (sum_ref, normed_ref) =
                    crate::ops::fast::fused_add_rms_norm(&x, &r, &nw, eps).unwrap();
                let y_ref =
                    quantized_matmul(&normed_ref, &wq, &scales, Some(&biases), true, 64, 4)
                        .unwrap();
                let _ = y_ref.eval();
                let _ = sum_ref.eval();
                let _ = normed_ref.eval();
                for write_norm in [false, true] {
                    let (y_new, sum_new, normed_new) =
                        crate::jit::affine_qmv_fast_addnorm(
                            x.t.device(),
                            &x.t,
                            &r.t,
                            &nw.t,
                            &wq.t,
                            &scales.t,
                            &biases.t,
                            eps,
                            64,
                            4,
                            write_norm,
                        )
                        .unwrap();
                    let _ = y_new.eval();
                    let _ = sum_new.eval();
                    let a = y_ref.as_slice::<u16>();
                    let b = y_new.as_slice::<u16>();
                    let bad: Vec<usize> = a
                        .iter()
                        .zip(b.iter())
                        .enumerate()
                        .filter_map(|(i, (p, q))| (p != q).then_some(i))
                        .collect();
                    let sa = sum_ref.as_slice::<u16>();
                    let sb = sum_new.as_slice::<u16>();
                    assert!(
                        bad.is_empty(),
                        "k={k} n={n} draw={draw} wn={write_norm}: {} y bit mismatches, first at {} (ref {:04x} new {:04x})",
                        bad.len(),
                        bad[0],
                        a[bad[0]],
                        b[bad[0]]
                    );
                    assert!(
                        sa.iter().zip(sb.iter()).all(|(p, q)| p == q),
                        "k={k} n={n} draw={draw} wn={write_norm}: sum mismatch"
                    );
                    if write_norm {
                        let na = normed_ref.as_slice::<u16>();
                        let nn_arr = normed_new.unwrap();
                        let nb = nn_arr.as_slice::<u16>();
                        let nbad: Vec<usize> = na
                            .iter()
                            .zip(nb.iter())
                            .enumerate()
                            .filter_map(|(i, (p, q))| (p != q).then_some(i))
                            .collect();
                        assert!(
                            nbad.is_empty(),
                            "k={k} n={n} draw={draw}: {} normed bit mismatches, first at {} (ref {:04x} new {:04x})",
                            nbad.len(),
                            nbad[0],
                            na[nbad[0]],
                            nb[nbad[0]]
                        );
                    }
                    println!("addnorm bit-exact: k={k} n={n} draw={draw} wn={write_norm} OK ({n} y + {k} sum words)");
                }
            }
        }
    }

    /// NORME-DANS-QMV tail-ULP (specs/16 §7, campaign close phase 1): the
    /// §9.7-contract variant must stay within 1 bf16 ULP + argmax equality
    /// vs the stock chain (`fast::fused_add_rms_norm` + `quantized_matmul`
    /// M=1) on the real decode shapes. sum is pinned BIT-exact (s = bf16(x+r)
    /// is the same elementwise chain; only the reduction order differs).
    /// Normed is allowed <= 1 ULP (different inv rounding); y inherits the
    /// normed drift through the GEMV.
    #[test]
    fn qmv_addnorm_tailulp_vs_fused_chain() {
        // bf16 sign-magnitude bits -> ordered integer (ULP distance metric).
        let ordered = |b: u16| -> i32 {
            if b & 0x8000 != 0 {
                -((b & 0x7fff) as i32)
            } else {
                b as i32
            }
        };
        let eps = 1e-6f32;
        for (k, n) in [
            (5120usize, 34816usize), // post-norm -> gate|up
            (5120, 11264),           // GDN in_proj_all-class width
            (5120, 8192),            // attention-class width
        ] {
            let mut seed = 919u64 + k as u64 * 17 + n as u64;
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 2.0 - 1.0).collect();
            let (wq, scales, biases) =
                quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
            let wq = wq.contiguous().unwrap();
            let scales = scales
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let biases = biases
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let nwv: Vec<f32> = (0..k).map(|_| 0.5 + rnd()).collect();
            let nw = Array::from_slice(&nwv, &[k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            for draw in 0..3u64 {
                let xv: Vec<f32> = (0..k).map(|_| rnd() * 4.0 - 2.0).collect();
                let rv: Vec<f32> = (0..k).map(|_| rnd() * 4.0 - 2.0).collect();
                let x = Array::from_slice(&xv, &[1i32, k as i32])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let r = Array::from_slice(&rv, &[1i32, k as i32])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let (sum_ref, normed_ref) =
                    crate::ops::fast::fused_add_rms_norm(&x, &r, &nw, eps).unwrap();
                let y_ref =
                    quantized_matmul(&normed_ref, &wq, &scales, Some(&biases), true, 64, 4)
                        .unwrap();
                let _ = y_ref.eval();
                let _ = normed_ref.eval();
                for write_norm in [false, true] {
                    let (y_new, sum_new, normed_new) =
                        crate::jit::affine_qmv_fast_addnorm_tu(
                            x.t.device(),
                            &x.t,
                            &r.t,
                            &nw.t,
                            &wq.t,
                            &scales.t,
                            &biases.t,
                            eps,
                            64,
                            4,
                            write_norm,
                        )
                        .unwrap();
                    let _ = y_new.eval();
                    let _ = sum_new.eval();
                    // sum: bit-exact (elementwise chain, no reduction).
                    let sa = sum_ref.as_slice::<u16>();
                    let sb = sum_new.as_slice::<u16>();
                    assert!(
                        sa.iter().zip(sb.iter()).all(|(p, q)| p == q),
                        "k={k} n={n} draw={draw} wn={write_norm}: sum mismatch"
                    );
                    // y: maxdiff <= 1 bf16 ULP + argmax equal.
                    let a = y_ref.as_slice::<u16>();
                    let b = y_new.as_slice::<u16>();
                    let mut maxd = 0i32;
                    let mut first_bad: Option<(usize, u16, u16)> = None;
                    for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                        let d = (ordered(*p) - ordered(*q)).abs();
                        if d > maxd {
                            maxd = d;
                        }
                        if d > 1 && first_bad.is_none() {
                            first_bad = Some((i, *p, *q));
                        }
                    }
                    let (ia, _) = a
                        .iter()
                        .enumerate()
                        .fold((0usize, f32::MIN), |(bi, bv), (i, &p)| {
                            let v = f32::from_bits((p as u32) << 16);
                            if v > bv {
                                (i, v)
                            } else {
                                (bi, bv)
                            }
                        });
                    let (ib, _) = b
                        .iter()
                        .enumerate()
                        .fold((0usize, f32::MIN), |(bi, bv), (i, &p)| {
                            let v = f32::from_bits((p as u32) << 16);
                            if v > bv {
                                (i, v)
                            } else {
                                (bi, bv)
                            }
                        });
                    assert!(
                        first_bad.is_none(),
                        "k={k} n={n} draw={draw} wn={write_norm}: y exceeds 1 ULP at {:?} (maxdiff {maxd})",
                        first_bad
                    );
                    assert_eq!(
                        ia, ib,
                        "k={k} n={n} draw={draw} wn={write_norm}: argmax moved"
                    );
                    if write_norm {
                        let na = normed_ref.as_slice::<u16>();
                        let nn_arr = normed_new.unwrap();
                        let nb = nn_arr.as_slice::<u16>();
                        let mut nmax = 0i32;
                        for (p, q) in na.iter().zip(nb.iter()) {
                            let d = (ordered(*p) - ordered(*q)).abs();
                            if d > nmax {
                                nmax = d;
                            }
                        }
                        assert!(
                            nmax <= 1,
                            "k={k} n={n} draw={draw}: normed maxdiff {nmax} ULP"
                        );
                    }
                    println!(
                        "addnorm tail-ULP: k={k} n={n} draw={draw} wn={write_norm} OK (maxdiff {maxd} ULP, argmax equal)"
                    );
                }
            }
        }
    }

    /// The bit-exact tile must be BIT-identical to the per-row qmv_fast
    /// chain it replaces in the MTP verify forward (7×qmv at depth 6).
    /// Five shapes covering the verify width range; ≥65k compared values per
    /// shape (the fused-kernel-wave pinning convention).
    #[test]
    fn verify_qmm_bitexact_vs_qmv() {
        for (k, n) in [
            (5120usize, 17408usize),
            (6144, 6144),
            (2560, 13824),
            (1024, 512),
            (5120, 32768),
        ] {
            let mut seed = 77u64;
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 2.0 - 1.0).collect();
            let (wq, scales, biases) =
                quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
            let wq = wq.contiguous().unwrap();
            let scales = scales.contiguous().unwrap();
            let biases = biases.contiguous().unwrap();
            for m in [2usize, 3, 5, 7] {
                let xv: Vec<f32> = (0..m * k).map(|_| rnd() * 4.0 - 2.0).collect();
                let x = Array::from_slice(&xv, &[m as i32, k as i32])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                // Reference: m × affine_qmv_fast (M=1 each), compared row by
                // row against the tiled output.
                let y_new = crate::jit::affine_verify_qmm(
                    x.t.device(),
                    &x.t,
                    &wq.t,
                    &scales.t,
                    &biases.t,
                    64,
                    4,
                )
                .unwrap();
                let _ = y_new.eval();
                let b = y_new.as_slice::<u16>().to_vec();
                assert_eq!(b.len(), m * n);
                for r in 0..m {
                    let rowv = &xv[r * k..(r + 1) * k];
                    let row = Array::from_slice(rowv, &[1i32, k as i32])
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let y_ref = crate::jit::affine_qmv_fast(
                        x.t.device(),
                        &row.t,
                        &wq.t,
                        &scales.t,
                        &biases.t,
                        64,
                        4,
                    )
                    .unwrap();
                    let _ = y_ref.eval();
                    let a = y_ref.as_slice::<u16>();
                    let bad: Vec<usize> = a
                        .iter()
                        .zip(b[r * n..(r + 1) * n].iter())
                        .enumerate()
                        .filter_map(|(i, (p, q))| (p != q).then_some(i))
                        .collect();
                    assert!(
                        bad.is_empty(),
                        "k={k} n={n} m={m} row={r}: {} bit mismatches, first at {:?} (ref {:04x} new {:04x})",
                        bad.len(),
                        bad.first(),
                        a[bad[0]],
                        b[r * n + bad[0]],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::ops::quantize;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn bench_splitk_widths() {
        for m in [3usize, 5, 7] {
            let (k, n) = (5120usize, 17408usize);
            let mut seed = 9u64;
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 0.02 - 0.01).collect();
            let (wq, scales, biases) =
                quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
            let wq = wq.contiguous().unwrap();
            let sc = scales.as_dtype(Dtype::Bfloat16).unwrap().contiguous().unwrap();
            let bi = biases.as_dtype(Dtype::Bfloat16).unwrap().contiguous().unwrap();
            let xv: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let x = Array::from_slice(&xv, &[m as i32, k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let mut f = || {
                let _ = crate::jit::affine_verify_qmm_splitk(
                    x.t.device(), &x.t, &wq.t, &sc.t, &bi.t, 64, 4,
                );
            };
            for _ in 0..3 { f(); }
            let _ = x.eval();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..10 {
                let t0 = std::time::Instant::now();
                f();
                let _ = x.eval();
                total += t0.elapsed();
            }
            println!("splitk m={m}: {:?}/iter", total / 10);
        }
    }

    #[test]
    #[ignore]
    fn bench_verify_qmm_paths() {
        let shapes = [(5120usize, 17408usize), (5120usize, 151936usize)];
        for (k, n) in shapes {
            let mut seed = 5u64;
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let m = 7usize;
            let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 0.02 - 0.01).collect();
            let (wq, scales, biases) =
                quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
            let wq = wq.contiguous().unwrap();
            let scales = scales.contiguous().unwrap();
            let biases = biases.contiguous().unwrap();
            let xv: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let x = Array::from_slice(&xv, &[m as i32, k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let warm = |f: &mut dyn FnMut()| {
                for _ in 0..3 {
                    f();
                }
                let _ = x.eval();
            };
            let time = |name: &str, f: &mut dyn FnMut()| {
                warm(f);
                // Single-shot: sync EVERY rep so launches cannot pipeline and
                // hide the drain tail (the e2e regime — one launch per
                // projection between other kernels).
                let mut total = std::time::Duration::ZERO;
                let reps = 20;
                for _ in 0..reps {
                    let t0 = Instant::now();
                    f();
                    let _ = x.eval();
                    total += t0.elapsed();
                }
                println!("{name} [k={k} n={n}]: {:?}/iter", total / reps);
            };
            let mut f1 = || {
                let _ =
                    crate::jit::qmv_wide(x.t.device(), &x.t, &wq.t, &scales.t, &biases.t, 64, 4);
            };
            time("qmv_wide      ", &mut f1);
            let mut f2 = || {
                let _ = crate::jit::affine_verify_qmm(
                    x.t.device(),
                    &x.t,
                    &wq.t,
                    &scales.t,
                    &biases.t,
                    64,
                    4,
                );
            };
            time("verify_qmm    ", &mut f2);
            let mut f4 = || {
                let _ = crate::jit::affine_verify_qmm_splitk(
                    x.t.device(),
                    &x.t,
                    &wq.t,
                    &scales.t,
                    &biases.t,
                    64,
                    4,
                );
            };
            time("splitk        ", &mut f4);
            let mut f5 = || {
                let _ = crate::jit::affine_verify_qmm_msg(
                    x.t.device(), &x.t, &wq.t, &scales.t, &biases.t, 64, 4,
                );
            };
            time("msg           ", &mut f5);
            let mut f3 = || {
                for r in 0..m {
                    let rowv: Vec<f32> = xv[r * k..(r + 1) * k].to_vec();
                    let row = Array::from_slice(&rowv, &[1i32, k as i32])
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap();
                    let _ = crate::jit::affine_qmv_fast(
                        x.t.device(),
                        &row.t,
                        &wq.t,
                        &scales.t,
                        &biases.t,
                        64,
                        4,
                    );
                }
            };
            time("7x qmv_fast   ", &mut f3);
        }
    }

    /// Sweep the split-K verify tile over packs-per-thread (1,2,4) and BN
    /// (1,2,4) at the shipping trunk shapes (specs/15 §6 lever). Single-shot
    /// timing (sync every rep) per the specs/14 lesson that pipelined reps
    /// lie; the verdict that matters is still in-situ round-cost.
    #[test]
    #[ignore]
    fn bench_verify_qmm_splitk_ppt_sweep() {
        let k = 5120usize;
        let n = 17408usize;
        for m in [5usize, 7usize] {
            let mut seed = 7u64;
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 0.02 - 0.01).collect();
            let (wq, scales, biases) =
                quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
            let wq = wq.contiguous().unwrap();
            let scales = scales.contiguous().unwrap();
            let biases = biases.contiguous().unwrap();
            let xv: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let x = Array::from_slice(&xv, &[m as i32, k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let dev = x.t.device();
            let k_parts = if n >= 4096 { 2usize } else { 4usize };
            // reference output from the shipped lane (ppt from lane const)
            for bn in [1usize, 2usize, 4usize] {
                for ppt in [1usize, 2usize, 4usize] {
                    // correctness spot-check vs ppt=1 (bit-identity)
                    let f = || {
                        let _ = crate::jit::affine_verify_qmm_splitk_tile(
                            dev, &x.t, &wq.t, &scales.t, &biases.t, 64, 4, bn, k_parts, ppt,
                        );
                    };
                    // warm + JIT compile
                    for _ in 0..3 {
                        f();
                    }
                    let _ = x.eval();
                    let y1 = crate::jit::affine_verify_qmm_splitk_tile(
                        dev, &x.t, &wq.t, &scales.t, &biases.t, 64, 4, bn, k_parts, 1,
                    )
                    .unwrap();
                    let _ = y1.eval();
                    let yp = crate::jit::affine_verify_qmm_splitk_tile(
                        dev, &x.t, &wq.t, &scales.t, &biases.t, 64, 4, bn, k_parts, ppt,
                    )
                    .unwrap();
                    let _ = yp.eval();
                    let av: Vec<u16> = y1.to_vec::<u16>().unwrap();
                    let bv: Vec<u16> = yp.to_vec::<u16>().unwrap();
                    let exact = av == bv;
                    let mut total = std::time::Duration::ZERO;
                    let reps = 20;
                    for _ in 0..reps {
                        let t0 = Instant::now();
                        f();
                        let _ = x.eval();
                        total += t0.elapsed();
                    }
                    println!(
                        "ppt-sweep m={m} bn={bn} ppt={ppt} kp={k_parts}: {:?}/iter bitexact={exact}",
                        total / reps
                    );
                }
            }
        }
    }

    /// Isolated RANKING bench (isolated numbers LIE — in-situ round-cost is
    /// the arbiter) of the tail-ULP variant axis: stock vs VLOAD vs Math::Fast
    /// vs both, at the production tile (bn=2, k_parts), trunk shape.
    #[test]
    #[ignore]
    fn bench_verify_qmm_splitk_variant_sweep() {
        let k = 5120usize;
        let n = 17408usize;
        for m in [3usize, 5usize, 7usize] {
            let mut seed = 11u64;
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let wv: Vec<f32> = (0..n * k).map(|_| rnd() * 0.02 - 0.01).collect();
            let (wq, scales, biases) =
                quantize(&Array::from_slice(&wv, &[n as i32, k as i32]), 64, 4).unwrap();
            let wq = wq.contiguous().unwrap();
            let scales = scales.as_dtype(Dtype::Bfloat16).unwrap().contiguous().unwrap();
            let biases = biases.as_dtype(Dtype::Bfloat16).unwrap().contiguous().unwrap();
            let xv: Vec<f32> = (0..m * k).map(|_| rnd() * 4.0 - 2.0).collect();
            let x = Array::from_slice(&xv, &[m as i32, k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
                .contiguous()
                .unwrap();
            let dev = x.t.device();
            let k_parts = if n >= 4096 { 2usize } else { 4usize };
            for (vl, fm) in [(false, false), (true, false), (false, true), (true, true)] {
                let f = || {
                    crate::jit::affine_verify_qmm_splitk_tile_ex(
                        dev, &x.t, &wq.t, &scales.t, &biases.t, 64, 4, 2, k_parts, 1, vl, fm,
                    )
                    .unwrap()
                };
                for _ in 0..3 {
                    let _ = f();
                }
                let _ = x.eval();
                let mut total = std::time::Duration::ZERO;
                let reps = 20;
                for _ in 0..reps {
                    let t0 = Instant::now();
                    let _ = f();
                    let _ = x.eval();
                    total += t0.elapsed();
                }
                println!(
                    "variant-sweep m={m} vl={vl} fm={fm} kp={k_parts}: {:?}/iter",
                    total / reps
                );
            }
        }
    }
}
