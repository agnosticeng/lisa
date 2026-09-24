//! Normalization modules and the partial rotary embedding.

use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{fast, ops, Array, Dtype};

use crate::core::loader::TensorSource;

/// RMSNorm with the checkpoint's baked weight-offset convention
/// (`rms_norm_weight_offset == 0` for this checkpoint: the stored weight IS
/// the scale). With `group_size` set, each group is normalized on its own
/// statistic and the flat scale applies after the normalized rounding.
#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Array,
    pub eps: f32,
    pub group_size: Option<usize>,
}

impl RmsNorm {
    pub fn load<S: TensorSource>(src: &mut S, name: &str, eps: f32, group_size: Option<usize>) -> anyhow::Result<Self> {
        Ok(Self {
            weight: src.get_bf16(&format!("{name}.weight"))?,
            eps,
            group_size,
        })
    }

    pub fn forward(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        match self.group_size {
            None => fast::rms_norm(x, Some(&self.weight), self.eps),
            Some(g) => {
                // The engine's rms_single_row reduction (pinned association).
                rms_row_exact(x, Some(&self.weight), self.eps, g)
            }
        }
    }
}


/// Partial rotary embedding evaluated at explicit positions.
///
/// cos/sin are computed ON DEVICE from two integers so the arithmetic matches
/// the reference exactly (MLX f32 exp/cos, not libm).
pub fn bf16_scalar(v: f32) -> Array {
    // Host-built bf16 scalar (no GPU round trip): `from_f32().as_dtype(BF16)`
    // hits `cast_host`, which evals the device (a full GPU sync per layer per
    // token — the dominant decode cost).
    Array::from_f32_as(v, lisa_mlx::Dtype::Bfloat16)
}

/// The engine's EXACT row RMS reduction (rms_single_row / injectNorm layout):
///
/// Channels are dealt to 20 simdgroups x 32 lanes x 4 consecutive elements
/// (per 2560-channel stream; generalizes to any H % 128 == 0). Each lane
/// accumulates its 4 squares sequentially in f32, each simdgroup folds its 32
/// lanes with the hardware xor-butterfly (pairwise tree), the per-simdgroup
/// partials are zero-padded to 32 and folded with a second butterfly, and
/// `inv_mean = precise::rsqrt(total / H + eps)`. The output is
/// `bf16(x * inv_mean) * scale` — the weight applied AFTER the bf16 rounding.
///
/// This pins the reduction association independently of the MLX build, which
/// is what the engine's AOT metallib bakes in.

/// Temporary: dump `rms_row_exact` intermediates for main-vs-branch comparison.
fn rms_trace(name: &str, a: &Array) {
    if std::env::var("LISA_RMS_TRACE").is_err() {
        return;
    }
    // Only the first rms_row_exact call (layer 0's hc_norm) is of interest.
    static FIRST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = FIRST.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if call > 12 {
        return;
    }
    let f = match a.as_dtype(Dtype::Float32) {
        Ok(f) => f,
        Err(_) => return,
    };
    let v = f.as_slice::<f32>().to_vec();
    std::fs::write(format!("/tmp/rms-trace/{call}_{name}.bin"), unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
    })
    .ok();
}

pub fn rms_row_exact(
    x: &Array,
    scale: Option<&Array>,
    eps: f32,
    h: usize,
) -> lisa_mlx::error::Result<Array> {
    use lisa_mlx::ops::indexing::IndexOp;
    if lisa_mlx::env_flag("LISA_RMS_EVAL_TOP") {
        lisa_mlx::transforms::eval([x])?;
    }
    let w_full = x.dim(-1) as usize;
    let groups = w_full / h;
    let slices = (h / 128) as i32;
    let lead = &x.shape()[..x.shape().len() - 1];
    // [lead..., groups, slices, 32, 4]
    let grouped = x.reshape(&[lead, &[groups as i32, slices, 32i32, 4i32]].concat())?;
    let sq = lisa_mlx::ops::multiply(
        &grouped.as_dtype(Dtype::Float32)?,
        &grouped.as_dtype(Dtype::Float32)?,
    )?;
    if std::env::var("E_SQ").is_ok() { eprintln!("[flag E_SQ]"); let _ = lisa_mlx::transforms::eval([&sq]); }
    // per-lane sequential sum over the 4 elements -> [.., groups, slices, 32]
    let sq_keep = sq.clone();
    let mut lane = sq.index((Ellipsis, 0..1)).contiguous().unwrap();
    for i in 1..4i32 {
        lane = lane.add(&sq.index((Ellipsis, i..i + 1)).contiguous().unwrap())?;
    }
    if std::env::var("E_LANE").is_ok() { eprintln!("[flag E_LANE]"); let _ = lisa_mlx::transforms::eval([&lane]); }
    let lane_v = lane.reshape(&[lead, &[groups as i32, slices, 32i32]].concat())?;
    let lane_keep = lane_v.clone();
    rms_trace("a_lane", &lane_v);
    // per-simdgroup xor-butterfly over the 32 lanes -> [.., groups, slices]
    let mut v = lane_v;
    let mut width = 32i32;
    while width >= 2 {
        let half = width / 2;
        let a = v.index((Ellipsis, 0..half)).contiguous().unwrap();
        let b = v.index((Ellipsis, half..width)).contiguous().unwrap();
        v = a.add(&b)?;
        width /= 2;
    }
    if lisa_mlx::env_flag("LISA_SHAPE_DEBUG") {
        eprintln!("rms_row: x {:?} groups {groups} slices {slices} lane {:?} v {:?}", x.shape(), lane.shape(), v.shape());
    }
    rms_trace("b_partials", &v);
    let partials = v.reshape(&[lead, &[groups as i32, slices]].concat())?;
    // zero-pad the partials to 32 slots, then the second butterfly
    let mut zshape = lead.to_vec();
    zshape.push(groups as i32);
    zshape.push(32i32 - slices);
    let zeros = lisa_mlx::ops::zeros::<f32>(&zshape)
        .map_err(|e| lisa_mlx::error::Exception::custom(e.to_string()))?;
    let padded = lisa_mlx::ops::concatenate(&[&partials, &zeros], -1)?;
    rms_trace("c_padded", &padded);
    let v_keep = v.clone();
    if std::env::var("E_PADDED").is_ok() { eprintln!("[flag E_PADDED]"); let _ = lisa_mlx::transforms::eval([&padded]); }
    let padded_keep = padded.clone();
    let mut v2 = padded;
    width = 32;
    while width >= 2 {
        let half = width / 2;
        let a = v2.index((Ellipsis, 0..half)).contiguous().unwrap();
        let b = v2.index((Ellipsis, half..width)).contiguous().unwrap();
        v2 = a.add(&b)?;
        width /= 2;
    }
    // total: [lead..., groups] -> unsqueeze for broadcasting
    let total = v2.reshape(&[lead, &[groups as i32, 1i32, 1i32, 1i32]].concat())?;
    rms_trace("d_total", &total);
    if std::env::var("E_TOTAL").is_ok() { eprintln!("[flag E_TOTAL]"); let _ = lisa_mlx::transforms::eval([&total]); }
    let scaled = total.clone() / (h as f32);
    let biased = scaled.add(&Array::from_f32(eps))?;
    let inv = lisa_mlx::ops::rsqrt(&biased)
        .map_err(|e| lisa_mlx::error::Exception::custom(e.to_string()))?;
    rms_trace("e_inv", &inv);
    rms_trace("f_grouped", &grouped);
    if std::env::var("E_INV").is_ok() { eprintln!("[flag E_INV]"); let _ = lisa_mlx::transforms::eval([&inv]); }
    let normalized = (grouped.as_dtype(Dtype::Float32)? * &inv).as_dtype(Dtype::Bfloat16)?;
    rms_trace("g_normalized", &normalized);
    if std::env::var("E_NORMED").is_ok() { eprintln!("[flag E_NORMED]"); let _ = lisa_mlx::transforms::eval([&normalized]); }
    let flat = normalized.reshape(&x.shape())?;
    rms_trace("h_flat", &flat);
    let out = match scale {
        Some(w) => flat.multiply(w),
        None => Ok(flat.clone()),
    };
    if let Ok(o) = &out { rms_trace("i_out", o); }
    if lisa_mlx::env_flag("LISA_DUMP_NORMED") {
        let d = |name: &str, a: &Array| {
            if let Ok(f) = a.as_dtype(Dtype::Float32) {
                { let v = f.as_slice::<f32>();
                    let v: &[f32] = v;
                    std::fs::write(format!("/tmp/rms-trace/{name}.bin"), unsafe {
                        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                    }).ok();
                }
            }
        };
        d("nn_sq", &sq_keep);
        d("nn_lane", &lane_keep);
        d("nn_v1", &v_keep);
        d("nn_padded", &padded_keep);
        d("nn_v2", &v2);
        d("nn_total", &total);
        d("nn_inv", &inv);
        d("nn_grouped_f32", &grouped);
        d("nn_normalized", &normalized);
        if let Ok(f2) = normalized.reshape(&x.shape()) { d("nn_flat", &f2); }
        if let Ok(o) = &out { d("nn_out", o); }
    }
    out
}

pub fn bf16_silu(x: &Array) -> lisa_mlx::error::Result<Array> {
    lisa_mlx::nn::silu(x)
}

/// The fork's `MLXNN.softplus` / `logaddexp(x, 0)` on bf16 rounds through
/// bf16 at every step of the op chain.
pub fn bf16_logaddexp0(x: &Array) -> lisa_mlx::error::Result<Array> {
    let zero = bf16_scalar(0.0);
    let mx_ = ops::maximum(x, &zero)?;
    let mn = ops::minimum(x, &zero)?;
    let d = mn - &mx_; // bf16 (operator- rounds)
    // exp/log1p are the ported MLX unary kernels, so the chain rounds exactly
    // as the engine's does.
    let e = Array::new(lisa_mlx::mlx_rt::exp(d.t.device(), &d.t)?);
    let l = Array::new(lisa_mlx::mlx_rt::log1p(e.t.device(), &e.t)?);
    mx_.add(&l)
}

/// Positions of `count` tokens starting at `offset`, shaped `[1, count]`.
/// Conventional RMSNorm with an optional sigmoid output gate, used by the
/// gated deltanet. The weight is a plain scale.
#[derive(Clone)]
pub struct RmsNormGated {
    pub weight: Array,
    pub eps: f32,
}

impl RmsNormGated {
    pub fn load<S: TensorSource>(src: &mut S, name: &str, eps: f32) -> anyhow::Result<Self> {
        Ok(Self {
            weight: src.get_bf16(&format!("{name}.weight"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array, gate: &Array) -> lisa_mlx::error::Result<Array> {
        // The gated RMS runs the engine's butterfly reduction at H = the head
        // dim (the weight [128] applied after the bf16 rounding).
        let out = fast::rms_norm(x, Some(&self.weight), self.eps)?;
        let g = ops::sigmoid(&gate.as_dtype(Dtype::Float32)?)?;
        (g * out.as_dtype(Dtype::Float32)?).as_dtype(x.dtype())
    }
}

/// Partial rotary embedding evaluated at explicit positions.
#[derive(Clone)]
pub struct Rotary {
    pub dimensions: usize,
    pub base: f32,
}

impl Rotary {
    pub fn new(dimensions: usize, base: f32) -> Self {
        Self { dimensions, base }
    }

    /// cos/sin for `positions` (any shape), each `positions.shape + [dims]`.
    pub fn cos_sin(&self, positions: &Array) -> lisa_mlx::error::Result<(Array, Array)> {
        let even: Vec<f32> = (0..self.dimensions).step_by(2).map(|i| i as f32).collect();
        let even_len = even.len();
        let even = Array::from_slice(&even, &[even_len as i32]).as_dtype(Dtype::Float32)?;
        let arg = even * (-self.base.ln() / self.dimensions as f32);
        let inv_freq = if std::env::var("LISA_ROPE_EXP_KERNEL").is_ok() {
            Array::new(lisa_mlx::mlx_rt::exp(arg.t.device(), &arg.t)?)
        } else if std::env::var("LISA_ROPE_EXP_NATIVE").is_ok() {
            Array::new(arg.t.exp()?)
        } else {
            ops::exp(&arg)?
        };
        let freqs = positions
            .as_dtype(Dtype::Float32)?
            .expand_dims(-1)?
            .multiply(&inv_freq)?;
        let emb = ops::concatenate(&[&freqs, &freqs], -1)?;
        if lisa_mlx::env_flag("LISA_ATTN_DEBUG") {
            eprintln!("[cos_sin] dims={} positions={:?} even_len={} inv_freq={:?} freqs={:?} emb={:?}",
                self.dimensions, positions.shape(), even_len, inv_freq.shape(), freqs.shape(), emb.shape());
        }
        Ok((ops::cos(&emb)?, ops::sin(&emb)?))
    }
}

/// Rotate only the leading `cos.dim(-1)` entries of the last axis.
pub fn rope_partial(x: &Array, cos: &Array, sin: &Array) -> lisa_mlx::error::Result<Array> {
    if std::env::var("LISA_ROPE_DEBUG").is_ok() {
        eprintln!("[rope_partial] x={:?} cos={:?} sin={:?}", x.shape(), cos.shape(), sin.shape());
    }
    let d = cos.dim(-1) as usize;
    // cos/sin are float32; cast FIRST or the whole attention promotes to f32.
    let c = cos.as_dtype(x.dtype())?;
    let s = sin.as_dtype(x.dtype())?;
    let nd = x.dim(-1) as usize;
    let rotated = x.index((Ellipsis, 0..d as i32));
    let half = d / 2;
    let x1 = rotated.index((Ellipsis, 0..half as i32));
    let x2 = rotated.index((Ellipsis, half as i32..d as i32));
    let neg_x2 = -x2;
    let swapped = ops::concatenate(&[&neg_x2, &x1], -1)?;
    let out = rotated.multiply(&c)?.add(&swapped.multiply(&s)?)?;
    if nd == d {
        Ok(out)
    } else {
        let rest = x.index((Ellipsis, d as i32..nd as i32));
        ops::concatenate(&[&out, &rest], -1)
    }
}

/// Positions of `count` tokens starting at `offset`, shaped `[1, count]`.
///
/// Built as f32 so RoPE does not need an i32->f32 cast (Metal
/// `to_dtype` lacks that pair and falls back to a host round-trip that
/// synchronizes the GPU once per attention layer).
pub fn positions(offset: usize, count: usize) -> lisa_mlx::error::Result<Array> {
    let data: Vec<f32> = (offset..offset + count).map(|i| i as f32).collect();
    Ok(Array::from_slice(&data, &[1i32, count as i32]))
}


#[cfg(test)]
mod rope_ab {
    use super::*;
    #[test]
    fn rope_partial_ab() {
        let (b,h,s,d,dims) = (1i32, 24i32, 1i32, 256i32, 64i32);
        let mut seed=555u64;
        let mut rnd = || { seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed>>40) as f32/8_388_608.0)-1.0 };
        let xv: Vec<f32> = (0..(b*h*s*d) as usize).map(|_| half::bf16::from_f32(rnd()).to_f32()).collect();
        let cv: Vec<f32> = (0..(b*s*dims) as usize).map(|_| rnd()).collect();
        let sv: Vec<f32> = (0..(b*s*dims) as usize).map(|_| rnd()).collect();
        let x = Array::from_slice(&xv, &[b,h,s,d]).as_dtype(Dtype::Bfloat16).unwrap();
        let cos = Array::from_slice(&cv, &[b,s,dims]).as_dtype(Dtype::Float32).unwrap().expand_dims(1).unwrap();
        let sin = Array::from_slice(&sv, &[b,s,dims]).as_dtype(Dtype::Float32).unwrap().expand_dims(1).unwrap();
        let out = rope_partial(&x, &cos, &sin).unwrap();
        let f = out.as_dtype(Dtype::Float32).unwrap();
        let a = f.as_slice::<f32>();
        let dir = std::env::var("LISA_AB_OUT").unwrap_or_else(|_| "/tmp".into());
        let _ = std::fs::write(format!("{dir}/rope_partial.bin"), unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len()*4) });
    }
}
