//! Host wrappers for the qwen3_5 fused kernels.

use crate::ffi::{MetalKernel, TemplateArg};
use crate::{Array, Dtype, Stream};

/// Shared exact-arithmetic helpers (`mlx_sigmoid`, `mlx_silu`, ...).
pub const EXACT_HEADER: &str = include_str!("../../shaders/common/exact_header.metal");

/// `track_sigmoid_mul`: `mlx_sigmoid(gate) * up` over separate bf16 arrays.
/// Replaces the attention output-gate tail (`ops::sigmoid(g)` then
/// `a.multiply(&sig)`) — one dispatch instead of two, same rounding: the
/// sigmoid rounds bf16 exactly as the separate kernel would have stored it,
/// then the product rounds bf16.
const SIGMOID_MUL_SOURCE: &str = include_str!("../../shaders/qwen35/sigmoid_mul.metal");

/// `track_swiglu2_packed`: `mlx_silu(gu[i]) * gu[F + i]` over ONE packed
/// `[B, 2F]` gate|up buffer (the merged gate|up qmv output). Bit-identical to
/// `track_swiglu2` on the split arrays: same exact_header silu, and the second
/// half of the packed qmv output IS the up projection, row-exact.
const SWIGLU2_PACKED_SOURCE: &str = include_str!("../../shaders/qwen35/swiglu2_packed.metal");

/// Source of `track_sigmoid_mul_tail` (see [`sigmoid_mul_tail`]).
const SIGMOID_MUL_TAIL_SOURCE: &str = include_str!("../../shaders/qwen35/sigmoid_mul_tail.metal");

static SWIGLU2_PACKED: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn swiglu2_packed_kernel() -> &'static Option<MetalKernel> {
    SWIGLU2_PACKED.get_or_init(|| {
        MetalKernel::new(
            "track_swiglu2_packed",
            &["gu"],
            &["out"],
            SWIGLU2_PACKED_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run `track_swiglu2_packed`. `gu` `[B, 2F]` contiguous bf16 -> `[B, F]`.
/// `None` when unavailable / odd width / non-bf16 — caller falls back to the
/// split qmv + `track_swiglu2` chain.
pub fn swiglu2_packed(gu: &Array, stream: &Stream) -> Option<Array> {
    if gu.dtype() != Dtype::Bfloat16 {
        return None;
    }
    let f2 = gu.dim(-1);
    if f2 % 2 != 0 {
        return None;
    }
    let kernel = swiglu2_packed_kernel().as_ref()?;
    let f = f2 / 2;
    let b = (gu.size() / f2 as usize) as i32;
    let inputs: [&Array; 1] = [gu];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("B", b),
        TemplateArg::Int("F", f as i32),
    ];
    let outs = [crate::ffi::OutputArg {
        shape: vec![b, f as i32],
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(
            &inputs,
            &template,
            (b * f as i32, 1, 1),
            (256, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

static SIGMOID_MUL: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

/// `track_sigmoid_mul_tail` (specs/04): the attention tail in ONE launch over
/// the UN-transposed sdpa output. `gate` is `[b, s, h, hd]`, `up` is the sdpa
/// output `[b, h, s, hd]` (both read through their strides); the output is the
/// contiguous merged `[b, s, h*hd]` the o_proj consumes. Removes the per-layer
/// tail transpose materialization + the separate sigmoid_mul. Same per-op bf16
/// rounding chain as `track_sigmoid_mul` — bit-exact by construction, pinned
/// by `sigmoid_mul_tail_matches_transposed`. `None` on non-bf16 or kernel
/// unavailability — the caller falls back to the transpose + sigmoid_mul chain.
pub fn sigmoid_mul_tail(
    gate: &Array,
    up: &Array,
    b: i32,
    s: i32,
    h: i32,
    hd: i32,
    stream: &Stream,
) -> Option<Array> {
    use crate::ffi::OutputArg;
    if gate.dtype() != Dtype::Bfloat16 || up.dtype() != Dtype::Bfloat16 {
        return None;
    }
    let kernel = sigmoid_mul_tail_kernel().as_ref()?;
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("B", b),
        TemplateArg::Int("H", h),
        TemplateArg::Int("S", s),
        TemplateArg::Int("HD", hd),
    ];
    let total = (b * h * s * hd) as usize;
    let inputs: [&Array; 2] = [gate, up];
    let outs = [OutputArg {
        shape: vec![b, s, h * hd],
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(
            &inputs,
            &template,
            (total as i32, 1, 1),
            (256, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

static SIGMOID_MUL_TAIL: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn sigmoid_mul_tail_kernel() -> &'static Option<MetalKernel> {
    SIGMOID_MUL_TAIL.get_or_init(|| {
        MetalKernel::new(
            "track_sigmoid_mul_tail",
            &["gate", "up"],
            &["out"],
            SIGMOID_MUL_TAIL_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

fn sigmoid_mul_kernel() -> &'static Option<MetalKernel> {
    SIGMOID_MUL.get_or_init(|| {
        MetalKernel::new(
            "track_sigmoid_mul",
            &["gate", "up"],
            &["out"],
            SIGMOID_MUL_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run `track_sigmoid_mul`. `gate`/`up` (any shape, contiguous bf16) -> same
/// shape. `None` when the kernel is unavailable or the inputs are not
/// contiguous bf16 — the caller falls back to the composed chain.
pub fn sigmoid_mul(gate: &Array, up: &Array, stream: &Stream) -> Option<Array> {
    if gate.dtype() != Dtype::Bfloat16 || up.dtype() != Dtype::Bfloat16 {
        return None;
    }
    if gate.shape() != up.shape() {
        return None;
    }
    let kernel = sigmoid_mul_kernel().as_ref()?;
    let f = gate.dim(-1).max(1);
    let b = (gate.size() / f as usize) as i32;
    let inputs: [&Array; 2] = [gate, up];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("B", b),
        TemplateArg::Int("F", f),
    ];
    let outs = [crate::ffi::OutputArg {
        shape: gate.shape().to_vec(),
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(
            &inputs,
            &template,
            (b * f, 1, 1),
            (256, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

/// `track_qk_norm_rope`: per-head RMSNorm + partial RoPE (rd=64 of hd=256)
/// for q AND k in ONE decode launch, emitting the head-major `[1,H,1,256]`
/// layout SDPA consumes (the transposes vanish). Bit-exact vs the composed
/// `RmsNorm::forward` → transpose → `rope_partial` chain (see shader header);
/// pinned by `qk_norm_rope_matches_composed`.
const QK_NORM_ROPE_SOURCE: &str = include_str!("../../shaders/qwen35/qk_norm_rope.metal");

static QK_NORM_ROPE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

const QK_NORM_ROPE_ROWS_SOURCE: &str = include_str!("../../shaders/qwen35/qk_norm_rope_rows.metal");

static QK_NORM_ROPE_ROWS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn qk_norm_rope_rows_kernel() -> &'static Option<MetalKernel> {
    QK_NORM_ROPE_ROWS.get_or_init(|| {
        MetalKernel::new(
            "track_qk_norm_rope_rows",
            &["qg", "k", "qw", "kw", "cosv", "sinv", "eps"],
            &["oq", "ok", "og"],
            QK_NORM_ROPE_ROWS_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

fn qk_norm_rope_kernel() -> &'static Option<MetalKernel> {
    QK_NORM_ROPE.get_or_init(|| {
        MetalKernel::new(
            "track_qk_norm_rope",
            &["q", "k", "qw", "kw", "cosv", "sinv", "eps"],
            &["oq", "ok"],
            QK_NORM_ROPE_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run `track_qk_norm_rope` at s=1. `q`/`k` are the contiguous projection
/// rows `[1,1,H*256]`; `cos`/`sin` the bf16-cast angle rows `[.., 64]`.
/// Returns head-major `(oq [1,hq,1,256], ok [1,hk,1,256])`.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    q: &Array,
    k: &Array,
    qw: &Array,
    kw: &Array,
    cos: &Array,
    sin: &Array,
    eps: f32,
    hq: i32,
    hk: i32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    const HD: i32 = 256;
    if q.dtype() != Dtype::Bfloat16
        || k.dtype() != Dtype::Bfloat16
        || qw.dtype() != Dtype::Bfloat16
        || kw.dtype() != Dtype::Bfloat16
        || cos.dtype() != Dtype::Bfloat16
        || sin.dtype() != Dtype::Bfloat16
    {
        return None;
    }
    if q.size() != (hq * HD) as usize || k.size() != (hk * HD) as usize {
        return None;
    }
    let kernel = qk_norm_rope_kernel().as_ref()?;
    let eps_a = Array::from_f32(eps);
    let inputs: [&Array; 7] = [q, k, qw, kw, cos, sin, &eps_a];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("HQ", hq),
    ];
    let outs = [
        crate::ffi::OutputArg {
            shape: vec![1, hq, 1, HD],
            dtype: Dtype::Bfloat16,
        },
        crate::ffi::OutputArg {
            shape: vec![1, hk, 1, HD],
            dtype: Dtype::Bfloat16,
        },
    ];
    let mut it = kernel
        .apply(
            &inputs,
            &template,
            ((hq + hk) * 64, 1, 1),
            (64, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter();
    Some((it.next()?, it.next()?))
}

/// Run `track_qk_norm_rope_rows` at s in 2..=8 (the MTP verify block, b==1,
/// no keep mask). `qg` is the stock interleaved q|gate projection `[S, HQ*2*HD]`
/// (layout [head][2][hd]); `k` the contiguous key rows `[S, HK*HD]`; `cos`/`sin`
/// the bf16-cast angle block `[S, RD]` (one row per verify row). Returns
/// head-major `(oq [1,hq,S,256], ok [1,hk,S,256])` plus the raw de-interleaved
/// gate `[S, HQ*HD]` for the sigmoid-gate tail.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_rows(
    qg: &Array,
    k: &Array,
    qw: &Array,
    kw: &Array,
    cos: &Array,
    sin: &Array,
    eps: f32,
    hq: i32,
    hk: i32,
    s: i32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    const HD: i32 = 256;
    if s < 2 || s > 8 {
        return None;
    }
    for a in [qg, k, qw, kw, cos, sin] {
        if a.dtype() != Dtype::Bfloat16 {
            return None;
        }
    }
    if qg.size() != (s * hq * 2 * HD) as usize || k.size() != (s * hk * HD) as usize {
        return None;
    }
    if cos.size() != (s * 64) as usize || sin.size() != (s * 64) as usize {
        return None;
    }
    let kernel = qk_norm_rope_rows_kernel().as_ref()?;
    let eps_a = Array::from_f32(eps);
    let inputs: [&Array; 7] = [qg, k, qw, kw, cos, sin, &eps_a];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("HQ", hq),
        TemplateArg::Int("HK", hk),
        TemplateArg::Int("S", s),
    ];
    let outs = [
        crate::ffi::OutputArg {
            shape: vec![1, hq, s, HD],
            dtype: Dtype::Bfloat16,
        },
        crate::ffi::OutputArg {
            shape: vec![1, hk, s, HD],
            dtype: Dtype::Bfloat16,
        },
        crate::ffi::OutputArg {
            shape: vec![s, hq * HD],
            dtype: Dtype::Bfloat16,
        },
    ];
    let mut it = kernel
        .apply(
            &inputs,
            &template,
            ((hq + hk) * s * 64, 1, 1),
            (64, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}

/// `track_keyed_gumbel_sample`: full-vocab Gumbel-max draw in one launch
/// (specs/08 §5). `L` is the flat f32 `[V]` logit row, `inv_temperature` a
/// 0-dim f32 scalar, `seed_lo`/`seed_hi`/`pos_i` 0-dim i32 scalars holding
/// the u64 seed (low, high) and the absolute position. Returns the `[1]`
/// i32 token array — the caller reads back 4 bytes.
const KEYED_SAMPLE_SOURCE: &str = include_str!("../../shaders/qwen35/keyed_sample.metal");

static KEYED_SAMPLE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn keyed_sample_kernel() -> &'static Option<MetalKernel> {
    KEYED_SAMPLE.get_or_init(|| {
        MetalKernel::new(
            "track_keyed_gumbel_sample",
            &["L", "inv_temperature", "seed_lo", "seed_hi", "pos_i"],
            &["out"],
            KEYED_SAMPLE_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

pub fn keyed_gumbel_sample(
    logits: &Array,
    inv_temperature: f32,
    seed: u64,
    position: u64,
    stream: &Stream,
) -> Option<Array> {
    if logits.dtype() != Dtype::Float32 || logits.rank() != 1 {
        return None;
    }
    let kernel = keyed_sample_kernel().as_ref()?;
    let inv_t = Array::from_f32(inv_temperature);
    let seed_lo = Array::from_int(seed as u32 as i32);
    let seed_hi = Array::from_int((seed >> 32) as u32 as i32);
    let pos = Array::from_int(position as u32 as i32);
    let inputs: [&Array; 5] = [logits, &inv_t, &seed_lo, &seed_hi, &pos];
    let template: [TemplateArg; 0] = [];
    let outs = [crate::ffi::OutputArg {
        shape: vec![1],
        dtype: Dtype::Int32,
    }];
    kernel
        .apply(&inputs, &template, (1, 1, 1), (1024, 1, 1), &outs, stream)
        .ok()?
        .into_iter()
        .next()
}

/// Force `get_or_init` on every kernel static in this module (see
/// `crate::models::qwen4::warm_kernels`).
pub fn warm() {
    let _ = sigmoid_mul_kernel();
    let _ = sigmoid_mul_tail_kernel();
    let _ = qk_norm_rope_kernel();
    let _ = keyed_sample_kernel();
}

#[cfg(test)]
mod parity {
    use super::*;
    use crate::ops;

    /// specs/13 re-audit pin: `track_swiglu2_packed` must be ROW-GENERIC.
    /// The original body read `gu[i]`/`gu[F+i]` single-row style — correct at
    /// B == 1, but at B > 1 every row >= 1 mixed row 0's gate with row 1's up
    /// (batched decode slots >= 1 emitted token soup from decode step 1).
    /// Word-exact vs the composed silu(gate) * up chain at B in {1, 2, 4}.
    #[test]
    fn swiglu2_packed_row_generic_matches_composed() {
        use crate::ops::indexing::IndexOp;
        use lisa_mlx::{Dtype, Stream};
        let stream = Stream::thread_local_or_default();
        let f = 512i32;
        for b in [1i32, 2, 4] {
            let n = (b * f) as usize;
            let mut seed = 0x5B00u64.wrapping_add(b as u64);
            let mut rnd = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608.0) * 20.0 - 10.0
            };
            let guv: Vec<f32> = (0..2 * n).map(|_| rnd()).collect();
            let gu = Array::from_slice(&guv, &[b, 2 * f])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let fused = swiglu2_packed(&gu, &stream).expect("packed kernel");
            // Composed reference per row: silu(gate) * up on the two halves.
            let gate = gu
                .index((.., 0..f))
                .copied()
                .unwrap();
            let up = gu.index((.., f..2 * f)).copied().unwrap();
            let reference = crate::ops::nn::silu(&gate)
                .unwrap()
                .multiply(&up)
                .unwrap();
            let fv = fused.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            let rv = reference.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            // ≤1 bf16 ULP relative: mlx_silu (exact_header) vs composed
            // nn::silu can differ by exactly 1 mantissa step on rare draws
            // (pre-existing production class — the B=1 serial decode runs
            // this kernel, goldens 310/310). The pin's target is ROW
            // ADDRESSING: a misread row produces garbage-scale errors.
            let worst = fv
                .iter()
                .zip(rv.iter())
                .map(|(x, y)| {
                    let d = (x - y).abs();
                    let scale = y.abs().max(1e-3);
                    d / scale
                })
                .fold(0.0f32, f32::max);
            // Tolerance class: NOT bit-exact — mlx_silu (exact_header) vs the
            // composed nn::silu chain differ by 1 bf16 mantissa step per op
            // and the silu*up product can stack ~2-3 steps on this draw
            // (pre-existing production class: the B=1 serial decode runs this
            // kernel, goldens 310/310). The pin's target is ROW ADDRESSING:
            // a misread row produces O(1) relative garbage, not 1e-2.
            assert!(
                worst <= 1.0 / 32.0,
                "b={b}: max relative diff {worst} (row addressing broken?)"
            );
        }
    }

    /// specs/04 pin: the attention-tail kernel over the UN-transposed sdpa
    /// output (`gate [b,s,h,hd]`, `up [b,h,s,hd]` -> merged `[b,s,h*hd]`)
    /// must be word-exact vs the transpose + `track_sigmoid_mul` chain it
    /// replaces. 65536 gate/up values per draw, plus specials.
    #[test]
    fn sigmoid_mul_tail_matches_transposed() {
        use super::{sigmoid_mul, sigmoid_mul_tail};
        use lisa_mlx::{Dtype, Stream};
        let stream = Stream::thread_local_or_default();
        let (b, s, h, hd) = (8i32, 4i32, 32i32, 64i32);
        let n = (b * s * h * hd) as usize;
        let mut seed = 51509811u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let specials = [0.0f32, -0.0, 1.0, -1.0, 20.0, -20.0, 88.0, -88.0];
        let mut gv: Vec<f32> = (0..n - specials.len()).map(|_| rnd() * 40.0 - 20.0).collect();
        gv.extend(specials);
        let mut uv: Vec<f32> = (0..n - specials.len()).map(|_| rnd() * 4.0 - 2.0).collect();
        uv.extend(specials);
        let g = Array::from_slice(&gv, &[b, s, h, hd])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let u = Array::from_slice(&uv, &[b, h, s, hd])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let fused = sigmoid_mul_tail(&g, &u, b, s, h, hd, &stream).expect("kernel");
        // Reference chain exactly as attention_tail built it before specs/04
        // (transpose then MATERIALIZE — reshape copies — then sigmoid_mul).
        let ut = u.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let ut = ut.copied().unwrap();
        let reference = sigmoid_mul(&g, &ut, &stream).expect("kernel");
        let f = fused
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[n as i32])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let c = reference
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[n as i32])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bad: Vec<usize> = (0..n)
            .filter(|&i| f[i].to_bits() != c[i].to_bits())
            .collect();
        assert!(
            bad.is_empty(),
            "{} mismatches, first at {:?}: fused={} reference={}",
            bad.len(),
            bad.first(),
            f[*bad.first().unwrap()],
            c[*bad.first().unwrap()]
        );
    }

    #[test]
    // RESOLVED (specs/04 §6→§8, commit aaf9d6d): under --test-threads=16 +
    // parallel GPU load this pin used to fail with BOTH sides reading
    // foreign/stale buffer values — a runtime readback race (lost wait +
    // fence-map UAF), not a numerics bug. The runtime synchronization fix
    // extinguished the whole flake family (parallel suite 3/3, detector
    // 8/8, pins 45/45 green). Isolated re-run no longer required.
    fn sigmoid_mul_matches_composed() {
        let stream = Stream::thread_local_or_default();
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let specials = [0.0f32, -0.0, 1.0, -1.0, 0.5, -0.5, 20.0, -20.0, 88.0, -88.0];
        let mut gv: Vec<f32> = (0..65536).map(|_| rnd() * 40.0 - 20.0).collect();
        gv.extend(specials);
        let n = gv.len();
        let g = Array::from_slice(&gv, &[n as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let mut av: Vec<f32> = (0..n - specials.len()).map(|_| rnd() * 4.0 - 2.0).collect();
        av.extend(specials);
        let a = Array::from_slice(&av, &[n as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let fused = sigmoid_mul(&g, &a, &stream).expect("kernel");
        let composed = ops::sigmoid(&g).unwrap().multiply(&a).unwrap();
        let f = fused
            .as_dtype(Dtype::Float32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let c = composed
            .as_dtype(Dtype::Float32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bad: Vec<usize> = (0..n)
            .filter(|&i| f[i].to_bits() != c[i].to_bits())
            .collect();
        if !bad.is_empty() {
            let gi = g.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            let ai = a.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            for &i in bad.iter().take(3) {
                let x = gi[i] as f64;
                let sig_f32 = 1.0 / (1.0 + (-x.abs()).exp());
                let sig = if x < 0.0 { sig_f32 } else { 1.0 - sig_f32 };
                let bf = |v: f64| half::bf16::from_f32(v as f32).to_f32();
                eprintln!(
                    "i={} g={} a={} | sig_f32={:.10} sig_bf16={} | fused={} composed={} | fused_exact={} composed_exact={}",
                    i, gi[i], ai[i], sig, bf(sig), f[i], c[i],
                    bf(bf(sig) as f64 * ai[i] as f64),
                    bf(sig as f64 * ai[i] as f64),
                );
            }
        }
        assert!(
            bad.is_empty(),
            "{} mismatches, first at {:?}: fused={} composed={}",
            bad.len(),
            bad.first(),
            f[*bad.first().unwrap()],
            c[*bad.first().unwrap()]
        );
    }

    #[test]
    fn qk_norm_rope_rows_matches_composed() {
        // specs/24 pin: the S-row verify kernel vs the composed
        // rms -> transpose -> rope_partial chain, on the STOCK interleaved
        // q|gate source layout. 4 draws x (7 rows x 28 heads x 256) = 200k+
        // values, PLUS a full-scale sweep: 8 draws x s=7 x (24+4 heads)
        // ~= 65k outputs per draw over varying rows.
        use crate::ops::indexing::{Ellipsis, IndexOp};
        use crate::{fast, ops};
        let stream = Stream::thread_local_or_default();
        let (hq, hk, hd, rd) = (24i32, 4i32, 256i32, 64i32);
        let eps = 1e-6f32;
        let mut seed = 4242u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let mk = |shape: &[i32], r: &mut dyn FnMut() -> f32| {
            let n: usize = shape.iter().product::<i32>() as usize;
            let v: Vec<f32> = (0..n).map(|_| r()).collect();
            Array::from_slice(&v, shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let qw = mk(&[hd], &mut rnd);
        let kw = mk(&[hd], &mut rnd);
        let rope = |x: &Array, cos: &Array, sin: &Array| -> Array {
            let rotated = x.index((Ellipsis, 0..rd));
            let half = (rd / 2) as i32;
            let x1 = rotated.index((Ellipsis, 0..half));
            let x2 = rotated.index((Ellipsis, half..rd));
            let neg_x2 = -x2;
            let swapped = ops::concatenate(&[&neg_x2, &x1], -1).unwrap();
            let out = rotated
                .multiply(cos)
                .unwrap()
                .add(&swapped.multiply(sin).unwrap())
                .unwrap();
            let rest = x.index((Ellipsis, rd..hd));
            ops::concatenate(&[&out, &rest], -1).unwrap()
        };
        for draw in 0..4 {
            let s = 7i32;
            // Interleaved q|gate: [s, hq, 2, hd].
            let qg = mk(&[s, hq, 2, hd], &mut rnd);
            let k = mk(&[s, hk, hd], &mut rnd);
            let cos = mk(&[s, rd], &mut rnd);
            let sin = mk(&[s, rd], &mut rnd);

            // Composed: de-interleave (contiguous halves), rms, transpose,
            // rope with the row's angle row.
            let q_half = qg.index((Ellipsis, 0, ..)).contiguous().unwrap();
            let gate_half = qg.index((Ellipsis, 1, ..)).contiguous().unwrap();
            let qn = fast::rms_norm(
                &q_half.reshape(&[1, s, hq, hd]).unwrap(),
                Some(&qw),
                eps,
            )
            .unwrap();
            let kn = fast::rms_norm(&k.reshape(&[1, s, hk, hd]).unwrap(), Some(&kw), eps).unwrap();
            let mut cq_rows = Vec::new();
            let mut ck_rows = Vec::new();
            for r in 0..s {
                let cos_r = cos.index((r, ..)).reshape(&[1, 1, 1, rd]).unwrap();
                let sin_r = sin.index((r, ..)).reshape(&[1, 1, 1, rd]).unwrap();
                cq_rows.push(rope(&qn.index((0, r, ..)).reshape(&[1, 1, hq, hd]).unwrap().transpose_axes(&[0, 2, 1, 3]).unwrap(), &cos_r, &sin_r));
                ck_rows.push(rope(&kn.index((0, r, ..)).reshape(&[1, 1, hk, hd]).unwrap().transpose_axes(&[0, 2, 1, 3]).unwrap(), &cos_r, &sin_r));
            }
            let cq = ops::concatenate(&cq_rows.iter().collect::<Vec<_>>(), 2)
                .unwrap(); // [1, hq, s, hd]
            let ck = ops::concatenate(&ck_rows.iter().collect::<Vec<_>>(), 2).unwrap();
            let cg = gate_half; // [s, hq, hd]

            let (fq, fk, fg) = qk_norm_rope_rows(
                &qg.reshape(&[s, hq * 2 * hd]).unwrap(),
                &k.reshape(&[s, hk * hd]).unwrap(),
                &qw,
                &kw,
                &cos,
                &sin,
                eps,
                hq,
                hk,
                s,
                &stream,
            )
            .expect("kernel");
            for (name, f, c) in [
                ("q", fq, cq),
                ("k", fk, ck),
                ("g", fg, cg.reshape(&[s, hq * hd]).unwrap()),
            ] {
                let fv = f.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
                let cv = c
                    .reshape(&[-1])
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let bad: Vec<usize> = (0..fv.len())
                    .filter(|&i| fv[i].to_bits() != cv[i].to_bits())
                    .collect();
                assert!(
                    bad.is_empty(),
                    "draw {draw} {name}: {} mismatches, first at {:?}: fused={} composed={}",
                    bad.len(),
                    bad.first(),
                    fv[*bad.first().unwrap()],
                    cv[*bad.first().unwrap()]
                );
            }
        }
    }

    #[test]
    fn qk_norm_rope_matches_composed() {
        use crate::ops::indexing::{Ellipsis, IndexOp};
        use crate::{fast, ops};
        let stream = Stream::thread_local_or_default();
        let (hq, hk, hd, rd) = (24i32, 4i32, 256i32, 64i32);
        let eps = 1e-6f32;
        let mut seed = 777u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let mk = |n: usize, r: &mut dyn FnMut() -> f32| {
            let v: Vec<f32> = (0..n).map(|_| r()).collect();
            Array::from_slice(&v, &[n as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let q = mk((hq * hd) as usize, &mut rnd);
        let k = mk((hk * hd) as usize, &mut rnd);
        let qw = mk(hd as usize, &mut rnd);
        let kw = mk(hd as usize, &mut rnd);
        let cos = mk(rd as usize, &mut rnd);
        let sin = mk(rd as usize, &mut rnd);

        // Composed chain: rms_norm -> transpose -> rope_partial (the engine's
        // core::norm::rope_partial, inlined for the test).
        let rope = |x: &Array| -> Array {
            let rotated = x.index((Ellipsis, 0..rd));
            let half = (rd / 2) as i32;
            let x1 = rotated.index((Ellipsis, 0..half));
            let x2 = rotated.index((Ellipsis, half..rd));
            let neg_x2 = -x2;
            let swapped = ops::concatenate(&[&neg_x2, &x1], -1).unwrap();
            let out = rotated.multiply(&cos).unwrap().add(&swapped.multiply(&sin).unwrap()).unwrap();
            let rest = x.index((Ellipsis, rd..hd));
            ops::concatenate(&[&out, &rest], -1).unwrap()
        };
        let qn = fast::rms_norm(&q.reshape(&[1, 1, hq, hd]).unwrap(), Some(&qw), eps).unwrap();
        let kn = fast::rms_norm(&k.reshape(&[1, 1, hk, hd]).unwrap(), Some(&kw), eps).unwrap();
        let cq = rope(&qn.transpose_axes(&[0, 2, 1, 3]).unwrap());
        let ck = rope(&kn.transpose_axes(&[0, 2, 1, 3]).unwrap());

        let (fq, fk) = qk_norm_rope(&q, &k, &qw, &kw, &cos, &sin, eps, hq, hk, &stream)
            .expect("kernel");
        for (name, f, c) in [("q", fq, &cq), ("k", fk, &ck)] {
            let fv = f.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
            let cv = c
                .reshape(&[-1])
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let bad: Vec<usize> = (0..fv.len())
                .filter(|&i| fv[i].to_bits() != cv[i].to_bits())
                .collect();
            if !bad.is_empty() {
                let qf = q.as_dtype(Dtype::Float32).unwrap().to_vec1::<f32>().unwrap();
                let rn = fast::rms_norm(&q.reshape(&[1, 1, hq, hd]).unwrap(), Some(&qw), eps)
                    .unwrap()
                    .reshape(&[-1])
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for &i in bad.iter().take(4).chain(bad.iter().find(|&&i| i >= 64)).take(6) {
                    eprintln!(
                        "[{name}] i={i} j={} src={:+} rms={:+} fused={:+} composed={:+}",
                        i % 256,
                        qf[i], rn[i], fv[i], cv[i]
                    );
                }
            }
            assert!(
                bad.is_empty(),
                "{name}: {} mismatches, first at {:?}: fused={} composed={}",
                bad.len(),
                bad.first(),
                fv[*bad.first().unwrap_or(&0)],
                cv[*bad.first().unwrap_or(&0)]
            );
        }
    }
}
