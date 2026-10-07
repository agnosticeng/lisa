use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
use lisa_mlx::{Array, Dtype, Stream};

use super::common::EXACT_HEADER;

/// PLE helper header (adds `mlx_maximum`/`mlx_sign`/abs/sqrt).
const PLE_HEADER_EXTRA: &str = include_str!("../../../shaders/qwen4/ple_header_extra.metal");

/// `track_ple_prod`: `norm_key(keyFlat) * norm_query(stream)` with the engine's
/// butterfly RMS on both, in one launch.
const PLE_PROD_SOURCE: &str = include_str!("../../../shaders/qwen4/ple_prod.metal");

/// `track_ple_gated`: the gate scalar chain, the gated value, norm_conv.
const PLE_GATED_SOURCE: &str = include_str!("../../../shaders/qwen4/ple_gated.metal");

/// `track_ple_conv`: dilated depthwise conv + silu + residual add.
const PLE_CONV_SOURCE: &str = include_str!("../../../shaders/qwen4/ple_conv.metal");

static PLE_PROD: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static PLE_GATED: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static PLE_CONV: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn ple_prod_kernel() -> &'static Option<MetalKernel> {
    PLE_PROD.get_or_init(|| {
        MetalKernel::new(
            "track_ple_prod",
            &["keyFlat", "stream", "kscale", "qscale", "eps"],
            &["prod"],
            PLE_PROD_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}
fn ple_gated_kernel() -> &'static Option<MetalKernel> {
    PLE_GATED.get_or_init(|| {
        let header = format!("{EXACT_HEADER}{PLE_HEADER_EXTRA}");
        MetalKernel::new(
            "track_ple_gated",
            &["g0", "value", "cscale", "divisor", "floorv", "eps"],
            &["gated", "normed"],
            PLE_GATED_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}
fn ple_conv_kernel() -> &'static Option<MetalKernel> {
    PLE_CONV.get_or_init(|| {
        MetalKernel::new(
            "track_ple_conv",
            &["full", "convw", "gated"],
            &["out"],
            PLE_CONV_SOURCE,
            &format!("{EXACT_HEADER}{PLE_HEADER_EXTRA}"),
            true,
            false,
        )
        .ok()
    })
}

/// `track_ple_prod`. `keyFlat`/`stream` `[B,S,W]`, `kScale`/`qScale` `[W]`,
/// `eps` `[1]` -> `prod` `[B,S,W]`.
#[allow(clippy::too_many_arguments)]
pub fn ple_prod(
    key_flat: &Array,
    stream: &Array,
    k_scale: &Array,
    q_scale: &Array,
    eps: &Array,
    hc: i32,
    hidden: i32,
    stream_: &Stream,
) -> Option<Array> {
    let kernel = ple_prod_kernel().as_ref()?;
    let b = key_flat.dim(0);
    let s = key_flat.dim(1);
    let w = hc * hidden;
    let inputs: [&Array; 5] = [key_flat, stream, k_scale, q_scale, eps];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("H", hidden),
        TemplateArg::Int("W", w),
        TemplateArg::Int("HC", hc),
    ];
    let outs = [OutputArg {
        shape: vec![b, s, w],
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(
            &inputs,
            &template,
            (hidden / 4, hc, b * s),
            (hidden / 4, 1, 1),
            &outs,
            stream_,
        )
        .ok()?
        .into_iter()
        .next()
}

/// `track_ple_gated`. `g0` `[B,S,HC]`, `value` `[B,S,H]`, `cScale` `[W]`,
/// `divisor`/`floor` `[1]`, `eps` `[1]` -> `(gated, normed)` `[B,S,W]`.
#[allow(clippy::too_many_arguments)]
pub fn ple_gated(
    g0: &Array,
    value: &Array,
    c_scale: &Array,
    divisor: &Array,
    floor: &Array,
    eps: &Array,
    hc: i32,
    hidden: i32,
    stream_: &Stream,
) -> Option<(Array, Array)> {
    let kernel = ple_gated_kernel().as_ref()?;
    let b = value.dim(0);
    let s = value.dim(1);
    let w = hc * hidden;
    let inputs: [&Array; 6] = [g0, value, c_scale, divisor, floor, eps];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("H", hidden),
        TemplateArg::Int("W", w),
        TemplateArg::Int("HC", hc),
    ];
    let outs = [
        OutputArg {
            shape: vec![b, s, w],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, s, w],
            dtype: Dtype::Bfloat16,
        },
    ];
    let r = kernel
        .apply(
            &inputs,
            &template,
            (hidden / 4, hc, b * s),
            (hidden / 4, 1, 1),
            &outs,
            stream_,
        )
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?))
}

/// `track_ple_conv`. `full` `[B,N+S,W]`, `convw` `[W,KC]`, `gated` `[B,S,W]`
/// -> `[B,S,W]`.
pub fn ple_conv(
    full: &Array,
    conv_w: &Array,
    gated: &Array,
    dilation: i32,
    stream_: &Stream,
) -> Option<Array> {
    let kernel = ple_conv_kernel().as_ref()?;
    let b = gated.dim(0);
    let s = gated.dim(1);
    let w = gated.dim(2);
    let kc = conv_w.dim(1);
    let nin = full.dim(1);
    let inputs: [&Array; 3] = [full, conv_w, gated];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("W", w),
        TemplateArg::Int("S", s),
        TemplateArg::Int("KC", kc),
        TemplateArg::Int("DIL", dilation),
        TemplateArg::Int("NIN", nin),
    ];
    let outs = [OutputArg {
        shape: vec![b, s, w],
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(&inputs, &template, (w, s, b), (256, 1, 1), &outs, stream_)
        .ok()?
        .into_iter()
        .next()
}
/// `track_swiglu`: `mlx_silu(gate) * up` over a fused `[.., 2F]` gate|up array.
// --- PLE S=1 fusion: track_ple_prepare_fuse2 + track_ple_convolution_fuse2 ---

const PLE_FUSE_HEADER: &str = include_str!("../../../shaders/qwen4/ple_fuse_header.metal");

const PLE_PREPARE_SOURCE: &str = include_str!("../../../shaders/qwen4/ple_prepare.metal");

const PLE_CONV_FUSE_SOURCE: &str = include_str!("../../../shaders/qwen4/ple_conv_fuse.metal");

static PLE_PREPARE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static PLE_CONV_FUSE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn ple_prepare_kernel() -> &'static Option<MetalKernel> {
    PLE_PREPARE.get_or_init(|| {
        let h = format!("{EXACT_HEADER}{PLE_FUSE_HEADER}");
        MetalKernel::new(
            "track_ple_prepare_fuse2",
            &[
                "key",
                "query",
                "value",
                "keyScale",
                "queryScale",
                "convScale",
                "convState",
            ],
            &["gated", "full"],
            PLE_PREPARE_SOURCE,
            &h,
            true,
            false,
        )
        .ok()
    })
}
fn ple_conv_fuse_kernel() -> &'static Option<MetalKernel> {
    PLE_CONV_FUSE.get_or_init(|| {
        MetalKernel::new(
            "track_ple_convolution_fuse2",
            &["full", "weight", "gated"],
            &["out"],
            PLE_CONV_FUSE_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// The S=1 PLE fusion: `(full [1,10,10240], output [1,1,10240])`.
#[allow(clippy::too_many_arguments)]
pub fn ple_fuse2(
    key: &Array,
    stream_arr: &Array,
    value: &Array,
    key_scale: &Array,
    query_scale: &Array,
    conv_scale: &Array,
    conv_state: &Array,
    conv_w: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let prep = ple_prepare_kernel().as_ref()?;
    let conv = ple_conv_fuse_kernel().as_ref()?;
    let p_inputs: [&Array; 7] = [
        key,
        stream_arr,
        value,
        key_scale,
        query_scale,
        conv_scale,
        conv_state,
    ];
    let p_tmpl = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
        TemplateArg::Int("DIVISOR_BITS", (2560f32.sqrt()).to_bits() as i32),
    ];
    let p_outs = [
        OutputArg {
            shape: vec![1, 1, 10240],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![1, 10, 10240],
            dtype: Dtype::Bfloat16,
        },
    ];
    let r = prep
        .apply(
            &p_inputs,
            &p_tmpl,
            (640, 4, 1),
            (640, 1, 1),
            &p_outs,
            stream,
        )
        .ok()?;
    let mut it = r.into_iter();
    let gated = it.next()?;
    let full = it.next()?;
    let c_inputs: [&Array; 3] = [&full, conv_w, &gated];
    let c_tmpl = [TemplateArg::Dtype("InT", Dtype::Bfloat16)];
    let c_outs = [OutputArg {
        shape: vec![1, 1, 10240],
        dtype: Dtype::Bfloat16,
    }];
    let out = conv
        .apply(
            &c_inputs,
            &c_tmpl,
            (32, 1, 4 * 10240),
            (32, 1, 4),
            &c_outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()?;
    Some((full, out))
}

/// Force `get_or_init` on every kernel static in this module (see
/// `crate::models::qwen4::warm_kernels`).
pub fn warm() {
    let _ = ple_prod_kernel();
    let _ = ple_gated_kernel();
    let _ = ple_conv_kernel();
    let _ = ple_prepare_kernel();
    let _ = ple_conv_fuse_kernel();
}
