use objc2_metal::MTLSize;

use crate::array::Array;
use crate::runtime::{ComputePipeline, ConstVal};

use super::compile::{type_string, type_to_name};
use super::{DType, Device, Tensor};
use super::{NAX_GEMM_HEADER, NAX_QUANT_HEADER};

const GLM53_AFFINE_Q8_PREFILL_DS4_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_affine_q8_prefill_ds4.metal");
const GLM53_AFFINE_Q2_MPP_DS4_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_affine_q2_mpp_ds4.metal");
const GLM53_MOE_Q2_PAIR_GROUPED_DS4_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_moe_q2_pair_grouped_ds4.metal");
const GLM53_MOE_Q2_MM_GROUPED_DS4_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_moe_q2_mm_grouped_ds4.metal");
const GLM53_MOE_GROUPED_SWIGLU_WEIGHT_SOURCE: &str =
    include_str!("../shaders/common/elementwise/glm53_moe_grouped_swiglu_weight.metal");
const GLM53_MOE_Q2_DOWN_GROUPED_DS4_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_moe_q2_down_grouped_ds4.metal");
use crate::error::Result;

/// MLX `steel_matmul_regular_axpby_nax` (`matmul.cpp:178`), no batch, no
/// out-source. `a` is `[M,K]`; `b` is `[K,N]` (or `[N,K]` when `transpose_b`).
/// Source for one `steel_gemm_fused_nax` instantiation (the NAX GEMM).
pub fn matmul_nax_source(
    base_name: &str,
    a_ty: &str,
    out_ty: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    wm: usize,
    wn: usize,
    transpose_a: bool,
    transpose_b: bool,
) -> String {
    let _ = (a_ty, out_ty);
    // MLX instantiates the NAX GEMM on the *output* dtype's Metal type
    // (`get_type_string(out.dtype())`), so f32 matmuls use `gemm<float, ...>`.
    let metal_ty = match out_ty {
        "float32" => "float",
        "bfloat16" => "bfloat16_t",
        "float16" => "half",
        other => other,
    };
    format!(
        "{NAX_GEMM_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{base_name}\", gemm, {metal_ty}, {bm}, {bn}, {bk}, {wm}, {wn}, {}, {})\n",
        if transpose_a { "true" } else { "false" },
        if transpose_b { "true" } else { "false" },
    )
}

pub fn matmul_nax(
    device: &Device,
    a: &Tensor,
    b: &Tensor,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<Tensor> {
    let mdev = device;
    let ar = a.rank();
    let br = b.rank();
    if ar != 2 || br != 2 {
        crate::bail!("matmul_nax: 2-D only");
    }
    let m = a.dims()[0];
    let k = a.dims()[1];
    let n = if transpose_b {
        b.dims()[0]
    } else {
        b.dims()[1]
    };
    let lda = if transpose_a { m } else { k } as i32;
    let ldb = if transpose_b { k } else { n } as i32;
    let ldd = n as i32;

    let a_ty = type_to_name(a.dtype())?;
    let out_ty = type_to_name(a.dtype())?;
    let devc = mdev.architecture_name().chars().last().unwrap_or('s');
    let (mut bm, bn, mut bk, mut wm, wn) = (128usize, 128usize, 512usize, 4usize, 4usize);
    if devc == 's' || devc == 'c' || devc == 'd' {
        bk = if k >= 8192 && k > m + n { 64 } else { 256 };
        bm = 64;
        wm = 2;
    }
    let base_name = format!(
        "steel_gemm_fused_nax_{}{}_{a_ty}_{out_ty}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}",
        if transpose_a { 't' } else { 'n' },
        if transpose_b { 't' } else { 'n' },
    );
    let align_m = m % bm == 0;
    let align_n = n % bn == 0;
    let align_k = k % bk == 0;
    let pipeline = compile_nax_jit(
        device,
        &matmul_nax_source(
            &base_name,
            a_ty,
            out_ty,
            bm,
            bn,
            bk,
            wm,
            wn,
            transpose_a,
            transpose_b,
        ),
        &base_name,
        &[
            (10, false),
            (100, false),
            (110, false),
            (200, align_m),
            (201, align_n),
            (202, align_k),
        ],
    )?;
    let count = m * n;
    let obuf =
        mdev.buffer_for_full_gpu_write((count) as usize * (a.dtype()).size_of(), "gemm_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &vec![m, n], a.dtype());

    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let swizzle_log: i32 = if devc == 's' || devc == 'c' || devc == 'd' {
        2
    } else if tm <= 3 {
        0
    } else {
        1
    };
    let mut p: Vec<u8> = Vec::with_capacity(72);
    for v in [
        m as i32, n as i32, k as i32, lda, ldb, ldd, tn as i32, tm as i32,
    ] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    for v in [0i64, 0, 0] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    for v in [swizzle_log, (k / bk) as i32, 0i32] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    p.resize(72, 0);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, a)?;
    bind(1, b)?;
    enc.set_output(3, Some(&obuf), 0);
    enc.set_bytes_directly(4, p.len(), p.as_ptr().cast());

    let tile = 1usize << swizzle_log;
    let tm2 = tm.div_ceil(tile);
    let tn2 = tn * tile;
    enc.dispatch_groups_size(
        MTLSize {
            width: tn2,
            height: tm2,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: wn,
            depth: wm,
        },
    );
    Ok(out)
}

/// JIT-compile an NAX kernel from its flattened source and memoize the
/// pipeline in the runtime. Compile options `set_compile_options`: Safe + Precise
/// + language version; function constants supplied at pipeline creation.
pub(super) fn compile_nax_jit(
    device: &Device,
    source: &str,
    name: &str,
    consts: &[(usize, bool)],
) -> Result<ComputePipeline> {
    compile_nax_jit_with_math(device, source, name, consts, crate::runtime::Math::Safe)
}

fn compile_nax_jit_with_math(
    device: &Device,
    source: &str,
    name: &str,
    consts: &[(usize, bool)],
    math: crate::runtime::Math,
) -> Result<ComputePipeline> {
    let consts: Vec<(usize, ConstVal)> = consts
        .iter()
        .map(|&(i, v)| (i, ConstVal::Bool(v)))
        .collect();
    device.compile_with_constants(source, name, math, &consts)
}

/// Source for one `affine_gather_qmm_rhs_nax` instantiation (transpose path).
pub fn gather_qmm_rhs_nax_source(kname: &str, bm: usize, group_size: i32, bits: i32) -> String {
    format!(
        "{NAX_QUANT_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{kname}\", affine_gather_qmm_rhs_nax, bfloat, {group_size}, {bits}, {bm}, 64, 64, 2, 2, true)\n"
    )
}

/// MLX `gather_qmm_rhs_nax` (`quantized.cpp:1450`): the NAX affine gather-GEMM
/// with right-sorted indices. This is the branch the sorted MoE expert matmuls
/// actually take (`M == 1`, `B >= 16`, `B / E >= 4`, `right_sorted`), because
/// the engine feeds a `[rows, 1, K]` lhs so `M` collapses to 1 and `B` becomes
/// the row count. `indices` is one expert id per output row, walked in order so
/// consecutive rows sharing an expert reuse the loaded weight block.
///
/// `x` is `[M, 1, K]` (or `[M, K]`), `w` is `[E, N, K]` quantized, and
/// `out_dims` is the caller's output shape. `M`/`N`/`K` are derived from the
/// tensors, matching the arguments `GatherQMM::eval_gpu` passes down.
pub fn gather_qmm_rhs_nax(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    indices: &Tensor,
    out_dims: Vec<usize>,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let k = *x.dims().last().unwrap();
    let n = *out_dims.last().unwrap();
    let m = x.elem_count() / k;
    let total: usize = out_dims.iter().product();

    if !matches!(indices.dtype(), DType::U32 | DType::I32) {
        crate::bail!("gather_qmm_rhs_nax: indices must be U32/I32");
    }

    let e = w.elem_count() / w.dims()[w.rank() - 1] / w.dims()[w.rank() - 2];
    let bm = if m / e < 64 { 32usize } else { 64usize };
    let (bn, bk, wm, wn) = (64usize, 64usize, 2usize, 2usize);
    let align_m = m % bm == 0;
    let align_n = n % bn == 0;
    let align_k = k % bk == 0;

    let ty = type_to_name(x.dtype())?;
    let kname = format!(
        "affine_gather_qmm_rhs_nax_nt_{ty}_gs_{group_size}_b_{bits}_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}"
    );
    let pipeline = compile_nax_jit(
        device,
        &gather_qmm_rhs_nax_source(&kname, bm, group_size, bits),
        &kname,
        &[(200, align_m), (201, align_n), (202, align_k)],
    )?;

    let obuf = mdev.buffer_for_full_gpu_write(
        (total) as usize * (x.dtype()).size_of(),
        "gather_qmm_rhs_nax_out",
    )?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims, x.dtype());

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };

    bind(0, x)?;
    bind(1, w)?;
    bind(2, scales)?;
    bind(3, biases)?;
    bind(4, indices)?;
    enc.set_output(5, Some(&obuf), 0);
    for (i, v) in [m as i32, n as i32, k as i32].into_iter().enumerate() {
        enc.set_bytes(6 + i, &v);
    }

    enc.dispatch_groups_size(
        MTLSize {
            width: n.div_ceil(bn),
            height: m.div_ceil(bm),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: wn,
            depth: wm,
        },
    );
    Ok(out)
}

/// Source for one `affine_qmm_t_nax` instantiation.
pub fn qmm_nax_source(
    kname: &str,
    dtype: &str,
    group_size: i32,
    bits: i32,
    aligned_n: bool,
    batched: bool,
    bm: usize,
    bk: usize,
    bn: usize,
    wm: usize,
    wn: usize,
) -> String {
    format!(
        "{NAX_QUANT_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{kname}\", affine_qmm_t_nax, {dtype}, {group_size}, {bits}, {}, {}, {bm}, {bk}, {bn}, {wm}, {wn})\n",
        if aligned_n { "true" } else { "false" },
        if batched { "true" } else { "false" },
    )
}

/// MLX `qmm_nax` (`quantized.cpp:814`): the NAX affine quantized matmul used
/// for `M > 1` when NAX is available, `transpose`, and `K % 64 == 0`. It is the
/// non-split counterpart of `affine_qmm_splitk`. `x`/`w` are 2-D and `B == 1`
/// (the engine always flattens to `[M, K]`).
pub fn qmm_nax(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let (wm, wn) = (2usize, 2usize);
    let bm = if m <= 32 { 32usize } else { 64usize };
    let (bn, bk) = (64usize, 64usize);
    let aligned = n % 64 == 0;
    let batched = false;
    let ty = type_to_name(x.dtype())?;
    let kname = format!(
        "affine_qmm_t_nax_{ty}_gs_{group_size}_b_{bits}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}{}{}",
        if aligned { "_alN_true" } else { "_alN_false" },
        if batched { "_batch_1" } else { "_batch_0" },
    );
    let pipeline = compile_nax_jit(
        device,
        &qmm_nax_source(
            &kname,
            type_string(x.dtype())?,
            group_size,
            bits,
            aligned,
            batched,
            bm,
            bk,
            bn,
            wm,
            wn,
        ),
        &kname,
        &[],
    )?;

    let count = m * n;
    let obuf =
        mdev.buffer_for_full_gpu_write((count) as usize * (x.dtype()).size_of(), "qmm_nax_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &vec![m, n], x.dtype());

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&obuf), 0);
    for (i, v) in [k as i32, n as i32, m as i32].into_iter().enumerate() {
        enc.set_bytes(5 + i, &v);
    }

    enc.dispatch_groups_size(
        MTLSize {
            width: n.div_ceil(bn),
            height: m.div_ceil(bm),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: wn,
            depth: wm,
        },
    );
    Ok(out)
}

/// Kept for the existing prefill comparison harness. The candidate now uses
/// DS4's full crossover, including generic SIMD-matrix tails and BF16 metadata.
pub fn glm53_affine_q8_mpp_prefill(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    glm53_affine_q8_prefill(device, x, w, scales, biases)
}

#[repr(C)]
struct Glm53Q8MmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    padding0: u32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    padding1: u32,
}

const _: [(); 88] = [(); std::mem::size_of::<Glm53Q8MmArgs>()];

/// Candidate port of DS4's Q8 matrix schedule with original MLX affine storage.
/// Rows 17+ use the 64x32 generic kernel, except aligned MPP rows on capable
/// devices. Unaligned batches of 192+ split into an MPP prefix and generic tail.
/// The caller keeps rows <=16 on DS4's existing affine matvec path.
pub fn glm53_affine_q8_prefill(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    let (m, k) = x.dims2();
    let (n, packed_k) = w.dims2();
    if x.dtype() != DType::F32
        || w.dtype() != DType::U32
        || !matches!(scales.dtype(), DType::F32 | DType::BF16)
        || biases.dtype() != scales.dtype()
        || m <= 16
        || n == 0
        || k == 0
        || packed_k.checked_mul(4) != Some(k)
        || k % 64 != 0
        || [m, k, n].into_iter().any(|dim| dim > i32::MAX as usize)
    {
        crate::bail!(
            "glm53_affine_q8_prefill: unsupported geometry or dtype (M={m}, K={k}, N={n})"
        );
    }
    if scales.dims2() != (n, k / 64) || biases.dims2() != (n, k / 64) {
        crate::bail!("glm53_affine_q8_prefill: affine arrays must have shape [N, K/64]");
    }
    for tensor in [x, w, scales, biases] {
        if !tensor.buffer_and_layout().1.is_contiguous() {
            crate::bail!("glm53_affine_q8_prefill: non-contiguous input");
        }
    }
    let mpp_rows = if device.nax() && n % 64 == 0 && (m % 32 == 0 || m >= 192) {
        m - m % 32
    } else {
        0
    };
    let tile_rows = if mpp_rows % 128 == 0 {
        128
    } else if mpp_rows % 64 == 0 {
        64
    } else {
        32
    };
    // Source assembly is once per process, never a per-dispatch source hash.
    let source = if device.nax() {
        static MPP_SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        MPP_SOURCE
            .get_or_init(|| {
                format!("#define DS4_METAL_HAS_TENSOR 1\n{GLM53_AFFINE_Q8_PREFILL_DS4_SOURCE}")
            })
            .as_str()
    } else {
        GLM53_AFFINE_Q8_PREFILL_DS4_SOURCE
    };
    let output_bytes = m
        .checked_mul(n)
        .and_then(|size| size.checked_mul(4))
        .ok_or_else(|| crate::error::Error::msg("GLM Q8 output size overflow"))?;
    let output_buffer = device.buffer_for_full_gpu_write(output_bytes, "glm53_q8_prefill_out")?;
    let output = Array::from_parts(device, output_buffer.clone(), &[m, n], DType::F32);
    let affine = [
        k as u32,
        n as u32,
        m as u32,
        8,
        64,
        u32::from(scales.dtype() == DType::BF16),
    ];
    for (row0, rows, mpp) in [(0, mpp_rows, true), (mpp_rows, m - mpp_rows, false)] {
        if rows == 0 {
            continue;
        }
        let name = if mpp {
            format!("kernel_glm53_mlx_affine_q8_mpp_n{tile_rows}")
        } else {
            "kernel_glm53_mlx_affine_q8_mm".to_owned()
        };
        let boundary = n % 64 != 0 || rows % 32 != 0;
        let pipeline = compile_nax_jit_with_math(
            device,
            source,
            &name,
            &[(0, false), (1, !mpp && boundary)],
            crate::runtime::Math::CompilerDefault,
        )?;
        let args = Glm53Q8MmArgs {
            ne00: k as i32,
            ne02: 1,
            nb01: k as u64,
            nb02: (k * n) as u64,
            nb03: (k * n) as u64,
            ne12: 1,
            padding0: 0,
            nb10: 4,
            nb11: (k * 4) as u64,
            nb12: (k * rows * 4) as u64,
            nb13: (k * rows * 4) as u64,
            ne0: n as i32,
            ne1: rows as i32,
            r2: 1,
            r3: 1,
            padding1: 0,
        };
        let guard = device.commands.encoder()?;
        let enc = guard.encoder();
        enc.set_pipeline(&pipeline);
        enc.set_bytes(0, &args);
        for (index, tensor, extra) in [
            (1, w, 0),
            (2, x, row0 * k * 4),
            (4, scales, 0),
            (5, biases, 0),
        ] {
            let (buffer, layout) = tensor.buffer_and_layout();
            enc.set_input(
                index,
                Some(buffer),
                layout.offset * tensor.dtype().size_of() + extra,
            );
        }
        enc.set_output(3, Some(&output_buffer), row0 * n * 4);
        enc.set_bytes(6, &affine);
        enc.set_threadgroup_memory_length(0, if mpp || boundary { 8192 } else { 6144 });
        enc.dispatch_groups_size(
            MTLSize {
                width: if mpp {
                    rows / tile_rows
                } else {
                    rows.div_ceil(32)
                },
                height: n.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(output)
}

/// DS4-style MPP prefill tile for MLX affine-Q2 weights. The caller supplies
/// one contiguous expert matrix: weights `[N,K/4]`, scales/biases `[N,K/128]`,
/// and activations `[M,K]`. Each dequantized weight is rounded to half before
/// matrix accumulation, matching the weight staging class used by DS4's
/// grouped routed-expert kernels.
pub fn glm53_affine_q2_mpp_prefill(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    let (m, k) = x.dims2();
    let (n, packed_k) = w.dims2();
    let metadata_ty = match scales.dtype() {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        other => crate::bail!("glm53_affine_q2_mpp_prefill: unsupported metadata type {other:?}"),
    };
    if x.dtype() != DType::F32
        || biases.dtype() != scales.dtype()
        || packed_k * 4 != k
        || k % 128 != 0
        || m == 0
        || m % 32 != 0
        || n == 0
        || n % 64 != 0
    {
        crate::bail!(
            "glm53_affine_q2_mpp_prefill: unsupported shape/dtype (M={m}, K={k}, N={n}, packedK={packed_k}, metadata={:?})",
            scales.dtype()
        );
    }
    if scales.dims2() != (n, k / 128) || biases.dims2() != (n, k / 128) {
        crate::bail!("glm53_affine_q2_mpp_prefill: affine arrays must have shape [N, K/128]");
    }
    for tensor in [x, w, scales, biases] {
        if !tensor.buffer_and_layout().1.is_contiguous() {
            crate::bail!("glm53_affine_q2_mpp_prefill: non-contiguous input");
        }
    }

    let name = format!("glm53_affine_q2_mpp_ds4_{metadata_ty}_nr1_32_nr0_64_nk32");
    let source = format!("{NAX_QUANT_HEADER}\n{GLM53_AFFINE_Q2_MPP_DS4_SOURCE}");
    let pipeline = compile_nax_jit(device, &source, &name, &[])?;
    let output_buffer =
        device.buffer_for_full_gpu_write(m * n * std::mem::size_of::<f32>(), &name)?;
    let output = Array::from_parts(device, output_buffer.clone(), &[m, n], DType::F32);

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [w, scales, biases, x].into_iter().enumerate() {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
    }
    enc.set_output(4, Some(&output_buffer), 0);
    for (index, value) in [k as i32, n as i32, m as i32].into_iter().enumerate() {
        enc.set_bytes(5 + index, &value);
    }
    enc.dispatch_groups_size(
        MTLSize {
            width: m / 32,
            height: n / 64,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

/// Bind a full-write GLM output to an exclusively owned model workspace.
/// The caller must keep its workspace lease through the last consumer encode.
/// Each returned tensor gets a fresh host cache; a workspace view must never
/// escape as a persistent result or diagnostic capture.
fn glm53_grouped_output(
    device: &Device,
    reuse: Option<&Tensor>,
    shape: &[usize],
    dtype: DType,
    bytes: usize,
    name: &str,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    if let Some(output) = reuse {
        let (buffer, layout) = output.buffer_and_layout();
        if !std::sync::Arc::ptr_eq(device, output.device())
            || output.dims() != shape
            || output.dtype() != dtype
            || !layout.is_contiguous()
            || layout.offset != 0
            || buffer.length() < bytes
            || inputs
                .iter()
                .any(|input| input.buffer_and_layout().0.id() == buffer.id())
        {
            crate::bail!("{name}: incompatible or aliased grouped MoE workspace output");
        }
        Ok(Array::from_parts(device, buffer.clone(), shape, dtype))
    } else {
        let buffer = device.buffer_for_full_gpu_write(bytes, name)?;
        Ok(Array::from_parts(device, buffer, shape, dtype))
    }
}

/// DS4-style expert-major affine-Q2 gate/up simdgroup prefill. `route_ids`,
/// counts, work count, and work items come from the GLM expert-route map. The
/// address table contains nine GPU addresses per expert; `keepalive` retains
/// indirect buffers when no model-wide residency set is attached.
#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_q2_pair_grouped_ds4_prefill(
    device: &Device,
    address_table: &Tensor,
    x: &Tensor,
    route_ids: &Tensor,
    counts: &Tensor,
    work_count: &Tensor,
    work_items: &Tensor,
    route_weights: &Tensor,
    keepalive: &[&Tensor],
    residency_set: Option<&crate::runtime::ResidencySet>,
    metadata_dtype: DType,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    mid_dim: usize,
    limit: f32,
) -> Result<Tensor> {
    glm53_moe_q2_pair_grouped_ds4_prefill_with_output(
        device,
        address_table,
        x,
        route_ids,
        counts,
        work_count,
        work_items,
        route_weights,
        keepalive,
        residency_set,
        metadata_dtype,
        n_tokens,
        top_k,
        n_experts,
        mid_dim,
        limit,
        None,
    )
}

/// Reuse a model-owned full-write output; its lease must cover every consumer.
#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_q2_pair_grouped_ds4_prefill_with_output(
    device: &Device,
    address_table: &Tensor,
    x: &Tensor,
    route_ids: &Tensor,
    counts: &Tensor,
    work_count: &Tensor,
    work_items: &Tensor,
    route_weights: &Tensor,
    keepalive: &[&Tensor],
    residency_set: Option<&crate::runtime::ResidencySet>,
    metadata_dtype: DType,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    mid_dim: usize,
    limit: f32,
    reuse_output: Option<&Tensor>,
) -> Result<Tensor> {
    let metadata_ty = match metadata_dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        other => {
            crate::bail!(
                "glm53_moe_q2_pair_grouped_ds4_prefill: unsupported metadata dtype {other:?}"
            )
        }
    };
    if address_table.rank() != 1
        || address_table.dtype() != DType::I64
        || address_table.dim(0) != n_experts.checked_mul(9).unwrap_or(usize::MAX)
        || x.rank() != 2
        || x.dtype() != DType::F32
        || x.dim(0) != n_tokens
        || route_ids.rank() != 1
        || route_ids.dtype() != DType::I32
        || route_ids.dim(0) != n_experts.checked_mul(n_tokens).unwrap_or(usize::MAX)
        || counts.rank() != 1
        || counts.dtype() != DType::I32
        || counts.dim(0) != n_experts
        || work_count.rank() != 1
        || work_count.dtype() != DType::I32
        || work_count.dim(0) != 1
        || work_items.rank() != 1
        || work_items.dtype() != DType::I32
        || route_weights.rank() != 1
        || route_weights.dtype() != DType::F32
        || n_tokens == 0
        || top_k == 0
        || n_experts == 0
        || mid_dim == 0
        || mid_dim % 64 != 0
        || n_tokens > i32::MAX as usize
        || top_k > i32::MAX as usize
        || n_experts > i32::MAX as usize
        || mid_dim > i32::MAX as usize
        || x.dim(1) == 0
        || x.dim(1) % 128 != 0
        || x.dim(1) > i32::MAX as usize
        || [n_tokens, top_k, n_experts, mid_dim, x.dim(1)]
            .into_iter()
            .any(|dim| dim > i32::MAX as usize)
    {
        crate::bail!("glm53_moe_q2_pair_grouped_ds4_prefill: unsupported route/weight shape");
    }
    let n_slots = route_weights.dim(0);
    let in_dim = x.dim(1);
    if n_slots != n_tokens.checked_mul(top_k).unwrap_or(usize::MAX) {
        crate::bail!(
            "glm53_moe_q2_pair_grouped_ds4_prefill: route weights do not match [tokens, top_k]"
        );
    }
    let max_work_items = (n_slots / 32 + usize::from(n_slots % 32 != 0))
        .checked_add(n_experts)
        .ok_or_else(|| {
            crate::error::Error::msg("glm53_moe_q2_pair_grouped_ds4_prefill: work size overflow")
        })?;
    if work_items.dim(0) < max_work_items.saturating_mul(2) {
        crate::bail!("glm53_moe_q2_pair_grouped_ds4_prefill: route work buffer is too small");
    }
    let output_elems = n_slots.checked_mul(mid_dim).ok_or_else(|| {
        crate::error::Error::msg("glm53_moe_q2_pair_grouped_ds4_prefill: output size overflow")
    })?;
    let output_bytes = output_elems
        .checked_mul(std::mem::size_of::<half::f16>())
        .ok_or_else(|| {
            crate::error::Error::msg("glm53_moe_q2_pair_grouped_ds4_prefill: output byte overflow")
        })?;
    let name = format!("glm53_moe_q2_pair_grouped_ds4_{metadata_ty}_nr1_32_nr0_64_nk32");
    let source = format!("{NAX_QUANT_HEADER}\n{GLM53_MOE_Q2_PAIR_GROUPED_DS4_SOURCE}");
    let pipeline = compile_nax_jit(device, &source, &name, &[])?;
    let output = glm53_grouped_output(
        device,
        reuse_output,
        &[n_slots, mid_dim],
        DType::F16,
        output_bytes,
        &name,
        &[
            address_table,
            x,
            route_ids,
            counts,
            work_count,
            work_items,
            route_weights,
        ],
    )?;
    let output_buffer = output.buffer_and_layout().0.clone();

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    if let Some(residency_set) = residency_set {
        enc.use_residency_set(residency_set);
    } else {
        for tensor in keepalive {
            let (buffer, _) = tensor.buffer_and_layout();
            enc.retain_until_completion(buffer);
        }
    }
    for (index, tensor) in [
        address_table,
        x,
        route_ids,
        counts,
        work_count,
        work_items,
        route_weights,
    ]
    .into_iter()
    .enumerate()
    {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
    }
    enc.set_output(7, Some(&output_buffer), 0);
    for (index, value) in [n_tokens, top_k, n_experts, mid_dim, in_dim]
        .into_iter()
        .enumerate()
    {
        enc.set_bytes(8 + index, &(value as i32));
    }
    enc.set_bytes(13, &limit);
    enc.dispatch_groups_size(
        MTLSize {
            width: max_work_items,
            height: mid_dim / 64,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

/// DS4's split expert-major affine-Q2 gate or up projection. The caller
/// dispatches both projections independently, then applies the separate
/// route-weighted F16 SwiGLU stage used by DS4's grouped MoE path.
#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_q2_mm_grouped_ds4_prefill(
    device: &Device,
    address_table: &Tensor,
    x: &Tensor,
    route_ids: &Tensor,
    counts: &Tensor,
    work_count: &Tensor,
    work_items: &Tensor,
    keepalive: &[&Tensor],
    residency_set: Option<&crate::runtime::ResidencySet>,
    metadata_dtype: DType,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    out_dim: usize,
    is_up: bool,
    vector_load: bool,
) -> Result<Tensor> {
    glm53_moe_q2_mm_grouped_ds4_prefill_with_output(
        device,
        address_table,
        x,
        route_ids,
        counts,
        work_count,
        work_items,
        keepalive,
        residency_set,
        metadata_dtype,
        n_tokens,
        top_k,
        n_experts,
        out_dim,
        is_up,
        vector_load,
        None,
    )
}

/// Reuse a model-owned full-write output; its lease must cover every consumer.
#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_q2_mm_grouped_ds4_prefill_with_output(
    device: &Device,
    address_table: &Tensor,
    x: &Tensor,
    route_ids: &Tensor,
    counts: &Tensor,
    work_count: &Tensor,
    work_items: &Tensor,
    keepalive: &[&Tensor],
    residency_set: Option<&crate::runtime::ResidencySet>,
    metadata_dtype: DType,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    out_dim: usize,
    is_up: bool,
    vector_load: bool,
    reuse_output: Option<&Tensor>,
) -> Result<Tensor> {
    let metadata_ty = match metadata_dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        other => {
            crate::bail!(
                "glm53_moe_q2_mm_grouped_ds4_prefill: unsupported metadata dtype {other:?}"
            )
        }
    };
    if address_table.rank() != 1
        || address_table.dtype() != DType::I64
        || address_table.dim(0) != n_experts.checked_mul(9).unwrap_or(usize::MAX)
        || x.rank() != 2
        || x.dtype() != DType::F32
        || x.dim(0) != n_tokens
        || route_ids.rank() != 1
        || route_ids.dtype() != DType::I32
        || route_ids.dim(0) != n_experts.checked_mul(n_tokens).unwrap_or(usize::MAX)
        || counts.rank() != 1
        || counts.dtype() != DType::I32
        || counts.dim(0) != n_experts
        || work_count.rank() != 1
        || work_count.dtype() != DType::I32
        || work_count.dim(0) != 1
        || work_items.rank() != 1
        || work_items.dtype() != DType::I32
        || n_tokens == 0
        || top_k == 0
        || n_experts == 0
        || out_dim == 0
        || out_dim % 64 != 0
        || n_tokens > i32::MAX as usize
        || top_k > i32::MAX as usize
        || n_experts > i32::MAX as usize
        || out_dim > i32::MAX as usize
        || x.dim(1) == 0
        || x.dim(1) % 128 != 0
        || x.dim(1) > i32::MAX as usize
    {
        crate::bail!("glm53_moe_q2_mm_grouped_ds4_prefill: unsupported route/weight shape");
    }
    let n_slots = n_tokens.checked_mul(top_k).unwrap_or(usize::MAX);
    let in_dim = x.dim(1);
    let max_work_items = (n_slots / 32 + usize::from(n_slots % 32 != 0))
        .checked_add(n_experts)
        .ok_or_else(|| {
            crate::error::Error::msg("glm53_moe_q2_mm_grouped_ds4_prefill: work size overflow")
        })?;
    if work_items.dim(0) < max_work_items.saturating_mul(2) {
        crate::bail!("glm53_moe_q2_mm_grouped_ds4_prefill: route work buffer is too small");
    }
    let output_elems = n_slots.checked_mul(out_dim).ok_or_else(|| {
        crate::error::Error::msg("glm53_moe_q2_mm_grouped_ds4_prefill: output size overflow")
    })?;
    let output_bytes = output_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            crate::error::Error::msg("glm53_moe_q2_mm_grouped_ds4_prefill: output byte overflow")
        })?;
    let direction = if is_up { "up" } else { "gate" };
    let load_variant = if vector_load { "" } else { "_scalar" };
    let name = format!(
        "glm53_moe_q2_mm_grouped_ds4_{metadata_ty}_{direction}{load_variant}_nr1_32_nr0_64_nk32"
    );
    let source = format!("{NAX_QUANT_HEADER}\n{GLM53_MOE_Q2_MM_GROUPED_DS4_SOURCE}");
    let pipeline = compile_nax_jit(device, &source, &name, &[])?;
    let output = glm53_grouped_output(
        device,
        reuse_output,
        &[n_slots, out_dim],
        DType::F32,
        output_bytes,
        &name,
        &[address_table, x, route_ids, counts, work_count, work_items],
    )?;
    let output_buffer = output.buffer_and_layout().0.clone();

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    if let Some(residency_set) = residency_set {
        enc.use_residency_set(residency_set);
    } else {
        for tensor in keepalive {
            let (buffer, _) = tensor.buffer_and_layout();
            enc.retain_until_completion(buffer);
        }
    }
    for (index, tensor) in [address_table, x, route_ids, counts, work_count, work_items]
        .into_iter()
        .enumerate()
    {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
    }
    enc.set_output(6, Some(&output_buffer), 0);
    for (index, value) in [n_tokens, top_k, n_experts, out_dim, in_dim]
        .into_iter()
        .enumerate()
    {
        enc.set_bytes(7 + index, &(value as i32));
    }
    enc.dispatch_groups_size(
        MTLSize {
            width: max_work_items,
            height: out_dim / 64,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

/// Apply DS4's separate clamped, route-weighted SwiGLU and store the compact
/// F16 intermediate consumed by grouped down projection.
pub fn glm53_moe_grouped_swiglu_weight_prefill(
    device: &Device,
    gate: &Tensor,
    up: &Tensor,
    route_weights: &Tensor,
    n_slots: usize,
    width: usize,
    limit: f32,
) -> Result<Tensor> {
    // DS4 compiles its monolithic Metal library with Metal's default math
    // settings. Match that mode for its weighted SwiGLU schedule.
    glm53_moe_grouped_swiglu_weight_prefill_with_math(
        device,
        gate,
        up,
        route_weights,
        n_slots,
        width,
        limit,
        crate::runtime::Math::CompilerDefault,
    )
}

fn glm53_moe_grouped_swiglu_weight_prefill_with_math(
    device: &Device,
    gate: &Tensor,
    up: &Tensor,
    route_weights: &Tensor,
    n_slots: usize,
    width: usize,
    limit: f32,
    math: crate::runtime::Math,
) -> Result<Tensor> {
    glm53_moe_grouped_swiglu_weight_prefill_impl(
        device,
        gate,
        up,
        route_weights,
        n_slots,
        width,
        limit,
        math,
        None,
    )
}

/// Reuse the model-owned F16 activation output without changing DS4 math.
#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_grouped_swiglu_weight_prefill_with_output(
    device: &Device,
    gate: &Tensor,
    up: &Tensor,
    route_weights: &Tensor,
    n_slots: usize,
    width: usize,
    limit: f32,
    reuse_output: Option<&Tensor>,
) -> Result<Tensor> {
    glm53_moe_grouped_swiglu_weight_prefill_impl(
        device,
        gate,
        up,
        route_weights,
        n_slots,
        width,
        limit,
        crate::runtime::Math::CompilerDefault,
        reuse_output,
    )
}

#[allow(clippy::too_many_arguments)]
fn glm53_moe_grouped_swiglu_weight_prefill_impl(
    device: &Device,
    gate: &Tensor,
    up: &Tensor,
    route_weights: &Tensor,
    n_slots: usize,
    width: usize,
    limit: f32,
    math: crate::runtime::Math,
    reuse_output: Option<&Tensor>,
) -> Result<Tensor> {
    if gate.rank() != 2
        || up.rank() != 2
        || gate.dtype() != DType::F32
        || up.dtype() != DType::F32
        || gate.dims2() != (n_slots, width)
        || up.dims2() != (n_slots, width)
        || route_weights.rank() != 1
        || route_weights.dtype() != DType::F32
        || route_weights.dim(0) != n_slots
        || n_slots == 0
        || width == 0
        || n_slots > i32::MAX as usize
        || width > i32::MAX as usize
    {
        crate::bail!("glm53_moe_grouped_swiglu_weight_prefill: unsupported activation shape");
    }
    for tensor in [gate, up, route_weights] {
        if !tensor.buffer_and_layout().1.is_contiguous() {
            crate::bail!("glm53_moe_grouped_swiglu_weight_prefill: non-contiguous input");
        }
    }
    let source = format!("{NAX_QUANT_HEADER}\n{GLM53_MOE_GROUPED_SWIGLU_WEIGHT_SOURCE}");
    let pipeline = compile_nax_jit_with_math(
        device,
        &source,
        "glm53_moe_grouped_swiglu_weight_f16",
        &[],
        math,
    )?;
    let output_elems = n_slots.checked_mul(width).ok_or_else(|| {
        crate::error::Error::msg("glm53_moe_grouped_swiglu_weight_prefill: output size overflow")
    })?;
    let output_bytes = output_elems
        .checked_mul(std::mem::size_of::<half::f16>())
        .ok_or_else(|| {
            crate::error::Error::msg(
                "glm53_moe_grouped_swiglu_weight_prefill: output byte overflow",
            )
        })?;
    let output = glm53_grouped_output(
        device,
        reuse_output,
        &[n_slots, width],
        DType::F16,
        output_bytes,
        "glm53_moe_grouped_swiglu_weight_f16",
        &[gate, up, route_weights],
    )?;
    let output_buffer = output.buffer_and_layout().0.clone();
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [gate, up, route_weights].into_iter().enumerate() {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
        enc.retain_until_completion(buffer);
    }
    enc.set_output(3, Some(&output_buffer), 0);
    enc.set_bytes(4, &(n_slots as i32));
    enc.set_bytes(5, &(width as i32));
    enc.set_bytes(6, &limit);
    let thread_count = 256usize.min(width);
    enc.dispatch_groups_size(
        MTLSize {
            width: n_slots,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_count,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

/// DS4-style expert-major affine-Q2 down projection over F16 SwiGLU rows.
/// Route-map tensors use the same expert-major layout as the gate/up kernel.
#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_q2_down_grouped_ds4_prefill(
    device: &Device,
    address_table: &Tensor,
    x: &Tensor,
    route_ids: &Tensor,
    counts: &Tensor,
    work_count: &Tensor,
    work_items: &Tensor,
    keepalive: &[&Tensor],
    residency_set: Option<&crate::runtime::ResidencySet>,
    metadata_dtype: DType,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    out_dim: usize,
    vector_load: bool,
    output_tile: usize,
) -> Result<Tensor> {
    glm53_moe_q2_down_grouped_ds4_prefill_with_output(
        device,
        address_table,
        x,
        route_ids,
        counts,
        work_count,
        work_items,
        keepalive,
        residency_set,
        metadata_dtype,
        n_tokens,
        top_k,
        n_experts,
        out_dim,
        vector_load,
        output_tile,
        None,
    )
}

/// Reuse a model-owned full-write output; its lease must cover every consumer.
#[allow(clippy::too_many_arguments)]
pub fn glm53_moe_q2_down_grouped_ds4_prefill_with_output(
    device: &Device,
    address_table: &Tensor,
    x: &Tensor,
    route_ids: &Tensor,
    counts: &Tensor,
    work_count: &Tensor,
    work_items: &Tensor,
    keepalive: &[&Tensor],
    residency_set: Option<&crate::runtime::ResidencySet>,
    metadata_dtype: DType,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    out_dim: usize,
    vector_load: bool,
    output_tile: usize,
    reuse_output: Option<&Tensor>,
) -> Result<Tensor> {
    let metadata_ty = match metadata_dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        other => {
            crate::bail!(
                "glm53_moe_q2_down_grouped_ds4_prefill: unsupported metadata dtype {other:?}"
            )
        }
    };
    if address_table.rank() != 1
        || address_table.dtype() != DType::I64
        || address_table.dim(0) != n_experts.checked_mul(9).unwrap_or(usize::MAX)
        || x.rank() != 2
        || x.dtype() != DType::F16
        || x.dim(0) != n_tokens.checked_mul(top_k).unwrap_or(usize::MAX)
        || route_ids.rank() != 1
        || route_ids.dtype() != DType::I32
        || route_ids.dim(0) != n_experts.checked_mul(n_tokens).unwrap_or(usize::MAX)
        || counts.rank() != 1
        || counts.dtype() != DType::I32
        || counts.dim(0) != n_experts
        || work_count.rank() != 1
        || work_count.dtype() != DType::I32
        || work_count.dim(0) != 1
        || work_items.rank() != 1
        || work_items.dtype() != DType::I32
        || n_tokens == 0
        || top_k == 0
        || n_experts == 0
        || out_dim == 0
        || !matches!(output_tile, 64 | 128)
        || out_dim % output_tile != 0
        || x.dim(1) == 0
        || x.dim(1) % 128 != 0
        || [n_tokens, top_k, n_experts, out_dim, x.dim(1)]
            .into_iter()
            .any(|dim| dim > i32::MAX as usize)
    {
        crate::bail!("glm53_moe_q2_down_grouped_ds4_prefill: unsupported route/weight shape");
    }
    let n_slots = n_tokens.checked_mul(top_k).unwrap_or(usize::MAX);
    let in_dim = x.dim(1);
    let max_work_items = (n_slots / 32 + usize::from(n_slots % 32 != 0))
        .checked_add(n_experts)
        .ok_or_else(|| {
            crate::error::Error::msg("glm53_moe_q2_down_grouped_ds4_prefill: work size overflow")
        })?;
    if work_items.dim(0) < max_work_items.saturating_mul(2) {
        crate::bail!("glm53_moe_q2_down_grouped_ds4_prefill: route work buffer is too small");
    }
    let output_elems = n_slots.checked_mul(out_dim).ok_or_else(|| {
        crate::error::Error::msg("glm53_moe_q2_down_grouped_ds4_prefill: output size overflow")
    })?;
    let output_bytes = output_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            crate::error::Error::msg("glm53_moe_q2_down_grouped_ds4_prefill: output byte overflow")
        })?;
    let load_variant = if vector_load { "" } else { "_scalar" };
    let name = format!(
        "glm53_moe_q2_down_grouped_ds4_{metadata_ty}{load_variant}_nr1_32_nr0_{output_tile}_nk32"
    );
    let source = format!("{NAX_QUANT_HEADER}\n{GLM53_MOE_Q2_DOWN_GROUPED_DS4_SOURCE}");
    let pipeline = compile_nax_jit(device, &source, &name, &[])?;
    let output = glm53_grouped_output(
        device,
        reuse_output,
        &[n_slots, out_dim],
        DType::F32,
        output_bytes,
        &name,
        &[address_table, x, route_ids, counts, work_count, work_items],
    )?;
    let output_buffer = output.buffer_and_layout().0.clone();

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    if let Some(residency_set) = residency_set {
        enc.use_residency_set(residency_set);
    } else {
        for tensor in keepalive {
            let (buffer, _) = tensor.buffer_and_layout();
            enc.retain_until_completion(buffer);
        }
    }
    for (index, tensor) in [address_table, x, route_ids, counts, work_count, work_items]
        .into_iter()
        .enumerate()
    {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
    }
    enc.set_output(6, Some(&output_buffer), 0);
    for (index, value) in [n_tokens, top_k, n_experts, out_dim, in_dim]
        .into_iter()
        .enumerate()
    {
        enc.set_bytes(7 + index, &(value as i32));
    }
    enc.dispatch_groups_size(
        MTLSize {
            width: max_work_items,
            height: out_dim / output_tile,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

#[cfg(test)]
mod glm53_moe_grouped_ds4_tests {
    use super::*;
    use crate::ops::Array as OpsArray;

    #[test]
    fn grouped_q2_pair_ds4_matches_route_map_and_half_weight_oracle() {
        if !crate::device::metal_available() {
            eprintln!("[glm53-moe-q2-ds4] skipping: no Metal device available");
            return;
        }
        const TOKENS: usize = 32;
        const TOP_K: usize = 2;
        const EXPERTS: usize = 4;
        const MID: usize = 128;
        const IN: usize = 256;
        const GROUPS: usize = IN / 128;

        let mut expert_arrays = Vec::<OpsArray>::new();
        let mut addresses = Vec::<i64>::new();
        let mut packed_weights = vec![vec![0u8; MID * (IN / 4)]; EXPERTS * 2];
        let mut scales = vec![vec![0.0f32; MID * GROUPS]; EXPERTS * 2];
        let mut biases = vec![vec![0.0f32; MID * GROUPS]; EXPERTS * 2];
        for expert in 0..EXPERTS {
            for projection in 0..2 {
                let matrix = expert * 2 + projection;
                for row in 0..MID {
                    for k in 0..IN {
                        let code = ((expert * 3 + projection * 2 + row + k) % 4) as u8;
                        packed_weights[matrix][row * (IN / 4) + k / 4] |= code << ((k % 4) * 2);
                    }
                    for group in 0..GROUPS {
                        let index = row * GROUPS + group;
                        scales[matrix][index] =
                            0.0006 + ((expert + row + group) % 5) as f32 * 0.00004;
                        biases[matrix][index] =
                            0.00008 + ((projection + row + group) % 3) as f32 * 0.00001;
                    }
                }
            }
            let gw =
                OpsArray::from_slice(&packed_weights[expert * 2], &[MID as i32, (IN / 4) as i32]);
            let gs = OpsArray::from_slice(&scales[expert * 2], &[MID as i32, GROUPS as i32]);
            let gb = OpsArray::from_slice(&biases[expert * 2], &[MID as i32, GROUPS as i32]);
            let uw = OpsArray::from_slice(
                &packed_weights[expert * 2 + 1],
                &[MID as i32, (IN / 4) as i32],
            );
            let us = OpsArray::from_slice(&scales[expert * 2 + 1], &[MID as i32, GROUPS as i32]);
            let ub = OpsArray::from_slice(&biases[expert * 2 + 1], &[MID as i32, GROUPS as i32]);
            let gate_addresses = [
                gw.gpu_address().unwrap() as i64,
                gs.gpu_address().unwrap() as i64,
                gb.gpu_address().unwrap() as i64,
            ];
            let up_addresses = [
                uw.gpu_address().unwrap() as i64,
                us.gpu_address().unwrap() as i64,
                ub.gpu_address().unwrap() as i64,
            ];
            addresses.extend_from_slice(&gate_addresses);
            addresses.extend_from_slice(&up_addresses);
            addresses.extend_from_slice(&gate_addresses);
            expert_arrays.extend([gw, gs, gb, uw, us, ub]);
        }

        let selected_values = (0..TOKENS)
            .flat_map(|token| [(token % EXPERTS) as i32, ((token + 1) % EXPERTS) as i32])
            .collect::<Vec<_>>();
        let route_weights = (0..TOKENS * TOP_K)
            .map(|slot| if slot % 2 == 0 { 0.4f32 } else { 0.6f32 })
            .collect::<Vec<_>>();
        let x_values = (0..TOKENS * IN)
            .map(|i| 0.02 + (i * 17 % 127) as f32 * 0.0001)
            .collect::<Vec<_>>();
        let address_table = OpsArray::from_slice(&addresses, &[(EXPERTS * 9) as i32]);
        let selected = OpsArray::from_slice(&selected_values, &[(TOKENS * TOP_K) as i32]);
        let weights = OpsArray::from_slice(&route_weights, &[(TOKENS * TOP_K) as i32]);
        let x = OpsArray::from_slice(&x_values, &[TOKENS as i32, IN as i32]);
        let stream = crate::ops::Stream::thread_local_or_default();
        let (route_ids, counts, work_count, work_items) =
            crate::models::glm53::kernels::moe_expert_route_map(
                &selected, TOKENS, TOP_K, EXPERTS, 32, &stream,
            )
            .expect("route map should dispatch");
        let keepalive = expert_arrays
            .iter()
            .map(|array| &array.t)
            .collect::<Vec<_>>();
        let actual = glm53_moe_q2_pair_grouped_ds4_prefill(
            x.t.device(),
            &address_table.t,
            &x.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &weights.t,
            &keepalive,
            None,
            DType::F32,
            TOKENS,
            TOP_K,
            EXPERTS,
            MID,
            10.0,
        )
        .expect("grouped Q2 DS4 simdgroup kernel should compile and dispatch")
        .to_vec::<half::f16>()
        .expect("read grouped Q2 F16 output")
        .into_iter()
        .map(f32::from)
        .collect::<Vec<_>>();

        let split_gate = glm53_moe_q2_mm_grouped_ds4_prefill(
            x.t.device(),
            &address_table.t,
            &x.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::F32,
            TOKENS,
            TOP_K,
            EXPERTS,
            MID,
            false,
            true,
        )
        .expect("split grouped Q2 gate matmul should compile and dispatch");
        let split_up = glm53_moe_q2_mm_grouped_ds4_prefill(
            x.t.device(),
            &address_table.t,
            &x.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::F32,
            TOKENS,
            TOP_K,
            EXPERTS,
            MID,
            true,
            true,
        )
        .expect("split grouped Q2 up matmul should compile and dispatch");
        let split_pair = glm53_moe_grouped_swiglu_weight_prefill(
            x.t.device(),
            &split_gate,
            &split_up,
            &weights.t,
            TOKENS * TOP_K,
            MID,
            10.0,
        )
        .expect("split grouped Q2 SwiGLU should compile and dispatch")
        .to_vec::<half::f16>()
        .expect("read split grouped Q2 F16 output")
        .into_iter()
        .map(f32::from)
        .collect::<Vec<_>>();
        assert_eq!(
            split_pair, actual,
            "split DS4 schedule must match fused pair"
        );

        let down_input_values = (0..TOKENS * TOP_K)
            .flat_map(|slot| {
                let token = slot / TOP_K;
                x_values[token * IN..(token + 1) * IN].iter().copied()
            })
            .collect::<Vec<_>>();
        let down_input =
            OpsArray::from_slice(&down_input_values, &[(TOKENS * TOP_K) as i32, IN as i32])
                .as_dtype(crate::ops::Dtype::Float16)
                .expect("convert down input to F16");
        let actual_down = glm53_moe_q2_down_grouped_ds4_prefill(
            down_input.t.device(),
            &address_table.t,
            &down_input.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::F32,
            TOKENS,
            TOP_K,
            EXPERTS,
            MID,
            true,
            64,
        )
        .expect("grouped Q2 DS4 down kernel should compile and dispatch")
        .to_vec1::<f32>()
        .expect("read grouped Q2 DS4 down output");
        let actual_down_wide = glm53_moe_q2_down_grouped_ds4_prefill(
            down_input.t.device(),
            &address_table.t,
            &down_input.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::F32,
            TOKENS,
            TOP_K,
            EXPERTS,
            MID,
            true,
            128,
        )
        .expect("wide grouped Q2 DS4 down kernel should compile and dispatch")
        .to_vec1::<f32>()
        .expect("read wide grouped Q2 DS4 down output");
        assert_eq!(
            actual_down_wide, actual_down,
            "128-row tiling must preserve every 64-row DS4 output bit-for-bit"
        );

        let mut expected = vec![0.0f32; TOKENS * TOP_K * MID];
        for slot in 0..TOKENS * TOP_K {
            let expert = selected_values[slot] as usize;
            let token = slot / TOP_K;
            for row in 0..MID {
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for k in 0..IN {
                    let gate_code =
                        (packed_weights[expert * 2][row * (IN / 4) + k / 4] >> ((k % 4) * 2)) & 3;
                    let up_code = (packed_weights[expert * 2 + 1][row * (IN / 4) + k / 4]
                        >> ((k % 4) * 2))
                        & 3;
                    let metadata = row * GROUPS + k / 128;
                    let gate_weight = f32::from(half::f16::from_f32(
                        f32::from(gate_code) * scales[expert * 2][metadata]
                            + biases[expert * 2][metadata],
                    ));
                    let up_weight = f32::from(half::f16::from_f32(
                        f32::from(up_code) * scales[expert * 2 + 1][metadata]
                            + biases[expert * 2 + 1][metadata],
                    ));
                    let input = f32::from(half::f16::from_f32(x_values[token * IN + k]));
                    gate += input * gate_weight;
                    up += input * up_weight;
                }
                let gate = gate.min(10.0);
                let up = up.clamp(-10.0, 10.0);
                expected[slot * MID + row] = f32::from(half::f16::from_f32(
                    gate / (1.0 + (-gate).exp()) * up * route_weights[slot],
                ));
            }
        }
        let (max_abs, rel_l2) = actual.iter().zip(&expected).fold(
            (0.0f32, (0.0f64, 0.0f64)),
            |(max_abs, (error_sq, reference_sq)), (&got, &want)| {
                let delta = got - want;
                (
                    max_abs.max(delta.abs()),
                    (
                        error_sq + f64::from(delta) * f64::from(delta),
                        reference_sq + f64::from(want) * f64::from(want),
                    ),
                )
            },
        );
        let rel_l2 = (rel_l2.0 / rel_l2.1.max(f64::MIN_POSITIVE)).sqrt();
        eprintln!("[glm53-moe-q2-ds4] max_abs={max_abs:e} rel_l2={rel_l2:e}");
        assert!(max_abs < 2.0e-6, "grouped Q2 DS4 max abs error {max_abs:e}");
        assert!(rel_l2 < 4.0e-3, "grouped Q2 DS4 rel L2 error {rel_l2:e}");

        let mut expected_down = vec![0.0f32; TOKENS * TOP_K * MID];
        for slot in 0..TOKENS * TOP_K {
            let expert = selected_values[slot] as usize;
            for row in 0..MID {
                for k in 0..IN {
                    let code =
                        (packed_weights[expert * 2][row * (IN / 4) + k / 4] >> ((k % 4) * 2)) & 3;
                    let metadata = row * GROUPS + k / 128;
                    let weight = f32::from(half::f16::from_f32(
                        f32::from(code) * scales[expert * 2][metadata]
                            + biases[expert * 2][metadata],
                    ));
                    let input = f32::from(half::f16::from_f32(down_input_values[slot * IN + k]));
                    expected_down[slot * MID + row] += input * weight;
                }
            }
        }
        let (down_max_abs, down_l2) = actual_down.iter().zip(&expected_down).fold(
            (0.0f32, (0.0f64, 0.0f64)),
            |(max_abs, (error_sq, reference_sq)), (&got, &want)| {
                let delta = got - want;
                (
                    max_abs.max(delta.abs()),
                    (
                        error_sq + f64::from(delta) * f64::from(delta),
                        reference_sq + f64::from(want) * f64::from(want),
                    ),
                )
            },
        );
        let down_rel_l2 = (down_l2.0 / down_l2.1.max(f64::MIN_POSITIVE)).sqrt();
        eprintln!("[glm53-moe-q2-ds4-down] max_abs={down_max_abs:e} rel_l2={down_rel_l2:e}");
        assert!(
            down_max_abs < 2.0e-6,
            "grouped Q2 DS4 down max abs error {down_max_abs:e}"
        );
        assert!(
            down_rel_l2 < 1.0e-5,
            "grouped Q2 DS4 down rel L2 error {down_rel_l2:e}"
        );
    }

    fn fixture_f32(path: &std::path::Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn fixture_i32(path: &std::path::Path) -> Vec<i32> {
        std::fs::read(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn fixture_bf16(path: &std::path::Path) -> Vec<half::bf16> {
        std::fs::read(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
            .chunks_exact(2)
            .map(|bytes| half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect()
    }

    fn fixture_error(actual: &[f32], expected: &[f32]) -> (f32, f64, usize) {
        assert_eq!(actual.len(), expected.len());
        let (max_abs, (error_sq, reference_sq, exact)) = actual.iter().zip(expected).fold(
            (0.0f32, (0.0f64, 0.0f64, 0usize)),
            |(max_abs, (error_sq, reference_sq, exact)), (&got, &want)| {
                let delta = got - want;
                (
                    max_abs.max(delta.abs()),
                    (
                        error_sq + f64::from(delta) * f64::from(delta),
                        reference_sq + f64::from(want) * f64::from(want),
                        exact + usize::from(got.to_bits() == want.to_bits()),
                    ),
                )
            },
        );
        (
            max_abs,
            (error_sq / reference_sq.max(f64::MIN_POSITIVE)).sqrt(),
            exact,
        )
    }

    #[test]
    #[ignore = "requires the exported MLX Q2 fixture and corrected DS4 sidecars"]
    fn grouped_q2_real_fixture_matches_ds4_production_simdgroup_oracle() {
        if !crate::device::metal_available() {
            eprintln!("[glm53-moe-q2-real] skipping: no Metal device available");
            return;
        }
        const TOKENS: usize = 96;
        const TOP_K: usize = 8;
        const EXPERTS: usize = 288;
        const HIDDEN: usize = 4096;
        const INTER: usize = 2048;

        let exports = std::path::PathBuf::from(
            std::env::var("LISA_GLM53_Q2_EXPORT_ROOT").expect("set export root"),
        );
        let capture = std::path::PathBuf::from(
            std::env::var("LISA_GLM53_Q2_CAPTURE_ROOT").expect("set capture root"),
        );
        let ds4_prefix = std::path::PathBuf::from(
            std::env::var("LISA_GLM53_Q2_DS4_OUTPUT_PREFIX").expect("set DS4 output prefix"),
        );

        let input_values = fixture_f32(&capture.join("input.f32"));
        let selected_values = fixture_i32(&capture.join("selected_all.i32"));
        let route_weight_values = fixture_f32(&capture.join("route_weights_all.f32"));
        assert_eq!(input_values.len(), TOKENS * HIDDEN);
        assert_eq!(selected_values.len(), TOKENS * TOP_K);
        assert_eq!(route_weight_values.len(), TOKENS * TOP_K);

        let mut expert_arrays = Vec::<OpsArray>::new();
        let mut address_values = vec![0i64; EXPERTS * 9];
        for expert in 0..EXPERTS {
            if !selected_values.contains(&(expert as i32)) {
                continue;
            }
            for (projection, input_dim, output_dim) in [
                ("gate_proj", HIDDEN, INTER),
                ("up_proj", HIDDEN, INTER),
                ("down_proj", INTER, HIDDEN),
            ] {
                let dir = exports.join(format!("expert_{expert}/{projection}"));
                let packed = std::fs::read(dir.join("weight.u32"))
                    .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()));
                let scales = fixture_bf16(&dir.join("scales.bf16"));
                let biases = fixture_bf16(&dir.join("biases.bf16"));
                let groups = input_dim / 128;
                assert_eq!(packed.len(), output_dim * (input_dim / 4));
                assert_eq!(scales.len(), output_dim * groups);
                assert_eq!(biases.len(), output_dim * groups);

                let weight =
                    OpsArray::from_slice(&packed, &[(output_dim * (input_dim / 4)) as i32]);
                let scale = OpsArray::from_slice(&scales, &[(output_dim as i32), groups as i32]);
                let bias = OpsArray::from_slice(&biases, &[(output_dim as i32), groups as i32]);
                let index = expert * 9
                    + match projection {
                        "gate_proj" => 0,
                        "up_proj" => 3,
                        "down_proj" => 6,
                        _ => unreachable!(),
                    };
                address_values[index] = weight.gpu_address().expect("weight GPU address") as i64;
                address_values[index + 1] = scale.gpu_address().expect("scale GPU address") as i64;
                address_values[index + 2] = bias.gpu_address().expect("bias GPU address") as i64;
                expert_arrays.extend([weight, scale, bias]);
            }
        }

        let address_table = OpsArray::from_slice(&address_values, &[(EXPERTS * 9) as i32]);
        let input = OpsArray::from_slice(&input_values, &[TOKENS as i32, HIDDEN as i32]);
        let selected = OpsArray::from_slice(&selected_values, &[(TOKENS * TOP_K) as i32]);
        let route_weights = OpsArray::from_slice(&route_weight_values, &[(TOKENS * TOP_K) as i32]);
        let stream = crate::ops::Stream::thread_local_or_default();
        let (route_ids, counts, work_count, work_items) =
            crate::models::glm53::kernels::moe_expert_route_map(
                &selected, TOKENS, TOP_K, EXPERTS, 32, &stream,
            )
            .expect("route-map dispatch");
        let keepalive = expert_arrays
            .iter()
            .map(|array| &array.t)
            .collect::<Vec<_>>();

        let split_gate = glm53_moe_q2_mm_grouped_ds4_prefill(
            input.t.device(),
            &address_table.t,
            &input.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::BF16,
            TOKENS,
            TOP_K,
            EXPERTS,
            INTER,
            false,
            true,
        )
        .expect("split grouped gate simdgroup dispatch");
        let split_up = glm53_moe_q2_mm_grouped_ds4_prefill(
            input.t.device(),
            &address_table.t,
            &input.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::BF16,
            TOKENS,
            TOP_K,
            EXPERTS,
            INTER,
            true,
            true,
        )
        .expect("split grouped up simdgroup dispatch");
        let scalar_gate = glm53_moe_q2_mm_grouped_ds4_prefill(
            input.t.device(),
            &address_table.t,
            &input.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::BF16,
            TOKENS,
            TOP_K,
            EXPERTS,
            INTER,
            false,
            false,
        )
        .expect("scalar grouped gate simdgroup dispatch");
        let scalar_up = glm53_moe_q2_mm_grouped_ds4_prefill(
            input.t.device(),
            &address_table.t,
            &input.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::BF16,
            TOKENS,
            TOP_K,
            EXPERTS,
            INTER,
            true,
            false,
        )
        .expect("scalar grouped up simdgroup dispatch");
        let pair_f16 = OpsArray::new(
            glm53_moe_grouped_swiglu_weight_prefill(
                input.t.device(),
                &split_gate,
                &split_up,
                &route_weights.t,
                TOKENS * TOP_K,
                INTER,
                10.0,
            )
            .expect("split grouped SwiGLU dispatch"),
        );
        let scalar_pair_f16 = OpsArray::new(
            glm53_moe_grouped_swiglu_weight_prefill(
                input.t.device(),
                &scalar_gate,
                &scalar_up,
                &route_weights.t,
                TOKENS * TOP_K,
                INTER,
                10.0,
            )
            .expect("scalar split grouped SwiGLU dispatch"),
        );
        let down = glm53_moe_q2_down_grouped_ds4_prefill(
            pair_f16.t.device(),
            &address_table.t,
            &pair_f16.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::BF16,
            TOKENS,
            TOP_K,
            EXPERTS,
            HIDDEN,
            true,
            64,
        )
        .expect("grouped down simdgroup dispatch");
        let down_array = OpsArray::new(down);
        let down_wide = glm53_moe_q2_down_grouped_ds4_prefill(
            pair_f16.t.device(),
            &address_table.t,
            &pair_f16.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::BF16,
            TOKENS,
            TOP_K,
            EXPERTS,
            HIDDEN,
            true,
            128,
        )
        .expect("wide grouped down simdgroup dispatch");
        let down_wide_array = OpsArray::new(down_wide);
        let wide_values = down_wide_array
            .to_vec1::<f32>()
            .expect("read wide down F32");
        let standard_values = down_array.to_vec1::<f32>().expect("read standard down F32");
        let first_mismatch = wide_values
            .iter()
            .zip(&standard_values)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        let max_delta = wide_values
            .iter()
            .zip(&standard_values)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0, f64::max);
        assert!(
            first_mismatch.is_none(),
            "128-row down tile differs from DS4 64-row tile: max_abs={max_delta}, first={first_mismatch:?}"
        );
        let scalar_down = glm53_moe_q2_down_grouped_ds4_prefill(
            scalar_pair_f16.t.device(),
            &address_table.t,
            &scalar_pair_f16.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            &keepalive,
            None,
            DType::BF16,
            TOKENS,
            TOP_K,
            EXPERTS,
            HIDDEN,
            false,
            64,
        )
        .expect("scalar grouped down simdgroup dispatch");
        let scalar_down_array = OpsArray::new(scalar_down);
        let sum = crate::models::glm53::kernels::moe_sum8(&down_array, TOKENS, HIDDEN, &stream)
            .expect("ordered sum8 dispatch");

        let actual_pair = pair_f16
            .t
            .to_vec::<half::f16>()
            .expect("read pair F16")
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        let actual_down = down_array.to_vec1::<f32>().expect("read down F32");
        let scalar_actual_pair = scalar_pair_f16
            .t
            .to_vec::<half::f16>()
            .expect("read scalar pair F16")
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        let scalar_actual_down = scalar_down_array
            .to_vec1::<f32>()
            .expect("read scalar down F32");
        assert!(
            actual_pair
                .iter()
                .zip(&scalar_actual_pair)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "vector and scalar grouped gate/up results differ"
        );
        assert!(
            actual_down
                .iter()
                .zip(&scalar_actual_down)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "vector and scalar grouped down results differ"
        );
        let actual_sum = sum.to_vec1::<f32>().expect("read sum F32");
        let mut pair_path = ds4_prefix.as_os_str().to_os_string();
        pair_path.push(".pair.f32");
        let mut down_path = ds4_prefix.as_os_str().to_os_string();
        down_path.push(".down.f32");
        let expected_pair = fixture_f32(std::path::Path::new(&pair_path));
        let expected_down = fixture_f32(std::path::Path::new(&down_path));
        let expected_sum = fixture_f32(&ds4_prefix);
        let pair_error = fixture_error(&actual_pair, &expected_pair);
        let down_error = fixture_error(&actual_down, &expected_down);
        let sum_error = fixture_error(&actual_sum, &expected_sum);
        eprintln!(
            "[glm53-moe-q2-real] pair max_abs={:e} rel_l2={:e} exact={}/{}",
            pair_error.0,
            pair_error.1,
            pair_error.2,
            actual_pair.len()
        );
        eprintln!(
            "[glm53-moe-q2-real] down max_abs={:e} rel_l2={:e} exact={}/{}",
            down_error.0,
            down_error.1,
            down_error.2,
            actual_down.len()
        );
        eprintln!(
            "[glm53-moe-q2-real] sum max_abs={:e} rel_l2={:e} exact={}/{}",
            sum_error.0,
            sum_error.1,
            sum_error.2,
            actual_sum.len()
        );
        assert!(pair_error.1 < 1.0e-5, "grouped pair relative L2 error");
        assert!(down_error.1 < 1.0e-5, "grouped down relative L2 error");
        assert!(sum_error.1 < 1.0e-5, "grouped sum relative L2 error");
    }
}
