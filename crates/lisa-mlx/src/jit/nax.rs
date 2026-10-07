use objc2_metal::MTLSize;

use crate::array::Array;
use crate::runtime::{ComputePipeline, ConstVal};

use super::compile::type_to_name;
use super::{DType, Device, Tensor};
use super::{NAX_GEMM_HEADER, NAX_QUANT_HEADER};
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
    let obuf = mdev.buffer((count) as usize * (a.dtype()).size_of(), "gemm_out")?;
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
    let consts: Vec<(usize, ConstVal)> = consts
        .iter()
        .map(|&(i, v)| (i, ConstVal::Bool(v)))
        .collect();
    device.compile_with_constants(source, name, crate::runtime::Math::Safe, &consts)
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

    let obuf = mdev.buffer(
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
         instantiate_kernel(\"{kname}\", affine_qmm_t_nax, bfloat16_t, {group_size}, {bits}, {}, {}, {bm}, {bk}, {bn}, {wm}, {wn})\n",
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
            &kname, group_size, bits, aligned, batched, bm, bk, bn, wm, wn,
        ),
        &kname,
        &[],
    )?;

    let count = m * n;
    let obuf = mdev.buffer((count) as usize * (x.dtype()).size_of(), "qmm_nax_out")?;
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
