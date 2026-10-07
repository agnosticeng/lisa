use objc2_metal::MTLSize;

use crate::array::{Array, Dtype};
use crate::runtime::ConstVal;

use super::compile::{
    builtin_template_def, compile_builtin, compile_builtin_bool_consts,
    compile_builtin_typed_consts, type_string, type_to_name,
};
use super::nax::{compile_nax_jit, matmul_nax};
use super::{DType, Device, Tensor};
use super::{MLX_SDPA_VECTOR_PREAMBLE, MLX_SOFTMAX_PREAMBLE, MLX_UTILS_PREAMBLE, NAX_ATTN_HEADER};
use crate::error::Result;

/// MLX `fast::scaled_dot_product_attention` vector path (`sdpa_vector`,
/// `scaled_dot_product_attention.cpp:364`) — single/few query rows, no array
/// mask, no sinks. `q`/`k`/`v` are `[B, H, L, D]`, `[B, Hk, N, D]`, `[B, Hk, N, V]`.
/// Mask strides per MLX `sdpa_vector` (`scaled_dot_product_attention.cpp:437`):
/// an axis of extent 1 broadcasts, so its stride reads as 0 inside the kernel.
fn mask_strides(m: &Tensor) -> [i32; 3] {
    let (sh, st) = (m.dims(), m.stride());
    [
        (if sh[3] > 1 { st[3] } else { 0 }) as i32,
        (if sh[2] > 1 { st[2] } else { 0 }) as i32,
        (if sh[1] > 1 {
            st[1]
        } else if sh[0] > 1 {
            st[0]
        } else {
            0
        }) as i32,
    ]
}

pub fn sdpa_vector(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let mdev = device;
    let qt = type_string(q.dtype())?;
    let d = q.dims()[3];
    let vd = v.dims()[3];
    let kernel_name = format!("sdpa_vector_{qt}_{d}_{vd}");
    let def = builtin_template_def(
        &kernel_name,
        "sdpa_vector",
        &[qt.to_string(), d.to_string(), vd.to_string()],
    );
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SDPA_VECTOR_PREAMBLE}{def}");
    let has_mask = mask.is_some();
    let bool_mask = has_mask && mask.unwrap().dtype() == Dtype::Bool;
    let float_mask = has_mask && !bool_mask;
    let pipeline = compile_builtin_bool_consts(
        device,
        &source,
        &kernel_name,
        &[
            (20, has_mask),
            (21, false), // query_transposed
            (22, do_causal),
            (23, bool_mask),
            (24, float_mask),
            (25, false), // has_sinks
        ],
    )?;

    let qd = q.dims().to_vec();
    let (b, h, ql) = (qd[0], qd[1], qd[2]);
    let out_dims = vec![b, h, ql, vd];
    let count: usize = out_dims.iter().product();
    let obuf = mdev.buffer((count) as usize * (q.dtype()).size_of(), "sdpa_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims, q.dtype());

    let gqa_factor = (q.dims()[1] / k.dims()[1]) as i32;
    let n = k.dims()[2] as i32;
    let k_head_stride: usize = if k.dims()[1] == 1 {
        k.stride()[0]
    } else {
        k.stride()[1]
    };
    let k_seq_stride: usize = k.stride()[2];
    let v_head_stride: usize = if v.dims()[1] == 1 {
        v.stride()[0]
    } else {
        v.stride()[1]
    };
    let v_seq_stride: usize = v.stride()[2];

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, q)?;
    bind(1, k)?;
    bind(2, v)?;
    enc.set_output(3, Some(&obuf), 0);
    enc.set_bytes(4, &gqa_factor);
    enc.set_bytes(5, &n);
    enc.set_bytes(6, &k_head_stride);
    enc.set_bytes(7, &k_seq_stride);
    enc.set_bytes(8, &v_head_stride);
    enc.set_bytes(9, &v_seq_stride);
    enc.set_bytes(10, &scale);
    if let Some(m) = mask {
        bind(11 + float_mask as usize, m)?;
        let [kvs, qs, hs] = mask_strides(m);
        enc.set_bytes(13, &kvs);
        enc.set_bytes(14, &qs);
        enc.set_bytes(15, &hs);
    }

    enc.dispatch_groups_size(
        MTLSize {
            width: b * h,
            height: ql,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

/// MLX `fast::scaled_dot_product_attention` 2-pass vector path
/// (`sdpa_vector_2pass`, `scaled_dot_product_attention.cpp:454`).
pub fn sdpa_vector_2pass(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let mdev = device;
    let qt = type_string(q.dtype())?;
    let d = q.dims()[3];
    let vd = v.dims()[3];
    let kname1 = format!("sdpa_vector_2pass_1_{qt}_{d}_{vd}");
    let kname2 = format!("sdpa_vector_2pass_2_{qt}_{vd}");
    let mut defs = builtin_template_def(
        &kname1,
        "sdpa_vector_2pass_1",
        &[qt.to_string(), d.to_string(), vd.to_string()],
    );
    defs.push_str(&builtin_template_def(
        &kname2,
        "sdpa_vector_2pass_2",
        &[qt.to_string(), vd.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SDPA_VECTOR_PREAMBLE}{defs}");

    let gqa_factor = q.dims()[1] / k.dims()[1];
    let n_simds = gqa_factor * q.dims()[2];
    let n = k.dims()[2];
    let devc = mdev.architecture_name().chars().last().unwrap_or('s');
    let blocks: usize = if devc == 's' {
        let mut bl = 64;
        if n > 1024 && n_simds > 4 {
            if n <= 8192 {
                bl = 128;
            } else if n <= 32768 {
                bl = 256;
            } else if n <= 65536 {
                bl = 512;
            } else {
                bl = 1024;
            }
        }
        bl
    } else if devc == 'd' {
        let mut bl = 128;
        if n_simds <= 2 && n > 8192 {
            bl = 256;
        } else if n_simds >= 6 {
            if (16384..65536).contains(&n) {
                bl = 512;
            } else if n >= 65536 {
                bl = 1024;
            }
        }
        bl
    } else if n_simds >= 4 {
        64
    } else {
        32
    };

    let pipeline1 = compile_builtin_typed_consts(
        device,
        &source,
        &kname1,
        &[
            (20, ConstVal::Bool(mask.is_some())),
            (21, ConstVal::Bool(false)), // query_transposed
            (22, ConstVal::Bool(do_causal)),
            (23, ConstVal::Bool(mask.is_some() && mask.unwrap().dtype() == Dtype::Bool)),
            (24, ConstVal::Bool(
                mask.is_some() && mask.unwrap().dtype() != Dtype::Bool,
            )),
            (25, ConstVal::Bool(false)), // has_sinks
            (26, ConstVal::Int(blocks as i32)),
        ],
    )?;
    let pipeline2 = compile_builtin(device, &source, &kname2)?;

    let qd = q.dims().to_vec();
    let (b, h, ql) = (qd[0], qd[1], qd[2]);
    let out_dims = vec![b, h, ql, vd];
    let out_count: usize = out_dims.iter().product();
    let obuf = mdev.buffer((out_count) as usize * (q.dtype()).size_of(), "sdpa_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims.clone(), q.dtype());

    // intermediates: [b,h,ql,blocks,vd] (q dtype) and [b,h,ql,blocks] (f32)
    let interm_dims = vec![b, h, ql, blocks, vd];
    let interm_count: usize = interm_dims.iter().product();
    let ibuf = mdev.buffer(
        (interm_count) as usize * (q.dtype()).size_of(),
        "sdpa_interm",
    )?;
    let red_dims = vec![b, h, ql, blocks];
    let red_count: usize = red_dims.iter().product();
    let sbuf = mdev.buffer((red_count) as usize * (DType::F32).size_of(), "sdpa_sums")?;
    let mbuf = mdev.buffer((red_count) as usize * (DType::F32).size_of(), "sdpa_maxs")?;

    let k_head_stride: usize = if k.dims()[1] == 1 {
        k.stride()[0]
    } else {
        k.stride()[1]
    };
    let k_seq_stride: usize = k.stride()[2];
    let v_head_stride: usize = if v.dims()[1] == 1 {
        v.stride()[0]
    } else {
        v.stride()[1]
    };
    let v_seq_stride: usize = v.stride()[2];
    let n_i = n as i32;

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();

    // pass 1
    enc.set_pipeline(&pipeline1);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, q)?;
    bind(1, k)?;
    bind(2, v)?;
    enc.set_output(3, Some(&ibuf), 0);
    enc.set_output(4, Some(&sbuf), 0);
    enc.set_output(5, Some(&mbuf), 0);
    enc.set_bytes(7, &n_i);
    enc.set_bytes(8, &k_head_stride);
    enc.set_bytes(9, &k_seq_stride);
    enc.set_bytes(10, &v_head_stride);
    enc.set_bytes(11, &v_seq_stride);
    enc.set_bytes(12, &scale);
    if let Some(m) = mask {
        bind(13 + (m.dtype() != Dtype::Bool) as usize, m)?;
        let [kvs, qs, hs] = mask_strides(m);
        enc.set_bytes(15, &kvs);
        enc.set_bytes(16, &qs);
        enc.set_bytes(17, &hs);
    }
    enc.dispatch_groups_size(
        MTLSize {
            width: k.dims()[1],
            height: b,
            depth: blocks,
        },
        MTLSize {
            width: 32,
            height: gqa_factor,
            depth: ql,
        },
    );

    // pass 2
    enc.set_pipeline(&pipeline2);
    enc.set_input(0, Some(&ibuf), 0);
    enc.set_input(1, Some(&sbuf), 0);
    enc.set_input(2, Some(&mbuf), 0);
    enc.set_output(3, Some(&obuf), 0);
    let blocks_i = blocks as i32;
    enc.set_bytes(4, &blocks_i);
    enc.dispatch_groups_size(
        MTLSize {
            width: b * h,
            height: ql,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    let _ = interm_dims;
    Ok(out)
}

/// MLX `ops::softmax_axis` over the last axis, `Softmax::eval_gpu` block path
/// (`softmax.cpp:16`). `precise` forces an f32 accumulator.
pub fn softmax_last_axis(device: &Device, x: &Tensor, precise: bool) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("softmax: non-contiguous or scalar input");
    }
    let dims = x.dims().to_vec();
    let dt = x.dtype();
    let axis_size = *dims.last().unwrap();
    let n_rows = x.elem_count() / axis_size.max(1);
    let looped = axis_size > 4096;
    let in_t = type_string(dt)?;
    let acc_t = if precise { "float" } else { in_t };
    let ty = type_to_name(dt)?;

    let mut kernel_name = String::from(if looped {
        "looped_softmax_"
    } else {
        "block_softmax_"
    });
    if dt != DType::F32 && precise {
        kernel_name.push_str("precise_");
    }
    kernel_name.push_str(ty);
    let lib_name = kernel_name
        .split_once('_')
        .map(|(_, r)| r)
        .unwrap_or(&kernel_name)
        .to_string();

    let mut defs = builtin_template_def(
        &format!("block_{lib_name}"),
        "softmax_single_row",
        &[in_t.to_string(), acc_t.to_string()],
    );
    defs.push_str(&builtin_template_def(
        &format!("looped_{lib_name}"),
        "softmax_looped",
        &[in_t.to_string(), acc_t.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SOFTMAX_PREAMBLE}{defs}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let count = x.elem_count();
    let obuf = mdev.buffer((count) as usize * (dt).size_of(), "softmax_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &dims.clone(), dt);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    enc.set_output(1, Some(&obuf), 0);
    let asize = axis_size as i32;
    enc.set_bytes(2, &asize);

    let simd = 32usize;
    let (tgs, n_threads) = if looped {
        // MLX looped path: fixed 1024-wide threadgroups, one row per group.
        (1024usize, n_rows * 1024)
    } else {
        let tg_needed = axis_size.div_ceil(4);
        let tgs = simd * tg_needed.div_ceil(simd);
        (tgs, n_rows * tgs)
    };
    enc.dispatch_threads_size(
        MTLSize {
            width: n_threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tgs,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

/// Source for one `attention_nax` / `attention_nax_dsplit` instantiation.
pub fn sdpa_full_nax_source(
    base_name: &str,
    split_d: bool,
    bq: usize,
    bk: usize,
    bd: usize,
    wm: usize,
    wn: usize,
    q_ty: &str,
    mask_ty: &str,
) -> String {
    format!(
        "{NAX_ATTN_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{base_name}\", {}, {q_ty}, {bq}, {bk}, {bd}, {wm}, {wn}, {mask_ty})\n",
        if split_d {
            "attention_nax_dsplit"
        } else {
            "attention_nax"
        }
    )
}

/// MLX `sdpa_full_self_attention_nax` (`scaled_dot_product_attention.cpp:18`),
/// JIT-compiled from the vendored MLX `.metal` source.
pub fn sdpa_full_nax(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
) -> Result<Tensor> {
    let mdev = device;
    let bd = q.dims()[3];
    let bq = 64usize;
    let bk = 32usize;
    let split_d = bd == 256;
    let wm = 4usize;
    let wn = if split_d { 2 } else { 1 };
    let (b, h, ql) = (q.dims()[0], q.dims()[1], q.dims()[2]);
    let k_l = k.dims()[2];
    let gqa = q.dims()[1] / k.dims()[1];
    let qt = type_to_name(q.dtype())?;

    let prefix = if split_d {
        "steel_attention_dsplit_"
    } else {
        "steel_attention_"
    };
    let base_name = format!("{prefix}{qt}_bq{bq}_bk{bk}_bd{bd}_wm{wm}_wn{wn}_mask{qt}");
    let align_q = ql % bq == 0;
    let align_k = k_l % bk == 0;
    let pipeline = compile_nax_jit(
        device,
        &sdpa_full_nax_source(&base_name, split_d, bq, bk, bd, wm, wn, "bfloat", "bfloat"),
        &base_name,
        &[
            (200, align_q),
            (201, align_k),
            (300, false),
            (301, do_causal),
            (302, false),
        ],
    )?;

    let store_dims = vec![b, h, ql, bd];
    let o_str = [(h * ql * bd) as i64, (ql * bd) as i64, bd as i64];
    let count: usize = store_dims.iter().product();
    let obuf = mdev.buffer((count) as usize * (q.dtype()).size_of(), "sdpa_full_out")?;
    let store = Array::from_parts(mdev, obuf.clone(), &store_dims, q.dtype());

    let nq = ql.div_ceil(bq);
    let nk = k_l.div_ceil(bk);
    let nq_aligned = ql / bq;
    let nk_aligned = k_l / bk;
    let ql_rem = ql - nq_aligned * bq;
    let kl_rem = k_l - nk_aligned * bk;
    let ql_off = k_l as i64 - ql as i64;

    let mut p: Vec<u8> = Vec::with_capacity(152);
    for val in [
        b as i32, h as i32, bd as i32, ql as i32, k_l as i32, gqa as i32,
    ] {
        p.extend_from_slice(&val.to_le_bytes());
    }
    p.extend_from_slice(&scale.to_le_bytes());
    for val in [
        nq as i32,
        nk as i32,
        nq_aligned as i32,
        nk_aligned as i32,
        ql_rem as i32,
        kl_rem as i32,
        ql_off as i32,
    ] {
        p.extend_from_slice(&val.to_le_bytes());
    }
    let q_str = [
        q.stride()[0] as i64,
        q.stride()[1] as i64,
        q.stride()[2] as i64,
    ];
    let k_str = [
        k.stride()[0] as i64,
        k.stride()[1] as i64,
        k.stride()[2] as i64,
    ];
    let v_str = [
        v.stride()[0] as i64,
        v.stride()[1] as i64,
        v.stride()[2] as i64,
    ];
    for s in [q_str, k_str, v_str, o_str] {
        for val in s {
            p.extend_from_slice(&val.to_le_bytes());
        }
    }

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, q)?;
    bind(1, k)?;
    bind(2, v)?;
    enc.set_output(3, Some(&obuf), 0);
    enc.set_bytes_directly(4, p.len(), p.as_ptr().cast());

    enc.dispatch_groups_size(
        MTLSize {
            width: nq,
            height: h,
            depth: b,
        },
        MTLSize {
            width: 32,
            height: wm,
            depth: wn,
        },
    );
    store.contiguous()
}

/// MLX `fast::scaled_dot_product_attention` fused dispatch for the causal /
/// array-mask case: ALL `qL <= 8` query runs route to the vector kernels
/// (masked 2-pass at long KV), the NAX full kernel or dense op chain for
/// wider ones — the old `qL * gqa > 32` dense diversion is gone (it matched
/// no MLX path and degraded super-linearly with kL).
pub fn sdpa(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // The KV cache hands us non-contiguous views (a [0..len] slice of a larger
    // capacity buffer); the dense op chain's stride handling diverges from
    // MLX's there, but the vector kernels bind head/seq strides natively — so
    // at qL <= 8 we copy only when MLX would (`copy_unless` predicates,
    // `scaled_dot_product_attention.cpp:803-830`), never as a reflex.
    let ql = q.dims()[2];
    let d = q.dims()[3];
    let gqa = q.dims()[1] / k.dims()[1];
    let nax = device.nax();
    if ql <= 8 {
        // MLX eval_gpu vector-mode layout checks: q may stay strided only when
        // a singleton batch/head makes it effectively [H, L, D]; k/v may stay
        // strided when the last dim is dense AND the view is singleton-batch
        // (a KV-cache slice) or head-major contiguous.
        let q_ok = {
            let (sh, st) = (q.dims(), q.stride());
            q.buffer_and_layout().1.is_contiguous()
                || sh[0] == 1
                || sh[1] == 1 && st[3] == 1
                    && st[2] == sh[3] * sh[if sh[0] == 1 { 1 } else { 0 }]
                    && st[if sh[0] == 1 { 1 } else { 0 }] == sh[3]
        };
        // MLX kv_copy_unless: the last dim must be dense; a singleton batch or
        // head (any KV-cache slice view) is accepted as-is, otherwise the view
        // must be head-major contiguous.
        let kv_ok = |t: &Tensor| {
            let (sh, st) = (t.dims(), t.stride());
            st[3] == 1 && (sh[0] == 1 || sh[1] == 1 || st[0] == st[1] * sh[1])
        };
        let qc = if q_ok { q.clone() } else { q.contiguous()? };
        let kc = if kv_ok(k) { k.clone() } else { k.contiguous()? };
        let vc = if kv_ok(v) { v.clone() } else { v.contiguous()? };
        let (q, k, v) = (&qc, &kc, &vc);
        // MLX eval_gpu: 2-pass on 's'/'d' with kL >= 1024 (or short KV heads
        // with kL >= 4096); causal is dropped for a single query. ALL qL <= 8
        // route here — including the MTP verify rows: the extra dense
        // diversion our port once had paid a per-(kv-head, repeat) GEMM +
        // looped-softmax chain whose cost grows with kL. MEASURED DEVIATION
        // (specs/23): below kL ~2048 the dense chain is still cheaper at wide
        // verify (S7/kv1024: 66 ms dense vs 114 fused) — the fused route
        // engages from 2048 up, where the dense chain's kL-linear term
        // dominates (S7/16k: 148 → 140 ms, and the chain's premium over the
        // same-path S5 arm −32 → −15 ms).
        let kl = k.dims()[2];
        let devc = device.architecture_name().chars().last().unwrap_or('s');
        let two_pass = ((devc == 'd' || devc == 's') && kl >= 1024)
            || (k.dims()[1] < q.dims()[1] && kl >= 4096);
        let dc = do_causal && ql > 1;
        if two_pass && !(ql * gqa > 32 && kl < 2048) {
            return if two_pass {
                sdpa_vector_2pass(device, q, k, v, scale, dc, mask)
            } else {
                sdpa_vector(device, q, k, v, scale, dc, mask)
            };
        }
        let (q, k, v) = (
            q.contiguous()?,
            k.contiguous()?,
            v.contiguous()?,
        );
        let (q, k, v) = (&q, &k, &v);
        if mask.is_some() && !matches!(d, 192 | 256) {
            // Unchanged legacy semantics: the dense chain only takes array
            // masks for head_dim 192/256 (the fused vector kernels above
            // handle masks at qL <= 8 for every d).
            crate::bail!("sdpa: array mask with head_dim {d} not ported");
        }
        if ql * gqa > 32 {
            if nax && d == 256 && do_causal {
                return sdpa_full_nax(device, q, k, v, scale, do_causal);
            }
            return sdpa_dense(device, q, k, v, scale, do_causal, mask);
        }
        if two_pass {
            return sdpa_vector_2pass(device, q, k, v, scale, dc, mask);
        }
        return sdpa_vector(device, q, k, v, scale, dc, mask);
    }
    let ql = q.dims()[2];
    if mask.is_some() {
        // The fused kernels' array-mask branches are not ported; MLX itself
        // routes d == 192/256 array-mask attention to the dense op chain.
        if matches!(d, 192 | 256) {
            return sdpa_dense(device, q, k, v, scale, false, mask);
        }
        crate::bail!("sdpa: array mask with head_dim {d} not ported");
    }
    if nax && ql >= 1024 && d == 256 && do_causal {
        sdpa_full_nax(device, q, k, v, scale, do_causal)
    } else if matches!(d, 192 | 256) {
        sdpa_dense(device, q, k, v, scale, do_causal, None)
    } else if matches!(d, 64 | 96 | 128) {
        if nax {
            sdpa_full_nax(device, q, k, v, scale, do_causal)
        } else {
            sdpa_dense(device, q, k, v, scale, do_causal, None)
        }
    } else {
        crate::bail!("sdpa: dense fallback for head_dim {d} not ported")
    }
}

/// MLX's dense (unfused) SDPA op chain from `fast.cpp` (`use_fallback ==
/// true`): `softmax(q*scale @ kᵀ [+ mask]) @ v`. Used for MTP verify
/// (`qL*gqa > 32`) and for array-mask / ragged batching. The batched matmuls
/// are the MLX NAX GEMM (`matmul_nax`) applied per (batch, kv-head, repeat)
/// slice; the softmax is the ported precise softmax kernel.
/// 2-D GEMM that follows the device: the NAX tensor-core GEMM on M5+, the
/// generic steel GEMM (`Array::matmul` -> `dense_gemm`) on older GPUs.
fn matmul_2d(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if device.nax() {
        return matmul_nax(device, a, b, false, false);
    }
    a.matmul(b)
}

pub fn sdpa_dense(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, h, ql, d) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let hk = k.dims()[1];
    let kl = k.dims()[2];
    let r = h / hk;
    let dt = q.dtype();

    // `multiply(array(scale, dtype), q)` — a bf16 scalar times q.
    let scal = Array::scalar_of(device, scale, dt)?;
    let qs = q.broadcast_mul(&scal)?;

    // scores = matmul(q, swapaxes(k, -1, -2)) -> [B, Hk, R, qL, kL]
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(b * hk * r);
    for bi in 0..b {
        for hi in 0..hk {
            let kt = k_t.i((bi, hi))?.contiguous()?;
            for ri in 0..r {
                let qq = qs.i((bi, hi * r + ri))?.contiguous()?;
                rows.push(matmul_2d(device, &qq, &kt)?);
            }
        }
    }
    let mut scores = Array::stack(&rows, 0)?.reshape(&[b, hk, r, ql, kl])?;

    let has_mask = mask.is_some() || do_causal;
    if has_mask {
        let m = match mask {
            Some(m) => m.contiguous()?,
            None => {
                let offset = kl as i64 - ql as i64;
                let mut mv = vec![0u8; ql * kl];
                for i in 0..ql {
                    for j in 0..kl {
                        mv[i * kl + j] = ((offset + i as i64) >= j as i64) as u8;
                    }
                }
                Array::from_slice_dt(device, &mv, &[ql as usize, kl as usize], Dtype::Uint8)?
            }
        };
        // bool mask -> `where(mask, scores, finfo(dtype).min)`
        let minv = Array::scalar_of(device, half::bf16::MIN.to_f32(), dt)?;
        let mb = m.broadcast_as(scores.shape())?;
        let minb = minv.broadcast_as(scores.shape())?;
        scores = mb.where_cond(&scores, &minb)?;
    }

    let probs = softmax_last_axis(device, &scores, true)?;

    let mut outs: Vec<Tensor> = Vec::with_capacity(b * hk * r);
    for bi in 0..b {
        for hi in 0..hk {
            let vv = v.i((bi, hi))?.contiguous()?;
            for ri in 0..r {
                let sc = probs.i3(bi, hi, ri)?.contiguous()?;
                outs.push(matmul_2d(device, &sc, &vv)?);
            }
        }
    }
    let out = Array::stack(&outs, 0)?.reshape(&[b, hk, r, ql, d])?;
    out.reshape(&[b, h, ql, d])
}
