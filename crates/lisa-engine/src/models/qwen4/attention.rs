//! Full attention layer with per-head q|gate split projections, qk-norm,
//! partial rope, and a contiguous KV cache.

use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{fast, ops, Array, Dtype};

use crate::core::cache::FullAttentionCache;
use crate::models::qwen4::indexer::{QsaIndexer, QsaSelection};
use crate::core::loader::TensorSource;
use crate::core::norm::{rope_partial, RmsNorm, Rotary};
use crate::core::quant::QuantizedLinear;



/// Full attention layer.
pub struct Attention {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: QuantizedLinear,
    pub o_proj: QuantizedLinear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub indexer: QsaIndexer,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl Attention {
    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<Self> {
        // Reorder q_proj's rows to [all q of every head | all gate of every
        // head] so the prep kernel can read q as a contiguous slice.
        let mut q_proj = QuantizedLinear::load(src, prefix, "q_proj")?;
        {
            let d = head_dim as i32;
            let hq = heads as i32;
            let mut order: Vec<u32> = Vec::with_capacity((hq * d * 2) as usize);
            for h in 0..hq { for i in 0..d { order.push((h * 2 * d + i) as u32); } }
            for h in 0..hq { for i in 0..d { order.push((h * 2 * d + d + i) as u32); } }
            let idx = Array::from_slice(&order, &[(hq * d * 2) as i32]);
            q_proj.weight = q_proj.weight.take_axis(&idx, 0)?;
            q_proj.scales = q_proj.scales.take_axis(&idx, 0)?;
            q_proj.biases = q_proj.biases.take_axis(&idx, 0)?;
        }
        Ok(Self {
            q_proj,
            k_proj: QuantizedLinear::load(src, prefix, "k_proj")?,
            v_proj: QuantizedLinear::load(src, prefix, "v_proj")?,
            o_proj: QuantizedLinear::load(src, prefix, "o_proj")?,
            q_norm: RmsNorm::load(src, &format!("{prefix}.q_norm"), eps, None)?,
            k_norm: RmsNorm::load(src, &format!("{prefix}.k_norm"), eps, None)?,
            indexer: QsaIndexer::load(src, &format!("{prefix}.indexer"), eps)?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// x: [B, S, hidden]; cache: the layer's KV cache (updated in place).
    ///
    /// Causal masking is always applied; for s == 1 that is a no-op because
    /// every cached key lies at or before the query.
    pub fn forward(
        &self,
        x: &Array,
        rope: &Rotary,
        mut cache: Option<&mut FullAttentionCache>,
        offset: usize,
        positions: &Array,
    ) -> lisa_mlx::error::Result<Array> {
        let b = x.dim(0);
        let s_usize = x.dim(1) as usize;
        let s = x.dim(1);

        // The indexer runs BEFORE the projections so its tape sees this step's
        // raw keys; below the budget it returns None and attention stays plain
        // causal.
        let sparse = self
            .indexer
            .forward(
                x,
                rope,
                cache.as_deref_mut().and_then(|c| {
                    if c.next_pos.is_some() || !c.indexer_ok {
                        None
                    } else {
                        Some(&mut c.indexer_tape)
                    }
                }),
                offset,
            )
            .map_err(|e| lisa_mlx::error::Exception::custom(e.to_string()))?;

        if sparse.is_some() && std::env::var("LISA_INDEXER_DEBUG").is_ok() {
            eprintln!("[indexer] sparse keep mask active (offset={offset}, s={s_usize})");
        }

        // q_proj carries [all q of every head | all gate of every head] after
        // the load-time row reorder.
        let projected = self.q_proj.forward(x)?;
        let qw = (self.heads * self.head_dim) as i32;
        let q_part = projected.index((.., .., 0..qw)).contiguous()?;
        if std::env::var("LISA_ATTN_DEBUG").is_ok() {
            eprintln!("[attn] x={:?} projected={:?} q_part={:?} qw={qw}", x.shape(), projected.shape(), q_part.shape());
        }
        let gate = projected
            .index((.., .., qw..(2 * qw)))
            .contiguous()?
            .reshape(&[b, s, -1])?;

        let (cos, sin) = rope.cos_sin(positions)?;
        // The engine's fused `track_attn_prep` covers every width it serves
        // (S <= 8 uses the fused qkv form, wide prefill the split form); the
        // split kernel is the same body, so use it at every width.
        let (queries, keys, values) = if let Some(t) = {
                let k_raw = self.k_proj.forward(x)?;
                let v_raw = self.v_proj.forward(x)?;
                let stream = lisa_mlx::Stream::thread_local_or_default();
                let cosb = cos.as_dtype(Dtype::Bfloat16)?;
                let sinb = sin.as_dtype(Dtype::Bfloat16)?;
                lisa_mlx::kernels::attn_prep_split(
                    &q_part, &k_raw, &v_raw, &self.q_norm.weight, &self.k_norm.weight,
                    &cosb, &sinb, self.heads as i32, self.kv_heads as i32,
                    self.head_dim as i32, rope.dimensions as i32, self.q_norm.eps, &stream,
                )
            }
        {
            t
        } else {
            let queries = q_part.reshape(&[b, s, self.heads as i32, self.head_dim as i32])?;
            let queries = self.q_norm.forward(&queries)?;
            let keys = self.k_proj.forward(x)?;
            let keys = keys.reshape(&[b, s, self.kv_heads as i32, self.head_dim as i32])?;
            let keys = self.k_norm.forward(&keys)?;
            let values = self.v_proj.forward(x)?;
            let values = values.reshape(&[b, s, self.kv_heads as i32, self.head_dim as i32])?;
            if std::env::var("LISA_ATTN_DEBUG").is_ok() {
                eprintln!("[attn] pre-transpose queries={:?} keys={:?} values={:?}", queries.shape(), keys.shape(), values.shape());
            }
            let queries = queries.transpose_axes(&[0, 2, 1, 3])?;
            let keys = keys.transpose_axes(&[0, 2, 1, 3])?;
            let values = values.transpose_axes(&[0, 2, 1, 3])?;
            let cos = cos.expand_dims(1)?;
            let sin = sin.expand_dims(1)?;
            if std::env::var("LISA_ATTN_DEBUG").is_ok() {
                eprintln!("[attn] cos={:?} sin={:?} queries={:?}", cos.shape(), sin.shape(), queries.shape());
            }
            let queries = rope_partial(&queries, &cos, &sin)?;
            let keys = rope_partial(&keys, &cos, &sin)?;
            (queries, keys, values)
        };

        if std::env::var("LISA_DUMP_DRAFT").is_ok() {
            static ACALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let r = ACALL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let d3 = |name: &str, a: &lisa_mlx::Array, r: usize| {
                if r < 200 { return; }
                if let Ok(f) = a.as_dtype(lisa_mlx::Dtype::Float32) {
                    {
                                let v = f.as_slice::<f32>();
                        let v: &[f32] = v;
                        let sum: f32 = v.iter().sum();
                        eprintln!("[attn] #{r} {name} n={} sum={:.6e} h={:?}", v.len(), sum, &v[..4.min(v.len())]);
                    }
                }
            };
            let kk = keys.as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap_or_else(|_| keys.clone());
            let vv = values.as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap_or_else(|_| values.clone());
            d3("queries", &queries, r);
            d3("keys", &kk, r);
            d3("values", &vv, r);
        }
        let attn_mask = cache.as_deref().and_then(|c| c.attn_mask.clone());
        let (live_keys, live_values) = match cache {
            Some(c) => c.update(&keys, &values)?,
            None => (keys.clone(), values),
        };

        // Ragged batching supplies its own per-stream keep mask.
        if let Some(m) = attn_mask {
            let out = fast::scaled_dot_product_attention(
                &queries,
                &live_keys,
                &live_values,
                self.scale,
                &m,
                None,
            )?;
            let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, -1])?;
            let gated = {
                let stream = lisa_mlx::Stream::thread_local_or_default();
                match lisa_mlx::kernels::attn_gate(&out, &gate, self.heads as i32, self.head_dim as i32, &stream) {
                    Some(g) => g,
                    None => out.multiply(&ops::sigmoid(&gate)?)?,
                }
            };
            return self.o_proj.forward(&gated);
        }

        // QSA sparse attention. Prefer the block-sparse kernel, which
        // streams only the selected four-token blocks (plus the query's causal
        // tail) and never builds a dense `[S, L]` mask. Fall back to the fused
        // causal SDPA with an expanded keep mask when the kernel is unavailable
        // (non-NAX machines) or the geometry is not the production contract.
        let out = match sparse {
            Some(sel) => {
                let stream = lisa_mlx::Stream::thread_local_or_default();
                // The block-sparse consumer is O(budget) per token, never
                // O(kv), so use it at every width once the indexer activates;
                // the dense mask fallback is only for a declined geometry.
                let kernel_out = if offset >= 2048 && sel.block_ids.dim(2) == 512 {
                    let bids = sel.block_ids.reshape(&[s_usize as i32, 512])?;
                    let bval = sel.block_valid.reshape(&[s_usize as i32, 512])?;
                    lisa_mlx::qsa::prefill_flash(
                        &queries,
                        &live_keys,
                        &live_values,
                        &bids,
                        &bval,
                        offset as i32,
                        (offset + s_usize) as i32,
                        self.scale,
                        &stream,
                    )
                } else {
                    None
                };
                match kernel_out {
                    Some(o) => o,
                    None => {
                        let keep = dense_keep_from_selection(&sel)?;
                        let kv_len = (offset + s_usize) as i32;
                        // f32 (not i32): the strided copy/cmp kernels have no
                        // i32 variant, and positions fit f32 exactly below 2^24.
                        let rinds: Vec<f32> = (0..kv_len).map(|x| x as f32).collect();
                        let rinds = Array::from_slice(&rinds, &[kv_len]);
                        let linds: Vec<f32> = ((kv_len - s)..kv_len).map(|x| x as f32).collect();
                        let linds = Array::from_slice(&linds, &[s as i32, 1]);
                        let l = lisa_mlx::ops::broadcast_to(&linds, &[s as i32, kv_len])?.contiguous()?;
                        let r = lisa_mlx::ops::broadcast_to(&rinds.expand_dims(0)?, &[s as i32, kv_len])?.contiguous()?;
                        let causal = l.ge(&r)?;
                        let mask = causal.expand_dims(0)?.logical_and(&keep)?;
                        fast::scaled_dot_product_attention(
                            &queries,
                            &live_keys,
                            &live_values,
                            self.scale,
                            &mask,
                            None,
                        )?
                    }
                }
            }
            None => {
                if std::env::var("LISA_DUMP_DRAFT").is_ok() {
                    static LCALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    let r = LCALL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if r >= 200 {
                        for (nm, a) in [("lk", &live_keys), ("lv", &live_values)] {
                            if let Ok(f) = a.as_dtype(lisa_mlx::Dtype::Float32) {
                                {
                                let v = f.as_slice::<f32>();
                                    let v: &[f32] = v;
                                    let sum: f32 = v.iter().sum();
                                    eprintln!("[live] #{r} {nm} n={} sum={:.6e} h={:?}", v.len(), sum, &v[..4.min(v.len())]);
                                }
                            }
                        }
                    }
                }
                fast::scaled_dot_product_attention(
                    &queries,
                    &live_keys,
                    &live_values,
                    self.scale,
                    fast::ScaledDotProductAttentionMask::Causal,
                    None,
                )?
            }
        };
        // [B, S, H*D]
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, -1])?;
        if std::env::var("LISA_DUMP_DRAFT").is_ok() {
            static OCALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let r = OCALL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if r >= 200 {
                if let Ok(f) = out.as_dtype(lisa_mlx::Dtype::Float32) {
                    {
                                let v = f.as_slice::<f32>();
                        let v: &[f32] = v;
                        let sum: f32 = v.iter().sum();
                        eprintln!("[attn-out] #{r} n={} sum={:.6e} h={:?}", v.len(), sum, &v[..4.min(v.len())]);
                    }
                }
            }
        }
        if std::env::var("LISA_ATTN_DEBUG").is_ok() {
            eprintln!("[attn] q={:?} k={:?} v={:?} out={:?} gate={:?} b={b} s={s} offset={offset}",
                queries.shape(), live_keys.shape(), live_values.shape(), out.shape(), gate.shape());
        }
        let gated = {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            match lisa_mlx::kernels::attn_gate(&out, &gate, self.heads as i32, self.head_dim as i32, &stream) {
                Some(g) => g,
                None => out.multiply(&ops::sigmoid(&gate)?)?,
            }
        };
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            static AC2: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let ac = AC2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let want = std::env::var("LISA_ATTN_CALL").ok().and_then(|v| v.parse::<usize>().ok());
            if want == Some(ac) {
                for (nm, arr) in [("q", &queries), ("lk", &live_keys), ("lv", &live_values), ("out", &out), ("gate", &gate), ("gated", &gated)] {
                    let a = arr.as_dtype(Dtype::Float32)?;
                    let a = a.as_slice::<f32>();
                    let _ = std::fs::write(format!("{dir}/aout_{nm}.bin"), unsafe {
                        std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                    });
                }
            }
        }
        let ret = self.o_proj.forward(&gated)?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            static AC9: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let ac = AC9.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let want = std::env::var("LISA_ATTN_CALL").ok().and_then(|v| v.parse::<usize>().ok());
            if want == Some(ac) {
                let a = ret.as_dtype(Dtype::Float32)?;
                let a = a.as_slice::<f32>();
                let _ = std::fs::write(format!("{dir}/aout_ret.bin"), unsafe {
                    std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                });
            }
        }
        Ok(ret)
    }
}


/// Expand a sparse selection into the dense boolean keep mask `[1, S, kv_len]`
/// (the fallback for when the block-sparse kernel is unavailable).
fn dense_keep_from_selection(sel: &QsaSelection) -> lisa_mlx::error::Result<Array> {
    let s = sel.q_pos.dim(1);
    let cr = sel.cr;
    let kv = sel.kv_len;
    let blocks = kv / cr;
    // `where_cond` has no I32 output, so do the select in f32
    // (block ids are < 2^24) and cast back.
    let sentinel = lisa_mlx::ops::broadcast_to(&Array::from_f32(blocks as f32), sel.block_ids.shape())?;
    let ids_f = sel.block_ids.as_dtype(Dtype::Float32)?;
    let picked_f = lisa_mlx::ops::r#where(&sel.block_valid, &ids_f, &sentinel)?;
    let picked_safe = picked_f.as_dtype(Dtype::Int32)?;
    let mut keep_blocks = ops::zeros::<bool>(&[1, s, blocks + 1])?;
    // The shim's scatter needs `updates` to share the index rank (no scalar).
    let upd = lisa_mlx::ops::broadcast_to(&Array::from_bool(true), picked_safe.shape())?;
    keep_blocks = keep_blocks.put_along_axis(&picked_safe, &upd, -1)?;
    let keep_blocks = keep_blocks.index((Ellipsis, 0..blocks));
    let mut keep = ops::repeat_axis::<bool>(keep_blocks, cr, -1)?;
    let rest = kv - blocks * cr;
    if rest > 0 {
        let zeros = ops::zeros::<bool>(&[1, s, rest])?;
        keep = ops::concatenate(&[&keep, &zeros], -1)?;
    }
    // `complete` is f32 (no i32 Metal arithmetic); compare in f32.
    let own_start = sel.complete.multiply(Array::from_f32(cr as f32))?;
    let tokens = Array::from_slice(&(0..kv).map(|x| x as f32).collect::<Vec<f32>>(), &[1, 1, kv]);
    let kv_pos = Array::from_slice(&((kv - s)..kv).map(|x| x as f32).collect::<Vec<f32>>(), &[1, s]);
    let tok_b = lisa_mlx::ops::broadcast_to(&tokens, &[1, s, kv])?;
    // `cmp` does not broadcast; expand both bounds to the full shape.
    let own_start_b = lisa_mlx::ops::broadcast_to(&own_start.expand_dims(-1)?, &[1, s, kv])?;
    let kv_pos_b = lisa_mlx::ops::broadcast_to(&kv_pos.expand_dims(-1)?, &[1, s, kv])?;
    let own = tok_b.ge(&own_start_b)?.logical_and(&tok_b.le(&kv_pos_b)?)?;
    keep.logical_or(&own)
}
