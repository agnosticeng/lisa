#![cfg(test)]
//! Frozen pre-migration computation used only as a numerical oracle.
use super::*;
use crate::core::norm::rope_partial;
use crate::core::quant::QuantizedLinearExecution;
use anyhow::Context;
use lisa_metal::ops::indexing::{Ellipsis, IndexMutOp, IndexOp};
use lisa_metal::{Dtype, fast, ops};
impl Qwen35Attention {
    fn reference_forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mut cache: Option<&mut FullAttentionCache>,
        _offset: usize,
    ) -> lisa_metal::error::Result<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let hd = self.head_dim as i32;
        let heads = self.heads as i32;

        let joined = if b == 1 && s <= 8 {
            self.qkv_proj.as_ref().map(|p| p.forward(x)).transpose()?
        } else {
            None
        };
        let project = |projection: &QuantizedLinear, offset: i32| {
            if let Some(all) = &joined {
                Ok(all.index((.., .., offset..offset + projection.dims_out() as i32)))
            } else {
                projection.forward(x)
            }
        };
        let (mut q, mut gate, qg_raw) = {
            let qg = project(&self.q_proj, 0)?;
            let qg = qg.split_last_axis(&[heads, 2, hd])?;
            let eager = s == 1;
            let q = if eager {
                Some(qg.index((Ellipsis, 0, ..)).contiguous()?)
            } else {
                None
            };
            let gate = if self.gated {
                if eager {
                    Some(qg.index((Ellipsis, 1, ..)).contiguous()?)
                } else {
                    None
                }
            } else {
                None
            };
            (q, gate, Some(qg))
        };
        let qn = self.q_proj.dims_out() as i32;
        let kn = self.k_proj.dims_out() as i32;
        let k = project(&self.k_proj, qn)?.split_last_axis(&[self.kv_heads as i32, hd])?;
        let v = project(&self.v_proj, qn + kn)?.split_last_axis(&[self.kv_heads as i32, hd])?;

        let attn_mask = cache.as_deref().and_then(|c| c.attn_mask.clone());
        let mask5: Option<lisa_metal::Array> = match &attn_mask {
            Some(m) => Some(m.reshape(&[b, 1, 1, s, m.dim(-1)])?),
            None => None,
        };
        let sdpa_mask = match &mask5 {
            Some(m) => fast::ScaledDotProductAttentionMask::Array(m),
            None => fast::ScaledDotProductAttentionMask::Causal,
        };
        let cos_bf = cos;
        let sin_bf = sin;
        let per_slot = b > 1 && attn_mask.is_some();

        // The old wrapper admitted B=2,D=128 by element count alone, then
        // emitted B=1,D=256 and read a 128-wide weight as 256-wide. Such a
        // shape has no valid fused oracle; use its original composed chain.
        if s == 1 && b == 1 && self.head_dim == 256 {
            let stream = lisa_metal::Stream::thread_local_or_default();
            if let Some((qk, kk)) = lisa_metal::kernels::gated_pointwise::qk_norm_rope(
                q.as_ref().expect("s==1 q is eager"),
                &k,
                &self.q_norm.weight,
                &self.k_norm.weight,
                cos_bf,
                sin_bf,
                self.q_norm.eps,
                heads,
                self.kv_heads as i32,
                &stream,
            ) {
                let v = v.transpose_axes(&[0, 2, 1, 3])?;
                let (live_k, live_v) = match cache.as_deref_mut() {
                    Some(c) => reference_update(c, &kk, &v)?,
                    None => (kk, v),
                };
                let out = fast::scaled_dot_product_attention(
                    &qk, &live_k, &live_v, self.scale, sdpa_mask, None,
                )?;
                return self.reference_tail(out, gate, b, s, heads, hd);
            }
        }

        if b == 1
            && s > 1
            && self.gated
            && attn_mask.is_none()
            && let Some(qg) = qg_raw.as_ref()
        {
            let stream = lisa_metal::Stream::thread_local_or_default();
            let (qg2, k2) = if let Some(all) = joined.as_ref() {
                let rows = all
                    .reshape_rows_view(s as usize, all.dim(2) as usize)
                    .expect("joined projection has uniform rows");
                (rows.index((.., 0..qn)), rows.index((.., qn..qn + kn)))
            } else {
                (
                    qg.reshape(&[s, heads * 2 * hd])?,
                    k.reshape(&[s, self.kv_heads as i32 * hd])?,
                )
            };
            if let Some((qk, kk, gate)) = lisa_metal::kernels::gated_pointwise::qk_norm_rope_rows(
                &qg2,
                &k2,
                &self.q_norm.weight,
                &self.k_norm.weight,
                cos_bf,
                sin_bf,
                self.q_norm.eps,
                heads,
                self.kv_heads as i32,
                s,
                &stream,
            ) {
                let v = v.transpose_axes(&[0, 2, 1, 3])?;
                let (live_k, live_v) = match cache.as_deref_mut() {
                    Some(c) => reference_update(c, &kk, &v)?,
                    None => (kk, v),
                };
                let out = fast::scaled_dot_product_attention(
                    &qk,
                    &live_k,
                    &live_v,
                    self.scale,
                    fast::ScaledDotProductAttentionMask::Causal,
                    None,
                )?;
                let gate = gate.reshape(&[b, s, heads, hd])?;
                return self.reference_tail(out, Some(gate), b, s, heads, hd);
            }
        }

        if q.is_none()
            && let Some(qg) = qg_raw.as_ref()
        {
            let qg4 = qg.reshape(&[b, s, heads, 2, hd])?;
            q = Some(qg4.index((Ellipsis, 0, ..)).contiguous()?);
            if gate.is_none() && self.gated {
                gate = Some(qg4.index((Ellipsis, 1, ..)).contiguous()?);
            }
        }
        let q = q.expect("q materialized for the composed path");
        let gate = if self.gated {
            Some(gate.expect("gate materialized for the composed path"))
        } else {
            None
        };

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = rope_partial(&q.transpose_axes(&[0, 2, 1, 3])?, cos_bf, sin_bf)?;
        let k = rope_partial(&k.transpose_axes(&[0, 2, 1, 3])?, cos_bf, sin_bf)?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let (live_k, live_v) = match cache.as_deref_mut() {
            Some(c) => reference_update(c, &k, &v)?,
            None => (k, v),
        };
        let out = if per_slot {
            match cache
                .as_deref()
                .and_then(|c| Self::reference_per_slot(&q, &live_k, &live_v, c, self.scale).ok())
                .flatten()
            {
                Some(o) => o,
                None => fast::scaled_dot_product_attention(
                    &q, &live_k, &live_v, self.scale, sdpa_mask, None,
                )?,
            }
        } else {
            fast::scaled_dot_product_attention(&q, &live_k, &live_v, self.scale, sdpa_mask, None)?
        };
        self.reference_tail(out, gate, b, s, heads, hd)
    }

    fn reference_per_slot(
        q: &Array,
        live_k: &Array,
        live_v: &Array,
        cache: &FullAttentionCache,
        scale: f32,
    ) -> lisa_metal::error::Result<Option<Array>> {
        let Some(next_pos) = &cache.next_pos else {
            return Ok(None);
        };
        if q.dim(2) != 1 || next_pos.len() != q.dim(0) as usize {
            return Ok(None);
        }
        let offset = cache.offset;
        if next_pos.iter().any(|&l| l > offset) {
            return Ok(None);
        }
        if *next_pos.iter().max().unwrap_or(&0) == 0 {
            return Ok(None);
        }
        let b = q.dim(0);
        static PER_SLOT_ENGAGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !PER_SLOT_ENGAGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[batched] per-slot attention engaged (slots={b}): no KV pad or mask on the read"
            );
        }
        let mut outs = Vec::with_capacity(b as usize);
        for i in 0..b {
            let kv_len = next_pos[i as usize] as i32;
            let origin = offset as i32 - kv_len;
            let q_i = q
                .index((i, ..))
                .reshape(&[1, q.dim(1), q.dim(2), q.dim(3)])?; // [1, heads, 1, hd]
            let k_i = live_k.index((i, .., origin..offset as i32, ..)).reshape(&[
                1,
                live_k.dim(1),
                kv_len,
                live_k.dim(3),
            ])?; // [1, kv_h, kl, hd]
            let v_i = live_v.index((i, .., origin..offset as i32, ..)).reshape(&[
                1,
                live_v.dim(1),
                kv_len,
                live_v.dim(3),
            ])?;
            outs.push(fast::scaled_dot_product_attention(
                &q_i,
                &k_i,
                &v_i,
                scale,
                fast::ScaledDotProductAttentionMask::Causal,
                None,
            )?);
        }
        let refs: Vec<&Array> = outs.iter().collect();
        Ok(Some(ops::concatenate(&refs, 0)?))
    }

    fn reference_tail(
        &self,
        out: Array,
        gate: Option<Array>,
        b: i32,
        s: i32,
        heads: i32,
        hd: i32,
    ) -> lisa_metal::error::Result<Array> {
        let out = match gate {
            Some(g) => {
                let stream = lisa_metal::Stream::thread_local_or_default();
                match lisa_metal::kernels::gated_pointwise::sigmoid_mul_tail(
                    &g, &out, b, s, heads, hd, &stream,
                ) {
                    Some(r) => r,
                    None => {
                        let ot = out.transpose_axes(&[0, 2, 1, 3])?; // [b,s,h,hd]
                        let stream = lisa_metal::Stream::thread_local_or_default();
                        match lisa_metal::kernels::gated_pointwise::sigmoid_mul(&g, &ot, &stream) {
                            Some(r) => r,
                            None => {
                                let sig = ops::sigmoid(&g)?;
                                ot.multiply(&sig)?
                            }
                        }
                    }
                }
            }
            None => out
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, heads * hd])?,
        };
        let out = out.as_dtype(Dtype::Bfloat16)?;
        self.o_proj.forward(&out)
    }
}

fn equal(a: &Array, b: &Array, case: &str) {
    assert_eq!(a.shape(), b.shape(), "{case}");
    assert_eq!(
        a.to_vec1::<half::bf16>().unwrap(),
        b.to_vec1::<half::bf16>().unwrap(),
        "{case}"
    );
}
fn tensor(shape: &[i32], salt: i32) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>())
            .map(|i| half::bf16::from_f32(((i * 37 + salt) % 127 - 63) as f32 / 128.))
            .collect::<Vec<_>>(),
        shape,
    )
}
fn projection(output: i32, input: i32) -> QuantizedLinear {
    QuantizedLinear {
        weight: Array::from_slice(
            &(0..output * input / 8)
                .map(|i| (i as u32).wrapping_mul(1664525).wrapping_add(1013904223))
                .collect::<Vec<_>>(),
            &[output, input / 8],
        ),
        scales: tensor(&[output, input / 64], 13),
        biases: tensor(&[output, input / 64], 27),
        group_size: 64,
        bits: 4,
    }
}
#[test]
fn attention_plan_matches_original_paths_and_cache_bytes() -> anyhow::Result<()> {
    for width in [128, 256] {
        for gated in [false, true] {
            let (h, kh, hidden) = (4, 2, 256);
            let q_proj = projection(h * 2 * width, hidden);
            let model =
                Qwen35Attention(lisa_engine::bindings::qwen35::attention::Qwen35Attention {
                    context: crate::metal_context::execution_context(),
                    q_proj,
                    qkv_proj: None,
                    k_proj: projection(kh * width, hidden),
                    v_proj: projection(kh * width, hidden),
                    o_proj: projection(hidden, h * width),
                    q_norm: RmsNorm {
                        weight: tensor(&[width], 7),
                        eps: 1e-6,
                        group_size: None,
                    },
                    k_norm: RmsNorm {
                        weight: tensor(&[width], 11),
                        eps: 1e-6,
                        group_size: None,
                    },
                    heads: h as usize,
                    kv_heads: kh as usize,
                    head_dim: width as usize,
                    scale: (width as f32).powf(-0.5),
                    gated,
                    plans: Default::default(),
                });
            for batch in [1, 2] {
                let mut a = FullAttentionCache::new(
                    crate::metal_context::execution_context(),
                    kh as usize,
                    width as usize,
                );
                let mut b = FullAttentionCache::new(
                    crate::metal_context::execution_context(),
                    kh as usize,
                    width as usize,
                );
                // Starts cold, reuses storage, crosses capacity, then exercises rollback.
                for (step, rows) in (if batch == 1 {
                    [1, 2, 7, 9, 256, 1, 3]
                } else {
                    [1; 7]
                })
                .into_iter()
                .enumerate()
                {
                    let x = tensor(&[batch, rows, hidden], step as i32);
                    let cos = tensor(&[batch, 1, rows, 64], 43);
                    let sin = tensor(&[batch, 1, rows, 64], 31);
                    let case = format!(
                        "width={width} gated={gated} batch={batch} rows={rows} step={step}"
                    );
                    let expected = model
                        .reference_forward(&x, &cos, &sin, Some(&mut a), 0)
                        .with_context(|| format!("reference {case}"))?;
                    let got = model
                        .forward(&x, &cos, &sin, Some(&mut b), 0)
                        .with_context(|| format!("plan {case}"))?;
                    equal(&got, &expected, &case);
                    assert_eq!((a.offset, a.capacity), (b.offset, b.capacity), "{case}");
                    equal(a.keys.as_ref().unwrap(), b.keys.as_ref().unwrap(), &case);
                    equal(
                        a.values.as_ref().unwrap(),
                        b.values.as_ref().unwrap(),
                        &case,
                    );
                    if step == 4 {
                        a.trim(3);
                        b.trim(3);
                    }
                }
                let x = tensor(&[batch, 1, hidden], 5);
                let cos = tensor(&[batch, 1, 1, 64], 43);
                let sin = tensor(&[batch, 1, 1, 64], 31);
                equal(
                    &model.forward(&x, &cos, &sin, None, 0)?,
                    &model.reference_forward(&x, &cos, &sin, None, 0)?,
                    "no cache",
                );
                if batch == 1 {
                    let cf = cos.as_dtype(Dtype::Float32)?;
                    let sf = sin.as_dtype(Dtype::Float32)?;
                    equal(
                        &model.forward(&x, &cf, &sf, None, 0)?,
                        &model.reference_forward(&x, &cf, &sf, None, 0)?,
                        "f32 frequencies",
                    );
                }
                if batch == 2 {
                    a.next_pos = Some(vec![a.offset + 1, a.offset - 2]);
                    b.next_pos = a.next_pos.clone();
                    let mask = Array::from_slice(
                        &vec![true; 2 * (a.offset + 1)],
                        &[2, 1, 1, (a.offset + 1) as i32],
                    );
                    a.attn_mask = Some(mask.clone());
                    b.attn_mask = Some(mask);
                    equal(
                        &model.forward(&x, &cos, &sin, Some(&mut b), 0)?,
                        &model.reference_forward(&x, &cos, &sin, Some(&mut a), 0)?,
                        "ragged",
                    );
                }
            }
        }
    }
    Ok(())
}

fn reference_grow(
    c: &mut FullAttentionCache,
    needed: usize,
    b: i32,
) -> lisa_metal::error::Result<()> {
    let same_batch = c.keys.as_ref().is_none_or(|k| k.dim(0) == b);
    if c.capacity >= needed && same_batch {
        return Ok(());
    }
    let mut new_cap = if c.capacity == 0 {
        needed + 256
    } else {
        c.capacity
    };
    while new_cap < needed {
        new_cap *= 2;
    }
    let shape = &[b, c.kv_heads as i32, new_cap as i32, c.head_dim as i32];
    let zeros = || match c.dtype {
        lisa_backend::ScalarType::F32 => ops::zeros::<f32>(shape),
        _ => ops::zeros::<half::bf16>(shape),
    };
    let mut new_keys = zeros()?;
    let mut new_values = zeros()?;
    if let (Some(old_k), Some(old_v)) = (&c.keys, &c.values) {
        let range = 0..c.offset as i32;
        let old_k = old_k.index((.., .., range.clone(), ..));
        let old_v = old_v.index((.., .., range.clone(), ..));
        new_keys.index_mut((.., .., range.clone(), ..), old_k);
        new_values.index_mut((.., .., range, ..), old_v);
    }
    c.keys = Some(new_keys);
    c.values = Some(new_values);
    c.capacity = new_cap;
    Ok(())
}

fn reference_update(
    c: &mut FullAttentionCache,
    keys: &Array,
    values: &Array,
) -> lisa_metal::error::Result<(Array, Array)> {
    let s = keys.dim(2) as usize;
    reference_grow(c, c.offset + s, keys.dim(0))?;
    let stream = lisa_metal::Stream::thread_local_or_default();
    let off = c.offset as i32;
    let mut done = true;
    {
        let k = c.keys.as_ref().unwrap();
        done &= lisa_metal::kernels::copy_rows_into(keys, k, off, &stream).is_some();
        let v = c.values.as_ref().unwrap();
        done &= lisa_metal::kernels::copy_rows_into(values, v, off, &stream).is_some();
    }
    if !done {
        let range = c.offset as i32..(c.offset + s) as i32;
        let k = c.keys.as_mut().unwrap();
        k.index_mut((.., .., range.clone(), ..), keys.clone());
        let v = c.values.as_mut().unwrap();
        v.index_mut((.., .., range, ..), values.clone());
    }
    c.offset += s;
    let live = 0..c.offset as i32;
    Ok((
        c.keys.as_ref().unwrap().index((.., .., live.clone(), ..)),
        c.values.as_ref().unwrap().index((.., .., live, ..)),
    ))
}

#[test]
fn shared_cache_plan_preserves_bf16_and_f32_storage() -> anyhow::Result<()> {
    for dtype in [Dtype::Bfloat16, Dtype::Float32] {
        let mut expected =
            FullAttentionCache::with_dtype(crate::metal_context::execution_context(), 2, 64, dtype);
        let mut actual =
            FullAttentionCache::with_dtype(crate::metal_context::execution_context(), 2, 64, dtype);
        for rows in [1, 7, 256, 3, 1] {
            let x = tensor(&[2, rows, 2, 64], 31).as_dtype(dtype)?;
            let k = x.transpose_axes(&[0, 2, 1, 3])?;
            let v = tensor(&[2, rows, 2, 64], 97)
                .as_dtype(dtype)?
                .transpose_axes(&[0, 2, 1, 3])?;
            let want = reference_update(&mut expected, &k, &v)?;
            let got = actual.update(&k, &v)?;
            for (a, b) in [
                (&want.0, &got.0),
                (&want.1, &got.1),
                (
                    expected.keys.as_ref().unwrap(),
                    actual.keys.as_ref().unwrap(),
                ),
                (
                    expected.values.as_ref().unwrap(),
                    actual.values.as_ref().unwrap(),
                ),
            ] {
                assert_eq!(a.shape(), b.shape());
                let bits = |a: &Array| {
                    a.as_dtype(Dtype::Float32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(a), bits(b));
            }
            assert_eq!(
                (expected.offset, expected.capacity),
                (actual.offset, actual.capacity)
            );
        }
    }
    Ok(())
}
