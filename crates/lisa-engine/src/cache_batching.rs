//! Portable cache packing and survivor extraction for the engine scheduler.
use crate::{
    attention_cache::FullAttentionCache,
    cache::LayerCache,
    execution::DeviceContext,
    indexer_cache::{CacheDevice, IndexerTape},
    recurrent_cache::GdnCache,
    tensor_ops::{TensorLaw, TensorOps},
};
use lisa_backend::{Kernel, TensorAccess, TensorOperation as Op};

/// Pack one layer's per-stream caches into a single batched cache.
pub fn pack_layer<D: CacheDevice>(
    context: &DeviceContext<D>,
    per: &[Vec<LayerCache<D>>],
    li: usize,
    lens: &[usize],
    lmax: usize,
    cap: usize,
) -> anyhow::Result<LayerCache<D>> {
    anyhow::ensure!(
        per.len() == lens.len(),
        "cache pack has {} streams but {} position lengths",
        per.len(),
        lens.len()
    );
    anyhow::ensure!(
        !per.is_empty() && per.iter().all(|c| li < c.len()),
        "invalid cache pack layer"
    );
    anyhow::ensure!(
        cap >= lmax && lens.iter().all(|&len| len <= lmax),
        "invalid cache pack positions"
    );
    let ops = TensorOps::new(context.device(), TensorLaw::Native);
    match &per[0][li] {
        LayerCache::Native(cache) => {
            anyhow::ensure!(
                per.len() == 1,
                "native cache batching requires a model-specific implementation"
            );
            Ok(LayerCache::Native(cache.clone()))
        }
        LayerCache::Full(tmpl) => {
            if tmpl.keys.is_none() && tmpl.values.is_none() {
                let mut tapes = Vec::with_capacity(per.len());
                for caches in per {
                    let LayerCache::Full(cache) = &caches[li] else {
                        anyhow::bail!("cache kind mismatch");
                    };
                    anyhow::ensure!(
                        cache.keys.is_none() && cache.values.is_none(),
                        "full-attention cache has mixed expanded-KV and indexer-tape state"
                    );
                    tapes.push(&cache.indexer_tape);
                }
                anyhow::ensure!(
                    lens.iter().all(|&len| len == lens[0]),
                    "GLM indexer-tape batching requires equal stream positions"
                );
                let indexer_tape = IndexerTape::pack_equal(context.clone(), &tapes)?;
                anyhow::ensure!(
                    indexer_tape.total == lens[0],
                    "GLM indexer tape length {} differs from packed stream position {}",
                    indexer_tape.total,
                    lens[0]
                );
                return Ok(LayerCache::Full(FullAttentionCache {
                    dtype: tmpl.dtype,
                    keys: None,
                    values: None,
                    offset: lmax,
                    capacity: cap,
                    kv_heads: tmpl.kv_heads,
                    head_dim: tmpl.head_dim,
                    indexer_tape,
                    indexer_ok: per.iter().all(|caches| match &caches[li] {
                        LayerCache::Full(cache) => cache.indexer_ok,
                        LayerCache::Linear(_) | LayerCache::Native(_) => false,
                    }),
                    ple_conv: None,
                    capture_ple_full: None,
                    next_pos: None,
                    attn_mask: None,
                }));
            }
            anyhow::ensure!(
                tmpl.keys.is_some() && tmpl.values.is_some(),
                "full-attention cache contains only one expanded KV tensor"
            );
            let h = tmpl.kv_heads as i32;
            let d = tmpl.head_dim as i32;
            let n = per.len() as i32;
            let zeros = || {
                ops.required(
                    Kernel::Zeros {
                        shape: vec![n, h, cap as i32, d],
                        dtype: tmpl.dtype,
                    },
                    &[],
                )
            };
            let mut keys = zeros()?;
            let mut values = zeros()?;
            for (i, caches) in per.iter().enumerate() {
                let LayerCache::Full(c) = &caches[li] else {
                    anyhow::bail!("cache kind mismatch");
                };
                let li_len = lens[i] as i32;
                let dst = lmax as i32 - li_len;
                if let Some(k) = &c.keys {
                    let src = k.slice(2, 0, Some(li_len))?;
                    keys = ops.op(
                        Op::AssignRegion(vec![
                            (i as i32, i as i32 + 1),
                            (0, h),
                            (dst, dst + li_len),
                            (0, d),
                        ]),
                        &[&keys, &src],
                    )?;
                }
                if let Some(v) = &c.values {
                    let src = v.slice(2, 0, Some(li_len))?;
                    values = ops.op(
                        Op::AssignRegion(vec![
                            (i as i32, i as i32 + 1),
                            (0, h),
                            (dst, dst + li_len),
                            (0, d),
                        ]),
                        &[&values, &src],
                    )?;
                }
            }
            Ok(LayerCache::Full(FullAttentionCache {
                dtype: tmpl.dtype,
                keys: Some(keys),
                values: Some(values),
                offset: lmax,
                capacity: cap,
                kv_heads: tmpl.kv_heads,
                head_dim: tmpl.head_dim,
                indexer_tape: IndexerTape::new(context.clone()),
                indexer_ok: false,
                ple_conv: None,
                capture_ple_full: None,
                next_pos: Some(lens.to_vec()),
                attn_mask: None,
            }))
        }
        LayerCache::Linear(_) => {
            let conv = concat_opt::<D>(per, li, |c| c.conv.as_ref())?;
            let ssm = concat_opt::<D>(per, li, |c| c.ssm.as_ref())?;
            let ple = concat_opt::<D>(per, li, |c| c.ple_conv.as_ref())?;
            Ok(LayerCache::Linear(GdnCache {
                conv,
                ssm,
                capture_ssm: None,
                capture_replay: None,
                capture_conv_input: None,
                rec_y_buf: None,
                ple_conv: ple,
                capture_ple_full: None,
            }))
        }
    }
}

fn concat_opt<D: CacheDevice>(
    per: &[Vec<LayerCache<D>>],
    li: usize,
    get: impl Fn(&GdnCache<D::Tensor>) -> Option<&D::Tensor>,
) -> anyhow::Result<Option<D::Tensor>> {
    let mut parts: Vec<&D::Tensor> = Vec::new();
    for caches in per {
        let LayerCache::Linear(g) = &caches[li] else {
            anyhow::bail!("cache kind mismatch");
        };
        match get(g) {
            Some(a) => parts.push(a),
            None => return Ok(None),
        }
    }
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(D::Tensor::concatenate(&parts, 0)?))
}

/// Continuous batching: streams are admitted and retired as they arrive and
/// finish, and the packed caches are compacted on every membership change.
///
/// The layout is the same right-aligned packing as [`crate::batching::Batch::prefill_ragged`]
/// (`next_pos` + per-stream `attn_mask`). When only one stream is live the
/// packing is collapsed to a contiguously left-aligned cache with no mask, so a
/// lone stream takes the exact single-stream decode path.
/// Slice the packed caches back into per-stream (B=1) caches for `survivors`,
/// each holding exactly its live tokens (right-aligned `[origin, offset)`).
pub fn extract_streams<D: CacheDevice>(
    packed: &[LayerCache<D>],
    survivors: &[usize],
    pack_lens: &[usize],
    lmax: usize,
) -> anyhow::Result<Vec<Vec<LayerCache<D>>>> {
    let offset = packed
        .iter()
        .find_map(|c| match c {
            LayerCache::Full(f) => Some(f.offset),
            _ => None,
        })
        .unwrap_or(0);
    let mut out = Vec::with_capacity(survivors.len());
    anyhow::ensure!(
        survivors
            .iter()
            .all(|&i| i < pack_lens.len() && pack_lens[i] <= lmax),
        "invalid cache survivor"
    );
    for &i in survivors {
        let origin = lmax - pack_lens[i];
        let range = origin as i32..offset as i32;
        let row = i as i32..i as i32 + 1;
        let mut layers = Vec::with_capacity(packed.len());
        for c in packed {
            match c {
                LayerCache::Native(cache) => {
                    anyhow::ensure!(
                        pack_lens.len() == 1 && i == 0,
                        "native cache extraction only supports one stream"
                    );
                    layers.push(LayerCache::Native(cache.clone()));
                }
                LayerCache::Full(f) => {
                    let (k, v, indexer_tape, indexer_ok) = match (&f.keys, &f.values) {
                        (Some(keys), Some(values)) => (
                            Some(
                                keys.slice(0, row.start, Some(row.end))?
                                    .slice(2, range.start, Some(range.end))?
                                    .contiguous()?,
                            ),
                            Some(
                                values
                                    .slice(0, row.start, Some(row.end))?
                                    .slice(2, range.start, Some(range.end))?
                                    .contiguous()?,
                            ),
                            IndexerTape::new(f.indexer_tape.context.clone()),
                            false,
                        ),
                        (None, None) => (None, None, f.indexer_tape.extract_row(i)?, f.indexer_ok),
                        _ => anyhow::bail!(
                            "full-attention cache contains only one expanded KV tensor"
                        ),
                    };
                    layers.push(LayerCache::Full(FullAttentionCache {
                        dtype: f.dtype,
                        keys: k,
                        values: v,
                        offset: offset - origin,
                        capacity: offset - origin,
                        kv_heads: f.kv_heads,
                        head_dim: f.head_dim,
                        indexer_tape,
                        indexer_ok,
                        ple_conv: None,
                        capture_ple_full: None,
                        next_pos: None,
                        attn_mask: None,
                    }));
                }
                LayerCache::Linear(g) => {
                    let one = |a: &Option<D::Tensor>| -> anyhow::Result<Option<D::Tensor>> {
                        match a {
                            Some(x) => {
                                Ok(Some(x.slice(0, row.start, Some(row.end))?.contiguous()?))
                            }
                            None => Ok(None),
                        }
                    };
                    layers.push(LayerCache::Linear(GdnCache {
                        conv: one(&g.conv)?,
                        ssm: one(&g.ssm)?,
                        ple_conv: one(&g.ple_conv)?,
                        capture_ssm: None,
                        capture_replay: None,
                        capture_conv_input: None,
                        rec_y_buf: None,
                        capture_ple_full: None,
                    }));
                }
            }
        }
        out.push(layers);
    }
    Ok(out)
}
