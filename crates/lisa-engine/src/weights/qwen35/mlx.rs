//! Qwen affine checkpoint import, binding and draft sidecar preparation.
//! Execution receives device tensors; all checkpoint paths remain in this adapter.
use crate::bindings::{
    qwen35::{attention::Qwen35Attention, decoder::DecoderLayer, mlx::MlxExecution},
    shared::{
        dense::DenseMlp,
        gated_delta::GatedDeltaNet,
        normalization::{RmsNorm, Rotary},
    },
};
use crate::weights::affine::{QuantizedEmbedding, QuantizedLinear};
use crate::weights::{
    mlx::{Checkpoint, Shard, get_tensor},
    source::Weights,
};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{HostTensor, ManagedDevice, ScalarType as Dtype, TensorAccess};
use lisa_models::{
    checkpoint::sanitize_name,
    qwen3_5::{cache_law, config::Qwen35Config},
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn load<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    config: Qwen35Config,
) -> anyhow::Result<MlxExecution<D>> {
    let checkpoint = Checkpoint::open(dir)?;
    let map = checkpoint.weight_map.clone();
    let mut shard_order: Vec<String> = Vec::new();
    for shard in map.values() {
        if !shard_order.contains(shard) {
            shard_order.push(shard.clone());
        }
    }
    let mut weights: Weights<D::Tensor> = HashMap::new();
    let t_shards = std::time::Instant::now();
    for shard_name in &shard_order {
        let shard = Shard::open(context.device().clone(), &dir.join(shard_name))?;
        for name in shard.tensor_names() {
            if let Some(k) = sanitize_name(name)
                && let std::collections::hash_map::Entry::Vacant(e) = weights.entry(k)
            {
                e.insert(get_tensor(&shard, name)?);
            }
        }
    }
    crate::ttft_mark(&format!(
        "model.load: {} shards read (mmap+host dequant) {:.2}s",
        shard_order.len(),
        t_shards.elapsed().as_secs_f64()
    ));
    // MTP head: a sibling `mtp/weights.safetensors`. The sidecar's tensor
    // names already carry the `mtp.` prefix (e.g. `mtp.fc.weight`); keys
    // are stored as-is so the head reads them back unchanged.
    let mtp_file = dir.join("mtp").join("weights.safetensors");
    if mtp_file.is_file() {
        let shard = Shard::open(context.device().clone(), &mtp_file)?;
        for name in shard.tensor_names() {
            let key = if name.starts_with("mtp.") {
                sanitize_name(name).unwrap_or_else(|| name.clone())
            } else {
                format!("mtp.{name}")
            };
            if let std::collections::hash_map::Entry::Vacant(e) = weights.entry(key) {
                e.insert(get_tensor(&shard, name)?);
            }
        }
    }
    let mut tower = from_weights_cached(context.clone(), weights, config, Some(dir))?;
    tower.vision = crate::weights::vision::load(context.clone(), dir, tower.rope.dimensions)?;
    Ok(tower)
}

pub fn from_weights<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    w: Weights<D::Tensor>,
    config: Qwen35Config,
) -> anyhow::Result<MlxExecution<D>> {
    from_weights_cached(context.clone(), w, config, None)
}

pub fn from_weights_cached<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    mut w: Weights<D::Tensor>,
    config: Qwen35Config,
    cache_dir: Option<&Path>,
) -> anyhow::Result<MlxExecution<D>> {
    let t_fw = std::time::Instant::now();
    let eps = config.rms_norm_eps;
    let gs = config.quant_group_size;
    let bits = config.quant_bits;

    crate::memory::report(context.device(), "t0-mmap-only");
    let mut embed_tokens = QuantizedEmbedding::load(&mut w, "model.embed_tokens")?;
    embed_tokens.set_quant(gs, bits);
    let mut lm_head = if config.tie_word_embeddings {
        QuantizedLinear {
            weight: embed_tokens.weight.clone(),
            scales: embed_tokens.scales.clone(),
            biases: embed_tokens.biases.clone(),
            group_size: embed_tokens.group_size,
            bits: embed_tokens.bits,
        }
    } else {
        QuantizedLinear::load(&mut w, "lm_head", "")?
    };
    lm_head.set_quant(gs, bits);
    crate::memory::report(context.device(), "t1-heads");

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let is_full = config.is_full(i);
        let attn = if is_full {
            Some(Qwen35Attention::load(
                context.clone(),
                &mut w,
                &format!("{p}.self_attn"),
                eps,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.attn_output_gate,
                gs,
                bits,
            )?)
        } else {
            None
        };
        let mut gdn = if is_full {
            None
        } else {
            Some(GatedDeltaNet::load(
                context.clone(),
                &mut w,
                &format!("{p}.linear_attn"),
                eps,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                gs,
                bits,
            )?)
        };
        if let Some(g) = gdn.as_mut() {
            g.output_gate_silu = true;
        }
        layers.push(DecoderLayer {
            context: context.clone(),
            plan: crate::core::decoder::PreparedDecoder::new(lisa_models::qwen3_5::decoder(
                is_full,
            ))?,
            attn,
            gdn,
            input_norm: RmsNorm::load_on(
                &context,
                &mut w,
                &format!("{p}.input_layernorm"),
                eps,
                None,
            )?,
            post_norm: RmsNorm::load_on(
                &context,
                &mut w,
                &format!("{p}.post_attention_layernorm"),
                eps,
                None,
            )?,
            mlp: DenseMlp::load(context.clone(), &mut w, &format!("{p}.mlp"), gs, bits)?,
        });
    }
    let norm = RmsNorm::load_on(&context, &mut w, "model.norm", eps, None)?;
    crate::memory::report(context.device(), "t2-64-layers");
    let mtp = crate::bindings::qwen35::mtp::MtpHead::load(context.clone(), &mut w, &config)?;
    crate::memory::report(context.device(), "t3-mtp-head");
    let t_draft = std::time::Instant::now();
    // The default draft head shortlists with a cached 2-bit projection,
    // then rescores its candidates through the original trunk head.
    // See build_draft_lm_head_cached.
    let draft_head = if mtp.is_some() {
        build_draft_lm_head_cached(context.clone(), &lm_head, cache_dir)
    } else {
        None
    };
    crate::memory::report(context.device(), "t4-draft-head");
    // The map's explicit teardown. The loader relies on the HashMap being
    // dropped at the end of this function; do it here instead, so the
    // release is visible, and report what was never claimed — an entry still
    // present is a tensor the model does not use, and it must not stay
    // resident. Anything left is either a key this build never asks for or a
    // name the accessors miss, and its byte count is the difference between
    // what the checkpoint holds and what the model actually needs.
    let unclaimed = w.len();
    let unclaimed_bytes: usize = w
        .values()
        .map(|a| {
            a.size()
                * context
                    .device()
                    .dtype(a)
                    .expect("known imported dtype")
                    .size_of()
        })
        .sum();
    w.clear();
    eprintln!(
        "[load] map teardown: {unclaimed} unclaimed tensors, {:.1} MB freed",
        unclaimed_bytes as f64 / 1e6
    );
    crate::memory::report(context.device(), "t5-map-cleared");
    crate::ttft_mark(&format!(
        "model.load: tower build {:.2}s (draft head {:.2}s inside)",
        t_fw.elapsed().as_secs_f64(),
        t_draft.elapsed().as_secs_f64()
    ));
    // Bound before any pipeline compiles.
    let cache_law = cache_law(&config, &context.device().identity().arithmetic);
    Ok(MlxExecution {
        context,
        cache_law,
        vision: None,
        embed_tokens,
        layers,
        norm,
        lm_head,
        draft_head,
        rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
        mtp,
        config,
    })
}

/// The draft-only low-bit lm_head: a requantized coarse copy of the
/// trunk head, used ONLY by `draft_step`. Verification always reads
/// the trunk head, so the output distribution is untouched — drafts just
/// read fewer bytes per full-vocab projection, the dominant draft cost.
/// Width is 2-bit/gs64, lisa's default coarse width (a 3-bit width would
/// need MLX's permuted 3-bit packing, which our kernels decode but no
/// host encoder here reproduces yet). Built once at load when the
/// checkpoint carries an MTP head; `None` keeps the trunk head.
/// `[`build_draft_lm_head`]` with an optional on-disk cache. The
/// requant costs ~5.2 s per process (GPU dequant of the [V, K] trunk
/// head + host requant) and is DETERMINISTIC, so the packed result is
/// persisted next to the checkpoint and reloaded verbatim — bit-identical
/// to a fresh rebuild (same tensors, same layout). A missing/corrupt/
/// mismatched cache falls back to the rebuild and rewrites the file.
fn build_draft_lm_head_cached<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    lm_head: &QuantizedLinear<D::Tensor>,
    cache_dir: Option<&Path>,
) -> Option<QuantizedLinear<D::Tensor>> {
    const DRAFT_BITS: i32 = 2;
    const DRAFT_GROUP: i32 = 64;
    if DRAFT_BITS >= lm_head.bits {
        return None; // no byte saving over the trunk head
    }
    let t0 = std::time::Instant::now();
    if let Some(dir) = cache_dir
        && let Some(dh) =
            load_draft_lm_head_cache(context.clone(), dir, DRAFT_GROUP, DRAFT_BITS, lm_head)
    {
        eprintln!(
            "[mtp] draft-only lm_head loaded from cache in {:.2}s",
            t0.elapsed().as_secs_f32()
        );
        return Some(dh);
    }
    // Dequantize on the GPU ([V, K] in the scales' dtype), requantize on
    // the host (the lisa-metal `quantize` is a host kernel), materialize.
    let (wq, scales, biases) = crate::bindings::shared::affine::requantize(
        &context,
        &lm_head.weight,
        &lm_head.scales,
        &lm_head.biases,
        (lm_head.group_size, lm_head.bits),
        (DRAFT_GROUP, DRAFT_BITS),
    )
    .ok()?;
    if let Some(dir) = cache_dir {
        let _ = save_draft_lm_head_cache(
            context.clone(),
            dir,
            DRAFT_GROUP,
            DRAFT_BITS,
            &wq,
            &scales,
            &biases,
        );
    }
    eprintln!(
        "[mtp] draft-only lm_head requantized to {DRAFT_BITS}-bit/gs{DRAFT_GROUP} in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    Some(QuantizedLinear {
        weight: wq,
        scales,
        biases,
        group_size: DRAFT_GROUP,
        bits: DRAFT_BITS,
    })
}

/// Cache-file layout: magic "LSDH1" | gs:i32 | bits:i32 | V:usize | K:usize
/// | packed-weight u32 bytes | scales bf16 bytes | biases bf16 bytes. The
/// V×K/dtype guard makes a stale file (different checkpoint geometry)
/// detectable by size alone.
fn draft_head_cache_path(dir: &Path) -> PathBuf {
    dir.join("mtp").join("draft_head_2bit_gs64.lsdh")
}

fn load_draft_lm_head_cache<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    gs: i32,
    bits: i32,
    reference: &QuantizedLinear<D::Tensor>,
) -> Option<QuantizedLinear<D::Tensor>> {
    let data = std::fs::read(draft_head_cache_path(dir)).ok()?;
    if data.len() < 8 + 4 + 4 + 8 + 8 {
        return None;
    }
    if &data[0..5] != b"LSDH1" {
        return None;
    }
    let rd_i32 = |off: usize| -> Option<i32> {
        Some(i32::from_le_bytes(data.get(off..off + 4)?.try_into().ok()?))
    };
    let rd_usize = |off: usize| -> Option<usize> {
        Some(usize::from_le_bytes(
            data.get(off..off + 8)?.try_into().ok()?,
        ))
    };
    if rd_i32(5)? != gs || rd_i32(9)? != bits {
        return None;
    }
    let v = rd_usize(13)?;
    let k = rd_usize(21)?;
    // `k` is the LOGICAL (unpacked) head width; the packed weight rows
    // are k*bits/32 u32 words. The trunk head supplies the expected
    // geometry so a different checkpoint is detected by value.
    // The trunk head's weight is ALSO packed (reference.bits per value),
    // so its logical width is dim(1)*32/bits — compare logical to logical.
    let ref_k = reference.weight.dim(1) as usize * 32 / reference.bits as usize;
    if v != reference.weight.dim(0) as usize || k != ref_k {
        return None;
    }
    let mut off = 29usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = data.get(off..off.checked_add(n)?)?;
        off += n;
        Some(s)
    };
    let wq_bytes = take(v * k * bits as usize / 8)?; // packed: bits/value
    let scales_bytes = take(v * (k / gs as usize) * 2)?; // bf16
    let biases_bytes = take(v * (k / gs as usize) * 2)?;
    let wq = context
        .device()
        .pull(
            HostTensor {
                bytes: wq_bytes,
                shape: &[v as i32, (k * bits as usize / 32) as i32],
                dtype: Dtype::U32,
            }
            .into(),
            Default::default(),
        )
        .ok()?;
    let scales = context
        .device()
        .pull(
            HostTensor {
                bytes: scales_bytes,
                shape: &[v as i32, k as i32 / gs],
                dtype: Dtype::BF16,
            }
            .into(),
            Default::default(),
        )
        .ok()?;
    let biases = context
        .device()
        .pull(
            HostTensor {
                bytes: biases_bytes,
                shape: &[v as i32, k as i32 / gs],
                dtype: Dtype::BF16,
            }
            .into(),
            Default::default(),
        )
        .ok()?;
    Some(QuantizedLinear {
        weight: wq,
        scales,
        biases,
        group_size: gs,
        bits,
    })
}

fn save_draft_lm_head_cache<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    gs: i32,
    bits: i32,
    wq: &D::Tensor,
    scales: &D::Tensor,
    biases: &D::Tensor,
) -> anyhow::Result<()> {
    // `wq` rows are PACKED (bits/value): words per row = k*bits/32, so the
    // logical width k = words*32/bits.
    let v = wq.dim(0) as usize;
    let k = wq.dim(1) as usize * 32 / bits as usize;
    let mut out = Vec::with_capacity(29 + wq.size() * 4 + (scales.size() + biases.size()) * 2);
    out.extend_from_slice(b"LSDH1");
    out.extend_from_slice(&gs.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(&(v as u64).to_le_bytes());
    out.extend_from_slice(&(k as u64).to_le_bytes());
    out.extend_from_slice(&context.device().read_bytes(wq)?);
    out.extend_from_slice(&context.device().read_bytes(scales)?);
    out.extend_from_slice(&context.device().read_bytes(biases)?);
    let path = draft_head_cache_path(dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("lsdh.tmp");
    std::fs::write(&tmp, &out)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}
