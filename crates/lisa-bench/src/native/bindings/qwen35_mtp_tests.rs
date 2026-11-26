#![cfg(test)]
//! Pre-migration MTP orchestration retained as a numerical oracle.
use super::*;
use crate::core::norm::positions;
use crate::core::quant::QuantizedEmbeddingExecution;
use crate::core::quant::QuantizedLinear;
use crate::core::quant::QuantizedLinearExecution;
use lisa_metal::ops::indexing::IndexOp;
use std::collections::HashMap;
trait MtpReference {
    fn reference_forward(
        &mut self,
        tokens: &Array,
        hidden: &Array,
        embed_tokens: &QuantizedEmbedding,
        offset: usize,
    ) -> anyhow::Result<(Array, Array)>;
}
impl MtpReference for MtpHead {
    fn reference_forward(
        &mut self,
        tokens: &Array,
        hidden: &Array,
        embed_tokens: &QuantizedEmbedding,
        offset: usize,
    ) -> anyhow::Result<(Array, Array)> {
        let s = tokens.dim(1) as usize;

        let e = self
            .pre_fc_norm_embedding
            .forward(&embed_tokens.forward(tokens)?)?;
        let h = self.pre_fc_norm_hidden.forward(hidden)?;
        anyhow::ensure!(
            e.dim(1) == h.dim(1) && e.dim(-1) == h.dim(-1),
            "mtp head prime shape mismatch: tokens [{}, {}] e [{}, {}, {}] h [{}, {}, {}]",
            tokens.dim(0),
            tokens.dim(1),
            e.dim(0),
            e.dim(1),
            e.dim(-1),
            h.dim(0),
            h.dim(1),
            h.dim(-1)
        );
        let cat =
            lisa_metal::ops::concatenate(&[&e, &h], -1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let x = self.fc.forward(&cat)?;

        let pos = positions(offset, s)?;
        let (cos, sin) = self.rope.cos_sin(&self.context, &pos)?;
        let cos = cos.expand_dims(1)?;
        let sin = sin.expand_dims(1)?;
        // Same hoist as tower.forward_inner: bf16 cast once.
        let cos_bf = cos.as_dtype(lisa_metal::Dtype::Bfloat16)?;
        let sin_bf = sin.as_dtype(lisa_metal::Dtype::Bfloat16)?;
        let cache = self.caches.first_mut();
        let (sum, m) = self
            .layer
            .forward(&x, None, cache, &cos_bf, &sin_bf, offset, false, false)?;
        let hidden_next = self.norm.forward_residual_add(&sum, &m)?.1;
        let last = hidden_next
            .index((.., hidden_next.dim(1) - 1, ..))
            .contiguous()?;
        Ok((hidden_next, last))
    }
}

fn same(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape());
    let bits = |a: &Array| {
        a.to_vec1::<half::bf16>()
            .unwrap()
            .into_iter()
            .map(half::bf16::to_bits)
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(a), bits(b));
}
fn projection(n: i32, k: i32, bits: i32, gs: i32) -> QuantizedLinear {
    QuantizedLinear {
        weight: Array::from_slice(
            &(0..n * k * bits / 32)
                .map(|i| (i as u32).wrapping_mul(0x1234567).wrapping_add(0x89abcdef))
                .collect::<Vec<_>>(),
            &[n, k * bits / 32],
        ),
        scales: Array::from_slice(
            &vec![half::bf16::from_f32(1. / 128.); (n * k / gs) as usize],
            &[n, k / gs],
        ),
        biases: Array::from_slice(
            &vec![half::bf16::from_f32(-0.0625); (n * k / gs) as usize],
            &[n, k / gs],
        ),
        group_size: gs,
        bits,
    }
}
fn weights(
    bits: i32,
    width: i32,
) -> (
    HashMap<String, Array>,
    lisa_models::qwen3_5::config::Qwen35Config,
    QuantizedEmbedding,
) {
    let mut config = lisa_models::qwen3_5::config::Qwen35Config::parse("{}").unwrap();
    config.hidden_size = 256;
    config.intermediate_size = 512;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = width as usize;
    config.partial_rotary_factor = 64. / width as f32;
    config.quant_group_size = 64;
    config.quant_bits = bits;
    let mut w = HashMap::new();
    for (name, n, k) in [
        ("mtp.fc", 256, 512),
        ("mtp.layers.0.self_attn.q_proj", 4 * width, 256),
        ("mtp.layers.0.self_attn.k_proj", width, 256),
        ("mtp.layers.0.self_attn.v_proj", width, 256),
        ("mtp.layers.0.self_attn.o_proj", 256, 2 * width),
        ("mtp.layers.0.mlp.gate_proj", 512, 256),
        ("mtp.layers.0.mlp.up_proj", 512, 256),
        ("mtp.layers.0.mlp.down_proj", 256, 512),
    ] {
        let p = projection(n, k, bits, 64);
        w.insert(format!("{name}.weight"), p.weight);
        w.insert(format!("{name}.scales"), p.scales);
        w.insert(format!("{name}.biases"), p.biases);
    }
    for (name, n) in [
        ("mtp.pre_fc_norm_embedding", 256),
        ("mtp.pre_fc_norm_hidden", 256),
        ("mtp.norm", 256),
        ("mtp.layers.0.input_layernorm", 256),
        ("mtp.layers.0.post_attention_layernorm", 256),
        ("mtp.layers.0.self_attn.q_norm", width),
        ("mtp.layers.0.self_attn.k_norm", width),
    ] {
        w.insert(
            format!("{name}.weight"),
            Array::from_slice(&vec![half::bf16::from_f32(0.875); n as usize], &[n]),
        );
    }
    let p = projection(32, 256, bits, 64);
    (
        w,
        config,
        QuantizedEmbedding {
            weight: p.weight,
            scales: p.scales,
            biases: p.biases,
            group_size: 64,
            bits,
        },
    )
}
#[test]
fn full_draft_graph_preserves_hidden_and_all_cache_bytes() -> anyhow::Result<()> {
    for bits in [2, 4, 8] {
        for width in [128, 256] {
            let (weights, config, embedding) = weights(bits, width);
            let mut old = MtpHead::load(
                crate::metal_context::execution_context(),
                &mut weights.clone(),
                &config,
            )?
            .unwrap();
            let mut new = MtpHead::load(
                crate::metal_context::execution_context(),
                &mut weights.clone(),
                &config,
            )?
            .unwrap();
            for (step, rows) in [1, 3, 7, 16, 256, 1, 2].into_iter().enumerate() {
                let tokens = Array::from_slice(
                    &(0..rows).map(|i| (i % 32) as u32).collect::<Vec<_>>(),
                    &[1, rows],
                );
                let hidden = Array::from_slice(
                    &(0..rows * 256)
                        .map(|i| {
                            half::bf16::from_f32(((i * 11 + step as i32) % 127 - 63) as f32 / 128.)
                        })
                        .collect::<Vec<_>>(),
                    &[1, rows, 256],
                );
                let offset = old.cache_offset();
                let want = old.reference_forward(&tokens, &hidden, &embedding, offset)?;
                let got = new.forward(&tokens, &hidden, &embedding, offset)?;
                same(&want.0, &got.0);
                same(&want.1, &got.1);
                let (
                    crate::core::cache::LayerCache::Full(a),
                    crate::core::cache::LayerCache::Full(b),
                ) = (&old.caches[0], &new.caches[0])
                else {
                    unreachable!()
                };
                assert_eq!((a.offset, a.capacity), (b.offset, b.capacity));
                same(a.keys.as_ref().unwrap(), b.keys.as_ref().unwrap());
                same(a.values.as_ref().unwrap(), b.values.as_ref().unwrap());
                if step == 4 {
                    old.trim_caches(3);
                    new.trim_caches(3);
                }
            }
            old.reset_caches();
            new.reset_caches();
            assert_eq!(old.cache_offset(), new.cache_offset());
        }
    }
    Ok(())
}
