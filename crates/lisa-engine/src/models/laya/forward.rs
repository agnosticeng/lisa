//! Laya forward: ModernBERT encoder + decision head, composed from the generic
//! ops (matmul, softmax, reductions, rope). `B = 1`.
//!
//! The reference is `laya_mlx/model.py` (which loads the same original
//! checkpoint): pre-norm ModernBERT with standard RoPE, GeGLU MLP, boolean
//! key masks; then a 2-layer decision head (ReLU MLP), an option-marker scorer,
//! and an act head. Weights are computed in f32 (cast up from the checkpoint's
//! fp16) for a first correct implementation; fp16 parity is a follow-up.

use anyhow::{Context, Result};
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::Array;

use super::{Laya, LayaConfig};
use crate::core::loader::array_from_bytes;

/// First 5 channels of token 0, as f32 — for the `LISA_LAYATRACE` diagnostic.
fn first5(x: &Array) -> Vec<f32> {
    let Ok(row) = x.index((0, .., ..)).as_dtype(lisa_mlx::Dtype::Float32) else {
        return Vec::new();
    };
    let Ok(flat) = row.reshape(&[-1]) else {
        return Vec::new();
    };
    flat.as_slice::<f32>().iter().take(5).copied().collect()
}

impl Laya {
    /// Materialize an original-name tensor as an f32 `Array`.
    fn w(&self, name: &str) -> Result<Array> {
        let (bytes, dtype, shape) = self
            .weights()
            .tensor_bytes(name)
            .with_context(|| format!("laya: missing weight {name}"))?;
        let a = array_from_bytes(bytes, dtype, shape).map_err(|e| anyhow::anyhow!("{e}"))?;
        a.as_dtype(lisa_mlx::Dtype::Float32)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// `x @ wᵀ (+ b)`. `w` is `[out, in]`, `b` optional `[out]`.
    fn linear(&self, x: &Array, w: &str, b: Option<&str>) -> Result<Array> {
        let wt = self.w(w)?.transpose_axes(&[1, 0])?;
        let mut y = x.matmul(&wt).map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some(b) = b {
            y = y.add(&self.w(b)?).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(y)
    }

    /// LayerNorm over the last axis (optionally with weight/bias).
    fn layer_norm(x: &Array, w: Option<Array>, b: Option<Array>, eps: f32) -> Result<Array> {
        let mean = x.mean_axis(-1, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let xc = x.subtract(&mean).map_err(|e| anyhow::anyhow!("{e}"))?;
        let var = xc
            .multiply(&xc)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .mean_axis(-1, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let inv = var
            .add(&Array::from_f32(eps))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .rsqrt()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut y = xc.multiply(&inv).map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some(w) = w {
            y = y.multiply(&w).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        if let Some(b) = b {
            y = y.add(&b).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(y)
    }

    /// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    fn gelu(x: &Array) -> Result<Array> {
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let e = x
            .multiply(Array::from_f32(INV_SQRT2))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .unary("Erf")
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let half = x.multiply(Array::from_f32(0.5)).map_err(|e| anyhow::anyhow!("{e}"))?;
        half.multiply(&e.add(Array::from_f32(1.0)).map_err(|e| anyhow::anyhow!("{e}"))?)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// GPT-NeoX RoPE (MLX `traditional=False`): rotate pairs `(i, i+d/2)` with
    /// `theta_i = pos * base^(-2i/d)`. `x` is `[heads, S, d]`.
    fn rope_neox(x: &Array, base: f32) -> Result<Array> {
        let d = x.dim(-1) as usize;
        let s = x.dim(-2) as usize;
        let half = d / 2;
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for pos in 0..s {
            for i in 0..half {
                let inv = base.powf(-2.0 * i as f32 / d as f32);
                let a = pos as f32 * inv;
                cos[pos * half + i] = a.cos();
                sin[pos * half + i] = a.sin();
            }
        }
        let shape = [1, s as i32, half as i32];
        let cos = Array::from_slice(&cos, &shape);
        let sin = Array::from_slice(&sin, &shape);

        let halves = x.split_equal(2, -1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (a, b) = (&halves[0], &halves[1]);
        let a_cos = a.multiply(&cos).map_err(|e| anyhow::anyhow!("{e}"))?;
        let b_sin = b.multiply(&sin).map_err(|e| anyhow::anyhow!("{e}"))?;
        let b_cos = b.multiply(&cos).map_err(|e| anyhow::anyhow!("{e}"))?;
        let a_sin = a.multiply(&sin).map_err(|e| anyhow::anyhow!("{e}"))?;
        let out_a = a_cos.subtract(&b_sin).map_err(|e| anyhow::anyhow!("{e}"))?;
        let out_b = b_cos.add(&a_sin).map_err(|e| anyhow::anyhow!("{e}"))?;
        Array::concatenate(&[&out_a, &out_b], -1).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn relu(x: &Array) -> Result<Array> {
        x.maximum(Array::from_f32(0.0)).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Additive attention mask `[1,1,S,S]`: 0 where a key is visible, else a
    /// large negative. `kind`: full (all keys) or sliding (`|i-j| <= window/2`).
    fn key_mask(s: usize, global: bool, window: usize, valid: &[bool]) -> Array {
        let mut m = vec![0f32; s * s];
        for q in 0..s {
            for k in 0..s {
                let visible = if global {
                    valid.get(k).copied().unwrap_or(true)
                } else {
                    let near = (q as isize - k as isize).unsigned_abs() <= window / 2;
                    (near || !valid.get(q).copied().unwrap_or(true))
                        && valid.get(k).copied().unwrap_or(true)
                };
                m[q * s + k] = if visible { 0.0 } else { -1e9 };
            }
        }
        Array::from_slice(&m, &[s as i32, s as i32])
    }

    /// ModernBERT attention block. `x` is `[S, H]`.
    fn encoder_attn(&self, x: &Array, prefix: &str, base: f32, mask: &Array, cfg: &LayaConfig) -> Result<Array> {
        let s = x.dim(0);
        let h = cfg.num_heads as i32;
        let d = cfg.head_dim as i32;
        let qkv = self.linear(x, &format!("{prefix}.attn.Wqkv.weight"), None)?;
        let qkv = qkv.reshape(&[s, 3, h, d]).map_err(|e| anyhow::anyhow!("{e}"))?;
        let parts = qkv.split_equal(3, 1).map_err(|e| anyhow::anyhow!("{e}"))?;
        // split_equal keeps the split axis (size 1); drop it, then [S,h,d] -> [1,h,S,d]
        let to_bhsd = |a: &Array| -> Result<Array> {
            a.reshape(&[s, h, d])?
                .transpose_axes(&[1, 0, 2])?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("{e}"))
        };
        let q = to_bhsd(&parts[0])?;
        let k = to_bhsd(&parts[1])?;
        let v = to_bhsd(&parts[2])?;
        let (q, k) = (Self::rope_neox(&q, base)?, Self::rope_neox(&k, base)?);
        let scale = 1.0 / (d as f32).sqrt();
        let kt = k.transpose_axes(&[0, 2, 1]).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut scores = q.matmul(&kt).map_err(|e| anyhow::anyhow!("{e}"))?;
        scores = scores.multiply(Array::from_f32(scale)).map_err(|e| anyhow::anyhow!("{e}"))?;
        scores = scores.add(mask).map_err(|e| anyhow::anyhow!("{e}"))?;
        let probs = scores.softmax_axis(-1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let out = probs.matmul(&v).map_err(|e| anyhow::anyhow!("{e}"))?;
        // [h, S, d] -> [S, h*d]
        let out = out
            .transpose_axes(&[1, 0, 2])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .reshape(&[s, h * d])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.linear(&out, &format!("{prefix}.attn.Wo.weight"), None)
    }

    /// ModernBERT MLP: `Wo(gelu(Wi_value) * Wi_gate)`.
    fn encoder_mlp(&self, x: &Array, prefix: &str, cfg: &LayaConfig) -> Result<Array> {
        let inter = cfg.intermediate_size as i32;
        let y = self.linear(x, &format!("{prefix}.mlp.Wi.weight"), None)?;
        let parts = y.split_equal(2, -1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let value = Self::gelu(&parts[0])?;
        let gated = value.multiply(&parts[1]).map_err(|e| anyhow::anyhow!("{e}"))?;
        debug_assert_eq!(gated.dim(-1), inter);
        self.linear(&gated, &format!("{prefix}.mlp.Wo.weight"), None)
    }

    /// The ModernBERT encoder; returns the `final_norm` hidden states `[S, H]`.
    pub fn encode_metal(&self, ids: &[u32]) -> Result<Array> {
        let cfg = self.config();
        let s = ids.len();
        let valid = vec![true; s];

        let ids_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let ids_arr = Array::from_slice(&ids_i32, &[s as i32]);
        let emb_w = self.w("encoder.embeddings.tok_embeddings.weight")?;
        let mut x = emb_w
            .take_axis(&ids_arr, 0)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        x = Self::layer_norm(
            &x,
            Some(self.w("encoder.embeddings.norm.weight")?),
            None,
            cfg.norm_eps,
        )?;

        let trace = std::env::var("LISA_LAYATRACE").is_ok();
        if trace { eprintln!("[laya] emb   {:?}", first5(&x)); }
        let full_mask = Self::key_mask(s, true, cfg.local_attention, &valid);
        let local_mask = Self::key_mask(s, false, cfg.local_attention, &valid);

        for i in 0..cfg.num_layers {
            let prefix = format!("encoder.layers.{i}");
            let global = cfg.layer_global[i];
            let base = if global { cfg.rope_theta_global } else { cfg.rope_theta_local };
            let mask = if global { &full_mask } else { &local_mask };

            // Pre-attention norm (layer 0 reuses the embeddings norm, applied
            // above; its attn_norm is Identity).
            let normed = if i == 0 {
                x.clone()
            } else {
                Self::layer_norm(
                    &x,
                    Some(self.w(&format!("{prefix}.attn_norm.weight"))?),
                    None,
                    cfg.norm_eps,
                )?
            };
            let att = self.encoder_attn(&normed, &prefix, base, mask, cfg)?;
            x = x.add(&att).map_err(|e| anyhow::anyhow!("{e}"))?;
            if trace && i == 0 { eprintln!("[laya] L0att  {:?}", first5(&x)); }

            let normed = Self::layer_norm(
                &x,
                Some(self.w(&format!("{prefix}.mlp_norm.weight"))?),
                None,
                cfg.norm_eps,
            )?;
            let mlp = self.encoder_mlp(&normed, &prefix, cfg)?;
            x = x.add(&mlp).map_err(|e| anyhow::anyhow!("{e}"))?;
            if trace && (i < 3 || i + 1 == cfg.num_layers) {
                eprintln!("[laya] L{i:<2}    {:?}", first5(&x));
            }
        }
        Self::layer_norm(
            &x,
            Some(self.w("encoder.final_norm.weight")?),
            None,
            cfg.norm_eps,
        )
    }

    /// One decision-head transformer layer (pre-norm, ReLU MLP, biases).
    fn head_layer(&self, x: &Array, i: usize, mask: &Array, cfg: &LayaConfig) -> Result<Array> {
        let p = format!("head.layers.{i}");
        let dims = cfg.hidden_size as i32;
        let heads = (dims / 64).max(1);
        let d = dims / heads;

        let normed = Self::layer_norm(
            &x,
            Some(self.w(&format!("{p}.norm1.weight"))?),
            Some(self.w(&format!("{p}.norm1.bias"))?),
            cfg.norm_eps,
        )?;
        let s = normed.dim(0);
        let qkv = self.linear(&normed, &format!("{p}.self_attn.in_proj_weight"), Some(&format!("{p}.self_attn.in_proj_bias")))?;
        let qkv = qkv.reshape(&[s, 3, heads, d]).map_err(|e| anyhow::anyhow!("{e}"))?;
        let parts = qkv.split_equal(3, 1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let to_bhsd = |a: &Array| -> Result<Array> {
            a.reshape(&[s, heads, d])?
                .transpose_axes(&[1, 0, 2])?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("{e}"))
        };
        let (q, k, v) = (to_bhsd(&parts[0])?, to_bhsd(&parts[1])?, to_bhsd(&parts[2])?);
        let kt = k.transpose_axes(&[0, 2, 1]).map_err(|e| anyhow::anyhow!("{e}"))?;
        let scores = q
            .matmul(&kt)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .multiply(Array::from_f32(1.0 / (d as f32).sqrt()))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .add(mask)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let probs = scores.softmax_axis(-1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let att = probs
            .matmul(&v)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .transpose_axes(&[1, 0, 2])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .reshape(&[s, dims])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let att = self.linear(&att, &format!("{p}.self_attn.out_proj.weight"), Some(&format!("{p}.self_attn.out_proj.bias")))?;
        let mut x = x.add(&att).map_err(|e| anyhow::anyhow!("{e}"))?;

        let normed = Self::layer_norm(
            &x,
            Some(self.w(&format!("{p}.norm2.weight"))?),
            Some(self.w(&format!("{p}.norm2.bias"))?),
            cfg.norm_eps,
        )?;
        let hid = self.linear(&normed, &format!("{p}.linear1.weight"), Some(&format!("{p}.linear1.bias")))?;
        let hid = Self::relu(&hid)?;
        let mlp = self.linear(&hid, &format!("{p}.linear2.weight"), Some(&format!("{p}.linear2.bias")))?;
        x = x.add(&mlp).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(x)
    }

    /// Full decision forward. Returns `(logits per marker, action)`.
    pub fn forward_metal(
        &self,
        ids: &[u32],
        qtype: i32,
        marker_pos: &[i32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = self.config();
        let s = ids.len();
        let mut h = self.encode_metal(ids)?;

        // h += type_emb[qtype]
        let type_w = self.w("type_emb.weight")?;
        let te = type_w
            .take_axis(&Array::from_slice(&[qtype], &[1]), 0)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .reshape(&[1, cfg.hidden_size as i32])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        h = h.add(&te).map_err(|e| anyhow::anyhow!("{e}"))?;

        // Decision head (full key mask; padded queries excluded later).
        let valid = vec![true; s];
        let mask = Self::key_mask(s, true, cfg.local_attention, &valid);
        for i in 0..cfg.head_layers {
            h = self.head_layer(&h, i, &mask, cfg)?;
        }

        // Gather marker rows.
        let pos = Array::from_slice(marker_pos, &[marker_pos.len() as i32]);
        let markers = h.take_axis(&pos, 0).map_err(|e| anyhow::anyhow!("{e}"))?;

        // scorer: LayerNorm, Linear, GELU, Linear -> [K]
        let m = Self::layer_norm(
            &markers,
            Some(self.w("scorer.0.weight")?),
            Some(self.w("scorer.0.bias")?),
            cfg.norm_eps,
        )?;
        let m = self.linear(&m, "scorer.1.weight", Some("scorer.1.bias"))?;
        let m = Self::gelu(&m)?;
        let logits = self
            .linear(&m, "scorer.3.weight", Some("scorer.3.bias"))?
            .reshape(&[marker_pos.len() as i32])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let logits_v: Vec<f32> = logits
            .as_dtype(lisa_mlx::Dtype::Float32)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .as_slice::<f32>()
            .to_vec();

        // probabilities over markers
        let p = logits.softmax_axis(-1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let p_v: Vec<f32> = p.as_slice::<f32>().to_vec();
        let k = marker_pos.len().max(2) as f32;
        let entropy: f32 = -p_v
            .iter()
            .map(|&x| x * x.max(1e-9).ln())
            .sum::<f32>()
            / k.ln();
        let mut sorted = p_v.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top1 = *sorted.last().unwrap();
        let top2 = *sorted.get(sorted.len().saturating_sub(2)).unwrap();
        let features = Array::from_slice(&[top1, top1 - top2, entropy, k / 255.0], &[1, 4]);

        // pooled = [h[:,0], features]; action = act_head(pooled)
        let h0 = h.index((0, ..)).reshape(&[1, cfg.hidden_size as i32]).map_err(|e| anyhow::anyhow!("{e}"))?;
        let pooled = Array::concatenate(&[&h0, &features], 1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let a = self.linear(&pooled, "act_head.0.weight", Some("act_head.0.bias"))?;
        let a = Self::gelu(&a)?;
        let action = self.linear(&a, "act_head.2.weight", Some("act_head.2.bias"))?;
        let action_v = action
            .as_dtype(lisa_mlx::Dtype::Float32)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .as_slice::<f32>()
            .to_vec();

        Ok((logits_v, action_v))
    }
}