//! Laya forward on the host (CPU), f32 — the portable no-Metal path.
//!
//! Mirrors `forward.rs` (the Metal path) op for op. Activations are row-major
//! f32 `Vec`s; matmuls are parallelised over rows with rayon. It is much slower
//! than the Metal path but needs no GPU.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use super::{Laya, LayaConfig};

/// A row-major f32 weight/activation with its shape.
type Mat = (Vec<f32>, Vec<usize>);

impl Laya {
    /// Materialize an original-name tensor as row-major f32.
    fn wf32(&self, name: &str) -> Result<Mat> {
        let (bytes, dtype, shape) = self
            .weights()
            .tensor_bytes(name)
            .with_context(|| format!("laya: missing weight {name}"))?;
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let data: Vec<f32> = match dtype {
            lisa_mlx::Dtype::Float32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            lisa_mlx::Dtype::Float16 => bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                .collect(),
            lisa_mlx::Dtype::Bfloat16 => bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                .collect(),
            other => bail!("laya cpu: unsupported dtype {other:?} for {name}"),
        };
        Ok((data, shape))
    }

    /// `x [rows, in] @ wᵀ` with `w [out, in]`; optional bias `[out]`.
    fn cpu_linear(&self, x: &[f32], rows: usize, w: &Mat, b: Option<&Mat>) -> Result<Vec<f32>> {
        let (wf, ws) = w;
        let in_dim = ws[1];
        let out_dim = ws[0];
        let bias = b.as_ref().map(|(v, _)| v.as_slice());
        let mut y = vec![0f32; rows * out_dim];
        y.par_chunks_mut(out_dim).enumerate().for_each(|(r, o)| {
            let xr = &x[r * in_dim..(r + 1) * in_dim];
            for (oc, ov) in o.iter_mut().enumerate() {
                let wr = &wf[oc * in_dim..(oc + 1) * in_dim];
                let mut acc = 0f32;
                for k in 0..in_dim {
                    acc += xr[k] * wr[k];
                }
                *ov = acc + bias.map_or(0.0, |bb| bb[oc]);
            }
        });
        Ok(y)
    }

    /// LayerNorm over the last axis, rows independent.
    fn cpu_layer_norm(x: &[f32], rows: usize, cols: usize, w: &Mat, b: Option<&Mat>, eps: f32) -> Vec<f32> {
        let (wf, _) = w;
        let bias = b.as_ref().map(|(v, _)| v.as_slice());
        let mut y = vec![0f32; rows * cols];
        y.par_chunks_mut(cols).enumerate().for_each(|(r, o)| {
            let xr = &x[r * cols..(r + 1) * cols];
            let mean = xr.iter().sum::<f32>() / cols as f32;
            let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..cols {
                let n = (xr[i] - mean) * inv;
                let s = n * wf[i] + bias.map_or(0.0, |bb| bb[i]);
                o[i] = s;
            }
        });
        y
    }

    fn cpu_gelu(x: &[f32]) -> Vec<f32> {
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        x.iter()
            .map(|&v| 0.5 * v * (1.0 + libm::erff(v * INV_SQRT2)))
            .collect()
    }

    /// GPT-NeoX RoPE on `[heads, S, d]` (pairs `(i, i+d/2)`).
    fn cpu_rope_neox(x: &mut [f32], heads: usize, s: usize, d: usize, base: f32) {
        let half = d / 2;
        for h in 0..heads {
            for pos in 0..s {
                let row = (h * s + pos) * d;
                for i in 0..half {
                    let inv = base.powf(-2.0 * i as f32 / d as f32);
                    let a = pos as f32 * inv;
                    let (c, si) = (a.cos(), a.sin());
                    let x0 = x[row + i];
                    let x1 = x[row + i + half];
                    x[row + i] = x0 * c - x1 * si;
                    x[row + i + half] = x1 * c + x0 * si;
                }
            }
        }
    }

    /// Additive key mask `[S, S]`: 0 visible, else large negative.
    fn cpu_key_mask(s: usize, global: bool, window: usize) -> Vec<f32> {
        let mut m = vec![0f32; s * s];
        for q in 0..s {
            for k in 0..s {
                let visible = global || (q as isize - k as isize).unsigned_abs() <= window / 2;
                m[q * s + k] = if visible { 0.0 } else { -1e9 };
            }
        }
        m
    }

    /// Attention over `[heads, S, d]` q/k/v with an additive `[S, S]` mask.
    fn cpu_attn(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
        s: usize,
        d: usize,
        mask: &[f32],
    ) -> Vec<f32> {
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0f32; heads * s * d];
        out.par_chunks_mut(s * d).enumerate().for_each(|(h, o)| {
            let qh = &q[h * s * d..(h + 1) * s * d];
            let kh = &k[h * s * d..(h + 1) * s * d];
            let vh = &v[h * s * d..(h + 1) * s * d];
            let mut scores = vec![0f32; s];
            for qi in 0..s {
                let qr = &qh[qi * d..(qi + 1) * d];
                for ki in 0..s {
                    let kr = &kh[ki * d..(ki + 1) * d];
                    let mut acc = 0f32;
                    for di in 0..d {
                        acc += qr[di] * kr[di];
                    }
                    scores[ki] = acc * scale + mask[qi * s + ki];
                }
                let p = super::prompt::softmax(&scores);
                let orow = &mut o[qi * d..(qi + 1) * d];
                for ki in 0..s {
                    let pv = p[ki];
                    let vr = &vh[ki * d..(ki + 1) * d];
                    for di in 0..d {
                        orow[di] += pv * vr[di];
                    }
                }
            }
        });
        out
    }

    /// Slice `[S, heads*d]` into `[heads, S, d]`.
    fn split_heads(x: &[f32], s: usize, heads: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0f32; heads * s * d];
        for si in 0..s {
            for h in 0..heads {
                for di in 0..d {
                    out[(h * s + si) * d + di] = x[si * heads * d + h * d + di];
                }
            }
        }
        out
    }

    /// Merge `[heads, S, d]` back to `[S, heads*d]`.
    fn merge_heads(x: &[f32], s: usize, heads: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0f32; s * heads * d];
        for si in 0..s {
            for h in 0..heads {
                for di in 0..d {
                    out[si * heads * d + h * d + di] = x[(h * s + si) * d + di];
                }
            }
        }
        out
    }

    fn cpu_encoder_attn(&self, x: &[f32], prefix: &str, base: f32, mask: &[f32], cfg: &LayaConfig) -> Result<Vec<f32>> {
        let s = x.len() / cfg.hidden_size;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let qkv = self.cpu_linear(x, s, &self.wf32(&format!("{prefix}.attn.Wqkv.weight"))?, None)?;
        // [S, 3, h, d] -> per-part [S, h, d] -> [h, S, d]
        let mut parts = vec![vec![0f32; s * h * d]; 3];
        for si in 0..s {
            for p in 0..3 {
                for hi in 0..h {
                    for di in 0..d {
                        parts[p][si * h * d + hi * d + di] = qkv[si * 3 * h * d + p * h * d + hi * d + di];
                    }
                }
            }
        }
        let mut q = Self::split_heads(&parts[0], s, h, d);
        let mut k = Self::split_heads(&parts[1], s, h, d);
        let v = Self::split_heads(&parts[2], s, h, d);
        Self::cpu_rope_neox(&mut q, h, s, d, base);
        Self::cpu_rope_neox(&mut k, h, s, d, base);
        let out = Self::cpu_attn(&q, &k, &v, h, s, d, mask);
        let merged = Self::merge_heads(&out, s, h, d);
        self.cpu_linear(&merged, s, &self.wf32(&format!("{prefix}.attn.Wo.weight"))?, None)
    }

    fn cpu_encoder_mlp(&self, x: &[f32], prefix: &str, cfg: &LayaConfig) -> Result<Vec<f32>> {
        let s = x.len() / cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let wi = self.cpu_linear(x, s, &self.wf32(&format!("{prefix}.mlp.Wi.weight"))?, None)?;
        let mut value = vec![0f32; s * inter];
        let mut gate = vec![0f32; s * inter];
        for r in 0..s {
            value[r * inter..(r + 1) * inter].copy_from_slice(&wi[r * 2 * inter..r * 2 * inter + inter]);
            gate[r * inter..(r + 1) * inter].copy_from_slice(&wi[r * 2 * inter + inter..(r + 1) * 2 * inter]);
        }
        let g = Self::cpu_gelu(&value);
        let gated: Vec<f32> = g.iter().zip(gate.iter()).map(|(a, b)| a * b).collect();
        self.cpu_linear(&gated, s, &self.wf32(&format!("{prefix}.mlp.Wo.weight"))?, None)
    }

    /// The ModernBERT encoder on the host; returns final-norm hidden `[S, H]`.
    pub fn encode_cpu(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let cfg = self.config();
        let s = ids.len();
        let h = cfg.hidden_size;

        // Embedding lookup + embeddings norm.
        let (emb, es) = self.wf32("encoder.embeddings.tok_embeddings.weight")?;
        let mut x = vec![0f32; s * h];
        for (i, &id) in ids.iter().enumerate() {
            x[i * h..(i + 1) * h].copy_from_slice(&emb[id as usize * es[1]..(id as usize + 1) * es[1]]);
        }
        let emb_norm = self.wf32("encoder.embeddings.norm.weight")?;
        x = Self::cpu_layer_norm(&x, s, h, &emb_norm, None, cfg.norm_eps);

        let full = Self::cpu_key_mask(s, true, cfg.local_attention);
        let local = Self::cpu_key_mask(s, false, cfg.local_attention);

        for i in 0..cfg.num_layers {
            let prefix = format!("encoder.layers.{i}");
            let global = cfg.layer_global[i];
            let base = if global { cfg.rope_theta_global } else { cfg.rope_theta_local };
            let mask = if global { &full } else { &local };
            let normed = if i == 0 {
                x.clone()
            } else {
                Self::cpu_layer_norm(&x, s, h, &self.wf32(&format!("{prefix}.attn_norm.weight"))?, None, cfg.norm_eps)
            };
            let att = self.cpu_encoder_attn(&normed, &prefix, base, mask, cfg)?;
            for (a, b) in x.iter_mut().zip(att.iter()) {
                *a += *b;
            }
            let normed = Self::cpu_layer_norm(&x, s, h, &self.wf32(&format!("{prefix}.mlp_norm.weight"))?, None, cfg.norm_eps);
            let mlp = self.cpu_encoder_mlp(&normed, &prefix, cfg)?;
            for (a, b) in x.iter_mut().zip(mlp.iter()) {
                *a += *b;
            }
        }
        let fin = self.wf32("encoder.final_norm.weight")?;
        Ok(Self::cpu_layer_norm(&x, s, h, &fin, None, cfg.norm_eps))
    }

    fn head_layer_cpu(&self, x: &[f32], i: usize, mask: &[f32], cfg: &LayaConfig) -> Result<Vec<f32>> {
        let p = format!("head.layers.{i}");
        let s = x.len() / cfg.hidden_size;
        let dims = cfg.hidden_size;
        let heads = (dims / 64).max(1);
        let d = dims / heads;

        let n1w = self.wf32(&format!("{p}.norm1.weight"))?;
        let n1b = self.wf32(&format!("{p}.norm1.bias"))?;
        let normed = Self::cpu_layer_norm(x, s, dims, &n1w, Some(&n1b), cfg.norm_eps);
        let ipw = self.wf32(&format!("{p}.self_attn.in_proj_weight"))?;
        let ipb = self.wf32(&format!("{p}.self_attn.in_proj_bias"))?;
        let qkv = self.cpu_linear(&normed, s, &ipw, Some(&ipb))?;
        let mut parts = vec![vec![0f32; s * heads * d]; 3];
        for si in 0..s {
            for pi in 0..3 {
                for hi in 0..heads {
                    for di in 0..d {
                        parts[pi][si * heads * d + hi * d + di] = qkv[si * 3 * heads * d + pi * heads * d + hi * d + di];
                    }
                }
            }
        }
        let q = Self::split_heads(&parts[0], s, heads, d);
        let k = Self::split_heads(&parts[1], s, heads, d);
        let v = Self::split_heads(&parts[2], s, heads, d);
        let out = Self::cpu_attn(&q, &k, &v, heads, s, d, mask);
        let merged = Self::merge_heads(&out, s, heads, d);
        let opw = self.wf32(&format!("{p}.self_attn.out_proj.weight"))?;
        let opb = self.wf32(&format!("{p}.self_attn.out_proj.bias"))?;
        let att = self.cpu_linear(&merged, s, &opw, Some(&opb))?;
        let mut x2: Vec<f32> = x.iter().zip(att.iter()).map(|(a, b)| a + b).collect();

        let n2w = self.wf32(&format!("{p}.norm2.weight"))?;
        let n2b = self.wf32(&format!("{p}.norm2.bias"))?;
        let normed = Self::cpu_layer_norm(&x2, s, dims, &n2w, Some(&n2b), cfg.norm_eps);
        let l1w = self.wf32(&format!("{p}.linear1.weight"))?;
        let l1b = self.wf32(&format!("{p}.linear1.bias"))?;
        let hid = self.cpu_linear(&normed, s, &l1w, Some(&l1b))?;
        let hid: Vec<f32> = hid.iter().map(|&v| v.max(0.0)).collect();
        let l2w = self.wf32(&format!("{p}.linear2.weight"))?;
        let l2b = self.wf32(&format!("{p}.linear2.bias"))?;
        let mlp = self.cpu_linear(&hid, s, &l2w, Some(&l2b))?;
        for (a, b) in x2.iter_mut().zip(mlp.iter()) {
            *a += *b;
        }
        Ok(x2)
    }

    /// The Laya decision forward on the host.
    pub fn forward_cpu(
        &self,
        ids: &[u32],
        qtype: i32,
        marker_pos: &[i32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = self.config();
        let s = ids.len();
        let dims = cfg.hidden_size;
        let mut h = self.encode_cpu(ids)?;

        // h += type_emb[qtype]
        let (te, ts) = self.wf32("type_emb.weight")?;
        let row = &te[qtype as usize * ts[1]..(qtype as usize + 1) * ts[1]];
        for si in 0..s {
            for d in 0..dims {
                h[si * dims + d] += row[d];
            }
        }

        let mask = Self::cpu_key_mask(s, true, cfg.local_attention);
        for i in 0..cfg.head_layers {
            h = self.head_layer_cpu(&h, i, &mask, cfg)?;
        }

        // scorer on the marker rows
        let mut markers = vec![0f32; marker_pos.len() * dims];
        for (mi, &pos) in marker_pos.iter().enumerate() {
            markers[mi * dims..(mi + 1) * dims].copy_from_slice(&h[pos as usize * dims..(pos as usize + 1) * dims]);
        }
        let m = marker_pos.len();
        let s0w = self.wf32("scorer.0.weight")?;
        let s0b = self.wf32("scorer.0.bias")?;
        let m = Self::cpu_layer_norm(&markers, m, dims, &s0w, Some(&s0b), cfg.norm_eps);
        let s1w = self.wf32("scorer.1.weight")?;
        let s1b = self.wf32("scorer.1.bias")?;
        let m = self.cpu_linear(&m, marker_pos.len(), &s1w, Some(&s1b))?;
        let m = Self::cpu_gelu(&m);
        let s3w = self.wf32("scorer.3.weight")?;
        let s3b = self.wf32("scorer.3.bias")?;
        let logits = self.cpu_linear(&m, marker_pos.len(), &s3w, Some(&s3b))?;

        // features from the unscaled softmax
        let p = super::prompt::softmax(&logits);
        let kf = marker_pos.len().max(2) as f32;
        let entropy: f32 = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f32>() / kf.ln();
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top1 = *sorted.last().unwrap();
        let top2 = *sorted.get(sorted.len().saturating_sub(2)).unwrap();

        // pooled = [h[0], top1, top1-top2, entropy, k/255]
        let mut pooled = vec![0f32; dims + 4];
        pooled[..dims].copy_from_slice(&h[..dims]);
        pooled[dims] = top1;
        pooled[dims + 1] = top1 - top2;
        pooled[dims + 2] = entropy;
        pooled[dims + 3] = kf / 255.0;

        let a0w = self.wf32("act_head.0.weight")?;
        let a0b = self.wf32("act_head.0.bias")?;
        let a = self.cpu_linear(&pooled, 1, &a0w, Some(&a0b))?;
        let a = Self::cpu_gelu(&a);
        let a2w = self.wf32("act_head.2.weight")?;
        let a2b = self.wf32("act_head.2.bias")?;
        let action = self.cpu_linear(&a, 1, &a2w, Some(&a2b))?;

        Ok((logits, action))
    }
}