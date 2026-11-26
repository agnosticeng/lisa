#![cfg(test)]
//! Frozen operation chains for strict packed-weight plan migration checks.
use super::weights::Config;
use crate::core::cache::{FullAttentionCache, GdnCache, LayerCache};
use anyhow::Result;
use lisa_metal::ops::indexing::IndexOp;
use lisa_metal::{Array, Stream, fast, ops};
struct Projection {
    bytes: Array,
    kind: lisa_backend::OrderedBlockEncoding,
}
impl Projection {
    pub fn forward(&self, x: &Array) -> lisa_metal::error::Result<Array> {
        let shape = x.shape();
        let k = x.dim(-1);
        let rows = shape[..shape.len() - 1].iter().product::<i32>();
        let flat = x.reshape(&[rows, k])?.contiguous()?;
        let y = lisa_metal::kernels::blockwise::matvec(
            self.kind,
            &self.bytes,
            &flat,
            &Stream::thread_local_or_default(),
        )?;
        y.reshape(&[&shape[..shape.len() - 1], &[y.dim(1)]].concat())
    }
}
struct Attention {
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    qn: Array,
    kn: Array,
}
struct Gdn {
    qkv: Projection,
    z: Projection,
    alpha: Projection,
    beta: Projection,
    out: Projection,
    conv: Array,
    a: Array,
    dt: Array,
    norm: Array,
}

#[allow(clippy::large_enum_variant, reason = "one value per layer")]
enum Mixer {
    Full(Attention),
    Linear(Gdn),
}
pub(super) struct Layer {
    mixer: Mixer,
    anorm: Array,
    fnorm: Array,
    gate: Projection,
    up: Projection,
    down: Projection,
}
fn norm(x: &Array, w: &Array, eps: f32) -> lisa_metal::error::Result<Array> {
    fast::rms_norm(x, Some(w), eps)
}
impl Layer {
    pub(super) fn load(w: &super::weights::Bound, i: usize) -> Result<Self> {
        let proj = |n: &str| -> Result<Projection> {
            let name = format!("blk.{i}.{n}.weight");
            Ok(Projection {
                bytes: w.arrays[&name].clone(),
                kind: w.projections[&name].encoding,
            })
        };
        let p = |n: &str| -> Result<Array> { Ok(w.arrays[&format!("blk.{i}.{n}")].clone()) };
        let mixer = if i == 64 || (i + 1).is_multiple_of(4) {
            Mixer::Full(Attention {
                q: proj("attn_q")?,
                k: proj("attn_k")?,
                v: proj("attn_v")?,
                out: proj("attn_output")?,
                qn: p("attn_q_norm.weight")?,
                kn: p("attn_k_norm.weight")?,
            })
        } else {
            Mixer::Linear(Gdn {
                qkv: proj("attn_qkv")?,
                z: proj("attn_gate")?,
                alpha: proj("ssm_alpha")?,
                beta: proj("ssm_beta")?,
                out: proj("ssm_out")?,
                conv: p("ssm_conv1d.weight")?.reshape(&[10240, 4, 1])?,
                a: p("ssm_a")?,
                dt: p("ssm_dt.bias")?,
                norm: p("ssm_norm.weight")?,
            })
        };
        Ok(Self {
            mixer,
            anorm: p("attn_norm.weight")?,
            fnorm: p("post_attention_norm.weight")?,
            gate: proj("ffn_gate")?,
            up: proj("ffn_up")?,
            down: proj("ffn_down")?,
        })
    }
    pub(super) fn forward(
        &self,
        x: &Array,
        cache: &mut LayerCache,
        capture: bool,
        c: &Config,
    ) -> Result<Array> {
        let n = norm(x, &self.anorm, c.eps)?;
        let y = match (&self.mixer, cache) {
            (Mixer::Full(a), LayerCache::Full(cache)) => a.forward(&n, cache, c)?,
            (Mixer::Linear(g), LayerCache::Linear(cache)) => g.forward(&n, cache, capture, c)?,
            _ => anyhow::bail!("Qwen GGUF cache kind mismatch"),
        };
        let residual = x.add(&y)?;
        let n = norm(&residual, &self.fnorm, c.eps)?;
        let ff = lisa_metal::kernels::blockwise::silu_mul(
            &self.gate.forward(&n)?,
            &self.up.forward(&n)?,
        )?;
        Ok(residual.add(&self.down.forward(&ff)?)?)
    }
}
impl Attention {
    fn forward(&self, x: &Array, cache: &mut FullAttentionCache, c: &Config) -> Result<Array> {
        let t = x.dim(1);
        let qg = self.q.forward(x)?.reshape(&[1, t, 24, 512])?;
        let q = norm(
            &qg.index((.., .., .., 0..256)).contiguous()?,
            &self.qn,
            c.eps,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .contiguous()?;
        let gate = qg
            .index((.., .., .., 256..512))
            .contiguous()?
            .reshape(&[1, t, 6144])?;
        let k = norm(
            &self.k.forward(x)?.reshape(&[1, t, 4, 256])?,
            &self.kn,
            c.eps,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape(&[1, t, 4, 256])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let q = fast::rope(&q, 64, false, Some(c.theta), 1.0, cache.offset as i32, None)?;
        let k = fast::rope(&k, 64, false, Some(c.theta), 1.0, cache.offset as i32, None)?;
        let (k, v) = cache.update(&k, &v)?;
        // One F32 reduction policy for serial, prefill and verification. Each
        // query is independent; causal masking preserves its live KV window.
        let y = Array::new(lisa_metal::jit::sdpa_vector(
            q.t.device(),
            &q.t,
            &k.t,
            &v.t,
            1.0 / 16.0,
            t > 1,
            None,
        )?);
        let y = y
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[1, t, 6144])?
            .multiply(&gate.sigmoid()?)?;
        Ok(self.out.forward(&y)?)
    }
}
impl Gdn {
    fn forward(&self, x: &Array, cache: &mut GdnCache, capture: bool, c: &Config) -> Result<Array> {
        let t = x.dim(1);
        let qkv = self.qkv.forward(x)?;
        let previous = match &cache.conv {
            Some(v) => v.clone(),
            None => ops::zeros::<f32>(&[1, 3, 10240])?,
        };
        let conv_input = Array::concatenate(&[&previous, &qkv], 1)?;
        cache.conv = Some(conv_input.index((.., t..t + 3, ..)).contiguous()?);
        let conv = lisa_metal::kernels::blockwise::conv_silu(&conv_input, &self.conv)?;
        let l2 = |a: Array| -> lisa_metal::error::Result<Array> {
            fast::rms_norm(
                &a.reshape(&[t, 16, 128])?.contiguous()?,
                None,
                c.eps / 128.0,
            )?
            .multiply(Array::from_f32(1.0 / 128f32.sqrt()))
        };
        let q = l2(conv.index((.., .., 0..2048)).contiguous()?)?;
        let k = l2(conv.index((.., .., 2048..4096)).contiguous()?)?;
        let v = conv
            .index((.., .., 4096..10240))
            .contiguous()?
            .reshape(&[t, 48, 128])?;
        let (g, b) = lisa_metal::kernels::blockwise::gates(
            &self.alpha.forward(x)?,
            &self.beta.forward(x)?,
            &self.dt,
            &self.a,
        )?;
        let mut state = match &cache.ssm {
            Some(s) => s.clone(),
            None => ops::zeros::<f32>(&[48, 128, 128])?,
        };
        let stream = Stream::thread_local_or_default();
        let y = if capture {
            let mut ys = Vec::new();
            let mut states = Vec::new();
            for i in 0..t {
                let (y, next) = lisa_metal::kernels::blockwise::gdn(
                    &q.index((i..i + 1, .., ..)).contiguous()?,
                    &k.index((i..i + 1, .., ..)).contiguous()?,
                    &v.index((i..i + 1, .., ..)).contiguous()?,
                    &g.index((i..i + 1, ..)).contiguous()?,
                    &b.index((i..i + 1, ..)).contiguous()?,
                    &state,
                    &stream,
                )?;
                ys.push(y);
                states.push(next.reshape(&[1, 48, 128, 128])?);
                state = next;
            }
            cache.capture_ssm = Some(Array::concatenate(&states.iter().collect::<Vec<_>>(), 0)?);
            cache.capture_conv_input = Some(conv_input);
            Array::concatenate(&ys.iter().collect::<Vec<_>>(), 0)?
        } else {
            cache.capture_ssm = None;
            cache.capture_conv_input = None;
            let (y, next) =
                lisa_metal::kernels::blockwise::gdn(&q, &k, &v, &g, &b, &state, &stream)?;
            state = next;
            y
        };
        cache.ssm = Some(state);
        let y = lisa_metal::kernels::blockwise::mul_silu(
            &norm(&y, &self.norm, c.eps)?,
            &self.z.forward(x)?.reshape(&[t, 48, 128])?,
        )?;
        Ok(self.out.forward(&y.reshape(&[1, t, 6144])?)?)
    }
}
