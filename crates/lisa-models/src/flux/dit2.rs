//! FLUX.2 denoising computation; produces a portable tensor graph.
use super::symbolic::{
    self as ops, Program, Schema, Tensor as Array, TensorInfo, Trace, Weights, layer_norm, silu,
};
use anyhow::ensure;
use lisa_backend::ScalarType as Dtype;

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub inner: i32,
    pub heads: i32,
    pub hd: i32,
    pub doubles: usize,
    pub singles: usize,
}
impl Geometry {
    pub fn from_schema(schema: &Schema) -> anyhow::Result<Self> {
        let dim = |name: &str| {
            schema
                .get(name)
                .and_then(|x| x.shape.first())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing FLUX geometry {name}"))
        };
        let inner = dim("x_embedder.weight")?;
        let hd = dim("transformer_blocks.0.attn.norm_q.weight")?;
        let doubles = ops::count(schema, "transformer_blocks", "attn.to_q.weight");
        let singles = ops::count(schema, "single_transformer_blocks", "attn.to_out.weight");
        ensure!(
            hd == 128 && matches!((inner, doubles, singles), (3072, 5, 20) | (4096, 8, 24)),
            "expected a distilled FLUX.2 Klein 4B or 9B transformer"
        );
        Ok(Self {
            inner,
            heads: inner / hd,
            hd,
            doubles,
            singles,
        })
    }
}
struct Dit<'a> {
    w: Weights<'a>,
    geometry: Geometry,
}
impl std::ops::Deref for Dit<'_> {
    type Target = Geometry;
    fn deref(&self) -> &Geometry {
        &self.geometry
    }
}
pub fn program(
    schema: &Schema,
    geometry: Geometry,
    latent: TensorInfo,
    enc: TensorInfo,
    time: f32,
    h: i32,
    w: i32,
) -> anyhow::Result<Program> {
    ensure!(
        geometry.hd == 128
            && geometry.heads > 0
            && geometry.heads.checked_mul(geometry.hd) == Some(geometry.inner)
            && geometry.doubles > 0
            && geometry.doubles <= 128
            && geometry.singles > 0
            && geometry.singles <= 128,
        "invalid FLUX transformer geometry"
    );
    ensure!(
        h > 0
            && w > 0
            && h.checked_mul(w).is_some()
            && latent.shape == [1, h * w, 128]
            && enc.shape.len() == 3
            && enc.shape[0] == 1
            && enc.shape[1] > 0,
        "invalid FLUX token shapes"
    );
    let trace = Trace::new();
    let latent = trace.input(&latent.shape, latent.dtype);
    let enc = trace.input(&enc.shape, enc.dtype);
    let dit = Dit {
        w: Weights {
            trace: trace.clone(),
            schema,
        },
        geometry,
    };
    let output = dit.forward(&latent, &enc, time, h, w)?;
    trace.finish(&[output])
}
fn slice(x: &Array, start: i32, end: i32) -> Array {
    x.slice(-1, start, end).expect("validated FLUX slice")
}
fn modulate(x: &Array, m: &[Array], shift: usize, scale: usize) -> anyhow::Result<Array> {
    Ok(layer_norm(x)?
        .multiply(&m[scale].add(&x.scalar(1.0))?)?
        .add(&m[shift])?)
}
impl Dit<'_> {
    fn heads(&self, x: Array) -> anyhow::Result<Array> {
        Ok(x.reshape(&[1, x.dim(1), self.heads, self.hd])?
            .transpose_axes(&[0, 2, 1, 3])?)
    }
    fn heads_norm(&self, x: Array, key: &str) -> anyhow::Result<Array> {
        let x = self.heads(x)?.as_dtype(Dtype::F32)?.contiguous()?;
        Ok(x.rms(&self.w.get(key)?.as_dtype(Dtype::F32)?, 1e-5)?
            .as_dtype(Dtype::BF16)?)
    }
    fn attention(
        &self,
        q: Array,
        k: Array,
        v: Array,
        cos: &Array,
        sin: &Array,
    ) -> anyhow::Result<Array> {
        let rope = |x: &Array| -> anyhow::Result<Array> {
            let shape = x.shape();
            let pairs = x
                .as_dtype(Dtype::F32)?
                .reshape(&[1, self.heads, x.dim(2), 64, 2])?;
            let re = pairs
                .slice(-1, 0, 1)?
                .reshape(&[1, self.heads, x.dim(2), 64])?;
            let im = pairs
                .slice(-1, 1, 2)?
                .reshape(&[1, self.heads, x.dim(2), 64])?;
            let r = re.multiply(cos)?.subtract(&im.multiply(sin)?)?;
            let i = im.multiply(cos)?.add(&re.multiply(sin)?)?;
            Ok(
                ops::concatenate(&[&r.expand_dims(-1)?, &i.expand_dims(-1)?], -1)?
                    .reshape(shape)?
                    .as_dtype(Dtype::BF16)?,
            )
        };
        let q = rope(&q)?.as_dtype(v.dtype())?;
        let k = rope(&k)?.as_dtype(v.dtype())?;
        let out = ops::attention(&q, &k, &v, 1.0 / (self.hd as f32).sqrt(), None)?;
        Ok(out
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[1, q.dim(2), self.inner])?)
    }
    fn ff(&self, x: &Array, p: &str) -> anyhow::Result<Array> {
        let x = self.w.lin(&format!("{p}.linear_in"), x)?;
        let half = x.dim(-1) / 2;
        self.w.lin(
            &format!("{p}.linear_out"),
            &silu(&slice(&x, 0, half))?.multiply(&slice(&x, half, 2 * half))?,
        )
    }
    pub fn forward(
        &self,
        latent: &Array,
        enc: &Array,
        time: f32,
        h: i32,
        w: i32,
    ) -> anyhow::Result<Array> {
        ensure!(
            enc.dim(-1) == if self.inner == 3072 { 7680 } else { 12288 },
            "FLUX transformer and text encoder dimensions do not match"
        );
        let mut time_features = [0f32; 256];
        for i in 0..128 {
            let angle = time * (-10000f32.ln() * i as f32 / 128.0).exp();
            time_features[i] = angle.cos();
            time_features[i + 128] = angle.sin();
        }
        let time = self
            .w
            .trace
            .constant(&time_features, &[1, 256])
            .as_dtype(Dtype::BF16)?;
        let time = self
            .w
            .lin(
                "time_guidance_embed.linear_2",
                &silu(&self.w.lin("time_guidance_embed.linear_1", &time)?)?,
            )?
            .reshape(&[1, 1, self.inner])?;
        let st = silu(&time)?;
        let mods = |key: &str, n: i32| -> anyhow::Result<Vec<Array>> {
            let x = self.w.lin(key, &st)?;
            Ok((0..n)
                .map(|i| slice(&x, i * self.inner, (i + 1) * self.inner))
                .collect())
        };
        let mi = mods("double_stream_modulation_img.linear", 6)?;
        let mt = mods("double_stream_modulation_txt.linear", 6)?;
        let ms = mods("single_stream_modulation.linear", 3)?;
        let nt = enc.dim(1);
        let nj = nt + h * w;
        let mut cos = Vec::with_capacity((nj * 64) as usize);
        let mut sin = Vec::with_capacity((nj * 64) as usize);
        for p in 0..nj {
            let ids = if p < nt {
                [0, 0, 0, p]
            } else {
                [0, (p - nt) / w, (p - nt) % w, 0]
            };
            for pos in ids {
                for k in 0..16 {
                    let a = pos as f64 * 2000f64.powf(-(2 * k) as f64 / 32.0);
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        let cos = self.w.trace.constant(&cos, &[1, 1, nj, 64]);
        let sin = self.w.trace.constant(&sin, &[1, 1, nj, 64]);
        let mut image = self.w.lin("x_embedder", latent)?;
        let mut text = self.w.lin("context_embedder", enc)?;
        for i in 0..self.doubles {
            let p = format!("transformer_blocks.{i}");
            let a = format!("{p}.attn");
            let im = modulate(&image, &mi, 0, 1)?;
            let tm = modulate(&text, &mt, 0, 1)?;
            let q = self.heads_norm(
                self.w.lin(&format!("{a}.to_q"), &im)?,
                &format!("{a}.norm_q.weight"),
            )?;
            let k = self.heads_norm(
                self.w.lin(&format!("{a}.to_k"), &im)?,
                &format!("{a}.norm_k.weight"),
            )?;
            let v = self.heads(self.w.lin(&format!("{a}.to_v"), &im)?)?;
            let tq = self.heads_norm(
                self.w.lin(&format!("{a}.add_q_proj"), &tm)?,
                &format!("{a}.norm_added_q.weight"),
            )?;
            let tk = self.heads_norm(
                self.w.lin(&format!("{a}.add_k_proj"), &tm)?,
                &format!("{a}.norm_added_k.weight"),
            )?;
            let tv = self.heads(self.w.lin(&format!("{a}.add_v_proj"), &tm)?)?;
            let out = self.attention(
                ops::concatenate(&[&tq, &q], 2)?,
                ops::concatenate(&[&tk, &k], 2)?,
                ops::concatenate(&[&tv, &v], 2)?,
                &cos,
                &sin,
            )?;
            image =
                image
                    .add(&mi[2].multiply(
                        &self.w.lin(&format!("{a}.to_out"), &out.slice(1, nt, nj)?)?,
                    )?)?;
            text = text.add(
                &mt[2].multiply(
                    &self
                        .w
                        .lin(&format!("{a}.to_add_out"), &out.slice(1, 0, nt)?)?,
                )?,
            )?;
            image =
                image
                    .add(&mi[5].multiply(
                        &self.ff(&modulate(&image, &mi, 3, 4)?, &format!("{p}.ff"))?,
                    )?)?;
            text =
                text.add(&mt[5].multiply(
                    &self.ff(&modulate(&text, &mt, 3, 4)?, &format!("{p}.ff_context"))?,
                )?)?;
        }
        let mut x = ops::concatenate(&[&text, &image], 1)?;
        for i in 0..self.singles {
            let p = format!("single_transformer_blocks.{i}.attn");
            let proj = self
                .w
                .lin(&format!("{p}.to_qkv_mlp_proj"), &modulate(&x, &ms, 0, 1)?)?;
            let d = self.inner;
            let mlp = (proj.dim(-1) - 3 * d) / 2;
            let q = self.heads_norm(slice(&proj, 0, d), &format!("{p}.norm_q.weight"))?;
            let k = self.heads_norm(slice(&proj, d, 2 * d), &format!("{p}.norm_k.weight"))?;
            let v = self.heads(slice(&proj, 2 * d, 3 * d))?;
            let attn = self.attention(q, k, v, &cos, &sin)?;
            let act = silu(&slice(&proj, 3 * d, 3 * d + mlp))?.multiply(&slice(
                &proj,
                3 * d + mlp,
                3 * d + 2 * mlp,
            ))?;
            let out = self.w.lin(
                &format!("{p}.to_out"),
                &ops::concatenate(&[&attn, &act], -1)?,
            )?;
            x = x.add(&ms[2].multiply(&out)?)?;
        }
        let modulation = self.w.lin("norm_out.linear", &st)?;
        let x = layer_norm(&x.slice(1, nt, nj)?)?
            .multiply(&slice(&modulation, 0, self.inner).add(&x.scalar(1.0))?)?
            .add(&slice(&modulation, self.inner, 2 * self.inner))?;
        self.w.lin("proj_out", &x)
    }
}
