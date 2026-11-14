//! FLUX.2 Qwen3 text conditioning computation and token policy.
use super::symbolic::{self as ops, Program, Schema, Tensor as Array, Trace, Weights, silu};
use anyhow::ensure;
use lisa_backend::ScalarType as Dtype;
pub fn prompt(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}
pub fn pad(tokens: &mut Vec<u32>) -> usize {
    tokens.truncate(512);
    let real = tokens.len();
    tokens.resize(512, 151643);
    real
}
#[derive(Clone, Copy)]
pub struct Geometry {
    pub hidden: i32,
    pub hd: i32,
    pub heads: i32,
    pub kv: i32,
    pub depth: usize,
}
impl Geometry {
    pub fn from_schema(schema: &Schema) -> anyhow::Result<Self> {
        let dim = |name: &str| {
            schema
                .get(name)
                .and_then(|x| x.shape.first())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing text geometry {name}"))
        };
        let hidden = dim("layers.0.input_layernorm.weight")?;
        let hd = dim("layers.0.self_attn.q_norm.weight")?;
        ensure!(hd == 128, "invalid Qwen text head dimension");
        let heads = dim("layers.0.self_attn.q_proj.weight")? / hd;
        let kv = dim("layers.0.self_attn.k_proj.weight")? / hd;
        let depth = ops::count(schema, "layers", "input_layernorm.weight");
        ensure!(
            heads == 32 && kv == 8 && matches!((depth, hidden), (36, 2560) | (36, 4096)),
            "expected the Klein 4B or 9B Qwen3 text encoder"
        );
        Ok(Self {
            hidden,
            hd,
            heads,
            kv,
            depth,
        })
    }
}
pub fn program(schema: &Schema, geometry: Geometry, real: usize) -> anyhow::Result<Program> {
    ensure!(
        geometry.hidden > 0
            && geometry.hd == 128
            && geometry.heads > 0
            && geometry.kv > 0
            && geometry.heads % geometry.kv == 0
            && geometry.depth >= 27
            && geometry.depth <= 128,
        "invalid FLUX text geometry"
    );
    ensure!(real > 0 && real <= 512, "empty encoder input");
    let Geometry {
        hidden,
        hd,
        heads,
        kv,
        depth,
    } = geometry;
    let trace = Trace::new();
    let w = Weights {
        trace: trace.clone(),
        schema,
    };
    let ids = trace.input(&[512], Dtype::U32);
    let ew = w.get("embed_tokens.weight")?.take_axis(&ids, 0)?;
    let es = w.get("embed_tokens.scales")?.take_axis(&ids, 0)?;
    let eb = w.get("embed_tokens.biases")?.take_axis(&ids, 0)?;
    let mut x = ew.dequantize(&es, &eb)?.reshape(&[1, 512, hidden])?;
    let mask: Vec<f32> = (0..512)
        .flat_map(|i| {
            (0..512).map(move |j| {
                if j > i || j >= real {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    let mask = trace.constant(&mask, &[1, 1, 512, 512]);
    let mut taps = Vec::new();
    for i in 0..depth {
        let p = format!("layers.{i}");
        let norm = |x: &Array, key: &str| -> anyhow::Result<Array> {
            Ok(x.rms(&w.get(&format!("{p}.{key}.weight"))?, 1e-6)?)
        };
        let n = norm(&x, "input_layernorm")?;
        let q = norm(
            &w.lin(&format!("{p}.self_attn.q_proj"), &n)?
                .reshape(&[1, 512, heads, hd])?,
            "self_attn.q_norm",
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = norm(
            &w.lin(&format!("{p}.self_attn.k_proj"), &n)?
                .reshape(&[1, 512, kv, hd])?,
            "self_attn.k_norm",
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = w
            .lin(&format!("{p}.self_attn.v_proj"), &n)?
            .reshape(&[1, 512, kv, hd])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q = q.contiguous()?.rotary(hd, 1e6, 0)?.as_dtype(Dtype::F32)?;
        let k = k.contiguous()?.rotary(hd, 1e6, 0)?.as_dtype(Dtype::F32)?;
        let a = ops::attention(
            &q,
            &k,
            &v.as_dtype(Dtype::F32)?,
            1.0 / (hd as f32).sqrt(),
            Some(&mask),
        )?
        .as_dtype(Dtype::BF16)?;
        let a = a
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[1, 512, heads * hd])?;
        x = x.add(&w.lin(&format!("{p}.self_attn.o_proj"), &a)?)?;
        let n = norm(&x, "post_attention_layernorm")?;
        let act = silu(&w.lin(&format!("{p}.mlp.gate_proj"), &n)?)?
            .multiply(&w.lin(&format!("{p}.mlp.up_proj"), &n)?)?;
        x = x.add(&w.lin(&format!("{p}.mlp.down_proj"), &act)?)?;
        if [8, 17, 26].contains(&i) {
            taps.push(x.clone());
        }
    }
    let out = ops::concatenate(&taps.iter().collect::<Vec<_>>(), -1)?;
    trace.finish(&[out])
}
