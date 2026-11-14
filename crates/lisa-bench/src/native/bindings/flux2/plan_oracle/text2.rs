#![cfg(test)]
use super::{silu, weights::Weights};
use lisa_metal::{Array, Dtype, fast, ops};
pub fn forward(w: &Weights, tokens: &[u32], real: usize) -> anyhow::Result<Array> {
    let (hidden, hd, heads, kv, depth) = (128, 128, 2, 1, 27);
    let ids = Array::from_slice(&tokens, &[512]);
    let ew = w.get("embed_tokens.weight")?.take_axis(&ids, 0)?;
    let es = w.get("embed_tokens.scales")?.take_axis(&ids, 0)?;
    let eb = w.get("embed_tokens.biases")?.take_axis(&ids, 0)?;
    let mut x = ops::dequantize(&ew, &es, &eb, 64, 4)?.reshape(&[1, 512, hidden])?;
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
    let mask = Array::from_slice(&mask, &[1, 1, 512, 512]);
    let mut taps = Vec::new();
    for i in 0..depth {
        let p = format!("layers.{i}");
        let norm = |x: &Array, key: &str| -> anyhow::Result<Array> {
            Ok(fast::rms_norm(
                x,
                Some(&w.get(&format!("{p}.{key}.weight"))?),
                1e-6,
            )?)
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
        let q = fast::rope(&q.contiguous()?, hd, false, Some(1e6), 1.0, 0, None)?
            .as_dtype(Dtype::Float32)?;
        let k = fast::rope(&k.contiguous()?, hd, false, Some(1e6), 1.0, 0, None)?
            .as_dtype(Dtype::Float32)?;
        let a = crate::core::media_attention::attention(
            &q,
            &k,
            &v.as_dtype(Dtype::Float32)?,
            1.0 / (hd as f32).sqrt(),
            Some(&mask),
        )?
        .as_dtype(Dtype::Bfloat16)?;
        let a = a
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[1, 512, heads * hd])?;
        x = x.add(&w.lin(&format!("{p}.self_attn.o_proj"), &a)?)?;
        let n = norm(&x, "post_attention_layernorm")?;
        let act = silu(&w.lin(&format!("{p}.mlp.gate_proj"), &n)?)?
            .multiply(&w.lin(&format!("{p}.mlp.up_proj"), &n)?)?;
        x = x.add(&w.lin(&format!("{p}.mlp.down_proj"), &act)?)?;
        x.eval_enqueue()?;
        if [8, 17, 26].contains(&i) {
            taps.push(x.clone());
        }
    }
    let out = ops::concatenate(&taps.iter().collect::<Vec<_>>(), -1)?;
    out.eval()?;
    Ok(out)
}
