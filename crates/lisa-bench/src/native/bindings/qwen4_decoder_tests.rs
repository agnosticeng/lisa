#![cfg(test)]
use super::*;
use crate::core::norm::RmsNormExecution;
use crate::core::quant::QuantizedEmbeddingExecution;
use crate::core::quant::QuantizedLinearExecution;
use crate::core::{cache::FullAttentionCache, norm::RmsNorm};
use lisa_metal::{Dtype, Stream};
fn residual() -> anyhow::Result<GatedResidual> {
    use crate::bindings::qwen4_hyper::tests::linear;
    let up = linear(10240, 320, 13)?;
    let spec = lisa_models::qwen4::hyper::decode_order(2560, 4, 10240).unwrap();
    let p = PreparedGraph::new(
        MetalDevice::system_default(),
        lisa_models::qwen4::hyper::pack_rows(&spec),
        vec![],
    )?;
    let mut packed = p
        .run(&[&up.weight, &up.scales, &up.biases], &mut [])?
        .into_iter();
    Ok(GatedResidual {
        context: crate::metal_context::execution_context(),
        hc_norm: RmsNorm {
            weight: Array::from_slice(&vec![half::bf16::ONE; 10240], &[10240]),
            eps: 1e-6,
            group_size: Some(2560),
        },
        mix_down: linear(320, 10240, 29)?,
        mix_up: up,
        decode_up: Some(crate::core::quant::QuantizedLinear {
            weight: packed.next().unwrap(),
            scales: packed.next().unwrap(),
            biases: packed.next().unwrap(),
            group_size: 32,
            bits: 4,
        }),
        block_inject: Some(linear(4, 10240, 37)?),
        hc_count: 4,
        hidden: 10240,
        force_norm_operations: false,
    })
}
fn same(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    let (a, b) = (a.to_vec1::<u16>()?, b.to_vec1::<u16>()?);
    if let Some(i) = a.iter().zip(&b).position(|(a, b)| a != b) {
        anyhow::bail!("decoder bit mismatch at {i}: {} vs {}", a[i], b[i]);
    }
    Ok(())
}
#[test]
fn hyper_decoder_schedule_preserves_carried_stream_and_ple_bits() -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    let stream = Stream::gpu();
    lisa_metal::moe_decode::bf16_sigmoid_table(&stream).expect("sigmoid table");
    let mut attention = crate::bindings::qwen4_attention::tests::source(2560);
    let attn = Attention::load_on(
        crate::metal_context::execution_context(),
        &mut attention,
        "a",
        1e-6,
        24,
        2,
        256,
    )?;
    drop(attention);
    let mut source = crate::bindings::qwen4_moe::tests::source(512, 2560, 640)?;
    let moe = SparseMoeBlock::load_on(
        crate::metal_context::execution_context(),
        &mut source,
        "block",
        10,
    )?;
    drop(source);
    let warm = Array::from_slice(&vec![half::bf16::ONE; 2560], &[1, 1, 2560]);
    let _ = moe.forward(&warm)?;
    let hc = residual()?;
    let mut layer = DecoderLayer {
        context: crate::metal_context::execution_context(),
        block: Block::Attention(attn),
        mlp: moe,
        attn_hc: hc.clone(),
        mlp_hc: hc,
        ple: None,
        has_ple: false,
    };
    let rope = Rotary::new(64, 10000000.);
    for ple in [false, true] {
        layer.ple = if ple {
            Some(crate::bindings::qwen4_ple::tests::layer()?)
        } else {
            None
        };
        layer.has_ple = ple;
        for (batch, rows, carry) in [
            (1, 1, false),
            (1, 1, true),
            (1, 8, true),
            (1, 9, true),
            (2, 1, true),
            (2, 8, true),
        ] {
            let make = |width| {
                Array::from_slice(
                    &(0..batch * rows * width)
                        .map(|i| half::bf16::from_f32(((i * 13 % 251) as f32 - 125.) / 311.))
                        .collect::<Vec<_>>(),
                    &[batch, rows, width],
                )
            };
            let x = make(10240);
            let carried = make(2560);
            let inj = make(4);
            let pos =
                Array::from_slice(&(0..rows).map(|i| i as f32).collect::<Vec<_>>(), &[1, rows]);
            let ids = vec![vec![17; rows as usize]; batch as usize];
            let history = vec![vec![248044; 2]; batch as usize];
            let args = ple.then_some((&ids[..], &history[..]));
            let mut a = LayerCache::Full(FullAttentionCache::new(
                crate::metal_context::execution_context(),
                2,
                256,
            ));
            let mut b = LayerCache::Full(FullAttentionCache::new(
                crate::metal_context::execution_context(),
                2,
                256,
            ));
            stream.runtime().commands.flush_and_wait()?;
            let first = counts();
            let left = layer.original_forward(
                &x,
                carry.then_some(&carried),
                carry.then_some(&inj),
                Some(&mut a),
                &rope,
                0,
                &pos,
                args,
                true,
            )?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let right = layer.forward(
                &x,
                carry.then_some(&carried),
                carry.then_some(&inj),
                Some(&mut b),
                &rope,
                0,
                &pos,
                args,
                true,
            )?;
            stream.runtime().commands.flush_and_wait()?;
            let last = counts();
            for (a, b) in [&left.0, &left.1, &left.2]
                .into_iter()
                .zip([&right.0, &right.1, &right.2])
            {
                same(a, b)?;
                assert_eq!(a.dtype(), Dtype::Bfloat16);
            }
            if ple {
                let (a, ac) = a.ple_conv_mut();
                let (b, bc) = b.ple_conv_mut();
                same(a.as_ref().unwrap(), b.as_ref().unwrap())?;
                same(ac.as_ref().unwrap(), bc.as_ref().unwrap())?;
            }
            if !last.is_empty() {
                assert_eq!(
                    difference(&first, &middle),
                    difference(&middle, &last),
                    "batch={batch} rows={rows} carry={carry} ple={ple}"
                );
            }
        }
    }
    Ok(())
}
#[test]
fn draft_input_and_final_mix_plans_preserve_bits_and_dispatch() -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    use crate::bindings::qwen4_hyper::tests::linear;
    use crate::core::quant::QuantizedEmbedding;
    let stream = Stream::gpu();
    lisa_metal::moe_decode::bf16_sigmoid_table(&stream).expect("sigmoid table");
    let table = linear(32, 2560, 41)?;
    let embedding = QuantizedEmbedding {
        weight: table.weight,
        scales: table.scales,
        biases: table.biases,
        group_size: 32,
        bits: 4,
    };
    let fc_e = linear(2560, 2560, 47)?;
    let fc_h = linear(2560, 2560, 53)?;
    let norm = |n| RmsNorm {
        weight: Array::from_slice(&vec![half::bf16::ONE; n as usize], &[n]),
        eps: 1e-6,
        group_size: None,
    };
    let en = norm(2560);
    let hn = norm(10240);
    let mut mixer = residual()?;
    mixer.block_inject = None;
    for (batch, rows) in [(1, 1), (1, 8), (1, 9), (2, 3)] {
        let make = |n| {
            Array::from_slice(
                &(0..batch * rows * n)
                    .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.) / 311.))
                    .collect::<Vec<_>>(),
                &[batch, rows, n],
            )
        };
        let x = make(10240);
        let ids = Array::from_slice(
            &(0..batch * rows).map(|i| i % 32).collect::<Vec<_>>(),
            &[batch, rows],
        );
        let spec = lisa_models::qwen4::mtp::Input {
            hidden: 2560,
            streams: 4,
            embedding: (32, 4),
            projection: [(32, 4); 2],
            epsilon: [1e-6f32.to_bits(); 2],
        };
        let plan = PreparedGraph::new(
            MetalDevice::system_default(),
            spec.graph(batch, rows)?,
            vec![],
        )?;
        stream.runtime().commands.flush_and_wait()?;
        let first = counts();
        let e = en.forward(&embedding.forward(&ids)?)?;
        let e = fc_e.forward(&e)?;
        let h = hn.forward(&x)?.reshape(&[batch, rows, 4, 2560])?;
        let h = fc_h.forward(&h)?;
        let old = e.expand_dims(-2)?.add(&h)?.reshape(&[batch, rows, 10240])?;
        stream.runtime().commands.flush_and_wait()?;
        let middle = counts();
        let new = plan
            .run(
                &[
                    &ids,
                    &x,
                    &embedding.weight,
                    &embedding.scales,
                    &embedding.biases,
                    &en.weight,
                    &hn.weight,
                    &fc_e.weight,
                    &fc_e.scales,
                    &fc_e.biases,
                    &fc_h.weight,
                    &fc_h.scales,
                    &fc_h.biases,
                ],
                &mut [],
            )?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        stream.runtime().commands.flush_and_wait()?;
        let last = counts();
        same(&old, &new)?;
        if !last.is_empty() {
            assert_eq!(difference(&first, &middle), difference(&middle, &last));
        }
        for enabled in [false, true] {
            let out = make(2560);
            let inject = make(4);
            stream.runtime().commands.flush_and_wait()?;
            let first = counts();
            let scale = mixer.norm_scale_q()?;
            let (multi, normed) = lisa_metal::kernels::inject_norm(
                &x,
                enabled.then_some(&out),
                enabled.then_some(&inject),
                &scale,
                4,
                2560,
                batch * rows,
                false,
                1e-6,
                &stream,
            )
            .unwrap();
            let multi = multi.reshape(&[batch, rows, 10240])?;
            let normed = normed.reshape(&[batch, rows, 10240])?;
            let (sample, _) = mixer.mix_from_normed(&normed, false)?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let (new, new_multi) = mixer.finish(
                &x,
                enabled.then_some(&out),
                enabled.then_some(&inject),
                1e-6,
            )?;
            stream.runtime().commands.flush_and_wait()?;
            let last = counts();
            same(&sample, &new)?;
            same(&multi, &new_multi)?;
            if !last.is_empty() {
                assert_eq!(difference(&first, &middle), difference(&middle, &last));
            }
        }
    }
    Ok(())
}
