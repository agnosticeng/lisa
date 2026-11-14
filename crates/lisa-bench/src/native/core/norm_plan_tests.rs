#![cfg(test)]
use super::*;
// Frozen pre-plan arithmetic oracle; keep association and cast boundaries.
fn original_rms_row_exact(
    x: &Array,
    scale: Option<&Array>,
    eps: f32,
    h: usize,
    force_ops: bool,
) -> lisa_metal::error::Result<Array> {
    // ds4 parity: the fused single-kernel port (same pinned association as the
    // op path below — 4 sequential squares per lane, simd_sum butterfly,
    // rsqrt(total/128 + eps), bf16 rounding BEFORE the weight multiply).
    // LISA_GLM53_RMS_OPS=1 forces the op path for A/B comparison.
    if !force_ops {
        if let Some(fused) = lisa_metal::kernels::sparse::kernels::rms_row_fused(
            x,
            eps,
            h,
            &lisa_metal::Stream::thread_local_or_default(),
        ) {
            let flat = fused.reshape(&x.shape())?;
            return match scale {
                Some(w) => flat.multiply(w),
                None => Ok(flat),
            };
        }
    }
    use lisa_metal::ops::indexing::IndexOp;
    let w_full = x.dim(-1) as usize;
    let groups = w_full / h;
    let slices = (h / 128) as i32;
    let lead = &x.shape()[..x.shape().len() - 1];
    // [lead..., groups, slices, 32, 4]
    let grouped = x.reshape(&[lead, &[groups as i32, slices, 32i32, 4i32]].concat())?;
    let sq = lisa_metal::ops::multiply(
        &grouped.as_dtype(Dtype::Float32)?,
        &grouped.as_dtype(Dtype::Float32)?,
    )?;
    if std::env::var("E_SQ").is_ok() {
        eprintln!("[flag E_SQ]");
        let _ = lisa_metal::transforms::eval([&sq]);
    }
    // per-lane sequential sum over the 4 elements -> [.., groups, slices, 32]
    let mut lane = sq.index((Ellipsis, 0..1)).contiguous().unwrap();
    for i in 1..4i32 {
        lane = lane.add(&sq.index((Ellipsis, i..i + 1)).contiguous().unwrap())?;
    }
    if std::env::var("E_LANE").is_ok() {
        eprintln!("[flag E_LANE]");
        let _ = lisa_metal::transforms::eval([&lane]);
    }
    let lane_v = lane.reshape(&[lead, &[groups as i32, slices, 32i32]].concat())?;
    // per-simdgroup xor-butterfly over the 32 lanes -> [.., groups, slices]
    let mut v = lane_v;
    let mut width = 32i32;
    while width >= 2 {
        let half = width / 2;
        let a = v.index((Ellipsis, 0..half)).contiguous().unwrap();
        let b = v.index((Ellipsis, half..width)).contiguous().unwrap();
        v = a.add(&b)?;
        width /= 2;
    }
    let partials = v.reshape(&[lead, &[groups as i32, slices]].concat())?;
    // zero-pad the partials to 32 slots, then the second butterfly
    let mut zshape = lead.to_vec();
    zshape.push(groups as i32);
    zshape.push(32i32 - slices);
    let zeros = lisa_metal::ops::zeros::<f32>(&zshape)
        .map_err(|e| lisa_metal::error::Exception::custom(e.to_string()))?;
    let padded = lisa_metal::ops::concatenate(&[&partials, &zeros], -1)?;
    if std::env::var("E_PADDED").is_ok() {
        eprintln!("[flag E_PADDED]");
        let _ = lisa_metal::transforms::eval([&padded]);
    }
    let mut v2 = padded;
    width = 32;
    while width >= 2 {
        let half = width / 2;
        let a = v2.index((Ellipsis, 0..half)).contiguous().unwrap();
        let b = v2.index((Ellipsis, half..width)).contiguous().unwrap();
        v2 = a.add(&b)?;
        width /= 2;
    }
    // total: [lead..., groups] -> unsqueeze for broadcasting
    let total = v2.reshape(&[lead, &[groups as i32, 1i32, 1i32, 1i32]].concat())?;
    if std::env::var("E_TOTAL").is_ok() {
        eprintln!("[flag E_TOTAL]");
        let _ = lisa_metal::transforms::eval([&total]);
    }
    let scaled = total.clone() / (h as f32);
    let biased = scaled.add(&Array::from_f32(eps))?;
    let inv = lisa_metal::ops::rsqrt(&biased)
        .map_err(|e| lisa_metal::error::Exception::custom(e.to_string()))?;
    if std::env::var("E_INV").is_ok() {
        eprintln!("[flag E_INV]");
        let _ = lisa_metal::transforms::eval([&inv]);
    }
    let normalized = (grouped.as_dtype(Dtype::Float32)? * &inv).as_dtype(Dtype::Bfloat16)?;
    if std::env::var("E_NORMED").is_ok() {
        eprintln!("[flag E_NORMED]");
        let _ = lisa_metal::transforms::eval([&normalized]);
    }
    let flat = normalized.reshape(&x.shape())?;
    let out = match scale {
        Some(w) => flat.multiply(w),
        None => Ok(flat.clone()),
    };
    out
}

#[test]
fn grouped_rms_plans_preserve_bits_and_dispatch_order() -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    let stream = lisa_metal::Stream::gpu();
    for h in [128usize, 256, 2560, 4096] {
        for (batch, rows) in [(1, 1), (1, 9), (2, 3)] {
            for dtype in [Dtype::Bfloat16, Dtype::Float32] {
                let width = (2 * h) as i32;
                let x = Array::from_slice(
                    &(0..batch * rows * width)
                        .map(|i| ((i * 37 % 241) as f32 - 120.) / 93.)
                        .collect::<Vec<_>>(),
                    &[batch, rows, width],
                )
                .as_dtype(dtype)?;
                let weight = Array::from_slice(
                    &(0..width)
                        .map(|i| 0.5 + (i % 13) as f32 / 19.)
                        .collect::<Vec<_>>(),
                    &[width],
                )
                .as_dtype(Dtype::Bfloat16)?;
                for weighted in [false, true] {
                    for force_ops in [false, true] {
                        let scale = weighted.then_some(&weight);
                        stream.runtime().commands.flush_and_wait()?;
                        let before = counts();
                        let a = original_rms_row_exact(&x, scale, 1e-6, h, force_ops)?;
                        stream.runtime().commands.flush_and_wait()?;
                        let middle = counts();
                        let b = grouped_rms_plan(&x, scale, 1e-6, h, force_ops)?;
                        stream.runtime().commands.flush_and_wait()?;
                        let after = counts();
                        assert_eq!(a.shape(), b.shape());
                        assert_eq!(a.dtype(), b.dtype());
                        assert_eq!(
                            a.to_vec1::<u16>()?,
                            b.to_vec1::<u16>()?,
                            "h={h} batch={batch} rows={rows} dtype={dtype:?} scale={weighted} ops={force_ops}"
                        );
                        if !after.is_empty() {
                            assert_eq!(
                                difference(&before, &middle),
                                difference(&middle, &after),
                                "h={h} batch={batch} rows={rows} dtype={dtype:?} scale={weighted} ops={force_ops}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
