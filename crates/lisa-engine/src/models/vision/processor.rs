//! Qwen still-image preprocessing. Resize coefficients follow Pillow's
//! separable bicubic filter (22-bit fixed point); patch order follows Qwen3-VL.
use anyhow::ensure;
use image::RgbImage;
use serde_json::Value;

pub struct Pixels {
    pub values: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let values: Vec<f64> = (start..end)
                .map(|p| cubic((p as f64 - center + 0.5) / filter_scale))
                .collect();
            let sum: f64 = values.iter().sum();
            let values = values
                .into_iter()
                .map(|v| {
                    let v = v / sum * (1 << 22) as f64;
                    (v + if v < 0.0 { -0.5 } else { 0.5 }) as i32
                })
                .collect();
            (start, values)
        })
        .collect()
}

fn resize(input: &RgbImage, width: usize, height: usize) -> RgbImage {
    let iw = input.width() as usize;
    let ih = input.height() as usize;
    let horizontal = if iw == width {
        input.clone()
    } else {
        let coeffs = coefficients(iw, width);
        let mut out = RgbImage::new(width as u32, ih as u32);
        for y in 0..ih {
            for (x, (start, weights)) in coeffs.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 1i64 << 21;
                    for (j, weight) in weights.iter().enumerate() {
                        acc += i64::from(input.get_pixel((start + j) as u32, y as u32)[c])
                            * i64::from(*weight);
                    }
                    out.get_pixel_mut(x as u32, y as u32)[c] = (acc >> 22).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    if ih == height {
        return horizontal;
    }
    let coeffs = coefficients(ih, height);
    let mut out = RgbImage::new(width as u32, height as u32);
    for (y, (start, weights)) in coeffs.iter().enumerate() {
        for x in 0..width {
            for c in 0..3 {
                let mut acc = 1i64 << 21;
                for (j, weight) in weights.iter().enumerate() {
                    acc += i64::from(horizontal.get_pixel(x as u32, (start + j) as u32)[c])
                        * i64::from(*weight);
                }
                out.get_pixel_mut(x as u32, y as u32)[c] = (acc >> 22).clamp(0, 255) as u8;
            }
        }
    }
    out
}

pub fn qwen(
    image: &RgbImage,
    patch: usize,
    temporal: usize,
    merge: usize,
    processor: &Value,
) -> anyhow::Result<Pixels> {
    ensure!(
        patch > 0 && patch <= 32 && temporal > 0 && temporal <= 8 && merge > 0 && merge <= 8,
        "invalid vision patch geometry"
    );
    let factor = (patch * merge) as f64;
    let h = f64::from(image.height());
    let w = f64::from(image.width());
    ensure!(
        h > 0.0 && w > 0.0 && h.max(w) / h.min(w) <= 200.0,
        "unsupported image aspect ratio"
    );
    let min = processor
        .pointer("/size/shortest_edge")
        .and_then(Value::as_u64)
        .unwrap_or(3136) as f64;
    // Bound materialized bidirectional attention independently of input size.
    let max = (processor
        .pointer("/size/longest_edge")
        .and_then(Value::as_u64)
        .unwrap_or(1003520) as f64)
        .min(1003520.0);
    ensure!(
        min > 0.0 && min <= max && factor * factor <= max,
        "unsupported image processor pixel bounds"
    );
    let mut rh = (h / factor).round_ties_even().max(1.0) * factor;
    let mut rw = (w / factor).round_ties_even().max(1.0) * factor;
    if rh * rw > max {
        let beta = (h * w / max).sqrt();
        rh = (h / beta / factor).floor().max(1.0) * factor;
        rw = (w / beta / factor).floor().max(1.0) * factor;
    } else if rh * rw < min {
        let beta = (min / (h * w)).sqrt();
        rh = (h * beta / factor).ceil() * factor;
        rw = (w * beta / factor).ceil() * factor;
    }
    ensure!(rh * rw <= max, "resized image exceeds the processor budget");
    let (rh, rw) = (rh as usize, rw as usize);
    let resized = resize(image, rw, rh);
    let (gh, gw) = (rh / patch, rw / patch);
    let feature_size = 3 * temporal * patch * patch;
    let mut values = Vec::with_capacity(gh * gw * feature_size);
    let mut mean = [0.5f32; 3];
    let mut std = [0.5f32; 3];
    for (field, out) in [("image_mean", &mut mean), ("image_std", &mut std)] {
        if let Some(v) = processor.get(field).and_then(Value::as_array) {
            ensure!(v.len() == 3, "{field} must have three channels");
            for i in 0..3 {
                out[i] = v[i]
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("invalid {field}"))?
                    as f32;
            }
        }
    }
    ensure!(
        std.iter().all(|v| v.is_finite() && *v > 0.0) && mean.iter().all(|v| v.is_finite()),
        "invalid image normalization"
    );
    for by in 0..gh / merge {
        for bx in 0..gw / merge {
            for iy in 0..merge {
                for ix in 0..merge {
                    for c in 0..3 {
                        for _ in 0..temporal {
                            for py in 0..patch {
                                for px in 0..patch {
                                    let value = resized.get_pixel(
                                        ((bx * merge + ix) * patch + px) as u32,
                                        ((by * merge + iy) * patch + py) as u32,
                                    )[c];
                                    values.push(
                                        ((f64::from(value) / 255.0) as f32 - mean[c]) / std[c],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Pixels {
        values,
        height: gh,
        width: gw,
    })
}

/// GLM keeps aspect ratio inside a black, patch-aligned canvas.
pub fn glm(
    image: &RgbImage,
    patch: usize,
    temporal: usize,
    merge: usize,
) -> anyhow::Result<Pixels> {
    ensure!(
        patch > 0 && patch <= 32 && temporal > 0 && temporal <= 8 && merge > 0 && merge <= 8,
        "invalid GLM image geometry"
    );
    let factor = patch * merge;
    let h = image.height() as usize;
    let w = image.width() as usize;
    ensure!(h > 0 && w > 0, "empty image");
    let align = |v: usize| v.div_ceil(factor) * factor;
    let min = 16 * factor * factor;
    let max = 1024 * factor * factor;
    let (mut rh, mut rw) = (align(h), align(w));
    if rh * rw < min {
        let scale = (min as f64 / (h * w) as f64).sqrt();
        rh = align((h as f64 * scale).ceil() as usize);
        rw = align((w as f64 * scale).ceil() as usize);
    }
    if rh * rw > max {
        let (mut lo, mut hi) = (1, h);
        rh = factor;
        rw = factor;
        while lo <= hi {
            let ch = (lo + hi) / 2;
            let cw = (w * ch / h).max(1);
            let (ah, aw) = (align(ch), align(cw));
            if ah * aw <= max {
                rh = ah;
                rw = aw;
                lo = ch + 1;
            } else {
                hi = ch - 1;
            }
        }
    }
    let mut scale = (rh as f64 / h as f64).min(rw as f64 / w as f64);
    if h * w >= min {
        scale = scale.min(1.0);
    }
    let (ch, cw) = (
        ((h as f64 * scale).floor() as usize).clamp(1, rh),
        ((w as f64 * scale).floor() as usize).clamp(1, rw),
    );
    let resized = resize(image, cw, ch);
    let (gh, gw) = (rh / patch, rw / patch);
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.26130258, 0.27577711];
    let mut values = Vec::with_capacity(gh * gw * 3 * temporal * patch * patch);
    for by in 0..gh / merge {
        for bx in 0..gw / merge {
            for iy in 0..merge {
                for ix in 0..merge {
                    for c in 0..3 {
                        for _ in 0..temporal {
                            for py in 0..patch {
                                for px in 0..patch {
                                    let y = (by * merge + iy) * patch + py;
                                    let x = (bx * merge + ix) * patch + px;
                                    let v = if y < ch && x < cw {
                                        resized.get_pixel(x as u32, y as u32)[c]
                                    } else {
                                        0
                                    };
                                    values.push(
                                        ((v as f64 * (1.0 / 255.0)) as f32 - mean[c]) / std[c],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Pixels {
        values,
        height: gh,
        width: gw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_matches_pillow_fixed_point_fixture() {
        let image = RgbImage::from_raw(3, 2, (0..18).map(|i| (i * 13) as u8).collect()).unwrap();
        assert_eq!(
            resize(&image, 2, 3).as_raw(),
            &[
                4, 17, 30, 58, 71, 84, 71, 84, 97, 125, 138, 151, 137, 150, 163, 191, 204, 217,
            ]
        );
    }

    #[test]
    fn processors_align_and_bound_patch_grids() -> anyhow::Result<()> {
        let image = RgbImage::from_pixel(97, 61, image::Rgb([255, 0, 0]));
        let q = qwen(&image, 16, 2, 2, &serde_json::json!({}))?;
        assert_eq!((q.height, q.width), (4, 6));
        assert_eq!(q.values.len(), 4 * 6 * 3 * 2 * 16 * 16);
        let g = glm(&image, 14, 2, 2)?;
        assert_eq!((g.height, g.width), (8, 12));
        assert_eq!(g.values.len(), 8 * 12 * 3 * 2 * 14 * 14);
        let image = RgbImage::new(3840, 2160);
        let g = glm(&image, 14, 2, 2)?;
        assert!(g.height * g.width <= 1024 * 4);
        assert_eq!((g.height % 2, g.width % 2), (0, 0));
        Ok(())
    }
}
