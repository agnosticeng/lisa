//! Pure host image geometry and patch ordering; no file or device access.
use anyhow::{Result, ensure};
pub struct Pixels {
    pub values: Vec<f32>,
    pub height: usize,
    pub width: usize,
}
/// Bound the ViT workload to 1024 patches. Small inputs keep their exact pixels;
/// padding uses the reference's raw -1 value before normalization.
pub fn preprocess(image: &image::RgbImage) -> Result<Pixels> {
    let (w, h) = image.dimensions();
    ensure!(w > 0 && h > 0, "empty Inkling image");
    let mut resized = None;
    let (mut w, mut h) = (w as usize, h as usize);
    if w.div_ceil(40) * h.div_ceil(40) > 256 {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..48 {
            let mid = (lo + hi) * 0.5;
            let nw = ((w as f64 * mid) as usize).max(1);
            let nh = ((h as f64 * mid) as usize).max(1);
            if nw.div_ceil(40) * nh.div_ceil(40) <= 256 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let nw = ((w as f64 * lo) as usize).max(1);
        let nh = ((h as f64 * lo) as usize).max(1);
        resized = Some(image::imageops::resize(
            image,
            nw as u32,
            nh as u32,
            image::imageops::FilterType::Triangle,
        ));
        w = nw;
        h = nh;
    }
    let img = resized.as_ref().unwrap_or(image);
    let width = w.div_ceil(40) * 2;
    let height = h.div_ceil(40) * 2;
    let mut values = Vec::with_capacity(width * height * 1200);
    for py in 0..height {
        for px in 0..width {
            for c in 0..3 {
                for y in 0..20 {
                    for x in 0..20 {
                        let xx = px * 20 + x;
                        let yy = py * 20 + y;
                        let value = if xx < w && yy < h {
                            img.get_pixel(xx as u32, yy as u32)[c] as f32
                        } else {
                            -1.
                        };
                        values.push((value - 127.5) / 127.5);
                    }
                }
            }
        }
    }
    Ok(Pixels {
        values,
        height,
        width,
    })
}
pub fn rotary(height: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![1.; 32];
    let mut sin = vec![0.; 32];
    for y in 0..height {
        for x in 0..width {
            for pos in [x + 1, y + 1] {
                for i in 0..16 {
                    let theta = pos as f32 * 10000f32.powf(-(i as f32) / 16.);
                    cos.push(theta.cos());
                    sin.push(theta.sin());
                }
            }
        }
    }
    (cos, sin)
}
pub fn positions(table: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
    ensure!(
        table.len() == 32 * 32 * 1024,
        "invalid Inkling learned position table"
    );
    let mut out = vec![0.; height * width * 1024];
    for y in 0..height {
        for x in 0..width {
            let fy = (y as f32 + 0.5) / (height as f32 / 32.) - 0.5;
            let fx = (x as f32 + 0.5) / (width as f32 / 32.) - 0.5;
            let y0 = (fy.floor() as i32).clamp(0, 31) as usize;
            let y1 = (fy.floor() as i32 + 1).clamp(0, 31) as usize;
            let x0 = (fx.floor() as i32).clamp(0, 31) as usize;
            let x1 = (fx.floor() as i32 + 1).clamp(0, 31) as usize;
            let dy = (fy - y0 as f32).clamp(0., 1.);
            let dx = (fx - x0 as f32).clamp(0., 1.);
            for c in 0..1024 {
                let a = table[(y0 * 32 + x0) * 1024 + c];
                let b = table[(y0 * 32 + x1) * 1024 + c];
                let cc = table[(y1 * 32 + x0) * 1024 + c];
                let d = table[(y1 * 32 + x1) * 1024 + c];
                out[(y * width + x) * 1024 + c] = a * (1. - dx) * (1. - dy)
                    + b * dx * (1. - dy)
                    + cc * (1. - dx) * dy
                    + d * dx * dy;
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resized_images_respect_patch_budget_for_extreme_aspect_ratios() {
        for (w, h) in [(641, 641), (20000, 1), (1, 20000), (1920, 1080)] {
            let p = preprocess(&image::RgbImage::from_pixel(w, h, image::Rgb([0, 0, 0]))).unwrap();
            assert!(p.width * p.height <= 1024);
            assert_eq!(p.width % 2, 0);
            assert_eq!(p.height % 2, 0);
            assert_eq!(p.values.len(), p.width * p.height * 1200);
        }
    }
    #[test]
    fn patch_padding_and_rotary_cls() {
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 128]));
        let p = preprocess(&img).unwrap();
        assert_eq!((p.height, p.width, p.values.len()), (2, 2, 4800));
        assert_eq!(p.values[0], 1.);
        assert_eq!(p.values[400], -1.);
        assert_eq!(p.values[1], (-1. - 127.5) / 127.5);
        let (c, s) = rotary(2, 2);
        assert_eq!(c.len(), 160);
        assert_eq!(&c[..32], &[1.; 32]);
        assert_eq!(&s[..32], &[0.; 32]);
    }
}
