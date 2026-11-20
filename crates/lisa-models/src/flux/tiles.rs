//! Bounded VAE decode geometry and cosine overlap blending.
//! Adapted from PrismML mflux-prism bcd13e8, VAETiler (MIT; see NOTICE).
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Kernel, TensorOperation as T};

#[derive(Clone, Copy, Debug)]
pub struct Tile {
    pub y: usize,
    pub x: usize,
    pub h: usize,
    pub w: usize,
}

pub fn tiles(height: usize, width: usize) -> Vec<Tile> {
    let mut result = Vec::new();
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let h = 8.min(height - y);
            let w = 8.min(width - x);
            if (y > 0 && h <= 4) || (x > 0 && w <= 4) {
                continue;
            }
            result.push(Tile { y, x, h, w });
        }
    }
    result
}
impl Tile {
    pub fn graph(self) -> Graph {
        let mut b = Builder::new(1, 0, 0);
        let y = b.one(
            Kernel::Tensor(T::Slice {
                axis: 2,
                start: self.y as i32,
                end: (self.y + self.h) as i32,
            }),
            vec![Value::Input(0)],
        );
        let x = b.one(
            Kernel::Tensor(T::Slice {
                axis: 3,
                start: self.x as i32,
                end: (self.x + self.w) as i32,
            }),
            vec![y],
        );
        let x = b.one(Kernel::Tensor(T::Contiguous), vec![x]);
        b.finish(vec![x])
    }
    pub fn blend(
        self,
        height: usize,
        width: usize,
        pixels: &[f32],
        out: &mut [f32],
        count: &mut [f32],
    ) -> anyhow::Result<()> {
        let (h, w) = (self.h * 16, self.w * 16);
        anyhow::ensure!(
            pixels.len() == h * w * 3
                && out.len() == height * width * 3
                && count.len() == height * width,
            "invalid VAE tile pixels"
        );
        let ramp = |i: usize| 0.5 - 0.5 * ((i as f32 / 63.0) * std::f32::consts::PI).cos();
        let weight = |i: usize, size: usize, offset: usize, total: usize| {
            let overlap = 64.min(size - 1);
            let mut v = 1.0;
            if offset > 0 && i < overlap {
                v = ramp(i);
            }
            if offset + size < total && i >= size - overlap {
                v = 1.0 - ramp(i - (size - overlap));
            }
            v
        };
        for y in 0..h {
            for x in 0..w {
                let v = weight(y, h, self.y * 16, height) * weight(x, w, self.x * 16, width);
                let dst = (self.y * 16 + y) * width + self.x * 16 + x;
                count[dst] += v;
                for c in 0..3 {
                    out[dst * 3 + c] += pixels[(y * w + x) * 3 + c] * v;
                }
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiled_blend_covers_rectangular_images_without_seams() {
        for (h, w) in [(4, 4), (8, 8), (10, 14), (32, 32), (64, 64)] {
            let mut out = vec![0.0; h * w * 256 * 3];
            let mut count = vec![0.0; h * w * 256];
            for t in tiles(h, w) {
                t.graph().validate().unwrap();
                t.blend(
                    h * 16,
                    w * 16,
                    &vec![0.75; t.h * t.w * 256 * 3],
                    &mut out,
                    &mut count,
                )
                .unwrap();
            }
            assert!(count.iter().all(|&v| v > 0.0));
            for (i, v) in out.iter().enumerate() {
                assert!((v / count[i / 3] - 0.75).abs() < 1e-6);
            }
        }
    }
}
