//! FLUX request policy, sampling schedule, latent arithmetic and pixel conversion.
use crate::graph::{Builder, Graph, Value};
use anyhow::ensure;
use lisa_backend::{Binary, Kernel, ScalarType as D, TensorOperation as T};
#[derive(Clone, Debug)]
pub struct Options {
    pub width: u32,
    pub height: u32,
    /// None selects the loaded model's native default (4 for FLUX.2, 16 for FLUX 3).
    pub steps: Option<u32>,
    pub seed: u64,
    /// Skip FLUX 3's local prompt expansion when supplying a complete caption.
    pub prompt_is_caption: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            steps: None,
            seed: 0,
            prompt_is_caption: false,
        }
    }
}
impl Options {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            (64..=1024).contains(&self.width)
                && (64..=1024).contains(&self.height)
                && self.width % 32 == 0
                && self.height % 32 == 0,
            "image width and height must be multiples of 32 between 64 and 1024"
        );
        ensure!(
            self.steps.is_none_or(|steps| (1..=50).contains(&steps)),
            "steps must be between 1 and 50"
        );
        Ok(())
    }
}
pub fn schedule2(tokens: usize, steps: usize) -> Vec<f32> {
    let a = 0.00016927 * tokens as f64 + 0.45666666;
    let b = 8.73809524e-5 * tokens as f64 + 1.89833333;
    let slope = (b - a) / (10.0 - 200.0);
    let mu = if tokens > 4300 {
        a
    } else {
        slope * steps as f64 + (a - slope * 200.0)
    };
    let emu = mu.exp();
    let mut out = Vec::new();
    for i in 0..steps {
        let t = if steps == 1 {
            1.0
        } else {
            1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps - 1) as f64
        };
        out.push((emu / (emu + (1.0 / t - 1.0))) as f32);
    }
    out.push(0.0);
    out
}
/// SplitMix64 + Box–Muller: stable LISA seeds, independent of process-global RNG.
/// The same numeric seed is not a promise of MLX's Threefry noise sequence.
pub fn normal_noise(count: usize, mut state: u64) -> Vec<f32> {
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let r = (-2.0 * uniform().ln()).sqrt();
        let a = std::f64::consts::TAU * uniform();
        out.push((r * a.cos()) as f32);
        if out.len() < count {
            out.push((r * a.sin()) as f32);
        }
    }
    out
}
/// Follow the executable reference's alpha=16. The exported config says 6.93,
/// but pipeline.generate uses 16 and the public CLI does not override it.
pub fn schedule3(steps: u32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            16.0 * t / (1.0 + 15.0 * t)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Klein2,
    Klein3,
}
pub fn initialize(family: Family, h: i32, w: i32) -> anyhow::Result<Graph> {
    ensure!(
        h > 0 && w > 0 && h.checked_mul(w).is_some(),
        "invalid latent dimensions"
    );
    let mut b = Builder::new(1, 0, 0);
    let x = b.one(Kernel::Tensor(T::Cast(D::BF16)), vec![Value::Input(0)]);
    let x = match family {
        Family::Klein2 => {
            let x = b.one(Kernel::Tensor(T::Transpose(vec![0, 2, 1])), vec![x]);
            b.one(Kernel::Tensor(T::Contiguous), vec![x])
        }
        Family::Klein3 => b.one(Kernel::Tensor(T::Cast(D::F32)), vec![x]),
    };
    Ok(b.finish(vec![x]))
}
pub fn noise_shape(family: Family, h: i32, w: i32) -> [i32; 3] {
    match family {
        Family::Klein2 => [1, 128, h * w],
        Family::Klein3 => [1, h * w, 128],
    }
}
pub fn step(family: Family, dt: f32) -> Graph {
    let mut b = Builder::new(2, 0, 0);
    let dt = b.scalar(dt);
    let mut delta = b.one(
        Kernel::Tensor(T::Binary(Binary::Multiply)),
        vec![Value::Input(1), dt],
    );
    if family == Family::Klein3 {
        delta = b.one(Kernel::Tensor(T::Cast(D::BF16)), vec![delta]);
        delta = b.one(Kernel::Tensor(T::Cast(D::F32)), vec![delta]);
    }
    let x = b.one(
        Kernel::Tensor(T::Binary(Binary::Add)),
        vec![Value::Input(0), delta],
    );
    b.finish(vec![x])
}
pub fn decoder_input(h: i32, w: i32) -> Graph {
    let mut b = Builder::new(1, 0, 0);
    let x = b.one(
        Kernel::Tensor(T::Reshape(vec![1, h, w, 128])),
        vec![Value::Input(0)],
    );
    let x = b.one(Kernel::Tensor(T::Transpose(vec![0, 3, 1, 2])), vec![x]);
    b.finish(vec![x])
}
pub fn pixel_readback() -> Graph {
    let mut b = Builder::new(1, 0, 0);
    let x = b.one(Kernel::Tensor(T::Cast(D::F32)), vec![Value::Input(0)]);
    let x = b.one(Kernel::Tensor(T::Contiguous), vec![x]);
    b.finish(vec![x])
}
pub fn rgb(family: Family, values: &[f32]) -> anyhow::Result<Vec<u8>> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "image generation produced non-finite pixels"
    );
    Ok(values
        .iter()
        .map(|v| match family {
            Family::Klein2 => ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8,
            Family::Klein3 => ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8,
        })
        .collect())
}
pub fn validate3(config: &serde_json::Value) -> anyhow::Result<()> {
    let d = &config["dit"];
    ensure!(
        config["quantization"].is_null()
            && d["hidden"] == 3072
            && d["heads"] == 24
            && d["depth"] == 5
            && d["depth_single"] == 28
            && d["theta"] == 10000
            && d["image_rope_scale"] == 0.5,
        "unsupported FLUX 3 architecture or quantization; select the original BF16 package"
    );
    Ok(())
}
/// Adapt BFL decoder names and storage layout without changing payload precision.
pub fn decoder_weight(key: &str, shape: &[i32]) -> anyhow::Result<(String, Option<Graph>)> {
    let mut name = key
        .replace("decoder.post_quant_conv", "post_quant_conv")
        .replace("decoder.mid.block_1", "decoder.mid_block.resnets.0")
        .replace("decoder.mid.block_2", "decoder.mid_block.resnets.1")
        .replace(
            "decoder.mid.attn_1.norm",
            "decoder.mid_block.attentions.0.group_norm",
        )
        .replace(
            "decoder.mid.attn_1.proj_out",
            "decoder.mid_block.attentions.0.to_out",
        )
        .replace(
            "decoder.mid.attn_1.q",
            "decoder.mid_block.attentions.0.to_q",
        )
        .replace(
            "decoder.mid.attn_1.k",
            "decoder.mid_block.attentions.0.to_k",
        )
        .replace(
            "decoder.mid.attn_1.v",
            "decoder.mid_block.attentions.0.to_v",
        )
        .replace("decoder.norm_out", "decoder.conv_norm_out")
        .replace("nin_shortcut", "conv_shortcut");
    for level in 0..4 {
        name = name
            .replace(
                &format!("decoder.up.{level}.block."),
                &format!("decoder.up_blocks.{}.resnets.", 3 - level),
            )
            .replace(
                &format!("decoder.up.{level}.upsample.conv"),
                &format!("decoder.up_blocks.{}.upsamplers.0.conv", 3 - level),
            );
    }

    let preparation = if shape.len() == 4 {
        let mut b = Builder::new(1, 0, 0);
        let x = if name.contains(".attentions.") {
            ensure!(
                shape[2] == 1 && shape[3] == 1,
                "expected pointwise VAE attention projection"
            );
            b.one(
                Kernel::Tensor(T::Reshape(vec![shape[0], shape[1]])),
                vec![Value::Input(0)],
            )
        } else {
            let x = b.one(
                Kernel::Tensor(T::Transpose(vec![0, 2, 3, 1])),
                vec![Value::Input(0)],
            );
            b.one(Kernel::Tensor(T::Contiguous), vec![x])
        };
        Some(b.finish(vec![x]))
    } else {
        None
    };
    Ok((name, preparation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_reproducible_and_schedule_reaches_zero() {
        let a = normal_noise(101, 42);
        assert_eq!(a, normal_noise(101, 42));
        assert_ne!(a, normal_noise(101, 43));
        assert!(a.iter().all(|x| x.is_finite()));
        for steps in [1, 4, 50] {
            for tokens in [16, 1024, 4096] {
                let s = schedule2(tokens, steps);
                assert_eq!(s.len(), steps + 1);
                assert_eq!(s[0], 1.0);
                assert_eq!(s[steps], 0.0);
                assert!(s.windows(2).all(|pair| pair[0] > pair[1]));
            }
        }
    }
}
