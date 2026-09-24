//! Hyper-connections: the 4-stream gated residual mixer.

use lisa_mlx::{ops, Array, Dtype};

use crate::core::norm::RmsNorm;
use crate::core::loader::TensorSource;
use lisa_mlx::ops::indexing::IndexOp;
use crate::core::quant::QuantizedLinear;

/// Gated residual mixer for the hyper-connection stream.
///
/// The residual carries `hc_count` streams side by side. Before a block the
/// mixer normalizes each stream, builds a low-rank gate over all of them, and
/// averages them into one width-`hidden` input. After the block the output is
/// injected back into each stream with its own weight.
///
/// The tower's last mixer runs without an inject head; it is what stands in
/// for the final `norm` tensor.
#[derive(Clone)]
pub struct GatedResidual {
    pub hc_norm: RmsNorm,
    pub mix_down: QuantizedLinear,
    pub mix_up: QuantizedLinear,
    pub decode_up: Option<QuantizedLinear>,
    pub block_inject: Option<QuantizedLinear>,
    pub hc_count: usize,
    pub hidden: usize,
}

impl GatedResidual {
    pub fn load<S: TensorSource>(src: &mut S, prefix: &str, hidden: usize, hc_count: usize, use_inject: bool) -> anyhow::Result<Self> {
        let wide = hidden * hc_count;
        let mix_up = QuantizedLinear::load(src, prefix, "input_mix_weight_up")?;
        // OPT-PACKEDUP: reorder the up rows so each decode tile's eight rows
        // are contiguous (for each hidden pair, the four HC streams), letting
        // the mixer walk `qmv_reg` instead of the strided `qmv_reg_rows`.
        let decode_up = if hc_count == 4 && hidden % 2 == 0 && mix_up.weight.dim(0) as usize == wide {
            let base = hidden as i32;
            let mut order: Vec<i32> = Vec::with_capacity(wide);
            let mut d = 0i32;
            while d < base {
                for s in 0..hc_count as i32 {
                    order.push(s * base + d);
                    order.push(s * base + d + 1);
                }
                d += 2;
            }
            let idx = Array::from_slice(&order, &[wide as i32]);
            let w = mix_up.weight.take_axis(&idx, 0)?;
            let sc = mix_up.scales.take_axis(&idx, 0)?;
            let bi = mix_up.biases.take_axis(&idx, 0)?;
            Some(QuantizedLinear { weight: w, scales: sc, biases: bi })
        } else {
            None
        };
        Ok(Self {
            hc_norm: RmsNorm::load(src, &format!("{prefix}.hc_norm"), 1e-6, Some(hidden))?,
            mix_down: QuantizedLinear::load(src, prefix, "input_mix_weight_down")?,
            mix_up,
            decode_up,
            block_inject: if use_inject {
                Some(QuantizedLinear::load(src, prefix, "block_inject_weight")?)
            } else {
                None
            },
            hc_count,
            hidden: wide,
        })
    }

    pub fn mixed_from_normed(&self, normed: &Array) -> lisa_mlx::error::Result<Array> {
        self.mixed(normed, false)
    }

    /// hc_norm with the 1/hc_count pre-multiplied into the scale (the fast
    /// path's exact trick: the mix sum then equals the reference mean).
    pub fn hc_norm_quarter(&self, hyper: &Array) -> lisa_mlx::error::Result<Array> {
        let scaled = &self.hc_norm.weight * crate::core::norm::bf16_scalar(1.0 / self.hc_count as f32);
        let h = self.hc_norm.group_size.unwrap_or_else(|| self.hidden / self.hc_count);
        crate::core::norm::rms_row_exact(hyper, Some(&scaled), self.hc_norm.eps, h)
    }

    /// The fast-kernel mix (upMix): the normed arrives PRE-QUARTERED (the
    /// /hc_count baked into the norm scale), silu is bf16-stepped, and the
    /// stream fold is a SEQUENTIAL BF16 accumulation of bf16 products.
    fn mixed(&self, normed: &Array, mix_dump: bool) -> lisa_mlx::error::Result<Array> {
        let mean_form = true;
        let dump = |name: &str, arr: &Array| {
            if mix_dump {
                if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
                    let f = arr.as_dtype(Dtype::Float32).unwrap();
                    let a = f.as_slice::<f32>();
                    let _ = std::fs::write(
                        format!("{dir}/rs_{name}.bin"),
                        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                    );
                }
            }
        };
        dump("mix_normed", normed);
        let lo = self.mix_down.forward(normed)?;
        dump("mix_down", &lo);
        let w = if mean_form {
            let lo = lo / crate::core::norm::bf16_scalar(self.hc_count as f32);
            lisa_mlx::nn::silu(&lo)?
        } else {
            crate::core::norm::bf16_silu(&lo)?
        };
        dump("mix_act", &w);
        let w = self.mix_up.forward(&w)?;
        dump("mix_up", &w);
        let w = ops::sigmoid(&w)?;
        dump("mix_w", &w);
        let mut lead = w.shape().to_vec();
        lead.pop();
        let hc = self.hc_count as i32;
        let h = (self.hidden / self.hc_count) as i32;
        lead.push(hc);
        lead.push(h);
        let w4 = w.reshape(&lead)?;
        let n4 = normed.reshape(&lead)?;
        if mean_form {
            return (w4 * n4).mean_axis(-2, None);
        }
        let prod = w4.multiply(&n4)?;
        dump("mix_prod", &prod);
        let mut acc = prod.index((.., .., 0, ..)).contiguous().unwrap();
        for s in 1..self.hc_count {
            acc = acc.add(&prod.index((.., .., s as i32, ..)))?;
        }
        dump("mix_acc", &acc);
        Ok(acc)
    }

    /// The hc_norm scale pre-divided by `hc_count` (the engine's `normScaleQ`).
    pub fn norm_scale_q(&self) -> lisa_mlx::error::Result<Array> {
        Ok(&self.hc_norm.weight * crate::core::norm::bf16_scalar(1.0 / self.hc_count as f32))
    }

    /// The engine's `hcMix` over a normed produced by `inject_norm`: returns
    /// `(block_input, inject_weights)`. `has_inject=false` is the final mixer.
    pub fn mix_from_normed(
        &self,
        normed: &Array,
        has_inject: bool,
    ) -> lisa_mlx::error::Result<(Array, Array)> {
        let hc = self.hc_count as i32;
        let h = (self.hidden / self.hc_count) as i32;
        let (b, s) = (normed.dim(0), normed.dim(1));
        let stream = lisa_mlx::Stream::thread_local_or_default();
        // Narrow windows (S <= 8): the engine's fused mixer pair
        // (`track_mixer_down_inject` + `track_mixer_up_mix`). Those kernels are
        // single-batch, so a batched window takes the generic path.
        if b == 1 && s <= 8 {
            let inj_q = self
                .block_inject
                .as_ref()
                .map(|i| (&i.weight, &i.scales, &i.biases));
            let nd = self.mix_down.weight.dim(0);
            let (wi, si, bi) = inj_q.unwrap_or((&self.mix_down.weight, &self.mix_down.scales, &self.mix_down.biases));
            if let Some((_lo, act, injw)) = lisa_mlx::moe_decode::down_inject(
                normed,
                &self.mix_down.weight, &self.mix_down.scales, &self.mix_down.biases,
                wi, si, bi,
                self.hidden as i32, nd, hc, s, has_inject, &stream,
            ) {
                if let Some((input, inject_w)) = {
                    let packed = if s == 1 { self.decode_up.as_ref() } else { None };
                    let (wu, su, bu) = match packed {
                        Some(p) => (&p.weight, &p.scales, &p.biases),
                        None => (&self.mix_up.weight, &self.mix_up.scales, &self.mix_up.biases),
                    };
                    lisa_mlx::moe_decode::up_mix(
                        &act, normed, wu, su, bu, &injw,
                        h, hc, s, has_inject, packed.is_some(), &stream,
                    )
                } {
                    return Ok((input.reshape(&[b, s, h])?, inject_w.reshape(&[b, s, hc])?));
                }
            }
        }
        let inj = match &self.block_inject {
            Some(inj) => inj.forward(normed)?,
            None => normed.clone(),
        };
        let lo = self.mix_down.forward(normed)?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            static H0: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
            if normed.dim(1) == 1024 && H0.swap(false, std::sync::atomic::Ordering::SeqCst) {
                for (nm, arr) in [("hx_inj", &inj), ("hx_lo", &lo)] {
                    let f = arr.as_dtype(lisa_mlx::Dtype::Float32)?;
                    let a = f.as_slice::<f32>();
                    let _ = std::fs::write(format!("{dir}/{nm}.bin"),
                        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) });
                }
            }
        }
        let act = match lisa_mlx::kernels::silu_head(&lo, lo.dim(-1), &stream) {
            Some(r) => r.reshape(lo.shape())?,
            None => crate::core::norm::bf16_silu(&lo)?,
        };
        let w = self.mix_up.forward(&act)?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            static H1: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
            if normed.dim(1) == 1024 && H1.swap(false, std::sync::atomic::Ordering::SeqCst) {
                for (nm, arr) in [("hx_act", &act), ("hx_w", &w)] {
                    let f = arr.as_dtype(lisa_mlx::Dtype::Float32)?;
                    let a = f.as_slice::<f32>();
                    let _ = std::fs::write(format!("{dir}/{nm}.bin"),
                        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) });
                }
            }
        }
        if let Some((input, inject_w)) =
            lisa_mlx::kernels::hc_mix(&w, normed, &inj, hc, h, b * s, has_inject, &stream)
        {
            return Ok((input.reshape(&[b, s, h])?, inject_w.reshape(&[b, s, hc])?));
        }
        let normed4 = normed.reshape(&[b, s, hc, h])?;
        let w4 = w.reshape(&[b, s, hc, h])?;
        let prod = (w4 * normed4).mean_axis(-2, None)?;
        let inject_w = inj.reshape(&[b, s, hc])? * crate::core::norm::bf16_scalar(2.0);
        Ok((prod, inject_w))
    }

    /// Returns `(block_input, residual, inject_weights)`.
    pub fn mix_with_inject(&self, hyper: &Array) -> lisa_mlx::error::Result<(Array, Array, Array)> {
        static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        let dump_dir = std::env::var("LISA_DUMP_DIR").ok();
        let first = FIRST.swap(false, std::sync::atomic::Ordering::SeqCst) && dump_dir.is_some();
        let dump = |name: &str, arr: &Array| {
            if first {
                let dir = dump_dir.as_ref().unwrap();
                let f = arr.as_dtype(Dtype::Float32).unwrap();
                let a = f.as_slice::<f32>();
                let _ = std::fs::write(
                    format!("{dir}/rsx_{name}.bin"),
                    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                );
            }
        };
        let inject = self
            .block_inject
            .as_ref()
            .expect("mix_with_inject on a mixer without an inject head");
        let mean_form = true;
        if mean_form {
            // Engine fast path: quartered norm, one fused `track_hc_mix` launch
            // (sigmoid + bf16 stream fold + the inject in a single kernel).
            let h = (self.hidden / self.hc_count) as i32;
            let hc = self.hc_count as i32;
            let normed = self.hc_norm_quarter(hyper)?;
            let inj = inject.forward(&normed)?;
            let lo = self.mix_down.forward(&normed)?;
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let act = match lisa_mlx::kernels::silu_head(&lo, lo.dim(-1), &stream) {
                Some(r) => r.reshape(lo.shape())?,
                None => crate::core::norm::bf16_silu(&lo)?,
            };
            let w = self.mix_up.forward(&act)?;
            let (b, s) = (hyper.dim(0), hyper.dim(1));
            if let Some((input, inject_w)) =
                lisa_mlx::kernels::hc_mix(&w, &normed, &inj, hc, h, b * s, true, &stream)
            {
                let block_input = input.reshape(&[b, s, h])?;
                let inject_w = inject_w.reshape(&[b, s, hc])?;
                dump("inject_w", &inject_w);
                dump("block_input", &block_input);
                return Ok((block_input, hyper.clone(), inject_w));
            }
        }
        let normed = if mean_form {
            self.hc_norm.forward(hyper)?
        } else {
            self.hc_norm_quarter(hyper)?
        };
        let x = inject.forward(&normed)?;
        let x = if mean_form {
            x / crate::core::norm::bf16_scalar(self.hc_count as f32)
        } else {
            x
        };
        let inject_w = ops::sigmoid(&x)? * crate::core::norm::bf16_scalar(2.0);
        dump("inject_w", &inject_w);
        let block_input = self.mixed(&normed, first)?;
        dump("block_input", &block_input);
        Ok((block_input, hyper.clone(), inject_w))
    }
}

/// Inject a block output back into the hyper-connection stream.
///
/// output: [B, S, hidden]; inject: [B, S, hc]; returns residual + spread.
pub fn inject(
    residual: &Array,
    output: &Array,
    inject: &Array,
) -> lisa_mlx::error::Result<Array> {
    let spread = output.expand_dims(-2)?.multiply(&inject.expand_dims(-1)?)?;
    // [B, S, hc, hidden] -> [B, S, hc * hidden] (stream-major, like the ref)
    let mut shape = spread.shape().to_vec();
    shape.pop();
    shape.pop();
    shape.push(-1);
    let flat = spread.reshape(&shape)?;
    residual.add(&flat)
}
