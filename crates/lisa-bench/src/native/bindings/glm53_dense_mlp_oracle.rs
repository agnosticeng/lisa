#![cfg(test)]
#![allow(dead_code)]
//! Frozen dense/shared expert orchestration before extraction.
use crate::bindings::glm53_oracle_helpers::narrow;
use crate::core::quant::QuantizedLinearExecution;
use crate::core::{capture::Capture, loader::TensorSource, quant::QuantizedLinear};
use lisa_metal::{Array, Dtype, ops};
/// Concatenate gate and up output rows so one affine-quant projection computes
/// both results. Real layer-0 weights match the separate DS4 production path
/// for decode and prefill shapes; the layer-45 shared expert has the same
/// checkpoint-backed exactness check.
pub fn fuse_gate_up(
    gate: &QuantizedLinear,
    up: &QuantizedLinear,
) -> anyhow::Result<QuantizedLinear> {
    debug_assert_eq!(gate.group_size, up.group_size);
    debug_assert_eq!(gate.bits, up.bits);
    let w = ops::concatenate(&[&gate.weight, &up.weight], 0)?;
    let s = ops::concatenate(&[&gate.scales, &up.scales], 0)?;
    let b = ops::concatenate(&[&gate.biases, &up.biases], 0)?;
    Ok(QuantizedLinear {
        weight: w,
        scales: s,
        biases: b,
        group_size: gate.group_size,
        bits: gate.bits,
    })
}

/// Split a fused gate_up projection output [.., 2*inter] into (gate, up).
pub fn split_gate_up(fused_out: &Array, inter: usize) -> anyhow::Result<(Array, Array)> {
    let n = fused_out.dim(-1) as usize;
    debug_assert_eq!(n, 2 * inter);
    let g = narrow(fused_out, fused_out.rank() - 1, 0, inter)?;
    let u = narrow(fused_out, fused_out.rank() - 1, inter, inter)?;
    Ok((g, u))
}

/// Clamped SwiGLU: `silu(min(gate, limit)) * clip(up, -limit, limit)`.
pub fn clamped_swiglu(up: &Array, gate: &Array, limit: f32) -> anyhow::Result<Array> {
    let gc = gate.contiguous()?;
    let uc = up.contiguous()?;
    let stream = lisa_metal::Stream::thread_local_or_default();
    lisa_metal::kernels::sparse::kernels::swiglu_fused(&gc, &uc, limit, &stream)
        .ok_or_else(|| anyhow::anyhow!("GLM fused clamped-SwiGLU Metal kernel failed"))
}

/// The paired Q8 projection is the production path. The separate mode keeps
/// the original two-dispatch schedule available for controlled end-to-end
/// measurements; it is selected once while loading the model.
pub(super) enum GateUpProjection {
    Paired {
        projection: QuantizedLinear,
        gate: QuantizedLinear,
        up: QuantizedLinear,
    },
    Separate {
        gate: QuantizedLinear,
        up: QuantizedLinear,
    },
}

struct ActivatedProjection {
    projections: Option<(Array, Array)>,
    activated: Array,
}

impl GateUpProjection {
    fn load_mode() -> anyhow::Result<bool> {
        match std::env::var("LISA_GLM53_Q8_GATE_UP").as_deref() {
            Ok("separate") => Ok(false),
            Ok("paired") | Err(std::env::VarError::NotPresent) => Ok(true),
            Ok(value) => {
                anyhow::bail!("LISA_GLM53_Q8_GATE_UP must be `paired` or `separate`, got {value:?}")
            }
            Err(error) => anyhow::bail!("cannot read LISA_GLM53_Q8_GATE_UP: {error}"),
        }
    }

    pub(super) fn from_parts(
        gate: QuantizedLinear,
        up: QuantizedLinear,
        paired: bool,
    ) -> anyhow::Result<Self> {
        if paired {
            let projection = fuse_gate_up(&gate, &up)?;
            let (gate, up) = split_gate_up_linear(&projection, projection.dims_out() / 2)?;
            Ok(Self::Paired {
                projection,
                gate,
                up,
            })
        } else {
            Ok(Self::Separate { gate, up })
        }
    }

    fn inter(&self) -> usize {
        match self {
            Self::Paired { projection, .. } => projection.dims_out() / 2,
            Self::Separate { gate, .. } => gate.dims_out(),
        }
    }

    fn split_linears(&self) -> (&QuantizedLinear, &QuantizedLinear) {
        match self {
            Self::Paired { gate, up, .. } | Self::Separate { gate, up } => (gate, up),
        }
    }

    fn forward(&self, x: &Array) -> anyhow::Result<(Array, Array)> {
        match self {
            Self::Paired { projection, .. } => {
                let output = projection.forward_glm53_prefill(x)?;
                let inter = projection.dims_out() / 2;
                let n = output.dim(-1) as usize;
                let (gate, up) = split_gate_up(&output, inter)?;
                debug_assert_eq!(up.dim(-1) as usize, n - inter);
                Ok((gate, up))
            }
            Self::Separate { gate, up } => {
                Ok((gate.forward_glm53_prefill(x)?, up.forward_glm53_prefill(x)?))
            }
        }
    }

    fn forward_native_activated(
        &self,
        x: &Array,
        limit: f32,
        capture_projections: bool,
    ) -> anyhow::Result<Option<ActivatedProjection>> {
        if !crate::bindings::glm53_affine_moe::native_gate_up_enabled()
            || matches!(self, Self::Separate { .. })
        {
            return Ok(None);
        }
        let shape = x.shape();
        let rows = shape[..shape.len() - 1].iter().product::<i32>() as usize;
        let k = x.dim(-1) as usize;
        if !(1..=16).contains(&rows)
            || k % 128 != 0
            || self.inter() % 2 != 0
            || x.dtype() != Dtype::Float32
        {
            return Ok(None);
        }
        let (gate, up) = self.split_linears();
        if [gate, up]
            .iter()
            .any(|projection| projection.bits != 8 || projection.group_size != 64)
        {
            return Ok(None);
        }
        let flat = x.reshape(&[rows as i32, k as i32])?.contiguous()?;
        // Gate/up are contiguous output-row views of the already-owned
        // projection storage. No repack or new weight allocation occurs here.
        for tensor in [
            &flat,
            &gate.weight,
            &gate.scales,
            &gate.biases,
            &up.weight,
            &up.scales,
            &up.biases,
        ] {
            tensor.eval_enqueue()?;
        }
        let output_shape = [&shape[..shape.len() - 1], &[self.inter() as i32]].concat();
        if !capture_projections {
            let activated = lisa_metal::jit::glm53_affine_q8_swiglu_native(
                flat.t.device(),
                &flat.t,
                &gate.weight.t,
                &gate.scales.t,
                &gate.biases.t,
                &up.weight.t,
                &up.scales.t,
                &up.biases.t,
                limit,
            )?;
            let activated = Array::new(activated).reshape(&output_shape)?;
            return Ok(Some(ActivatedProjection {
                projections: None,
                activated,
            }));
        }
        let (g, u, activated) = lisa_metal::jit::glm53_affine_q8_gate_up_native(
            flat.t.device(),
            &flat.t,
            &gate.weight.t,
            &gate.scales.t,
            &gate.biases.t,
            &up.weight.t,
            &up.scales.t,
            &up.biases.t,
            limit,
        )?;
        Ok(Some(ActivatedProjection {
            projections: Some((
                Array::new(g).reshape(&output_shape)?,
                Array::new(u).reshape(&output_shape)?,
            )),
            activated: Array::new(activated).reshape(&output_shape)?,
        }))
    }
}

/// Clamped GLM MLP. Leading dense layers use a concatenated gate/up Q8
/// projection. Shared experts use the same exact paired-projection schedule.
pub struct DenseMlp {
    pub(super) gate_up: GateUpProjection,
    pub(super) down: QuantizedLinear,
    pub(super) limit: f32,
}

impl DenseMlp {
    /// The projection mode captured at load, independent of later environment changes.
    pub(super) fn paired_storage(&self) -> bool {
        matches!(&self.gate_up, GateUpProjection::Paired { .. })
    }

    pub fn load<S: TensorSource>(w: &mut S, prefix: &str, limit: f32) -> anyhow::Result<Self> {
        let (gate, up, down) = Self::load_parts(w, prefix)?;
        Ok(Self {
            gate_up: GateUpProjection::from_parts(gate, up, GateUpProjection::load_mode()?)?,
            down,
            limit,
        })
    }

    fn load_shared<S: TensorSource>(w: &mut S, prefix: &str, limit: f32) -> anyhow::Result<Self> {
        Self::load::<S>(w, prefix, limit)
    }

    fn load_parts<S: TensorSource>(
        w: &mut S,
        prefix: &str,
    ) -> anyhow::Result<(QuantizedLinear, QuantizedLinear, QuantizedLinear)> {
        let mut gate = QuantizedLinear::load(w, &format!("{prefix}.gate_proj"), "")?;
        let mut up = QuantizedLinear::load(w, &format!("{prefix}.up_proj"), "")?;
        let mut down = QuantizedLinear::load(w, &format!("{prefix}.down_proj"), "")?;
        for q in [&mut gate, &mut up, &mut down] {
            q.set_quant(64, 8);
            if q.scales.dtype() != Dtype::Bfloat16 {
                q.scales = q.scales.as_dtype(Dtype::Float32)?;
                q.biases = q.biases.as_dtype(Dtype::Float32)?;
            }
        }
        Ok((gate, up, down))
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        self.forward_with_capture(x, None, 0)
    }

    pub fn forward_with_capture(
        &self,
        x: &Array,
        capture: Option<&Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dense_mlp_input"), x);
        }
        // Logit-only captures must use the activation-only production path.
        let projection_capture =
            capture.filter(|capture| capture.accepts_key(&format!("L{layer}.dense_gate_up")));
        let native =
            self.gate_up
                .forward_native_activated(x, self.limit, projection_capture.is_some())?;
        let ActivatedProjection {
            projections,
            activated,
        } = if let Some(native) = native {
            native
        } else {
            let (gate, up) = self.gate_up.forward(x)?;
            let activated = clamped_swiglu(&up, &gate, self.limit)?;
            ActivatedProjection {
                projections: Some((gate, up)),
                activated,
            }
        };
        if let Some(capture) = projection_capture {
            let (gate, up) = projections
                .as_ref()
                .expect("capture requested projection outputs");
            let gate_up = ops::concatenate(&[gate, up], -1)?;
            capture.store(&format!("L{layer}.dense_gate_up"), &gate_up);
        }
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dense_swiglu"), &activated);
        }
        let out = self.forward_projection(&self.down, &activated)?;
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dense_out"), &out);
        }
        Ok(out)
    }

    /// Use DS4's row-count-specific Q8 projection schedule over the original
    /// affine storage: M=1, small-row and matrix paths keep their own arithmetic.
    fn forward_projection(
        &self,
        projection: &QuantizedLinear,
        x: &Array,
    ) -> lisa_metal::error::Result<Array> {
        projection.forward_glm53_prefill(x)
    }
}

fn split_gate_up_linear(
    gate_up: &QuantizedLinear,
    inter: usize,
) -> anyhow::Result<(QuantizedLinear, QuantizedLinear)> {
    let split = |array: &Array, start: usize| -> anyhow::Result<Array> {
        Ok(narrow(array, 0, start, inter)?.contiguous()?)
    };
    let make = |start: usize| -> anyhow::Result<QuantizedLinear> {
        Ok(QuantizedLinear {
            weight: split(&gate_up.weight, start)?,
            scales: split(&gate_up.scales, start)?,
            biases: split(&gate_up.biases, start)?,
            group_size: gate_up.group_size,
            bits: gate_up.bits,
        })
    };
    Ok((make(0)?, make(inter)?))
}
