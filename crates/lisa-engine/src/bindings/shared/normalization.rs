//! Normalization and rotary bindings over explicit engine execution contexts.
use crate::{
    bindings::BindingDevice,
    core::graph::PlanCache,
    execution::DeviceContext,
    tensor_ops::{TensorLaw, TensorOps},
    weights::source::TensorSource,
};
use lisa_backend::{Binary, Outputs, ScalarType, TensorAccess, TensorOperation, Unary};
use std::cell::RefCell;
/// RMSNorm with the checkpoint's baked weight-offset convention
/// (`rms_norm_weight_offset == 0` for this checkpoint: the stored weight IS
/// the scale). With `group_size` set, each group is normalized on its own
/// statistic and the flat scale applies after the normalized rounding.
#[derive(Clone)]
pub struct RmsNorm<T> {
    pub weight: T,
    pub eps: f32,
    pub group_size: Option<usize>,
}

impl<T: TensorAccess> RmsNorm<T> {
    pub fn load_on<D: BindingDevice<Tensor = T>, S: TensorSource<T>>(
        context: &DeviceContext<D>,
        src: &mut S,
        name: &str,
        eps: f32,
        group_size: Option<usize>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_bf16(context, src, &format!("{name}.weight"))?,
            eps,
            group_size,
        })
    }

    pub fn forward_on<D: BindingDevice<Tensor = T>>(
        &self,
        context: &DeviceContext<D>,
        x: &D::Tensor,
    ) -> anyhow::Result<D::Tensor> {
        match self.group_size {
            None => run_plan(
                context,
                PlanKey::Rms(self.eps.to_bits(), false),
                &[x, &self.weight],
                || lisa_models::normalization::rms(self.eps, false),
            )?
            .into_one()
            .map_err(anyhow::Error::msg),
            Some(g) => {
                // The engine's rms_single_row reduction (pinned association).
                rms_row_exact(context, x, Some(&self.weight), self.eps, g)
            }
        }
    }

    /// Fused residual-add + RMSNorm: `(bf16(x + r), normed)` in
    /// one dispatch. Bit-exact against `x.add(r)` → `self.forward(&sum)` on
    /// the plain `fast::rms_norm` path (the kernel mirrors rms_norm.metal and
    /// rounds the sum before the reduction); other paths fall back to the
    /// composed pair.
    pub fn forward_residual_add_on<D: BindingDevice<Tensor = T>>(
        &self,
        context: &DeviceContext<D>,
        x: &D::Tensor,
        r: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        match self.group_size {
            None => {
                let output = run_plan(
                    context,
                    PlanKey::Rms(self.eps.to_bits(), true),
                    &[x, r, &self.weight],
                    || lisa_models::normalization::rms(self.eps, true),
                )?;
                let mut output = output.into_iter();
                Ok((
                    output.next().expect("validated sum output"),
                    output.next().expect("validated norm output"),
                ))
            }
            Some(_) => {
                let sum = TensorOps::new(context.device(), TensorLaw::Native)
                    .op(TensorOperation::Binary(Binary::Add), &[x, r])?;
                let normed = self.forward_on(context, &sum)?;
                Ok((sum, normed))
            }
        }
    }
}

/// Partial rotary embedding evaluated at explicit positions.
///
/// cos/sin are computed ON DEVICE from two integers so the arithmetic is exactly
/// MLX's f32 exp/cos, not libm.
pub fn bf16_scalar<D: BindingDevice>(
    context: &DeviceContext<D>,
    v: f32,
) -> anyhow::Result<D::Tensor> {
    // Host-built bf16 scalar (no GPU round trip): `from_f32().as_dtype(BF16)`
    // hits `cast_host`, which evals the device (a full GPU sync per layer per
    // token — the dominant decode cost).
    TensorOps::new(context.device(), TensorLaw::Native).pull(
        &half::bf16::from_f32(v).to_bits().to_ne_bytes(),
        &[],
        ScalarType::BF16,
    )
}

/// The engine's EXACT row RMS reduction (rms_single_row / injectNorm layout):
///
/// Channels are dealt to 20 simdgroups x 32 lanes x 4 consecutive elements
/// (per 2560-channel stream; generalizes to any H % 128 == 0). Each lane
/// accumulates its 4 squares sequentially in f32, each simdgroup folds its 32
/// lanes with the hardware xor-butterfly (pairwise tree), the per-simdgroup
/// partials are zero-padded to 32 and folded with a second butterfly, and
/// `inv_mean = precise::rsqrt(total / H + eps)`. The output is
/// `bf16(x * inv_mean) * scale` — the weight applied AFTER the bf16 rounding.
///
/// This pins the reduction association independently of the MLX build.
pub fn rms_row_exact<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    scale: Option<&D::Tensor>,
    eps: f32,
    h: usize,
) -> anyhow::Result<D::Tensor> {
    grouped_rms_plan(context, x, scale, eps, h, false)
}
pub fn grouped_rms_plan<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    scale: Option<&D::Tensor>,
    eps: f32,
    h: usize,
    force_operations: bool,
) -> anyhow::Result<D::Tensor> {
    let dtype = dtype(context, x)?;
    let mut inputs = vec![x];
    if let Some(w) = scale {
        inputs.push(w);
    }
    plans(context, |cache| {
        cache
            .borrow_mut()
            .try_get_or_prepare(
                PlanKey::GroupedRms(
                    x.shape().to_vec(),
                    dtype,
                    eps.to_bits(),
                    h,
                    scale.is_some(),
                    force_operations,
                ),
                || {
                    lisa_models::normalization::grouped_rms(
                        x.shape(),
                        dtype,
                        eps,
                        h,
                        scale.is_some(),
                        force_operations,
                    )
                },
            )?
            .run(&inputs, &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    })
    .map_err(|e: anyhow::Error| anyhow::Error::msg(e.to_string()))
}

pub fn bf16_silu<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    TensorOps::new(context.device(), TensorLaw::Native)
        .op(TensorOperation::Unary(Unary::Silu), &[x])
}

/// The fork's `MLXNN.softplus` / `logaddexp(x, 0)` on bf16 rounds through
/// bf16 at every step of the op chain.
pub fn bf16_logaddexp0<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    run_plan(
        context,
        PlanKey::LogAddExp0,
        &[x],
        lisa_models::normalization::bf16_logaddexp0,
    )?
    .into_one()
    .map_err(anyhow::Error::msg)
}

/// Positions of `count` tokens starting at `offset`, shaped `[1, count]`.
/// Conventional RMSNorm with an optional sigmoid output gate, used by the
/// gated deltanet. The weight is a plain scale.
#[derive(Clone)]
pub struct RmsNormGated<T> {
    pub weight: T,
    pub eps: f32,
}

impl<T: TensorAccess> RmsNormGated<T> {
    pub fn load_on<D: BindingDevice<Tensor = T>, S: TensorSource<T>>(
        context: &DeviceContext<D>,
        src: &mut S,
        name: &str,
        eps: f32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_bf16(context, src, &format!("{name}.weight"))?,
            eps,
        })
    }

    pub fn forward_on<D: BindingDevice<Tensor = T>>(
        &self,
        context: &DeviceContext<D>,
        x: &D::Tensor,
        gate: &D::Tensor,
    ) -> anyhow::Result<D::Tensor> {
        let dtype = dtype(context, x)?;
        run_plan(
            context,
            PlanKey::Gated(self.eps.to_bits(), dtype),
            &[x, gate, &self.weight],
            || lisa_models::normalization::gated_rms(self.eps, dtype),
        )?
        .into_one()
        .map_err(anyhow::Error::msg)
    }
}

/// Partial rotary embedding evaluated at explicit positions.
#[derive(Clone)]
pub struct Rotary {
    pub dimensions: usize,
    pub base: f32,
}

impl Rotary {
    pub fn new(dimensions: usize, base: f32) -> Self {
        Self { dimensions, base }
    }

    /// cos/sin for `positions` (any shape), each `positions.shape + [dims]`.
    pub fn cos_sin<D: BindingDevice>(
        &self,
        context: &DeviceContext<D>,
        positions: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.frequencies(context, positions, false)
    }

    /// The final axis selects a temporal, height or width position for each
    /// rotary frequency. All other arithmetic matches the scalar text path.
    pub fn cos_sin_multimodal<D: BindingDevice>(
        &self,
        context: &DeviceContext<D>,
        positions: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.frequencies(context, positions, true)
    }
    fn frequencies<D: BindingDevice>(
        &self,
        context: &DeviceContext<D>,
        positions: &D::Tensor,
        multimodal: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let key = PlanKey::Frequencies(
            self.dimensions,
            self.base.to_bits(),
            positions.shape().to_vec(),
            multimodal,
        );
        let output = plans(context, |cache| {
            cache
                .borrow_mut()
                .try_get_or_prepare(key, || {
                    lisa_models::position::frequencies(
                        self.dimensions,
                        self.base,
                        positions.shape(),
                        multimodal,
                    )
                })?
                .run(&[positions], &mut [])
        })
        .map_err(|e: anyhow::Error| anyhow::Error::msg(e.to_string()))?;
        let mut output = output.into_iter();
        Ok((
            output.next().expect("validated cosine output"),
            output.next().expect("validated sine output"),
        ))
    }
}

/// Rotate only the leading `cos.dim(-1)` entries of the last axis.
pub fn rope_partial<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    cos: &D::Tensor,
    sin: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    if x.rank() == 0 || cos.rank() == 0 || sin.shape() != cos.shape() {
        return Err(anyhow::Error::msg("invalid rotary input shape"));
    }
    let d = cos.dim(-1);
    let width = x.dim(-1);
    let dtype = dtype(context, x)?;
    if d <= 0 || d % 2 != 0 || d > width {
        return Err(anyhow::Error::msg("invalid rotary dimension"));
    }
    run_plan(
        context,
        PlanKey::Rotary(d, width, dtype),
        &[x, cos, sin],
        || lisa_models::position::partial_rotary(d, width, dtype),
    )?
    .into_one()
    .map_err(anyhow::Error::msg)
}

/// Positions of `count` tokens starting at `offset`, shaped `[1, count]`.
///
/// Built as f32 so RoPE does not need an i32->f32 cast (Metal
/// `to_dtype` lacks that pair and falls back to a host round-trip that
/// synchronizes the GPU once per attention layer).
pub fn positions<D: BindingDevice>(
    context: &DeviceContext<D>,
    offset: usize,
    count: usize,
) -> anyhow::Result<D::Tensor> {
    let data: Vec<f32> = (offset..offset + count).map(|i| i as f32).collect();
    TensorOps::new(context.device(), TensorLaw::Native).pull(
        &data
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
        &[1, count as i32],
        ScalarType::F32,
    )
}

#[derive(PartialEq)]
enum PlanKey {
    Frequencies(usize, u32, Vec<i32>, bool),
    LogAddExp0,
    GroupedRms(Vec<i32>, ScalarType, u32, usize, bool, bool),
    Rms(u32, bool),
    Gated(u32, ScalarType),
    Rotary(i32, i32, ScalarType),
}

fn plans<D: BindingDevice, R>(
    context: &DeviceContext<D>,
    f: impl FnOnce(&RefCell<PlanCache<D, PlanKey>>) -> R,
) -> R {
    context.with_slot(
        "normalization.plans",
        || RefCell::new(PlanCache::new(context.device().clone(), 16)),
        f,
    )
}
fn run_plan<D: BindingDevice>(
    context: &DeviceContext<D>,
    key: PlanKey,
    inputs: &[&D::Tensor],
    build: impl FnOnce() -> lisa_models::graph::Graph,
) -> anyhow::Result<Outputs<D::Tensor>> {
    plans(context, |cache| {
        cache
            .borrow_mut()
            .get_or_prepare(key, build)?
            .run(inputs, &mut [])
    })
}
fn dtype<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<ScalarType> {
    context
        .device()
        .dtype(x)
        .ok_or_else(|| anyhow::anyhow!("unsupported normalization dtype"))
}
pub fn load_bf16<D: BindingDevice, S: TensorSource<D::Tensor>>(
    context: &DeviceContext<D>,
    src: &mut S,
    name: &str,
) -> anyhow::Result<D::Tensor> {
    let x = src.get(name)?;
    if dtype(context, &x)? == ScalarType::BF16 {
        Ok(x)
    } else {
        TensorOps::new(context.device(), TensorLaw::Native)
            .op(TensorOperation::Cast(ScalarType::BF16), &[&x])
    }
}
