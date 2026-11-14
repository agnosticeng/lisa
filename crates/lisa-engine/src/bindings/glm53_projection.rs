use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
// Original GGUF blocks, dispatched without affine conversion.
use crate::core::graph::PlanCache;
use crate::weights::gguf::{PackedTensor, ResidentWeights};
use anyhow::{Context, Result, ensure};
use lisa_gguf::TensorType;

use lisa_backend::ScalarType;
use lisa_models::glm53::packed as model;

pub fn plain<D: BindingDevice, L>(
    weights: &ResidentWeights<D, L>,
    name: &str,
) -> Result<D::Tensor> {
    weights
        .tensors
        .get(name)
        .with_context(|| format!("missing GLM GGUF {name}"))?
        .plain_view()
}

pub enum ProjectionValue<T, P> {
    Plain(T),
    Packed(P),
}
pub type Projection<D> = ProjectionValue<<D as lisa_backend::Device>::Tensor, PackedTensor<D>>;
impl<T, P> From<T> for ProjectionValue<T, P> {
    fn from(value: T) -> Self {
        Self::Plain(value)
    }
}
impl<D: BindingDevice> std::ops::Deref for ProjectionValue<D::Tensor, PackedTensor<D>> {
    type Target = D::Tensor;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Plain(t) => t,
            Self::Packed(p) => p.packed_bytes(),
        }
    }
}
impl<D: BindingDevice> ProjectionValue<D::Tensor, PackedTensor<D>> {
    pub fn load<L>(w: &ResidentWeights<D, L>, name: &str) -> Result<Self> {
        let p = w
            .tensors
            .get(name)
            .with_context(|| format!("missing GLM GGUF {name}"))?;
        Ok(match p.info.kind {
            TensorType::F32 | TensorType::BF16 | TensorType::F16 => Self::Plain(plain(w, name)?),
            _ => Self::Packed(p.clone()),
        })
    }
    pub fn descriptor_on(&self, device: &D) -> Result<model::Projection> {
        let (kind, input, output) = match self {
            Self::Plain(w) => {
                ensure!(w.rank() == 2, "projection weight must be a matrix");
                (
                    if device.dtype(w) == Some(ScalarType::BF16) {
                        model::Kind::Bf16
                    } else {
                        model::Kind::Plain
                    },
                    w.dim(1),
                    w.dim(0),
                )
            }
            Self::Packed(w) => {
                ensure!(
                    w.info.dimensions.len() == 2,
                    "packed projection weight must be a matrix"
                );
                let kind = match w.info.kind {
                    TensorType::Q8_0 => model::Kind::Q8,
                    TensorType::Q4_K => model::Kind::Q4K,
                    _ => anyhow::bail!("unsupported GLM packed projection"),
                };
                (
                    kind,
                    i32::try_from(w.info.dimensions[0])?,
                    i32::try_from(w.info.dimensions[1])?,
                )
            }
        };
        Ok(model::Projection {
            kind,
            input,
            output,
        })
    }
    pub fn forward_on(&self, context: &DeviceContext<D>, x: &D::Tensor) -> Result<D::Tensor> {
        ensure!(
            context.device().dtype(x) == Some(ScalarType::F32),
            "GLM projection requires F32 activations"
        );
        let descriptor = self.descriptor_on(context.device())?;
        context.with_slot(
            "glm53.projection.plans",
            || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 64)),
            |plans| {
                plans
                    .borrow_mut()
                    .try_get_or_prepare((descriptor, x.shape().to_vec()), || {
                        descriptor.graph(x.shape())
                    })?
                    .run(&[self, x], &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)
            },
        )
    }
}
