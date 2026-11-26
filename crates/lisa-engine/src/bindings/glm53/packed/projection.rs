use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
// Original GGUF blocks, dispatched without affine conversion.
use crate::core::graph::PlanCache;
use crate::weights::prepared::{PreparedTensor, PreparedWeights};
use anyhow::{Context, Result, ensure};
use lisa_backend::PackedEncoding;
use lisa_models::packed::Projection as Encoding;

use lisa_backend::ScalarType;
use lisa_models::glm53::packed as model;

pub fn plain<T: Clone, L>(weights: &PreparedWeights<T, L>, name: &str) -> Result<T> {
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
pub type Projection<D> = ProjectionValue<
    <D as lisa_backend::Device>::Tensor,
    PreparedTensor<<D as lisa_backend::Device>::Tensor>,
>;
impl<T, P> From<T> for ProjectionValue<T, P> {
    fn from(value: T) -> Self {
        Self::Plain(value)
    }
}
impl<T: Clone> std::ops::Deref for ProjectionValue<T, PreparedTensor<T>> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Plain(t) => t,
            Self::Packed(p) => p.packed_bytes(),
        }
    }
}
impl<T: Clone + TensorAccess> ProjectionValue<T, PreparedTensor<T>> {
    pub fn load<L>(w: &PreparedWeights<T, L>, name: &str) -> Result<Self> {
        let p = w
            .tensors
            .get(name)
            .with_context(|| format!("missing GLM GGUF {name}"))?;
        Ok(match p.layout.projection {
            Encoding::Plain => Self::Plain(p.plain_view()?),
            _ => Self::Packed(p.clone()),
        })
    }
    pub fn descriptor_on<D: BindingDevice<Tensor = T>>(
        &self,
        device: &D,
    ) -> Result<model::Projection> {
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
                    w.layout.logical_shape.len() == 2,
                    "packed projection weight must be a matrix"
                );
                let kind = match w.layout.projection {
                    Encoding::Q8 => model::Kind::Q8,
                    Encoding::Block(PackedEncoding::Q4K) => model::Kind::Q4K,
                    _ => anyhow::bail!("unsupported GLM packed projection"),
                };
                (kind, w.layout.logical_shape[1], w.layout.logical_shape[0])
            }
        };
        Ok(model::Projection {
            kind,
            input,
            output,
        })
    }
    pub fn forward_on<D: BindingDevice<Tensor = T>>(
        &self,
        context: &DeviceContext<D>,
        x: &T,
    ) -> Result<T> {
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
