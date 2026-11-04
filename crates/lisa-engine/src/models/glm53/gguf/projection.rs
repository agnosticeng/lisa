//! Original GGUF blocks, dispatched without affine conversion.
use crate::core::gguf_resident::{PackedTensor, ResidentWeights};
use anyhow::{Context, Result, ensure};
use lisa_gguf::TensorType;
use lisa_mlx::models::{deepseek41 as gpu, glm53::kernels};
use lisa_mlx::{Array, Dtype, Stream};

pub fn plain(weights: &ResidentWeights, name: &str) -> Result<Array> {
    weights
        .tensors
        .get(name)
        .with_context(|| format!("missing GLM GGUF {name}"))?
        .plain_view()
}

pub enum Projection {
    Plain(Array),
    Packed(PackedTensor),
}
impl From<Array> for Projection {
    fn from(a: Array) -> Self {
        Self::Plain(a)
    }
}
// Plain-only fusion eligibility is checked by dtype before inspecting shape.
impl std::ops::Deref for Projection {
    type Target = Array;
    fn deref(&self) -> &Array {
        match self {
            Self::Plain(a) => a,
            Self::Packed(p) => p.packed_bytes(),
        }
    }
}
impl Projection {
    pub fn load(w: &ResidentWeights, name: &str) -> Result<Self> {
        let p = w
            .tensors
            .get(name)
            .with_context(|| format!("missing GLM GGUF {name}"))?;
        Ok(match p.info.kind {
            TensorType::F32 | TensorType::BF16 | TensorType::F16 => Self::Plain(plain(w, name)?),
            _ => Self::Packed(p.clone()),
        })
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(
            x.rank() == 3 && x.dtype() == Dtype::Float32,
            "GLM projection expects F32 [batch,tokens,width]"
        );
        let rows = x.dim(0) * x.dim(1);
        let input = x.reshape(&[rows, x.dim(2)])?.contiguous()?;
        let s = Stream::thread_local_or_default();
        let out = match self {
            Self::Plain(w) if w.dtype() == Dtype::Bfloat16 => if rows <= 8 {
                kernels::glm53_bf16_gemv(w, &input, &s)
            } else {
                kernels::glm53_bf16_gemm(w, &input, &s)
            }
            .context("GLM BF16 projection declined")?,
            Self::Plain(w) => gpu::plain_matvec(w, &input, &s)?,
            Self::Packed(w) => match w.info.kind {
                TensorType::Q8_0 => gpu::q8_matvec(w.packed_bytes(), &input, false, false, &s)?,
                TensorType::Q4_K => {
                    gpu::packed_matvec(gpu::BlockFormat::Q4K, w.packed_bytes(), &input, &s)?
                }
                _ => anyhow::bail!("unsupported GLM projection {:?}", w.info.kind),
            },
        };
        Ok(out.reshape(&[x.dim(0), x.dim(1), out.dim(1)])?)
    }
}
