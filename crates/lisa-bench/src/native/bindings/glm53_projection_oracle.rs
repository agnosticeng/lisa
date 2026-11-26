#![cfg(test)]
//! Original GGUF blocks, dispatched without affine conversion.
use crate::core::gguf_resident::PackedTensor;
use anyhow::{Context, Result, ensure};
use lisa_gguf::TensorType;
use lisa_metal::kernels::{ggml as gpu, sparse::kernels};
use lisa_metal::{Array, Dtype, Stream};

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
                kernels::bf16_gemv(w, &input, &s)
            } else {
                kernels::bf16_gemm(w, &input, &s)
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
