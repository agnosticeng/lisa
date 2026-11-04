//! GGUF v3 storage. Quantized payloads remain in their original block format.
//!
//! Header parsing never maps or reads the weight payload. A resident tensor
//! mapping and a disk-backed row reader are separate, explicit operations.
//! This crate has no dependency on LISA's engine or Metal runtime. GPU weight
//! views and RAM admission belong to `lisa-engine::core::gguf_resident`.

mod header;
mod package;
pub mod quant;

pub use header::{Header, TensorInfo, Value};
pub use package::{Package, Tensor};

#[cfg(test)]
mod tests;

/// GGML block geometry, independent of whether an execution kernel exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorType(pub u32);

impl TensorType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q8_0: Self = Self(8);
    pub const Q2_K: Self = Self(10);
    pub const Q4_K: Self = Self(12);
    pub const Q5_K: Self = Self(13);
    pub const IQ2_XXS: Self = Self(16);
    pub const IQ2_XS: Self = Self(17);
    pub const IQ4_NL: Self = Self(20);
    pub const IQ1_M: Self = Self(29);
    pub const BF16: Self = Self(30);

    pub fn layout(self) -> anyhow::Result<(&'static str, u64, u64)> {
        Ok(match self.0 {
            0 => ("F32", 1, 4),
            1 => ("F16", 1, 2),
            2 => ("Q4_0", 32, 18),
            3 => ("Q4_1", 32, 20),
            6 => ("Q5_0", 32, 22),
            7 => ("Q5_1", 32, 24),
            8 => ("Q8_0", 32, 34),
            9 => ("Q8_1", 32, 36),
            10 => ("Q2_K", 256, 84),
            11 => ("Q3_K", 256, 110),
            12 => ("Q4_K", 256, 144),
            13 => ("Q5_K", 256, 176),
            14 => ("Q6_K", 256, 210),
            15 => ("Q8_K", 256, 292),
            16 => ("IQ2_XXS", 256, 66),
            17 => ("IQ2_XS", 256, 74),
            18 => ("IQ3_XXS", 256, 98),
            19 => ("IQ1_S", 256, 50),
            // IQ4_NL is 18 bytes per 32 values, including the FP16 scale.
            // DS4's unused type-table entry has different geometry; use GGML's ABI.
            20 => ("IQ4_NL", 32, 18),
            21 => ("IQ3_S", 256, 110),
            22 => ("IQ2_S", 256, 82),
            23 => ("IQ4_XS", 256, 136),
            24 => ("I8", 1, 1),
            25 => ("I16", 1, 2),
            26 => ("I32", 1, 4),
            27 => ("I64", 1, 8),
            28 => ("F64", 1, 8),
            29 => ("IQ1_M", 256, 56),
            30 => ("BF16", 1, 2),
            39 => ("MXFP4", 32, 17),
            _ => anyhow::bail!("unsupported GGUF tensor type {}", self.0),
        })
    }

    pub fn row_bytes(self, columns: u64) -> anyhow::Result<u64> {
        let (name, block, bytes) = self.layout()?;
        anyhow::ensure!(
            columns > 0 && columns % block == 0,
            "{name} row width {columns} is not a positive multiple of {block}"
        );
        (columns / block)
            .checked_mul(bytes)
            .ok_or_else(|| anyhow::anyhow!("GGUF row byte size overflow"))
    }
}
