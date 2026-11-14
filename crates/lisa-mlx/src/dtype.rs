//! Scalar storage tags accepted by the MLX safetensors reader.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint32,
    Int32,
    Int64,
    Float16,
    Float32,
    Bfloat16,
}

impl Dtype {
    pub const fn size_of(self) -> usize {
        match self {
            Self::Bool | Self::Uint8 => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Int64 => 8,
        }
    }
}
