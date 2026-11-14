//! Scalar mapping at the Metal storage boundary.
pub(crate) fn dtype(value: lisa_backend::ScalarType) -> lisa_metal::Dtype {
    use lisa_backend::ScalarType as S;
    use lisa_metal::Dtype as D;
    match value {
        S::Bool => D::Bool,
        S::U8 => D::Uint8,
        S::U32 => D::Uint32,
        S::I32 => D::Int32,
        S::I64 => D::Int64,
        S::F16 => D::Float16,
        S::F32 => D::Float32,
        S::BF16 => D::Bfloat16,
        S::U16 => D::Uint16,
        S::I8 => D::Int8,
        S::I16 => D::Int16,
        S::F64 => D::Float64,
    }
}
