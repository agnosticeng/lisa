// ─────────────────────────────── dtype ───────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
}

impl Dtype {
    // short aliases, so `mlx_rt`'s `DType::F32` etc. keep
    // reading the same.
    #[allow(non_upper_case_globals)]
    pub const U8: Dtype = Dtype::Uint8;
    #[allow(non_upper_case_globals)]
    pub const U16: Dtype = Dtype::Uint16;
    #[allow(non_upper_case_globals)]
    pub const I8: Dtype = Dtype::Int8;
    #[allow(non_upper_case_globals)]
    pub const U32: Dtype = Dtype::Uint32;
    #[allow(non_upper_case_globals)]
    pub const I16: Dtype = Dtype::Int16;
    #[allow(non_upper_case_globals)]
    pub const I32: Dtype = Dtype::Int32;
    #[allow(non_upper_case_globals)]
    pub const I64: Dtype = Dtype::Int64;
    #[allow(non_upper_case_globals)]
    pub const F16: Dtype = Dtype::Float16;
    #[allow(non_upper_case_globals)]
    pub const F32: Dtype = Dtype::Float32;
    #[allow(non_upper_case_globals)]
    pub const F64: Dtype = Dtype::Float64;
    #[allow(non_upper_case_globals)]
    pub const BF16: Dtype = Dtype::Bfloat16;

    /// Bytes per element.
    pub fn size_of(self) -> usize {
        match self {
            Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
            Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Int64 | Dtype::Float64 => 8,
        }
    }

    /// `size_in_bytes()`.
    pub fn size_in_bytes(self) -> usize {
        self.size_of()
    }

    pub fn is_float(self) -> bool {
        matches!(
            self,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16
        )
    }

    /// The MLX type tag; matches `mlx_rt`'s `type_to_name`.
    pub fn mlx_name(self) -> &'static str {
        match self {
            Dtype::Uint8 => "uint8",
            Dtype::Uint16 => "uint16",
            Dtype::Uint32 => "uint32",
            Dtype::Int16 => "int16",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Float16 => "float16",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "double",
            Dtype::Bfloat16 => "bfloat16",
            Dtype::Bool | Dtype::Int8 => "uint8",
        }
    }

    /// The Metal spelling; matches `mlx_rt`'s `type_string`.
    pub fn metal_name(self) -> &'static str {
        match self {
            Dtype::Bool | Dtype::Uint8 => "uint8_t",
            Dtype::Uint16 => "uint16_t",
            Dtype::Uint32 => "uint32_t",
            Dtype::Int8 => "int8_t",
            Dtype::Int16 => "int16_t",
            Dtype::Int32 => "int32_t",
            Dtype::Int64 => "int64_t",
            Dtype::Float16 => "float16_t",
            Dtype::Float32 => "float",
            Dtype::Float64 => "double",
            Dtype::Bfloat16 => "bfloat16_t",
        }
    }
}

/// The `Dtype` a Rust scalar type corresponds to (for `item` casting).
pub fn dtype_of<T: 'static>() -> Option<Dtype> {
    use std::any::TypeId;
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() {
        Some(Dtype::Float32)
    } else if t == TypeId::of::<f64>() {
        Some(Dtype::Float64)
    } else if t == TypeId::of::<i32>() {
        Some(Dtype::Int32)
    } else if t == TypeId::of::<i64>() {
        Some(Dtype::Int64)
    } else if t == TypeId::of::<u32>() {
        Some(Dtype::Uint32)
    } else if t == TypeId::of::<u8>() {
        Some(Dtype::Uint8)
    } else if t == TypeId::of::<i8>() {
        Some(Dtype::Int8)
    } else if t == TypeId::of::<i16>() {
        Some(Dtype::Int16)
    } else if t == TypeId::of::<u16>() {
        Some(Dtype::Uint16)
    } else if t == TypeId::of::<half::f16>() {
        Some(Dtype::Float16)
    } else if t == TypeId::of::<half::bf16>() {
        Some(Dtype::Bfloat16)
    } else {
        None
    }
}

pub(super) fn promote(a: Dtype, b: Dtype) -> Dtype {
    if a == b {
        return a;
    }
    let rank = |d: Dtype| match d {
        Dtype::Float64 => 6,
        Dtype::Float32 => 5,
        Dtype::Bfloat16 => 4,
        Dtype::Float16 => 3,
        Dtype::Int64 => 2,
        Dtype::Uint32 | Dtype::Int32 => 1,
        _ => 0,
    };
    if a.is_float() == b.is_float() {
        if rank(a) >= rank(b) { a } else { b }
    } else if a.is_float() {
        a
    } else {
        b
    }
}
