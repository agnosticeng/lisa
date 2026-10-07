/// View a POD slice as bytes.
pub(super) fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Max rank the strided-copy kernel takes (kept small so the params struct is
/// a plain `setBytes`).
pub const MAX_COPY_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CopyParams {
    pub(super) ndim: u32,
    pub(super) shape: [u32; MAX_COPY_RANK],
    pub(super) strides: [u32; MAX_COPY_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Copy2Params {
    pub(super) ndim: u32,
    pub(super) shape: [u32; MAX_COPY_RANK],
    pub(super) src_strides: [u32; MAX_COPY_RANK],
    pub(super) dst_strides: [u32; MAX_COPY_RANK],
}

pub(super) fn pad_rank(v: &[usize]) -> [u32; MAX_COPY_RANK] {
    let mut o = [0u32; MAX_COPY_RANK];
    for (i, &x) in v.iter().enumerate() {
        o[i] = x as u32;
    }
    o
}

/// `copy_gg` (concatenate slices): template kernels + per-type instantiations,
/// prepended with the MLX utils preamble the helpers live in.
pub(super) const COPY_GG_SOURCE: &str = include_str!("../shaders/common/data/copy_gg.metal");

/// Strided -> strided copy body (`slice_assign` into a view).
pub(super) const COPY2_STRIDED_SOURCE: &str =
    include_str!("../shaders/common/data/copy2_strided.metal");

/// Strided -> contiguous materialisation of a view.
pub(super) const COPY_STRIDED_SOURCE: &str =
    include_str!("../shaders/common/data/copy_strided.metal");

/// Source for a `copy_gg_{ndim}` kernel of `dtype`: the preamble plus the static
/// template file (which carries every explicit instantiation).
pub(super) fn copy_gg_source() -> String {
    format!("{}{COPY_GG_SOURCE}", crate::jit::MLX_UTILS_PREAMBLE)
}
