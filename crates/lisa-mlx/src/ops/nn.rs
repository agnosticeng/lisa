//! `lisa_mlx::nn`-shaped wrappers, routed through the MLX unary kernels so the
//! rounding matches (an op chain is not bit-exact for `silu`).
use super::*;
use crate::jit;

pub fn silu(a: &Array) -> Result<Array> {
    Ok(Array::new(jit::silu(a.t.device(), &a.t)?))
}
