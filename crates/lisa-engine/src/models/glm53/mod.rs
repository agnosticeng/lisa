//! GLM-5.3 Flash (`glm5_next` / `glm5_next_text`).
//!
//! 45-layer hybrid: KDA (`linear_attention`, 34 layers) + NoPE-MLA DSA
//! (`deepseek_sparse_attention`, 11 layers), 288-expert MoE with clamped
//! SwiGLU, mHC hyper-connections (4 streams, sinkhorn-normalized mixing),
//! one MTP block. Ports ds4's execution graph to lisa's Rust + Metal stack
//! with MLX-format weights. Campaign:
//! `specs/11-glm53-flash.md`.

pub mod config;
pub mod kda;
pub mod mhc;
pub mod moe;
#[cfg(test)]
pub mod smoke;
pub mod sparse;
#[cfg(test)]
mod test_fixtures;
pub mod tower;

// Explicit cache-law revision for DS4 b4e3faf, original MLX affine metadata,
// native Q8 wide/tiny arithmetic (including MTP projections), the wide MTP
// attention fallback and the current KDA/DSA/MoE graph. Serial generated
// tokens use the decode graph shared by MTP seed/replay. Increment on any
// production arithmetic change; do not derive it from lazy pipelines.
const NUMERICAL_LAW_VERSION: u16 = 4;

const fn law_for_modes(version: u16, modes: u8) -> u64 {
    0x474c_4d35_3300_0000 | ((version as u64) << 8) | modes as u64
}

pub(crate) fn cache_numerical_law() -> u64 {
    let modes = u8::from(crate::core::quant::glm53_q8_native_prefill_enabled())
        | (u8::from(lisa_mlx::models::glm53::kernels::glm53_dsa_native_enabled()) << 1)
        | (u8::from(moe::native_gate_up_enabled()) << 2)
        | (u8::from(lisa_mlx::runtime::nax_available()) << 3);
    law_for_modes(NUMERICAL_LAW_VERSION, modes)
}

#[cfg(test)]
mod numerical_law_tests {
    use super::*;

    #[test]
    fn glm53_explicit_cache_law_distinguishes_versions_and_candidate_modes_without_jit() {
        let production = cache_numerical_law();
        let modes = 1 | 2 | 4 | (u8::from(lisa_mlx::runtime::nax_available()) << 3);
        assert_eq!(production, law_for_modes(NUMERICAL_LAW_VERSION, modes));
        assert_ne!(production, law_for_modes(NUMERICAL_LAW_VERSION + 1, modes));
        {
            let _scope = crate::core::quant::Glm53Q8PrefillScope::new(false);
            assert_ne!(cache_numerical_law(), production);
        }
        {
            let _scope = lisa_mlx::models::glm53::kernels::Glm53DsaNativeScope::new(false);
            assert_ne!(cache_numerical_law(), production);
        }
        {
            let _scope = moe::Glm53Q8GateUpNativeScope::new(false);
            assert_ne!(cache_numerical_law(), production);
        }
        assert_eq!(cache_numerical_law(), production);
    }
}
