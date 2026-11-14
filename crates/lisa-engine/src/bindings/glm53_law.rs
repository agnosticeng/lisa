//! Loaded GLM arithmetic identity, independent of pipeline compilation.
use lisa_backend::ManagedDevice;
fn law(provider: &str, arithmetic: &[u8], modes: &[u8], pairing: u64) -> u64 {
    b"glm53-affine-v4"
        .iter()
        .chain(provider.as_bytes())
        .chain(arithmetic)
        .chain(modes)
        .copied()
        .chain(pairing.to_le_bytes())
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}
pub fn cache_numerical_law<D: ManagedDevice>(device: &D, pairing: u64) -> u64 {
    let identity = device.identity();
    let modes = [
        u8::from(super::glm53_policy::glm53_q8_native_prefill_enabled()),
        u8::from(super::glm53_attention_mode::glm53_dsa_native_enabled()),
        u8::from(super::glm53_policy::native_gate_up_enabled()),
    ];
    law(&identity.provider, &identity.arithmetic, &modes, pairing)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loaded_projection_modes_cannot_be_masked_by_hash_bits() {
        let identities: std::collections::HashSet<_> = [0, 16, 32, 48]
            .into_iter()
            .flat_map(|pairing| {
                (0u8..8).map(move |mode| {
                    law(
                        "metal",
                        b"ordered-v1",
                        &[mode & 1, (mode >> 1) & 1, (mode >> 2) & 1],
                        pairing,
                    )
                })
            })
            .collect();
        assert_eq!(identities.len(), 32);
        assert_ne!(
            law("metal", b"ordered-v1", &[1, 1, 1], 16),
            law("rocm", b"ordered-v1", &[1, 1, 1], 16)
        );
        assert_ne!(
            law("metal", b"ordered-v1", &[1, 1, 1], 16),
            law("metal", b"ordered-v2", &[1, 1, 1], 16)
        );
    }
}
