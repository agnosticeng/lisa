#![cfg(test)]
use super::*;

#[test]
fn ds4_native_glm_batch_position_guard_matches_reference_boundaries() {
    let dense_compact_limit = 2048 + 4 - 1;
    assert!(ds4_native_session_batch_position_supported(
        2,
        dense_compact_limit - 1,
        dense_compact_limit
    ));
    assert!(!ds4_native_session_batch_position_supported(
        2,
        dense_compact_limit,
        dense_compact_limit
    ));
    assert!(!ds4_native_session_batch_position_supported(
        1,
        0,
        dense_compact_limit
    ));
    assert!(!ds4_native_session_batch_position_supported(
        DS4_GLM53_NATIVE_BATCH_ROW_LIMIT + 1,
        0,
        dense_compact_limit
    ));
}
