#![cfg(test)]
use super::*;

#[test]
fn prefetch_budget_uses_selected_payload_bytes() {
    let expert_bytes = vec![8; 64];
    let many_small_experts = (0..20).collect::<Vec<_>>();
    assert_eq!(
        selected_prefetch_bytes(&many_small_experts, &expert_bytes),
        Some(160)
    );
    assert!(
        selected_prefetch_bytes(&many_small_experts, &expert_bytes)
            .is_some_and(|bytes| bytes <= 160),
        "a prefill selection above the old 16-expert cutoff should fit by bytes"
    );
    assert!(
        !selected_prefetch_bytes(&many_small_experts, &expert_bytes)
            .is_some_and(|bytes| bytes <= 159),
        "read-ahead must stay within the per-pool byte budget"
    );
}

#[test]
fn prefetch_budget_rejects_invalid_or_overflowing_expert_sizes() {
    assert_eq!(selected_prefetch_bytes(&[0, 1], &[4]), None);
    assert_eq!(selected_prefetch_bytes(&[0, 1], &[usize::MAX, 1]), None);
}
