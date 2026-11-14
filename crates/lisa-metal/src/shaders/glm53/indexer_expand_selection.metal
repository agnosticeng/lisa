// Expand selected DSA pools into token ids and append the always-selected tail.
// This fuses the tensor broadcasts, pool expansion, causal masking, and tail
// construction used by the ds4 GLM-5.3 selector.
{
    const uint gid = thread_position_in_grid.x;
    const uint total = (uint)n_rows * (uint)output_width;
    if (gid >= total || output_width == 0 || pool_size == 0) return;

    const uint row = gid / (uint)output_width;
    const uint slot = gid - row * (uint)output_width;
    const uint query = row % (uint)n_tokens;
    const uint visible = (uint)pos0 + query + 1u;
    int32_t value = -1;

    const uint dense_limit = (uint)index_topk +
        (always_tail != 0 ? (uint)pool_size - 1u : 0u);
    if (causal_prefix != 0 && visible <= dense_limit) {
        if (slot < visible) value = (int32_t)slot;
        selected[gid] = value;
        return;
    }

    if (slot < index_topk) {
        const uint pool_slot = slot / (uint)pool_size;
        if (pool_slot < selected_pools) {
            const uint pool = (uint)pool_selected[(size_t)row * selected_pools + pool_slot];
            if ((pool + 1u) * (uint)pool_size <= visible) {
                value = (int32_t)(pool * (uint)pool_size + slot % (uint)pool_size);
            }
        }
    } else if (always_tail != 0) {
        const uint tail_slot = slot - (uint)index_topk;
        const uint tail_count = visible % (uint)pool_size;
        if (tail_slot < tail_count) {
            value = (int32_t)(visible - tail_count + tail_slot);
        }
    }

    selected[gid] = value;
}
