// Stable descending top-32 per block. Only comparisons are performed: values
// are promoted exactly to F32; NaNs sort last and equal values keep ID order.
{
    const uint lane = thread_position_in_threadgroup.x;
    const uint block = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint index = block * NTH + lane;
    uint id = index < uint(width) ? index : UINT_MAX;
    float value = id != UINT_MAX ? float(scores[size_t(row) * width + id]) : 0.0f;
    threadgroup uint ids[NTH];
    threadgroup float values[NTH];
    for (uint span = 2; span <= NTH; span <<= 1) {
        for (uint stride = span >> 1; stride; stride >>= 1) {
            uint other_id;
            float other_value;
            if (stride < 32) {
                other_id = simd_shuffle_xor(id, ushort(stride));
                other_value = simd_shuffle_xor(value, ushort(stride));
            } else {
                ids[lane] = id;
                values[lane] = value;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                other_id = ids[lane ^ stride];
                other_value = values[lane ^ stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const bool best_first = ((lane & span) == 0) == ((lane & stride) == 0);
            const bool replace = best_first
                ? lisa_topk_before(other_value, other_id, value, id)
                : lisa_topk_before(value, id, other_value, other_id);
            if (replace) {
                id = other_id;
                value = other_value;
            }
        }
    }
    if (lane < 32) {
        candidates[(size_t(row) * uint(blocks) + block) * 32 + lane] = id;
    }
}
