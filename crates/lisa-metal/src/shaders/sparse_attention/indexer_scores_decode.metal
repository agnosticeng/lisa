// GLM-5.3 one-token DSA indexer score, matching ds4's direct 32-head kernel.
// One 4-simdgroup threadgroup computes one pool score for one batch row.
{
    const uint group = threadgroup_position_in_grid.x;
    const uint pool = group % (uint)n_rows;
    const uint batch = group / (uint)n_rows;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    if (pool >= (uint)n_rows || batch >= (uint)n_batches) return;

    threadgroup float key_shared[128];
    threadgroup float partial_heads[8];
    if (tid < 128u) {
        key_shared[tid] = float(pooled[(size_t)batch * (uint)n_rows * 128u +
                                       (size_t)pool * 128u + tid]);
    }
    float score = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0u; head0 < 32u; head0 += 8u) {
        const uint head = head0 + simdgroup;
        const device float4 *q4 = (device const float4 *)(q +
            ((size_t)batch * 32u + head) * 128u);
        threadgroup const float4 *k4 = (threadgroup const float4 *)key_shared;
        float dot_value = dot(q4[lane], k4[lane]);
        dot_value = simd_sum(dot_value);
        if (lane == 0u) {
            partial_heads[simdgroup] = max(dot_value * scale, 0.0f) *
                weights[(size_t)batch * 32u + head];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            score += partial_heads[0];
            score += partial_heads[1];
            score += partial_heads[2];
            score += partial_heads[3];
            score += partial_heads[4];
            score += partial_heads[5];
            score += partial_heads[6];
            score += partial_heads[7];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) scores[(size_t)batch * (uint)n_rows + pool] = score;
}
