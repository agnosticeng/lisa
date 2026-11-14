// GLM-5.3 DSA indexer score batch, matching ds4's scalar batch kernel.
// One threadgroup computes one (query, pooled-key) score. q is [B,H,T,D],
// weights [B,T,H], pooled keys are F16 [B,P,D], output is [B,T,P] F32.
{
    const uint pool = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    const uint batch = threadgroup_position_in_grid.z;
    const uint tid = thread_index_in_threadgroup;
    const uint pools = (uint)n_rows;
    const uint tokens = (uint)n_tokens;
    if (pool >= pools || token >= tokens || batch >= (uint)n_batches) return;

    const uint visible = min(((uint)pos0 + token + 1u) / (uint)pool_size, pools);
    const size_t dst_index = ((size_t)batch * tokens + token) * pools + pool;
    if (pool >= visible) {
        if (tid == 0u) scores[dst_index] = -INFINITY;
        return;
    }

    threadgroup float scratch[32];
    const device float16_t *key_row = pooled +
        ((size_t)batch * pools + pool) * (uint)head_dim;
    const device float *weight_row = weights +
        ((size_t)batch * tokens + token) * (uint)n_heads;
    float score = 0.0f;
    for (uint head = 0u; head < (uint)n_heads; ++head) {
        const size_t q_base = (((size_t)batch * (uint)n_heads + head) * tokens + token) *
            (uint)head_dim;
        float partial = 0.0f;
        for (uint d = tid; d < (uint)head_dim; d += 32u) {
            partial += q[q_base + d] * float(key_row[d]);
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = 16u; step > 0u; step >>= 1u) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) score += max(scratch[0] * scale, 0.0f) * weight_row[head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) scores[dst_index] = score;
}
