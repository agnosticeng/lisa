// GLM q_a + kv_a normalization from DS4's fused compact-cache kernel.
// One threadgroup handles one (row, branch): q output is F32 and the
// normalized compact latent is written directly as F16.
{
    const uint task = threadgroup_position_in_grid.x;
    const uint row = task >> 1u;
    const bool kv_task = (task & 1u) != 0u;
    const uint cache_pos = (uint)cache_pos_arg;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint threads = threads_per_threadgroup.x;
    const uint q_n = (uint)q_dim;
    const uint kv_n = (uint)kv_dim;
    const uint n = kv_task ? kv_n : q_n;
    const uint n4 = n / 4u;
    if (n == 0u || (n & 3u) != 0u || row >= (uint)n_rows || tid >= threads) return;

    if (simdgroup == 0u) shared[lane] = 0.0f;
    const device float4 *x4 = kv_task
        ? (const device float4 *)kv_input + (size_t)row * (kv_n / 4u)
        : (const device float4 *)q_input + (size_t)row * (q_n / 4u);
    const device float4 *w4 = kv_task
        ? (const device float4 *)kv_weight
        : (const device float4 *)q_weight;

    float sum = 0.0f;
    for (uint i = tid; i < n4; i += threads) {
        const float4 value = x4[i];
        sum += dot(value, value);
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) shared[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(shared[lane]);

    const float scale = rsqrt(sum / (float)n + eps);
    if (kv_task) {
        const device float *x = kv_input + (size_t)row * kv_n;
        const device float *w = kv_weight;
        device half *out = kv_norm + (size_t)(cache_pos + row) * kv_n;
        for (uint i = tid; i < kv_n; i += threads) {
            out[i] = (half)((x[i] * scale) * w[i]);
        }
    } else {
        device float4 *out = (device float4 *)q_norm + (size_t)row * n4;
        for (uint i = tid; i < n4; i += threads) {
            out[i] = (x4[i] * scale) * w4[i];
        }
    }
}
