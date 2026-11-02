// DS4's fused DSA q-rank and KV-LoRA row normalization.
// Both branches use q_dim/4 threads, including the shorter 512-value KV row.
{
    const uint task = threadgroup_position_in_grid.x;
    const uint row = task >> 1u;
    const bool kv_task = (task & 1u) != 0u;
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
        sum += dot(x4[i], x4[i]);
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) shared[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(shared[lane]);

    const float scale = 1.0f / sqrt(sum / (float)n + eps);
    device float4 *out4 = kv_task
        ? (device float4 *)kv_output + (size_t)row * (kv_n / 4u)
        : (device float4 *)q_output + (size_t)row * (q_n / 4u);
    for (uint i = tid; i < n4; i += threads) {
        out4[i] = (x4[i] * scale) * w4[i];
    }
}
