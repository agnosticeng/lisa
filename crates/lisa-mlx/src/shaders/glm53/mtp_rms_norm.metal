// GLM-5.3 MTP weighted RMSNorm, following ds4's kernel_rms_norm_mul_f32_4.
// One threadgroup handles one row. Four contiguous channels are accumulated
// with float4 dot products, then reduced first within each SIMD group and
// across SIMD groups in the same order as ds4.
{
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint threads = threads_per_threadgroup.x;
    const uint n = (uint)n_dim;
    const uint n4 = n / 4u;
    if (n == 0u || (n & 3u) != 0u || row >= (uint)n_rows || tid >= threads) return;

    threadgroup float shared[32];
    if (simdgroup == 0u) shared[lane] = 0.0f;

    const device float4 *x = (const device float4 *)input + (size_t)row * n4;
    const device float4 *w = (const device float4 *)weight;
    float sum = 0.0f;
    for (uint i = tid; i < n4; i += threads) {
        sum += dot(x[i], x[i]);
    }
    sum = simd_sum(sum);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) shared[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = shared[lane];
    sum = simd_sum(sum);
    const float mean = sum / (float)n;
    const float scale = 1.0f / sqrt(mean + eps);

    device float4 *out = (device float4 *)output + (size_t)row * n4;
    for (uint i = tid; i < n4; i += threads) {
        out[i] = (x[i] * scale) * w[i];
    }
}
