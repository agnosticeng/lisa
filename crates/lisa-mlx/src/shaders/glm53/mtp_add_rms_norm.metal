// GLM MTP residual-add plus weighted RMSNorm, following DS4's
// kernel_add_rms_norm_mul_f32_4. One threadgroup handles one row; float4
// partials reduce within each SIMD group and then across SIMD groups.
{
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint threads = threads_per_threadgroup.x;
    const uint n = (uint)n_dim;
    const uint n4 = n / 4u;
    if (n == 0u || (n & 3u) != 0u || row >= (uint)n_rows || tid >= threads) return;

    if (simdgroup == 0u) shared[lane] = 0.0f;

    const device float4 *a4 = (const device float4 *)a + (size_t)row * n4;
    const device float4 *b4 = (const device float4 *)b + (size_t)row * n4;
    const device float4 *w4 = (const device float4 *)weight;
    device float4 *sum4 = (device float4 *)sum_out + (size_t)row * n4;
    device float4 *norm4 = (device float4 *)norm_out + (size_t)row * n4;

    float sum = 0.0f;
    for (uint i = tid; i < n4; i += threads) {
        const float4 value = a4[i] + b4[i];
        sum4[i] = value;
        sum += dot(value, value);
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) shared[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(shared[lane]);

    const float mean = sum / (float)n;
    const float scale = 1.0f / sqrt(mean + eps);
    for (uint i = tid; i < n4; i += threads) {
        norm4[i] = (sum4[i] * scale) * w4[i];
    }
}
