// DS4's F32 dense matvec path used by one-token GLM router projections.
// One threadgroup owns one output row and one input row. The chunk walk,
// float4 dot accumulation, and two-stage simdgroup reduction mirror
// `kernel_mul_mv_f32_f32_4` in ds4's `metal/dense.metal`.
{
    constexpr uint SIMD_WIDTH = 32u;
    constexpr uint SIMDGROUPS = 8u;
    constexpr uint VALUES_PER_LANE = 16u;
    constexpr uint CHUNK = 32u;
    const uint out_row = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint in_width = (uint)in_dim;
    const uint n_blocks = in_width / CHUNK;
    if (out_row >= (uint)out_dim || token >= (uint)n_tokens) return;

    const device float4 *weight4 = (const device float4 *)(weights + (size_t)out_row * in_width);
    const device float4 *x4 = (const device float4 *)(x + (size_t)token * in_width);
    const uint ix = lane / 2u;
    const uint il = lane % 2u;
    const uint block0 = simdgroup * 16u + ix;
    float sumf = 0.0f;

    for (uint block = block0; block < n_blocks; block += SIMDGROUPS * 16u) {
        const uint vec0 = (block * CHUNK + il * VALUES_PER_LANE) / 4u;
        float sumq = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE / 4u; i++) {
            sumq += dot(weight4[vec0 + i], x4[vec0 + i]);
        }
        sumf += sumq;
    }

    threadgroup float *partial = shared;
    if (simdgroup == 0u) partial[lane] = 0.0f;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) partial[simdgroup] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float total = simd_sum(partial[lane]);
    if (lane == 0u && simdgroup == 0u) {
        logits[(size_t)token * (uint)out_dim + out_row] = total;
    }
}
