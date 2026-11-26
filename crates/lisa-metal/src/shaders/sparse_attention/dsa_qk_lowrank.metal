// GLM DSA query/key low-rank projection.
// Ports ds4's `kernel_mul_mv_f32_f32_4` decode matvec reduction for the
// dequantized MLX affine rows. Each threadgroup owns one output latent row;
// each lane accumulates the same 16-value chunks before the simdgroup and
// threadgroup reductions used by ds4.
{
    constexpr uint SIMD_WIDTH = 32u;
    constexpr uint SIMDGROUPS = 2u;
    constexpr uint VALUES_PER_LANE = 16u;
    constexpr uint CHUNK = 32u;
    constexpr uint INPUT_WIDTH = 256u;
    constexpr uint OUTPUT_WIDTH = 512u;
    constexpr uint N_BLOCKS = INPUT_WIDTH / CHUNK;

    const uint latent_id = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    const uint head_batch = threadgroup_position_in_grid.z;
    const uint head = head_batch % (uint)n_heads;
    const uint batch = head_batch / (uint)n_heads;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    if (latent_id >= OUTPUT_WIDTH || head >= (uint)n_heads || token >= (uint)n_queries ||
        batch >= (uint)n_batches || group_size <= 0 ||
        OUTPUT_WIDTH % (uint)group_size != 0) {
        return;
    }

    const size_t q_row =
        (((size_t)batch * (size_t)n_heads + (size_t)head) *
         (size_t)n_queries + (size_t)token) * INPUT_WIDTH;
    device const float4 *q4 = (device const float4 *)(q + q_row);
    device const uint *packed = (device const uint *)kv_weight;
    device const float *scale = kv_scales;
    device const float *bias = kv_biases;

    const uint ix = lane / (SIMD_WIDTH / VALUES_PER_LANE);
    const uint il = lane % (SIMD_WIDTH / VALUES_PER_LANE);
    const uint block0 = simdgroup * VALUES_PER_LANE + ix;
    float sumf = 0.0f;

    for (uint block = block0; block < N_BLOCKS;
         block += SIMDGROUPS * VALUES_PER_LANE) {
        const uint input0 = block * CHUNK + il * VALUES_PER_LANE;
        float sumq = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE / 4u; ++i) {
            const uint d = input0 + i * 4u;
            const uint row0 = head * OUTPUT_WIDTH + d;
            float4 weight4;
            for (uint k = 0; k < 4u; ++k) {
                const uint row = row0 + k;
                const uint word = packed[(size_t)row * (OUTPUT_WIDTH / 4u) +
                                         latent_id / 4u];
                const uint code = (word >> ((latent_id & 3u) * 8u)) & 0xffu;
                const uint affine = row * (OUTPUT_WIDTH / (uint)group_size) +
                                    latent_id / (uint)group_size;
                weight4[k] = fma((float)code, scale[affine], bias[affine]);
            }
            sumq += dot(weight4, q4[d / 4u]);
        }
        sumf += sumq;
    }

    threadgroup float *partial = shared;
    if (simdgroup == 0u) {
        partial[lane] = 0.0f;
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        partial[simdgroup] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float total = simd_sum(partial[lane]);
    if (lane == 0u && simdgroup == 0u) {
        const size_t out_row =
            (((size_t)batch * (size_t)n_queries + (size_t)token) *
             (size_t)n_heads + (size_t)head) * OUTPUT_WIDTH + latent_id;
        output[out_row] = total;
    }
}
