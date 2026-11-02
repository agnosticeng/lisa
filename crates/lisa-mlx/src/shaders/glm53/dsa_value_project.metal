// Apply the value half of packed MLX affine kv_b after indexed latent
// attention. For each output row this mirrors ds4's F32/F32 `_4` matvec
// schedule (K=512, 16-wide lanes, four simdgroups, then shared reduction).
{
    const uint rows = (uint)n_rows;
    const uint heads = (uint)n_heads;
    const uint row = threadgroup_position_in_grid.y;
    const uint d = threadgroup_position_in_grid.x;
    if (row >= rows || d >= 256u || heads == 0u || group_size <= 0 ||
        512 % (uint)group_size != 0u) {
        return;
    }
    const uint g = row * 256u + d;
    const uint head = row % heads;
    const uint weight_row = head * 512u + 256u + d;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    device const float *source = lora + (size_t)row * 512u;
    const uint ix = (uint)lane / 2u;
    const uint il = (uint)lane % 2u;
    float sumf = 0.0f;
    const uint ib0 = (uint)sg * 16u + ix;
    for (uint ib = ib0; ib < 16u; ib += 64u) {
        float sumq = 0.0f;
        const uint k0 = ib * 32u + il * 16u;
        for (uint i = 0; i < 4u; ++i) {
            const uint k = k0 + i * 4u;
            const uint word0 = kv_weight[(size_t)weight_row * 128u + (k >> 2)];
            const uint word1 = kv_weight[(size_t)weight_row * 128u + ((k + 1u) >> 2)];
            const uint code0 = (word0 >> ((k & 3u) * 8u)) & 0xffu;
            const uint code1 = (word1 >> (((k + 1u) & 3u) * 8u)) & 0xffu;
            const uint code2 = (word1 >> (((k + 2u) & 3u) * 8u)) & 0xffu;
            const uint code3 = (word1 >> (((k + 3u) & 3u) * 8u)) & 0xffu;
            const uint group0 = weight_row * (512u / (uint)group_size) + k / (uint)group_size;
            const uint group1 = weight_row * (512u / (uint)group_size) + (k + 1u) / (uint)group_size;
            const uint group2 = weight_row * (512u / (uint)group_size) + (k + 2u) / (uint)group_size;
            const uint group3 = weight_row * (512u / (uint)group_size) + (k + 3u) / (uint)group_size;
            const float4 weight = float4(
                fma((float)code0, kv_scales[group0], kv_biases[group0]),
                fma((float)code1, kv_scales[group1], kv_biases[group1]),
                fma((float)code2, kv_scales[group2], kv_biases[group2]),
                fma((float)code3, kv_scales[group3], kv_biases[group3]));
            const float4 activation = float4(
                source[k], source[k + 1u], source[k + 2u], source[k + 3u]);
            sumq += dot(weight, activation);
        }
        sumf += sumq;
    }

    sumf = simd_sum(sumf);
    if (sg == 0u) shared[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) shared[sg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(shared[lane]);
    if (lane == 0u && sg == 0u) output[g] = total;
}
