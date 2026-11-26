// DS4 GLM-5.3 raw MLX-affine matvec schedule for the DSA output projection.
// Adapted from ds4/metal/glm53_bf16.metal; scales and biases arrive widened
// from their checkpoint BF16 representation without changing their values.
{
    const uint rows = (uint)n_rows;
    const uint out_row = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    if (out_row >= 4096u || token >= rows || group_size <= 0 ||
        16384u % (uint)group_size != 0u) return;

    const uint groups_per_row = 16384u / (uint)group_size;
    const uint words_per_row = 4096u;
    const device uint *weight_row = weight + (size_t)out_row * words_per_row;
    const device float *input_row = input + (size_t)token * 16384u;
    const ushort lane = thread_index_in_simdgroup;
    float sum = 0.0f;
    for (uint k = (uint)lane; k < 16384u; k += 32u) {
        const uint packed = weight_row[k / 4u];
        const uint code = (packed >> ((k & 3u) * 8u)) & 0xffu;
        const uint affine_index = out_row * groups_per_row + k / (uint)group_size;
        const float value = fma((float)code, scales[affine_index], biases[affine_index]);
        sum = fma(value, input_row[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) output[(size_t)token * 4096u + out_row] = sum;
}
