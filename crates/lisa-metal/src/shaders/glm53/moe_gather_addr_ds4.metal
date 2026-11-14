// DS4's one-token MLX-affine Q2 expert down-projection schedule.
// One SIMD group computes one output/route pair; each lane visits k=lane+32n,
// then the SIMD group reduces once, matching glm53_mlx_affine_matvec.
{
    const uint output_row = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    if (output_row >= (uint)out_rows || slot >= (uint)n_slots) return;

    const int expert = selected[slot];
    if (expert < 0 || expert >= n_experts) return;

    const size_t address_base = (size_t)expert * 9;
    const device uchar *weights = reinterpret_cast<const device uchar *>(
        as_type<ulong>(address_table[address_base + 6]));
    const device M *scales = reinterpret_cast<const device M *>(
        as_type<ulong>(address_table[address_base + 7]));
    const device M *biases = reinterpret_cast<const device M *>(
        as_type<ulong>(address_table[address_base + 8]));

    const uint row_bytes = (uint)in_vec_size / 4u;
    const uint groups_per_row = (uint)in_vec_size / 128u;
    const device uchar *weight_row = weights + (size_t)output_row * row_bytes;
    const device float *input_row = x +
        (size_t)(x_per_slot ? (int)slot : 0) * (size_t)in_vec_size;
    float sum = 0.0f;
    for (uint k = (uint)thread_index_in_simdgroup;
         k < (uint)in_vec_size; k += 32u) {
        const uchar packed = weight_row[k / 4u];
        const uint code = (packed >> ((k % 4u) * 2u)) & 0x3u;
        const uint affine_index = output_row * groups_per_row + k / 128u;
        const float scale = float(scales[affine_index]);
        const float bias = float(biases[affine_index]);
        const float value = fma(float(code), scale, bias);
        sum = fma(value, input_row[k], sum);
    }
    sum = simd_sum(sum);
    if (thread_index_in_simdgroup == 0) {
        y[(size_t)slot * (size_t)out_rows + output_row] = sum;
    }
}
