// Port of DS4's kernel_glm53_mlx_affine_mv_f32 body. Geometry and metadata
// flags stay runtime arguments to preserve the reference kernel's code path.
{
    const uint row_bytes = (uint)in_dim * (uint)bits / 8u;
    const uint groups_per_row = (uint)in_dim / (uint)group_size;
    const uint codes_per_byte = 8u / (uint)bits;
    const uint mask = (1u << (uint)bits) - 1u;
    const uint output_row = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    if (output_row >= (uint)out_dim || token >= (uint)n_rows) return;

    const device uchar *weight_bytes =
        reinterpret_cast<device const uchar *>(weights);
    const device uchar *scale_bytes =
        reinterpret_cast<device const uchar *>(scales);
    const device uchar *bias_bytes =
        reinterpret_cast<device const uchar *>(biases);
    const device uchar *weight_row = weight_bytes + (ulong)output_row * row_bytes;
    const device float *input_row = x + (ulong)token * (uint)in_dim;
    float sum = 0.0f;
    for (uint k = thread_index_in_simdgroup; k < (uint)in_dim; k += 32u) {
        const uchar packed = weight_row[k / codes_per_byte];
        const uint code = (packed >> ((k % codes_per_byte) * (uint)bits)) & mask;
        const uint affine_index = output_row * groups_per_row + k / (uint)group_size;
        float scale;
        float bias;
        if (metadata_bf16 != 0) {
            const ushort scale_bf16 =
                *reinterpret_cast<device const ushort *>(scale_bytes + 2u * affine_index);
            const ushort bias_bf16 =
                *reinterpret_cast<device const ushort *>(bias_bytes + 2u * affine_index);
            scale = as_type<float>((uint)scale_bf16 << 16);
            bias = as_type<float>((uint)bias_bf16 << 16);
        } else {
            scale = *reinterpret_cast<device const float *>(scale_bytes + 4u * affine_index);
            bias = *reinterpret_cast<device const float *>(bias_bytes + 4u * affine_index);
        }
        const float value = fma(float(code), scale, bias);
        sum = fma(value, input_row[k], sum);
    }
    sum = simd_sum(sum);
    if (thread_index_in_simdgroup == 0u) {
        out[(ulong)token * (uint)out_dim + output_row] = sum;
    }
}
