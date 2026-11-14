// DS4-exact fused Q2 down projection for eight routed experts.
// One threadgroup owns one token and four adjacent output rows. Its eight
// SIMD groups each run four independent DS4 per-expert lane-strided matvecs;
// each lane reuses the route activation while preserving every output's FMA
// order. One lane then adds the eight reduced F32 results in route order.
{
    constexpr uint output_rows_per_group = 4u;
    const uint output_row = threadgroup_position_in_grid.x * output_rows_per_group;
    const uint token = threadgroup_position_in_grid.y;
    const uint slot = simdgroup_index_in_threadgroup;
    if (output_row >= (uint)out_rows || token >= (uint)n_tokens || slot >= 8u) return;

    const uint selected_index = token * 8u + slot;
    const int expert = selected[selected_index];
    const uint row_bytes = (uint)in_vec_size / 4u;
    const uint groups_per_row = (uint)in_vec_size / 128u;
    float sums[output_rows_per_group] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (expert >= 0 && expert < n_experts) {
        const size_t address_base = (size_t)expert * 9;
        const device uchar *weights = reinterpret_cast<const device uchar *>(
            as_type<ulong>(address_table[address_base + 6]));
        const device M *scales = reinterpret_cast<const device M *>(
            as_type<ulong>(address_table[address_base + 7]));
        const device M *biases = reinterpret_cast<const device M *>(
            as_type<ulong>(address_table[address_base + 8]));
        const device float *input_row = x + (size_t)selected_index * (size_t)in_vec_size;

        for (uint group = 0; group < groups_per_row; group++) {
            float row_scales[output_rows_per_group];
            float row_biases[output_rows_per_group];
            #pragma unroll
            for (uint row = 0; row < output_rows_per_group; row++) {
                const uint source_row = min(output_row + row, (uint)out_rows - 1u);
                const uint affine_index = source_row * groups_per_row + group;
                row_scales[row] = float(scales[affine_index]);
                row_biases[row] = float(biases[affine_index]);
            }
            // A lane visits four values in each affine group. Hoisting the
            // metadata leaves the original k += 32 FMA order unchanged.
            #pragma unroll
            for (uint item = 0; item < 4u; item++) {
                const uint k = group * 128u + (uint)thread_index_in_simdgroup + item * 32u;
                const float input_value = input_row[k];
                #pragma unroll
                for (uint row = 0; row < output_rows_per_group; row++) {
                    if (output_row + row < (uint)out_rows) {
                        const device uchar *weight_row =
                            weights + (size_t)(output_row + row) * row_bytes;
                        const uchar packed = weight_row[k / 4u];
                        const uint code = (packed >> ((k % 4u) * 2u)) & 0x3u;
                        const float value = fma(float(code), row_scales[row], row_biases[row]);
                        sums[row] = fma(value, input_value, sums[row]);
                    }
                }
            }
        }
    }

    threadgroup float route_sums[8][output_rows_per_group];
    float reduced[output_rows_per_group];
    #pragma unroll
    for (uint row = 0; row < output_rows_per_group; row++) {
        reduced[row] = simd_sum(sums[row]);
    }
    if (thread_index_in_simdgroup == 0) {
        #pragma unroll
        for (uint row = 0; row < output_rows_per_group; row++) {
            route_sums[slot][row] = reduced[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0 && thread_index_in_simdgroup == 0) {
        #pragma unroll
        for (uint row = 0; row < output_rows_per_group; row++) {
            if (output_row + row < (uint)out_rows) {
                float total = 0.0f;
                for (uint route = 0; route < 8u; route++) {
                    total += route_sums[route][row];
                }
                if (has_shared != 0) {
                    total += shared[(size_t)token * (size_t)out_rows + output_row + row];
                }
                y[(size_t)token * (size_t)out_rows + output_row + row] = total;
            }
        }
    }
}
