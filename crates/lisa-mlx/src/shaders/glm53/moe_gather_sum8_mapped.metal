// DS4-exact fused Q2 down projection for eight routed experts using ordinary
// encoder-bound resources. One SIMD group computes one selected expert's
// lane-strided affine matvec; the group leader sums the eight results in route
// order, matching moe_gather_sum8_addr_ds4.metal.
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
        const device uint8_t *weights = glm53_mapped_resource(
            as_type<ulong>(address_table[address_base + 6]), resource_00,
            resource_01, resource_02, resource_03, resource_04, resource_05,
            resource_06, resource_07, resource_08, resource_09, resource_10,
            resource_11, resource_12, resource_13, resource_14, resource_15);
        const device M *scales = reinterpret_cast<const device M *>(glm53_mapped_resource(
            as_type<ulong>(address_table[address_base + 7]), resource_00,
            resource_01, resource_02, resource_03, resource_04, resource_05,
            resource_06, resource_07, resource_08, resource_09, resource_10,
            resource_11, resource_12, resource_13, resource_14, resource_15));
        const device M *biases = reinterpret_cast<const device M *>(glm53_mapped_resource(
            as_type<ulong>(address_table[address_base + 8]), resource_00,
            resource_01, resource_02, resource_03, resource_04, resource_05,
            resource_06, resource_07, resource_08, resource_09, resource_10,
            resource_11, resource_12, resource_13, resource_14, resource_15));
        const device float *input_row = x + (size_t)selected_index * (size_t)in_vec_size;

        for (uint k = (uint)thread_index_in_simdgroup;
             k < (uint)in_vec_size; k += 32u) {
            const float input_value = input_row[k];
            #pragma unroll
            for (uint row = 0; row < output_rows_per_group; row++) {
                if (output_row + row < (uint)out_rows) {
                    const device uint8_t *weight_row =
                        weights + (size_t)(output_row + row) * row_bytes;
                    const uint8_t packed = weight_row[k / 4u];
                    const uint code = (packed >> ((k % 4u) * 2u)) & 0x3u;
                    const uint affine_index =
                        (output_row + row) * groups_per_row + k / 128u;
                    const float scale = float(scales[affine_index]);
                    const float bias = float(biases[affine_index]);
                    const float value = fma(float(code), scale, bias);
                    sums[row] = fma(value, input_value, sums[row]);
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
                out[(size_t)token * (size_t)out_rows + output_row + row] = total;
            }
        }
    }
}
