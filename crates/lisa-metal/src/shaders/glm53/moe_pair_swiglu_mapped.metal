// DS4-exact GLM-5.3 gate/up SwiGLU over MLX-native Q2 expert buffers.
// The nine-field expert table stores (resource slot, byte offset) entries;
// this kernel uses ordinary encoder-bound buffers while preserving the same
// per-lane FMA and reduction order as moe_pair_swiglu_addr.metal.
{
    constexpr int num_simdgroups = 4;
    constexpr int results_per_simdgroup = 1;
    constexpr int group_size = 128;

    const uint slot = threadgroup_position_in_grid.y / groups_per_slot;
    const uint row0 = (threadgroup_position_in_grid.y % groups_per_slot) *
            (num_simdgroups * results_per_simdgroup) +
        simdgroup_index_in_threadgroup * results_per_simdgroup;
    if (slot >= (uint)n_slots || row0 >= (uint)mid_dim) return;

    const int expert = selected[slot];
    if (expert < 0 || expert >= n_experts) return;
    const size_t address_base = (size_t)expert * 9;
    const device uint8_t *gate_w = glm53_mapped_resource(
        as_type<ulong>(address_table[address_base]), resource_00, resource_01,
        resource_02, resource_03, resource_04, resource_05, resource_06,
        resource_07, resource_08, resource_09, resource_10, resource_11,
        resource_12, resource_13, resource_14, resource_15);
    const device M *gate_s = reinterpret_cast<const device M *>(glm53_mapped_resource(
        as_type<ulong>(address_table[address_base + 1]), resource_00, resource_01,
        resource_02, resource_03, resource_04, resource_05, resource_06,
        resource_07, resource_08, resource_09, resource_10, resource_11,
        resource_12, resource_13, resource_14, resource_15));
    const device M *gate_b = reinterpret_cast<const device M *>(glm53_mapped_resource(
        as_type<ulong>(address_table[address_base + 2]), resource_00, resource_01,
        resource_02, resource_03, resource_04, resource_05, resource_06,
        resource_07, resource_08, resource_09, resource_10, resource_11,
        resource_12, resource_13, resource_14, resource_15));
    const device uint8_t *up_w = glm53_mapped_resource(
        as_type<ulong>(address_table[address_base + 3]), resource_00, resource_01,
        resource_02, resource_03, resource_04, resource_05, resource_06,
        resource_07, resource_08, resource_09, resource_10, resource_11,
        resource_12, resource_13, resource_14, resource_15);
    const device M *up_s = reinterpret_cast<const device M *>(glm53_mapped_resource(
        as_type<ulong>(address_table[address_base + 4]), resource_00, resource_01,
        resource_02, resource_03, resource_04, resource_05, resource_06,
        resource_07, resource_08, resource_09, resource_10, resource_11,
        resource_12, resource_13, resource_14, resource_15));
    const device M *up_b = reinterpret_cast<const device M *>(glm53_mapped_resource(
        as_type<ulong>(address_table[address_base + 5]), resource_00, resource_01,
        resource_02, resource_03, resource_04, resource_05, resource_06,
        resource_07, resource_08, resource_09, resource_10, resource_11,
        resource_12, resource_13, resource_14, resource_15));

    const int row_bytes = in_vec_size / 4;
    const int row_groups = in_vec_size / group_size;

    typedef float U;
    thread U gate_result[results_per_simdgroup] = {0};
    thread U up_result[results_per_simdgroup] = {0};
    const uint token = slot / (uint)slots_per_token;
    const device T *xx = x + (size_t)token * (size_t)in_vec_size;
    const uint lane = thread_index_in_simdgroup;

    for (int group = 0; group < row_groups; group++) {
        thread U gate_scale[results_per_simdgroup];
        thread U gate_bias[results_per_simdgroup];
        thread U up_scale[results_per_simdgroup];
        thread U up_bias[results_per_simdgroup];
        for (int row = 0; row < results_per_simdgroup; row++) {
            const size_t metadata_index = ((size_t)row0 + (size_t)row) *
                (size_t)row_groups + (size_t)group;
            gate_scale[row] = float(gate_s[metadata_index]);
            gate_bias[row] = float(gate_b[metadata_index]);
            up_scale[row] = float(up_s[metadata_index]);
            up_bias[row] = float(up_b[metadata_index]);
        }
        #pragma unroll
        for (int item = 0; item < group_size / SIMD_SIZE; item++) {
            const int k = group * group_size + (int)lane + item * SIMD_SIZE;
            const int byte_index = k / 4;
            const int shift = (k & 3) * 2;
            const float input = float(xx[k]);
            for (int row = 0; row < results_per_simdgroup; row++) {
                const size_t row_offset = (size_t)row0 + (size_t)row;
                const uint8_t gate_code = (gate_w[row_offset * (size_t)row_bytes +
                    (size_t)byte_index] >> shift) & 0x03u;
                const uint8_t up_code = (up_w[row_offset * (size_t)row_bytes +
                    (size_t)byte_index] >> shift) & 0x03u;
                const float gate_value = fma(float(gate_code), gate_scale[row],
                    gate_bias[row]);
                const float up_value = fma(float(up_code), up_scale[row], up_bias[row]);
                gate_result[row] = fma(gate_value, input, gate_result[row]);
                up_result[row] = fma(up_value, input, up_result[row]);
            }
        }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (thread_index_in_simdgroup == 0 && row0 + (uint)row < (uint)mid_dim) {
            float g = gate;
            float u = up;
            if (limit > 1.0e-6f) {
                g = min(g, limit);
                u = clamp(u, -limit, limit);
            }
            const float silu = g / (1.0f + exp(-g));
            y[(size_t)slot * (size_t)mid_dim + row0 + (uint)row] =
                silu * u * weights[slot];
        }
    }
}
