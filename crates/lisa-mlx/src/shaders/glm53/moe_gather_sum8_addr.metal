// GLM-5.3 routed down projection with DS4's selected-expert loop schedule.
// Each threadgroup owns one output-row tile and accumulates the eight routed
// experts in slot order, avoiding the intermediate [8, hidden] output and a
// second reduction dispatch. Each lane uses DS4's affine-dequant FMA and
// activation FMA, then SIMD-reduces each expert before ordered slot addition.
// Optional shared outputs are added after the routed sum; Q8 shared-down
// follows MLX's group-64 qmv_fast layout.
{
    constexpr int num_simdgroups = 4;
    constexpr int results_per_simdgroup = 1;
    constexpr int group_size = 128;

    const uint token = threadgroup_position_in_grid.y / groups_per_token;
    const uint row0 = (threadgroup_position_in_grid.y % groups_per_token) *
            (num_simdgroups * results_per_simdgroup) +
        simdgroup_index_in_threadgroup * results_per_simdgroup;
    if (token >= (uint)n_tokens || row0 >= (uint)out_rows) return;

    const int in_vec_size_w = in_vec_size / 4;
    const int in_vec_size_g = in_vec_size / group_size;
    typedef float U;
    thread U total[results_per_simdgroup] = {0};
    thread U shared_x_thread[8];
    thread U shared_result[results_per_simdgroup] = {0};
    U shared_slot_value[results_per_simdgroup] = {0};

    for (uint slot = 0; slot < (uint)slots_per_token; slot++) {
        const uint selected_index = token * (uint)slots_per_token + slot;
        const int expert = selected[selected_index];
        if (expert < 0 || expert >= n_experts) continue;

        const size_t address_base = (size_t)expert * 9;
        const device uint8_t *w = reinterpret_cast<const device uint8_t *>(
            as_type<ulong>(address_table[address_base + 6]));
        const device M *scales = reinterpret_cast<const device M *>(
            as_type<ulong>(address_table[address_base + 7]));
        const device M *biases = reinterpret_cast<const device M *>(
            as_type<ulong>(address_table[address_base + 8]));
        const device X *xx = x + (size_t)selected_index * (size_t)in_vec_size;
        w += (size_t)row0 * (size_t)in_vec_size_w;
        scales += (size_t)row0 * (size_t)in_vec_size_g;
        biases += (size_t)row0 * (size_t)in_vec_size_g;
        thread U result[results_per_simdgroup] = {0};

        for (int group = 0; group < in_vec_size_g; group++) {
            thread U scale[results_per_simdgroup];
            thread U bias[results_per_simdgroup];
            for (int row = 0; row < results_per_simdgroup; row++) {
                const size_t metadata_index = (size_t)row * (size_t)in_vec_size_g +
                    (size_t)group;
                scale[row] = float(scales[metadata_index]);
                bias[row] = float(biases[metadata_index]);
            }
            // A lane contributes four values per group. Keep DS4's exact
            // order but reuse each scale/bias pair for those four FMAs.
            #pragma unroll
            for (int item = 0; item < group_size / SIMD_SIZE; item++) {
                const int k = group * group_size +
                    (int)thread_index_in_simdgroup + item * SIMD_SIZE;
                const int byte_index = k / 4;
                const int shift = (k & 3) * 2;
                const float input = float(xx[k]);
                for (int row = 0; row < results_per_simdgroup; row++) {
                    const device uint8_t *row_w = w + (size_t)row * (size_t)in_vec_size_w;
                    const uint8_t code = (row_w[byte_index] >> shift) & 0x03u;
                    const float value = fma(float(code), scale[row], bias[row]);
                    result[row] = fma(value, input, result[row]);
                }
            }
        }

        for (int row = 0; row < results_per_simdgroup; row++) {
            const U slot_value = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                total[row] += slot_value;
            }
        }
    }

    if (has_shared_down != 0) {
        constexpr int shared_packs_per_thread = 2;
        constexpr int shared_pack_factor = get_pack_factor<8, 32>();
        constexpr int shared_bytes_per_pack = get_bytes_per_pack<8, 32>();
        constexpr int shared_values_per_thread = shared_pack_factor * shared_packs_per_thread;
        constexpr int shared_block_size = shared_values_per_thread * SIMD_SIZE;
        constexpr int shared_scale_step = 64 / shared_values_per_thread;
        const int shared_in_vec_size_w =
            shared_down_in_vec_size * shared_bytes_per_pack / shared_pack_factor;
        const int shared_in_vec_size_g = shared_down_in_vec_size / 64;
        const device T *shared_x = shared_down_x +
            (size_t)token * (size_t)shared_down_in_vec_size +
            (size_t)thread_index_in_simdgroup * shared_values_per_thread;
        const device uint8_t *shared_w = reinterpret_cast<const device uint8_t *>(shared_down_w) +
            (size_t)row0 * (size_t)shared_in_vec_size_w +
            (size_t)thread_index_in_simdgroup * shared_packs_per_thread * shared_bytes_per_pack;
        const device T *shared_scales = shared_down_scales +
            (size_t)row0 * (size_t)shared_in_vec_size_g +
            (size_t)(thread_index_in_simdgroup / shared_scale_step);
        const device T *shared_biases = shared_down_biases +
            (size_t)row0 * (size_t)shared_in_vec_size_g +
            (size_t)(thread_index_in_simdgroup / shared_scale_step);
        for (int k = 0; k < shared_down_in_vec_size; k += shared_block_size) {
            U sum = load_vector<T, U, shared_values_per_thread, 8>(shared_x, shared_x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
                const device uint8_t *row_w =
                    shared_w + (size_t)row * (size_t)shared_in_vec_size_w;
                const U scale = U(shared_scales[(size_t)row * (size_t)shared_in_vec_size_g]);
                const U bias = U(shared_biases[(size_t)row * (size_t)shared_in_vec_size_g]);
                shared_result[row] += qdot<U, shared_values_per_thread, 8>(
                    row_w, shared_x_thread, scale, bias, sum);
            }
            shared_w += (size_t)shared_block_size * shared_bytes_per_pack / shared_pack_factor;
            shared_scales += shared_block_size / 64;
            shared_biases += shared_block_size / 64;
            shared_x += shared_block_size;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            shared_slot_value[row] = simd_sum(shared_result[row]);
        }
    }

    if (thread_index_in_simdgroup == 0) {
        device T *yy = out + (size_t)token * (size_t)out_rows + row0;
        for (int row = 0; row < results_per_simdgroup; row++) {
            if (row0 + (uint)row < (uint)out_rows) {
                U value = total[row];
                if (has_shared != 0) {
                    value += U(shared[(size_t)token * (size_t)out_rows + row0 + (uint)row]);
                }
                if (has_shared_down != 0) {
                    value += shared_slot_value[row];
                }
                yy[row] = static_cast<T>(value);
            }
        }
    }
}
