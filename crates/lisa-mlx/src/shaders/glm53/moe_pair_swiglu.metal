// GLM-5.3 gathered gate/up quantized matmul fused with clamped SwiGLU and
// router weighting. Weight and quantization storage remains MLX-native; the
// expert-selection and activation order follows ds4's pair-SwiGLU path.
{
    constexpr int packs_per_thread = 1;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = get_pack_factor<2, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<2, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = 128 / values_per_thread;

    const uint slot = threadgroup_position_in_grid.y / groups_per_slot;
    const uint row0 = (threadgroup_position_in_grid.y % groups_per_slot) *
            (num_simdgroups * results_per_simdgroup) +
        simdgroup_index_in_threadgroup * results_per_simdgroup;
    if (slot >= (uint)n_slots || row0 >= (uint)mid_dim) return;

    const int expert = selected[slot];
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / 128;
    const size_t expert_rows = (size_t)expert * (size_t)(2 * mid_dim);
    const size_t row_bytes = (size_t)in_vec_size_w;
    const size_t row_groups = (size_t)in_vec_size_g;

    typedef float U;
    thread U x_thread[values_per_thread];
    thread U gate_result[results_per_simdgroup] = {0};
    thread U up_result[results_per_simdgroup] = {0};

    const device uint8_t *ws = (const device uint8_t *)w;
    const device T *sl = scales;
    const device T *bl = biases;
    ws += (expert_rows + row0) * row_bytes +
        (size_t)thread_index_in_simdgroup * (packs_per_thread * bytes_per_pack);
    sl += (expert_rows + row0) * row_groups +
        (size_t)(thread_index_in_simdgroup / scale_step_per_thread);
    bl += (expert_rows + row0) * row_groups +
        (size_t)(thread_index_in_simdgroup / scale_step_per_thread);
    const uint token = slot / (uint)slots_per_token;
    const device T *xx = x + (size_t)token * (size_t)in_vec_size +
        (size_t)thread_index_in_simdgroup * values_per_thread;

    const size_t up_row_offset = (size_t)mid_dim;
    for (int k = 0; k < in_vec_size; k += block_size) {
        U sum = load_vector<T, U, values_per_thread, 2>(xx, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
            const size_t row_offset = (size_t)row;
            const device uint8_t *gate_w = ws + row_offset * row_bytes;
            const device uint8_t *up_w = ws + (up_row_offset + row_offset) * row_bytes;
            const T gate_s = sl[row_offset * row_groups];
            const T gate_b = bl[row_offset * row_groups];
            const T up_s = sl[(up_row_offset + row_offset) * row_groups];
            const T up_b = bl[(up_row_offset + row_offset) * row_groups];
            gate_result[row] += qdot<U, values_per_thread, 2>(gate_w, x_thread, gate_s, gate_b, sum);
            up_result[row] += qdot<U, values_per_thread, 2>(up_w, x_thread, up_s, up_b, sum);
        }
        ws += (size_t)block_size * bytes_per_pack / pack_factor;
        sl += block_size / 128;
        bl += block_size / 128;
        xx += block_size;
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
