// GLM-5.3 routed Q2 affine QMV through MLX-native per-expert GPU addresses.
// The address table is [expert, 9]; down projection uses fields 6..8.
// Reduction association matches moe_gather.metal.
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
    if (slot >= (uint)n_slots) return;

    const int expert = selected[slot];
    if (expert < 0 || expert >= n_experts) return;
    const size_t address_base = (size_t)expert * 9;
    const device uint8_t *w = reinterpret_cast<const device uint8_t *>(
        as_type<ulong>(address_table[address_base + 6]));
    const device M *scales = reinterpret_cast<const device M *>(
        as_type<ulong>(address_table[address_base + 7]));
    const device M *biases = reinterpret_cast<const device M *>(
        as_type<ulong>(address_table[address_base + 8]));
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / 128;

    typedef float U;
    thread U x_thread[values_per_thread];
    thread U result[results_per_simdgroup] = {0};
    const device X *xx = x + (size_t)(x_per_slot * (int)slot) * (size_t)in_vec_size +
        (size_t)thread_index_in_simdgroup * values_per_thread;

    w += (size_t)row0 * (size_t)in_vec_size_w +
        (size_t)thread_index_in_simdgroup * bytes_per_pack;
    scales += (size_t)row0 * (size_t)in_vec_size_g +
        (size_t)(thread_index_in_simdgroup / scale_step_per_thread);
    biases += (size_t)row0 * (size_t)in_vec_size_g +
        (size_t)(thread_index_in_simdgroup / scale_step_per_thread);
    device T *yy = y + (size_t)slot * (size_t)out_rows + row0;

    for (int k = 0; k < in_vec_size; k += block_size) {
        U sum = load_vector<X, U, values_per_thread, 2>(xx, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
            const device uint8_t *row_w = w + (size_t)row * (size_t)in_vec_size_w;
            const float scale = float(scales[(size_t)row * (size_t)in_vec_size_g]);
            const float bias = float(biases[(size_t)row * (size_t)in_vec_size_g]);
            result[row] += qdot<U, values_per_thread, 2>(row_w, x_thread, scale, bias, sum);
        }
        w += (size_t)block_size * bytes_per_pack / pack_factor;
        scales += block_size / 128;
        biases += block_size / 128;
        xx += block_size;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (thread_index_in_simdgroup == 0) {
            yy[row] = static_cast<T>(result[row]);
        }
    }
}
