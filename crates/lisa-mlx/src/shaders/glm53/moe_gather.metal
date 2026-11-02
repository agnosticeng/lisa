// GLM-5.3 routed-MoE gathered qmv (ds4 moe.metal parity: `selected` is read
// INSIDE the kernel — no host round-trip for the expert dispatch).
// Body only — the framework generates the kernel signature from the wrapper's
// param list (n_slots, groups_per_slot, out_rows, in_vec_size, x_per_slot, w,
// scales, biases, x, selected, y) and injects threadgroup/simdgroup indices.
//
// The qmv math is MLX's `qmv_fast_impl` verbatim (same helpers, same
// association: per-thread qdot blocks, simd_sum over 32 lanes, one write per
// row) — the only delta is the expert indirection: each slot's rows live at
// `expert * out_rows` inside the stacked pools (axis-0 stack of the per-expert
// quantized linears), and `selected[slot]` picks the expert on the GPU.
//
// T = the compute dtype of x/scales/biases/out (F32 for the GLM-5.3 graph).
// group_size = 128, bits = 2 (glm53 routed experts).
{
    constexpr int packs_per_thread = 1; // bits == 2
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = get_pack_factor<2, 32>(); // 16
    constexpr int bytes_per_pack = get_bytes_per_pack<2, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread; // 16
    constexpr int block_size = values_per_thread * SIMD_SIZE;         // 512
    constexpr int scale_step_per_thread = 128 / values_per_thread;    // 8

    const uint slot = threadgroup_position_in_grid.y / groups_per_slot;
    const uint row0 = (threadgroup_position_in_grid.y % groups_per_slot) *
            (num_simdgroups * results_per_simdgroup) +
        simdgroup_index_in_threadgroup * results_per_simdgroup;
    if (slot >= (uint)n_slots) return;

    const int expert = selected[slot];
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / 128;

    typedef float U;
    thread U x_thread[values_per_thread];
    thread U result[results_per_simdgroup] = {0};

    const device uint8_t *ws = (const device uint8_t *)w;
    ws += ((size_t)expert * (size_t)out_rows + (size_t)row0) * (size_t)in_vec_size_w +
        (size_t)thread_index_in_simdgroup * (packs_per_thread * bytes_per_pack);
    device const T *sl = scales +
        ((size_t)expert * (size_t)out_rows + (size_t)row0) * (size_t)in_vec_size_g +
        (size_t)(thread_index_in_simdgroup / scale_step_per_thread);
    device const T *bl = biases +
        ((size_t)expert * (size_t)out_rows + (size_t)row0) * (size_t)in_vec_size_g +
        (size_t)(thread_index_in_simdgroup / scale_step_per_thread);
    device const T *xx = x + (size_t)(x_per_slot * (int)slot) * (size_t)in_vec_size +
        (size_t)thread_index_in_simdgroup * values_per_thread;
    device T *yy = y + (size_t)slot * (size_t)out_rows + row0;

    for (int k = 0; k < in_vec_size; k += block_size) {
        U sum = load_vector<T, U, values_per_thread, 2>(xx, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (const device uint8_t *)(ws + (size_t)row * (size_t)in_vec_size_w);
            U s = sl[(size_t)row * (size_t)in_vec_size_g];
            U b = bl[(size_t)row * (size_t)in_vec_size_g];
            result[row] += qdot<U, values_per_thread, 2>(wl, x_thread, s, b, sum);
        }
        ws += (size_t)block_size * bytes_per_pack / pack_factor;
        sl += block_size / 128;
        bl += block_size / 128;
        xx += block_size;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (thread_index_in_simdgroup == 0) {
            yy[row] = static_cast<T>(result[row]);
        }
    }
}
