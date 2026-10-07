// MoE prefill expert down projection over the tile table.
        alignas(16) threadgroup T Ws[128 * 40];
        alignas(16) threadgroup T As[32 * 40];
        track_prefill_indirect<T, 32, 4, 32, 128, 32, 2, 2, true, N, true>(
            x, w, scales, biases, indices, token_rows, tiles, y,
            N, K, Ws, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    