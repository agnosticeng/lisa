// MoE prefill expert gate|up projection + SwiGLU over the tile table.
        alignas(16) threadgroup T Ws0[64 * 72];
        alignas(16) threadgroup T Ws1[64 * 72];
        alignas(16) threadgroup T As[32 * 72];
        track_prefill_indirect_gu<T, 32, 4, 32, 64, 64, 2, 2, true, SILU, N>(
            x, w0, scales0, biases0, w1, scales1, biases1, indices, token_rows, tiles,
            y, y, N, K, Ws0, Ws1, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    