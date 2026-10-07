// Declarations of the wide (2..8-row) GEMV helpers for the one-token instantiations.
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, bool SILU>
        METAL_FUNC void qmv_wide_reg_full(
            const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
            const int in_vec_size, const int M, const int row, uint simd_lid, thread float (&result)[vecs_per_tg]);
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>
        METAL_FUNC void qmv_wide_reg_partial(
            const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
            const int in_vec_size, const int out_vec_size, const int M, threadgroup float* fold_partials,
            uint simd_gid, uint simd_lid, thread float (&result)[vecs_per_tg], thread bool& valid, thread int& row_out);
        