// GLM-5.3 DSA indexer tiled batch scorer, ported from ds4/dsv4_misc.metal.
// One 8-query x 32-pool tile per 4-simdgroup threadgroup. q/cache dot products
// use F16 simdgroup matrices and F32 accumulation, as in ds4's default path.
{
    constexpr uint TM = 8u;
    constexpr uint TN = 32u;
    constexpr uint TS = 8u;
    constexpr uint D = 128u;
    const uint row_base = threadgroup_position_in_grid.x * TN;
    const uint token_base = threadgroup_position_in_grid.y * TM;
    const uint batch = threadgroup_position_in_grid.z;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint pools = (uint)n_rows;
    const uint tokens = (uint)n_tokens;
    if (batch >= (uint)n_batches) return;

    threadgroup half shared_data[TM * D + TN * D];
    threadgroup half *qtg = shared_data;
    threadgroup half *ktg = qtg + TM * D;
    threadgroup float dot[4u * TS * TS];

    const uint last_token = min(token_base + TM, tokens);
    const uint max_visible = last_token > token_base
        ? min(((uint)pos0 + last_token) / (uint)pool_size, pools)
        : 0u;
    if (row_base >= max_visible) {
        for (uint i = tid; i < TM * TN; i += 128u) {
            const uint token = token_base + i / TN;
            const uint pool = row_base + i % TN;
            if (token < tokens && pool < pools) {
                scores[((size_t)batch * tokens + token) * pools + pool] = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN * D; i += 128u) {
        const uint col = i / D;
        const uint d = i - col * D;
        const uint pool = row_base + col;
        half value = half(0.0f);
        if (pool < pools) {
            value = half(float(pooled[((size_t)batch * pools + pool) * D + d]));
        }
        ktg[i] = value;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3u;
    const uint token_row1 = cell1 >> 3u;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = simdgroup * TS + sub0;
    const uint col1 = simdgroup * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0u; head < (uint)n_heads; ++head) {
        for (uint i = tid; i < TM * D; i += 128u) {
            const uint token_row = i / D;
            const uint d = i - token_row * D;
            const uint token = token_base + token_row;
            half value = half(0.0f);
            if (token < tokens) {
                const size_t q_index =
                    (((size_t)batch * (uint)n_heads + head) * tokens + token) * D + d;
                value = half(q[q_index]);
            }
            qtg[i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 matrix_dot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0u; db < D / TS; ++db) {
            simdgroup_half8x8 matrix_q;
            simdgroup_half8x8 matrix_k;
            simdgroup_load(matrix_q, qtg + db * TS, D, 0, false);
            simdgroup_load(matrix_k, ktg + simdgroup * TS * D + db * TS, D, 0, true);
            simdgroup_multiply_accumulate(matrix_dot, matrix_q, matrix_k, matrix_dot);
        }
        simdgroup_store(matrix_dot, dot + simdgroup * TS, TN, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < tokens && row0 < pools) {
            const float weight = weights[((size_t)batch * tokens + token0) *
                                          (uint)n_heads + head];
            const float value = dot[token_row0 * TN + col0];
            acc0 += max(value * scale, 0.0f) * weight;
        }
        if (token1 < tokens && row1 < pools) {
            const float weight = weights[((size_t)batch * tokens + token1) *
                                          (uint)n_heads + head];
            const float value = dot[token_row1 * TN + col1];
            acc1 += max(value * scale, 0.0f) * weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < tokens && row0 < pools) {
        const uint visible = min(((uint)pos0 + token0 + 1u) / (uint)pool_size, pools);
        scores[((size_t)batch * tokens + token0) * pools + row0] =
            row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < tokens && row1 < pools) {
        const uint visible = min(((uint)pos0 + token1 + 1u) / (uint)pool_size, pools);
        scores[((size_t)batch * tokens + token1) * pools + row1] =
            row1 < visible ? acc1 : -INFINITY;
    }
}
