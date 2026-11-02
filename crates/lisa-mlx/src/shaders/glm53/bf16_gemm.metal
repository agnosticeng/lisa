// GLM-5.3 BF16-weight/F32-activation prefill GEMM.
//
// This is the ds4 kernel_mul_mm tile and simdgroup schedule specialized to
// BF16 weights. The source operands are staged through F16 before F32
// simdgroup accumulation, matching ds4's kernel_glm53_mul_mm_bf16_f32 path.
{
    constexpr uint TILE_N = 64;
    constexpr uint TILE_M = 32;
    constexpr uint TILE_K = 32;

    threadgroup half shared_a[TILE_N * TILE_K];
    threadgroup half shared_b[TILE_N * TILE_K];
    threadgroup float shared_out[TILE_N * TILE_M];

    const uint tile_n = threadgroup_position_in_grid.y * TILE_N;
    const uint tile_m = threadgroup_position_in_grid.x * TILE_M;
    const uint n_valid = min((uint)out_dim - tile_n, TILE_N);
    const uint m_valid = min((uint)n_rows - tile_m, TILE_M);

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;

    const uint row_lane = tid / 2u;
    const uint m_lane = tid / 4u;
    const uint n_row = tile_n + min(row_lane, n_valid - 1u);
    const uint m_row = tile_m + min(m_lane, m_valid - 1u);
    const uint a_half = tid % 2u;
    const uint b_lane = tid % 4u;
    const uint b_offset = 8u * b_lane;

    device const bfloat16_t *weight_row = weights + (size_t)n_row * in_dim;
    device const float *activation_row = x + (size_t)m_row * in_dim;

    simdgroup_half8x8 a_fragments[4];
    simdgroup_half8x8 b_fragments[2];
    simdgroup_float8x8 accumulators[8];
    for (uint i = 0; i < 8u; ++i) {
        accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k_base = 0; k_base < (uint)in_dim; k_base += TILE_K) {
        // Retire every simdgroup's reads from the previous K tile before any
        // group overwrites shared_a/shared_b with the next one. ds4 places
        // this barrier after its private dequantize step and before staging.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread stages two 16-value weight chunks. Keep ds4's swizzle:
        // it is the layout consumed by its simdgroup_load sequence below.
        for (uint i = 0; i < 16u; ++i) {
            const uint sx = 2u * a_half + i / 8u;
            const uint sy = row_lane / 8u;
            const uint lx = row_lane % 8u;
            const uint ly = i % 8u;
            const uint index = 64u * (8u * sx + sy) + 8u * ly + lx;
            const uint k = k_base + 16u * a_half + i;
            half value = half(0.0f);
            if (n_row < (uint)out_dim && k < (uint)in_dim) {
                value = half(float(weights[(size_t)n_row * in_dim + k]));
            }
            shared_a[index] = value;
        }

        // The activation side uses ds4's four-way K interleave and swizzle.
        for (uint i = 0; i < 8u; ++i) {
            const uint sx = b_lane;
            const uint sy = m_lane / 8u;
            const uint ly = m_lane % 8u;
            const uint index = 64u * (4u * sx + sy) + 8u * ly + i;
            const uint k = k_base + b_offset + i;
            shared_b[index] = half(
                m_row < (uint)n_rows && k < (uint)in_dim
                    ? x[(size_t)m_row * in_dim + k]
                    : 0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *load_a = shared_a + 4u * 64u * (sg % 2u);
        threadgroup const half *load_b = shared_b + 2u * 64u * (sg / 2u);
        for (uint ik = 0; ik < TILE_K / 8u; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(a_fragments[i], load_a + 64u * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 2u; ++i) {
                simdgroup_load(b_fragments[i], load_b + 64u * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_multiply_accumulate(
                    accumulators[i], b_fragments[i / 4u],
                    a_fragments[i % 4u], accumulators[i]);
            }
            load_a += 8u * 64u;
            load_b += 4u * 64u;
        }
    }

    // ds4 selects its boundary-output kernel variant for the entire dispatch
    // when either matrix dimension has a tail. Keep that dispatch-wide choice
    // here too: otherwise interior tiles take the direct-store path while the
    // same logical operation in ds4 uses the shared tile and copy path.
    const bool boundary_output = (out_dim % TILE_N) != 0u || (n_rows % TILE_M) != 0u;
    if (!boundary_output) {
        device float *out_tile = out + tile_n + 32u * (sg & 1u) +
            (tile_m + 16u * (sg >> 1u)) * (size_t)out_dim;
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_store(
                accumulators[i],
                out_tile + 8u * (i % 4u) + 8u * (size_t)out_dim * (i / 4u),
                out_dim, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *local_tile = shared_out + 32u * (sg & 1u) +
            16u * (sg >> 1u) * TILE_N;
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_store(
                accumulators[i],
                local_tile + 8u * (i % 4u) + 8u * TILE_N * (i / 4u),
                TILE_N, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            for (uint m = tid; m < m_valid; m += TILE_M) {
                for (uint n = 0; n < n_valid; ++n) {
                    out[(size_t)(tile_m + m) * out_dim + tile_n + n] =
                        shared_out[(size_t)m * TILE_N + n];
                }
            }
        }
    }
}
