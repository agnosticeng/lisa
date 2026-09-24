// Block-sparse QSA prefill attention kernel (NAX). See NOTICE.
    constexpr int HEAD_DIM = 256;
    constexpr int KV_HEADS = 2;
    constexpr int GQA = 12;
    constexpr int BLOCK_TOKENS = 4;
    constexpr int TOP_K_BLOCKS = 512;
    constexpr int M_ROWS = 16;
    constexpr int TILE_BLOCKS = 8;
    constexpr int TOKENS_PER_TILE = TILE_BLOCKS * BLOCK_TOKENS;
    constexpr int MAX_SELECTED_TILES = TOP_K_BLOCKS / TILE_BLOCKS;
    constexpr int D_FRAGS = HEAD_DIM / 16;
    constexpr int OUT_GROUPS = HEAD_DIM / 32;

    // MLX Steel's wide-load idiom: a byte aggregate aligned only to one T.
    // Unlike vec<T,4>, this remains defined for unit-stride views whose base
    // begins at an odd fp16/bf16 element offset.
    struct alignas(sizeof(T)) QSAReadVector8 {
        uchar bytes[sizeof(T) * 8];
    };

    const ushort lane = ushort(thread_index_in_simdgroup);
    const int work = int(threadgroup_position_in_grid.x);
    const int row = work / KV_HEADS;
    const int kv_head = work - row * KV_HEADS;
    const int pos_start = int(params[0]);
    const int total_tokens = int(params[1]);
    const int query_pos = pos_start + row;
    const int complete_blocks = (query_pos + 1) / BLOCK_TOKENS;
    const int tail_start = complete_blocks * BLOCK_TOKENS;
    const int tail_count = query_pos + 1 - tail_start;
    // Find the highest selection slot that can actually contribute.  The
    // production selector emits a chronological valid prefix, but deriving
    // this from the slots themselves preserves the public kernel's existing
    // semantics for arbitrary validity holes as well.  Encoding slot+1 lets
    // zero mean "no selected tile" in the simd-wide max reduction.
    uint local_active_slots = 0u;
    for (uint block_slot = uint(lane);
         block_slot < uint(TOP_K_BLOCKS);
         block_slot += 32u) {
        const size_t id_at = size_t(row) * block_ids_strides[0] +
            size_t(block_slot) * block_ids_strides[1];
        const size_t valid_at = size_t(row) * block_valid_strides[0] +
            size_t(block_slot) * block_valid_strides[1];
        const int block_id = int(block_ids[id_at]);
        if (bool(block_valid[valid_at]) &&
            block_id >= 0 && block_id < complete_blocks) {
            local_active_slots = metal::max(local_active_slots, block_slot + 1u);
        }
    }
    const uint active_blocks = simd_max(local_active_slots);
    const int active_selected_tiles =
        (int(active_blocks) + TILE_BLOCKS - 1) / TILE_BLOCKS;
    const int active_tiles = active_selected_tiles + (tail_count > 0 ? 1 : 0);
    const short2 sc = qsa_nax_coord(lane);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> mm;
    auto ct_a = mm.get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = mm.get_right_input_cooperative_tensor<T, T, float>();
    // macOS 27 MPP SDK gates get_destination_cooperative_tensor on
    // __is_tensor_type_v / __is_cooperative_tensor_type_v, which an
    // address-space-qualified decltype (thread ...) fails (issue #404).
    // Strip the qualifier, matching mlx's own steel/gemm/nax.h pattern.
    auto ct_c = mm.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();

    // Exactly 32 * 256 * sizeof(T) == 16 KiB. The allocation first holds K
    // row-major and is then overwritten with V-transpose for the PV multiply.
    threadgroup T tg_tile[TOKENS_PER_TILE * HEAD_DIM];

    // Each lane carries two padded M rows. Four lanes share a logical row;
    // xor(1),xor(8) reduce the four score fragments for that row.
    float row_max[2] = {-1.0e38f, -1.0e38f};
    float row_sum[2] = {0.0f, 0.0f};
    float out_frag[OUT_GROUPS][2][QSA_ELEMS_PER_FRAG];
    for (int group = 0; group < OUT_GROUPS; ++group) {
        for (short row_part = 0; row_part < 2; ++row_part) {
            for (short elem = 0; elem < QSA_ELEMS_PER_FRAG; ++elem) {
                out_frag[group][row_part][elem] = 0.0f;
            }
        }
    }

    for (int tile_index = 0; tile_index < active_tiles; ++tile_index) {
        const bool tail_tile =
            tail_count > 0 && tile_index == active_selected_tiles;
        const int block_base = tile_index * TILE_BLOCKS;

        // Gather K into tg_tile[token,dim]. Metadata is deliberately recomputed
        // by the one simdgroup so the kernel owns only one 16 KiB TG allocation.
        for (int token_slot = 0; token_slot < TOKENS_PER_TILE; ++token_slot) {
            int token = 0;
            bool token_valid = false;
            if (tail_tile) {
                token = tail_start + token_slot;
                token_valid = token_slot < tail_count && token < total_tokens;
            } else {
                const int local_block = token_slot / BLOCK_TOKENS;
                const int within = token_slot - local_block * BLOCK_TOKENS;
                const int block_slot = block_base + local_block;
                const size_t id_at = size_t(row) * block_ids_strides[0] +
                    size_t(block_slot) * block_ids_strides[1];
                const size_t valid_at = size_t(row) * block_valid_strides[0] +
                    size_t(block_slot) * block_valid_strides[1];
                const int block_id = int(block_ids[id_at]);
                token_valid = bool(block_valid[valid_at]) &&
                    block_id >= 0 && block_id < complete_blocks;
                token = token_valid ? block_id * BLOCK_TOKENS + within : 0;
            }
            if (k_strides[3] == 1) {
                // Each lane exposes one compiler-visible 16-byte copy. Across
                // the simdgroup the full 256-wide cache row is contiguous.
                const int dim0 = int(lane) * 8;
                const int destination = token_slot * HEAD_DIM + dim0;
                if (token_valid) {
                    const size_t k_at = size_t(kv_head) * k_strides[1] +
                        size_t(token) * k_strides[2] + size_t(dim0);
                    *reinterpret_cast<threadgroup QSAReadVector8*>(
                        &tg_tile[destination]) =
                        *reinterpret_cast<const device QSAReadVector8*>(&k[k_at]);
                } else {
                    for (short elem = 0; elem < 8; ++elem) {
                        tg_tile[destination + int(elem)] = T(0);
                    }
                }
            } else {
                // Fail-correct for unusual cache views whose feature axis is
                // not contiguous; the production backing takes the wide-copy lane.
                for (int dim = int(lane); dim < HEAD_DIM; dim += 32) {
                    const size_t k_at = size_t(kv_head) * k_strides[1] +
                        size_t(token) * k_strides[2] +
                        size_t(dim) * k_strides[3];
                    tg_tile[token_slot * HEAD_DIM + dim] =
                        token_valid ? k[k_at] : T(0);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // QK: padded [16,256] x gathered [32,256]^T -> [16,32].
        for (short elem = 0; elem < 2 * QSA_ELEMS_PER_FRAG; ++elem) {
            ct_c[elem] = 0.0f;
        }
        for (int frag = 0; frag < D_FRAGS; ++frag) {
            for (short row_part = 0; row_part < 2; ++row_part) {
                const int m_row = int(sc.y) + int(row_part) * QSA_ELEM_ROWS_JUMP;
                if (m_row < GQA) {
                    const int q_head = kv_head * GQA + m_row;
                    const size_t q_base = size_t(q_head) * q_strides[1] +
                        size_t(row) * q_strides[2] +
                        size_t(frag * 16 + int(sc.x)) * q_strides[3];
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_a[row_part * QSA_ELEM_COLS + col] =
                            q[q_base + size_t(col) * q_strides[3]];
                    }
                } else {
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_a[row_part * QSA_ELEM_COLS + col] = T(0);
                    }
                }
            }
            for (short n_half = 0; n_half < 2; ++n_half) {
                for (short row_half = 0; row_half < 2; ++row_half) {
                    const int token_slot = int(n_half) * 16 + int(sc.y) +
                        int(row_half) * QSA_ELEM_ROWS_JUMP;
                    const int tile_base = token_slot * HEAD_DIM +
                        frag * 16 + int(sc.x);
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_b[n_half * QSA_ELEMS_PER_FRAG +
                             row_half * QSA_ELEM_COLS + col] =
                            tg_tile[tile_base + int(col)];
                    }
                }
            }
            mm.run(ct_a, ct_b, ct_c);
        }

        // Mask, scale, and exponentiate the score fragment in registers.
        float probabilities[2][QSA_ELEMS_PER_FRAG];
        float correction[2];
        for (short row_part = 0; row_part < 2; ++row_part) {
            const int m_row = int(sc.y) + int(row_part) * QSA_ELEM_ROWS_JUMP;
            const bool live_row = m_row < GQA;
            float tile_max = -1.0e38f;
            for (short n_half = 0; n_half < 2; ++n_half) {
                for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                    const int token_slot = int(n_half) * 16 + int(sc.x) + int(col);
                    bool token_valid = false;
                    if (tail_tile) {
                        const int token = tail_start + token_slot;
                        token_valid = token_slot < tail_count && token < total_tokens;
                    } else {
                        const int local_block = token_slot / BLOCK_TOKENS;
                        const int block_slot = block_base + local_block;
                        const size_t id_at = size_t(row) * block_ids_strides[0] +
                            size_t(block_slot) * block_ids_strides[1];
                        const size_t valid_at = size_t(row) * block_valid_strides[0] +
                            size_t(block_slot) * block_valid_strides[1];
                        const int block_id = int(block_ids[id_at]);
                        token_valid = bool(block_valid[valid_at]) &&
                            block_id >= 0 && block_id < complete_blocks;
                    }
                    const short at = row_part * QSA_ELEM_COLS + col;
                    const float score = live_row && token_valid
                        ? ct_c[n_half * QSA_ELEMS_PER_FRAG + at] * scale[0]
                        : -1.0e38f;
                    probabilities[n_half][at] = score;
                    tile_max = metal::max(tile_max, score);
                }
            }
            tile_max = metal::max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
            tile_max = metal::max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
            const float new_max = metal::max(row_max[row_part], tile_max);
            correction[row_part] = metal::exp(row_max[row_part] - new_max);
            float tile_sum = 0.0f;
            for (short n_half = 0; n_half < 2; ++n_half) {
                for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                    const short at = row_part * QSA_ELEM_COLS + col;
                    const float score = probabilities[n_half][at];
                    const float probability = score > -1.0e37f
                        ? metal::exp(score - new_max)
                        : 0.0f;
                    probabilities[n_half][at] = probability;
                    tile_sum += probability;
                }
            }
            tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
            tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
            row_max[row_part] = new_max;
            row_sum[row_part] = row_sum[row_part] * correction[row_part] + tile_sum;
        }

        for (short row_part = 0; row_part < 2; ++row_part) {
            const float factor = correction[row_part];
            for (int group = 0; group < OUT_GROUPS; ++group) {
                for (short dim_half = 0; dim_half < 2; ++dim_half) {
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        out_frag[group][dim_half]
                                [row_part * QSA_ELEM_COLS + col] *= factor;
                    }
                }
            }
        }

        // QK is finished: overwrite the same 16 KiB with V^T[dim,token].
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int token_slot = 0; token_slot < TOKENS_PER_TILE; ++token_slot) {
            int token = 0;
            bool token_valid = false;
            if (tail_tile) {
                token = tail_start + token_slot;
                token_valid = token_slot < tail_count && token < total_tokens;
            } else {
                const int local_block = token_slot / BLOCK_TOKENS;
                const int within = token_slot - local_block * BLOCK_TOKENS;
                const int block_slot = block_base + local_block;
                const size_t id_at = size_t(row) * block_ids_strides[0] +
                    size_t(block_slot) * block_ids_strides[1];
                const size_t valid_at = size_t(row) * block_valid_strides[0] +
                    size_t(block_slot) * block_valid_strides[1];
                const int block_id = int(block_ids[id_at]);
                token_valid = bool(block_valid[valid_at]) &&
                    block_id >= 0 && block_id < complete_blocks;
                token = token_valid ? block_id * BLOCK_TOKENS + within : 0;
            }
            if (v_strides[3] == 1) {
                const int dim0 = int(lane) * 8;
                thread T v_lane[8];
                if (token_valid) {
                    const size_t v_at = size_t(kv_head) * v_strides[1] +
                        size_t(token) * v_strides[2] + size_t(dim0);
                    *reinterpret_cast<thread QSAReadVector8*>(&v_lane[0]) =
                        *reinterpret_cast<const device QSAReadVector8*>(&v[v_at]);
                } else {
                    for (short elem = 0; elem < 8; ++elem) {
                        v_lane[elem] = T(0);
                    }
                }
                for (short elem = 0; elem < 8; ++elem) {
                    tg_tile[(dim0 + int(elem)) * TOKENS_PER_TILE + token_slot] =
                        v_lane[elem];
                }
            } else {
                for (int dim = int(lane); dim < HEAD_DIM; dim += 32) {
                    const size_t v_at = size_t(kv_head) * v_strides[1] +
                        size_t(token) * v_strides[2] +
                        size_t(dim) * v_strides[3];
                    tg_tile[dim * TOKENS_PER_TILE + token_slot] =
                        token_valid ? v[v_at] : T(0);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // PV: [16,32] x [256,32]^T -> [16,256]. MPP consumes probabilities
        // in T, while accumulation and all persistent online state remain fp32.
        for (int group = 0; group < OUT_GROUPS; ++group) {
            for (short elem = 0; elem < QSA_ELEMS_PER_FRAG; ++elem) {
                ct_c[elem] = out_frag[group][0][elem];
                ct_c[QSA_ELEMS_PER_FRAG + elem] = out_frag[group][1][elem];
            }
            for (short token_half = 0; token_half < 2; ++token_half) {
                for (short row_part = 0; row_part < 2; ++row_part) {
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_a[row_part * QSA_ELEM_COLS + col] = T(
                            probabilities[token_half][row_part * QSA_ELEM_COLS + col]);
                    }
                }
                for (short dim_half = 0; dim_half < 2; ++dim_half) {
                    for (short row_half = 0; row_half < 2; ++row_half) {
                        const int dim = group * 32 + int(dim_half) * 16 +
                            int(sc.y) + int(row_half) * QSA_ELEM_ROWS_JUMP;
                        const int tile_base = dim * TOKENS_PER_TILE +
                            int(token_half) * 16 + int(sc.x);
                        for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                            ct_b[dim_half * QSA_ELEMS_PER_FRAG +
                                 row_half * QSA_ELEM_COLS + col] =
                                tg_tile[tile_base + int(col)];
                        }
                    }
                }
                mm.run(ct_a, ct_b, ct_c);
            }
            for (short elem = 0; elem < QSA_ELEMS_PER_FRAG; ++elem) {
                out_frag[group][0][elem] = ct_c[elem];
                out_frag[group][1][elem] = ct_c[QSA_ELEMS_PER_FRAG + elem];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and store the twelve live rows into contiguous [1,24,S,256].
    for (short row_part = 0; row_part < 2; ++row_part) {
        const int m_row = int(sc.y) + int(row_part) * QSA_ELEM_ROWS_JUMP;
        if (m_row >= GQA) continue;
        const int q_head = kv_head * GQA + m_row;
        const float inv_sum = row_sum[row_part] > 0.0f ? 1.0f / row_sum[row_part] : 0.0f;
        const size_t out_base =
            (size_t(q_head) * size_t(params[2]) + size_t(row)) * HEAD_DIM;
        for (int group = 0; group < OUT_GROUPS; ++group) {
            for (short dim_half = 0; dim_half < 2; ++dim_half) {
                for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                    const int dim = group * 32 + int(dim_half) * 16 +
                        int(sc.x) + int(col);
                    out[out_base + size_t(dim)] = T(
                        out_frag[group][dim_half][row_part * QSA_ELEM_COLS + col] *
                        inv_sum);
                }
            }
        }
    }
