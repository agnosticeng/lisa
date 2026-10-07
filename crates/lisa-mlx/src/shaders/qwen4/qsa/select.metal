// Fused QSA indexer block selector.
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const int qpos = pos_start[0] + int(row);
        const int total = total_tokens[0];
        const int logical_value = logical_blocks[0];
        const uint logical = logical_value > 0
            ? metal::min(uint(logical_value), BACKING_BLOCKS)
            : 0u;
        // Mirror Matmul::eval_gpu's check_transpose + batch-collapse route.
        // A non-f32 astype materializes contiguous storage. For an existing
        // f32 view, check_transpose preserves recognized row/transposed
        // layouts and copies anything else. A copied operand is contiguous;
        // a copied broadcast B, however, no longer has a zero S-batch stride
        // and prevents folding S into M. When folding does not happen, M is
        // HEADS. MLX sends min(M,N)==1 to full-fp32 GEMV before NAX/TF32.
        const bool q_is_vector = HEADS == 1u;
        const bool q_kept_untransposed = Q_INPUT_IS_FLOAT32 &&
            q_strides[3] == 1u &&
            (!q_is_vector || q_strides[2] == HEAD_DIM);
        const bool q_kept_transposed = Q_INPUT_IS_FLOAT32 &&
            !q_kept_untransposed && q_strides[2] == 1u &&
            (!q_is_vector || q_strides[3] == HEADS);
        const bool q_copied =
            !Q_INPUT_IS_FLOAT32 ||
            (!q_kept_untransposed && !q_kept_transposed);
        const bool q_batch_contiguous = q_copied ||
            (q_kept_untransposed && q_strides[2] == HEAD_DIM &&
             q_strides[1] == size_t(HEADS) * HEAD_DIM);

        // pooled:[1,N,D] is swapped to B:[1,1,D,N]. For N>1 its two
        // recognized matrix layouts correspond to either original stride
        // being one. Otherwise check_transpose copies the broadcast view.
        const bool pooled_kept = !POOLED_INPUT_IS_FLOAT32 ||
            pooled_strides[1] == 1u || pooled_strides[2] == 1u;
        const bool collapse_s_into_m = q_shape[1] > 1u &&
            !q_kept_transposed && q_batch_contiguous && pooled_kept;
        const bool effective_m_gt_one =
            HEADS > 1u || collapse_s_into_m;
        const bool use_tf32_operands = ENABLE_TF32 &&
            effective_m_gt_one && logical > 1u;
        const uint gemm_operand_mask =
            use_tf32_operands ? 0xffffe000u : 0xffffffffu;
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(logical, complete);

        threadgroup float exchange_scores[WIDTH];
        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup atomic_uint selected_count;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup ulong threshold_key;

        // Score every visible complete block once. The scratch output keeps
        // those fp32 adjusted scores available to all eight radix passes
        // without recomputing HEADS*HEAD_DIM dot products. Only the visible
        // prefix is written/read; backing capacity does not tax an early row.
        const size_t scratch_base = (size_t)row * BACKING_BLOCKS;
        // The innermost dim is unit-stride and 8-element aligned for the
        // production layouts (a contiguous q view, the contiguous pooled bank);
        // when it is, the score loop loads 8 bf16 per transaction and keeps the
        // exact dim order, so the fp32 accumulation is bit-identical.
        const bool vector_loads = q_strides[3] == 1u && pooled_strides[2] == 1u &&
            (q_strides[1] % 8u) == 0u && (q_strides[2] % 8u) == 0u &&
            (pooled_strides[1] % 8u) == 0u;
        // Stage this query row's `[HEADS, HEAD_DIM]` once. Every lane's score
        // loop reads the same q element for a given (head, dim), so the
        // threadgroup read broadcasts and the device q traffic drops from
        // once-per-block to once-per-row.
        threadgroup float q_shared[HEADS * HEAD_DIM];
        const bool stage_q =
            q_strides[3] == 1u && q_strides[2] == HEAD_DIM;
        if (stage_q) {
            for (uint i = lane; i < HEADS * HEAD_DIM; i += WIDTH) {
                const uint head = i / HEAD_DIM;
                const uint dim = i % HEAD_DIM;
                q_shared[i] = qsa_mlx_gemm_operand(
                    float(q[(size_t)row * q_strides[1] +
                            (size_t)head * q_strides[2] + (size_t)dim]),
                    gemm_operand_mask);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stage_q && vector_loads) {
            // Staged-query fast path: load each pooled block element once and
            // dot all HEADS against it. Each head's fp32 accumulation still
            // walks the dims in the same order, so the score is bit-identical;
            // the device pooled traffic drops by a factor of HEADS.
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const size_t pooled_base = (size_t)block * pooled_strides[1];
                const device uint4* kv =
                    (const device uint4*)(pooled + pooled_base);
                float dots[HEADS];
                #pragma clang loop unroll(full)
                for (uint h = 0; h < HEADS; ++h) {
                    dots[h] = 0.0f;
                }
                #pragma clang loop unroll(full)
                for (uint chunk = 0; chunk < HEAD_DIM / 8u; ++chunk) {
                    const uint4 b = kv[chunk];
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; ++i) {
                        const float k_value = qsa_unpack_trunc(b, i, gemm_operand_mask);
                        #pragma clang loop unroll(full)
                        for (uint h = 0; h < HEADS; ++h) {
                            dots[h] += q_shared[h * HEAD_DIM + chunk * 8u + i] *
                                       k_value;
                        }
                    }
                }
                float score_sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint h = 0; h < HEADS; ++h) {
                    score_sum += metal::max(dots[h], 0.0f);
                }
                const float score = score_sum / SQRT_HEAD_DIM;
                score_scratch[scratch_base + block] =
                    score - float(block) * 1.0e-12f;
            }
        } else {
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const size_t pooled_base = (size_t)block * pooled_strides[1];
                float score_sum = 0.0f;
                for (uint head = 0; head < HEADS; ++head) {
                    float dot = 0.0f;
                    const size_t q_base =
                        (size_t)row * q_strides[1] +
                        (size_t)head * q_strides[2];
                    if (vector_loads) {
                        const device uint4* qv =
                            (const device uint4*)(q + q_base);
                        const device uint4* kv =
                            (const device uint4*)(pooled + pooled_base);
                        #pragma clang loop unroll(full)
                        for (uint chunk = 0; chunk < HEAD_DIM / 8u; ++chunk) {
                            const uint4 a = qv[chunk];
                            const uint4 b = kv[chunk];
                            #pragma clang loop unroll(full)
                            for (uint i = 0; i < 8u; ++i) {
                                dot += qsa_unpack_trunc(a, i, gemm_operand_mask) *
                                       qsa_unpack_trunc(b, i, gemm_operand_mask);
                            }
                        }
                    } else {
                        for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                            const float q_value = qsa_mlx_gemm_operand(
                                float(q[q_base + (size_t)dim * q_strides[3]]),
                                gemm_operand_mask);
                            const float k_value = qsa_mlx_gemm_operand(
                                float(pooled[
                                    pooled_base + (size_t)dim * pooled_strides[2]]),
                                gemm_operand_mask);
                            dot += q_value * k_value;
                        }
                    }
                    score_sum += metal::max(dot, 0.0f);
                }
                const float score = score_sum / SQRT_HEAD_DIM;
                const float adjusted = score - float(block) * 1.0e-12f;
                score_scratch[scratch_base + block] = adjusted;
            }
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup | mem_flags::mem_device);

        // Select the Kth largest strict composite key a byte at a time. Each
        // pass scans only candidates matching the already-selected high-byte
        // prefix. The rank is relative to that prefix bucket.
        const uint k_eff = metal::min(TOP_K, valid_count);
        if (lane == 0) {
            radix_prefix = 0ul;
            radix_rank = k_eff > 0 ? k_eff - 1 : 0;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k_eff > 0) {
            for (uint pass = 0; pass < 8; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 56u - pass * 8u;
                const ulong prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const float adjusted = score_scratch[scratch_base + block];
                    const ulong key = qsa_composite_key(adjusted, block);
                    bool prefix_matches = true;
                    if (pass > 0) {
                        prefix_matches = (key >> (shift + 8u)) == prefix;
                    }
                    if (prefix_matches) {
                        const uint digit = uint((key >> shift) & 0xfful);
                        atomic_fetch_add_explicit(
                            &radix_histogram[digit], 1u, memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0) {
                    uint rank = radix_rank;
                    uint chosen = 0;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8) | ulong(chosen);
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compact the exact winners into WIDTH-sized threadgroup storage. The
        // atomic arrival order is irrelevant: the following bitonic network
        // establishes the mode's deterministic output order.
        exchange_scores[lane] = -INFINITY;
        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0;
        if (lane == 0) {
            atomic_store_explicit(
                &selected_count, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_eff > 0) {
            const ulong threshold = threshold_key;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const float adjusted = score_scratch[scratch_base + block];
                if (qsa_composite_key(adjusted, block) >= threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_scores[slot] = adjusted;
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_index = exchange_indices[lane];
        bool my_valid = exchange_valid[lane] != 0;
        float my_score = exchange_scores[lane];
        for (uint sequence = 2; sequence <= WIDTH; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                exchange_scores[lane] = my_score;
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1 : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const float other_score = exchange_scores[partner];
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0;
                const float a_score = is_lower ? my_score : other_score;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const float b_score = is_lower ? other_score : my_score;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;

                const bool lower_wants_before = (lane & sequence) == 0;
                const bool b_before_a = ROW_TOKEN_MODE
                    ? qsa_row_score_before(
                        b_score, b_index, b_valid,
                        a_score, a_index, a_valid)
                    : qsa_index_before(
                        b_index, b_valid, a_index, a_valid);
                const bool a_before_b = ROW_TOKEN_MODE
                    ? qsa_row_score_before(
                        a_score, a_index, a_valid,
                        b_score, b_index, b_valid)
                    : qsa_index_before(
                        a_index, a_valid, b_index, b_valid);
                const bool swap = lower_wants_before ? b_before_a : a_before_b;
                if (swap) {
                    my_score = is_lower ? b_score : a_score;
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }
    
        if (lane < TOP_K) {
            const bool ok = my_valid;
            const size_t out_at = (size_t)row * TOP_K + lane;
            block_ids[out_at] = ok ? int(my_index) : 0;
            block_valid[out_at] = ok;
            adjusted_scores[out_at] = ok ? my_score : -INFINITY;
        }
        