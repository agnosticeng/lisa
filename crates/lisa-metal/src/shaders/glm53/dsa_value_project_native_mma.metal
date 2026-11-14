// Native DS4 GLM-5.3 value projection schedule with MLX affine storage.
{
    const uint n_tokens = (uint)n_rows / (uint)n_heads;
    const uint tid = thread_index_in_threadgroup;
    device const uchar *weight = (device const uchar *)kv_weight;
    device char *heads = (device char *)output;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup char *shmem = (threadgroup char *)shared;
    constexpr uint NR0 = 64u;
    constexpr uint NR1 = 32u;
    constexpr uint NK = 32u;
    constexpr uint NL0 = 2u;
    constexpr uint NL1 = 4u;

    const uint token0 = threadgroup_position_in_grid.x * NR1;
    const uint value0 = threadgroup_position_in_grid.y * NR0;
    const uint head = threadgroup_position_in_grid.z + 0u;
    if (head >= n_heads || token0 >= n_tokens || value0 >= value_dim) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096u);

    const uint nr0 = min(NR0, value_dim - value0);
    const uint nr1 = min(NR1, n_tokens - token0);

    const uint lr0 = min((uint)tid / NL0, nr0 - 1u);
    const uint lr1 = min((uint)tid / NL1, nr1 - 1u);
    const uint il0 = (uint)tid & 1u;
    const uint iy = 8u * ((uint)tid & (NL1 - 1u));

    const uint64_t lora_token_stride =
        (uint64_t)n_heads * kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride =
        (uint64_t)n_heads * value_dim * sizeof(float);
    const uint64_t head_lora_base =
        (uint64_t)head * kv_lora_dim * sizeof(float);
    const uint64_t head_out_base =
        (uint64_t)head * value_dim * sizeof(float);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < kv_lora_dim; loop_k += NK) {
        const uint value = value0 + lr0;
        const ulong weight_row = (ulong)head * kv_lora_dim + value_dim + value;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 16u; i++) {
            const uint k = loop_k + 16u * il0 + i;
            const uint sx = 2u * il0 + i / 8u;
            const uint sy = ((uint)tid / NL0) / 8u;
            const uint lx = ((uint)tid / NL0) & 7u;
            const uint ly = i & 7u;
            const uint ib = 8u * sx + sy;
            const ulong affine = weight_row * (kv_lora_dim / group_size) + k / group_size;
            const half v = (value < value_dim && k < kv_lora_dim) ?
                half(fma(float(weight[weight_row * kv_lora_dim + k]),
                    kv_scales[affine],
                    kv_biases[affine])) : half(0.0f);
            *(sa + 64u * ib + 8u * ly + lx) = v;
        }

        const uint token = token0 + lr1;
        device const float *y =
            (device const float *)((device const char *)lora +
                (uint64_t)token * lora_token_stride +
                head_lora_base +
                (uint64_t)loop_k * sizeof(float) +
                (uint64_t)iy * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            const uint k = loop_k + iy + i;
            const uint sx = ((uint)tid) & (NL1 - 1u);
            const uint sy = ((uint)tid / NL1) / 8u;
            const uint lx = i;
            const uint ly = ((uint)tid / NL1) & 7u;
            const uint ib = 4u * sx + sy;
            const half v = (token < n_tokens && k < kv_lora_dim) ?
                half(y[i]) :
                half(0.0f);
            *(sb + 64u * ib + 8u * ly + lx) = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4u * 64u * ((uint)sg & 1u);
        threadgroup const half *lsmb = sb + 2u * 64u * ((uint)sg >> 1);

        for (uint ik = 0; ik < NK / 8u; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 4u; i++) {
                simdgroup_load(ma[i], lsma + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 2u; i++) {
                simdgroup_load(mb[i], lsmb + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 8u; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i & 3u], mc[i]);
            }

            lsma += 8u * 64u;
            lsmb += 4u * 64u;
        }
    }

    if (nr0 == NR0 && nr1 == NR1) {
        device float *dst =
            (device float *)(heads +
                (uint64_t)(token0 + 16u * ((uint)sg >> 1)) * heads_token_stride +
                head_out_base +
                (uint64_t)(value0 + 32u * ((uint)sg & 1u)) * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            dst + 8u * (i & 3u) + 8u * (heads_token_stride / sizeof(float)) * (i / 4u),
                            heads_token_stride / sizeof(float),
                            0,
                            false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float *tmp = (threadgroup float *)shmem;
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            tmp + 32u * ((uint)sg & 1u) +
                                  16u * ((uint)sg >> 1) * NR0 +
                                  8u * (i & 3u) + 8u * NR0 * (i / 4u),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            for (uint t = tid; t < nr1; t += 128u) {
                device float *dst =
                    (device float *)(heads +
                        (uint64_t)(token0 + t) * heads_token_stride +
                        head_out_base +
                        (uint64_t)value0 * sizeof(float));
                threadgroup const float *src = tmp + t * NR0;
                for (uint v = 0; v < nr0; v++) {
                    dst[v] = src[v];
                }
            }
        }
    }
}
