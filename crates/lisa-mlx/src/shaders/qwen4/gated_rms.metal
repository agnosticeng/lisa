// GDN output gating: gate(z) * RMSNorm(out) in one launch. The gate is
// templated: SIGLU=false -> `sigmoid(z)` (Flash-Next), SIGLU=true ->
// `silu(z) = z * sigmoid(z)` (qwen3_5 swish output gate). The silu arm
// reproduces the composed `nn::silu(z)` rounding points exactly: the sigmoid
// rounds to bf16, the z*sigmoid product rounds to bf16, and the final
// gate*normed product rounds to bf16 (see the exact_header parity notes).
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hv = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        threadgroup float local_sums[32];
        const uint ybase = (row * Hv + hv) * Dv;
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            thread_x[i] = static_cast<float>(y[ybase + lid * N_READS + i]);
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        if (lane >= 1) { local_sums[lane] = 0; }
        if (lane == 0) { local_sums[0] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(local_sums[lane]);
        const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + as_type<float>((uint)EPS_BITS));
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            InT n = w[d] * static_cast<InT>(thread_x[i] * inv_mean);
            const uint zoff = row * (int)zproj_strides[0]
                            + (hv * Dv + d) * (int)zproj_strides[1];
            const float z = static_cast<float>(zproj[zoff]);
            if (SIGLU) {
                // The composed `nn::silu(z)` = bf16 Sigmoid (per-op bf16
                // rounding, the exact chain pinned by
                // sigmoid_mul_matches_composed) then bf16 z*sig; the final
                // gate*normed multiply rounds bf16 once more.
                InT zb = zproj[zoff];
                InT ax = static_cast<InT>(metal::abs(static_cast<float>(zb)));
                InT e = static_cast<InT>(metal::precise::exp(static_cast<float>(ax)));
                InT t = static_cast<InT>(1.0f + static_cast<float>(e));
                InT r = static_cast<InT>(1.0f / static_cast<float>(t));
                InT sg = (zb < InT(0)) ? r : static_cast<InT>(1.0f - static_cast<float>(r));
                InT sil = static_cast<InT>(static_cast<float>(zb) * static_cast<float>(sg));
                out[ybase + d] = static_cast<InT>(static_cast<float>(sil) * static_cast<float>(n));
            } else {
                const float g = mlx_sigmoid(z);
                out[ybase + d] = static_cast<InT>(g * static_cast<float>(n));
            }
        }
    
