// PLE key*query product reduction over the [hc, hidden] reshape.
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hc = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float ksums[32];
        threadgroup float qsums[32];
        const uint base = row * W + hc * H;
        float kx[N_READS];
        float qx[N_READS];
        float kacc = 0.0f;
        float qacc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            kx[i] = static_cast<float>(keyFlat[base + d]);
            qx[i] = static_cast<float>(stream[base + d]);
            kacc += kx[i] * kx[i];
            qacc += qx[i] * qx[i];
        }
        kacc = simd_sum(kacc);
        qacc = simd_sum(qacc);
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { ksums[lane] = 0; qsums[lane] = 0; }
        if (lane == 0) { ksums[sg] = kacc; qsums[sg] = qacc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kacc = simd_sum(ksums[lane]);
        qacc = simd_sum(qsums[lane]);
        const float kinv = metal::precise::rsqrt(kacc / (float)H + eps[0]);
        const float qinv = metal::precise::rsqrt(qacc / (float)H + eps[0]);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT kn = static_cast<InT>(kx[i] * kinv) * kscale[hc * H + d];
            const InT qn = static_cast<InT>(qx[i] * qinv) * qscale[hc * H + d];
            prod[base + d] = kn * qn;
        }
    