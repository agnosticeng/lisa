// GDN output gating: sigmoid(z) * RMSNorm(out) in one launch.
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
            const float z = static_cast<float>(zproj[row * (Hv * Dv) + hv * Dv + d]);
            const float g = mlx_sigmoid(z);
            out[ybase + d] = static_cast<InT>(g * static_cast<float>(n));
        }
    