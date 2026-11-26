// PLE gated add: `sqrt(|gate|)*sign(gate)` applied to the n-gram embedding.
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hc = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float sums[32];
        const uint base = row * W + hc * H;

        InT g = g0[row * HC + hc] / divisor[0];
        g = mlx_sqrt_t(mlx_maximum(mlx_abs_t(g), floorv[0])) * mlx_sign(g);
        const InT sgm = mlx_sigmoid(g);

        float gx[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT v = sgm * value[row * H + d];
            gated[base + d] = v;
            gx[i] = static_cast<float>(v);
            acc += gx[i] * gx[i];
        }
        acc = simd_sum(acc);
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { sums[lane] = 0; }
        if (lane == 0) { sums[sg] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(sums[lane]);
        const float inv = metal::precise::rsqrt(acc / (float)H + eps[0]);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            normed[base + d] = static_cast<InT>(gx[i] * inv) * cscale[hc * H + d];
        }
    