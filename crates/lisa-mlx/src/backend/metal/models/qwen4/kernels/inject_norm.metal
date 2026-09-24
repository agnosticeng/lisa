// Hyper-connection inject + RMSNorm for the 1..8-row window.
        constexpr int N_READS = 4;
        constexpr uint NT = H / N_READS;
        const uint row = thread_position_in_grid.z;
        const uint hc = thread_position_in_grid.y;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float local_sums[32];
        const uint base = row * W + hc * H;
        InT inj_t = InT(0);
        if (HAS_INJECT) { inj_t = inject[row * HC + hc]; }
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const uint src = TILE ? (row * H + d) : (base + d);
            InT r = residual[src];
            if (HAS_INJECT) {
                InT sp = out[row * H + d] * inj_t;
                r = r + sp;
            }
            stream[base + d] = r;
            thread_x[i] = static_cast<float>(r);
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { local_sums[lane] = 0; }
        if (lane == 0) { local_sums[sg] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(local_sums[lane]);
        const float inv_mean = metal::precise::rsqrt(acc / (float)H + as_type<float>((uint)EPS_BITS));
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            InT n = static_cast<InT>(thread_x[i] * inv_mean);
            normed[base + d] = n * scale[hc * H + d];
        }
    