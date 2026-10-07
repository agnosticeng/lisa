// Attention prep: per-head Q/K RMSNorm + partial RoPE in one launch.
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint h = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint b = row / S;
        const uint s = row % S;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float local_sums[32];
        threadgroup InT vec[D];

        const bool isQ = h < HQ;
        const bool isV = h >= HQ + HK;
        uint hh, src;
        if (isQ) { hh = h; src = row * QW + h * D; }
        else if (!isV) { hh = h - HQ; src = row * HK * D + hh * D; }
        else { hh = h - HQ - HK; src = row * HK * D + hh * D; }
        if (isV) {
            for (int i = 0; i < N_READS; ++i) {
                const uint d = lid * N_READS + i;
                vout[((b * HK + hh) * S + s) * D + d] = vproj[src + d];
            }
            return;
        }
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            thread_x[i] = static_cast<float>((isQ ? qkv : kproj)[src + lid * N_READS + i]);
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        constexpr uint simd_groups = (D + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { local_sums[lane] = 0; }
        if (lane == 0) { local_sums[sg] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(local_sums[lane]);
        const float inv_mean = metal::precise::rsqrt(acc / (float)D + as_type<float>((uint)EPS_BITS));
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT wgt = isQ ? qnorm[d] : knorm[d];
            vec[d] = wgt * static_cast<InT>(thread_x[i] * inv_mean);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr int hrot = ROT / 2;
        device InT* dst = isQ ? (qout + ((b * HQ + hh) * S + s) * D) : (kout + ((b * HK + hh) * S + s) * D);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            InT o = vec[d];
            if (d < ROT) {
                const InT c = cosb[s * ROT + d];
                const InT sn = sinb[s * ROT + d];
                if (d < hrot) {
                    InT x1 = vec[d];
                    InT x2 = vec[d + hrot];
                    InT t1 = x1 * c;
                    InT t2 = (-x2) * sn;
                    o = t1 + t2;
                } else {
                    InT x2 = vec[d];
                    InT x1 = vec[d - hrot];
                    InT t1 = x2 * c;
                    InT t2 = x1 * sn;
                    o = t1 + t2;
                }
            }
            dst[d] = o;
        }