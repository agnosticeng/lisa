// One-token mixer down+inject GEMV split across SIMD groups (ORDERED fold).
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int tile = int(threadgroup_position_in_grid.y);
        constexpr int DN = ND / RPS;
        constexpr int DOWN_SCRATCH = (K / 512) * RPS * 32;
        constexpr int INJ_SCRATCH = (K / 256) * 32;
        threadgroup float scratch[DOWN_SCRATCH > INJ_SCRATCH ? DOWN_SCRATCH : INJ_SCRATCH];
        if (tile < DN) {
            float r[RPS];
            research_split_qmv<T, K, 16, RPS, SPLIT>(
                wd, sd, bd, x, tile * RPS, sg, lane, scratch, r);
            if (sg == 0 && lane == 0) {
                for (int i = 0; i < RPS; ++i) {
                    T l = static_cast<T>(r[i]);
                    lo[tile * RPS + i] = l;
                    act[tile * RPS + i] = mlx_silu(l);
                }
            }
        } else if (HAS_INJECT) {
            float r[1];
            research_split_qmv<T, K, 8, 1, SPLIT>(
                wi, si, bi, x, tile - DN, sg, lane, scratch, r);
            if (sg == 0 && lane == 0) { inj[tile - DN] = static_cast<T>(r[0]); }
        }
