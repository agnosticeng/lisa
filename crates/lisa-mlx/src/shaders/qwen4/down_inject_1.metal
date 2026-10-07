// Hyper-connection mixer down projection + inject GEMV, one-token window.
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        constexpr int RPS = 1;
        constexpr int NT = ND / (2 * RPS);
        if (tile < NT) {
            float r[RPS];
            qmv_fast_reg<T, GS, BITS, RPS>(wd, sd, bd, normed, KD, tile * (2 * RPS) + (int)sg * RPS, lid, r);
            if (lid == 0) {
                for (int i = 0; i < RPS; ++i) {
                    const T l = static_cast<T>(r[i]);
                    lo[tile * (2 * RPS) + (int)sg * RPS + i] = l;
                    act[tile * (2 * RPS) + (int)sg * RPS + i] = mlx_silu(l);
                }
            }
        } else if (HAS_INJECT) {
            static_assert(HC == 4, "one-row inject tiles require four HC rows");
            const int row = (tile - NT) * 2 + (int)sg;
            switch (row) {
                case 0: track_inject_qmv_row<T, GS, BITS, KD, 0>(wi, si, bi, normed, inj, lid); break;
                case 1: track_inject_qmv_row<T, GS, BITS, KD, 1>(wi, si, bi, normed, inj, lid); break;
                case 2: track_inject_qmv_row<T, GS, BITS, KD, 2>(wi, si, bi, normed, inj, lid); break;
                case 3: track_inject_qmv_row<T, GS, BITS, KD, 3>(wi, si, bi, normed, inj, lid); break;
            }
        }
