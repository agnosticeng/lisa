// Hyper-connection mixer down projection + inject GEMV, 2..8-token window.
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        constexpr int RPS = 4;
        constexpr int NT = ND / (2 * RPS);
        if (tile < NT) {
            float r[VPT];
            const int row = tile * 8 + (int)sg * 4 + (int)(lid / 8);
            qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wd, sd, bd, normed, KD, VPT, row, lid, r);
            if ((lid % 8) == 0) {
                for (int v = 0; v < VPT; ++v) {
                    const T l = static_cast<T>(r[v]);
                    lo[v * ND + row] = l;
                    act[v * ND + row] = mlx_silu(l);
                }
            }
        } else if (HAS_INJECT) {
            threadgroup float fp[8 * VPT];
            float r[VPT];
            bool valid = false; int row = 0;
            qmv_wide_reg_partial<T, GS, BITS, VPT, 8>(wi, si, bi, normed, KD, HC, VPT, fp, sg, lid, r, valid, row);
            if (valid) {
                for (int v = 0; v < VPT; ++v) { inj[v * HC + row] = static_cast<T>(r[v]); }
            }
        }
