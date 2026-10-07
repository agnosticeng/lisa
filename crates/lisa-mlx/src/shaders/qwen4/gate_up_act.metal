// Wide MoE expert gate|up GEMV + SwiGLU for a 2..8-token window (the S=1 window
// uses gate_up_reuse.metal instead, so this file has no narrow branch).
        const uint z = threadgroup_position_in_grid.z;
        if (z == (uint)BR) {
            const uint kw2 = (uint)KD / 8;
            const uint kg2 = (uint)KD / GS;
            const int tile = (int)threadgroup_position_in_grid.y;
            const int row = tile * 8 + (int)simdgroup_index_in_threadgroup * 4 + (int)(thread_index_in_simdgroup / 8);
            float g[VPT], u[VPT];
            qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wsh, ssh, bsh, x, KD, VPT, row, thread_index_in_simdgroup, g);
            qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wsh + (size_t)N * kw2, ssh + (size_t)N * kg2, bsh + (size_t)N * kg2, x, KD, VPT, row, thread_index_in_simdgroup, u);
            if ((thread_index_in_simdgroup % 8) == 0) {
                for (int v = 0; v < VPT; ++v) {
                    act[(size_t)(BR + v) * (size_t)N + (size_t)row] = mlx_silu(static_cast<T>(g[v])) * static_cast<T>(u[v]);
                }
            }
            return;
        }
        const uint e = idx[z];
        const uint r = xrow[z];
        const uint kw = (uint)KD / 8;
        const uint kg = (uint)KD / GS;
        const int out_row = (int)threadgroup_position_in_grid.y * 8 + (int)simdgroup_index_in_threadgroup * 4;
        const device T* xb = x + (size_t)r * (size_t)KD;
        const size_t eoff = (size_t)e * (size_t)N;
        float g[4], u[4];
        if (FAST) {
            qmv_fast_reg<T, GS, BITS>(wg + eoff * kw, sg + eoff * kg, bg + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, g);
            qmv_fast_reg<T, GS, BITS>(wu + eoff * kw, su + eoff * kg, bu + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, u);
        } else {
            qmv_reg<T, GS, BITS, (KD % get_pack_factor<BITS, 32>()) == 0>(wg + eoff * kw, sg + eoff * kg, bg + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, g);
            qmv_reg<T, GS, BITS, (KD % get_pack_factor<BITS, 32>()) == 0>(wu + eoff * kw, su + eoff * kg, bu + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, u);
        }
        // OPT-ACTLANES: g/u are post-simd_sum, identical on every lane, so
        // each of the four entries can be stored by its own lane.
        if (thread_index_in_simdgroup < 4) {
            const int i = (int)thread_index_in_simdgroup;
            const T gv = static_cast<T>(g[i]);
            const T uv = static_cast<T>(u[i]);
            act[(size_t)z * (size_t)N + (size_t)(out_row + i)] = mlx_silu(gv) * uv;
        }
