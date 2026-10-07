// One-token MoE: fused routed+shared expert gate|up GEMV + SwiGLU.
        const uint z = threadgroup_position_in_grid.z;
        const bool shared = z == (uint)BR;
        const uint e = shared ? 0u : idx[z];
        const uint r = shared ? 0u : xrow[z];
        const size_t kw = (size_t)KD / 8;
        const size_t kg = (size_t)KD / GS;
        const size_t eoff = (size_t)e * (size_t)N;
        const device uint32_t* gw = shared ? wsh : wg + eoff * kw;
        const device T* gs = shared ? ssh : sg + eoff * kg;
        const device T* gb = shared ? bsh : bg + eoff * kg;
        const device uint32_t* uw = shared ? wsh + (size_t)N * kw : wu + eoff * kw;
        const device T* us = shared ? ssh + (size_t)N * kg : su + eoff * kg;
        const device T* ub = shared ? bsh + (size_t)N * kg : bu + eoff * kg;
        const int out_row = (int)threadgroup_position_in_grid.y * (2 * RPS)
            + (int)simdgroup_index_in_threadgroup * RPS;
        float g[RPS], u[RPS];
        qmv_fast_reg_dual<T, GS, BITS, RPS>(
            gw, gs, gb, uw, us, ub, x + (size_t)r * (size_t)KD,
            KD, out_row, thread_index_in_simdgroup, g, u);
        if (thread_index_in_simdgroup == 0) {
            for (int i = 0; i < RPS; ++i) {
                const T gv = static_cast<T>(g[i]);
                const T uv = static_cast<T>(u[i]);
                act[(size_t)z * (size_t)N + (size_t)(out_row + i)] = mlx_silu(gv) * uv;
            }
        }
    