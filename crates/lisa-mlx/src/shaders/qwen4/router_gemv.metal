// One-token router GEMV over the bf16 router weight.
        constexpr int TM = RPS, TN = 4, SN = 32, blockM = 4 * RPS, blockN = 128;
        const int tid_x = (int)threadgroup_position_in_grid.x;
        const int simd_gid = (int)simdgroup_index_in_threadgroup;
        const int simd_lid = (int)thread_index_in_simdgroup;
        float result[TM] = {0};
        float inter[TN];
        float v_coeff[TN];
        const int thrN = simd_lid;
        const int simdM = simd_gid;
        int bm = simdM * TM;
        int bn = thrN * TN;
        int out_row = tid_x * blockM + bm;
        if (out_row >= N) return;
        out_row = out_row + TM <= N ? out_row : N - TM;
        const device T* mat = w + (size_t)out_row * (size_t)K;
        const int n_iter = K / blockN;
        // OPT-ROUTERVEC4: both operand tiles are TN == 4 contiguous elements
        // at a TN-aligned offset (`bn` starts at `simd_lid * 4` and advances by
        // blockN == 128; every row base is a multiple of K), so the four scalar
        // loads per tile are one aligned vector load. The products, their order,
        // the accumulator and the simd reduction are untouched.
        const device float4* xv = reinterpret_cast<const device float4*>(x);
        for (int i = 0; i < n_iter; ++i) {
            const float4 vx = xv[bn / TN];
            for (int tn = 0; tn < TN; tn++) { v_coeff[tn] = vx[tn]; }
            int mat_offset = 0;
            for (int tm = 0; tm < TM; tm++) {
                const vec<T, 4> vw = *reinterpret_cast<const device vec<T, 4>*>(mat + mat_offset + bn);
                for (int tn = 0; tn < TN; tn++) { inter[tn] = static_cast<float>(vw[tn]); }
                for (int tn = 0; tn < TN; tn++) { result[tm] += inter[tn] * v_coeff[tn]; }
                mat_offset += K;
            }
            bn += blockN;
        }
        for (int tm = 0; tm < TM; tm++) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        if (simd_lid == 0) {
            for (int tm = 0; tm < TM; tm++) { out[out_row + tm] = result[tm]; }
        }
    