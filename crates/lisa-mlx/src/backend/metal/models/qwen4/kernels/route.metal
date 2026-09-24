// MoE router: top-k selection + softmax + shared-expert gate (one launch).

        constexpr int E_PER = (E + 31) / 32;
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        if constexpr (HAS_GATE) {
            const device T* xr = x + (size_t)row * (size_t)KD;
            if constexpr (VPT == 1) {
                track_inject_qmv<T, GS, BITS, KD, 1, 4>(wg, sgw, bgw, xr, gate + row, sg, lane);
            } else {
                threadgroup float fp[8];
                float r[1]; bool valid = false; int orow = 0;
                qmv_wide_reg_partial<T, GS, BITS, 1, 8>(wg, sgw, bgw, xr, KD, 1, 1, fp, sg, lane, r, valid, orow);
                if (valid) { gate[row] = static_cast<T>(r[0]); }
            }
        }
        // OPT-ROUTEREG: the walk's winners come from a simd_max/simd_min
        // pair, so logit and index are already broadcast across the walk's own
        // simdgroup. Normalize them there and skip the threadgroup round trip.
        constexpr bool REGISTER_RESULTS = VPT != 1 || HAS_GATE;
        constexpr int N_READS = 4;
        float ld[N_READS];
        uint selected[N_READS];
        for (int i = 0; i < N_READS; ++i) {
            ld[i] = -INFINITY;
            selected[i] = 0xffffffffu;
        }
        threadgroup float selv[K];
        threadgroup uint seli[K];
        constexpr uint SEL_SG = (VPT == 1) ? 1u : 0u;
        if (sg == SEL_SG) {
        const device float* lr = logits + (size_t)row * (size_t)E;
        float v[E_PER];
        bool taken[E_PER];
        for (int j = 0; j < E_PER; ++j) {
            const int e = (int)lane + 32 * j;
            v[j] = (e < E) ? lr[e] : -INFINITY;
            taken[j] = (e >= E);
        }
        for (int k = 0; k < K; ++k) {
            float bv = -INFINITY; int bj = -1;
            for (int j = 0; j < E_PER; ++j) {
                if (!taken[j] && (v[j] > bv)) { bv = v[j]; bj = j; }
            }
            const float gmax = simd_max(bv);
            const uint cand = (bv == gmax && bj >= 0) ? (uint)(lane + 32 * bj) : 0xffffffffu;
            const uint gidx = simd_min(cand);
            if constexpr (REGISTER_RESULTS) {
                for (int i = 0; i < N_READS; ++i) {
                    if (k == (int)lane * N_READS + i) { ld[i] = gmax; selected[i] = gidx; }
                }
            } else if (lane == 0) { selv[k] = gmax; seli[k] = gidx; }
            if (gidx == (uint)(lane + 32 * bj) && bj >= 0) { taken[bj] = true; }
        }
        }
        if constexpr (REGISTER_RESULTS) {
            if (sg != SEL_SG) { return; }
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg != 0) { return; }
            for (int i = 0; i < N_READS; i++) {
                const int p = (int)lane * N_READS + i;
                ld[i] = (p < K) ? selv[p] : -INFINITY;
            }
        }
        // softmax_single_row over the K selected logits (AccT = float)
        float maxval = -FLT_MAX;
        for (int i = 0; i < N_READS; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
        maxval = simd_max(maxval);
        float normalizer = 0;
        for (int i = 0; i < N_READS; i++) {
            float exp_x = fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        normalizer = 1 / normalizer;
        for (int i = 0; i < N_READS; i++) {
            const int p = (int)lane * N_READS + i;
            if (p < K) {
                w[(size_t)row * K + p] = ld[i] * normalizer;
                if constexpr (REGISTER_RESULTS) {
                    idx[(size_t)row * K + p] = selected[i];
                } else { idx[(size_t)row * K + p] = seli[p]; }
            }
        }
    