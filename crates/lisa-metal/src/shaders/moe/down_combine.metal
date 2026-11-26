// MoE expert down projection + expert-weighted combine + shared-gate add (one
// launch; the S=1 narrow and 2..8 wide windows share this templated source).

        const uint t = threadgroup_position_in_grid.z;
        static_assert(VPT == 1 || RPS == 4, "wide windows keep four rows");
        const int d0 = (int)threadgroup_position_in_grid.y * RPS;
        const uint kw = (uint)F / 8;
        const uint kg = (uint)F / GS;
        const uint sgi = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup float prod[K][RPS];
        threadgroup float shvT[RPS];
        float res[RPS];
        if (sgi < (uint)KSG) {
            for (int kk = 0; kk < K / KSG; ++kk) {
                const int k = (int)sgi + kk * KSG;
                const uint z = t * K + k;
                const uint e = idx[z];
                const size_t eoff = (size_t)e * (size_t)H;
                const device T* xb = act + (size_t)z * (size_t)F;
                if (FAST) { qmv_fast_reg<T, GS, BITS, RPS>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, lid, res); }
                else { qmv_reg<T, GS, BITS, (F % get_pack_factor<BITS, 32>()) == 0, RPS>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, lid, res); }
                const float wk = w[z];
                // OPT-STAGELANES: `qmv_reg`/`qmv_fast_reg` close with a
                // `simd_sum` on every row, so every lane of the simdgroup already
                // holds the identical `res[RPS]`. The staging write therefore only
                // needs *a* lane per entry, not lane 0 for all of them; each k slot
                // is written by the simdgroup that owns it (`k = sgi + kk * KSG`).
                if constexpr (VPT == 1 && RPS <= 32) {
                    if (lid < RPS) {
                        prod[k][lid] = static_cast<float>(static_cast<T>(res[lid])) * wk;
                    }
                } else {
                    if (lid == 0) {
                        for (int i = 0; i < RPS; ++i) { prod[k][i] = static_cast<float>(static_cast<T>(res[i])) * wk; }
                    }
                }
            }
        }
        // Shared expert down rows d0..d0+RPS-1 for token t. OPT-SHAREDROWSG:
        // the one-token path gives each shared row its own simdgroup, so the
        // barrier waits on ONE row walk instead of RPS serial ones. Each row
        // keeps its own qmv walk, accumulation order and reduction. Two to
        // eight tokens keep the wide tile (a row's walk does not depend on how
        // many vectors share its tile).
        constexpr uint shared_sg = VPT == 1 && K == 10 && KSG >= 5
            ? (uint)KSG : ((uint)KSG > (uint)K ? (uint)K : 0u);
        const device T* xs = act + (size_t)(BR + t) * (size_t)F;
        if constexpr (VPT == 1 && K == 10 && KSG >= 5) {
            if (sgi >= shared_sg && sgi < shared_sg + (uint)RPS) {
                const int i = (int)(sgi - shared_sg);
                float rs[1];
                qmv_reg<T, GS, BITS, (F % get_pack_factor<BITS, 32>()) == 0, 1>(
                    wsd, ssd, bsd, xs, F, d0 + i, lid, rs);
                if (lid == 0) { shvT[i] = static_cast<float>(static_cast<T>(rs[0])); }
            }
        } else if (sgi == shared_sg) {
            if constexpr (VPT == 1) {
                float rs[RPS];
                qmv_reg<T, GS, BITS, (F % get_pack_factor<BITS, 32>()) == 0, RPS>(wsd, ssd, bsd, xs, F, d0, lid, rs);
                // OPT-STAGELANES: same argument as the routed staging above —
                // `rs` is post-`simd_sum`, so the RPS entries are identical on
                // every lane and each can be stored by its own lane.
                if constexpr (RPS <= 32) {
                    if (lid < RPS) { shvT[lid] = static_cast<float>(static_cast<T>(rs[lid])); }
                } else {
                    if (lid == 0) { for (int i = 0; i < RPS; ++i) { shvT[i] = static_cast<float>(static_cast<T>(rs[i])); } }
                }
            } else {
                float rw[1];
                qmv_wide_reg_full<T, GS, BITS, 1, 8, false>(wsd, ssd, bsd, xs, F, 1, d0 + (int)(lid / 8), lid, rw);
                float sh4[4];
                for (int i = 0; i < 4; ++i) { sh4[i] = static_cast<float>(static_cast<T>(simd_shuffle(rw[0], (ushort)(i * 8)))); }
                if (lid == 0) { for (int i = 0; i < 4; ++i) { shvT[i] = sh4[i]; } }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // OPT-EPILANES: the RPS output columns of one tile fold
        // independently, one lane each, instead of all of them in a serial loop
        // on lane 0. `mlx_colsum_small_f32` is thread-local (no collectives, no
        // threadgroup memory), so each column keeps its own K iteration order
        // and its own fold; only which lane performs it changes.
        if constexpr (VPT == 1 && RPS <= 32) {
            if (sgi == 0 && lid < RPS) {
                const int i = (int)lid;
                const T sg = mlx_sigmoid(gate[t]);
                float col[K];
                for (int k = 0; k < K; ++k) { col[k] = prod[k][i]; }
                const T r = static_cast<T>(mlx_colsum_small_f32<K>(col));
                const T sh = sg * static_cast<T>(shvT[i]);
                out[(size_t)t * (size_t)H + (size_t)(d0 + i)] = r + sh;
            }
        } else {
            if (sgi == 0 && lid == 0) {
                const T sg = mlx_sigmoid(gate[t]);
                for (int i = 0; i < RPS; ++i) {
                    float col[K];
                    for (int k = 0; k < K; ++k) { col[k] = prod[k][i]; }
                    const T r = static_cast<T>(mlx_colsum_small_f32<K>(col));
                    const T sh = sg * static_cast<T>(shvT[i]);
                    out[(size_t)t * (size_t)H + (size_t)(d0 + i)] = r + sh;
                }
            }
        }
    