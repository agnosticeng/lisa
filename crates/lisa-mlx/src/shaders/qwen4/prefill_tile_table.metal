// Builds the per-expert tile table for the NAX indirect prefill GEMMs.
        static_assert(E <= TG, "one run per thread in pass 2");
        threadgroup uint run_begin[E + 1];
        threadgroup uint sg_runs[TG / 32];
        threadgroup uint sg_tiles[TG / 32];
        const uint t = thread_position_in_threadgroup.x;
        const uint sg = t / 32;
        constexpr uint PER = ((uint)R + (uint)TG - 1) / (uint)TG;
        const uint i0 = t * PER;
        const uint i1 = min(i0 + PER, (uint)R);
        uint starts = 0;
        for (uint i = i0; i < i1; ++i) {
            starts += (i == 0 || sorted_ids[i] != sorted_ids[i - 1]) ? 1u : 0u;
        }
        const uint ex = simd_prefix_exclusive_sum(starts);
        if ((t % 32) == 31) { sg_runs[sg] = ex + starts; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = ex;
        uint nruns = 0;
        for (uint g = 0; g < (uint)(TG / 32); ++g) {
            if (g < sg) { r += sg_runs[g]; }
            nruns += sg_runs[g];
        }
        for (uint i = i0; i < i1; ++i) {
            if (i == 0 || sorted_ids[i] != sorted_ids[i - 1]) { run_begin[r++] = i; }
        }
        if (t == 0) { run_begin[nruns] = (uint)R; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint len = 0;
        uint ntile = 0;
        if (t < nruns) {
            len = run_begin[t + 1] - run_begin[t];
            ntile = (len + (uint)BM - 1) / (uint)BM;
        }
        const uint ex2 = simd_prefix_exclusive_sum(ntile);
        if ((t % 32) == 31) { sg_tiles[sg] = ex2 + ntile; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint slot = ex2;
        uint total = 0;
        for (uint g = 0; g < (uint)(TG / 32); ++g) {
            if (g < sg) { slot += sg_tiles[g]; }
            total += sg_tiles[g];
        }
        if (t < nruns) {
            const uint b = run_begin[t];
            for (uint j = 0; j < ntile; ++j) {
                tiles[2 * (slot + j)] = b + j * (uint)BM;
                tiles[2 * (slot + j) + 1] = b + min(len, (j + 1) * (uint)BM);
            }
        }
        for (uint i = total + t; i < (uint)MAXT; i += (uint)TG) {
            tiles[2 * i] = 0u;
            tiles[2 * i + 1] = 0u;
        }
    