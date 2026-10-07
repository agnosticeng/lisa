// Counting sort, pass 2: stable scatter of the routed assignments.
        threadgroup uint vals[BLK];
        threadgroup uint tot[E];
        threadgroup uint pre[E];
        threadgroup uint sA[E];
        threadgroup uint sB[E];
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint gi = blk * BLK + t;
        vals[t] = (gi < (uint)R) ? ids[gi] : (uint)E;
        for (uint b = t; b < (uint)E; b += BLK) {
            uint s = 0, before = 0;
            for (uint n = 0; n < (uint)NB; ++n) {
                const uint c = counts[n * (uint)E + b];
                if (n < blk) { before += c; }
                s += c;
            }
            tot[b] = s;
            pre[b] = before;
            sA[b] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1; off < (uint)E; off <<= 1) {
            for (uint b = t; b < (uint)E; b += BLK) {
                sB[b] = sA[b] + ((b >= off) ? sA[b - off] : 0u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint b = t; b < (uint)E; b += BLK) { sA[b] = sB[b]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (gi >= (uint)R) { return; }
        const uint v = vals[t];
        // The last block may be padded with the out-of-range sentinel `E`
        // (see `route_block_counts`); such slots carry no assignment.
        if (v >= (uint)E) { return; }
        uint rank = 0;
        for (uint j = 0; j < t; ++j) { rank += (vals[j] == v) ? 1u : 0u; }
        const uint dest = (sA[v] - tot[v]) + pre[v] + rank;
        sorted_ids[dest] = v;
        token_rows[dest] = gi / (uint)TOPK;
        inverse[gi] = dest;
    