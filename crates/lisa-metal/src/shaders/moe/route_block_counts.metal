// Counting sort, pass 1: per-block expert bucket counts. The 512-bucket /
// 256-block geometry intersects SIMD ballot bit planes and popcounts them.
        threadgroup uint planes[8][10];
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint value = (blk * BLK + t < (uint)R) ? ids[blk * BLK + t] : (uint)E;
        const uint valid = (uint)((simd_vote::vote_t)simd_ballot(value < (uint)E));
        if (lane == 0) { planes[sg][9] = valid; }
        for (uint bit = 0; bit < 9; ++bit) {
            const uint mask = (uint)((simd_vote::vote_t)simd_ballot((value & (1u << bit)) != 0));
            if (lane == 0) { planes[sg][bit] = mask; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = t; b < (uint)E; b += BLK) {
            uint count = 0;
            for (uint word = 0; word < 8; ++word) {
                uint matches = planes[word][9];
                #pragma clang loop unroll(full)
                for (uint bit = 0; bit < 9; ++bit) {
                    const uint mask = planes[word][bit];
                    matches &= (b & (1u << bit)) ? mask : ~mask;
                }
                count += popcount(matches);
            }
            counts[blk * (uint)E + b] = count;
        }
