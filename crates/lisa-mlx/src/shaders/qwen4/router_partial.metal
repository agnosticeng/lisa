// Wide-window router GEMM partials.
        constexpr int tiles_m = (M + 63) / 64;
        constexpr int swizzle_log = tiles_m <= 3 ? 0 : 1;
        constexpr int tn_swizzled = 8 << swizzle_log;
        constexpr int tm_swizzled = (tiles_m + (1 << swizzle_log) - 1) >> swizzle_log;
        constexpr int tiles_per_partition = tn_swizzled * tm_swizzled;
        const int linear_tid = threadgroup_position_in_grid.x;
        const int partition = linear_tid / tiles_per_partition;
        const int xy_flat = linear_tid % tiles_per_partition;
        const int grid_x = xy_flat % tn_swizzled;
        const int grid_y = xy_flat / tn_swizzled;
        const int tid_y = (grid_y << swizzle_log) + (grid_x & ((1 << swizzle_log) - 1));
        const int tid_x = grid_x >> swizzle_log;
        if (tid_y >= tiles_m) { return; }
        const int c_row = tid_y * 64;
        const int c_col = tid_x * 64;
        const int k_start = partition * 2048;
        const int partition_k = min(2048, 2560 - k_start);
        const short tm = 32 * (simdgroup_index_in_threadgroup / 2);
        const short tn = 32 * (simdgroup_index_in_threadgroup % 2);
        const short sm = (M % 64 == 0) ? 32 : short(min(32, M - c_row - tm));
        const device bfloat16_t* A = x + size_t(c_row + tm) * 2560 + k_start;
        const device bfloat16_t* B = w + size_t(c_col + tn) * 2560 + k_start;
        device float* C = partials + size_t(partition) * M * 512 + size_t(c_row + tm) * 512 + c_col + tn;
        NAXTile<float, 2, 2> Dtile;
        dispatch_bool(M % 64 == 0 || sm == 32, [&](auto aligned_m) {
            Dtile = track_router_loop<bfloat16_t, 32, 32, 32, 256, false, true,
                aligned_m.value, true, true, float>(
                A, B, 2560, 2560, partition_k, partition_k / 256, sm, 32);
        });
        dispatch_bool(M % 64 == 0 || sm == 32, [&](auto aligned_m) {
            if constexpr (aligned_m) { Dtile.store(C, 512); }
            else { Dtile.store_safe(C, 512, short2(32, sm)); }
        });
        