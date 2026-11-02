// GLM-5.3 KDA output stage, matching DS4's 128-thread head/token group.
        const uint head = thread_position_in_grid.x / 128u;
        const uint token = thread_position_in_grid.y;
        if (head >= (uint)n_heads || token >= (uint)n_rows) return;
        constexpr uint D = 128u;
        const uint proj = (uint)n_heads * D;
        const uint tid = thread_index_in_threadgroup;
        const ushort lane = thread_index_in_simdgroup;
        const ushort sg = simdgroup_index_in_threadgroup;
        threadgroup float partial[4];
        const ulong row_base = (ulong)token * proj;
        const ulong idx = row_base + head * D + tid;
        const float raw = in[idx];
        const float sumsq = simd_sum(raw * raw);
        if (lane == 0u) partial[sg] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = lane < 4u ? partial[lane] : 0.0f;
        total = simd_sum(total);
        const float scale = rsqrt(total / (float)D + norm_eps);
        out[idx] = raw * scale * output_norm[tid] /
            (1.0f + exp(-output_gate[idx]));
