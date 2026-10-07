// Header for the PLE S=1 fusion kernels.
        METAL_FUNC float ple_row_sum(
            float acc, threadgroup float* partials, uint lane, uint sg) {
            acc = simd_sum(acc);
            if (sg == 0 && lane >= 20) { partials[lane] = 0.0f; }
            if (lane == 0) { partials[sg] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc = simd_sum(partials[lane]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return acc;
        }
    