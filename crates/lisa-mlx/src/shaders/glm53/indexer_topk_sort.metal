// GLM-5.3 DSA indexer blockwise top-k, ported from ds4/metal/argsort.metal.
// One threadgroup sorts one fixed-width score block and writes only its best
// entries. Merge passes reduce these block candidates to the requested top-k.
{
    const uint lane = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint blocks_per_row = (uint)(n_comp + NTH - 1) / (uint)NTH;
    const uint row = group / blocks_per_row;
    const uint block = group % blocks_per_row;
    if (row >= (uint)n_rows) return;

    const uint base = block * (uint)NTH;
    const uint block_start = block * (uint)block_top_k;
    const uint width = (uint)n_comp;
    threadgroup int32_t indices[NTH];
    threadgroup float values[NTH];
    const uint index = base + lane;
    indices[lane] = (int32_t)index;
    if (index < width) values[lane] = scores[(size_t)row * width + index];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int32_t reg_index = indices[lane];
    float reg_value = reg_index < (int32_t)width ? values[(uint)reg_index - base] : 0.0f;
    for (uint k = 2u; k <= (uint)NTH; k <<= 1u) {
        for (uint stride = k >> 1u; stride > 0u; stride >>= 1u) {
            if (SHUFFLE != 0 && stride < 32u) {
                if (k > 32u && stride == 16u) {
                    reg_index = indices[lane];
                    reg_value = reg_index < (int32_t)width
                        ? values[(uint)reg_index - base]
                        : 0.0f;
                }
                const int32_t other_index = simd_shuffle_xor(reg_index, (ushort)stride);
                const float other_value = simd_shuffle_xor(reg_value, (ushort)stride);
                const bool first = ((lane & k) == 0u) == ((lane & stride) == 0u);
                const bool exchange = first
                    ? (reg_index >= (int32_t)width ||
                       (other_index < (int32_t)width && reg_value < other_value))
                    : (other_index >= (int32_t)width ||
                       (reg_index < (int32_t)width && reg_value > other_value));
                if (exchange) {
                    reg_index = other_index;
                    reg_value = other_value;
                }
                continue;
            }
            const uint other = lane ^ stride;
            if (other > lane) {
                const int32_t a = indices[lane];
                const int32_t b = indices[other];
                const bool a_invalid = (uint)a >= width;
                const bool b_invalid = (uint)b >= width;
                const float av = a_invalid ? 0.0f : values[(uint)a - base];
                const float bv = b_invalid ? 0.0f : values[(uint)b - base];
                const bool descending = (lane & k) == 0u;
                const bool swap = descending
                    ? (a_invalid || (!b_invalid && av < bv))
                    : (b_invalid || (!a_invalid && av > bv));
                if (swap) {
                    indices[lane] = b;
                    indices[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (SHUFFLE != 0 && k >= 32u) {
            indices[lane] = reg_index;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < (uint)block_top_k && block_start + lane < (uint)work_width) {
        partial[(size_t)row * (uint)work_width + block_start + lane] =
            SHUFFLE != 0 ? reg_index : indices[lane];
    }
}
