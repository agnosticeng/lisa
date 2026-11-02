// DS4's one-row Q8_0 shared gate/up reduction schedule over MLX affine-Q8,
// group-64 weights. Each threadgroup computes two adjacent output rows.
{
    constexpr uint NSG = 4u;
    constexpr uint NQ = 8u;
    constexpr uint NR0 = 2u;
    constexpr uint SIMD_WIDTH = 32u;

    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint row_base = threadgroup_position_in_grid.x * NR0;
    const uint ix = lane / (SIMD_WIDTH / NQ);
    const uint il = lane % (SIMD_WIDTH / NQ);
    const uint block0 = simdgroup * NQ + ix;
    const uint n_blocks = (uint)in_dim / 32u;
    const uint bytes_per_row = (uint)in_dim;
    const uint groups_per_row = (uint)in_dim / (uint)group_size;
    device const uchar *gate_up_q = (device const uchar *)gate_up_weights;
    float sum_gate[NR0] = {0.0f, 0.0f};
    float sum_up[NR0] = {0.0f, 0.0f};

    for (uint block = block0; block < n_blocks; block += NSG * NQ) {
        const uint input0 = block * 32u + il * NQ;
        float sum_x = 0.0f;
        float dot_gate[NR0] = {0.0f, 0.0f};
        float dot_up[NR0] = {0.0f, 0.0f};
        for (uint i = 0u; i < NQ; ++i) {
            const float value = x[input0 + i];
            sum_x += value;
            for (uint row = 0u; row < NR0; ++row) {
                const uint input_index = input0 + i;
                const uint output_row = row_base + row;
                const uint gate_index = output_row * bytes_per_row + input_index;
                const uint up_index = ((uint)out_dim + output_row) * bytes_per_row + input_index;
                dot_gate[row] += (float)gate_up_q[gate_index] * value;
                dot_up[row] += (float)gate_up_q[up_index] * value;
            }
        }
        const uint group = (block * 32u) / (uint)group_size;
        for (uint row = 0u; row < NR0; ++row) {
            const uint gate_affine = (row_base + row) * groups_per_row + group;
            const uint up_affine = ((uint)out_dim + row_base + row) * groups_per_row + group;
            sum_gate[row] += dot_gate[row] * gate_up_scales[gate_affine] +
                sum_x * gate_up_biases[gate_affine];
            sum_up[row] += dot_up[row] * gate_up_scales[up_affine] +
                sum_x * gate_up_biases[up_affine];
        }
    }

    threadgroup float *scratch = (threadgroup float *)shared;
    for (uint row = 0u; row < NR0; ++row) {
        if (simdgroup == 0u) {
            scratch[row * SIMD_WIDTH + lane] = 0.0f;
            scratch[(NR0 + row) * SIMD_WIDTH + lane] = 0.0f;
        }
        sum_gate[row] = simd_sum(sum_gate[row]);
        sum_up[row] = simd_sum(sum_up[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0u; row < NR0; ++row) {
        if (lane == 0u) {
            scratch[row * SIMD_WIDTH + simdgroup] = sum_gate[row];
            scratch[(NR0 + row) * SIMD_WIDTH + simdgroup] = sum_up[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0u; row < NR0; ++row) {
        const float gate_value = simd_sum(scratch[row * SIMD_WIDTH + lane]);
        const float up_value = simd_sum(scratch[(NR0 + row) * SIMD_WIDTH + lane]);
        if (lane == 0u && simdgroup == 0u && row_base + row < (uint)out_dim) {
            const uint output_row = row_base + row;
            gate[output_row] = gate_value;
            up[output_row] = up_value;
            float g = gate_value;
            float u = up_value;
            if (limit > 1.0e-6f) {
                g = min(g, limit);
                u = clamp(u, -limit, limit);
            }
            const float silu = g / (1.0f + exp(-g));
            mid[output_row] = silu * u;
        }
    }
}
