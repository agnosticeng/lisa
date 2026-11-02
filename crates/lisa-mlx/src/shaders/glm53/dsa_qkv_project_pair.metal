// Pair DSA q_a and kv_a projections. q_a uses MLX qmv_fast's schedule.
// For 2..8 session rows, kv_a follows DS4's F32 small-batch matvec schedule:
// nxpsg and r1ptg select the same lane and row mapping as mul_mv_ext_f32.
// The packed MLX affine weights are expanded with DS4's FMA before the same
// four-value dot and shuffle reduction. One launch computes both projections.
{
    constexpr uint IN = 4096u;
    constexpr uint Q_ROWS = 1536u;
    constexpr uint KV_ROWS = 512u;
    constexpr uint Q_GROUPS = Q_ROWS / 32u;
    constexpr uint GROUP_SIZE = 64u;

    const uint batch_row = threadgroup_position_in_grid.x;
    const uint group = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint rows = (uint)kv_rows_arg;

    if (group < Q_GROUPS) {
        const uint3 virtual_tid = uint3(batch_row, group * 4u + sg / 2u, 0u);
        qmv_fast_impl<float, 64, 8>(
            q_weights,
            q_scales,
            q_biases,
            x,
            q_raw,
            in_dim,
            q_rows,
            virtual_tid,
            sg & 1u,
            lane);
        return;
    }

    if (rows > 1u && rows <= 8u) {
        if (batch_row != 0u) return;
        const uint nxpsg = (uint)nxpsg_arg;
        const uint r1ptg = (uint)r1ptg_arg;
        const uint nypsg = 32u / nxpsg;
        const uint pair = sg >> 1u;
        const uint local_sg = sg & 1u;
        const uint ty = lane / nxpsg;
        const uint tx = lane % nxpsg;
        const uint kv_group = group - Q_GROUPS;
        const uint outputs_per_group = nypsg * 8u;
        const uint out_row = kv_group * outputs_per_group
            + pair * nypsg * 2u + local_sg * nypsg + ty;
        if (out_row >= KV_ROWS) return;

        constexpr uint CHUNKS_PER_THREAD = 4u;
        constexpr uint INPUT_VECTORS = IN / 4u;
        const device uint *packed = kv_weights;
        const device float *scale = kv_scales;
        const device float *bias = kv_biases;
        const device float4 *x4 = (const device float4 *)x;
        const size_t row_words = IN / 4u;
        const size_t row_groups = IN / GROUP_SIZE;
        const uint token_base = threadgroup_position_in_grid.z * r1ptg;
        float sums[5] = {};

        for (uint ich = tx; ich < INPUT_VECTORS;
             ich += CHUNKS_PER_THREAD * nxpsg) {
            for (uint chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk) {
                const uint vector = ich + chunk * nxpsg;
                const uint input0 = vector * 4u;
                float4 weight4;
                for (uint k = 0; k < 4u; ++k) {
                    const uint input_index = input0 + k;
                    const uint word = packed[
                        (size_t)out_row * row_words + input_index / 4u];
                    const uint code =
                        (word >> ((input_index & 3u) * 8u)) & 0xffu;
                    const size_t affine =
                        (size_t)out_row * row_groups + input_index / GROUP_SIZE;
                    weight4[k] = fma((float)code, scale[affine], bias[affine]);
                }
                for (uint ir = 0; ir < r1ptg; ++ir) {
                    const uint token = token_base + ir;
                    if (token < rows) {
                        const device float4 *row_x = x4 + (size_t)token * INPUT_VECTORS;
                        sums[ir] += dot(weight4, row_x[vector]);
                    }
                }
            }
        }

        for (uint ir = 0; ir < r1ptg; ++ir) {
            if (nxpsg >= 16u) sums[ir] += simd_shuffle_down(sums[ir], 8);
            if (nxpsg >= 8u) sums[ir] += simd_shuffle_down(sums[ir], 4);
            if (nxpsg >= 4u) sums[ir] += simd_shuffle_down(sums[ir], 2);
            if (nxpsg >= 2u) sums[ir] += simd_shuffle_down(sums[ir], 1);
        }
        if (tx == 0u) {
            for (uint ir = 0; ir < r1ptg; ++ir) {
                const uint token = token_base + ir;
                if (token < rows) {
                    kv_raw[(size_t)token * KV_ROWS + out_row] = sums[ir];
                }
            }
        }
        return;
    }

    if (rows == 1u) {
        const uint kv_group = group - Q_GROUPS;
        const uint out_row = kv_group * 8u + sg;
        if (batch_row != 0u || out_row >= KV_ROWS) return;
        const device uint *packed = kv_weights;
        const device float *scale = kv_scales;
        const device float *bias = kv_biases;
        const device float *row_x = x;
        const size_t row_words = IN / 4u;
        const size_t row_groups = IN / GROUP_SIZE;
        float sum = 0.0f;
        for (uint input_index = lane; input_index < IN; input_index += 32u) {
            const uint word = packed[(size_t)out_row * row_words + input_index / 4u];
            const uint code = (word >> ((input_index & 3u) * 8u)) & 0xffu;
            const size_t affine = (size_t)out_row * row_groups + input_index / GROUP_SIZE;
            const float weight = fma((float)code, scale[affine], bias[affine]);
            sum = fma(weight, row_x[input_index], sum);
        }
        sum = simd_sum(sum);
        if (lane == 0u) kv_raw[out_row] = sum;
        return;
    }

    if (threadgroup_position_in_grid.z != 0u) return;
    const uint out_row = group - Q_GROUPS;
    if (out_row >= KV_ROWS) return;
    constexpr uint SIMD_WIDTH = 32u;
    constexpr uint SIMDGROUPS = 8u;
    constexpr uint VALUES_PER_LANE = 16u;
    constexpr uint CHUNK = 32u;
    constexpr uint N_BLOCKS = IN / CHUNK;
    const uint ix = lane / (SIMD_WIDTH / VALUES_PER_LANE);
    const uint il = lane % (SIMD_WIDTH / VALUES_PER_LANE);
    const uint block0 = sg * VALUES_PER_LANE + ix;
    const size_t row_words = IN / 4u;
    const size_t row_groups = IN / GROUP_SIZE;
    const device uint *packed = kv_weights;
    const device float *scale = kv_scales;
    const device float *bias = kv_biases;
    const device float4 *x4 = (const device float4 *)x;
    const size_t x_row = (size_t)batch_row * IN;
    float sumf = 0.0f;

    for (uint block = block0; block < N_BLOCKS;
         block += SIMDGROUPS * VALUES_PER_LANE) {
        const uint input0 = block * CHUNK + il * VALUES_PER_LANE;
        float sumq = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE / 4u; ++i) {
            const uint d = input0 + i * 4u;
            float4 weight4;
            for (uint k = 0; k < 4u; ++k) {
                const uint input_index = d + k;
                const uint word = packed[(size_t)out_row * row_words + input_index / 4u];
                const uint code = (word >> ((input_index & 3u) * 8u)) & 0xffu;
                const size_t affine = (size_t)out_row * row_groups + input_index / GROUP_SIZE;
                weight4[k] = fma((float)code, scale[affine], bias[affine]);
            }
            sumq += dot(weight4, x4[(x_row + d) / 4u]);
        }
        sumf += sumq;
    }

    threadgroup float *partial = shared;
    if (sg == 0u) partial[lane] = 0.0f;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) partial[sg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(partial[lane]);
    if (lane == 0u && sg == 0u) kv_raw[(size_t)batch_row * KV_ROWS + out_row] = total;
}
