// DS4's small-batch F32 router matvec schedule for 2–8 input rows.
// The lane mapping, four-vector chunk loop, row grouping, and shuffle tree
// follow kernel_mul_mv_ext_q4_f32_impl instantiated for float4 weights.
{
    constexpr uint NSG = 2u;
    constexpr uint CHPT = 4u;
    constexpr uint NYPSG = 32u / (uint)NXPSG;
    const uint row = threadgroup_position_in_grid.x * (NYPSG * NSG) +
                     NYPSG * simdgroup_index_in_threadgroup +
                     thread_index_in_simdgroup / (uint)NXPSG;
    const uint token_base = threadgroup_position_in_grid.y * (uint)R1PTG;
    const uint tx = thread_index_in_simdgroup % (uint)NXPSG;
    const uint in_width = (uint)in_dim;
    const uint n_vectors = in_width / 4u;
    if (row >= (uint)out_dim) return;

    const device float4 *weight4 =
        (const device float4 *)(weights + (size_t)row * in_width);
    const device float4 *x4[R1PTG];
    float sums[R1PTG] = {};
    for (int ir = 0; ir < R1PTG; ++ir) {
        const uint token = token_base + (uint)ir;
        x4[ir] = token < (uint)n_tokens
            ? (const device float4 *)(x + (size_t)token * in_width) + tx
            : (const device float4 *)x;
    }

    for (uint ich = tx; ich < n_vectors; ich += CHPT * (uint)NXPSG) {
        float4 weight_vectors[CHPT];
        for (uint ch = 0; ch < CHPT; ++ch) {
            weight_vectors[ch] = weight4[ich + ch * (uint)NXPSG];
        }
        for (uint ch = 0; ch < CHPT; ++ch) {
            for (int ir = 0; ir < R1PTG; ++ir) {
                const uint token = token_base + (uint)ir;
                if (token < (uint)n_tokens) {
                    sums[ir] += dot(weight_vectors[ch], x4[ir][ch * (uint)NXPSG]);
                }
            }
        }
        for (int ir = 0; ir < R1PTG; ++ir) {
            x4[ir] += CHPT * (uint)NXPSG;
        }
    }

    for (int ir = 0; ir < R1PTG; ++ir) {
        if (NXPSG >= 32) sums[ir] += simd_shuffle_down(sums[ir], 16);
        if (NXPSG >= 16) sums[ir] += simd_shuffle_down(sums[ir], 8);
        if (NXPSG >= 8) sums[ir] += simd_shuffle_down(sums[ir], 4);
        if (NXPSG >= 4) sums[ir] += simd_shuffle_down(sums[ir], 2);
        if (NXPSG >= 2) sums[ir] += simd_shuffle_down(sums[ir], 1);
    }

    if (tx == 0u) {
        for (int ir = 0; ir < R1PTG; ++ir) {
            const uint token = token_base + (uint)ir;
            if (token < (uint)n_tokens) {
                logits[(size_t)token * (uint)out_dim + row] = sums[ir];
            }
        }
    }
}
