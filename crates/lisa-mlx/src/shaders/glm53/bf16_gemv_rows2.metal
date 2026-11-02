// Two-row version of DS4's BF16/F32 GEMV. Each weight serves both rows;
// each accumulator retains the original lane-strided FMA and SIMD sum order.
{
    const uint out_row = threadgroup_position_in_grid.x * simdgroups_per_threadgroup +
                         simdgroup_index_in_threadgroup;
    if (out_row >= (uint)out_dim) return;
    // Preserve the binding's address space for small constant tensors too.
    auto w = weights + (size_t)out_row * (size_t)in_dim;
    auto x0 = x;
    auto x1 = x + (size_t)in_dim;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint k = thread_index_in_simdgroup;
    for (; k + 224u < (uint)in_dim; k += 256u) {
        const float w0 = float(w[k]);
        const float w1 = float(w[k + 32u]);
        const float w2 = float(w[k + 64u]);
        const float w3 = float(w[k + 96u]);
        const float w4 = float(w[k + 128u]);
        const float w5 = float(w[k + 160u]);
        const float w6 = float(w[k + 192u]);
        const float w7 = float(w[k + 224u]);
        sum0 = fma(w0, x0[k], sum0);
        sum0 = fma(w1, x0[k + 32u], sum0);
        sum0 = fma(w2, x0[k + 64u], sum0);
        sum0 = fma(w3, x0[k + 96u], sum0);
        sum0 = fma(w4, x0[k + 128u], sum0);
        sum0 = fma(w5, x0[k + 160u], sum0);
        sum0 = fma(w6, x0[k + 192u], sum0);
        sum0 = fma(w7, x0[k + 224u], sum0);
        sum1 = fma(w0, x1[k], sum1);
        sum1 = fma(w1, x1[k + 32u], sum1);
        sum1 = fma(w2, x1[k + 64u], sum1);
        sum1 = fma(w3, x1[k + 96u], sum1);
        sum1 = fma(w4, x1[k + 128u], sum1);
        sum1 = fma(w5, x1[k + 160u], sum1);
        sum1 = fma(w6, x1[k + 192u], sum1);
        sum1 = fma(w7, x1[k + 224u], sum1);
    }
    for (; k < (uint)in_dim; k += 32u) {
        const float value = float(w[k]);
        sum0 = fma(value, x0[k], sum0);
        sum1 = fma(value, x1[k], sum1);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (thread_index_in_simdgroup == 0u) {
        out[out_row] = sum0;
        out[(size_t)out_dim + out_row] = sum1;
    }
}
