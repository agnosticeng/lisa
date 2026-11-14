// GLM-5.3 BF16 weight / F32 activation GEMV from ds4's glm53_bf16.metal.

{
    const uint out_row = threadgroup_position_in_grid.x * simdgroups_per_threadgroup +
                         simdgroup_index_in_threadgroup;
    const uint token = threadgroup_position_in_grid.y;
    if (out_row >= (uint)out_dim || token >= (uint)n_rows) return;
    // Tiny tensors bind in constant address space; larger tensors use device.
    auto w = weights + (size_t)out_row * (size_t)in_dim;
    auto xr = x + (size_t)token * (size_t)in_dim;
    float sum = 0.0f;
    uint k = thread_index_in_simdgroup;
    for (; k + 224u < (uint)in_dim; k += 256u) {
        const bfloat16_t w0 = w[k];
        const bfloat16_t w1 = w[k + 32u];
        const bfloat16_t w2 = w[k + 64u];
        const bfloat16_t w3 = w[k + 96u];
        const bfloat16_t w4 = w[k + 128u];
        const bfloat16_t w5 = w[k + 160u];
        const bfloat16_t w6 = w[k + 192u];
        const bfloat16_t w7 = w[k + 224u];
        sum = fma(float(w0), xr[k], sum);
        sum = fma(float(w1), xr[k + 32u], sum);
        sum = fma(float(w2), xr[k + 64u], sum);
        sum = fma(float(w3), xr[k + 96u], sum);
        sum = fma(float(w4), xr[k + 128u], sum);
        sum = fma(float(w5), xr[k + 160u], sum);
        sum = fma(float(w6), xr[k + 192u], sum);
        sum = fma(float(w7), xr[k + 224u], sum);
    }
    for (; k < (uint)in_dim; k += 32u) {
        sum = fma(float(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (thread_index_in_simdgroup == 0u) {
        out[(size_t)token * (size_t)out_dim + out_row] = sum;
    }
}
