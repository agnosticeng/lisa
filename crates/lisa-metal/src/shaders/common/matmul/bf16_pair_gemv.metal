// Paired GLM-5.3 BF16-weight / F32-activation GEMV.
// Each output row retains the DS4 lane-strided FMA and simd_sum order.
{
    const uint task_row = threadgroup_position_in_grid.x * simdgroups_per_threadgroup +
                          simdgroup_index_in_threadgroup;
    const uint token = threadgroup_position_in_grid.y;
    const bool which_b = task_row >= (uint)out_dim_a;
    const uint out_row = which_b ? task_row - (uint)out_dim_a : task_row;
    const uint in_dim = which_b ? (uint)in_dim_b : (uint)in_dim_a;
    const uint output_dim = which_b ? (uint)out_dim_b : (uint)out_dim_a;
    if (task_row >= (uint)out_dim_a + (uint)out_dim_b || token >= (uint)n_rows) return;

    device const bfloat16_t *w = (which_b ? weights_b : weights_a) +
                                  (size_t)out_row * (size_t)in_dim;
    device const float *xr = (which_b ? x_b : x_a) +
                              (size_t)token * (size_t)in_dim;
    device float *result = which_b ? out_b : out_a;
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
        result[(size_t)token * (size_t)output_dim + out_row] = sum;
    }
}
