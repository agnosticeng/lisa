// GLM-5.3 mHC weighted_sum (collapse), fused (ds4 dsv4_hc.metal
// kernel_dsv4_hc_weighted_sum port). Body only — the framework generates the
// kernel signature from the wrapper's param list (n_rows, hidden, x, pre, dst)
// and injects thread_position_in_grid.
//
// dst[t,d] = sum_c pre[t,c] * residual[t,c,d] — F32 operands and output,
// matching ds4's graph buffers.
// Grid: (H * n_rows, 1, 1), threadgroup (1, 1, 1).
{
    const uint hdim = (uint)hidden;
    const uint g = thread_position_in_grid.x;
    if ((int)g >= n_rows * hidden) return;
    const uint d = g % hdim;
    const uint t = g / hdim;
    float acc = 0.0f;
    for (uint c = 0u; c < 4u; c++) {
        const float xv = residual[((size_t)t * 4u + c) * hdim + d];
        const float wv = pre[(size_t)t * 4u + c];
        acc += xv * wv;
    }
    dst[(size_t)t * hdim + d] = acc;
}
