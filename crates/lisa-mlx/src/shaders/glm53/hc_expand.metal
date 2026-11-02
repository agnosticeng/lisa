// GLM-5.3 mHC expand4, fused (ds4 dsv4_hc.metal expand4 port).
// Body only — the framework generates the kernel signature from the wrapper's
// param list (n_rows, hidden, y, residual, post, comb, dst) and injects
// thread_position_in_grid.
//
// new[t, dst, d] = post[t,dst] * y[t,d] + sum_src comb[t,src,dst] * residual[t,src,d]
// all inputs and output f32, matching ds4's graph buffers.
// Grid: (H * n_rows, 1, 1), threadgroup (1, 1, 1).
{
    const uint h = (uint)hidden;
    const uint g = thread_position_in_grid.x;
    if ((int)g >= n_rows * hidden) return;
    const uint d = g % h;
    const uint t = g / h;
    const float bv = y[(size_t)t * h + d];
    const float r0 = residual[((size_t)t * 4u + 0u) * h + d];
    const float r1 = residual[((size_t)t * 4u + 1u) * h + d];
    const float r2 = residual[((size_t)t * 4u + 2u) * h + d];
    const float r3 = residual[((size_t)t * 4u + 3u) * h + d];
    device const float *cm = comb + (size_t)t * 16u;
    for (uint dst_hc = 0u; dst_hc < 4u; dst_hc++) {
        float acc = bv * post[(size_t)t * 4u + dst_hc];
        acc += cm[0u * 4u + dst_hc] * r0;
        acc += cm[1u * 4u + dst_hc] * r1;
        acc += cm[2u * 4u + dst_hc] * r2;
        acc += cm[3u * 4u + dst_hc] * r3;
        dst[((size_t)t * 4u + dst_hc) * h + d] = acc;
    }
}
