// GLM-5.3 mHC split+Sinkhorn, ported from ds4's HC=4 float4 specialization.
// `mixes` [N,24] -> `out` [N,24], containing pre[4], post[4], and comb[16].
{
    const uint row = thread_position_in_grid.x;
    if ((int)row >= n_rows) return;

    device const float *mix = mixes + (size_t)row * 24u;
    device float *o = out + (size_t)row * 24u;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    const float4 pre_z = *((device const float4 *)mix) * pre_scale +
        *((device const float4 *)base);
    const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + eps;
    *((device float4 *)o) = pre;

    const float4 post_z = *((device const float4 *)(mix + 4)) * post_scale +
        *((device const float4 *)(base + 4));
    *((device float4 *)(o + 4)) = 2.0f / (1.0f + exp(-post_z));

    float4 r0 = *((device const float4 *)(mix + 8)) * comb_scale +
        *((device const float4 *)(base + 8));
    float4 r1 = *((device const float4 *)(mix + 12)) * comb_scale +
        *((device const float4 *)(base + 12));
    float4 r2 = *((device const float4 *)(mix + 16)) * comb_scale +
        *((device const float4 *)(base + 16));
    float4 r3 = *((device const float4 *)(mix + 20)) * comb_scale +
        *((device const float4 *)(base + 20));

    const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
    const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
    const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
    const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));
    r0 = exp(r0 - m0);
    r1 = exp(r1 - m1);
    r2 = exp(r2 - m2);
    r3 = exp(r3 - m3);

    r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + eps;
    r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + eps;
    r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + eps;
    r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + eps;

    float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + eps);
    r0 *= col_inv;
    r1 *= col_inv;
    r2 *= col_inv;
    r3 *= col_inv;

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + eps);
        r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + eps);
        r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + eps);
        r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + eps);

        col_inv = 1.0f / (r0 + r1 + r2 + r3 + eps);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;
    }

    *((device float4 *)(o + 8)) = r0;
    *((device float4 *)(o + 12)) = r1;
    *((device float4 *)(o + 16)) = r2;
    *((device float4 *)(o + 20)) = r3;
}
