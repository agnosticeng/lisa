// GLM-5.3 HC=4 split, collapse, and RMSNorm ported from DS4's
// kernel_dsv4_hc_split_weighted_sum_norm4. `x` is [N,4,hidden].
{
    const uint row = threadgroup_position_in_grid.x;
    const ushort tid = (ushort)thread_index_in_threadgroup;
    const ushort sgitg = (ushort)simdgroup_index_in_threadgroup;
    const ushort tiisg = (ushort)thread_index_in_simdgroup;
    const ushort ntg = (ushort)threads_per_threadgroup.x;
    if ((int64_t)row >= n_rows || hidden != 4096 || (hidden & 3) != 0) {
        return;
    }

    const uint n_embd = (uint)hidden;
    const uint n4 = n_embd >> 2;
    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + n_embd;
    threadgroup float *sum_shmem = pre_shmem + 4;

    const uint64_t mix_hc = 24;
    device const float *mix = mixes + (uint64_t)row * mix_hc;
    device float *out = split + (uint64_t)row * mix_hc;

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        const float4 post_value = 2.0f / (1.0f + exp(-post_z));
        *((device float4 *)(out + 4)) = post_value;
        *((device float4 *)(post + (uint64_t)row * 4)) = post_value;

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
        device float4 *comb_row = (device float4 *)(comb + (uint64_t)row * 16);
        comb_row[0] = r0;
        comb_row[1] = r1;
        comb_row[2] = r2;
        comb_row[3] = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * hidden + (uint64_t)row * 4 * hidden);
        device const float4 *x1 = (device const float4 *)(x + 1 * hidden + (uint64_t)row * 4 * hidden);
        device const float4 *x2 = (device const float4 *)(x + 2 * hidden + (uint64_t)row * 4 * hidden);
        device const float4 *x3 = (device const float4 *)(x + 3 * hidden + (uint64_t)row * 4 * hidden);
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    const float norm_arg = sumf / float(n_embd) + norm_eps;
    const float norm_scale = n_rows > 1 ? 1.0f / sqrt(norm_arg) : rsqrt(norm_arg);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * hidden);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(normed + (uint64_t)row * hidden);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}
