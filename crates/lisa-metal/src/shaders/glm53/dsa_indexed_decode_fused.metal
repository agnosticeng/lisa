// GLM-5.3 serial DSA decode, following DS4's one-threadgroup-per-head kernel.
// The MLX kv_b affine rows are dequantized for the value projection in place,
// so the weighted latent vector never leaves threadgroup memory.
{
    constexpr uint HEADS = 64u;
    constexpr uint KV_DIM = 512u;
    constexpr uint VALUE_DIM = 256u;
    constexpr uint MAX_SELECTED = 2051u;

    const uint head = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint nth = threads_per_threadgroup.x;
    if (head >= HEADS || n_selected <= 0 || n_selected > (int)MAX_SELECTED ||
        tc <= 0 || group_size <= 0 || KV_DIM % (uint)group_size != 0u) {
        return;
    }

    threadgroup float *red = shared;
    threadgroup float *scores = red + 256u;
    threadgroup float *lora = scores + (uint)n_selected;
    device const float *query_low = q_low + (size_t)head * KV_DIM;
    device const half *latent = latents;
    device const int *rows = selected;

    // Match DS4's indexed-decode score order: one thread owns a set of rows,
    // and each row's 512-value dot accumulates in increasing component order.
    float local_max = -INFINITY;
    for (uint s = tid; s < (uint)n_selected; s += nth) {
        const int row = dense_all != 0 ? (int)s : rows[s];
        float score = -INFINITY;
        if (row >= 0 && row < tc) {
            const size_t row_offset = (size_t)row * KV_DIM;
            float dotv = 0.0f;
            for (uint k = 0u; k < KV_DIM; k += 4u) {
                device const half4 *kv4 =
                    (device const half4 *)(latent + row_offset + k);
                device const float4 *q4 =
                    (device const float4 *)(query_low + k);
                const float4 kv = (float4)(*kv4);
                const float4 qv = *q4;
                dotv += qv.x * kv.x + qv.y * kv.y +
                        qv.z * kv.z + qv.w * kv.w;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < (uint)n_selected; s += nth) {
        const float weight = exp(scores[s] - max_score);
        scores[s] = weight;
        local_sum += weight;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Two adjacent dimensions per thread, accumulated across selected rows
    // in order, as in DS4's GLM indexed-decode kernel.
    for (uint k0 = tid * 2u; k0 < KV_DIM; k0 += nth * 2u) {
        const uint k1 = k0 + 1u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0u; s < (uint)n_selected; s++) {
            const int row = dense_all != 0 ? (int)s : rows[s];
            if (row >= 0 && row < tc) {
                const float weight = scores[s];
                const device half2 *kv2 =
                    (device const half2 *)(latent + (size_t)row * KV_DIM + k0);
                const float2 kv = (float2)(*kv2);
                acc0 += weight * kv.x;
                acc1 += weight * kv.y;
            }
        }
        lora[k0] = acc0 / denom;
        lora[k1] = acc1 / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // __DSA_FUSED_LORA_CAPTURE__

    // Each lane owns one output row. Match DS4's native Q8 row-dot order by
    // accumulating components in increasing k after MLX affine dequantization.
    const uint d = tid;
    if (d < VALUE_DIM) {
        const uint weight_row = head * KV_DIM + 256u + d;
        const uint groups = KV_DIM / (uint)group_size;
        device const uint *packed = kv_weight + (size_t)weight_row * 128u;
        float sum = 0.0f;
        for (uint block = 0u; block < 128u; block++) {
            const uint k = block * 4u;
            for (uint j = 0u; j < 4u; j++) {
                const uint index = k + j;
                const uint word = packed[index >> 2u];
                const uint code = (word >> ((index & 3u) * 8u)) & 0xffu;
                const uint group = weight_row * groups + index / (uint)group_size;
                const float weight = fma((float)code, kv_scales[group], kv_biases[group]);
                sum = fma(weight, lora[index], sum);
            }
        }
        output[head * VALUE_DIM + d] = sum;
    }
}
