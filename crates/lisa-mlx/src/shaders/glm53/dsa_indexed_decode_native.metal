// Native GLM-5.3 serial DSA decode; DS4 uses one group per head.
// The MLX kv_b affine rows are dequantized for the value projection in place,
// so the weighted latent vector never leaves threadgroup memory.
{
    const uint HEADS = (uint)n_heads;
    const uint KV_DIM = (uint)kv_lora_dim;
    const uint VALUE_DIM = (uint)value_dim;
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
            uint k = 0u;
            for (; k + 3u < KV_DIM; k += 4u) {
                device const half4 *kv4 =
                    (device const half4 *)(latent + row_offset + k);
                device const float4 *q4 =
                    (device const float4 *)(query_low + k);
                const float4 kv = (float4)(*kv4);
                const float4 qv = *q4;
                dotv += qv.x * kv.x + qv.y * kv.y +
                        qv.z * kv.z + qv.w * kv.w;
            }
            if (k < KV_DIM) {
                for (; k < KV_DIM; k++) {
                    const float kv = (float)latent[row_offset + k];
                    dotv += query_low[k] * kv;
                }
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
        const bool use_k1 = k1 < KV_DIM;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0u; s < (uint)n_selected; s++) {
            const int row = dense_all != 0 ? (int)s : rows[s];
            if (row >= 0 && row < tc) {
                const float weight = scores[s];
                if (use_k1) {
                    const device half2 *kv2 =
                        (device const half2 *)(latent + (size_t)row * KV_DIM + k0);
                    const float2 kv = (float2)(*kv2);
                    acc0 += weight * kv.x;
                    acc1 += weight * kv.y;
                } else {
                    const float kv0 = (float)latent[(size_t)row * KV_DIM + k0];
                    acc0 += weight * kv0;
                }
            }
        }
        lora[k0] = acc0 / denom;
        if (use_k1) lora[k1] = acc1 / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // __DSA_FUSED_LORA_CAPTURE__

    // Native Q8 short-value block schedule with original affine metadata.
    device const uchar *weight_bytes = (device const uchar *)kv_weight;
    for (uint value = tid; value < VALUE_DIM; value += nth) {
        const ulong weight_row = (ulong)head * 512u + 256u + value;
        float acc = 0.0f;
        for (uint block = 0; block < 16u; block++) {
            const uint base = block << 5;
            const ulong affine = weight_row * 8u + base / 64u;
            const float scale = kv_scales[affine];
            const float bias = kv_biases[affine];
            // Native Q8's unrolled512-dot factors each32-code block's scale:
            // acc = fma(scale, dot(signed_codes, x), acc). MLX stores unsigned
            // codes with an independent bias. Center at128 and explicitly add
            // the remaining affine bias under that same block schedule.
            const float residual_bias = fma(128.0f, scale, bias);
            float dot = 0.0f;
            float input_sum = 0.0f;
            #pragma clang loop unroll(full)
            for (uint qi = 0; qi < 32u; qi++) {
                const uint k = base + qi;
                const float centered_code = float(weight_bytes[weight_row * 512u + k]) - 128.0f;
                {
#pragma clang fp reassociate(off)
                    dot = fma(centered_code, lora[k], dot);
                    input_sum += lora[k];
                }
            }
            acc = fma(scale, dot, acc);
            if (residual_bias != 0.0f) acc = fma(residual_bias, input_sum, acc);
        }
        output[head * VALUE_DIM + value] = acc;
    }
}
