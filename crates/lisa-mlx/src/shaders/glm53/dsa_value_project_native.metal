// Native DS4 GLM-5.3 value projection schedule with MLX affine storage.
{
    const uint n_tokens = (uint)n_rows / (uint)n_heads;
    const uint tid = thread_index_in_threadgroup;
    device const uchar *weight = (device const uchar *)kv_weight;
    device char *heads = (device char *)output;
    threadgroup float *x = shared;
    const uint head = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    if (head >= n_heads || token >= n_tokens) return;
    device const float *src = reinterpret_cast<device const float *>(lora) +
                             ((ulong)token * n_heads + head) * kv_lora_dim;
    device float *out = reinterpret_cast<device float *>(heads) +
                       ((ulong)token * n_heads + head) * value_dim;
    for (uint k = tid; k < kv_lora_dim; k += threads_per_threadgroup.x) x[k] = src[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint value = tid; value < value_dim; value += threads_per_threadgroup.x) {
        const ulong weight_row = (ulong)head * kv_lora_dim + value_dim + value;
        float acc = 0.0f;
        for (uint block = 0; block < 16u; block++) {
            const uint base = block << 5;
            const ulong affine = weight_row * (kv_lora_dim / group_size) + base / group_size;
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
                const float centered_code = float(weight[weight_row * kv_lora_dim + k]) - 128.0f;
                {
#pragma clang fp reassociate(off)
                    dot = fma(centered_code, x[k], dot);
                    input_sum += x[k];
                }
            }
            acc = fma(scale, dot, acc);
            if (residual_bias != 0.0f) acc = fma(residual_bias, input_sum, acc);
        }
        out[value] = acc;
    }
}
