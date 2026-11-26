// DS4 b4e3faf native GLM-5.3 QK schedule with MLX affine storage loads.
// A group has 256 threads per head/token; each thread computes two serial dots.
{
    const uint n_tokens = (uint)n_queries;
    const uint tid = thread_index_in_threadgroup;
    device const uchar *weight = (device const uchar *)kv_weight;
    const uint head_batch = threadgroup_position_in_grid.x;
    const uint head = head_batch % (uint)n_heads;
    const uint batch = head_batch / (uint)n_heads;
    const uint token = threadgroup_position_in_grid.y;
    if (head >= n_heads || token >= n_tokens) return;
    device const float *qh = q + (((ulong)batch * (uint)n_heads + head) * n_tokens + token) * qk_dim;
    device float *out = output + (((ulong)batch * n_tokens + token) * (uint)n_heads + head) * kv_lora_dim;
    for (uint latent = tid; latent < kv_lora_dim; latent += threads_per_threadgroup.x) {
        float acc = 0.0f;
        const uint n_blocks = (qk_nope + 31u) >> 5;
        for (uint block = 0; block < n_blocks; block++) {
            const uint base = block << 5;
            const uint count = min(32u, qk_nope - base);
            for (uint qi = 0; qi < count; qi++) {
                const uint col = base + qi;
                const ulong weight_row = (ulong)head * kv_lora_dim + col;
                const ulong affine = weight_row * (kv_lora_dim / group_size) + latent / group_size;
                const float value = fma(float(weight[weight_row * kv_lora_dim + latent]),
                    kv_scales[affine],
                    kv_biases[affine]);
                acc += value * qh[col];
            }
        }
        out[latent] = acc;
    }
}
