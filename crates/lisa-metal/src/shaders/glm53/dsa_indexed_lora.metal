// GLM indexed MLA attention stage for the compact latent cache.
// Ported from ds4's kernel_glm_attention_indexed_batch_lora_group8_vec_impl
// for the GLM-5.3 no-RoPE DSA layout. Eight SIMD groups share each staged
// block of F16 latent rows while each group owns one attention head.
{
    constexpr uint GROUP_HEADS = 8u;
    constexpr uint STAGE_ROWS = 16u;
    constexpr uint KV_DIM = 512u;
    constexpr uint KV_VECS = KV_DIM / 4u;
    constexpr uint MAX_THREADS = 256u;

    const uint token = threadgroup_position_in_grid.y;
    const uint batch = threadgroup_position_in_grid.z;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint head_in_group = simdgroup_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x * GROUP_HEADS + head_in_group;
    const uint valid_prefix = (uint)selected_valid_prefix;
    if (token >= (uint)n_tokens || batch >= (uint)n_batches ||
        n_selected <= 0 || n_selected > 2051 || tid >= MAX_THREADS) {
        return;
    }

    const bool valid_head = head < (uint)n_heads;
    const uint safe_head = valid_head ? head : 0u;
    threadgroup half4 kv_shared[STAGE_ROWS * KV_VECS];

    const size_t q_low_row =
        (((size_t)batch * (size_t)n_tokens + (size_t)token) * (size_t)n_heads +
         (size_t)safe_head) * KV_DIM;
    device const float4 *low4 = (device const float4 *)(q_low + q_low_row);
    device const int *token_selected =
        selected + ((size_t)batch * (size_t)n_tokens + (size_t)token) *
                       (size_t)n_selected;
    device const half *latent_rows =
        latents + (size_t)batch * (size_t)tc * KV_DIM;

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < (uint)n_selected; base += STAGE_ROWS) {
        const uint rows = min(STAGE_ROWS, (uint)n_selected - base);
        for (uint off = tid; off < rows * KV_VECS; off += MAX_THREADS) {
            const uint rr = off / KV_VECS;
            const uint vv = off - rr * KV_VECS;
            const int row = dense_all != 0
                ? (int)(base + rr)
                : token_selected[base + rr];
            const bool guaranteed_valid = base + rr < valid_prefix;
            if (guaranteed_valid || (row >= 0 && row < tc)) {
                device const half4 *src =
                    (device const half4 *)(latent_rows + (size_t)row * KV_DIM);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const int row = dense_all != 0
                ? (int)(base + rr)
                : token_selected[base + rr];
            const bool valid_row = base + rr < valid_prefix || (row >= 0 && row < tc);
            threadgroup const half4 *kv_row = kv_shared + rr * KV_VECS;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
            }
            const float sum = simd_sum(partial);
            const float score = (valid_head && valid_row)
                ? sum * scale
                : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        const size_t out_row =
            (((size_t)batch * (size_t)n_tokens + (size_t)token) * (size_t)n_heads +
             (size_t)head) * KV_DIM;
        device float4 *out4 = (device float4 *)(lora_out + out_row);
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}
