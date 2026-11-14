// GLM-5.3 indexer pool update, following ds4's kernel_glm53_indexer_pool_update.
// Each 128-thread group owns one four-token pool. Raw keys and gate features
// are separate F32 arrays; APE remains in its checkpoint BF16 representation.
// Keep the pool and head loops runtime-bounded like ds4's args-driven kernel:
// specializing these bounds changes Metal's reassociation near F16 boundaries.
{
    constexpr uint D = 128u;
    constexpr uint MAX_POOL = 4u;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint head_dim = (uint)hd_dim;
    const uint pool_size = (uint)pool_dim;
    if (head_dim != D || pool_size != MAX_POOL || row >= (uint)n_rows || tid >= D) return;

    threadgroup float *rows = shared;
    threadgroup float *means = rows + pool_size * head_dim;
    threadgroup float *invs = means + pool_size;
    const device float *raw = raw_key + (size_t)row * pool_size * head_dim;
    const device float *gates = gate + (size_t)row * pool_size * head_dim;
    const device bfloat16_t *ape_bf16 = ape;

    for (uint j = 0; j < pool_size; j++) {
        rows[(size_t)j * head_dim + tid] = raw[(size_t)j * head_dim + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < pool_size) {
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; d++) sum += rows[(size_t)tid * head_dim + d];
        const float mean = sum / (float)head_dim;
        float sumsq = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            const float delta = rows[(size_t)tid * head_dim + d] - mean;
            sumsq += delta * delta;
        }
        means[tid] = mean;
        invs[tid] = rsqrt(sumsq / (float)head_dim + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float logits[MAX_POOL];
    float maximum = -INFINITY;
    for (uint j = 0; j < pool_size; j++) {
        const ushort ape_bits = as_type<ushort>(ape_bf16[(size_t)j * head_dim + tid]);
        const float ape_value = as_type<float>((uint)ape_bits << 16);
        logits[j] = gates[(size_t)j * head_dim + tid] + ape_value;
        maximum = max(maximum, logits[j]);
    }
    float denom = 0.0f;
    for (uint j = 0; j < pool_size; j++) {
        logits[j] = exp(logits[j] - maximum);
        denom += logits[j];
    }
    float pooled = 0.0f;
    for (uint j = 0; j < pool_size; j++) {
        const float normalized =
            (rows[(size_t)j * head_dim + tid] - means[j]) * invs[j] * norm_weight[tid] + norm_bias[tid];
        // DS4's default-math output matches this effective contraction: fma(
        // logit * normalized, 1 / denom, pooled). Keep this operand grouping;
        // dividing the logit first changes F16 midpoint results even though
        // all input and normalization intermediates match.
        pooled = fma(logits[j] * normalized, 1.0f / denom, pooled);
    }
    pool[(size_t)row * head_dim + tid] = (half)pooled;
}
