// GLM-5.3 MoE router selection, ported from ds4's threadgroup bitonic sort.
// One group handles one token; padded lanes sort as -infinity. The chosen
// experts are ordered by (sigmoid(logit)+correction_bias) descending, with
// lower expert id first on ties. Route weights use sigmoid(logit), normalized
// across selected experts, then multiply by the configured route scale.
{
    const uint token = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint sort_width = n_expert > 256 ? 512u : 256u;
    const uint E = (uint)n_expert;
    const uint K = (uint)top_k;

    threadgroup float scores[512];
    threadgroup int indices[512];
    device const float *lg = logits + (size_t)token * E;

    const bool active = tid < E;
    float p = 0.0f;
    if (active) {
        const float x = lg[tid];
        if (x >= 0.0f) {
            const float e = exp(-x);
            p = 1.0f / (1.0f + e);
        } else {
            const float e = exp(x);
            p = e / (1.0f + e);
        }
        probs[(size_t)token * E + tid] = p;
    }
    scores[tid] = active ? p + bias[tid] : -INFINITY;
    indices[tid] = (int)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 2u; width <= sort_width; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            const uint other = tid ^ stride;
            if (other > tid) {
                const int a = indices[tid];
                const int b = indices[other];
                const bool b_better = scores[b] > scores[a] ||
                    (scores[b] == scores[a] && b < a);
                const bool a_better = scores[a] > scores[b] ||
                    (scores[a] == scores[b] && a < b);
                const bool descending = (tid & width) == 0u;
                const bool swap = descending ? b_better : a_better;
                if (swap) {
                    indices[tid] = b;
                    indices[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint k_used = min(K, E);
    if (tid < k_used) {
        selected[(size_t)token * K + tid] = indices[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < k_used) {
        float sum = 0.0f;
        for (uint i = 0u; i < k_used; i++) {
            const uint expert = (uint)indices[i];
            sum += probs[(size_t)token * E + expert];
        }
        sum = max(sum, 6.103515625e-5f);
        const uint expert = (uint)indices[tid];
        weights[(size_t)token * K + tid] =
            probs[(size_t)token * E + expert] / sum * scale;
    }
}
