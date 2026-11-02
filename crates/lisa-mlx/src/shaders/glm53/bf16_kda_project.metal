// One-dispatch KDA projections for one to eight decode rows. Small dense banks
// run first, then q/k/v row groups. Each output row keeps the DS4 BF16 GEMV
// lane order.
{
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = simdgroups_per_threadgroup;
    const uint task_linear = threadgroup_position_in_grid.x;
    const uint in_n = (uint)in_dim;
    const uint qkv_n = (uint)qkv_dim;
    const uint low_n = (uint)low_dim;
    const uint heads_n = (uint)heads;
    const uint small_rows = 2u * low_n + heads_n;
    const uint small_groups = (small_rows + nsg - 1u) / nsg;
    const uint qkv_groups = (qkv_n + nsg - 1u) / nsg;
    const uint qkv_tasks = 3u * qkv_groups;
    const uint groups_per_row = small_groups + qkv_tasks;
    if (groups_per_row == 0u || n_rows == 0u || n_rows > 8u) return;
    const uint row = task_linear / groups_per_row;
    const uint task = task_linear - row * groups_per_row;
    if (row >= n_rows) return;
    const uint packed_n = 3u * qkv_n + heads_n;

    uint out_row;
    device const bfloat16_t *weights;
    device float *out;
    if (task < small_groups) {
        const uint small_row = task * nsg + sg;
        if (small_row >= small_rows) return;
        if (small_row < low_n) {
            out_row = small_row;
            weights = weights_fa;
            out = a_low + (size_t)row * low_n;
        } else if (small_row < 2u * low_n) {
            out_row = small_row - low_n;
            weights = weights_ga;
            out = g_low + (size_t)row * low_n;
        } else {
            out_row = small_row - 2u * low_n;
            weights = weights_beta;
            out = proj + (size_t)row * packed_n + 3u * qkv_n;
        }
    } else {
        const uint qkv_task = task - small_groups;
        const uint which = qkv_task / qkv_groups;
        out_row = (qkv_task % qkv_groups) * nsg + sg;
        if (which >= 3u || out_row >= qkv_n) return;
        weights = which == 0u ? weights_q : (which == 1u ? weights_k : weights_v);
        out = proj + (size_t)row * packed_n + which * qkv_n;
    }

    const device float *x_row = x + (size_t)row * in_n;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < in_n; k += 256u) {
        const bfloat16_t w0 = weights[(size_t)out_row * in_n + k];
        const bfloat16_t w1 = weights[(size_t)out_row * in_n + k + 32u];
        const bfloat16_t w2 = weights[(size_t)out_row * in_n + k + 64u];
        const bfloat16_t w3 = weights[(size_t)out_row * in_n + k + 96u];
        const bfloat16_t w4 = weights[(size_t)out_row * in_n + k + 128u];
        const bfloat16_t w5 = weights[(size_t)out_row * in_n + k + 160u];
        const bfloat16_t w6 = weights[(size_t)out_row * in_n + k + 192u];
        const bfloat16_t w7 = weights[(size_t)out_row * in_n + k + 224u];
        sum = fma(float(w0), x_row[k], sum);
        sum = fma(float(w1), x_row[k + 32u], sum);
        sum = fma(float(w2), x_row[k + 64u], sum);
        sum = fma(float(w3), x_row[k + 96u], sum);
        sum = fma(float(w4), x_row[k + 128u], sum);
        sum = fma(float(w5), x_row[k + 160u], sum);
        sum = fma(float(w6), x_row[k + 192u], sum);
        sum = fma(float(w7), x_row[k + 224u], sum);
    }
    for (; k < in_n; k += 32u) {
        sum = fma(float(weights[(size_t)out_row * in_n + k]), x_row[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        out[out_row] = sum;
    }
}
