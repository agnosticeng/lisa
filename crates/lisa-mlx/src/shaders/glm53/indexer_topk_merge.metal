// GLM-5.3 DSA indexer merge stage, ported from ds4/metal/argsort.metal.
// Each thread merges a disjoint rank interval from a pair of sorted runs.
{
    const uint group = threadgroup_position_in_grid.x;
    const uint row = group % (uint)n_rows;
    const uint pair = group / (uint)n_rows;
    const uint lane = thread_position_in_threadgroup.x;
    const uint start = pair * 2u * (uint)run_len;
    const uint width = (uint)work_width;
    const uint len0 = min((uint)run_len, start < width ? width - start : 0u);
    const uint second_start = start + (uint)run_len;
    const uint len1 = min((uint)run_len, second_start < width ? width - second_start : 0u);
    const uint total = len0 + len1;
    if (total == 0u) return;

    const uint threads = threads_per_threadgroup.x;
    const uint chunk = (total + threads - 1u) / threads;
    const uint k0 = lane * chunk;
    const uint limit = (uint)top_k;
    const uint k1 = min(min(k0 + chunk, total), limit);
    if (k0 >= limit || k0 >= total) return;

    const size_t row_offset = (size_t)row * width;
    device const int32_t *tmp0 = partial + row_offset + start;
    device const int32_t *tmp1 = tmp0 + run_len;
    device const float *score_row = scores + (size_t)row * (uint)n_comp;
    const uint low_init = k0 > len1 ? k0 - len1 : 0u;
    uint low = low_init;
    uint high = min(k0, len0);

    // Partition the two sorted runs at rank k0. Equal scores take the left
    // run first, matching ds4's merge comparison and preserving its run order.
    while (low < high) {
        const uint mid = (low + high) >> 1u;
        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1u];
        const float val0 = score_row[(uint)idx0];
        const float val1 = score_row[(uint)idx1];
        if (val0 >= val1) low = mid + 1u;
        else high = mid;
    }

    uint i = low;
    uint j = k0 - i;
    int32_t idx0 = 0;
    float val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = score_row[(uint)idx0];
    }
    int32_t idx1 = 0;
    float val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = score_row[(uint)idx1];
    }

    const size_t dst_offset = (size_t)row * (uint)out_width + start;
    for (uint k = k0; k < k1; ++k) {
        int32_t selected;
        if (i >= len0) {
            while (k < k1) partial_out[dst_offset + k++] = tmp1[j++];
            break;
        } else if (j >= len1) {
            while (k < k1) partial_out[dst_offset + k++] = tmp0[i++];
            break;
        } else if (val0 >= val1) {
            selected = idx0;
            ++i;
            if (i < len0) {
                idx0 = tmp0[i];
                val0 = score_row[(uint)idx0];
            }
        } else {
            selected = idx1;
            ++j;
            if (j < len1) {
                idx1 = tmp1[j];
                val1 = score_row[(uint)idx1];
            }
        }
        partial_out[dst_offset + k] = selected;
    }
}
