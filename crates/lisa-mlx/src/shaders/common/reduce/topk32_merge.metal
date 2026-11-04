// Merge two top-32 lists into a compact top-32 list. Each candidate finds its
// rank in the other list; the total order makes all output ranks unique.
{
    const uint lane = thread_position_in_threadgroup.x;
    const uint pair = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    if (pair * 2 + 1 >= uint(runs)) {
        if (lane < 32) {
            merged[(size_t(row) * uint(out_runs) + pair) * 32 + lane] =
                candidates[(size_t(row) * uint(runs) + pair * 2) * 32 + lane];
        }
        return;
    }
    const uint own = pair * 2 + lane / 32;
    const uint other = own ^ 1;
    const uint local = lane % 32;
    const uint id = own < uint(runs)
        ? candidates[(size_t(row) * uint(runs) + own) * 32 + local] : UINT_MAX;
    if (id == UINT_MAX) return;
    const float value = float(scores[size_t(row) * width + id]);
    uint low = 0, high = other < uint(runs) ? 32 : 0;
    while (low < high) {
        const uint middle = (low + high) / 2;
        const uint other_id = candidates[(size_t(row) * uint(runs) + other) * 32 + middle];
        const float other_value = other_id != UINT_MAX
            ? float(scores[size_t(row) * width + other_id]) : 0.0f;
        if (lisa_topk_before(other_value, other_id, value, id)) low = middle + 1;
        else high = middle;
    }
    const uint rank = local + low;
    if (rank < 32) {
        merged[(size_t(row) * uint(out_runs) + pair) * 32 + rank] = id;
    }
}
