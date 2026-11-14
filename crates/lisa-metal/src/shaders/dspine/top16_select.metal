// The global top 16 must belong to the 16 chunks with the best maxima.
// Resolve cutoff ties by token ID, then sort the selected IDs for refinement.
const uint lane = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.x;
threadgroup uint group_ids[8];
threadgroup float group_values[8];
threadgroup uint chunks[16];
threadgroup uint result[16];
uint ids[SLOTS];
float values[SLOTS];
for (uint slot = 0; slot < SLOTS; ++slot) {
    const uint c = slot * 256 + lane;
    ids[slot] = c < CHUNKS ? maxima[row * CHUNKS + c] : UINT_MAX;
    values[slot] = ids[slot] != UINT_MAX ? float(scores[size_t(row) * V + ids[slot]]) : 0.0f;
}
for (uint pick = 0; pick < 16; ++pick) {
    uint id = UINT_MAX;
    float value = 0.0f;
    for (uint slot = 0; slot < SLOTS; ++slot) {
        if (dspine_before(values[slot], ids[slot], value, id)) {
            id = ids[slot];
            value = values[slot];
        }
    }
    uint best = dspine_group_best(id, value, lane, group_ids, group_values);
    if (lane == 0) chunks[pick] = best == UINT_MAX ? UINT_MAX : best / 64;
    for (uint slot = 0; slot < SLOTS; ++slot) {
        if (ids[slot] == best) ids[slot] = UINT_MAX;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint candidates_local[4];
float scores_local[4];
for (uint slot = 0; slot < 4; ++slot) {
    uint index = slot * 256 + lane;
    uint chunk = chunks[index / 64];
    uint id = chunk == UINT_MAX ? UINT_MAX : chunk * 64 + index % 64;
    if (id >= V) id = UINT_MAX;
    candidates_local[slot] = id;
    scores_local[slot] = id != UINT_MAX ? float(scores[size_t(row) * V + id]) : 0.0f;
}
for (uint pick = 0; pick < 16; ++pick) {
    uint id = UINT_MAX;
    float value = 0.0f;
    for (uint slot = 0; slot < 4; ++slot) {
        if (dspine_before(scores_local[slot], candidates_local[slot], value, id)) {
            id = candidates_local[slot];
            value = scores_local[slot];
        }
    }
    uint best = dspine_group_best(id, value, lane, group_ids, group_values);
    if (lane == 0) result[pick] = best;
    for (uint slot = 0; slot < 4; ++slot) {
        if (candidates_local[slot] == best) candidates_local[slot] = UINT_MAX;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
    for (uint i = 1; i < 16; ++i) {
        uint id = result[i];
        uint j = i;
        while (j > 0 && result[j - 1] > id) {
            result[j] = result[j - 1];
            --j;
        }
        result[j] = id;
    }
    for (uint i = 0; i < 16; ++i) selected[row * 16 + i] = result[i];
}
