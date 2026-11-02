// Build the expert-major route index used by the grouped GLM MoE prefill.
// `selected` is token-major [token, top_k]. Each expert owns a dense slot list
// with capacity n_tokens; router top-k IDs are unique within a token.
// `work_items` is a compact sequence of (expert, route-row-start) pairs.
{
    const uint expert = thread_index_in_threadgroup;
    if (expert >= (uint)n_experts) return;

    uint route_count = 0;
    for (int token = 0; token < n_tokens; ++token) {
        const int route_base = token * top_k;
        for (int rank = 0; rank < top_k; ++rank) {
            if (selected[route_base + rank] == (int)expert) {
                route_ids[(size_t)expert * (size_t)n_tokens + route_count] =
                    route_base + rank;
                ++route_count;
            }
        }
    }
    counts[expert] = (int)route_count;

    threadgroup_barrier(mem_flags::mem_device);

    const uint own_tiles = (route_count + (uint)tile_size - 1u) /
        (uint)tile_size;
    uint tile_base = 0;
    for (uint prior = 0; prior < expert; ++prior) {
        const uint prior_count = (uint)counts[prior];
        tile_base += (prior_count + (uint)tile_size - 1u) /
            (uint)tile_size;
    }
    for (uint tile = 0; tile < own_tiles; ++tile) {
        const uint work_index = tile_base + tile;
        work_items[work_index * 2u] = (int)expert;
        work_items[work_index * 2u + 1u] = (int)(tile * (uint)tile_size);
    }
    if (expert + 1u == (uint)n_experts) {
        work_count[0] = (int)(tile_base + own_tiles);
    }
}
