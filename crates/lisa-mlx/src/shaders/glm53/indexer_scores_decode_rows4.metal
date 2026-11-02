// DS4's four-row one-token DSA indexer scorer. One SIMD group computes four
// pooled-key rows; a threadgroup covers sixteen rows. Each score keeps the
// direct scorer's dot, simd_sum, and head-ordered accumulation.
{
    const uint group = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint rows_per_batch = ((uint)n_rows + 15u) / 16u;
    if (rows_per_batch == 0u) return;
    const uint batch = group / rows_per_batch;
    const uint tile = group - batch * rows_per_batch;
    if (batch >= (uint)n_batches) return;

    const uint first = tile * 16u + simdgroup * 4u;
    const uint r0 = first;
    const uint r1 = first + 1u;
    const uint r2 = first + 2u;
    const uint r3 = first + 3u;
    // Padded rows reuse row zero for safe reads; stores remain bounds-checked.
    const uint p0 = r0 < (uint)n_rows ? r0 : 0u;
    const uint p1 = r1 < (uint)n_rows ? r1 : 0u;
    const uint p2 = r2 < (uint)n_rows ? r2 : 0u;
    const uint p3 = r3 < (uint)n_rows ? r3 : 0u;
    const size_t base = (size_t)batch * (size_t)n_rows * 128u;
    const device half* keys = pooled + base;
    const float4 k0 = float4(float(keys[(size_t)p0 * 128u + lane * 4u + 0u]),
                             float(keys[(size_t)p0 * 128u + lane * 4u + 1u]),
                             float(keys[(size_t)p0 * 128u + lane * 4u + 2u]),
                             float(keys[(size_t)p0 * 128u + lane * 4u + 3u]));
    const float4 k1 = float4(float(keys[(size_t)p1 * 128u + lane * 4u + 0u]),
                             float(keys[(size_t)p1 * 128u + lane * 4u + 1u]),
                             float(keys[(size_t)p1 * 128u + lane * 4u + 2u]),
                             float(keys[(size_t)p1 * 128u + lane * 4u + 3u]));
    const float4 k2 = float4(float(keys[(size_t)p2 * 128u + lane * 4u + 0u]),
                             float(keys[(size_t)p2 * 128u + lane * 4u + 1u]),
                             float(keys[(size_t)p2 * 128u + lane * 4u + 2u]),
                             float(keys[(size_t)p2 * 128u + lane * 4u + 3u]));
    const float4 k3 = float4(float(keys[(size_t)p3 * 128u + lane * 4u + 0u]),
                             float(keys[(size_t)p3 * 128u + lane * 4u + 1u]),
                             float(keys[(size_t)p3 * 128u + lane * 4u + 2u]),
                             float(keys[(size_t)p3 * 128u + lane * 4u + 3u]));

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    const device float* q_row = q + (size_t)batch * 32u * 128u;
    const device float* w_row = weights + (size_t)batch * 32u;
    for (uint head = 0u; head < 32u; ++head) {
        const device float4* q4 = (device const float4*)(q_row + (size_t)head * 128u);
        float s0 = simd_sum(dot(q4[lane], k0));
        float s1 = simd_sum(dot(q4[lane], k1));
        float s2 = simd_sum(dot(q4[lane], k2));
        float s3 = simd_sum(dot(q4[lane], k3));
        if (lane == 0u) {
            const float t0 = max(s0 * scale, 0.0f) * w_row[head];
            const float t1 = max(s1 * scale, 0.0f) * w_row[head];
            const float t2 = max(s2 * scale, 0.0f) * w_row[head];
            const float t3 = max(s3 * scale, 0.0f) * w_row[head];
            {
#pragma clang fp contract(off)
                a0 += t0;
                a1 += t1;
                a2 += t2;
                a3 += t3;
            }
        }
    }
    const size_t score_base = (size_t)batch * (size_t)n_rows;
    if (lane == 0u) {
        if (r0 < (uint)n_rows) scores[score_base + r0] = a0;
        if (r1 < (uint)n_rows) scores[score_base + r1] = a1;
        if (r2 < (uint)n_rows) scores[score_base + r2] = a2;
        if (r3 < (uint)n_rows) scores[score_base + r3] = a3;
    }
}
