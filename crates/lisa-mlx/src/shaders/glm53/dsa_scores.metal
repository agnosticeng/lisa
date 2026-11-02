// GLM-5.3 DSA decode attention, stage 1 of 2 (ds4 fused flash_attn decode
// path port; the framework has no threadgroup scratch, so the single
// flash-attention kernel is split into scores + weighted-V stages with
// identical rounding points). Body only — the framework generates the kernel
// signature from the wrapper's param list (n_heads, tc, scale, q, k, keep,
// scores) and injects thread_position_in_grid.
//
// Decode (t == 1) only. q [B, 64, 256], k [B, 64, Tc, 256], and keep [B,Tc]
// are F32. Output scores [B*64,Tc] stays F32 like ds4's graph buffers.
// Rows are [B, heads] flattened (row = b * n_heads + h). One thread per
// (row, token): grid (rows*tc, 1, 1), threadgroup (1, 1, 1).
{
    const uint g = thread_position_in_grid.x;
    const uint T = (uint)tc;
    const uint rows = (uint)n_rows;
    if (g >= rows * T) return;
    const uint row = g / T;
    const uint j = g % T;
    const uint b = row / (uint)n_heads;
    device const float *qr = q + (size_t)row * 256u;
    device const float *kr = k + ((size_t)row * T + j) * 256u;
    float dot = 0.0f;
    for (uint d = 0u; d < 256u; d++) {
        dot += qr[d] * kr[d];
    }
    scores[g] = dot * scale + keep[(size_t)b * T + j];
}
