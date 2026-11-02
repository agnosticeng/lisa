// GLM-5.3 DSA decode attention, stage 2 of 2 (ds4 fused flash_attn decode
// path port; see dsa_scores.metal for the split rationale). Body only — the
// framework generates the kernel signature from the wrapper's param list
// (tc, scores, v, out) and injects thread_position_in_grid.
//
// scores [B*64,Tc], v [B,64,Tc,256], output [B*64,256], all F32.
// One thread per (row, channel): grid (rows*256, 1, 1), threadgroup (1, 1, 1).
{
    const uint g = thread_position_in_grid.x;
    const uint T = (uint)tc;
    const uint rows = (uint)n_rows;
    if (g >= rows * 256u) return;
    const uint row = g / 256u;
    const uint d = g % 256u;
    device const float *sr = scores + (size_t)row * T;
    device const float *vr = v + (size_t)row * T * 256u + d;

    float s0 = sr[0];
    float mx = s0;
    for (uint j = 1u; j < T; j++) {
        mx = max(mx, sr[j]);
    }
    float sum = 0.0f;
    for (uint j = 0u; j < T; j++) {
        sum += metal::precise::exp(sr[j] - mx);
    }
    float acc = 0.0f;
    for (uint j = 0u; j < T; j++) {
        const float p = metal::precise::exp(sr[j] - mx) / sum;
        acc += p * vr[(size_t)j * 256u];
    }
    out[g] = acc;
}
