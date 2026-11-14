// GLM-5.3 row/group RMS norm, fused (ds4 norm.metal `kernel_rms_norm_fuse_impl`
// port, geometry = the certified kda_output norm block).
// Body only — the framework generates the kernel signature from the wrapper's
// param list (n_groups, eps, x, y) and injects thread_position_in_grid /
// thread_index_in_simdgroup.
//
// Semantics = the engine's rms_row_exact for the h = 128 group case:
//   - one threadgroup of 32 lanes owns ONE 128-channel group (4 channels/lane),
//   - lane sum: the 4 squares accumulated sequentially in f32 (order pinned),
//   - simd_sum over the 32 lanes (xor butterfly, same association as the
//     engine's half-fold),
//   - inv = rsqrt(total / 128 + eps) per group,
//   - out = bf16(x_f32 * inv) — the weight multiply stays OUTSIDE (the engine
//     applies the bf16 weight after the rounding, so fusing it here with an
//     f32 multiply would change the rounding).
//
// Grid: (n_groups * 32, rows, 1), threadgroup (32, 1, 1), group = gid.x / 32.
{
    const uint group = thread_position_in_grid.x / 32u;
    const uint row = thread_position_in_grid.y;
    if (group >= (uint)n_groups) return;
    const ulong row_base = (ulong)row * (ulong)n_groups * 128u;
    const ulong base = row_base + (ulong)group * 128u + (ulong)thread_index_in_simdgroup * 4u;
    float vals[4];
    float sq = 0.0f;
    for (uint i = 0u; i < 4u; i++) {
        vals[i] = static_cast<float>(x[base + i]);
        sq += vals[i] * vals[i];
    }
    const float total = simd_sum(sq);
    const float inv = rsqrt(total / 128.0f + eps);
    for (uint i = 0u; i < 4u; i++) {
        y[base + i] = static_cast<bfloat16_t>(vals[i] * inv);
    }
}
