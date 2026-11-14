// SwiGLU on a PACKED gate|up buffer: `silu(gu[i]) * gu[F + i]` for a merged
// gate|up qmv output [B, 2F]. Values are bit-identical to track_swiglu2:
// gu[F + i] is the same bf16 element up[i] was, and mlx_silu is the same
// exact_header helper (same per-op bf16 rounding points).
//
// B-GENERIC (specs/13 re-audit): the original body indexed `gu[i]` /
// `gu[F + i]` directly, which is only correct for B == 1 — at B > 1 every
// row >= 1 read row 0's gate against row 1's up (batched-decode slots >= 1
// emitted token soup from the first step). Row-major decompose: at B == 1
// (r == 0, base == 0) the reads are IDENTICAL bytes in IDENTICAL order —
// bit-exact by construction on the serial path.
        const uint i = thread_position_in_grid.x;
        if (i >= (uint)(B * F)) return;
        const uint r = i / (uint)F;
        const uint j = i - r * (uint)F;
        const uint base = r * (uint)(2 * F);
        out[i] = mlx_silu(gu[base + j]) * gu[base + F + j];
    
