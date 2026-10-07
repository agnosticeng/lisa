// Fused VERIFY QK-norm + partial RoPE (specs/24): the s>1 sibling of
// qk_norm_rope.metal. One launch per attention layer replaces, for q AND k,
// the two de-interleave `.contiguous()` copies, q_norm, k_norm, both
// transposes and both `rope_partial` chains, and emits the gate half
// de-interleaved for the sigmoid-gate tail.
//
// Inputs: qg [S, HQ*2*HD] — the stock interleaved q|gate projection rows
// (layout [head][2][hd], gate at +HD); k [S, HK*HD] contiguous; qw/kw [HD];
// cosv/sinv [S, RD] pre-cast bf16 (one angle row per verify row); eps f32.
// Outputs: oq [HQ, S, HD] head-major ([1,HQ,S,HD] as SDPA consumes), ok
// [HK, S, HD] head-major, og [S, HQ*HD] row-major (raw de-interleaved gate).
// Launched with (HQ+HK)*S threadgroups, 64 threads each — ONE threadgroup
// per (row, head), so the per-row math is bit-identical to the pinned s==1
// kernel (same rms reduction over HD=256, same bf16 rounding points, same
// precise::rsqrt); only the addressing gained a row axis.
//
// Parity contract vs the composed chain (qwen3_5 geometry hd=256, rd=64):
// identical to qk_norm_rope.metal's (see its header) — rms mirrors the
// built-in single-row kernel (N_READS=4, per-lane sequential f32 sum,
// simd_sum butterfly, 32-slot partial fold, weight AFTER the bf16 rounding),
// rotation couples (j, j+32) through simd_shuffle with pre-cast bf16 angles,
// pass-through tail j >= rd untouched.

constexpr uint HD = 256;

  const uint HT = uint(HQ + HK);
  const uint head = threadgroup_position_in_grid.x % HT;
  const uint row = threadgroup_position_in_grid.x / HT;
  const uint lid = thread_index_in_threadgroup;
  const uint simd_lane = thread_index_in_simdgroup;
  const uint simd_group = simdgroup_index_in_threadgroup;

  const bool is_q = head < uint(HQ);
  const device InT* w = is_q ? qw : kw;
  device InT* dst = is_q ? (oq + (head * uint(S) + row) * HD)
                         : (ok + ((head - uint(HQ)) * uint(S) + row) * HD);
  const device InT* src = is_q
      ? (qg + (row * uint(HQ) + head) * 2 * HD)
      : (k + (row * uint(HK) + (head - uint(HQ))) * HD);

  constexpr uint RD = 64;
  constexpr uint HALF_RD = RD / 2;
  constexpr uint SIMDS = 32;

  threadgroup float local_sums[SIMDS];
  threadgroup float local_inv[1];

  const uint base = lid * 4;
  float xs[4];
  float acc = 0.0f;
  for (uint i = 0; i < 4; ++i) {
    xs[i] = static_cast<float>(src[base + i]);
    acc += xs[i] * xs[i];
  }
  acc = simd_sum(acc);
  if (simd_group == 0) {
    local_sums[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) {
    local_sums[simd_group] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    acc = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0) {
      local_inv[0] = metal::precise::rsqrt(acc / static_cast<float>(HD) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv[0];

  InT nrm[4];
  for (uint i = 0; i < 4; ++i) {
    const InT scaled = static_cast<InT>(xs[i] * inv);
    nrm[i] = static_cast<InT>(static_cast<float>(w[base + i]) * static_cast<float>(scaled));
  }

  // Partner element of the rotary pair: j <-> j + HALF_RD, four elements per
  // lane => partner lane is lane ^ (HALF_RD / 4).
  float pf[4];
  for (uint i = 0; i < 4; ++i) {
    pf[i] = simd_shuffle(static_cast<float>(nrm[i]), simd_lane ^ (HALF_RD / 4));
  }
  const device InT* crow = cosv + row * RD;
  const device InT* srow = sinv + row * RD;
  for (uint i = 0; i < 4; ++i) {
    const uint j = base + i;
    InT r;
    if (j < HALF_RD) {
      const InT t1 = static_cast<InT>(static_cast<float>(nrm[i]) * static_cast<float>(crow[j]));
      const InT t2 = static_cast<InT>(pf[i] * static_cast<float>(srow[j]));
      r = static_cast<InT>(static_cast<float>(t1) - static_cast<float>(t2));
    } else if (j < RD) {
      const InT t1 = static_cast<InT>(static_cast<float>(nrm[i]) * static_cast<float>(crow[j]));
      const InT t2 = static_cast<InT>(pf[i] * static_cast<float>(srow[j]));
      r = static_cast<InT>(static_cast<float>(t1) + static_cast<float>(t2));
    } else {
      r = nrm[i];
    }
    dst[j] = r;
  }

  // Gate de-interleave: the q head-group also copies its gate half through,
  // raw (no norm, no rotation) — replaces one of the two `.contiguous()`
  // copies the composed chain paid per layer.
  if (is_q) {
    const device InT* gsrc = src + HD;
    device InT* gdst = og + (row * uint(HQ) + head) * HD;
    for (uint i = 0; i < 4; ++i) {
      gdst[base + i] = gsrc[base + i];
    }
  }
