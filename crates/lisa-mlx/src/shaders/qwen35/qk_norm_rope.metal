// Fused decode QK-norm + partial RoPE (specs/08 §2): one launch replaces
// q_norm → k_norm → transpose → rope_partial → transpose for q AND k.
// Inputs: q [HQ*HD], k [HK*HD] (contiguous, s=1); qw/kw [HD]; cosv/sinv [RD]
// pre-cast bf16; eps f32 scalar. Outputs: oq [HQ*HD], ok [HK*HD] head-major.
// Launched with (HQ+HK)*64 threads, 64 per threadgroup.
//
// Parity contract vs the composed chain (qwen3_5 geometry hd=256, rd=64):
// - the RMS half mirrors the built-in `rms_single_row` (N_READS=4, 64 threads
//   per 256-wide row: each lane owns 4 consecutive channels, per-lane
//   sequential f32 sum, simd_sum butterfly, per-simdgroup partials folded by
//   a second simd_sum over the zero-padded 32 slots, precise::rsqrt, weight
//   applied AFTER the bf16 rounding: out = w * T(x*inv));
// - the rotation mirrors `rope_partial` with PRE-CAST bf16 angles: the pair
//   (j, j+32) couples through simd_shuffle, intermediates round bf16 at the
//   same points (T(x*c), T(partner*s), then one add/sub round), pass-through
//   tail j >= rd untouched. Built-in RMS kernels compile with precise math
//   functions, so this kernel (compiled Math::Jit) pins precise::rsqrt
//   explicitly.

constexpr uint HD = 256;

  const uint head = threadgroup_position_in_grid.x;
  const uint lid = thread_index_in_threadgroup;
  const uint simd_lane = thread_index_in_simdgroup;
  const uint simd_group = simdgroup_index_in_threadgroup;

  const bool is_q = head < uint(HQ);
  const device InT* src = is_q ? (q + head * HD) : (k + (head - uint(HQ)) * HD);
  const device InT* w = is_q ? qw : kw;
  device InT* dst = is_q ? (oq + head * HD) : (ok + (head - uint(HQ)) * HD);

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
  for (uint i = 0; i < 4; ++i) {
    const uint j = base + i;
    InT r;
    if (j < HALF_RD) {
      const InT t1 = static_cast<InT>(static_cast<float>(nrm[i]) * static_cast<float>(cosv[j]));
      const InT t2 = static_cast<InT>(pf[i] * static_cast<float>(sinv[j]));
      r = static_cast<InT>(static_cast<float>(t1) - static_cast<float>(t2));
    } else if (j < RD) {
      const InT t1 = static_cast<InT>(static_cast<float>(nrm[i]) * static_cast<float>(cosv[j]));
      const InT t2 = static_cast<InT>(pf[i] * static_cast<float>(sinv[j]));
      r = static_cast<InT>(static_cast<float>(t1) + static_cast<float>(t2));
    } else {
      r = nrm[i];
    }
    dst[j] = r;
  }
