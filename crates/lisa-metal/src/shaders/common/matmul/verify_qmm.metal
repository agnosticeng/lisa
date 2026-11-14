// Verify-width quantized matmul: one weight stream serves all M input rows.
//
// Port context: the `verifyQmm` lane family — a split-K verify tile and a
// wide-N msg tile. Which lane is taken is decided by the verify width and the
// vocab geometry, not by a fixed table. The
// tile wins by paying the weight DRAM stream ONCE for the S=2..8-row
// speculative-verify forward instead of once per row.
//
// This variant keeps MLX `qmv_fast_impl` (quantized.metal) NUMERICS
// bit-identically — same per-thread pack assignment, same qdot/load_vector
// expression trees (qdot_reg below is qdot with the weight pointer moved to
// thread storage; identical masks, identical fp order), same simd_sum
// reduction — and changes ONLY the tiling: tid.x (the qmv_fast input-row
// axis) is collapsed to a single threadgroup column, the M input rows ride
// an unrolled inner loop, and each thread's weight words for the block are
// loaded ONCE into thread storage before the row loop (the MTPLX-ledger
// rule: hoist the BYTES, not the pointer — a device-pointer re-read per row
// measured LSU-bound and lost to the per-row qmv chain). The result per
// output element is the exact same fp32 operation sequence as the M=1
// qmv_fast kernel, so outputs are bit-identical to the per-row chain — the
// contract the fused-kernel-wave tests pin. Dispatch gates: n % 8 == 0 and
// K aligned to qmv_fast_k_alignment(bits), i.e. exactly where qmv_fast is
// the M=1 reference.

template <typename U, int values_per_thread, int bits>
inline U qdot_reg(
    const thread uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (x_thread[4 * i] * (w[i] & 0x03) +
                x_thread[4 * i + 1] * (w[i] & 0x0c) +
                x_thread[4 * i + 2] * (w[i] & 0x30) +
                x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (x_thread[4 * i] * (ws[i] & 0x000f) +
                x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename T, int group_size, int bits, int mrows>
[[kernel]] void affine_verify_qmm(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  // Per-thread register tiles. mrows is a template constant and every loop
  // over it is fully unrolled, so these stay register-promoted (the
  // MTPLX-ledger stack-spill trap hits array-indexed rows behind a
  // runtime loop; here all indices are literal after unrolling).
  thread U x_thread[mrows][values_per_thread];
  thread U result[mrows][results_per_simdgroup];
  // The block's weight words, hoisted once (BYTES, not pointers).
  thread uint8_t wcopy[results_per_simdgroup][packs_per_thread * bytes_per_pack];

  // Adjust positions — identical to qmv_fast_impl except tid.x is always 0
  // and the input row is the inner loop variable.
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  _Pragma("unroll")
  for (int r = 0; r < mrows; ++r) {
    _Pragma("unroll")
    for (int q = 0; q < results_per_simdgroup; ++q) {
      result[r][q] = 0;
    }
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    _Pragma("unroll")
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      _Pragma("unroll")
      for (int p = 0; p < packs_per_thread * bytes_per_pack; ++p) {
        wcopy[row][p] = wl[p];
      }
    }

    _Pragma("unroll")
    for (int r = 0; r < mrows; ++r) {
      U sum = load_vector<T, U, values_per_thread, bits>(
          x + r * in_vec_size, x_thread[r]);

      _Pragma("unroll")
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[r][row] += qdot_reg<U, values_per_thread, bits>(
            wcopy[row], x_thread[r], s, b, sum);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  _Pragma("unroll")
  for (int r = 0; r < mrows; ++r) {
    _Pragma("unroll")
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[r][row] = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[r * out_vec_size + row] = static_cast<T>(result[r][row]);
      }
    }
  }
}

// Split-K verify tile — the splitk lane, for narrow verify widths.
// This is the variant lisa actually ships: the
// bit-exact tile above loses in situ because its full-K per-thread chains
// serialize the mixed pipeline; K_PARTS exists precisely to fix that ("deep
// occupancy queues + latency hiding under mixed scheduling with attention/
// GDN kernels"). Ported line-for-line from their comptime emission:
//
//   - one threadgroup owns BN output columns x all MROWS input rows;
//   - the K reduction splits across K_PARTS simdgroups of the SAME
//     threadgroup (threadgroup = (32*K_PARTS, 1, 1), grid = (32*K_PARTS,
//     N/BN)): simdgroup `part` covers packs [part*per_part, p_end) strided
//     by 32 lanes, per_part = (K/8)/K_PARTS, the last part runs to K/8;
//   - per 8-K pack: every row's Vec8 activation loaded once, every owned
//     column's weight WORD hoisted once (4-bit: one aligned uint32 at
//     w_q[(n0+j)*K/8 + pack]), then ONE sequential dequant+FMA chain per
//     column (wv = float(nibble)*s + b; acc += float(v[ki]) * wv — the
//     interleaved form measurably loses, their ledger's rule);
//   - fp32 accumulation throughout, simd_sum per part, then THEIR EXACT
//     partial reduction: lane 0 of each part writes its NACC sums to
//     threadgroup partials, one threadgroup barrier, part 0's lanes sum the
//     K_PARTS partials IN PART ORDER (p = 0..K_PARTS-1, deterministic) and
//     write y[row*N + n0+j] = T(total).
//
// PPT (packs per thread) — OUR extension, NOT in their shipped body (their
// loop is 1 pack/thread/iteration, transformer.zig:596 emission): each thread
// takes PPT packs per iteration — pack, pack+32, ..., pack+32*(PPT-1) — with
// all PPT weight WORDS per owned column hoisted above the dequant+FMA
// chains. The lane->pack assignment is IDENTICAL to PPT=1 and each thread
// still accumulates its packs in ascending order, so the fp32 sum order per
// output element is unchanged: PPT>1 is BIT-IDENTICAL to PPT=1 (stronger
// than the tail-ULP contract). Rationale: halves the weight-word load
// instructions per iteration slot and pairs two independent device loads
// with the FMA chains (latency hiding on the DRAM stream).
//
// Numerics: fp32 summed in a DIFFERENT order than qmv_wide (per-part chains
// + part-ordered reduction) — bf16 tail-ULP class, exactly their accepted
// class ("bf16 tail-ULP class differences ... parity pinned by the verifyQmm
// test"). NOT bit-exact; the tail-ULP contract (maxdiff <= 1 bf16 ULP +
// argmax equality on the verify shapes, MTP verify rows only) is pinned by
// ops::array_ops::tests::verify_qmm_splitk_tailulp_vs_qmv_wide.
// Requirements (enforced by the Rust gate, mirroring vqmmLaneForTile):
// 2 <= MROWS <= 7, N % BN == 0, N >= 512, N < 100000 (huge-N is their msg
// lane — not ported), K % 64 == 0, K % 8 == 0 (pack geometry), and
// (K/8) % K_PARTS == 0. 4-bit only (the shipping trunk class; their
// mixed-plain adoption is per-shape measured and stays off here).
//   - VL4 (VLOAD=1) — OUR tail-ULP extension, NOT in their shipped body:
//     each thread takes 4 CONSECUTIVE packs per iteration loaded with ONE
//     16-byte uint4 per owned column (lane->pack map lane*4+it, stride 128
//     packs) instead of 4 separate 4-byte words. Requires per_part % 4 == 0
//     so every part stays 16B-aligned (K % 64 == 0 already aligns row
//     strides). fp32 per-thread accumulation stays in ascending pack order
//     per output element (a DIFFERENT association than VLOAD=0 — bf16
//     tail-ULP class, pinned by the VLOAD tail-ULP test). Falsified-family
//     note: this is NOT the PPT=2 falsified shape (PPT keeps 4-byte loads
//     and the lane+32*it map); VL4 changes the LOAD WIDTH, halving weight
//     load instructions and widening every memory transaction.
template <typename T, int GS, int BITS, int MROWS, int BN, int K_PARTS, int PPT, int VLOAD>
[[kernel]] void affine_verify_qmm_splitk(
    const device uint32_t* w_q,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& K_size,
    const constant int& N_size,
    uint2 tg_n [[threadgroup_position_in_grid]],
    uint part [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NACC = BN * MROWS;
  constexpr int PP = VLOAD ? 4 : PPT;

  int K = int(K_size);
  int N = int(N_size);
  int K_by_p = K / 8;
  int K_bytes = (K * BITS) / 8;
  int K_by_gs = K / GS;
  int per_part = K_by_p / K_PARTS;
  int n0 = int(tg_n.y) * BN;
  int p_start = int(part) * per_part;
  int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;

  float acc[NACC];
  _Pragma("unroll")
  for (int i = 0; i < NACC; ++i) {
    acc[i] = 0.0f;
  }

  using Vec8 = vec<T, 8>;
  const device Vec8* xv = (const device Vec8*)x;

  for (int pack0 = VLOAD ? (p_start + int(lane) * 4) : (p_start + int(lane));
       pack0 < p_end;
       pack0 += 32 * PP) {
    // Weight WORDS per owned column, hoisted above the chains (each
    // chain then reads thread registers, not the device stream).
    // VLOAD: ONE uint4 (16B, 4 consecutive packs) per column per iteration.
    uint32_t pw[PP][BN];
    if constexpr (VLOAD) {
      _Pragma("unroll")
      for (int j = 0; j < BN; ++j) {
        uint4 w4 = *(const device uint4*)(w_q + (n0 + j) * K_by_p + pack0);
        pw[0][j] = w4.x;
        pw[1][j] = w4.y;
        pw[2][j] = w4.z;
        pw[3][j] = w4.w;
      }
    } else {
      _Pragma("unroll")
      for (int it = 0; it < PP; ++it) {
        int pk = pack0 + 32 * it;
        if (pk < p_end) {
          _Pragma("unroll")
          for (int j = 0; j < BN; ++j) {
            pw[it][j] = w_q[(n0 + j) * K_by_p + pk];
          }
        }
      }
    }

    _Pragma("unroll")
    for (int it = 0; it < PP; ++it) {
      int pk = pack0 + (VLOAD ? it : 32 * it);
      if (pk < p_end) {
        int k_base = pk * 8;
        int gi = k_base / GS;

        // Row activation loads, all up front.
        T v[MROWS][8];
        _Pragma("unroll")
        for (int r = 0; r < MROWS; ++r) {
          Vec8 vr = xv[(r * K + k_base) / 8];
          _Pragma("unroll")
          for (int ki = 0; ki < 8; ++ki) {
            v[r][ki] = vr[ki];
          }
        }

        float s[BN];
        float b[BN];
        _Pragma("unroll")
        for (int j = 0; j < BN; ++j) {
          s[j] = float(scales[(n0 + j) * K_by_gs + gi]);
          b[j] = float(biases[(n0 + j) * K_by_gs + gi]);
        }

        // One sequential dequant+FMA chain per output column.
        _Pragma("unroll")
        for (int j = 0; j < BN; ++j) {
          float sj = s[j];
          float bj = b[j];
          if constexpr (BITS == 4) {
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {
              float wv = float((pw[it][j] >> (ki * 4)) & 0xFu) * sj + bj;
              _Pragma("unroll")
              for (int r = 0; r < MROWS; ++r) {
                acc[j * MROWS + r] += float(v[r][ki]) * wv;
              }
            }
          }
        }
      }
    }
  }

  _Pragma("unroll")
  for (int i = 0; i < NACC; ++i) {
    acc[i] = simd_sum(acc[i]);
  }

  // Their exact partial reduction: per-part lane-0 spill, one barrier,
  // part-ordered sum by part 0.
  threadgroup float partials[K_PARTS * NACC];
  if (lane == 0) {
    _Pragma("unroll")
    for (int i = 0; i < NACC; ++i) {
      partials[int(part) * NACC + i] = acc[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (part == 0) {
    for (int i = int(lane); i < NACC; i += 32) {
      float total = 0.0f;
      _Pragma("unroll")
      for (int p2 = 0; p2 < K_PARTS; ++p2) {
        total += partials[p2 * NACC + i];
      }
      int j = i / MROWS;
      int row = i - j * MROWS;
      y[row * N + n0 + j] = static_cast<T>(total);
    }
  }
}

// Msg verify tile — the huge-N lane, for wide vocabularies where the msg pass
// amortises the read better than split-K. This is the lane `vqmmLaneForTile`
// (transformer.zig:1359) routes to at N >= 100 000 — the lm_head class
// (N = 151936 on qwen3_5) — because the tiny-tile split-K grid thrashes the
// scheduler there (their measurement: 2.1x stock at M=4). Their design:
//
//   - NSG simdgroups per threadgroup, each simdgroup owning BN output
//     columns x all MROWS input rows (threadgroup = (32*NSG, 1, 1) threads,
//     grid = (1, ceil(N / (BN*NSG)))); the in-kernel n0 guard covers the
//     N-tail threadgroup, so host geometry stays a plain ceil-div;
//   - full-K strided loop `for pack = lane; pack < K/8; pack += 32` — the
//     same weight-word hoist + one sequential dequant+FMA chain per column
//     as the splitk tile (their generated body is line-identical there);
//   - reduction: one simd_sum per accumulator over the full K (NO
//     threadgroup partials, NO barrier — simdgroups are independent);
//   - write: lanes < BN*MROWS map i -> (j = i/MROWS, row = i-j*MROWS).
//
// Numerics: fp32 accumulated in a DIFFERENT order than qmv_wide (strided
// lanes over full K + one 32-lane simd_sum) — bf16 tail-ULP class, the same
// accepted class as the splitk tile, pinned against qmv_wide by
// ops::array_ops::tests::verify_qmm_splitk_tailulp_vs_qmv_wide (which also
// covers the head shape). Requirements (enforced by the Rust gate, mirroring
// their msg arm): 2 <= MROWS <= 7, N >= 100000, K % 64 == 0, N % BN == 0,
// 4-bit only. BN per their vqmmMsgBn: 4 through M=6, 2 at M=7 ("14
// accumulators — under the 24 ceiling"). NOTE the ledger lesson from our
// splitk port (specs/15 §2): their BN=4 stack-spilled on OUR Metal compiler
// at M=5 in the splitk tile — the msg tile carries no threadgroup partials
// array, so BN=4 is kept per their design but the isolated bench must
// confirm no spill before in-situ adoption.
template <typename T, int GS, int BITS, int MROWS, int BN, int NSG>
[[kernel]] void affine_verify_qmm_msg(
    const device uint32_t* w_q,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& K_size,
    const constant int& N_size,
    uint2 tg_n [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NACC = BN * MROWS;

  int K = int(K_size);
  int N = int(N_size);
  int K_by_p = K / 8;
  int K_by_gs = K / GS;
  int n0 = (int(tg_n.y) * NSG + int(sg)) * BN;
  if (n0 + BN - 1 >= N) {
    return;
  }

  float acc[NACC];
  _Pragma("unroll")
  for (int i = 0; i < NACC; ++i) {
    acc[i] = 0.0f;
  }

  using Vec8 = vec<T, 8>;
  const device Vec8* xv = (const device Vec8*)x;

  for (int pack = int(lane); pack < K_by_p; pack += 32) {
    int k_base = pack * 8;
    int gi = k_base / GS;

    // Row activation loads, all up front.
    T v[MROWS][8];
    _Pragma("unroll")
    for (int r = 0; r < MROWS; ++r) {
      Vec8 vr = xv[(r * K + k_base) / 8];
      _Pragma("unroll")
      for (int ki = 0; ki < 8; ++ki) {
        v[r][ki] = vr[ki];
      }
    }

    // Weight WORDS for the owned columns, hoisted above the FMA chains.
    uint32_t p[BN];
    _Pragma("unroll")
    for (int j = 0; j < BN; ++j) {
      p[j] = w_q[(n0 + j) * K_by_p + pack];
    }

    float s[BN];
    float b[BN];
    _Pragma("unroll")
    for (int j = 0; j < BN; ++j) {
      s[j] = float(scales[(n0 + j) * K_by_gs + gi]);
      b[j] = float(biases[(n0 + j) * K_by_gs + gi]);
    }

    // One sequential dequant+FMA chain per output column.
    _Pragma("unroll")
    for (int j = 0; j < BN; ++j) {
      float sj = s[j];
      float bj = b[j];
      if constexpr (BITS == 4) {
        _Pragma("unroll")
        for (int ki = 0; ki < 8; ++ki) {
          float wv = float((p[j] >> (ki * 4)) & 0xFu) * sj + bj;
          _Pragma("unroll")
          for (int r = 0; r < MROWS; ++r) {
            acc[j * MROWS + r] += float(v[r][ki]) * wv;
          }
        }
      }
    }
  }

  _Pragma("unroll")
  for (int i = 0; i < NACC; ++i) {
    acc[i] = simd_sum(acc[i]);
  }

  if (lane < NACC) {
    int j = int(lane) / MROWS;
    int row = int(lane) - j * MROWS;
    y[row * N + n0 + j] = static_cast<T>(acc[int(lane)]);
  }
}
