#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// NORME-DANS-QMV (specs/16 phase 1): the fused residual-add + RMSNorm moved
// INTO the following quantized GEMV as a prologue — one launch replaces
// (fused_add_rms + qmv).
//
// Bit-exactness rests on two pinned facts (see the dedicated test
// `qmv_addnorm_bitexact_vs_fused_chain`):
// 1. the prologue's reduction mirrors `fused_add_rms_looped` exactly: the
//    original launch ran `emu_threads` threads/threadgroup with N_READS=4;
//    lane j accumulates its 4-wide reads sequentially over ro rounds, then
//    per-32-lane hardware `simd_sum` butterflies fold each original
//    simdgroup, and a final hardware `simd_sum` over the 32 partials (the
//    original's second stage) gives lane 0's total -> precise::rsqrt.
//    Stage 1 is simulated scalar-wise with the double-buffered xor-butterfly
//    (all lanes read pre-step partner values — the hardware lockstep
//    semantics); stage 2 runs the REAL simd_sum over the same 32 values in
//    the same order, so the hardware cannot disagree with itself. The
//    original launch had emu = max_total_threads_per_threadgroup threads
//    (1024 on Apple silicon); this kernel runs the qmv threadgroup (64
//    threads), so each real thread stridedly plays several emulated lanes
//    while preserving every emulated lane's sequential accumulation order.
// 2. the main loop is not a copy but the REAL `qmv_fast_impl` from the MLX
//    quantized preamble, invoked verbatim on the normed row. A hand-copied
//    expression tree compiled inside a different kernel function produced
//    different FMA-contraction codegen (1-ulp y drift on a few % of outputs
//    — measured); calling the actual stock function removes that divergence
//    class entirely. The prologue therefore stages `sum` and `normed` to
//    device scratch (the same buffers the replaced two-kernel chain wrote),
//    barriers on device memory, and re-reads its own threadgroup's row —
//    freshly written, hence L2-resident.
// TAIL-ULP variant (specs/16 §7, campaign-close phase 1 under the §9.7
// tail-ULP contract — same class of relaxation as the verify split-K port):
// the §5.2 in-situ falsification was specific to the BIT-EXACT prologue. Its
// cost was the 1024-lane strided emulation (every threadgroup replaying the
// looped-rms lane program) plus the device scratch staging + device barrier.
// Both exist only to preserve `fused_add_rms_looped`'s exact association;
// under tail-ULP (<= 1 bf16 ULP + argmax equality vs the stock chain, pinned
// by `qmv_addnorm_tailulp_vs_fused_chain`) they go away:
//
// - each 64-thread qmv threadgroup computes the sum-of-squares with a plain
//   strided fp32 accumulation + hardware simd_sum (2 simdgroups) — one pass;
// - the s = bf16(x + r) row lives in THREADGROUP memory (10 KB at k=5120) and
//   is normalized in place; the GEMV loop reads it from TG memory (a local
//   `load_vector` twin over the threadgroup address space, bits==4 path).
//   No device scratch, no device barrier.
//
// DETERMINISM: every threadgroup of a row executes the identical instruction
// sequence on identical data, so all of them derive the bit-identical `inv`
// and `normed` — the tid.y==0 threadgroup is the only device writer of
// `sum_out`/`norm_out` (no write amplification, no race semantics needed).
// Goldens therefore stay fully meaningful (no nondeterminism is introduced).

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_tg_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const threadgroup T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid);

template <typename T, int group_size, int bits, int write_norm>
[[kernel]] void affine_qmv_fast_addnorm_tu(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device T* r [[buffer(4)]],
    const device T* nw [[buffer(5)]],
    device T* sum_out [[buffer(6)]],
    device T* norm_out [[buffer(7)]],
    device T* y [[buffer(8)]],
    const constant float& eps [[buffer(9)]],
    const constant int& in_vec_size [[buffer(10)]],
    const constant int& out_vec_size [[buffer(11)]],
    const constant uint& axis_size [[buffer(12)]],
    const constant uint& nw_stride [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int MAXK = 5120;
  static_assert(bits == 4, "tail-ULP addnorm: only the 4-bit GEMV is pinned");
  if (axis_size > MAXK) {
    // Guarded by the Rust wrapper; unreachable in practice.
    return;
  }
  const uint lid = simd_gid * 32u + simd_lid;
  const uint base = tid.x * axis_size;

  threadgroup T tg_x[MAXK];
  threadgroup float tg_lane[1024];
  threadgroup float tg_groups[32];
  threadgroup float tg_inv[1];

  // ---- prologue: s = bf16(x + r) + the stock looped-rms reduction ----
  // IN SITU LESSON (specs/16 §7.1): modeling the hardware simd_sum as a
  // software xor-butterfly is WRONG — the real hardware tree has its own
  // association, invisible on narrow-range random draws but ~3e-5 relative
  // on real hidden states (heavy-tailed s^2), which flips ~1% of normed
  // bf16 roundings and is amplified by the GEMV into tens of ULP on y
  // (golden 210/310). The stock association is therefore reproduced with
  // REAL hardware reductions in the SAME operand order instead of any
  // model: 1024 emulated lanes mapped 16-consecutive-per-real-lane, so each
  // emulated 32-lane group is reduced by one hardware simd_sum over the
  // same 32 partials in lane order, then the stock second hardware simd_sum
  // over the 32 group totals. No device staging, TG-resident row.
  {
    const device T* xs = x + base;
    const device T* rs = r + base;
    // real lane `lid` plays emulated lanes 16*lid .. 16*lid+15, in order.
    for (uint l0 = lid * 16u; l0 < (lid + 1u) * 16u; l0++) {
      float acc = 0;
      for (uint ro = 0; ro < axis_size; ro += 4096u) {
        uint b4 = ro + l0 * 4u;
        for (int i = 0; i < 4; i++) {
          if (b4 + i < axis_size) {
            T s = static_cast<T>(
                static_cast<float>(xs[b4 + i]) + static_cast<float>(rs[b4 + i]));
            tg_x[b4 + i] = s;
            acc += static_cast<float>(s) * static_cast<float>(s);
          }
        }
      }
      tg_lane[l0] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Stage 1: each emulated 32-lane group reduced by the REAL hardware
  // simd_sum over its 32 partials in lane order (stock line for line).
  if (simd_gid == 0) {
    for (uint g = 0; g < 32u; g++) {
      float t = simd_sum(tg_lane[g * 32u + simd_lid]);
      if (simd_lid == 0) {
        tg_groups[g] = t;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Stage 2: the stock second stage — a real hardware simd_sum over the 32
  // group partials (simdgroup 0).
  if (simd_gid == 0) {
    float total = simd_sum(tg_groups[simd_lid]);
    if (simd_lid == 0) {
      tg_inv[0] = metal::precise::rsqrt(
          total / static_cast<float>(axis_size) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = tg_inv[0];

  // ---- normalize in place: nd = bf16(w * bf16(f32(s) * inv)) ----
  // Same rounding chain as fused_add_rms_looped's write-out.
  {
    const device T* nwp = nw;
    for (uint j = lid; j < axis_size; j += 64u) {
      tg_x[j] = static_cast<T>(
          static_cast<float>(nwp[nw_stride * j]) *
          static_cast<float>(
              static_cast<T>(static_cast<float>(tg_x[j]) * inv)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- device outputs: written by the tid.y==0 threadgroup ONLY ----
  // (Every threadgroup holds bit-identical values; one writer keeps the
  // device traffic at one row, not n/8 rows.)
  if (tid.y == 0) {
    const device T* xs = x + base;
    const device T* rs = r + base;
    for (uint j = lid; j < axis_size; j += 64u) {
      T s = static_cast<T>(static_cast<float>(xs[j]) + static_cast<float>(rs[j]));
      sum_out[base + j] = s;
      if (write_norm == 1) {
        norm_out[base + j] = static_cast<T>(
            static_cast<float>(nw[nw_stride * j]) *
            static_cast<float>(
                static_cast<T>(static_cast<float>(s) * inv)));
      }
    }
  }

  // ---- main loop: qmv_fast over the threadgroup-resident normed row ----
  qmv_fast_tg_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      tg_x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

// `load_vector` over the threadgroup address space — the bits==4 branch of
// the MLX helper (quantized.metal), pointer type swapped. Values are
// identical to what the stock kernel reads from its device-resident normed
// row, so x_thread and the per-thread partial `sum` are identical too.
template <typename T, typename U, int values_per_thread>
inline U load_vector_tg(const threadgroup T* x, thread U* x_thread) {
  U sum = 0;
  for (int i = 0; i < values_per_thread; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

// qmv_fast_impl (quantized.metal) with the activation vector in threadgroup
// memory. Expression trees, pack assignment and reduction order are the
// stock ones; the only divergence vs the two-kernel stock chain is the
// normed row itself (<= 1 bf16 ULP — tail-ULP contract).
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_tg_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const threadgroup T* x,
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

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector_tg<T, U, values_per_thread>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}


template <typename T, int group_size, int bits, int write_norm>
[[kernel]] void affine_qmv_fast_addnorm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device T* r [[buffer(4)]],
    const device T* nw [[buffer(5)]],
    device T* sum_out [[buffer(6)]],
    device T* norm_out [[buffer(7)]],
    device T* y [[buffer(8)]],
    const constant float& eps [[buffer(9)]],
    const constant int& in_vec_size [[buffer(10)]],
    const constant int& out_vec_size [[buffer(11)]],
    const constant uint& axis_size [[buffer(12)]],
    const constant uint& nw_stride [[buffer(13)]],
    const constant uint& emu_threads [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int MAX_EMU = 1024;
  const uint emu = emu_threads;
  const uint lid = simd_gid * 32u + simd_lid;

  // ---------- prologue: inv_mean, bit-for-bit (see header note 1) ----------
  threadgroup float tg_acc[MAX_EMU];
  threadgroup float tg_groups[MAX_EMU / 32];
  threadgroup float tg_inv[1];

  if (lid < MAX_EMU) {
    tg_acc[lid] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = lid; j < emu; j += 64u) {
    float acc = 0;
    // fused_add_rms_looped: lane j reads x[j*4 + i + ro], ro stepping
    // emu*4, four sequential adds per round into one f32 accumulator.
    for (uint ro = 0; ro < axis_size; ro += emu * 4u) {
      uint base = ro + j * 4u;
      for (int i = 0; i < 4; i++) {
        if (base + i < axis_size) {
          float s = static_cast<float>(x[base + i]) +
              static_cast<float>(r[base + i]);
          acc += static_cast<float>(static_cast<T>(s)) *
              static_cast<float>(static_cast<T>(s));
        }
      }
    }
    tg_acc[j] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Stage 1: per-original-simdgroup partials. The original ran
  // simd_sum(acc) over each group's 32 lanes; every lane ends at the total,
  // lane 0's is the one consumed downstream. Simulate the double-buffered
  // xor-butterfly for lane 0's association (hardware: all lanes read the
  // pre-step partner value in lockstep).
  if (lid < MAX_EMU / 32) {
    tg_groups[lid] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < emu / 32u) {
    float v[MAX_EMU / 32];
    for (int l = 0; l < 32; l++) {
      v[l] = tg_acc[lid * 32u + l];
    }
    for (int off = 16; off >= 1; off >>= 1) {
      float vo[MAX_EMU / 32];
      for (int l = 0; l < 32; l++) {
        vo[l] = v[l ^ off];
      }
      for (int l = 0; l < 32; l++) {
        v[l] = v[l] + vo[l];
      }
    }
    tg_groups[lid] = v[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Stage 2: the ORIGINAL second stage was a real hardware simd_sum over
  // the 32 group partials (simd_group 0, lane 0 keeps the result). Run the
  // real simd_sum over the same 32 values in the same order.
  if (simd_gid == 0) {
    float total = simd_sum(tg_groups[simd_lid]);
    if (simd_lid == 0) {
      tg_inv[0] = metal::precise::rsqrt(total / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = tg_inv[0];

  // ---- stage the sum + normed row (the rms kernel's outputs, note 2) ----
  // Same rounding chain as fused_add_rms_looped's write-out: s = bf16(x+r);
  // sum = s; normed = bf16(w * bf16(f32(s) * inv)).
  {
    const device T* xs = x + tid.x * axis_size;
    const device T* rs = r + tid.x * axis_size;
    device T* sd = sum_out + tid.x * axis_size;
    device T* nd = norm_out + tid.x * axis_size;
    for (uint ro = 0; ro < axis_size; ro += 64u * 4u) {
      uint base = ro + lid * 4u;
      for (int i = 0; i < 4; i++) {
        if (base + i < axis_size) {
          T s = static_cast<T>(
              static_cast<float>(xs[base + i]) +
              static_cast<float>(rs[base + i]));
          sd[base + i] = s;
          nd[base + i] = static_cast<T>(
              static_cast<float>(nw[nw_stride * (base + i)]) *
              static_cast<float>(
                  static_cast<T>(static_cast<float>(s) * inv)));
        }
      }
    }
  }
  // Device-memory writes above must be visible to this threadgroup's
  // re-read inside qmv_fast_impl.
  threadgroup_barrier(mem_flags::mem_device);

  // ---------- main loop: the REAL stock qmv_fast_impl, verbatim ----------
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      norm_out + tid.x * axis_size,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}
