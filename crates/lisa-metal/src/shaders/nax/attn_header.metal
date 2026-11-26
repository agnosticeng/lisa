// Flattened MLX NAX attention header closure (AOT for `sdpa_full_nax`).
// Auto-generated flattened MLX Metal header closure. Do not hand-edit;
// regenerate from the pinned MLX headers.
#line 1 "mlx/backend/metal/kernels/bf16.h"


#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}
//@include fragments/bf16_math_1.inc
//@include fragments/complex_1.inc
#line 1 "mlx/backend/metal/kernels/defines.h"


#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#line 1 "mlx/backend/metal/kernels/logging.h"


#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>

namespace mlx {
using os_log = metal::os_log;
} // namespace mlx

#else

namespace mlx {
struct os_log {
  constexpr os_log(constant char*, constant char*) constant {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const thread {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const constant {}
};
} // namespace mlx

#endif
//@include fragments/utils_5.inc
#line 1 "mlx/backend/metal/kernels/steel/defines.h"


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
//@include fragments/type_traits_1.inc
//@include fragments/integral_constant_1.inc
#line 1 "mlx/backend/metal/kernels/steel/attn/nax.h"

//@include fragments/bf16_2.inc
#line 1 "mlx/backend/metal/kernels/steel/attn/params.h"

//@include fragments/bf16_8.inc
#line 1 "mlx/backend/metal/kernels/steel/utils.h"

//@include fragments/defines_1.inc
#line 1 "mlx/backend/metal/kernels/steel/attn/transforms.h"


//@include fragments/bf16_6.inc

#line 1 "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"


using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
//@include fragments/bf16_10.inc
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  using otile_t = NAXTile<AccumType, TQ, TD>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(params->Q_strides[2]);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
//@include fragments/bf16_4.inc
    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
//@include fragments/bf16_3.inc
      for (short id = 0; id < TD; id += 2) {
        if constexpr (BD == 128) {
//@include fragments/bf16_7.inc
  // Store results
  O += tm * int(params->O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Head-dim split attention kernel
///////////////////////////////////////////////////////////////////////////////

// Variant of attention_nax for wide heads (bd = 256). There, the per-simdgroup
// accumulator working set of attention_nax (TD output fragments plus the S
// fragments) is what gates tensor-unit throughput, so this kernel splits the
// head dim across the WN = 2 simdgroups of the second warp dimension: each
// simdgroup of a pair owns one half of D for Q@K.T and one half of Dv for P@V,
// halving its accumulator set. The pair exchanges its partial Q@K.T sums
// through threadgroup memory, then both simdgroups run softmax redundantly on
// the full S tile (the row statistics are cheap) and each accumulates P@V for
// its own half of Dv.

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax_dsplit(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  // The WM simdgroups along the first warp dimension split the Q sequence;
  // the WN simdgroups along the second split the head dim. The exchange
  // below reduces exactly one peer, so WN is fixed at 2.
  static_assert(WN == 2, "The head-dim split kernel needs WN == 2");
  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags over the full head dim
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(TD % WN == 0, "The head dim must split evenly across WN");

  // HeadDim frags / columns owned by each of the WN simdgroups of a row group
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;

  static_assert(TDh % 2 == 0, "P@V accumulates output fragments in pairs");
  static_assert(TK % 2 == 0, "S fragments are exchanged pair by pair");

  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;

  using otile_t = NAXTile<AccumType, TQ, TDh>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * row_group;
  Q += tm * int(params->Q_strides[2]) + d_half * BDh;
  K += d_half * BDh;
  V += d_half * BDh;
  O += tm * int(params->O_strides[2]) + d_half * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_q = int(tid.x) == (params->NQ_aligned);
  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  using stile_t = NAXTile<AccumType, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;

  // One slot per (row group, half): a fragment pair in per-lane-linear
  // layout. Both halves share the fragment-to-lane mapping, so the
  // exchange needs no coordinate math.
  threadgroup AccumType s_xchg[WM][WN][2 * kEPF * 32];

  // Keep the simdgroup's Q half resident in registers for the whole KV
  // loop: TDh fragments of T are cheap next to the accumulators.
  NAXTile<T, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) {
    const int Q_load_off = id * kU;
    if (!align_Q && is_last_q) {
      Qtiles[id].load_rows(
          Q + Q_load_off, int(params->Q_strides[2]), lim_rows_q);
    } else {
      Qtiles[id].load(Q + Q_load_off, int(params->Q_strides[2]));
    }
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    stile_t Stile;
    Stile.clear();

    // S = Q @ K.T, this half of D only, exchanged pair by pair.
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Ktile.load_rows(
              K + K_load_off, int(params->K_strides[2]), lim_rows_k - ik * kU);
        } else {
          Ktile.load(K + K_load_off, int(params->K_strides[2]));
        }

        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }

      // Exchange the partial pair and reduce.
      threadgroup AccumType* slot = s_xchg[row_group][d_half];
      thread auto& s0 = Stile.frag_at(0, ik);
      thread auto& s1 = Stile.frag_at(0, ik + 1);
      const short base = short(simd_lane_id) * (2 * kEPF);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        slot[base + i] = s0[i];
        slot[base + kEPF + i] = s1[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup AccumType* peer = s_xchg[row_group][1 - d_half];
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        s0[i] += peer[base + i];
        s1[i] += peer[base + kEPF + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + ii * stile_t::kFragRowsJump + sm;
            const auto c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (r < c) ? neg_inf : fg[loc];
//@include fragments/bf16_5.inc
    // Do softmax (redundantly per half; the row statistics are cheap)
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O = P @ V, this half of Dv only.
//@include fragments/bf16_9.inc
  Otile.template row_bin_op<MulOp>(rcp);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;
    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}

