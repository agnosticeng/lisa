// Vendored for the native runtime (MIT/Apache-2.0, see NOTICE). MLX steel-gemm port (src/metal_src/mlx_gemm.metal, the kernels crate 0.11.0)
// MLX Kernel extracted from:
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm
// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

using namespace metal;

// https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/kernels/steel/gemm/params.h#L1
///////////////////////////////////////////////////////////////////////////////
// GEMM param classes
///////////////////////////////////////////////////////////////////////////////

struct GEMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldd;

  const int tiles_n;
  const int tiles_m;

  const size_t batch_stride_a;
  const size_t batch_stride_b;
  const size_t batch_stride_d;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;

  const int batch_ndim;
};

struct GEMMSpiltKParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  const int tiles_n;
  const int tiles_m;

  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;

  const int gemm_k_iterations_aligned;
};

struct GEMMAddMMParams {
  const int ldc;
  const int fdc;

  const size_t batch_stride_c;

  const float alpha;
  const float beta;
};

// https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/kernels/steel/gemm/loader.h#L1
///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoader {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = op.apply(dst[i * dst_ld + j]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
//@include fragments/shared_18.inc
  METAL_FUNC void next() {
    src += tile_stride;
  }
};

// https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/kernels/steel/gemm/transforms.h#L1
///////////////////////////////////////////////////////////////////////////////
// Transforms and Epilogues
///////////////////////////////////////////////////////////////////////////////

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT) {
    return static_cast<OutT>(x);
  }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(x * alpha + (beta * c));
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

// https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/kernels/steel/gemm/mma.h#L1
///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = 8 * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = 8 * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

  // Strides of A, B along reduction axis
  STEEL_CONST short simd_stride_a = {
      transpose_a ? TM_stride : TM_stride * lda_tgp};
  STEEL_CONST short simd_stride_b = {
      transpose_b ? TN_stride * ldb_tgp : TN_stride};

  // Jump between elements
  STEEL_CONST short jump_a = {transpose_a ? lda_tgp : 1};
  STEEL_CONST short jump_b = {transpose_b ? ldb_tgp : 1};

  STEEL_CONST short tile_stride_a = {transpose_a ? 8 * lda_tgp : 8};
  STEEL_CONST short tile_stride_b = {transpose_b ? 8 : 8 * ldb_tgp};

  // Simdgroup matrices
  simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
  simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
  simdgroup_matrix<AccumType, 8, 8> results[TM * TN] = {
      simdgroup_matrix<AccumType, 8, 8>(0)};

  // Offsets within threadgroup
  const short tm;
  const short tn;

  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : tm(8 * (simd_group_id / WN)), tn(8 * (simd_group_id % WN)) {
    // Determine thread position in simdgroup matrix
    short qid = simd_lane_id / 4;
    sm = (qid & 4) + (simd_lane_id / 2) % 4;
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

    // Determine thread and simdgroup offset
    As_offset =
        transpose_a ? ((sn)*lda_tgp + (tm + sm)) : ((sn) + (tm + sm) * lda_tgp);
    Bs_offset =
        transpose_b ? ((tn + sn) * ldb_tgp + (sm)) : ((sm)*ldb_tgp + (tn + sn));
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of 8
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += 8) {
      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup A as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        Asimd[i].thread_elements()[0] =
            static_cast<AccumType>(As[i * simd_stride_a + 0]);
        Asimd[i].thread_elements()[1] =
            static_cast<AccumType>(As[i * simd_stride_a + jump_a]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup B as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        Bsimd[j].thread_elements()[0] =
            static_cast<AccumType>(Bs[j * simd_stride_b + 0]);
        Bsimd[j].thread_elements()[1] =
            static_cast<AccumType>(Bs[j * simd_stride_b + jump_b]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Multiply and accumulate into result simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          short j_serp = (i % 2) ? (TN - 1 - j) : j;

          simdgroup_multiply_accumulate(
              results[i * TN + j_serp],
              Asimd[i],
              Bsimd[j_serp],
              results[i * TN + j_serp]);
        }
      }

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) const {
    // Adjust for simdgroup and thread location
    D += (sm + tm) * ldd + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {Epilogue::apply(accum[0]), Epilogue::apply(accum[1])};

        // Write out D
        D[offset] = outs[0];
        D[offset + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) const {
    // Adjust for simdgroup and thread location
    D += (sm + tm) * ldd + (tn + sn);
    dst_tile_dims -= short2(tn + sn, sm + tm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            D[offset] = Epilogue::apply(accum[0]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            D[offset + 1] = Epilogue::apply(accum[1]);
          }
        }
      }
    }
  }

  /* Apply epilogue */
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = results[i * TN + j].thread_elements();

        // Apply epilogue
        accum[0] = epilogue_op.apply(accum[0]);
        accum[1] = epilogue_op.apply(accum[1]);
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Apply epilogue
        accum[0] = epilogue_op.apply(accum[0], C[offset_c]);
        accum[1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    dst_tile_dims -= short2(tn + sn, sm + tm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Read C
        U c_elems[2] = {0};

        if ((j * TN_stride + 1) < dst_tile_dims.x) {
          c_elems[0] = C[offset_c];
          c_elems[1] = C[offset_c + fdc];
        } else if ((j * TN_stride) < dst_tile_dims.x) {
          c_elems[0] = C[offset_c];
        }

        // Apply epilogue
        accum[0] = epilogue_op.apply(accum[0], c_elems[0]);
        accum[1] = epilogue_op.apply(accum[1], c_elems[1]);
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {
            epilogue_op.apply(accum[0], C[offset_c]),
            epilogue_op.apply(accum[1], C[offset_c + fdc])};

        // Write out D
        D[offset_d] = outs[0];
        D[offset_d + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;
    dst_tile_dims -= short2(tn + sn, sm + tm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            D[offset_d] = epilogue_op.apply(accum[0], C[offset_c]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            D[offset_d + 1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
          }
        }
      }
    }
  }
};

// https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/kernels/steel/gemm/gemm.h#L1
///////////////////////////////////////////////////////////////////////////////
// GEMM kernel class
///////////////////////////////////////////////////////////////////////////////
//@include fragments/shared_3.inc
// utils.h
///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint elem,
    device const int* shape,
    device const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    stride_t elem,
    device const int* shape,
    device const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    stride_t elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = elem.x * strides[ndim - 1] + elem.y * strides[ndim - 2];
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}


METAL_FUNC ulong2 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
  }
  return ulong2(loc_a, loc_b);
}

METAL_FUNC ulong3 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  ulong loc_c{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
    loc_c += pos_in_dim * c_strides[i];
  }
  return ulong3(loc_a, loc_b, loc_c);
}


// https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L1
///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

constant bool do_gather [[function_constant(300)]];

constant bool gather_bias = do_gather && use_out_source;

// clang-format off
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2), function_constant(use_out_source)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
    const constant size_t* batch_strides [[buffer(7), function_constant(has_batch)]],
    const constant uint32_t* lhs_indices [[buffer(10), function_constant(do_gather)]],
    const constant uint32_t* rhs_indices [[buffer(11), function_constant(do_gather)]],
    const constant uint32_t* C_indices [[buffer(12), function_constant(gather_bias)]],
    const constant int* operand_shape [[buffer(13), function_constant(do_gather)]],
    const constant size_t* operand_strides [[buffer(14), function_constant(do_gather)]],
    const constant packed_int3& operand_batch_ndim [[buffer(15), function_constant(do_gather)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on
  // Pacifying compiler
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      true,
      true,
      AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  // Find block
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  // Exit early if out of bounds
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Adjust for batch

  // Handle gather
  if (do_gather) {
    // Read indices
    uint32_t indx_A, indx_B, indx_C;

    if (has_batch) {
      const constant size_t* indx_A_bstrides = batch_strides;
      const constant size_t* indx_B_bstrides =
          batch_strides + params->batch_ndim;

      ulong2 indx_offsets = elem_to_loc_broadcast(
          tid.z,
          batch_shape,
          indx_A_bstrides,
          indx_B_bstrides,
          params->batch_ndim);
      indx_A = lhs_indices[indx_offsets.x];
      indx_B = rhs_indices[indx_offsets.y];

      if (use_out_source) {
        const constant size_t* indx_C_bstrides =
            indx_B_bstrides + params->batch_ndim;
        auto indx_offset_C = elem_to_loc(
            tid.z, batch_shape, indx_C_bstrides, params->batch_ndim);
        indx_C = C_indices[indx_offset_C];
      }
    } else {
      indx_A = lhs_indices[params->batch_stride_a * tid.z];
      indx_B = rhs_indices[params->batch_stride_b * tid.z];

      if (use_out_source) {
        indx_C = C_indices[addmm_params->batch_stride_c * tid.z];
      }
    }

    // Translate indices to offsets
    int batch_ndim_A = operand_batch_ndim.x;
    const constant int* batch_shape_A = operand_shape;
    const constant size_t* batch_strides_A = operand_strides;
    A += elem_to_loc(indx_A, batch_shape_A, batch_strides_A, batch_ndim_A);

    int batch_ndim_B = operand_batch_ndim.y;
    const constant int* batch_shape_B = batch_shape_A + batch_ndim_A;
    const constant size_t* batch_strides_B = batch_strides_A + batch_ndim_A;
    B += elem_to_loc(indx_B, batch_shape_B, batch_strides_B, batch_ndim_B);

    if (use_out_source) {
      int batch_ndim_C = operand_batch_ndim.z;
      const constant int* batch_shape_C = batch_shape_B + batch_ndim_B;
      const constant size_t* batch_strides_C = batch_strides_B + batch_ndim_B;
      C += elem_to_loc(indx_C, batch_shape_C, batch_strides_C, batch_ndim_C);
    }

  }

  // Handle regular batch
  else {
    if (has_batch) {
      const constant size_t* A_bstrides = batch_strides;
      const constant size_t* B_bstrides = batch_strides + params->batch_ndim;

      ulong2 batch_offsets = elem_to_loc_broadcast(
          tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

      A += batch_offsets.x;
      B += batch_offsets.y;

      if (use_out_source) {
        const constant size_t* C_bstrides = B_bstrides + params->batch_ndim;
        C += elem_to_loc(tid.z, batch_shape, C_bstrides, params->batch_ndim);
      }
    } else {
      A += params->batch_stride_a * tid.z;
      B += params->batch_stride_b * tid.z;

      if (use_out_source) {
        C += addmm_params->batch_stride_c * tid.z;
      }
//@include fragments/shared_5.inc

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  template [[host_name("gemm_" #tname "_"  #iname "_" #oname "_" #bm "_" #bn "_" #bk "_" #wm "_" #wn)]] \
  [[kernel]] void gemm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, float>( \
      const device itype *A [[buffer(0)]], \
      const device itype *B [[buffer(1)]], \
      const device itype *C [[buffer(2), function_constant(use_out_source)]], \
      device itype *D [[buffer(3)]], \
      const constant GEMMParams* params [[buffer(4)]], \
      const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]], \
      const constant int* batch_shape [[buffer(6)]], \
      const constant size_t* batch_strides [[buffer(7)]], \
      const constant uint32_t* lhs_indices [[buffer(10), function_constant(do_gather)]], \
      const constant uint32_t* rhs_indices [[buffer(11), function_constant(do_gather)]], \
      const constant uint32_t* C_indices [[buffer(12), function_constant(gather_bias)]], \
      const constant int* operand_shape [[buffer(13), function_constant(do_gather)]], \
      const constant size_t* operand_strides [[buffer(14), function_constant(do_gather)]], \
      const constant packed_int3& operand_batch_ndim [[buffer(15), function_constant(do_gather)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

// ============================================================
// Tile Configuration: (32, 32, 16, 2, 2) - Original/fallback configuration
// ============================================================
instantiate_gemm_transpose_helper(f32, float, f32, float, 32, 32, 16, 2, 2)
instantiate_gemm_transpose_helper(f16, half, f16, half, 32, 32, 16, 2, 2)
#if defined(__HAVE_BFLOAT__)
instantiate_gemm_transpose_helper(bf16, bfloat, bf16, bfloat, 32, 32, 16, 2, 2)
#endif

// ============================================================
// Tile Configuration: (64, 64, 16, 2, 2) - Default for medium devices
// Reference: MLX steel_gemm_fused.metal
// ============================================================
instantiate_gemm_transpose_helper(f32, float, f32, float, 64, 64, 16, 2, 2)
instantiate_gemm_transpose_helper(f16, half, f16, half, 64, 64, 16, 2, 2)
#if defined(__HAVE_BFLOAT__)
instantiate_gemm_transpose_helper(bf16, bfloat, bf16, bfloat, 64, 64, 16, 2, 2)
#endif

// ============================================================
// Tile Configuration: (64, 64, 16, 1, 2) - For half/bfloat with small K
// Reference: MLX steel_gemm_fused.metal
// ============================================================
instantiate_gemm_transpose_helper(f32, float, f32, float, 64, 64, 16, 1, 2)
instantiate_gemm_transpose_helper(f16, half, f16, half, 64, 64, 16, 1, 2)
#if defined(__HAVE_BFLOAT__)
instantiate_gemm_transpose_helper(bf16, bfloat, bf16, bfloat, 64, 64, 16, 1, 2)
#endif

// ============================================================
// Tile Configuration: (64, 32, 32, 2, 2) - For nt mode with large K
// Reference: MLX steel_gemm_fused.metal
// ============================================================
instantiate_gemm_transpose_helper(f32, float, f32, float, 64, 32, 32, 2, 2)
instantiate_gemm_transpose_helper(f16, half, f16, half, 64, 32, 32, 2, 2)
#if defined(__HAVE_BFLOAT__)
instantiate_gemm_transpose_helper(bf16, bfloat, bf16, bfloat, 64, 32, 32, 2, 2)
#endif

// ============================================================
// Tile Configuration: (32, 64, 16, 1, 2) - For nn mode with large K
// Reference: MLX steel_gemm_fused.metal
// ============================================================
instantiate_gemm_transpose_helper(f32, float, f32, float, 32, 64, 16, 1, 2)
instantiate_gemm_transpose_helper(f16, half, f16, half, 32, 64, 16, 1, 2)
#if defined(__HAVE_BFLOAT__)
instantiate_gemm_transpose_helper(bf16, bfloat, bf16, bfloat, 32, 64, 16, 1, 2)
#endif

// LISA opt-in small-row F32 tiles. The MMA helper, BK=16 and K traversal are
// unchanged; only unused output-row tiles and SIMD groups are removed.
instantiate_gemm(nn, false, false, f32, float, f32, float, 8, 32, 16, 1, 2)
instantiate_gemm(nt, false, true,  f32, float, f32, float, 8, 32, 16, 1, 2)
