[[kernel]] void affine_gather_qmm_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device int32_t* offsets [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    const constant int& num_groups [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      transpose,
      BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(
          offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(
      wl + group * stride_w,
      scales + group * stride_s,
      biases + group * stride_s,
      transpose ? K : N,
      Ws,
      simd_group_id,
      simd_lane_id);

  // Tile aligned so check outside of the hot loop
  if (tgp_bm == BM && (align_N || tgp_bn == BN)) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(y, N);
  }

  // Tile partially aligned check rows
  else if (align_N || tgp_bn == BN) {
    gemm_loop_unaligned<false, true, transpose>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_safe(y, N, short2(BN, tgp_bm));
  }

  // Tile partially aligned check cols
  else if (tgp_bm == BM) {
    gemm_loop_unaligned<true, false, transpose>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_safe(y, N, short2(tgp_bn, BM));
  }

  // Nothing aligned so check both rows and cols
  else {
    gemm_loop_unaligned<false, false, transpose>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_safe(y, N, short2(tgp_bn, tgp_bm));
  }
}

template <typename T, int group_size, int bits, bool has_global_scale = false>
[[kernel]] void affine_quantize(
    const device T* w [[buffer(0)]],
    device uint8_t* out [[buffer(1)]],
    device T* scales [[buffer(2)]],
    device T* biases [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  constexpr float eps = 1e-7;
  constexpr int simd_size = 32;
  constexpr float n_bins = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int writes_per_reduce = pack_factor / values_per_reduce;
  constexpr int writes_per_pack =
      writes_per_reduce > 1 ? 1 : values_per_reduce / pack_factor;
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;

  static_assert(
      group_size % simd_size == 0,
      "Group size must be divisible by simd size.");

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t in_index = offset * values_per_reduce;
  size_t out_index = power_of_2_bits
      ? offset * writes_per_pack
      : offset * bytes_per_pack / writes_per_reduce;

  float w_thread[values_per_reduce];
  float w_min = Limits<T>::max;
