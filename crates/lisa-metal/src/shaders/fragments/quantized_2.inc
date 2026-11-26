      (const device uint32_t*)wl,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    bool batched,
    bool has_global_scale = false,
    int BM = 32,
    int BK = 32,
    int BN = 32>
[[kernel]] void affine_qmm_n(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool has_global_scale = false>
[[kernel]] void affine_gather_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool has_global_scale = false>
[[kernel]] void affine_gather_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmv_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool has_global_scale = false>
[[kernel]] void affine_gather_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      out_vec_size * M,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      in_vec_size,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
