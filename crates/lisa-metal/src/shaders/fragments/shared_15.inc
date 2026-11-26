[[kernel]] void affine_qmm_t(
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

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

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
  qmm_t_impl<T, group_size, bits, aligned_N, T, BM, BK, BN>(
      w,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
