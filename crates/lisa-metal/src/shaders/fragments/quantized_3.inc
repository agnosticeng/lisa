    bool has_global_scale = false,
    int results_per_simdgroup = 4>
[[kernel]] void affine_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
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

template <
    typename T,
    int group_size,
    int bits,
    int vecs_per_tg,
    int k_lanes,
    bool batched>
[[kernel]] void affine_qmv_wide(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
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
  qmv_wide_impl<T, group_size, bits, vecs_per_tg, k_lanes>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size,
      out_vec_size,
      M,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    bool batched,
    bool has_global_scale = false>
[[kernel]] void affine_qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        out_vec_size * M,
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

template <typename T, const int group_size, const int bits, int split_k = 32>
[[kernel]] void affine_qvm_split_k(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    const constant int& final_block_size [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      y,
      out_vec_size * M,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);

  // When (in_vec_size % split_k != 0) the final block needs to be smaller
  int in_vec_size_adj =
      tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;

  // The in_vec_stride is the full K dimension, not the partition size
  int in_vec_stride = (split_k - 1) * in_vec_size + final_block_size;

  qvm_impl<T, group_size, bits>(
      w,
      scales,
      biases,
      x,
      y,
      in_vec_size_adj,
      out_vec_size,
      in_vec_stride,
      tid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
