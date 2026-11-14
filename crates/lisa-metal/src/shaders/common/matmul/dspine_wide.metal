// Copyright © 2023-2024 Apple Inc.
// Ported unchanged from MLX v0.32.4 gemv.h (GemvWide + gemv_wide).
// MIT license; see NOTICE. The shared utils preamble supplies basic types.
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
template <typename T> struct DefaultAccT { using type = float; };

constant bool gemv_wide_has_batch [[function_constant(0)]];
constant bool gemv_wide_do_axpby [[function_constant(1)]];

// out[M, N] = x[M, K] @ mat[N, K]^T for small M: each threadgroup streams a
// block of matrix rows once and applies it to vecs_per_tg input vectors, so
// the matrix is read ceil(M / vecs_per_tg) times instead of once per padded
// GEMM tile. k_lanes lanes reduce K for one row, 32 / k_lanes rows per
// simdgroup, k_lanes / 8 simdgroups per threadgroup.
template <
    typename T,
    const int vecs_per_tg,
    const int k_lanes,
    typename AccT = typename DefaultAccT<T>::type>
struct GemvWide {
  static constexpr constant int unroll = 8;
  static constexpr constant int num_simdgroups = k_lanes / 8;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      int in_vec_size,
      int out_vec_size,
      int M,
      int matrix_ld,
      int vector_ld,
      float alpha,
      float beta,
      int bias_ld,
      int bias_fd,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 tgpg [[threadgroups_per_grid]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int rows_per_simdgroup = 32 / k_lanes;
    constexpr int rows_per_tg = rows_per_simdgroup * num_simdgroups;

    const short k_lane =
        simd_lid % k_lanes; // this lane's slot in the K reduction
    const short sg_row =
        simd_lid / k_lanes; // which output row of the simdgroup

    const int out_row =
        tid.y * rows_per_tg + rows_per_simdgroup * simd_gid + sg_row;

    // Clamped tail rows/vectors read valid memory; the guarded writes
    // below never store them.
    const int row = min(out_row, out_vec_size - 1);
    const device T* wrow = mat + int64_t(row) * matrix_ld;
    const device vec<T, 4>* w4 = (const device vec<T, 4>*)wrow;
    const int n_v4 = in_vec_size / 4;
    const int n_main = n_v4 - n_v4 % (k_lanes * unroll);

    // Vector chunks beyond grid.x round-robin onto the same threadgroups:
    // re-walking the row block hits cache where an extra grid column would
    // re-stream it from DRAM.
    const device vec<T, 4>* x4 = (const device vec<T, 4>*)in_vec;
    const int x_ld4 = vector_ld / 4;
    const int n_chunks = (M + vecs_per_tg - 1) / vecs_per_tg;
    for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {
      const int vec0 = chunk * vecs_per_tg;

      int x_off4[vecs_per_tg];
      for (int v = 0; v < vecs_per_tg; v++) {
        x_off4[v] = min(vec0 + v, M - 1) * x_ld4;
      }

      AccT result[vecs_per_tg] = {0};

      // Adjacent lanes read adjacent blocks so every transaction lands on
      // whole cachelines; the unroll keeps loads in flight on short rows.
      for (int base = 0; base < n_main; base += k_lanes * unroll) {
        vec<AccT, 4> wf[unroll];
        MLX_MTL_PRAGMA_UNROLL
        for (int i = 0; i < unroll; i++) {
          wf[i] = vec<AccT, 4>(w4[base + i * k_lanes + k_lane]);
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int v = 0; v < vecs_per_tg; v++) {
          AccT acc = 0;
          MLX_MTL_PRAGMA_UNROLL
          for (int i = 0; i < unroll; i++) {
            acc +=
                dot(wf[i],
                    vec<AccT, 4>(x4[x_off4[v] + base + i * k_lanes + k_lane]));
          }
          result[v] += acc;
        }
      }
      for (int idx = n_main + k_lane; idx < n_v4; idx += k_lanes) {
        const vec<AccT, 4> wf = vec<AccT, 4>(w4[idx]);
        MLX_MTL_PRAGMA_UNROLL
        for (int v = 0; v < vecs_per_tg; v++) {
          result[v] += dot(wf, vec<AccT, 4>(x4[x_off4[v] + idx]));
        }
      }

      // The halving shuffles reduce each row's k_lanes while rows sharing
      // the simdgroup stay separate.
      MLX_MTL_PRAGMA_UNROLL
      for (int v = 0; v < vecs_per_tg; v++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort off = k_lanes / 2; off >= 1; off >>= 1) {
          result[v] += simd_shuffle_down(result[v], off);
        }
      }

      if (k_lane == 0 && out_row < out_vec_size) {
        for (int v = 0; v < vecs_per_tg; v++) {
          if (vec0 + v < M) {
            int64_t out_idx = int64_t(vec0 + v) * out_vec_size + out_row;
            if (gemv_wide_do_axpby) {
              AccT bias_val = static_cast<AccT>(
                  bias[int64_t(vec0 + v) * bias_ld + out_row * bias_fd]);
              out_vec[out_idx] =
                  static_cast<T>(alpha * result[v] + beta * bias_val);
            } else {
              out_vec[out_idx] = static_cast<T>(result[v]);
            }
          }
        }
      }
    }
  }
};

template <typename T, const int vecs_per_tg, const int k_lanes>
[[kernel]] void gemv_wide(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2), function_constant(gemv_wide_do_axpby)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& matrix_ld [[buffer(7)]],
    const constant int& vector_ld [[buffer(8)]],
    const constant float& alpha
    [[buffer(9), function_constant(gemv_wide_do_axpby)]],
    const constant float& beta
    [[buffer(10), function_constant(gemv_wide_do_axpby)]],
    const constant int& batch_ndim [[buffer(11)]],
    const constant int* batch_shape [[buffer(12)]],
    const constant int64_t* vector_batch_stride [[buffer(13)]],
    const constant int64_t* matrix_batch_stride [[buffer(14)]],
    const constant int64_t* bias_batch_stride
    [[buffer(15), function_constant(gemv_wide_do_axpby)]],
    const constant int& bias_ld
    [[buffer(16), function_constant(gemv_wide_do_axpby)]],
    const constant int& bias_fd
    [[buffer(17), function_constant(gemv_wide_do_axpby)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (gemv_wide_has_batch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (gemv_wide_do_axpby) {
      bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
    }
  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (gemv_wide_do_axpby) {
      bias += tid.z * bias_batch_stride[0];
    }
  }
  out_vec += int64_t(tid.z) * M * out_vec_size;

  GemvWide<T, vecs_per_tg, k_lanes>::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      M,
      matrix_ld,
      vector_ld,
      alpha,
      beta,
      bias_ld,
      bias_fd,
      tid,
      tgpg,
      simd_gid,
      simd_lid);
}

