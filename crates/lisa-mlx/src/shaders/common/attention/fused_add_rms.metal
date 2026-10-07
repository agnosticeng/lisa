// Fused residual-add + RMSNorm (specs/08 §1): ONE kernel returns both the new
// residual sum and the normed input. Bit-exact by construction against the
// composed chain `bf16_add(x, r)` → `rms_single_row`/`rms_looped`:
// - the sum is computed as float(x)+float(r) and rounded to T (exactly what
//   MLX's bf16 binary add produces),
// - the rounded sum is what feeds the sum-of-squares (the composed chain reads
//   the bf16 sum back), and
// - the normalization half is a verbatim copy of rms_norm.metal.
#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void fused_add_rms_single_row(
    const device T* x,
    const device T* r,
    const device T* w,
    device T* sum_out,
    device T* norm_out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  float thread_x[N_READS];
  x += gid * size_t(axis_size) + lid * N_READS;
  r += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = static_cast<T>(static_cast<float>(x[i]) + static_cast<float>(r[i]));
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = (lid * N_READS + i < axis_size)
          ? static_cast<T>(static_cast<float>(x[i]) + static_cast<float>(r[i]))
          : 0;
      acc += thread_x[i] * thread_x[i];
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  sum_out += gid * size_t(axis_size) + lid * N_READS;
  norm_out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      sum_out[i] = static_cast<T>(thread_x[i]);
      norm_out[i] =
          w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        sum_out[i] = static_cast<T>(thread_x[i]);
        norm_out[i] =
            w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void fused_add_rms_looped(
    const device T* x,
    const device T* r,
    const device T* w,
    device T* sum_out,
    device T* norm_out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  r += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint ro = 0; ro < axis_size; ro += lsize * N_READS) {
    if (ro + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float s = static_cast<float>(x[i + ro]) + static_cast<float>(r[i + ro]);
        acc += static_cast<float>(static_cast<T>(s)) *
            static_cast<float>(static_cast<T>(s));
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((ro + lid * N_READS + i) < axis_size) {
          float s = static_cast<float>(x[i + ro]) + static_cast<float>(r[i + ro]);
          acc += static_cast<float>(static_cast<T>(s)) *
              static_cast<float>(static_cast<T>(s));
        }
      }
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  sum_out += gid * size_t(axis_size) + lid * N_READS;
  norm_out += gid * size_t(axis_size) + lid * N_READS;
  for (uint ro = 0; ro < axis_size; ro += lsize * N_READS) {
    if (ro + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        T s = static_cast<T>(
            static_cast<float>(x[i + ro]) + static_cast<float>(r[i + ro]));
        sum_out[ro + i] = s;
        norm_out[ro + i] = w[w_stride * (i + ro)] *
            static_cast<T>(static_cast<float>(s) * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((ro + lid * N_READS + i) < axis_size) {
          T s = static_cast<T>(
              static_cast<float>(x[i + ro]) + static_cast<float>(r[i + ro]));
          sum_out[ro + i] = s;
          norm_out[ro + i] = w[w_stride * (i + ro)] *
              static_cast<T>(static_cast<float>(s) * local_inv_mean[0]);
        }
      }
    }
  }
}
