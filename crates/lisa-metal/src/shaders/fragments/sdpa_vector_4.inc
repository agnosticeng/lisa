    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Duplication-free variant for high gqa_factor decode: each simdgroup owns a
// contiguous token sub-chunk and computes HPT of its group's query heads, so
// each K/V byte is read G / HPT times instead of G times. Single-token
// queries without mask or sinks only; the partials layout matches
// sdpa_vector_2pass_2.
template <typename T, int D, int V, int G, int HPT>
[[kernel]] void sdpa_vector_2pass_1_gqa(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int NT = G / HPT;

  typedef float U;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int blocks = tpg.z;
  const int g = tidtg.y;
  const int cchunk = g / NT;
  const int h0 = (g % NT) * HPT;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * G;
  const int base_head = batch_idx * num_q_heads + kv_head_idx * G;
