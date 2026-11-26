// MLX v0.32.4 affine projections for wide single-row inputs. See NOTICE.
//@include fragments/shared_1.inc
    bool batched,
    bool partial_rows = false,
    bool has_global_scale = false,
//@include fragments/shared_16.inc
    const int bits,
    bool batched,
    bool partial_rows = false,
//@include fragments/quantized_3.inc
    const bool has_global_scale = false,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
//@include fragments/shared_15.inc
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void affine_qmm_t_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& k_partition_size [[buffer(8)]],
    const constant int& split_k_partition_stride [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int k_start = tid.z * k_partition_size;
  x += k_start;

  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  scales += k_start / group_size;
  biases += k_start / group_size;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  qmm_t_impl<T, group_size, bits, aligned_N, float, BM, BK, BN>(
//@include fragments/quantized_2.inc
    const bool has_global_scale = false,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
//@include fragments/shared_14.inc
    bool has_global_scale = false,
//@include fragments/quantized_8.inc
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
//@include fragments/shared_6.inc
  float w_max = 0;
//@include fragments/shared_8.inc
using namespace mlx::steel;
template <
    typename AccT,
    typename OutT,
    typename Epilogue = TransformNone<OutT, AccT>>
[[kernel]] void gemm_splitk_accum(
    const device AccT* C_split [[buffer(0)]],
    device OutT* D [[buffer(1)]],
    const constant int& k_partitions [[buffer(2)]],
    const constant int& partition_stride [[buffer(3)]],
    const constant int& ldd [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  // Ajust D and C
  D += gid.x + gid.y * size_t(ldd);
  C_split += gid.x + gid.y * size_t(ldd);

  size_t offset = 0;
  AccT out = 0;

  for (int i = 0; i < k_partitions; i++) {
    out += C_split[offset];
    offset += partition_stride;
  }

  // Write output
  D[0] = Epilogue::apply(out);
}

