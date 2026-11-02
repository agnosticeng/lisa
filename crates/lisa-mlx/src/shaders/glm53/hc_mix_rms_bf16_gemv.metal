// Fused mHC RMSNorm and BF16 GEMV for GLM-5.3's [24,16384] projection.
// RMS reduction is the ds4 float4/SIMD/threadgroup tree from hc_mix_rms.metal;
// projection accumulation is the ds4 lane-strided BF16 GEMV order.
{
    constexpr uint RMS_VTHREADS = 1024u;
    constexpr uint OUT_DIM = 24u;

    const uint token = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint nsg = simdgroups_per_threadgroup;
    const uint threads = nsg * 32u;
    const uint n = (uint)n_dim;
    const uint n4 = n / 4u;
    if (token >= (uint)n_rows || n != 16384u || out_dim != (int)OUT_DIM ||
        (nsg != 4u && nsg != 8u)) return;

    // Match DS4's GLM BF16 GEMV scheduling: eight simdgroups per threadgroup,
    // or four on M3 Ultra. Fold those physical lanes over the same 1024
    // virtual RMS threads as the standalone norm so its reduction tree stays
    // unchanged.
    threadgroup float shared[32];

    const device float4 *x4 = (const device float4 *)input + (size_t)token * n4;
    const uint virtual_slices = RMS_VTHREADS / threads;
    for (uint v = 0; v < virtual_slices; ++v) {
        const uint virtual_tid = (simdgroup + nsg * v) * 32u + lane;
        float rms_sum = 0.0f;
        for (uint i = virtual_tid; i < n4; i += RMS_VTHREADS) {
            rms_sum += dot(x4[i], x4[i]);
        }
        rms_sum = simd_sum(rms_sum);
        if (lane == 0u) shared[simdgroup + nsg * v] = rms_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_square_sum = simd_sum(shared[lane]);
    const float scale = 1.0f / sqrt(total_square_sum / (float)n + eps);

    const uint out_row = threadgroup_position_in_grid.x * nsg + simdgroup;
    if (out_row >= OUT_DIM) return;
    device const bfloat16_t *w = weights + (size_t)out_row * n;
    device const float *x = input + (size_t)token * n;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < n; k += 256u) {
        const float x0 = x[k] * scale;
        const float x1 = x[k + 32u] * scale;
        const float x2 = x[k + 64u] * scale;
        const float x3 = x[k + 96u] * scale;
        const float x4v = x[k + 128u] * scale;
        const float x5 = x[k + 160u] * scale;
        const float x6 = x[k + 192u] * scale;
        const float x7 = x[k + 224u] * scale;
        sum = fma(float(w[k]), x0, sum);
        sum = fma(float(w[k + 32u]), x1, sum);
        sum = fma(float(w[k + 64u]), x2, sum);
        sum = fma(float(w[k + 96u]), x3, sum);
        sum = fma(float(w[k + 128u]), x4v, sum);
        sum = fma(float(w[k + 160u]), x5, sum);
        sum = fma(float(w[k + 192u]), x6, sum);
        sum = fma(float(w[k + 224u]), x7, sum);
    }
    for (; k < n; k += 32u) {
        const float normalized = x[k] * scale;
        sum = fma(float(w[k]), normalized, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        output[(size_t)token * OUT_DIM + out_row] = sum;
    }
}
