// Native DS4 Q8_0 decode schedule, b4e3faf5262b21db94ec309cf321640ff1eed460.
// Port of metal/dense.metal::kernel_mul_mv_q8_0_f32_impl and its two-level
// reduction. Only storage access is adapted to original MLX group-64 affine
// bytes. Centered codes preserve the native signed-Q8 dot; a separate residual
// bias term extends that schedule to affine weights without requantization.
#include <metal_stdlib>
using namespace metal;
#define GLM53_TINY_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct glm53_q8_tiny_native_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
    uint metadata_bf16;
};

static inline float glm53_q8_tiny_native_metadata(
        device const uchar *bytes, uint index, uint bf16) {
    if (bf16 != 0u) {
        return as_type<float>((uint)((device const ushort *)bytes)[index] << 16);
    }
    return ((device const float *)bytes)[index];
}

template<short NSG>
kernel void glm53_affine_q8_mv_native(
        constant glm53_q8_tiny_native_args &args [[buffer(0)]],
        device const uchar *weights [[buffer(1)]],
        device const float *x [[buffer(2)]],
        device float *out [[buffer(3)]],
        device const uchar *scales [[buffer(4)]],
        device const uchar *biases [[buffer(5)]],
        threadgroup float *shared [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NW = 32;
    constexpr short NQ = 8;
    const int nb = args.in_dim / 32u;
    const int row0 = group.x * NR0;
    const short ix = lane / (NW / NQ);
    const short il = lane % (NW / NQ);
    const int ib0 = simdgroup * NQ + ix;
    const uint groups_per_row = args.in_dim / 64u;
    float sumf[NR0] = {0.0f};
    float yl[NQ];
    device const float *yb = x + ib0 * 32 + il * NQ;
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        for (short row = 0; row < NR0; row++) {
            const uint output_row = min(uint(row0 + row), args.out_dim - 1u);
            const uint affine = output_row * groups_per_row + uint(ib) / 2u;
            const float scale = glm53_q8_tiny_native_metadata(scales, affine, args.metadata_bf16);
            const float bias = glm53_q8_tiny_native_metadata(biases, affine, args.metadata_bf16);
            const float residual = fma(128.0f, scale, bias);
            device const uchar *qs = weights + (ulong)output_row * args.in_dim
                + ib * 32 + il * NQ;
            float sumq = 0.0f;
            float sumx = 0.0f;
            GLM53_TINY_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += (float(qs[i]) - 128.0f) * yl[i];
                sumx += yl[i];
            }
            // The zero-residual path is exactly native Q8_0's block update.
            sumf[row] += sumq * scale;
            if (residual != 0.0f) sumf[row] += sumx * residual;
        }
        yb += NSG * NQ * 32;
    }

    // Preserve helper_mv_reduce_and_write: SIMD reduction, shared lane-zero
    // partials, then a second SIMD reduction. No XOR-tree substitution.
    for (short row = 0; row < NR0; ++row) {
        if (simdgroup == 0) shared[row * NW + lane] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        if (lane == 0) shared[row * NW + simdgroup] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && (uint)(row0 + row) < args.out_dim; ++row) {
        const float total = simd_sum(shared[row * NW + lane]);
        if (lane == 0 && simdgroup == 0) out[row0 + row] = total;
    }
}

typedef decltype(glm53_affine_q8_mv_native<4>) glm53_affine_q8_mv_native_t;
template [[host_name("glm53_affine_q8_mv_native_nsg4")]]
kernel glm53_affine_q8_mv_native_t glm53_affine_q8_mv_native<4>;
template [[host_name("glm53_affine_q8_mv_native_nsg8")]]
kernel glm53_affine_q8_mv_native_t glm53_affine_q8_mv_native<8>;
