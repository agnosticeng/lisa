// Native DS4 dense.metal small-row Q8 schedule, with MLX affine loads.
// The chunk walk, float4 dot and subgroup reduction match upstream b4e3faf.
#include <metal_stdlib>
using namespace metal;

constant int GLM53_Q8_EXT_NSG [[function_constant(0)]];
constant int GLM53_Q8_EXT_NXPSG [[function_constant(1)]];
struct glm53_affine_q8_ext_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
    uint metadata_bf16;
};

inline float glm53_q8_ext_metadata(device const uchar *values, ulong index, bool bf16) {
    return bf16 ? as_type<float>(uint(((device const ushort *)values)[index]) << 16)
                : ((device const float *)values)[index];
}

inline void glm53_q8_ext_load(
        device const uchar *weight, device const uchar *scales,
        device const uchar *biases, ulong row, uint in_dim,
        uint block, short chunk, bool bf16, thread float4 &reg) {
    const uint col = block * 32u + uint(chunk) * 4u;
    const ulong affine = row * (in_dim / 64u) + col / 64u;
    const float scale = glm53_q8_ext_metadata(scales, affine, bf16);
    const float bias = glm53_q8_ext_metadata(biases, affine, bf16);
    for (int i = 0; i < 4; i++) {
        reg[i] = fma(float(weight[row * in_dim + col + uint(i)]), scale, bias);
    }
}

template<short r1ptg>
kernel void glm53_affine_q8_ext_small_native(
        constant glm53_affine_q8_ext_args &args [[buffer(0)]],
        device const uchar *weight [[buffer(1)]],
        device const float *input [[buffer(2)]],
        device float *output [[buffer(3)]],
        device const uchar *scales [[buffer(4)]],
        device const uchar *biases [[buffer(5)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = short(GLM53_Q8_EXT_NSG);
    const short nxpsg = short(GLM53_Q8_EXT_NXPSG);
    const short chpt = 4;
    const short chpb = 8;
    const short nypsg = 32 / nxpsg;
    const short tx = tiisg % nxpsg;
    const short ty = tiisg / nxpsg;
    const int i01 = tgpig.x * (nypsg * NSG) + nypsg * sgitg + ty;
    const int i11 = tgpig.y * r1ptg;
    const ulong row = i01 < int(args.out_dim) ? ulong(i01) : 0;
    uint block = i01 < int(args.out_dim) ? uint(tx / chpb) : 0u;
    device const float4 *y4[r1ptg];
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = i11 + ir1 < int(args.n_rows)
            ? (device const float4 *)(input + ulong(i11 + ir1) * args.in_dim) + tx
            : (device const float4 *)input;
    }
    float sumf[r1ptg] = { [0 ... r1ptg - 1] = 0.0f };
    short cch = tx % chpb;
    for (int ich = tx; 4 * ich < int(args.in_dim); ich += chpt * nxpsg) {
        float4 lx[chpt];
#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            glm53_q8_ext_load(weight, scales, biases, row, args.in_dim,
                             block, cch, args.metadata_bf16 != 0u, lx[ch]);
            cch += nxpsg;
            if (cch >= chpb) {
                block += cch / chpb;
                cch %= chpb;
            }
        }
#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch * nxpsg]);
            }
        }
#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) y4[ir1] += chpt * nxpsg;
    }
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        if (nxpsg >= 16) sumf[ir1] += simd_shuffle_down(sumf[ir1], 8);
        if (nxpsg >= 8) sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
        if (nxpsg >= 4) sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
        if (nxpsg >= 2) sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }
    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < int(args.n_rows); ++ir1) {
            if (i01 < int(args.out_dim)) {
                output[ulong(i11 + ir1) * args.out_dim + i01] = sumf[ir1];
            }
        }
    }
}

typedef decltype(glm53_affine_q8_ext_small_native<2>) glm53_affine_q8_ext_small_native_t;
template [[host_name("glm53_affine_q8_ext_small_native_r2")]] kernel glm53_affine_q8_ext_small_native_t glm53_affine_q8_ext_small_native<2>;
template [[host_name("glm53_affine_q8_ext_small_native_r3")]] kernel glm53_affine_q8_ext_small_native_t glm53_affine_q8_ext_small_native<3>;
template [[host_name("glm53_affine_q8_ext_small_native_r4")]] kernel glm53_affine_q8_ext_small_native_t glm53_affine_q8_ext_small_native<4>;
template [[host_name("glm53_affine_q8_ext_small_native_r5")]] kernel glm53_affine_q8_ext_small_native_t glm53_affine_q8_ext_small_native<5>;
