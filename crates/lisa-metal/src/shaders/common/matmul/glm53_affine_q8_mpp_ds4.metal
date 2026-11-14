#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// GLM prefill adapter for MLX affine-Q8 weights using DS4's packed Q8_0 NAX
// prefill schedule. MLX codes are dequantized to F32, rounded to half for the
// staged A tile, then accumulated by the same 64x32x32 TensorOps descriptor
// and direct-RHS path as DS4.
template <int NR1, int NR0, int NK>
[[kernel]] void glm53_affine_q8_mpp_ds4(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device float* biases [[buffer(2)]],
    device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]]) {

    constexpr int GROUP_SIZE = 64;
    constexpr int NL = NK / 16;
    constexpr int NUM_THREADS = 128;

    const int r0 = int(tgpig.y) * NR0;
    const int r1 = int(tgpig.x) * NR1;

    threadgroup half sa[2 * NR0 * NK];
    threadgroup half* sa0 = sa;
    threadgroup half* sa1 = sa + NR0 * NK;

    auto tA0 = tensor(sa0, dextents<int32_t, 2>(NK, NR0));
    auto tA1 = tensor(sa1, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(x, dextents<int32_t, 2>(K, M), array<int, 2>({1, K}));

    matmul2d<
        matmul2d_descriptor(
            NR1,
            NR0,
            NK,
            false,
            true,
            true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    auto stage_tile = [&](const int loop_k, threadgroup half* buf) {
        for (int work = int(tiitg); work < NR0 * NL; work += NUM_THREADS) {
            const int row = work / NL;
            const int k_chunk = work % NL;
            const int k_base = k_chunk * 16;
            const int global_k = loop_k + k_base;
            const int weight_row = r0 + row;
            const int scale_index = weight_row * (K / GROUP_SIZE) + global_k / GROUP_SIZE;
            const float scale = scales[scale_index];
            const float bias = biases[scale_index];
            const uint32_t packed0 =
                w[weight_row * (K / 4) + global_k / 4 + 0];
            const uint32_t packed1 =
                w[weight_row * (K / 4) + global_k / 4 + 1];
            const uint32_t packed2 =
                w[weight_row * (K / 4) + global_k / 4 + 2];
            const uint32_t packed3 =
                w[weight_row * (K / 4) + global_k / 4 + 3];
            const float4 values0 = float4(as_type<uchar4>(packed0)) * scale + bias;
            const float4 values1 = float4(as_type<uchar4>(packed1)) * scale + bias;
            const float4 values2 = float4(as_type<uchar4>(packed2)) * scale + bias;
            const float4 values3 = float4(as_type<uchar4>(packed3)) * scale + bias;
            half4x4 values = half4x4(
                half4(values0), half4(values1), half4(values2), half4(values3));
            threadgroup half4* dst4 =
                (threadgroup half4*)(buf + row * NK + k_base);
            dst4[0] = values[0];
            dst4[1] = values[1];
            dst4[2] = values[2];
            dst4[3] = values[3];
        }
    };

    stage_tile(0, sa0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_sel = 0;
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        auto mA = (buf_sel ? tA1 : tA0).slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        const int next_k = loop_k + NK;
        if (next_k < K) {
            buf_sel ^= 1u;
            stage_tile(next_k, buf_sel ? sa1 : sa0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(y, dextents<int32_t, 2>(N, M), array<int, 2>({1, N}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

template [[host_name("glm53_affine_q8_mpp_ds4_f32_gs64_b8_nr1_32_nr0_64_nk32")]]
[[kernel]] decltype(glm53_affine_q8_mpp_ds4<32, 64, 32>)
    glm53_affine_q8_mpp_ds4<32, 64, 32>;
template [[host_name("glm53_affine_q8_mpp_ds4_f32_gs64_b8_nr1_64_nr0_64_nk32")]]
[[kernel]] decltype(glm53_affine_q8_mpp_ds4<64, 64, 32>)
    glm53_affine_q8_mpp_ds4<64, 64, 32>;
template [[host_name("glm53_affine_q8_mpp_ds4_f32_gs64_b8_nr1_128_nr0_64_nk32")]]
[[kernel]] decltype(glm53_affine_q8_mpp_ds4<128, 64, 32>)
    glm53_affine_q8_mpp_ds4<128, 64, 32>;
