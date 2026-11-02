#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// DS4-style MPP tile for MLX affine-Q2 weights. Each affine group has 128
// values and stores its unsigned 2-bit codes in four low-to-high fields per
// byte. Dequantized weights are rounded to half before MPP accumulation.
template <int NR1, int NR0, int NK, typename M>
[[kernel]] void glm53_affine_q2_mpp_ds4(
    const device uint8_t* w [[buffer(0)]],
    const device M* scales [[buffer(1)]],
    const device M* biases [[buffer(2)]],
    device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M_rows [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]]) {

    constexpr int GROUP_SIZE = 128;
    constexpr int NUM_THREADS = 128;
    const int WEIGHT_ROW_BYTES = K / 4;
    const int SCALE_GROUPS = K / GROUP_SIZE;

    const int m0 = int(tgpig.x) * NR1;
    const int n0 = int(tgpig.y) * NR0;

    threadgroup half sa[2 * NR0 * NK];
    threadgroup half* sa0 = sa;
    threadgroup half* sa1 = sa + NR0 * NK;

    auto tA0 = tensor(sa0, dextents<int32_t, 2>(NK, NR0));
    auto tA1 = tensor(sa1, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(x, dextents<int32_t, 2>(K, M_rows), array<int, 2>({1, K}));

    matmul2d<
        matmul2d_descriptor(
            NR1,
            NR0,
            NK,
            false,
            true,
            false,
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
        for (int work = int(tiitg); work < NR0 * NK; work += NUM_THREADS) {
            const int row = work / NK;
            const int local_k = work % NK;
            const int global_k = loop_k + local_k;
            const int weight_row = n0 + row;
            const uint8_t packed = w[weight_row * WEIGHT_ROW_BYTES + global_k / 4];
            const int shift = (global_k & 3) * 2;
            const float code = float((packed >> shift) & 0x03);
            const int scale_index = weight_row * SCALE_GROUPS + global_k / GROUP_SIZE;
            const float value = code * float(scales[scale_index]) + float(biases[scale_index]);
            buf[row * NK + local_k] = half(value);
        }
    };

    stage_tile(0, sa0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_sel = 0;
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        auto mA = (buf_sel ? tA1 : tA0).slice(0, 0);
        auto mB = tB.slice(loop_k, m0);
        mm.run(mB, mA, cT);

        const int next_k = loop_k + NK;
        if (next_k < K) {
            buf_sel ^= 1u;
            stage_tile(next_k, buf_sel ? sa1 : sa0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(y, dextents<int32_t, 2>(N, M_rows), array<int, 2>({1, N}));
    auto mD = tD.slice(n0, m0);
    cT.store(mD);
}

template [[host_name("glm53_affine_q2_mpp_ds4_f32_nr1_32_nr0_64_nk32")]]
[[kernel]] decltype(glm53_affine_q2_mpp_ds4<32, 64, 32, float>)
    glm53_affine_q2_mpp_ds4<32, 64, 32, float>;
template [[host_name("glm53_affine_q2_mpp_ds4_bf16_nr1_32_nr0_64_nk32")]]
[[kernel]] decltype(glm53_affine_q2_mpp_ds4<32, 64, 32, bfloat16_t>)
    glm53_affine_q2_mpp_ds4<32, 64, 32, bfloat16_t>;
