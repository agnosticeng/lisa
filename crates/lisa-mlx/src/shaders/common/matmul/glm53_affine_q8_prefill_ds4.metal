// DS4 Q8 prefill schedule, commit b4e3faf5262b21db94ec309cf321640ff1eed460.
// Adapted from metal/dense.metal via the independent MLX storage adapter.
// Only affine byte decoding replaces native Q8 block loads. Both original
// F32 and BF16 metadata are read in their stored precision.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#ifdef DS4_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
constant bool FC_mul_mm_bc_inp [[function_constant(0)]];
constant bool FC_mul_mm_bc_out [[function_constant(1)]];

struct ds4_metal_args_mul_mm {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

static inline float glm53_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

struct glm53_bf16_matmul_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

struct glm53_mlx_affine_matvec_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
    uint bits;
    uint group_size;
    uint metadata_bf16;
};

static inline float glm53_mlx_affine_metadata(
        device const uchar *metadata,
        uint index,
        uint metadata_bf16) {
    if (metadata_bf16 != 0u) {
        const ushort value = *reinterpret_cast<device const ushort *>(metadata + 2u * index);
        return glm53_bf16_to_f32(value);
    }
    return *reinterpret_cast<device const float *>(metadata + 4u * index);
}

// Dense affine Q8 candidate adapted directly from upstream dense.metal at
// b4e3faf5262b21db94ec309cf321640ff1eed460. Storage decoding is the only
// change inside each matrix schedule; scalar admission remains separately
// selectable. This is not the historical scalar affine numerical law.
#ifdef DS4_METAL_HAS_TENSOR
template<short NR1>
kernel void kernel_glm53_mlx_affine_q8_mpp(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        device const uchar *scales,
        device const uchar *biases,
        constant glm53_mlx_affine_matvec_args &affine,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;
    using SA = half;
    using T1 = float;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    auto tA0 = tensor(sa,          dextents<int32_t, 2>(NK, NR0));
    auto tA1 = tensor(sa + NR0*NK, dextents<int32_t, 2>(NK, NR0));

    device T1 *ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA0), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    // NR0*NL/NUM_THREADS 16-value weight chunks per thread (1 at NR0=64).
    auto stage_tile = [&](const int loop_k, threadgroup SA *buf) {
        FOR_UNROLL (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work / NL;
            const int k_chunk = work % NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;
            const uint output_row = r0 + row;
            const uint affine_index = output_row * (affine.in_dim / affine.group_size) +
                                      uint(k_pos) / affine.group_size;
            const float scale = glm53_mlx_affine_metadata(scales, affine_index, affine.metadata_bf16);
            const float bias = glm53_mlx_affine_metadata(biases, affine_index, affine.metadata_bf16);
            device const uchar *row_ptr = reinterpret_cast<device const uchar *>(
                srcA + args.nb01 * output_row + offset0);
            half4x4 temp_a;
            FOR_UNROLL (short i = 0; i < 16; i++) {
                temp_a[i / 4][i % 4] = half(fma(float(row_ptr[k_pos + i]), scale, bias));
            }
            threadgroup half4 *dst4 = (threadgroup half4 *)(buf + row*NK + k_base);
            dst4[0] = temp_a[0];
            dst4[1] = temp_a[1];
            dst4[2] = temp_a[2];
            dst4[3] = temp_a[3];
        }
    };

    stage_tile(0, sa);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf_sel = 0;
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        auto mA = (buf_sel ? tA1 : tA0).slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        const int next_k = loop_k + NK;
        if (next_k < K) {
            buf_sel ^= 1u;
            stage_tile(next_k, buf_sel ? sa + NR0*NK : sa);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

template [[host_name("kernel_glm53_mlx_affine_q8_mpp_n32")]]
kernel decltype(kernel_glm53_mlx_affine_q8_mpp<32>) kernel_glm53_mlx_affine_q8_mpp<32>;
template [[host_name("kernel_glm53_mlx_affine_q8_mpp_n64")]]
kernel decltype(kernel_glm53_mlx_affine_q8_mpp<64>) kernel_glm53_mlx_affine_q8_mpp<64>;
template [[host_name("kernel_glm53_mlx_affine_q8_mpp_n128")]]
kernel decltype(kernel_glm53_mlx_affine_q8_mpp<128>) kernel_glm53_mlx_affine_q8_mpp<128>;
#endif

kernel void kernel_glm53_mlx_affine_q8_mm(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const uchar *scales,
        device const uchar *biases,
        constant glm53_mlx_affine_matvec_args &affine,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    using S0 = half;
    using S1 = half;
    using T1 = float;
    using T1_2x4 = float2x4;
    using S1_2x4 = half2x4;
    using S0_8x8 = simdgroup_half8x8;
    using S1_8x8 = simdgroup_half8x8;
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 64u*32u*sizeof(S0));

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const uint output_row = r0 + lr0;
    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    device const uchar *weight_row = reinterpret_cast<device const uchar *>(
        src0 + args.nb01 * output_row + offset0);

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // Only the affine load differs from upstream kernel_mul_mm. Keep the
        // half staging, SIMD matrix order, barriers and boundary stores.
        const uint k_pos = uint(loop_k) + 16u * uint(il0);
        const uint affine_index = output_row * (affine.in_dim / affine.group_size) +
                                  k_pos / affine.group_size;
        const float scale = glm53_mlx_affine_metadata(scales, affine_index, affine.metadata_bf16);
        const float bias = glm53_mlx_affine_metadata(biases, affine_index, affine.metadata_bf16);
        half4x4 temp_a;
        FOR_UNROLL (short i = 0; i < 16; i++) {
            temp_a[i/4][i%4] = half(fma(float(weight_row[k_pos + i]), scale, bias));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}
