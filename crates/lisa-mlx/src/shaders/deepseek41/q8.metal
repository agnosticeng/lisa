// Verbatim DS4 kernel sections; see quant_provenance.json and LICENSE.
#include <metal_stdlib>
using namespace metal;
#define QK8_0 32
#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
struct block_q8_0 { half d; int8_t qs[QK8_0]; };
// DS4 Metal matvec kernels used by generation.

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

// Round to the nearest bfloat16 value in f32 storage (DeepSeek V4.1 keeps its
// activations at bf16 precision between operators).
static inline float ds4_bf16_round(float v) {
    uint bits = as_type<uint>(v);
    if ((bits & 0x7f800000u) != 0x7f800000u) bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}
static inline float4 ds4_bf16_round(float4 v) {
    uint4 bits = as_type<uint4>(v);
    const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
    bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
    return as_type<float4>(bits & 0xffff0000u);
}

struct ds4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};

struct ds4_metal_args_compressor_pair_store {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
};

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

struct ds4_metal_args_mul_mv_ext {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
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

template<short NR0, bool ROUND = false>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = ROUND ? ds4_bf16_round(tot) : tot;
        }
    }
}

// Q8's single-use reduction keeps the same two simd_sum stages and the
// same positive-zero lanes as helper_mv_reduce_and_write. Each SIMD group
// publishes only its own partial; padding is supplied in registers instead
// of clearing shared memory before a separate barrier. Only group 0 needs
// the final reduction. No caller may reuse this scratch before another
// threadgroup barrier: the other groups can finish before group 0 reads it.
template<short NR0>
static inline void helper_q8_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;
    const short NSG = FC_mul_mv_nsg;
    threadgroup float * partials = (threadgroup float *) shmem;

    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
        if (NSG > 1 && tiisg == 0) {
            partials[NW*row + sgitg] = sumf[row];
        }
    }

    if (NSG > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sgitg == 0) {
        for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
            // Preserve the second reduction even for NSG=1, including its
            // positive-zero padding rather than directly storing sumf.
            float value = 0.0f;
            if (NSG == 1) {
                if (tiisg == 0) value = sumf[row];
            } else if (tiisg < NSG) {
                value = partials[NW*row + tiisg];
            }
            const float tot = simd_sum(value);
            if (tiisg == 0) dst_f32[r0 + row] = tot;
        }
    }
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_plain_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        // Packed activation loads also permit scalar-aligned tensor views.
        const float4 y0 = float4(((device const packed_float4 *)yb)[0]);
        const float4 y1 = float4(((device const packed_float4 *)yb)[1]);
        FOR_UNROLL (short i = 0; i < 4; ++i) {
            yl[i] = y0[i];
            yl[i + 4] = y1[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            // Preserve the reference eight-term loop. Pairing quants as
            // char2 changes Metal fast-math code generation and rounding;
            // small projection errors can amplify through recurrent decode.
            // The full GEMV oracle pins this arithmetic, not just reduction.
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_q8_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Rope folded into a matvec: the last 64 columns of every `width`-wide head
// rotate at position `start` (ROPE_IN on the activation as it loads, after
// rounding it to bf16; ROPE_OUT on the bf16 outputs before they store).
struct ds4_metal_args_mv_rope {
    uint  width;
    uint  start;
    uint  inverse;
    float frequencies[32];
};

template<short NR0>
static inline void helper_mv_reduce_and_write_rope(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem,
        constant ds4_metal_args_mv_rope & rope) {
    static_assert(NR0 % 2 == 0, "rope pairs are adjacent output rows");
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot[NR0];
    for (short row = 0; row < NR0; ++row) {
        tot[row] = ds4_bf16_round(simd_sum(shmem_f32[row][tiisg]));
    }

    if (tiisg == 0 && sgitg == 0) {
        for (short p = 0; p < NR0 && r0 + p + 1 < ne01; p += 2) {
            const uint col = (uint)(r0 + p) % rope.width;
            if (col >= rope.width - 64u) {
                const float theta = float(rope.start) * rope.frequencies[(col - (rope.width - 64u)) >> 1u];
                const float c = precise::cos(theta);
                const float s = rope.inverse ? -precise::sin(theta) : precise::sin(theta);
                const float re = tot[p], im = tot[p + 1];
                dst_f32[r0 + p] = ds4_bf16_round(fma(re, c, -(im * s)));
                dst_f32[r0 + p + 1] = ds4_bf16_round(fma(im, c, re * s));
            } else {
                dst_f32[r0 + p] = tot[p];
                dst_f32[r0 + p + 1] = tot[p + 1];
            }
        }
    }
}

// The HC expand (kernel_dsv4_hc_expand4, one token, four streams) applied to
// each bf16 output as it is written: the row joins `add` when has_add and
// spreads into the streams with the same per-stream accumulation order.
struct ds4_metal_args_mv_hc_expand4 {
    uint n_embd;
    uint has_add;
    uint split_stride;   /* floats between the rows' post/comb weights */
};

template<short NR0>
static inline void helper_mv_reduce_and_write_hc_expand4(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem,
        constant ds4_metal_args_mv_hc_expand4 & hc,
        device const float * add,
        device const float * residual,
        device const float * post,
        device const float * comb) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float block_v = ds4_bf16_round(simd_sum(shmem_f32[row][tiisg]));

        if (tiisg == 0 && sgitg == 0) {
            const int d = r0 + row;
            if (hc.has_add) {
                block_v += add[d];
                block_v = ds4_bf16_round(block_v);
            }
            const float r0v = residual[d];
            const float r1v = residual[d + hc.n_embd];
            const float r2v = residual[d + 2*hc.n_embd];
            const float r3v = residual[d + 3*hc.n_embd];

            for (int dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * post[dst_hc];

                acc += comb[dst_hc + 0*4] * r0v;
                acc += comb[dst_hc + 1*4] * r1v;
                acc += comb[dst_hc + 2*4] * r2v;
                acc += comb[dst_hc + 3*4] * r3v;

                dst_f32[d + dst_hc*hc.n_embd] = ds4_bf16_round(acc);
            }
        }
    }
}

template<short NR0, typename args_t, bool ROUND = false, bool ROUND_IN = false,
         bool ROPE_IN = false, bool ROPE_OUT = false, bool HC_OUT = false>
void kernel_mul_mv_q8_0_f32_rope_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg,
        constant ds4_metal_args_mv_rope * rope,
        constant ds4_metal_args_mv_hc_expand4 * hc = nullptr,
        device const float * hc_add = nullptr,
        device const float * hc_residual = nullptr,
        device const float * hc_post = nullptr,
        device const float * hc_comb = nullptr) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = ROUND_IN ? ds4_bf16_round(yb[i]) : yb[i];
        }

        if (ROPE_IN) {
            const uint col = (uint)(ib*QK8_0 + il*NQ) % rope->width;
            if (col >= rope->width - 64u) {
                const uint pair0 = (col - (rope->width - 64u)) >> 1u;
                FOR_UNROLL (short p = 0; p < NQ/2; ++p) {
                    const float theta = float(rope->start) * rope->frequencies[pair0 + p];
                    float c;
                    const float sn = precise::sincos(theta, c);
                    const float s = rope->inverse ? -sn : sn;
                    const float re = yl[2*p], im = yl[2*p + 1];
                    yl[2*p] = ds4_bf16_round(fma(re, c, -(im * s)));
                    yl[2*p + 1] = ds4_bf16_round(fma(im, c, re * s));
                }
            }
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (HC_OUT) {
        helper_mv_reduce_and_write_hc_expand4<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem,
                                                   *hc, hc_add, hc_residual, hc_post, hc_comb);
    } else if (ROPE_OUT) {
        helper_mv_reduce_and_write_rope<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem, *rope);
    } else {
        helper_mv_reduce_and_write<NR0, ROUND>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
    }
}

template<short NR0, typename args_t, bool ROUND = false, bool ROUND_IN = false>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    kernel_mul_mv_q8_0_f32_rope_impl<NR0, args_t, ROUND, ROUND_IN>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg, (constant ds4_metal_args_mv_rope *)nullptr);
}


[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_plain_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}
[[host_name("kernel_mul_mv_q8_0_f32_bf16")]]
kernel void kernel_mul_mv_q8_0_f32_bf16(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &, true>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}
[[host_name("kernel_mul_mv_q8_0_f32_bf16io")]]
kernel void kernel_mul_mv_q8_0_f32_bf16io(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &, true, true>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

static inline ds4_metal_args_mul_mv ds4_q8_0_static_args(int in, int out) {
    const ulong row = (ulong)(in / 32) * 34;
    return ds4_metal_args_mul_mv{in, out, 1, 34, row, row * out, row * out, in, 1, 1, 4,
                                 (ulong)in * 4, (ulong)in * 4, (ulong)in * 4, out, 1, 2, 1, 1};
}


// Two Q8_0 matrices over one activation row in one dispatch: the leading
// threadgroups take the first matrix's row groups, the rest the second's.
[[host_name("kernel_mul_mv_q8_0_f32_bf16_pair")]]
kernel void kernel_mul_mv_q8_0_f32_bf16_pair(
        constant ds4_metal_args_mul_mv & args_a,
        constant ds4_metal_args_mul_mv & args_b,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint groups_a = ((uint)args_a.ne01 + N_R0_Q8_0 - 1u) / N_R0_Q8_0;
    if (tgpig.x < groups_a) {
        kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &, true>(args_a, src0_a, src1, dst_a, shmem, tgpig, tiisg, sgitg);
    } else {
        tgpig.x -= groups_a;
        kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &, true>(args_b, src0_b, src1, dst_b, shmem, tgpig, tiisg, sgitg);
    }
}

/* The q_a/kv pair at its V4.1 decode shape (5120 -> 1280, 5120 -> 512) as literals. */
[[host_name("kernel_mul_mv_q8_0_f32_bf16_pair_qa_kv")]]
kernel void kernel_mul_mv_q8_0_f32_bf16_pair_qa_kv(
        constant ds4_metal_args_mul_mv & args_a,
        constant ds4_metal_args_mul_mv & args_b,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)args_a; (void)args_b;
    if (tgpig.x < 640u) {
        ds4_metal_args_mul_mv sa = ds4_q8_0_static_args(5120, 1280);
        kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &, true>(sa, src0_a, src1, dst_a, shmem, tgpig, tiisg, sgitg);
    } else {
        ds4_metal_args_mul_mv sa = ds4_q8_0_static_args(5120, 512);
        tgpig.x -= 640u;
        kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &, true>(sa, src0_b, src1, dst_b, shmem, tgpig, tiisg, sgitg);
    }
}

