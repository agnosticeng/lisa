// Native DS4 paired Q8 gate/up plus SwiGLU. The embedding source supplies
// affine metadata helpers; arithmetic and reductions follow dense.metal.

template<short r1ptg>
kernel void glm53_affine_q8_pair_native(
        constant glm53_affine_q8_ext_args &args [[buffer(0)]],
        device const uchar *src0_gate [[buffer(1)]],
        device const uchar *src0_up [[buffer(2)]],
        device const char *src1 [[buffer(3)]],
        device char *dst_gate [[buffer(4)]],
        device char *dst_up [[buffer(5)]],
        device char *dst_mid [[buffer(6)]],
        constant float &clamp_value [[buffer(7)]],
        device const uchar *gate_scales [[buffer(8)]],
        device const uchar *gate_biases [[buffer(9)]],
        device const uchar *up_scales [[buffer(10)]],
        device const uchar *up_biases [[buffer(11)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG   = short(GLM53_Q8_EXT_NSG);
    const short nxpsg = short(GLM53_Q8_EXT_NXPSG);

    const short chpt = 4;
    const short chpb = 8;
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const ulong row = i01 < int(args.out_dim) ? ulong(i01) : 0;
    uint block = i01 < int(args.out_dim) ? uint(tx / chpb) : 0u;
    const ulong offset1 = ulong(i11) * args.in_dim * sizeof(float);

    device const float4 * y4[r1ptg];
    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.n_rows)
            ? (device const float4 *)(src1 + offset1 + ir1*(ulong(args.in_dim) * sizeof(float))) + tx
            : (device const float4 *)src1;
    }

    float sum_gate[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };
    float sum_up[r1ptg]   = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb;
    for (int ich = tx; 4*ich < args.in_dim; ich += chpt*nxpsg) {
        float4 lg[chpt];
        float4 lu[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            glm53_q8_ext_load(src0_gate, gate_scales, gate_biases, row, args.in_dim, block, cch, args.metadata_bf16 != 0u, lg[ch]);
            glm53_q8_ext_load(src0_up, up_scales, up_biases, row, args.in_dim, block, cch, args.metadata_bf16 != 0u, lu[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                block += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                const float4 y = y4[ir1][ch*nxpsg];
                sum_gate[ir1] += dot(lg[ch], y);
                sum_up[ir1] += dot(lu[ch], y);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1], 16);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],   16);
        }
        if (nxpsg >= 16) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  8);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    8);
        }
        if (nxpsg >= 8) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  4);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    4);
        }
        if (nxpsg >= 4) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  2);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    2);
        }
        if (nxpsg >= 2) {
            sum_gate[ir1] += simd_shuffle_down(sum_gate[ir1],  1);
            sum_up[ir1]   += simd_shuffle_down(sum_up[ir1],    1);
        }
    }

    if (tx == 0 && i01 < args.out_dim) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.n_rows; ++ir1) {
            const uint64_t dst_base =
                (uint64_t)i1m*args.out_dim*args.n_rows + (uint64_t)(i11 + ir1)*args.out_dim;
            device float * gate_f32 = (device float *)dst_gate + dst_base;
            device float * up_f32   = (device float *)dst_up   + dst_base;
            device float * mid_f32  = (device float *)dst_mid  + dst_base;

            const float gate = sum_gate[ir1];
            const float up = sum_up[ir1];
            gate_f32[i01] = gate;
            up_f32[i01] = up;

            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[i01] = silu * u;
        }
    }
}

typedef decltype(glm53_affine_q8_pair_native<2>) glm53_affine_q8_pair_native_t;
template [[host_name("glm53_affine_q8_pair_native_r2")]] kernel glm53_affine_q8_pair_native_t glm53_affine_q8_pair_native<2>;
template [[host_name("glm53_affine_q8_pair_native_r3")]] kernel glm53_affine_q8_pair_native_t glm53_affine_q8_pair_native<3>;
template [[host_name("glm53_affine_q8_pair_native_r4")]] kernel glm53_affine_q8_pair_native_t glm53_affine_q8_pair_native<4>;
template [[host_name("glm53_affine_q8_pair_native_r5")]] kernel glm53_affine_q8_pair_native_t glm53_affine_q8_pair_native<5>;

#define FOR_UNROLL _Pragma("clang loop unroll(full)") for
kernel void glm53_affine_q8_pair_native_m1(
        constant glm53_affine_q8_ext_args &args [[buffer(0)]],
        device const uchar *src0_gate [[buffer(1)]],
        device const uchar *src0_up [[buffer(2)]],
        device const char *src1 [[buffer(3)]],
        device char *dst_gate [[buffer(4)]],
        device char *dst_up [[buffer(5)]],
        device char *dst_mid [[buffer(6)]],
        constant float &clamp_value [[buffer(7)]],
        device const uchar *gate_scales [[buffer(8)]],
        device const uchar *gate_biases [[buffer(9)]],
        device const uchar *up_scales [[buffer(10)]],
        device const uchar *up_biases [[buffer(11)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 4;
    constexpr short NR0 = 2;
    constexpr short NW = 32;
    constexpr short NQ = 8;

    const int nb = args.in_dim / 32;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    device const float *y = (device const float *)(src1 + ulong(r1) * args.in_dim * sizeof(float));

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * 32 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const uint row_index = min(uint(r0 + row), args.out_dim - 1u);
            const ulong affine = (ulong)row_index * (args.in_dim / 64u) + uint(ib) / 2u;
            const float dg = glm53_q8_ext_metadata(gate_scales, affine, args.metadata_bf16 != 0u);
            const float du = glm53_q8_ext_metadata(up_scales, affine, args.metadata_bf16 != 0u);
            const float residual_g = fma(128.0f, dg, glm53_q8_ext_metadata(gate_biases, affine, args.metadata_bf16 != 0u));
            const float residual_u = fma(128.0f, du, glm53_q8_ext_metadata(up_biases, affine, args.metadata_bf16 != 0u));
            device const uchar *qg = src0_gate + (ulong)row_index * args.in_dim + ib * 32u + il * NQ;
            device const uchar *qu = src0_up + (ulong)row_index * args.in_dim + ib * 32u + il * NQ;
            float sg = 0.f;
            float su = 0.f;
            float sy = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += (float(qg[i]) - 128.0f) * yl[i];
                su += (float(qu[i]) - 128.0f) * yl[i];
                sy += yl[i];
            }
            sumg[row] += sg * dg;
            if (residual_g != 0.0f) sumg[row] += sy * residual_g;
            sumu[row] += su * du;
            if (residual_u != 0.0f) sumu[row] += sy * residual_u;
        }

        yb += NSG * NQ * 32;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *gate_f32 = (device float *)dst_gate +
        (uint64_t)im * args.out_dim * args.n_rows + (uint64_t)r1 * args.out_dim;
    device float *up_f32 = (device float *)dst_up +
        (uint64_t)im * args.out_dim * args.n_rows + (uint64_t)r1 * args.out_dim;
    device float *mid_f32 = (device float *)dst_mid +
        (uint64_t)im * args.out_dim * args.n_rows + (uint64_t)r1 * args.out_dim;

    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.out_dim; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const uint out_row = r0 + row;
            {
                gate_f32[out_row] = gate;
                up_f32[out_row] = up;
            }
            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u;
        }
    }
}
