// Adapted from OVHai-LLM/llama.cpp 55c22dca0c987f247c21b427c3b3a09e1a28187f.
// MIT; see LICENSE and provenance.json. Kernel arithmetic is unchanged.
#include <metal_stdlib>
using namespace metal;
#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define OP_MAMBA3_MIMO_NTH 256
#define OP_MAMBA3_MIMO_NSG 8
#define OP_MAMBA3_MIMO_CHUNK 32
#define OP_MAMBA3_MIMO_MAX_R 8
typedef struct {
    int32_t D_qk;
    int32_t R;
    int32_t D_v;
    int32_t H;
    int32_t T;            // tokens per sequence
    int32_t n_seqs;
    int32_t n_chunks;     // chunked path only (1 otherwise)
    int32_t pad0;
    int64_t pdyn_s;       // token stride of pdyn
    int64_t pstat_s;      // token stride of pstat
    int64_t s_in_s;       // seq stride of the input state
    int64_t s_out_s;      // seq stride of the output state (n_embd_s, or the cache row stride when fused)
    int64_t off_K;        // packed state section offsets: [S | k_prev | v_prev | ang]
    int64_t off_V;
    int64_t off_A;
    int64_t ws_state_off; // chunked path workspace: [ang | chunk states | chunk decays]
    int64_t ws_decay_off;
    float   eps;
    float   a_floor;
} ggml_metal_kargs_mamba3_mimo;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    inv_depth;
} ggml_metal_kargs_geodesic;

// Olala fused ops: GGML_OP_MAMBA3_MIMO (Mamba3-MIMO mixer core, raw
// projections in) and GGML_OP_GEODESIC (geodesic residual).
//
// semantic reference: ggml_compute_forward_mamba3_mimo_f32 / _geodesic_f32 in
// ggml-cpu/ops.cpp; the design mirrors ggml-cuda/mamba3-mimo.cu.
//
// MAMBA3_MIMO: one threadgroup of M3_NTH = 256 threads (8 simdgroups) per
// (chunk, head, seq). Each token step builds its operands in threadgroup memory
// from the raw projections (B/C rms-norm + weight + bias + rotary, x/z MIMO
// scaling, softplus/exp/sigmoid discretization), then updates the state and,
// when producing outputs, applies the D skip, the SiLU(z) gate and the MIMO
// output combine. Simdgroup w owns the state rows p = w, w + 8, ...; lane l owns
// the columns d = l, l + 32, ..., so the state update and the q.S contraction
// are per thread (one simd_sum per (row, rank)) and need no extra barriers.
//
// The per-head state (D_v*D_qk floats, 32 KB for Olala) does not fit next to
// the staging buffers in the 32 KB of threadgroup memory, so:
//   - fast path (template dims, Olala: D_qk = 128, D_v = 64, R = 4): the state
//     lives in registers (D_v/8 x D_qk/32 = 32 floats per thread)
//   - generic path (DQK == 0, runtime dims): the state lives in device memory
//     (output state slot / chunk workspace), each element owned by one thread
//
// short ubatches (T < 2*M3_CHUNK tokens per sequence, e.g. decode): one serial
// kernel. long ubatches: the recurrence is affine in S, so with chunks of
// M3_CHUNK tokens S_end(n) = P_n S_end(n-1) + L_n (P_n the product of the
// chunk's alphas, L_n the chunk-local state run from S = 0):
//   angles:  per (head, seq): serial phase cumsum -> per-token angle buffer
//   phase 1: per (chunk, head, seq): L_n and P_n                 (parallel)
//   phase 2: per (state element, head, seq): scan over chunks, leaves the
//            chunk carry-ins in the workspace, writes the final state
//   phase 3: per (chunk, head, seq): outputs from the carry-in   (parallel)

#define M3_NTH     OP_MAMBA3_MIMO_NTH
#define M3_NSG     OP_MAMBA3_MIMO_NSG
#define M3_MAX_R   OP_MAMBA3_MIMO_MAX_R
#define M3_CHUNK   OP_MAMBA3_MIMO_CHUNK

#define M3_MODE_SERIAL 0
#define M3_MODE_PHASE1 1
#define M3_MODE_PHASE3 2

#define M3_PI      3.14159265358979f
#define M3_TWO_PI  6.283185307179586f
#define M3_INV_2PI 0.15915494309189535f

static inline float m3_softplus(float x) {
    return fmax(x, 0.0f) + precise::log(precise::exp(-fabs(x)) + 1.0f);
}

static inline float m3_wrap(float a) {
    return a - M3_TWO_PI*rint(a*M3_INV_2PI);
}

// rms scales of the first n_groups D_qk-groups of the pdyn row pd (one simdgroup per group)
static inline void m3_rms(
        device const float * pd,
        threadgroup  float * rms,
        int    n_groups,
        int    D_qk,
        float  eps,
        ushort sgitg,
        ushort tiisg) {
    for (int g = sgitg; g < n_groups; g += M3_NSG) {
        float s = 0.0f;
        for (int d = tiisg; d < D_qk; d += N_SIMDWIDTH) {
            const float x = pd[g*D_qk + d];
            s += x*x;
        }
        s = simd_sum(s);
        if (tiisg == 0) {
            rms[g] = 1.0f/sqrt(s/D_qk + eps);
        }
    }
}

// builds k (and q, gz if want_qz) of one token into threadgroup memory from the raw
// projections; cs/sn must hold the token's rotary cos/sin, rms the 2R (R) scales.
// gz holds the output gate mimo_o*silu(z) instead of z itself.
// v is always built. The caller puts barriers before and after.
template<short DQK, short DV, short RR>
static inline void m3_build_operands(
        constant ggml_metal_kargs_mamba3_mimo & args,
        device const float * pd,      // pdyn row of the token
        device const float * ps,      // pstat segment of (token, head): [z | x | dt | A | trap]
        device const float * bias_h,  // (D_qk, 2R) of the head: [b_bias | c_bias]
        device const float * mx_h,    // (D_v,  2R) of the head: [mimo_x | mimo_z]
        device const float * mo_h,    // (D_v,  R)  of the head
        device const float * norms,   // (D_qk, 2): [w_b | w_c]
        threadgroup const float * cs,
        threadgroup const float * sn,
        threadgroup const float * rms,
        threadgroup float * k,
        threadgroup float * q,
        threadgroup float * v,
        threadgroup float * gz,
        bool   want_qz,
        ushort tiitg) {
    const int D_qk = DQK > 0 ? DQK : args.D_qk;
    const int D_v  = DQK > 0 ? DV  : args.D_v;
    const int R    = DQK > 0 ? RR  : args.R;

    const int hf      = D_qk/2;
    const int quarter = D_qk/4;

    const int n_kq = (want_qz ? 2 : 1)*R*D_qk;
    for (int j = tiitg; j < n_kq; j += M3_NTH) {
        const int g = j / D_qk;
        const int d = j % D_qk;

        device const float * src = pd + g*D_qk;
        device const float * w   = norms + (g < R ? 0 : D_qk);
        device const float * b   = bias_h + g*D_qk;

        const float sc = rms[g];

        float out;
        if (d < quarter) {
            const float v0 = src[d       ]*sc*w[d       ] + b[d       ];
            const float v2 = src[d + hf]*sc*w[d + hf] + b[d + hf];
            out = v0*cs[d] - v2*sn[d];
        } else if (d >= hf && d < hf + quarter) {
            const int i = d - hf;
            const float v0 = src[i]*sc*w[i] + b[i];
            const float v2 = src[d]*sc*w[d] + b[d];
            out = v0*sn[i] + v2*cs[i];
        } else {
            out = src[d]*sc*w[d] + b[d];
        }

        if (g < R) {
            k[j] = out;
        } else {
            q[j - R*D_qk] = out;
        }
    }

    for (int j = tiitg; j < R*D_v; j += M3_NTH) {
        const int p = j % D_v;
        v[j] = ps[D_v + p]*mx_h[j];
        if (want_qz) {
            const float zv = ps[p]*mx_h[R*D_v + j];
            gz[j] = mo_h[j]*(zv/(1.0f + precise::exp(-zv)));
        }
    }
}

// MODE: M3_MODE_SERIAL / M3_MODE_PHASE1 / M3_MODE_PHASE3
// DQK, DV, RR: compile-time dims of the register-state fast path, DQK == 0 selects the
// generic path (runtime dims, state in device memory)
template<short MODE, short DQK, short DV, short RR>
kernel void kernel_mamba3_mimo_impl(
        constant ggml_metal_kargs_mamba3_mimo & args,
        device const float * pdyn,
        device const float * pstat,
        device const float * bias,
        device const float * mxz,
        device const float * mimo_o,
        device const float * norms,
        device const float * misc,
        device const float * s_in,
        device       float * y,
        device       float * s_out,
        device       float * ws,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr bool FAST = DQK > 0;

    // fast path register tiling: NP rows per simdgroup, ND columns per lane
    constexpr short NP = FAST ? DV/M3_NSG      : 1;
    constexpr short ND = FAST ? DQK/N_SIMDWIDTH : 1;
    constexpr short NR = FAST ? RR              : 1;

    static_assert(!FAST || (DV % M3_NSG == 0 && DQK % N_SIMDWIDTH == 0 && DQK % 4 == 0), "invalid fast-path dims");

    constexpr bool want_y = MODE != M3_MODE_PHASE1;

    const int D_qk  = FAST ? DQK : args.D_qk;
    const int D_v   = FAST ? DV  : args.D_v;
    const int R     = FAST ? RR  : args.R;
    const int H     = args.H;
    const int n_ang = D_qk/4;
    const int SD    = D_qk*D_v;

    const int nc  = tgpig.x;
    const int h   = tgpig.y;
    const int seq = tgpig.z;

    const int t0 = MODE == M3_MODE_SERIAL ? 0      : nc*M3_CHUNK;
    const int tc = MODE == M3_MODE_SERIAL ? args.T : min(M3_CHUNK, args.T - t0);

    const int64_t tok0 = (int64_t) seq*args.T;

    device const float * s_in_q  = s_in  + seq*args.s_in_s;
    device       float * s_out_q = s_out + seq*args.s_out_s;

    // chunked-path workspace: [ang (n_ang, H, n_tok) | state (SD, n_chunks, H, n_seqs) | decay (n_chunks, H, n_seqs)]
    device const float * ws_ang   = ws;
    device       float * ws_state = ws + args.ws_state_off;
    device       float * ws_decay = ws + args.ws_decay_off;

    device float * W = ws_state + (((int64_t) seq*H + h)*args.n_chunks + nc)*SD;

    // threadgroup: [ k x2 | q | v x2 | gz | cs | sn | ang | rms | coef ]
    threadgroup float * kb  = shmem;
    threadgroup float * qb  = kb  + 2*R*D_qk;
    threadgroup float * vb  = qb  + R*D_qk;
    threadgroup float * zb  = vb  + 2*R*D_v;
    threadgroup float * cs  = zb  + R*D_v;
    threadgroup float * sn  = cs  + n_ang;
    threadgroup float * ang = sn  + n_ang;
    threadgroup float * rms = ang + n_ang;
    threadgroup float * cfb = rms + 2*R;

    device const float * bias_h = bias   + h*2*R*D_qk;
    device const float * mx_h   = mxz    + h*2*R*D_v;
    device const float * mo_h   = mimo_o + h*R*D_v;

    const float dt_bias = misc[h];
    const float D_h     = misc[H + h];

    // generic path state (device memory): the output state slot (serial) or the chunk workspace
    device float * S = MODE == M3_MODE_SERIAL ? s_out_q + h*SD : W;

    // fast path state (registers)
    float st[NP][ND];

    if (FAST) {
        FOR_UNROLL (short j = 0; j < NP; ++j) {
            const int p = sgitg + M3_NSG*j;
            FOR_UNROLL (short i = 0; i < ND; ++i) {
                const int d = tiisg + N_SIMDWIDTH*i;
                if (MODE == M3_MODE_SERIAL) {
                    st[j][i] = s_in_q[h*SD + p*D_qk + d];
                } else if (MODE == M3_MODE_PHASE1) {
                    st[j][i] = 0.0f;
                } else {
                    st[j][i] = W[p*D_qk + d]; // carry-in left by phase 2
                }
            }
        }
    } else {
        // each element is owned by the thread that updates it below: no barrier needed
        if (MODE != M3_MODE_PHASE3) {
            for (int p = sgitg; p < D_v; p += M3_NSG) {
                for (int d = tiisg; d < D_qk; d += N_SIMDWIDTH) {
                    S[p*D_qk + d] = MODE == M3_MODE_SERIAL ? s_in_q[h*SD + p*D_qk + d] : 0.0f;
                }
            }
        }
    }

    // rotary phases (serial path): owned by the lanes of simdgroup 0
    if (MODE == M3_MODE_SERIAL && sgitg == 0) {
        for (int i = tiisg; i < n_ang; i += N_SIMDWIDTH) {
            ang[i] = s_in_q[args.off_A + h*n_ang + i];
        }
    }

    // previous token's k/v: from the cache at the sequence start, else rebuilt
    short prv = 1;
    if (t0 == 0) {
        for (int j = tiitg; j < R*D_qk; j += M3_NTH) {
            kb[prv*R*D_qk + j] = s_in_q[args.off_K + h*R*D_qk + j];
        }
        for (int j = tiitg; j < R*D_v; j += M3_NTH) {
            vb[prv*R*D_v + j] = s_in_q[args.off_V + h*R*D_v + j];
        }
    } else {
        const int64_t ptok = tok0 + t0 - 1;

        device const float * pd = pdyn  + ptok*args.pdyn_s;
        device const float * ps = pstat + ptok*args.pstat_s + h*(2*D_v + 3);

        for (int i = tiitg; i < n_ang; i += M3_NTH) {
            const float a = ws_ang[(ptok*H + h)*n_ang + i];
            cs[i] = precise::cos(a);
            sn[i] = precise::sin(a);
        }
        m3_rms(pd, rms, R, D_qk, args.eps, sgitg, tiisg);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        m3_build_operands<DQK, DV, RR>(args, pd, ps, bias_h, mx_h, mo_h, norms, cs, sn, rms,
                kb + prv*R*D_qk, qb, vb + prv*R*D_v, zb, false, tiitg);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pa = 1.0f; // running alpha product (phase 1)

    for (int t = 0; t < tc; ++t) {
        const int64_t tok = tok0 + t0 + t;
        const short   cur = prv ^ 1;

        device const float * pd = pdyn  + tok*args.pdyn_s;
        device const float * ps = pstat + tok*args.pstat_s + h*(2*D_v + 3);

        // token coefficients + rotary phases: simdgroup 0, published through threadgroup memory
        //
        // note: the coef/cs/sn/rms buffers are overwritten here while other simdgroups may still be
        //       in the state update of the previous token - this is safe because those buffers are
        //       only read between the two barriers below (the coefs are copied to registers)
        if (sgitg == 0) {
            const float dt    = m3_softplus(ps[2*D_v + 0] + dt_bias);
            const float a     = fmin(-m3_softplus(ps[2*D_v + 1]), -args.a_floor);
            const float alpha = precise::exp(a*dt);
            const float trap  = 1.0f/(1.0f + precise::exp(-ps[2*D_v + 2]));

            if (tiisg == 0) {
                cfb[0] = alpha;
                cfb[1] = (1.0f - trap)*dt*alpha; // beta
                cfb[2] = trap*dt;                // gamma
            }

            for (int i = tiisg; i < n_ang; i += N_SIMDWIDTH) {
                float a_i;
                if (MODE == M3_MODE_SERIAL) {
                    a_i = m3_wrap(ang[i] + precise::tanh(pd[2*R*D_qk + i])*M3_PI*dt);
                    ang[i] = a_i;
                } else {
                    a_i = ws_ang[(tok*H + h)*n_ang + i];
                }
                cs[i] = precise::cos(a_i);
                sn[i] = precise::sin(a_i);
            }
        }
        m3_rms(pd, rms, want_y ? 2*R : R, D_qk, args.eps, sgitg, tiisg);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float alpha = cfb[0];
        const float beta  = cfb[1];
        const float gamma = cfb[2];

        if (MODE == M3_MODE_PHASE1) {
            pa *= alpha;
        }

        threadgroup float * k_c = kb + cur*R*D_qk;
        threadgroup float * v_c = vb + cur*R*D_v;
        threadgroup float * k_p = kb + prv*R*D_qk;
        threadgroup float * v_p = vb + prv*R*D_v;

        m3_build_operands<DQK, DV, RR>(args, pd, ps, bias_h, mx_h, mo_h, norms, cs, sn, rms,
                k_c, qb, v_c, zb, want_y, tiitg);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * y_t = y + (tok*H + h)*D_v;

        if (FAST) {
            // per-lane operand columns, with beta/gamma folded in
            float kcg[NR][ND];
            float kpb[NR][ND];
            float qq [NR][ND];

            FOR_UNROLL (short r = 0; r < NR; ++r) {
                FOR_UNROLL (short i = 0; i < ND; ++i) {
                    const int d = tiisg + N_SIMDWIDTH*i;
                    kcg[r][i] = gamma*k_c[r*D_qk + d];
                    kpb[r][i] = beta *k_p[r*D_qk + d];
                    qq [r][i] = want_y ? qb[r*D_qk + d] : 0.0f;
                }
            }

            FOR_UNROLL (short j = 0; j < NP; ++j) {
                const int p = sgitg + M3_NSG*j;

                float vc [NR];
                float vp [NR];
                float acc[NR];

                FOR_UNROLL (short r = 0; r < NR; ++r) {
                    vc [r] = v_c[r*D_v + p];
                    vp [r] = v_p[r*D_v + p];
                    acc[r] = 0.0f;
                }

                FOR_UNROLL (short i = 0; i < ND; ++i) {
                    float s = alpha*st[j][i];
                    FOR_UNROLL (short r = 0; r < NR; ++r) {
                        s += vp[r]*kpb[r][i] + vc[r]*kcg[r][i];
                    }
                    st[j][i] = s;

                    if (want_y) {
                        FOR_UNROLL (short r = 0; r < NR; ++r) {
                            acc[r] += s*qq[r][i];
                        }
                    }
                }

                if (want_y) {
                    float yp = 0.0f;
                    FOR_UNROLL (short r = 0; r < NR; ++r) {
                        const float o = simd_sum(acc[r]) + D_h*vc[r];
                        yp += o*zb[r*D_v + p];
                    }
                    if (tiisg == 0) {
                        y_t[p] = yp;
                    }
                }
            }
        } else {
            for (int p = sgitg; p < D_v; p += M3_NSG) {
                float vc [M3_MAX_R];
                float vp [M3_MAX_R];
                float acc[M3_MAX_R];

                for (int r = 0; r < R; ++r) {
                    vc [r] = v_c[r*D_v + p];
                    vp [r] = v_p[r*D_v + p];
                    acc[r] = 0.0f;
                }

                device float * Sp = S + p*D_qk;

                for (int d = tiisg; d < D_qk; d += N_SIMDWIDTH) {
                    float c_acc = 0.0f;
                    float p_acc = 0.0f;
                    for (int r = 0; r < R; ++r) {
                        c_acc += vc[r]*k_c[r*D_qk + d];
                        p_acc += vp[r]*k_p[r*D_qk + d];
                    }
                    const float s = alpha*Sp[d] + beta*p_acc + gamma*c_acc;
                    Sp[d] = s;

                    if (want_y) {
                        for (int r = 0; r < R; ++r) {
                            acc[r] += s*qb[r*D_qk + d];
                        }
                    }
                }

                if (want_y) {
                    float yp = 0.0f;
                    for (int r = 0; r < R; ++r) {
                        const float o = simd_sum(acc[r]) + D_h*vc[r];
                        yp += o*zb[r*D_v + p];
                    }
                    if (tiisg == 0) {
                        y_t[p] = yp;
                    }
                }
            }
        }

        prv = cur;
    }

    // k/v of the last token feed the next ubatch's trapezoid term
    // (written by the barrier-separated build of the last step, so readable by any thread)
    const bool last = MODE == M3_MODE_SERIAL || (MODE == M3_MODE_PHASE3 && nc == args.n_chunks - 1);
    if (last) {
        for (int j = tiitg; j < R*D_qk; j += M3_NTH) {
            s_out_q[args.off_K + h*R*D_qk + j] = kb[prv*R*D_qk + j];
        }
        for (int j = tiitg; j < R*D_v; j += M3_NTH) {
            s_out_q[args.off_V + h*R*D_v + j] = vb[prv*R*D_v + j];
        }
    }

    if (MODE == M3_MODE_SERIAL && sgitg == 0) {
        for (int i = tiisg; i < n_ang; i += N_SIMDWIDTH) {
            s_out_q[args.off_A + h*n_ang + i] = ang[i];
        }
    }

    if (FAST && MODE != M3_MODE_PHASE3) {
        // serial: final state; phase 1: chunk-local state
        device float * dst_s = MODE == M3_MODE_SERIAL ? s_out_q + h*SD : W;
        FOR_UNROLL (short j = 0; j < NP; ++j) {
            const int p = sgitg + M3_NSG*j;
            FOR_UNROLL (short i = 0; i < ND; ++i) {
                const int d = tiisg + N_SIMDWIDTH*i;
                dst_s[p*D_qk + d] = st[j][i];
            }
        }
    }

    if (MODE == M3_MODE_PHASE1 && tiitg == 0) {
        ws_decay[((int64_t) seq*H + h)*args.n_chunks + nc] = pa;
    }
}

typedef decltype(kernel_mamba3_mimo_impl<M3_MODE_SERIAL, 0, 0, 0>) kernel_mamba3_mimo_t;

template [[host_name("kernel_mamba3_mimo_serial_gen")]]        kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_SERIAL, 0,   0,  0>;
template [[host_name("kernel_mamba3_mimo_phase1_gen")]]        kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE1, 0,   0,  0>;
template [[host_name("kernel_mamba3_mimo_phase3_gen")]]        kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE3, 0,   0,  0>;
template [[host_name("kernel_mamba3_mimo_serial_d128_v64_r4")]] kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_SERIAL, 128, 64, 4>;
template [[host_name("kernel_mamba3_mimo_phase1_d128_v64_r4")]] kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE1, 128, 64, 4>;
template [[host_name("kernel_mamba3_mimo_phase3_d128_v64_r4")]] kernel kernel_mamba3_mimo_t kernel_mamba3_mimo_impl<M3_MODE_PHASE3, 128, 64, 4>;

kernel void kernel_geodesic_f32(
        constant ggml_metal_kargs_geodesic & args,
        device const char  * x,
        device const char  * g,
        device const float * scale,
        device const float * bias,
        device       char  * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short nsg = (ntg.x + N_SIMDWIDTH - 1)/N_SIMDWIDTH;

    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    const int n = args.ne00;

    device const float * xr = (device const float *) (x   + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device const float * gr = (device const float *) (g   + i1*args.nb11 + i2*args.nb12 + i3*args.nb13);
    device       float * yr = (device       float *) (dst + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    float xx = 0.0f;
    float xg = 0.0f;
    for (int i = tiitg; i < n; i += ntg.x) {
        const float xv = xr[i];
        xx += xv*xv;
        xg += xv*gr[i];
    }
    xx = simd_sum(xx);
    xg = simd_sum(xg);
    if (tiisg == 0) {
        shmem[sgitg]               = xx;
        shmem[N_SIMDWIDTH + sgitg] = xg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    xx = 0.0f;
    xg = 0.0f;
    for (short k = 0; k < nsg; ++k) {
        xx += shmem[k];
        xg += shmem[N_SIMDWIDTH + k];
    }

    const float x_norm_sq = fmax(xx, 1e-12f);
    const float coef      = xg/x_norm_sq;

    // |g - coef*x|^2 accumulated directly (the expanded form is cancellation-prone)
    float pp = 0.0f;
    for (int i = tiitg; i < n; i += ntg.x) {
        const float gp = gr[i] - coef*xr[i];
        pp += gp*gp;
    }
    pp = simd_sum(pp);

    // all threads have read the first partials before they are overwritten
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = pp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    pp = 0.0f;
    for (short k = 0; k < nsg; ++k) {
        pp += shmem[k];
    }

    const float pi4      = 0.785398163f;
    const float tan_norm = fmax(sqrt(pp), 1e-8f);
    const float xn       = sqrt(x_norm_sq);

    float theta = fmin(tan_norm/fmax(xn, 1e-6f), pi4);
    theta = fmin((theta*scale[0] + bias[0])*args.inv_depth, pi4);

    const float c = precise::cos(theta);
    const float s = precise::sin(theta)*xn/tan_norm;

    for (int i = tiitg; i < n; i += ntg.x) {
        const float xv = xr[i];
        yr[i] = xv*c + (gr[i] - coef*xv)*s;
    }
}
