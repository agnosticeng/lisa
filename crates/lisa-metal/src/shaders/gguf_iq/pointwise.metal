// Verbatim DS4 kernel sections; see provenance.json and LICENSE.
#include <metal_stdlib>
using namespace metal;

// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}
static inline float4 dsv41_bf16(float4 v) {
    return float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_router {
    uint n_expert, top_k, has_bias;
    float scale;
};

// One-token router for up to 1024 experts in one dispatch: the softplus/sqrt
// probability transform, the biased top-k through the same bitonic network as
// kernel_argsort_f32_i32_desc (padding indices sort last), and the weights
// normalised as the generic chain does: gather, SIMD sum over the top_k lanes,
// clamp, divide, scale. Threadgroup memory holds nth ints and 2 * nth floats.
kernel void kernel_dsv41_router_one(
        constant ds4_metal_args_dsv41_router &args,
        device const float *logits,
        device const float *bias,
        device float *probs,
        device int32_t *selected,
        device float *weights,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 tpos [[thread_position_in_threadgroup]],
        uint2 tsize [[threads_per_threadgroup]]) {
    const uint col = tpos.x, ntg = tsize.x;
    const int width = (int)args.n_expert;
    logits += tg.y * args.n_expert;   /* one threadgroup per row */
    probs += tg.y * args.n_expert;
    selected += tg.y * args.top_k;
    weights += tg.y * args.top_k;
    threadgroup float *score = (threadgroup float *)(shmem_i32 + ntg);
    threadgroup float *prob = score + ntg;
    shmem_i32[col] = (int)col;
    if ((int)col < width) {
        const float x = logits[col];
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(sp);
        probs[col] = p;
        prob[col] = p;
        score[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = (int)col ^ j;
            if (ixj > (int)col) {
                const int a = shmem_i32[col], b = shmem_i32[ixj];
                const bool swap = ((int)col & k) == 0 ?
                    (a >= width || (b < width && score[a] < score[b])) :
                    (b >= width || (a < width && score[a] > score[b]));
                if (swap) { shmem_i32[col] = b; shmem_i32[ixj] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < args.top_k) {
        const int32_t idx = shmem_i32[col];
        selected[col] = idx;
        const float w = prob[idx];
        const float total = clamp(simd_sum(w), 6.103515625e-5f, INFINITY);
        // two stores as in the generic chain: the quotient is rounded before the scale
        threadgroup volatile float *quotient = (threadgroup volatile float *)(prob + ntg) + col;
        *quotient = w / total;
        weights[col] = *quotient * args.scale;
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
    uint round_all;
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong head = ((ulong)group.y * args.heads + group.x) * args.width;
    if (args.round_all) {
        for (uint k = lane; k + 64u < args.width; k += 32u) x[head + k] = dsv41_bf16(x[head + k]);
    }
    const ulong i = head + args.width - 64u + 2u * lane;
    float re = x[i], im = x[i + 1u];
    if (args.round_all) { re = dsv41_bf16(re); im = dsv41_bf16(im); }
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}


struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

/* A single product, rounded as written: fast math may not reassociate it into a neighbouring product. */
static inline float dsv41_engram_mul(float a, float b) {
#pragma METAL fp math_mode(safe)
    return a * b;
}

/* kernel_dsv41_engram_add over eight SIMD groups per stream (width 5120): each lane's chain runs in order, segment by
 * segment, handing its three accumulators on, with the fma contractions the 32-lane kernel compiles to written out. */
kernel void kernel_dsv41_engram_add_tg256(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        threadgroup float *sh [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (args.masked && !mask[group.x]) return;
    constexpr uint W = 5120u, J = W / 256u;
    const uint e0 = sg * 32u * J + lane;
    device float *hp = residual + ((ulong)group.x * 4u + group.y) * W + e0;
    device const float *kp = kv + ((ulong)group.x * 5u + group.y) * W + e0;
    device const float *vp = kv + ((ulong)group.x * 5u + 4u) * W + e0;
    device const float *qp = q_weight + (ulong)group.y * W + e0;
    device const float *wp = k_weight + (ulong)group.y * W + e0;
    float h[J], k[J], p[J], v[J];
    for (uint j = 0; j < J; j++) {
        h[j] = hp[32u * j];
        k[j] = dsv41_bf16(kp[32u * j]);
        p[j] = dsv41_engram_mul(dsv41_engram_mul(h[j], qp[32u * j]), wp[32u * j]);
        v[j] = vp[32u * j];
    }
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint s = 0; s < 8u; s++) {
        if (sg == s) {
            if (s) { h2 = sh[lane]; k2 = sh[32u + lane]; dot = sh[64u + lane]; }
            for (uint j = 0; j < J; j++) {
                h2 = fma(h[j], h[j], h2);
                k2 = fma(k[j], k[j], k2);
                dot = fma(p[j], k[j], dot);
            }
            if (s < 7u) { sh[lane] = h2; sh[32u + lane] = k2; sh[64u + lane] = dot; }
        }
        if (s < 7u) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 7u) {
        h2 = simd_sum(h2);
        k2 = simd_sum(k2);
        dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
              rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
        const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
        if (lane == 0u) sh[96] = gate;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = sh[96];
    for (uint j = 0; j < J; j++)
        hp[32u * j] = dsv41_bf16(fma(gate, dsv41_bf16(v[j]), h[j]));
}

