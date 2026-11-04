#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}
static inline float4 dsv41_bf16(float4 v) {
    return float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
}

struct ds4_metal_args_glm_indexer_score_one {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t cache_f16;
    float    scale;
};


static inline float glm_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}
kernel void kernel_glm_indexer_score_one_direct(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_rows || args.n_head != 32u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;
    threadgroup float *psum = ktg + 128u;   /* two 4-float phases */

    if (tid < 128u) {
        ktg[tid] = glm_cache_load_f32_or_f16(indexer_key_cache,
                                             (uint64_t)row * 128u + tid,
                                             args.cache_f16);
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * 128u * sizeof(float));
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        /* Alternate the partial buffer so the next iteration cannot overwrite
         * the slots thread 0 is still reading. This removes the second
         * barrier while keeping the accumulation order identical. */
        threadgroup float *ps = psum + (head0 & 4u);
        if (lane == 0) {
            ps[sg] = max(s * args.scale, 0.0f) * weights[head];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += ps[0];
            acc += ps[1];
            acc += ps[2];
            acc += ps[3];
        }
    }

    if (tid == 0) {
        scores[row] = acc;
    }
}

/* Same math as kernel_glm_indexer_score_one_direct, but the row for each
 * output slot comes from a compact candidate list. Used by V4.1 layers whose
 * indexer candidate set is restricted by an earlier block selection. */
/* One SIMD group scores four key rows at a time. Every lane keeps its own
 * float4 of each row in a register, so a row never round-trips through
 * threadgroup memory; the query float4 is read once per head and reused by all
 * four rows, which is what keeps the per-row query traffic from growing with
 * the head count. Per-row math is unchanged (32-lane float4 dot, then
 * simd_sum, then max(s*scale,0)*w[head] accumulated on lane 0 in head order),
 * so the scores are bit-identical to kernel_glm_indexer_score_one_direct. */
static inline float4 glm_indexer_cache_row4(
        device const char *indexer_key_cache,
        uint row,
        ushort lane,
        uint cache_f16) {
    const uint64_t idx = (uint64_t)row * 128u + (uint64_t)lane * 4u;
    if (cache_f16 != 0u) {
        return float4(glm_cache_load_f32_or_f16(indexer_key_cache, idx + 0u, cache_f16),
                      glm_cache_load_f32_or_f16(indexer_key_cache, idx + 1u, cache_f16),
                      glm_cache_load_f32_or_f16(indexer_key_cache, idx + 2u, cache_f16),
                      glm_cache_load_f32_or_f16(indexer_key_cache, idx + 3u, cache_f16));
    }
    return *(device const float4 *)((device const float *)indexer_key_cache + idx);
}

static inline float4 glm_indexer_score_rows_in_simd(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const float *q,
        device const float *weights,
        device const char *indexer_key_cache,
        uint r0, uint r1, uint r2, uint r3,
        ushort lane) {
    const float4 kv0 = glm_indexer_cache_row4(indexer_key_cache, r0, lane, args.cache_f16);
    const float4 kv1 = glm_indexer_cache_row4(indexer_key_cache, r1, lane, args.cache_f16);
    const float4 kv2 = glm_indexer_cache_row4(indexer_key_cache, r2, lane, args.cache_f16);
    const float4 kv3 = glm_indexer_cache_row4(indexer_key_cache, r3, lane, args.cache_f16);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    /* The scalar kernel rounds max(s*scale,0)*w into threadgroup memory and
     * then adds it, so its term is rounded twice. Contracting that add into an
     * fma would remove a rounding and change the scores, and this library is
     * built with fast math. Keep the contraction off around the accumulate
     * only, so the dot and the reduction compile exactly as in the scalar
     * kernel. The pragma has to start a compound statement. */
    for (uint head = 0; head < 32u; head++) {
        device const float4 *q4 = (device const float4 *)(q + (uint64_t)head * 128u);
        float s0 = dot(q4[lane], kv0);
        float s1 = dot(q4[lane], kv1);
        float s2 = dot(q4[lane], kv2);
        float s3 = dot(q4[lane], kv3);
        s0 = simd_sum(s0);
        s1 = simd_sum(s1);
        s2 = simd_sum(s2);
        s3 = simd_sum(s3);
        if (lane == 0u) {
            const float t0 = max(s0 * args.scale, 0.0f) * weights[head];
            const float t1 = max(s1 * args.scale, 0.0f) * weights[head];
            const float t2 = max(s2 * args.scale, 0.0f) * weights[head];
            const float t3 = max(s3 * args.scale, 0.0f) * weights[head];
            {
#pragma clang fp contract(off)
                a0 += t0;
                a1 += t1;
                a2 += t2;
                a3 += t3;
            }
        }
    }
    return float4(a0, a1, a2, a3);
}

/* Grid: ceil(n_rows / 16); the threadgroup is 32x4, so each SIMD group covers
 * four rows and a threadgroup covers sixteen. */
static inline void glm_indexer_score_four_rows(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const float *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        uint first, ushort lane) {
    const uint r0 = first, r1 = first + 1u, r2 = first + 2u, r3 = first + 3u;
    /* Rows past the end fall back to row 0 so the cache load stays in bounds;
     * their results are dropped instead of being written. n_rows >= 1. */
    const float4 acc = glm_indexer_score_rows_in_simd(args, q, weights, indexer_key_cache,
            r0 < args.n_rows ? r0 : 0u, r1 < args.n_rows ? r1 : 0u,
            r2 < args.n_rows ? r2 : 0u, r3 < args.n_rows ? r3 : 0u, lane);
    if (lane == 0u) {
        if (r0 < args.n_rows) scores[r0] = acc.x;
        if (r1 < args.n_rows) scores[r1] = acc.y;
        if (r2 < args.n_rows) scores[r2] = acc.z;
        if (r3 < args.n_rows) scores[r3] = acc.w;
    }
}

kernel void kernel_glm_indexer_score_one_rows4(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const float *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        uint tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;
    glm_indexer_score_four_rows(args, q, weights, indexer_key_cache, scores, tg * 16u + sg * 4u, lane);
}


struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}


struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
    uint mask_stride;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * args.mask_stride + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

/* The 2048 candidate blocks of a 0/-inf mask in ascending order; one
 * threadgroup of 1024 threads, four mask entries per thread and step. */

struct ds4_metal_args_dsv41_index_keys_bf16 {
    uint32_t n_vec;
    uint32_t check;
};


kernel void kernel_glm_indexer_score_one_wide(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        uint3 ntg_u [[threads_per_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;
    const uint ntg = ntg_u.x;
    threadgroup float *qs = shared;           /* [32][128] */
    threadgroup float *ws = shared + 4096u;   /* [32] */
    device const float *qf = (device const float *)q;
    for (uint i = tid; i < 4096u; i += ntg) qs[i] = qf[i];
    if (tid < 32u) ws[tid] = weights[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tgpig.x * ntg + tid;
    if (row >= args.n_rows) return;
    float acc[32];
    FOR_UNROLL (uint h = 0; h < 32u; h++) acc[h] = 0.0f;
    const uint64_t base = (uint64_t)row * 128u;
    for (uint d0 = 0; d0 < 128u; d0 += 16u) {
        float4 k[4];
        if (args.cache_f16 != 0u) {
            device const half4 *kh = (device const half4 *)indexer_key_cache + ((base + d0) >> 2);
            FOR_UNROLL (uint i = 0; i < 4u; i++) k[i] = float4(kh[i]);
        } else {
            device const float4 *kf = (device const float4 *)indexer_key_cache + ((base + d0) >> 2);
            FOR_UNROLL (uint i = 0; i < 4u; i++) k[i] = kf[i];
        }
        FOR_UNROLL (uint h = 0; h < 32u; h++) {
            threadgroup const float4 *q4 = (threadgroup const float4 *)(qs + h * 128u + d0);
            acc[h] += dot(q4[0], k[0]) + dot(q4[1], k[1]) + dot(q4[2], k[2]) + dot(q4[3], k[3]);
        }
    }
    float score = 0.0f;
    FOR_UNROLL (uint h = 0; h < 32u; h++) {
#pragma clang fp contract(off)
        score += max(acc[h] * args.scale, 0.0f) * ws[h];
    }
    scores[row] = score;
}

/* Index keys as bf16 (FP4 values on a power-of-two scale are bf16-exact); with check, a key that is not raises *flag. */
kernel void kernel_dsv41_index_keys_bf16(
        constant ds4_metal_args_dsv41_index_keys_bf16 & args,
        device const uint4 *src,
        device ushort4 *dst,
        device atomic_uint *flag,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_vec) return;
    const uint4 b = src[gid];
    dst[gid] = ushort4(b >> 16);
    if (args.check != 0u && any((b & 0xffffu) != 0u))
        atomic_fetch_or_explicit(flag, 1u, memory_order_relaxed);
}

/* The wide scorer over the bf16 keys, two keys per thread: each dot and the head sum (0..31) in its order, its bits. */
kernel void kernel_dsv41_indexer_score_bf16(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const float *qf,
        device const float *weights,
        device const ushort4 *keys,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        uint3 ntg_u [[threads_per_threadgroup]]) {
    const uint ntg = ntg_u.x;
    threadgroup float *qs = shared;           /* [32][128] */
    threadgroup float *ws = shared + 4096u;   /* [32] */
    for (uint i = tid; i < 4096u; i += ntg) qs[i] = qf[i];
    if (tid < 32u) ws[tid] = weights[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (args.n_head != 32u || args.head_dim != 128u) return;

    const uint first = tgpig.x * ntg * 2u + tid;
    /* the tail's second key rereads the last row and is not written */
    const uint row[2] = {min(first, args.n_rows - 1u), min(first + ntg, args.n_rows - 1u)};
    float score[2] = {0.0f, 0.0f};
    for (uint g = 0; g < 32u; g += 16u) {
        float acc[2][16];
        FOR_UNROLL (uint r = 0; r < 2u; r++) FOR_UNROLL (uint h = 0; h < 16u; h++) acc[r][h] = 0.0f;
        for (uint d0 = 0; d0 < 128u; d0 += 16u) {
            float4 k[2][4];
            FOR_UNROLL (uint r = 0; r < 2u; r++) {
                device const ushort4 *kb = keys + (((uint64_t)row[r] * 128u + d0) >> 2);
                FOR_UNROLL (uint i = 0; i < 4u; i++) k[r][i] = as_type<float4>(uint4(kb[i]) << 16);
            }
            FOR_UNROLL (uint h = 0; h < 16u; h++) {
                threadgroup const float4 *q4 = (threadgroup const float4 *)(qs + (g + h) * 128u + d0);
                FOR_UNROLL (uint r = 0; r < 2u; r++)
                    acc[r][h] += dot(q4[0], k[r][0]) + dot(q4[1], k[r][1]) + dot(q4[2], k[r][2]) + dot(q4[3], k[r][3]);
            }
        }
        FOR_UNROLL (uint r = 0; r < 2u; r++) FOR_UNROLL (uint h = 0; h < 16u; h++) {
#pragma clang fp contract(off)
            score[r] += max(acc[r][h] * args.scale, 0.0f) * ws[g + h];
        }
    }
    if (first < args.n_rows) scores[first] = score[0];
    if (first + ntg < args.n_rows) scores[first + ntg] = score[1];
}

/* The wide kernel over V4.1 candidate keys: key i is position i % 8 of block
 * list[i / 8], so the scores keep ascending key order. */
kernel void kernel_dsv41_indexer_score_candidates(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        device const uint *list,
        threadgroup float *shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        uint3 ntg_u [[threads_per_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;
    const uint ntg = ntg_u.x;
    threadgroup float *qs = shared;           /* [32][128] */
    threadgroup float *ws = shared + 4096u;   /* [32] */
    device const float *qf = (device const float *)q;
    for (uint i = tid; i < 4096u; i += ntg) qs[i] = qf[i];
    if (tid < 32u) ws[tid] = weights[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint slot = tgpig.x * ntg + tid;
    if (slot >= args.n_rows) return;
    const uint row = list[slot >> 3] * 8u + (slot & 7u);
    float acc[32];
    FOR_UNROLL (uint h = 0; h < 32u; h++) acc[h] = 0.0f;
    const uint64_t base = (uint64_t)row * 128u;
    for (uint d0 = 0; d0 < 128u; d0 += 16u) {
        float4 k[4];
        if (args.cache_f16 != 0u) {
            device const half4 *kh = (device const half4 *)indexer_key_cache + ((base + d0) >> 2);
            FOR_UNROLL (uint i = 0; i < 4u; i++) k[i] = float4(kh[i]);
        } else {
            device const float4 *kf = (device const float4 *)indexer_key_cache + ((base + d0) >> 2);
            FOR_UNROLL (uint i = 0; i < 4u; i++) k[i] = kf[i];
        }
        FOR_UNROLL (uint h = 0; h < 32u; h++) {
            threadgroup const float4 *q4 = (threadgroup const float4 *)(qs + h * 128u + d0);
            acc[h] += dot(q4[0], k[0]) + dot(q4[1], k[1]) + dot(q4[2], k[2]) + dot(q4[3], k[3]);
        }
    }
    float score = 0.0f;
    FOR_UNROLL (uint h = 0; h < 32u; h++) {
#pragma clang fp contract(off)
        score += max(acc[h] * args.scale, 0.0f) * ws[h];
    }
    scores[slot] = score;
}

/* The same with S lanes per key, the head sum passed lane to lane in order (2T / S keys per T threads). */
template <uint S>
kernel void kernel_dsv41_indexer_score_rows_split_bf16(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const float *qf,
        device const float *weights,
        device const ushort *keys,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        uint3 ntg_u [[threads_per_threadgroup]]) {
    const uint ntg = ntg_u.x;
    threadgroup float *qs = shared;           /* [32][128] */
    threadgroup float *ws = shared + 4096u;   /* [32] */
    for (uint i = tid; i < 4096u; i += ntg) qs[i] = qf[i];
    if (tid < 32u) ws[tid] = weights[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (args.n_head != 32u || args.head_dim != 128u) return;
    constexpr uint HS = 32u / S;
    const uint part = tid % S, lanes = ntg / S;
    uint slot[2], row[2];
    FOR_UNROLL (uint r = 0; r < 2u; r++) {
        slot[r] = tgpig.x * lanes * 2u + r * lanes + tid / S;
        row[r] = min(slot[r], args.n_rows - 1u);
    }
    float acc[2][HS];
    FOR_UNROLL (uint r = 0; r < 2u; r++) FOR_UNROLL (uint i = 0; i < HS; i++) acc[r][i] = 0.0f;
    for (uint d0 = 0; d0 < 128u; d0 += 16u) {
        float4 k[2][4];
        FOR_UNROLL (uint r = 0; r < 2u; r++) {
            device const ushort4 *kb = (device const ushort4 *)keys + (((uint64_t)row[r] * 128u + d0) >> 2);
            FOR_UNROLL (uint i = 0; i < 4u; i++) k[r][i] = as_type<float4>(uint4(kb[i]) << 16);
        }
        FOR_UNROLL (uint i = 0; i < HS; i++) {
            threadgroup const float4 *q4 = (threadgroup const float4 *)(qs + (part * HS + i) * 128u + d0);
            FOR_UNROLL (uint r = 0; r < 2u; r++)
                acc[r][i] += dot(q4[0], k[r][0]) + dot(q4[1], k[r][1]) + dot(q4[2], k[r][2]) + dot(q4[3], k[r][3]);
        }
    }
    FOR_UNROLL (uint r = 0; r < 2u; r++) {
        float score = 0.0f;
        FOR_UNROLL (uint p = 0; p < S; p++) {
            if (p) score = simd_shuffle_up(score, (ushort)1);
            if (part == p) {
                FOR_UNROLL (uint i = 0; i < HS; i++) {
#pragma clang fp contract(off)
                    score += max(acc[r][i] * args.scale, 0.0f) * ws[p * HS + i];
                }
            }
        }
        if (part == S - 1u && slot[r] < args.n_rows) scores[slot[r]] = score;
    }
}
#define DSV41_ROWS_SPLIT(NAME, S) template [[host_name(NAME)]] \
    kernel void kernel_dsv41_indexer_score_rows_split_bf16<S>( \
    constant ds4_metal_args_glm_indexer_score_one &, device const float *, device const float *, device const ushort *, \
    device float *, threadgroup float *, uint3, ushort, uint3);
DSV41_ROWS_SPLIT("kernel_dsv41_indexer_score_rows_split2_bf16", 2)
DSV41_ROWS_SPLIT("kernel_dsv41_indexer_score_rows_split4_bf16", 4)


kernel void kernel_dsv41_candidate_list(
        constant uint &blocks,
        device const float *mask,
        device uint *list,
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[33];
    uint base = 0u;
    for (uint b0 = 0u; b0 < blocks; b0 += 4096u) {
        const uint b = b0 + tid * 4u;
        bool keep[4];
        uint count = 0u;
        for (uint j = 0u; j < 4u; j++) {
            keep[j] = b + j < blocks && mask[b + j] == 0.0f;
            count += keep[j] ? 1u : 0u;
        }
        const uint before = simd_prefix_exclusive_sum(count);
        if (lane == 31u) part[simd] = before + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const uint c = part[lane], c_before = simd_prefix_exclusive_sum(c);
            part[lane] = c_before;
            if (lane == 31u) part[32] = c_before + c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint at = base + part[simd] + before;
        for (uint j = 0u; j < 4u; j++) {
            if (keep[j] && at < 2048u) list[at] = b + j;
            at += keep[j] ? 1u : 0u;
        }
        base += part[32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Positions in the candidate list back to key ids, in place and in order. */
kernel void kernel_dsv41_candidate_ids(
        constant uint &count,
        device const uint *list,
        device int *selected,
        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint p = (uint)selected[i];
    selected[i] = (int)(list[p >> 3] * 8u + (p & 7u));
}


struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
};


kernel void kernel_dsv4_topk_mask(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    (void)topk;
    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

// Enables the selected compressed rows in the dense mask. This replaces the
// old O(n_comp * n_tokens * top_k) membership test with O(top_k * n_tokens)
// writes while preserving exactly the same 0/-inf mask consumed by attention.
kernel void kernel_dsv4_topk_mask_scatter(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((device const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((device float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

