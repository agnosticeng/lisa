// DS4 indexed attention-lora fallback, copied from metal/dsv4_misc.metal.
// Reference: b4e3faf5262b21db94ec309cf321640ff1eed460, same-weight adapter
// ds4-mlx-affine-native-dsa-complete. The ABI, runtime geometry, F16 staging,
// online softmax and reduction body are unchanged. GLM-5.3 dispatches the
// <false, false> specialization with qk_rope=0; retain the runtime branches.
// Copyright (c) 2026 The ds4.c authors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2023 DeepSeek
// SPDX-License-Identifier: MIT
// Full notice: ../common/attention/ds4_flash_attn.LICENSE
#include <metal_stdlib>
using namespace metal;
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct ds4_metal_args_glm_attention_indexed_batch {
    uint32_t n_tokens;
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t value_type;
    uint32_t pos0;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t head_base;
};

static float glm_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static void glm_rope_yarn(
        float theta_extrap,
        float freq_scale,
        float corr_dims[2],
        int   i0,
        float ext_factor,
        float mscale,
        thread float *cos_theta,
        thread float *sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = glm_rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

static float glm_rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void glm_rope_yarn_corr_dims(
        int   n_dims,
        int   n_ctx_orig,
        float freq_base,
        float beta_fast,
        float beta_slow,
        float dims[2]) {
    dims[0] = max(0.0f,
                  floor(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f,
                  ceil(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

static inline float glm_cache_load_f16_only(
        device const char *base,
        uint64_t index) {
    return (float)((device const half *)base)[index];
}

static inline float2 glm_cache_load_rotated_rope_pair_f16_only(
        device const char *base,
        uint64_t           rope_base,
        uint               r,
        uint               row,
        uint               qk_rope,
        float              freq_base,
        float              freq_scale,
        float              ext_factor,
        float              attn_factor,
        float              corr0,
        float              corr1) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = theta_base * exp2(inv_ndims * (float)r * log2(freq_base));
#else
    const float theta = theta_base * pow(freq_base, inv_ndims * (float)r);
#endif
    float corr_dims[2] = {corr0, corr1};
    float cos_theta;
    float sin_theta;
    glm_rope_yarn(theta,
                  freq_scale,
                  corr_dims,
                  (int)r,
                  ext_factor,
                  attn_factor,
                  &cos_theta,
                  &sin_theta);
    const float x0 = glm_cache_load_f16_only(base, rope_base + r);
    const float x1 = glm_cache_load_f16_only(base, rope_base + r + 1u);
    return float2(x0 * cos_theta - x1 * sin_theta,
                  x0 * sin_theta + x1 * cos_theta);
}

template <bool assume_valid_rows, bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        (args.qk_rope != 0u && args.qk_rope != 64u)) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.qk_rope != 0u && args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>)
        glm_attention_indexed_batch_lora_group8_vec_t;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, true>;
