// GLM MTP's kv_a_layernorm from ds4's kernel_glm_kv_lora_rms_norm.
// This deliberately uses scalar strided accumulation and a threadgroup tree;
// ds4 does not use the float4/SIMD reduction used by its ordinary RMSNorm.
{
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint threads = threads_per_threadgroup.x;
    const uint n = (uint)n_dim;
    const uint raw_n = (uint)raw_dim;
    if (n == 0u || n > raw_n || row >= (uint)n_rows || tid >= threads) return;

    threadgroup float scratch[128];
    const device float *x = input + (size_t)row * raw_n;
    float ss = 0.0f;
    for (uint i = tid; i < n; i += threads) {
        const float v = x[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = threads >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(scratch[0] / (float)n + eps);
    device float *out = output + (size_t)row * n;
    for (uint i = tid; i < n; i += threads) {
        out[i] = x[i] * inv * weight[i];
    }
}
