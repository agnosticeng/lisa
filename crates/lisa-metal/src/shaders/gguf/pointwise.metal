// F32 Qwen GGUF pointwise chains. Preserve the separate operation rounding
// points and the MLX unary compiler contract (Safe + Precise, default MSL).
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline float lisa_gguf_sigmoid(float x) {
    const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
    return x < 0.0f ? y : 1.0f - y;
}

kernel void lisa_gguf_conv_silu(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    const uint c = i % channels;
    float acc = 0.0f;
    // Same four products and left-to-right adds as ops::conv1d. No FMA.
    for (uint j = 0; j < 4; ++j) {
        const float product = x[i + j * channels] * w[c * 4 + j];
        acc = acc + product;
    }
    out[i] = acc * lisa_gguf_sigmoid(acc);
}

kernel void lisa_gguf_gates(
    device const float *alpha [[buffer(0)]],
    device const float *beta [[buffer(1)]],
    device const float *dt [[buffer(2)]],
    device const float *a [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint2 &shape [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    const uint h = i % shape.x;
    const float x = alpha[i] + dt[h];
    const float e = metal::precise::exp(metal::min(x, 20.0f));
    const float soft = metal::precise::log(e + 1.0f);
    out[i] = (x > 20.0f ? x : soft) * a[h];
    out[shape.y + i] = lisa_gguf_sigmoid(beta[i]);
}

kernel void lisa_gguf_silu_mul(
    device const float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &reverse [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    const float silu = x[i] * lisa_gguf_sigmoid(x[i]);
    out[i] = reverse ? y[i] * silu : silu * y[i];
}
