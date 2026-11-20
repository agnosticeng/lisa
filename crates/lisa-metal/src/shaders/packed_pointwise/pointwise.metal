// F32 fusion adapters for the existing native elementwise/reduction law.
// Sigmoid and four-consecutive-read sum follow Apple's MIT-licensed MLX
// kernels retained under ../common; see repository NOTICE. Stable selection
// and ordered expert accumulation are LISA implementations.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct PointwiseArgs { uint width, rows; float scale; uint pad; };

kernel void lisa_relu_squared(
    constant PointwiseArgs & a [[buffer(0)]],
    device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= a.width * a.rows) return;
    float v = x[i] > 0.0f ? x[i] : 0.0f;
    y[i] = v * v;
}

kernel void lisa_weighted_sum6(
    constant PointwiseArgs & a [[buffer(0)]],
    device const float * x [[buffer(1)]], device const float * w [[buffer(2)]],
    device float * y [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.width * a.rows) return;
    uint token = i / a.width, col = i % a.width;
    x += ulong(token) * 6 * a.width + col;
    w += token * 6;
    float value = x[0] * w[0];
    for (uint k = 1; k < 6; ++k) {
        float term = x[k * a.width] * w[k];
        value = value + term;
    }
    y[i] = value;
}

// One SIMD group per 256-expert row. Six stable selections require no
// threadgroup barriers, temporary buffers, full sort, or host readback.
// The input contract is finite logits and selection biases.
kernel void lisa_sigmoid_router256(
    constant PointwiseArgs & a [[buffer(0)]],
    device const float * logits [[buffer(1)]], device const float * bias [[buffer(2)]],
    device int * ids [[buffer(3)]], device float * weights [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    float p[8], score[8], chosen[6];
    for (uint j = 0; j < 8; ++j) {
        uint id = lane * 8 + j;
        float x = logits[row * 256 + id];
        float z = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        p[j] = x < 0.0f ? z : 1.0f - z;
        score[j] = p[j] + bias[id];
    }
    for (uint k = 0; k < 6; ++k) {
        float best = -INFINITY;
        for (uint j = 0; j < 8; ++j) best = max(best, score[j]);
        best = simd_max(best);
        uint id = 0xffffffffu;
        for (uint j = 0; j < 8; ++j) if (score[j] == best) id = min(id, uint(lane) * 8 + j);
        id = simd_min(id);
        chosen[k] = simd_shuffle(p[id % 8], id / 8);
        if (lane == 0) ids[row * 6 + k] = int(id);
        if (lane == id / 8) score[id % 8] = -INFINITY;
    }
    // Match row_reduce_simple: lane 0 accumulates p0..p3, lane 1 p4..p5,
    // each as value + accumulator from positive zero, then simd_sum.
    float sum = 0.0f;
    for (uint j = uint(lane) * 4; j < min(uint(lane) * 4 + 4, 6u); ++j) sum = chosen[j] + sum;
    sum = simd_sum(sum);
    float denominator = sum > 6.103515625e-5f ? sum : 6.103515625e-5f;
    if (lane < 6) {
        // The standalone binary Divide uses Fast math, while Sigmoid and
        // Sum use Safe + Precise. Preserve that quotient explicitly.
        float normalized = metal::fast::divide(chosen[lane], denominator);
        weights[row * 6 + lane] = normalized * a.scale;
    }
}

static inline float standalone_sigmoid(float x) {
    float z = 1.0f / (1.0f + metal::exp(metal::abs(x)));
    return x < 0.0f ? z : 1.0f - z;
}
kernel void lisa_row_gate_blend_pair(
    constant PointwiseArgs & a [[buffer(0)]],
    device const float * k [[buffer(1)]], device const float * v [[buffer(2)]],
    device const float * pk [[buffer(3)]], device const float * pv [[buffer(4)]],
    device const float * kg [[buffer(5)]], device const float * vg [[buffer(6)]],
    device float * ko [[buffer(7)]], device float * vo [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= a.width * a.rows) return;
    uint row = i / a.width;
    float ka = standalone_sigmoid(kg[row]);
    float va = standalone_sigmoid(vg[row]);
    if (a.pad) { ka = ka * 0.0f; va = va * 0.0f; }
    float kp = pk[i] * ka, vp = pv[i] * va;
    float kc = k[i] * (1.0f - ka), vc = v[i] * (1.0f - va);
    ko[i] = kp + kc;
    vo[i] = vp + vc;
}
kernel void lisa_differential_silu_gate(
    constant PointwiseArgs & a [[buffer(0)]],
    device const float * attention [[buffer(1)]], device const float * lambda [[buffer(2)]],
    device const float * gate [[buffer(3)]], device float * out [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= a.width * a.rows) return;
    uint head = i / (3 * a.width), signal = i % (3 * a.width), col = i % a.width;
    float l = standalone_sigmoid(lambda[head]);
    float noise = l * attention[(head * 4 + 3) * a.width + col];
    float mixed = attention[head * 4 * a.width + signal] - noise;
    float g = gate[i] + a.scale;
    float activation = g * standalone_sigmoid(g);
    out[i] = mixed * activation;
}
