// DSpine bridge normalization, preserving MLX 0.32.4 row_reduce_simple's
// four consecutive reads, value+accumulator order and two simd_sum stages.
// Adapted from Apple's MIT-licensed reduction kernels; see NOTICE.
#pragma clang fp contract(off)
uint row = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint group = lane / 32;
uint simd_lane = lane % 32;
threadgroup float partials[32];
float total = 0.0f;
for (uint base = lane * 4; base < H; base += THREADS * 4) {
    for (uint j = 0; j < 4 && base + j < H; ++j) {
        float x = float(raw[row * H + base + j]);
        float square = x * x;
        total = square + total;
    }
}
total = simd_sum(total);
if (simd_lane == 0) partials[group] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
total = lane < THREADS / 32 ? partials[lane] : 0.0f;
total = simd_sum(total);
if (lane == 0) partials[0] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = partials[0] * reciprocal[0];
float scale;
if (READ) {
    scale = metal::sqrt(metal::max(mean, 1.1754943508222875e-38f));
} else {
    scale = metal::rsqrt(mean + eps[0]);
}
if (lane == 0) rms[row] = scale;
for (uint h = lane; h < H; h += THREADS) {
    float x = float(raw[row * H + h]);
    if (READ) {
        normalized[row * H + h] = bfloat16_t(x / (scale + eps[0]));
    } else {
        bfloat16_t y = bfloat16_t(x * scale);
        normalized[row * H + h] = bfloat16_t(float(y) * float(weight[h]));
    }
}
