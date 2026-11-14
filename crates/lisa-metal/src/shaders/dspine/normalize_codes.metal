// DSpine normalize_codes with MLX's exact four-read reduction schedule.
// Apple MLX MIT-licensed reduction design and Tencent DSpine; see NOTICE.
#pragma clang fp contract(off)
uint row = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
if (row % T == 0) {
    uint token = uint(input_ids[row]);
    for (uint c = lane; c < C; c += 128)
        codes[row * C + c] = bfloat16_t(float(codebook[token * C + c]) * 32.0f);
    return;
}
threadgroup float partials[4];
float total = 0.0f;
for (uint base = lane * 4; base < C; base += 128 * 4) {
    for (uint j = 0; j < 4; ++j) {
        float x = float(projected[row * C + base + j]);
        float square = x * x;
        total = square + total;
    }
}
total = simd_sum(total);
if (lane % 32 == 0) partials[lane / 32] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
total = lane < 4 ? partials[lane] : 0.0f;
total = simd_sum(total);
if (lane == 0) partials[0] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
float denominator = metal::sqrt(partials[0]) + eps[0];
for (uint c = lane; c < C; c += 128) {
    float numerator = float(projected[row * C + c]) * 32.0f;
    codes[row * C + c] = bfloat16_t(numerator / denominator);
}
