// Tencent DSpine's chunk-max candidate retrieval, adapted to Metal.
// Comparisons use Lisa's stable total order, including NaNs and signed zero.
const uint lane = thread_position_in_threadgroup.x;
const uint chunk = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
uint id = chunk * 64 + lane;
float value = id < V ? float(scores[size_t(row) * V + id]) : 0.0f;
if (id >= V) id = UINT_MAX;
const uint other = chunk * 64 + lane + 32;
const float other_value = other < V ? float(scores[size_t(row) * V + other]) : 0.0f;
if (other < V && dspine_before(other_value, other, value, id)) {
    id = other;
    value = other_value;
}
for (ushort stride = 16; stride; stride >>= 1) {
    const uint next = simd_shuffle_xor(id, stride);
    const float next_value = simd_shuffle_xor(value, stride);
    if (dspine_before(next_value, next, value, id)) {
        id = next;
        value = next_value;
    }
}
if (lane == 0) maxima[row * CHUNKS + chunk] = id;
