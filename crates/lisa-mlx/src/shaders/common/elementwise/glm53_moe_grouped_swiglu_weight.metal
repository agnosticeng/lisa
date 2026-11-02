using namespace metal;

[[kernel]] void glm53_moe_grouped_swiglu_weight_f16(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    const device float* route_weights [[buffer(2)]],
    device half* y [[buffer(3)]],
    const constant int& n_slots [[buffer(4)]],
    const constant int& width [[buffer(5)]],
    const constant float& limit [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]) {
    if (row >= uint(n_slots)) return;

    const device float* gate_row = gate + size_t(row) * size_t(width);
    const device float* up_row = up + size_t(row) * size_t(width);
    device half* mid_row = y + size_t(row) * size_t(width);
    const float route_weight = route_weights[row];
    const float c = limit;

    for (uint i = tid; i < uint(width); i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = half(silu * u * route_weight);
    }
}
