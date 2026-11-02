// Ordered eight-expert reduction used after the routed down projections.
// One threadgroup handles each token, matching ds4's
// `kernel_dsv4_moe_sum8_f32` launch and preserving slot order.
{
    const uint token = threadgroup_position_in_grid.x;
    if (token >= (uint)n_tokens) return;

    const uint lane = thread_index_in_threadgroup;
    const uint lanes = threads_per_threadgroup.x;
    const size_t row = (size_t)token * (size_t)width * 8;
    for (uint column = lane; column < (uint)width; column += lanes) {
        float value = src[row + column];
        value += src[row + (size_t)width + column];
        value += src[row + (size_t)width * 2 + column];
        value += src[row + (size_t)width * 3 + column];
        value += src[row + (size_t)width * 4 + column];
        value += src[row + (size_t)width * 5 + column];
        value += src[row + (size_t)width * 6 + column];
        value += src[row + (size_t)width * 7 + column];
        out[(size_t)token * (size_t)width + column] = value;
    }
}
