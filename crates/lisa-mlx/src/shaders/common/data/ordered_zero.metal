#include <metal_stdlib>
using namespace metal;

// Ordered GPU initialization for reused scratch storage. A 2-D grid keeps
// indexing correct for allocations larger than 2^32 words.
kernel void lisa_ordered_zero_u32(
    device uint* output [[buffer(0)]],
    constant ulong& words [[buffer(1)]],
    constant uint& row_width [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong index = ulong(tid.y) * ulong(row_width) + ulong(tid.x);
    if (index < words) output[index] = 0;
}
