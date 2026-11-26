// Copies of the Inkling step's persistent state (the vendored reference's
// paged decode state, shaders/reference/, provenance.json) for snapshots and
// growth: the committed pages of every region of a key/value extent, and
// flat word copies of the convolution state and the draft's rings.
#include <metal_stdlib>
using namespace metal;

struct LisaExtentCopyParams {
    uint source_pages;  // pages of each region of the source extent
    uint target_pages;  // pages of each region of the target extent
    uint pages;         // leading pages of each region to copy
};

// Threadgroups (x, tensor, region) of 256 threads, one 16-byte word each.
// Per region an extent holds the keys of every page, then their scales,
// the values and their scales; a page holds 32 tokens of 8 heads.
[[kernel]] void lisa_kv_extent_copy(
    device const uint4 *source [[buffer(0)]],
    device uint4 *target [[buffer(1)]],
    constant LisaExtentCopyParams &p [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong data = 32u * 8u * 128u / 16u, scales = 32u * 8u * 4u / 16u;
    const uint tensor = group.y, region = group.z;
    const ulong page_words = (tensor & 1) ? scales : data;
    const ulong word = ulong(group.x) * 256 + lane;
    if (word >= ulong(p.pages) * page_words) return;
    const ulong before = (tensor + 1) / 2 * data + tensor / 2 * scales;
    const ulong from = ulong(region) * p.source_pages * 2 * (data + scales) +
                       ulong(p.source_pages) * before + word;
    const ulong to = ulong(region) * p.target_pages * 2 * (data + scales) +
                     ulong(p.target_pages) * before + word;
    target[to] = source[from];
}

// One thread per 4-byte word.
[[kernel]] void lisa_words_copy(
    device const uint *source [[buffer(0)]],
    device uint *target [[buffer(1)]],
    constant uint &words [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < words) target[i] = source[i];
}
