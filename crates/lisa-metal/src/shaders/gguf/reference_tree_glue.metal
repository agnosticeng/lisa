// Glue around the vendored reference's draft tree selection
// (shaders/reference/sampling.metal, provenance.json) for one lane: the draft's
// candidate lattice into the selector buffers its tree kernel reads, and the
// verify rows it lays out into LISA's tree tensor.
#include <metal_stdlib>
using namespace metal;

#define LISA_TREE_POSITIONS 7u
#define LISA_TREE_CANDIDATES 16u
#define LISA_TREE_SHARDS 8u
#define LISA_TREE_ROOT 0xffffu

struct LisaTreeRow {
    uint ancestors;
    ushort parent;
    ushort depth;
};

// One threadgroup of 7 x 16 threads, thread (position, i) for candidate i of
// a proposal position. Inputs: candidate IDs `I32 [7, 16]` in any order,
// their raw logits `F32 [7, 16]`, and edges `F32 [7, 16, 16]`, entry
// [p, a, k] the selector's edge from candidate a of position p - 1 (any of
// position 0's rows for the anchor) to candidate k of position p, both in
// the input order. Writes the candidates in the order of the reference's
// merged top 16 (larger logit first, the lower ID among equal ones), their
// logits, and the edge table where its tree kernel reads it: after the
// selector's shard partials, `[position][predecessor][candidate]` in the
// sorted order. Position 0's predecessor rows past the first are unread.
[[kernel]] void lisa_tree_inputs(
    device const int *ids [[buffer(0)]],
    device const float *logits [[buffer(1)]],
    device const float *edges [[buffer(2)]],
    device uint *candidates [[buffer(3)]],
    device float *unary [[buffer(4)]],
    device float *partial_values [[buffer(5)]],
    uint index [[thread_index_in_threadgroup]]) {
    constexpr uint K = LISA_TREE_CANDIDATES;
    threadgroup uint order[LISA_TREE_POSITIONS][K];
    const uint position = index / K, i = index % K;
    if (position < LISA_TREE_POSITIONS) {
        const float value = logits[index];
        const uint id = uint(ids[index]);
        uint rank = 0;
        for (uint j = 0; j < K; ++j) {
            const float other = logits[position * K + j];
            const uint other_id = uint(ids[position * K + j]);
            rank += other > value || (other == value && other_id < id) ? 1u : 0u;
        }
        order[position][rank] = i;
        candidates[position * K + rank] = id;
        unary[position * K + rank] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (position < LISA_TREE_POSITIONS) {
        // Thread (position, a): sorted predecessor a's row of edges.
        const uint predecessor = position > 0 ? order[position - 1][i] : 0u;
        device const float *row = edges + (position * K + predecessor) * K;
        device float *out = partial_values + LISA_TREE_POSITIONS * LISA_TREE_SHARDS * K +
                            (position * K + i) * K;
        for (uint k = 0; k < K; ++k)
            out[k] = row[order[position][k]];
    }
}

// One thread: the verify block's `rows` rows (the reference's verify-row
// layout) as LISA's tree tensor `I32 [2, rows]`, parents (-1 for a root, and
// for a row of no lane) then depths, and their tokens as `I32 [rows]`.
[[kernel]] void lisa_tree_rows(
    device const uint *tokens [[buffer(0)]],
    device const LisaTreeRow *tree [[buffer(1)]],
    device int *ids [[buffer(2)]],
    device int *lisa_tree [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows)
        return;
    ids[row] = int(tokens[row]);
    const LisaTreeRow node = tree[row];
    lisa_tree[row] = node.parent == LISA_TREE_ROOT ? -1 : int(node.parent);
    lisa_tree[rows + row] = int(node.depth);
}
