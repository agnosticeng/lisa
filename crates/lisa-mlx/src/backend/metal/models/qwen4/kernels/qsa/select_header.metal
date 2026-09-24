// Header/helpers for the QSA block selector.
#include <metal_stdlib>
using namespace metal;

// Total order for the block/dense output network: selected blocks by
// ascending id, followed by invalid/padded lanes.
inline bool qsa_index_before(
    uint a_index, bool a_valid, uint b_index, bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    return a_index < b_index;
}

// MLX's Metal ArgPartition currently delegates to its stable ascending merge
// sort.  The v2.10 rows-gather lane consumes the final K indices directly, so
// its selected valid blocks appear in ascending adjusted-score order.  Equal
// values preserve input order, which is ascending block id.  Keep invalid
// lanes after valid lanes inside the network; the epilogue places the exact
// number of selected masked fillers before the winners.
inline bool qsa_row_score_before(
    float a_score,
    uint a_index,
    bool a_valid,
    float b_score,
    uint b_index,
    bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    if (!a_valid) {
        return a_index < b_index;
    }
    bool a_nan = metal::isnan(a_score);
    bool b_nan = metal::isnan(b_score);
    if (a_nan || b_nan) {
        if (a_nan != b_nan) {
            return !a_nan;
        }
        return a_index < b_index;
    }
    if (a_score < b_score) {
        return true;
    }
    if (b_score < a_score) {
        return false;
    }
    return a_index < b_index;
}

// Monotonic IEEE-754 mapping: numerically larger finite floats produce larger
// unsigned keys. Appending the block id makes the 64-bit key unique and mirrors
// v2.10's stable ascending GPU sort followed by a final-K slice: if the
// 1e-12 adjustment itself rounds away, the later/higher id wins the cutoff.
inline uint qsa_float_order_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

inline ulong qsa_composite_key(float adjusted_score, uint block_id) {
    return (ulong(qsa_float_order_key(adjusted_score)) << 32) |
           ulong(block_id);
}

// MLX's NAX float32 GEMM path truncates each operand to a 10-bit mantissa
// before fp32 accumulation, but the legacy GEMV route remains full fp32.
// A uniform runtime mask mirrors that eager-matmul contract without adding a
// query-row or logical-history specialization to the Python kernel cache.
inline float qsa_mlx_gemm_operand(float value, uint operand_mask) {
    return as_type<float>(as_type<uint>(value) & operand_mask);
}

// Unpack element `i` (0..7, little-endian) of a 16-byte bf16 chunk and apply
// the operand truncation. Vectorizing the score loop's loads keeps the exact
// dim order (and therefore the fp32 accumulation order) while issuing 8x
// fewer loads; the loop was load-latency-bound, not bandwidth-bound.
inline float qsa_unpack_trunc(uint4 packed, uint i, uint operand_mask) {
    const uint word = (i < 2u) ? packed.x
        : (i < 4u) ? packed.y
        : (i < 6u) ? packed.z
        : packed.w;
    const ushort half_word =
        (i & 1u) ? ushort(word >> 16) : ushort(word & 0xffffu);
    return qsa_mlx_gemm_operand(float(uint16_to_bfloat16(half_word)), operand_mask);
}

constant constexpr uint HEADS = 4;
constant constexpr uint HEAD_DIM = 128;
constant constexpr uint BACKING_BLOCKS = 65536;
constant constexpr uint TOP_K = 512;
constant constexpr uint RATIO = 4;
constant constexpr uint WIDTH = 512;
constant constexpr uint RADIX_BINS = 256;
constant constexpr uint OUTPUT_TOKENS = 0;
constant constexpr bool ENABLE_TF32 = true;
constant constexpr bool ROW_TOKEN_MODE = false;
constant constexpr bool Q_INPUT_IS_FLOAT32 = false;
constant constexpr bool POOLED_INPUT_IS_FLOAT32 = false;
constant constexpr float SQRT_HEAD_DIM = 11.313708498984761f;
