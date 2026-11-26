// Vendored from Splash (https://github.com/incoai/splash), revision
// 2afc7337aacc044722fa95e4cb3da75c281a19a8, Apache License 2.0 (LICENSE-APACHE-2.0).
// Quantized GGUF decoding follows ggml (MIT, LICENSE-GGML). Each original
// header is inlined once at its first include, between begin/end markers; the
// code is otherwise unchanged. Source hashes: provenance.json.
// ---- begin runtime/metal/kernels/prefill/gguf_prefill_int8.metal (Splash 2afc733, Apache-2.0) ----
// Int8-activation prefill GEMMs for MLX affine 4- and 5-bit weights in groups of 64 (af4g64, af5g64) and for Q8_0
// (ops::LinearTile::GgufPrefillInt8). Each 64-input group g of a chunk row is quantized to int8 codes qa with its own
// scale sa_g, and with w = s * q + z per affine weight group:
//
//   y[m][n] = sum_g ( (sa_g[m] s_g[n]) sum_{k in g} qa[m][k] q[n][k]  +  z_g[n] (sa_g[m] sum_{k in g} qa[m][k]) )
//
// and with w = d * q per Q8_0 32-group h of g (its codes int8 already):
//
//   y[m][n] = sum_g sa_g[m] sum_{h < 2} d_{g,h}[n] sum_{k in h} qa[m][k] q[n][k]
//
// The weights' codes feed int8 16x32x16 matmul2d fragments straight from the planes; each group's int32 sums (each
// Q8_0 32-group's) fold into fp32 once. The weights are exact; only the activations round, per row and group.
//
// The fragment schedule and the activations' slot order are adapted from oMLX's oQ A8 kernel (Apache-2.0 notice in
// THIRD_PARTY_NOTICES): in fragment step t (0-3) of a group, lane column x (coord.x / 4) supplies inputs
// 16x + 4t .. 16x + 4t + 3, 32-group x >> 1, chunk 2 (x & 1) + (t >> 1) of the planes (each weight column's 16 codes
// of the lane are one uint2 of its 4-bit plane), nibbles (word >> 4 (t & 1)) & 0x0f0f0f0f: chunk slots {0, 4, 1, 5}
// or {2, 6, 3, 7} (metal/abi/QuantFormat.h). The quantizer writes each 32-group in that order. Q8_0's tile reads the
// same codes in another schedule, so that a fused projection of both quantizes its rows once: steps 2h and 2h + 1
// take 32-group h alone, lane column x the inputs at 32h + 8x + 4 (t & 1) .. +3, chunk x of the 32-group's plane
// record (one uint2 of its bytes, chunk slots {0, 4, 1, 5} from it in the even step, {2, 6, 3, 7} in the odd one).
// ---- begin runtime/metal/abi/Gguf.h (Splash 2afc733, Apache-2.0) ----

// GGUF quantized projection parameters and tile geometry shared by host
// dispatch code and Metal kernels: the staged tiles of
// kernels/shared/gguf_linear.metal and kernels/shared/moe_gguf.metal, the
// Apple9 register tile of kernels/decode/linear_gguf_sgmatrix.metal, and the
// int8 prefill tile of kernels/prefill/gguf_prefill_int8.metal.
// ---- begin runtime/metal/abi/QuantFormat.h (Splash 2afc733, Apache-2.0) ----

// Quantized weight formats of the MDGG0001 image, shared by the host planner
// and reader, the load-time repack, the GEMM kernels and the tests. A [N, K]
// tensor with G = K / 32 groups per row is stored in tiles of T =
// QUANT_TILE_ROWS rows as
//   plane0 [N / T][G][T][plane0_bytes]
//   plane1 [N / T][G][T][plane1_bytes]   (none when plane1_bytes is 0)
//   meta   [N / T][G / meta_groups][T][meta_bytes]
// A meta unit is one native block and holds its scale fields.
//
// It holds what defines the image's bytes, plus each format's kernel name
// token, which the host reads. The decode-only value tables are in
// metal/abi/QuantTables.h.
//
// Inside a group of 32 the elements are in lane-owned chunk order: chunk c
// (0..3) holds elements 4c..4c+3 and 16+4c..16+4c+3 as pairs p = 0..3, pair p
// being elements e0 and e0 + 1 with e0 = 16 (p >> 1) + 4c + 2 (p & 1).
// Element e is at slot quant_slot(e) = 8c + 2p + (e & 1). The planes pack the
// slots as follows.
//   4-bit linear codes (Q4_K, Q4_0, Q4_1, the low bits of Q5_K, Q5_0, Q5_1
//   and Q6_K):
//     word c holds slots 8c..8c+7, pair p's e0 at bits 4p and e1 at bits
//     16 + 4p.
//   Every other field is a little-endian bit string of the 32 slots: IQ4 and
//     MXFP4 indices (4 bits, so byte p of word c is pair p), BF16 and F16
//     values (16, so word 4c + p is pair p), Q8_0 values (8), Q6_K high, Q3_K
//     low, Q2_K, PQ2_0 and Q2_0 bits (2), Q5_K, Q5_0 and Q5_1 fifth and Q3_K
//     hmask bits (1).
//   IQ3_S word c: bits 0..7 and 8..15 the low grid index bits of elements
//     4c..4c+3 and 16+4c..16+4c+3, 16..23 the sign bits of slots 8c..8c+7,
//     24 and 25 the two ninth index bits, 26..29 the group's scale.
//   IQ3_XXS, IQ2 and IQ1 keep the group's native bytes, whose grid entries
//     span four or eight elements and so several chunks: its block_*'s fields
//     for the group's 32 elements, split between the planes as its row below
//     states. kernels/common/quant_formats.h decodes a chunk from them.
//
// The MLX affine formats (mlx.core.quantize, mode "affine") hold b-bit codes
// q, b of 2, 3, 4, 5, 6 or 8, in groups of 32, 64 or 128 elements with a
// bf16 scale s and bias z each: value = s * q + z. Their planes hold the codes
// as the GGUF formats of that width do (2 bits as Q2_K, 3 as Q3_K's low and
// high bits, 4 as Q4_K, 5 as Q5_K, 6 as Q6_K, 8 as Q8_0's bytes), and their
// meta unit is the group's s, z. MLX keeps codes, scales and biases in tensors
// of their own; the loader writes each group as the native block the repack
// reads: s, z, then the group's codes as MLX packs a row's, a little-endian
// bit string of b bits per code.
//
// NVFP4 (NVIDIA's Model Optimizer and compressed-tensors, MLX's mode "nvfp4",
// llama.cpp's type 40) holds E2M1 codes with an E4M3 scale per 16 elements and
// an FP32 scale g: value = g * e4m3 * e2m1. FP8 (Model Optimizer's and
// compressed-tensors') holds E4M3 values with an FP32 scale g: value = g *
// e4m3. g is the tensor's, or each row's or expert's when a checkpoint scales
// those apart. Their meta unit spans eight groups and holds g, as a K-quant's
// super-block holds its d, so the kernels read every scale a value needs from
// its meta unit. The loader writes each 256 elements of a row as the native
// block the repack reads: NVFP4's 16 E4M3 scales (byte i scaling elements
// 16i..16i+15), g at QUANT_NVFP4_G, then from QUANT_NVFP4_CODES the codes as a
// row packs them (element e in bits 4 (e % 2) of byte e / 2); FP8's g, then
// from QUANT_FP8_VALUES the 256 E4M3 bytes. Their planes hold the codes as
// MXFP4's indices and the values as Q8_0's. An NVFP4 tensor whose rows are not
// whole 256-element blocks (llama.cpp's blocks are 64 elements) takes NVFP4's
// form of 64-element blocks, NVFP4_64 (its 4 E4M3 scales, g at QUANT_NVFP464_G,
// its codes from QUANT_NVFP464_CODES), whose meta unit spans two groups.
//
// MXFP8 (the OCP microscaling format: MLX's mode "mxfp8", Inkling 2's
// release) holds E4M3 values with an E8M0 exponent e per 32 elements: value =
// 2^(e - 127) * e4m3. Its planes hold the values as FP8's do and its meta
// unit, one group's, is e, as MXFP4's is. The loader writes each 32 elements
// of a row as the native block the repack reads: e, then the 32 E4M3 bytes.
//
// BF16 and F16 hold 16-bit float values x with an FP32 scale g per 256
// elements, as FP8 holds g: value = g * x. A GGUF tensor of either type has
// no scale, so its g is 1, which the repack writes; a native block is 256 of
// its values, a whole number of the type's one-element blocks. That block is
// the repack's alone: rows kept as stored (quant_stored_row_bytes) hold each
// value as its own block, at any width.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define QUANT_CONSTANT constant constexpr
#else
#include <stdint.h>
#define QUANT_CONSTANT inline constexpr
#endif

// Format ids: kQuantFormats index, GgufRepackParams.fmt and the run-time format
// of the fused and MoE expert kernels. A new format takes the next id: the
// weight goldens' fixtures are seeded by format id.
#define GGUF_FMT_Q4K 0u
#define GGUF_FMT_IQ4XS 1u
#define GGUF_FMT_IQ4NL 2u
#define GGUF_FMT_Q5K 3u
#define GGUF_FMT_Q6K 4u
#define GGUF_FMT_Q3K 5u
#define GGUF_FMT_Q80 6u
#define GGUF_FMT_IQ3S 7u
#define GGUF_FMT_Q2K 8u
#define GGUF_FMT_IQ3XXS 9u
#define GGUF_FMT_IQ2XXS 10u
#define GGUF_FMT_IQ2XS 11u
#define GGUF_FMT_IQ2S 12u
#define GGUF_FMT_IQ1S 13u
#define GGUF_FMT_IQ1M 14u
#define GGUF_FMT_Q40 15u
#define GGUF_FMT_Q41 16u
#define GGUF_FMT_MXFP4 17u
#define GGUF_FMT_PQ20 18u
// MLX affine, b bits in groups of g (GGUF_FMT_AF<b>G<g>).
#define GGUF_FMT_AF2G32 19u
#define GGUF_FMT_AF2G64 20u
#define GGUF_FMT_AF2G128 21u
#define GGUF_FMT_AF3G32 22u
#define GGUF_FMT_AF3G64 23u
#define GGUF_FMT_AF3G128 24u
#define GGUF_FMT_AF4G32 25u
#define GGUF_FMT_AF4G64 26u
#define GGUF_FMT_AF4G128 27u
#define GGUF_FMT_AF5G32 28u
#define GGUF_FMT_AF5G64 29u
#define GGUF_FMT_AF5G128 30u
#define GGUF_FMT_AF6G32 31u
#define GGUF_FMT_AF6G64 32u
#define GGUF_FMT_AF6G128 33u
#define GGUF_FMT_AF8G32 34u
#define GGUF_FMT_AF8G64 35u
#define GGUF_FMT_AF8G128 36u
#define GGUF_FMT_NVFP4 37u
#define GGUF_FMT_FP8 38u
#define GGUF_FMT_Q50 39u
#define GGUF_FMT_Q51 40u
#define GGUF_FMT_BF16 41u
#define GGUF_FMT_F16 42u
#define GGUF_FMT_Q20 43u
#define GGUF_FMT_NVFP464 44u
#define GGUF_FMT_MXFP8 45u
#define GGUF_FMT_COUNT 46u

// The tensor type an image descriptor names for an MLX affine format: no GGUF
// type is of this form.
#define QUANT_AFFINE_TYPE(bits, group) (0x4D4C0000u | ((bits) << 8) | (group))
// The tensor types of NVFP4's, FP8's and MXFP8's native blocks, which the
// loader writes ("NVF4", "FP8E", "MXF8"): no GGUF type is any of them.
#define QUANT_NVFP4_TYPE 0x4E564634u
#define QUANT_FP8_TYPE 0x46503845u
#define QUANT_NVFP464_TYPE 0x4E563634u
#define QUANT_MXFP8_TYPE 0x4D584638u
// Byte offsets in NVFP4's native block (its scales at 0) and FP8's (its g at 0).
#define QUANT_NVFP4_G 16u
#define QUANT_NVFP4_CODES 20u
#define QUANT_FP8_VALUES 4u
#define QUANT_NVFP464_G 4u
#define QUANT_NVFP464_CODES 8u

struct QuantFormat {
  uint32_t ggml_type;      // GGUF tensor type, or a loader's (QUANT_AFFINE_TYPE, QUANT_NVFP4_TYPE, ...)
  uint32_t block_elements; // elements per native block
  uint32_t block_bytes;    // bytes per native block
  uint32_t plane0_bytes;   // per row and group of 32
  uint32_t plane1_bytes;   // per row and group of 32
  uint32_t meta_bytes;     // per row and meta unit
  uint32_t meta_groups;    // groups of 32 per meta unit
  char name[8];            // kernel name suffix
};

QUANT_CONSTANT QuantFormat kQuantFormats[GGUF_FMT_COUNT] = {
    {12, 256, 144, 16, 0, 16, 8, "q4k"},  // meta: d, dmin, 12 scale bytes
    {23, 256, 136, 16, 0, 8, 8, "iq4xs"}, // meta: d, scales_h, scales_l
    {20, 32, 18, 16, 0, 2, 1, "iq4nl"},   // meta: d
    {13, 256, 176, 16, 4, 16, 8, "q5k"},  // plane1: fifth bits; meta as Q4_K
    {14, 256, 210, 16, 8, 20, 8, "q6k"},  // plane1: high 2 bits; meta: 16 int8 scales, d, 0, 0
    {11, 256, 110, 8, 4, 16, 8, "q3k"},   // plane1: hmask bits; meta: d, 0, 0, 12 scale bytes
    {8, 32, 34, 32, 0, 2, 1, "q80"},      // meta: d
    {21, 256, 110, 16, 0, 2, 8, "iq3s"},  // plane0 also holds signs, qh and the scale; meta: d
    {10, 256, 84, 8, 0, 20, 8, "q2k"},    // meta: d, dmin, the 16 scale bytes
    {18, 256, 98, 8, 4, 2, 8, "iq3xxs"},  // plane0: qs; plane1: the sign and scale word; meta: d
    {16, 256, 66, 8, 0, 2, 8, "iq2xxs"},  // plane0: the grid index and the sign and scale words; meta: d
    {17, 256, 74, 8, 1, 2, 8, "iq2xs"},   // plane0: the four qs; plane1: the scales byte; meta: d
    {22, 256, 82, 8, 2, 2, 8, "iq2s"},    // plane0: qs, then signs; plane1: qh, then the scales byte; meta: d
    {19, 256, 50, 4, 2, 2, 8, "iq1s"},    // plane0: qs; plane1: qh; meta: d
    {29, 256, 56, 4, 2, 8, 8, "iq1m"},    // plane0: qs; plane1: qh; meta: the 8 scale bytes, which hold d
    {2, 32, 18, 16, 0, 2, 1, "q40"},      // meta: d
    {3, 32, 20, 16, 0, 4, 1, "q41"},      // meta: d, m
    {39, 32, 17, 16, 0, 1, 1, "mxfp4"},   // meta: e
    {142, 128, 34, 8, 0, 2, 4, "pq20"},   // Prism's block_pq2_0, one d per 128 elements; meta: d
    // MLX affine: the loader's native block s | z | codes; meta: s, z.
    {QUANT_AFFINE_TYPE(2, 32), 32, 12, 8, 0, 4, 1, "af2g32"},
    {QUANT_AFFINE_TYPE(2, 64), 64, 20, 8, 0, 4, 2, "af2g64"},
    {QUANT_AFFINE_TYPE(2, 128), 128, 36, 8, 0, 4, 4, "af2g128"},
    {QUANT_AFFINE_TYPE(3, 32), 32, 16, 8, 4, 4, 1, "af3g32"},
    {QUANT_AFFINE_TYPE(3, 64), 64, 28, 8, 4, 4, 2, "af3g64"},
    {QUANT_AFFINE_TYPE(3, 128), 128, 52, 8, 4, 4, 4, "af3g128"},
    {QUANT_AFFINE_TYPE(4, 32), 32, 20, 16, 0, 4, 1, "af4g32"},
    {QUANT_AFFINE_TYPE(4, 64), 64, 36, 16, 0, 4, 2, "af4g64"},
    {QUANT_AFFINE_TYPE(4, 128), 128, 68, 16, 0, 4, 4, "af4g128"},
    {QUANT_AFFINE_TYPE(5, 32), 32, 24, 16, 4, 4, 1, "af5g32"},
    {QUANT_AFFINE_TYPE(5, 64), 64, 44, 16, 4, 4, 2, "af5g64"},
    {QUANT_AFFINE_TYPE(5, 128), 128, 84, 16, 4, 4, 4, "af5g128"},
    {QUANT_AFFINE_TYPE(6, 32), 32, 28, 16, 8, 4, 1, "af6g32"},
    {QUANT_AFFINE_TYPE(6, 64), 64, 52, 16, 8, 4, 2, "af6g64"},
    {QUANT_AFFINE_TYPE(6, 128), 128, 100, 16, 8, 4, 4, "af6g128"},
    {QUANT_AFFINE_TYPE(8, 32), 32, 36, 32, 0, 4, 1, "af8g32"},
    {QUANT_AFFINE_TYPE(8, 64), 64, 68, 32, 0, 4, 2, "af8g64"},
    {QUANT_AFFINE_TYPE(8, 128), 128, 132, 32, 0, 4, 4, "af8g128"},
    // The loader's native blocks of 256 elements (NVFP4_64's of 64, last).
    {QUANT_NVFP4_TYPE, 256, 148, 16, 0, 20, 8, "nvfp4"},   // meta: the 16 E4M3 scales, g
    {QUANT_FP8_TYPE, 256, 260, 32, 0, 4, 8, "fp8"},        // meta: g
    // Q5_K's planes with Q4_0's and Q4_1's meta.
    {6, 32, 22, 16, 4, 2, 1, "q50"},      // plane1: fifth bits; meta: d
    {7, 32, 24, 16, 4, 4, 1, "q51"},      // plane1: fifth bits; meta: d, m
    // 16-bit floats in native blocks of 256 GGUF values; meta: g, which the repack writes.
    {30, 256, 512, 64, 0, 4, 8, "bf16"},
    {1, 256, 512, 64, 0, 4, 8, "f16"},
    {42, 64, 18, 8, 0, 2, 2, "q20"},      // GGUF's block_q2_0, one d per 64 elements; meta: d
    {QUANT_NVFP464_TYPE, 64, 40, 16, 0, 8, 2, "nvfp464"},   // meta: the 4 E4M3 scales, g
    {QUANT_MXFP8_TYPE, 32, 33, 32, 0, 1, 1, "mxfp8"},       // FP8's planes; meta: e, as MXFP4's
};

// The format that stores a GGUF tensor type (or a loader's);
// GGUF_FMT_COUNT when none does.
inline constexpr uint32_t gguf_format_of(uint32_t ggml_type) {
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format)
    if (kQuantFormats[format].ggml_type == ggml_type) return format;
  return GGUF_FMT_COUNT;
}

// The bytes of a row of `columns` values kept as stored, which a gather or
// the float projection reads in place: BF16's and F16's values each their
// own block (GGUF's, of one value), so any width; any other format's whole
// native blocks, else 0.
inline constexpr uint64_t quant_stored_row_bytes(uint32_t format, uint64_t columns) {
  if (format == GGUF_FMT_BF16 || format == GGUF_FMT_F16) return columns * 2;
  if (format >= GGUF_FMT_COUNT || columns % kQuantFormats[format].block_elements) return 0;
  return columns / kQuantFormats[format].block_elements * kQuantFormats[format].block_bytes;
}

// Whether a format is MLX affine, and its code bits.
inline constexpr bool quant_affine_format(uint32_t format) {
  return format >= GGUF_FMT_AF2G32 && format <= GGUF_FMT_AF8G128;
}
inline constexpr uint32_t quant_affine_bits(uint32_t format) { return (kQuantFormats[format].ggml_type >> 8) & 0xFF; }
// Whether a format's native block is the loader's rather than a GGUF type's:
// the MLX affine formats, NVFP4 (of both block sizes), FP8 and MXFP8.
inline constexpr bool quant_loader_format(uint32_t format) {
  return quant_affine_format(format) || format == GGUF_FMT_NVFP4 || format == GGUF_FMT_NVFP464 ||
         format == GGUF_FMT_FP8 || format == GGUF_FMT_MXFP8;
}
// Whether a format's values enter a bf16 matrix multiply exactly, as the
// Apple9 register tile takes them (kernels/decode/linear_gguf_sgmatrix.metal):
// every format's but F16's, whose 11-bit significands bf16 does not hold.
inline constexpr bool quant_register_format(uint32_t format) { return format != GGUF_FMT_F16; }
// Whether the int8 decode tiles (kernels/shared/gguf_linear.metal
// gguf_decode_int8_*) take a format: PQ2_0, whose two-bit codes q stand for
// q - 1, exact in int8, with one d per 128-input block.
inline constexpr bool quant_decode_int8_format(uint32_t format) { return format == GGUF_FMT_PQ20; }
// Whether the int8 prefill tile (kernels/prefill/gguf_prefill_int8.metal
// gguf_prefill_int8_*) takes a format: MLX affine 4 and 5 bits in groups of
// 64 and Q8_0, whose codes are int8 exactly, each Q8_0 32-group with its d.
inline constexpr bool quant_prefill_int8_format(uint32_t format) {
  return format == GGUF_FMT_AF4G64 || format == GGUF_FMT_AF5G64 || format == GGUF_FMT_Q80;
}
// Whether the MoE experts' int8 prefill passes (kernels/shared/moe_int8.metal
// moe_expert_int8_*) take a format: MLX affine 4 and 5 bits in groups of 64.
// Not Q8_0, which the dense tile takes: no real model has run Q8_0 experts on
// these passes, and their six routes an expert (ops::selectMoe) were measured
// on af4g64.
inline constexpr bool quant_moe_int8_format(uint32_t format) {
  return format == GGUF_FMT_AF4G64 || format == GGUF_FMT_AF5G64;
}
// The MLX affine format of b bits in groups of g; GGUF_FMT_COUNT when none
// (QUANT_AFFINE_TYPE holds b and g in a byte each).
inline constexpr uint32_t quant_affine_format_of(uint32_t bits, uint32_t group) {
  return bits <= 0xFF && group <= 0xFF ? gguf_format_of(QUANT_AFFINE_TYPE(bits, group)) : GGUF_FMT_COUNT;
}

// The chunk order slot of element e (0..31) of a group.
inline constexpr uint32_t quant_slot(uint32_t e) { return 8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3); }

// A plane tile holds one GGUF_TILE_COLUMNS column tile of every kernel, so
// any projection or expert of a multiple of 64 rows starts on a tile.
#define QUANT_TILE_ROWS 64u

// The columns a [N, K] tensor is a multiple of, from its format's native
// block of `block_elements` elements: one 64-input span of the activation
// tables for a block that fits in one (32 or 64 elements: Q2_0's 64 and the
// MLX affine formats' groups of 64 too), else a whole 256-input superblock
// (PQ2_0's 128-input blocks and the groups of 128 too).
inline constexpr uint32_t quant_column_unit(uint32_t block_elements) {
  return block_elements > 64 ? 256 : 64;
}

// Whether the Apple9 register tile (kernels/decode/linear_gguf_sgmatrix.metal)
// takes a tensor of `columns` inputs in `format`: a format it takes in rows
// of whole coefficient units, which for a format of several groups per meta
// unit span four 64-input spans (Q2_K's two), so 256 columns.
inline constexpr bool quant_register_tensor(uint32_t format, uint64_t columns) {
  return quant_register_format(format) && columns % (kQuantFormats[format].meta_groups > 1 ? 256 : 64) == 0;
}
// The inputs of the register tile's coefficient unit in `format`, of which
// each K partition takes whole ones: one 64-input span for a format of one
// group per meta unit, else four, and Q2_K's two, whose coefficients per span
// take twice Q4_K's storage.
inline constexpr uint32_t quant_register_unit(uint32_t format) {
  return kQuantFormats[format].meta_groups == 1 ? 64 : format == GGUF_FMT_Q2K ? 128 : 256;
}

// Whether a format stores at least four bits per weight.
inline constexpr bool quant_four_bits(uint32_t format) {
  return 2 * kQuantFormats[format].block_bytes >= kQuantFormats[format].block_elements;
}

// Index of (row, block) in a [rows / T][blocks][T] plane (blocks = G) or meta
// plane (blocks = meta units per row).
inline constexpr uint64_t quant_tile_index(uint32_t row, uint32_t block, uint32_t blocks) {
  return (uint64_t(row / QUANT_TILE_ROWS) * blocks + block) * QUANT_TILE_ROWS + row % QUANT_TILE_ROWS;
}

#undef QUANT_CONSTANT
// ---- end runtime/metal/abi/QuantFormat.h ----

// Vocabulary: a group is 32 inputs (the image's quantization unit), a span 64
// (Table16's); a plane tile is QUANT_TILE_ROWS rows of the weight image, a
// column tile the GGUF_TILE_COLUMNS output columns of one threadgroup (grid.x
// counts column tiles), and an expert tile the grouped rows of one MoE expert.
// The int8 prefill tile's activation group is the MLX weight group of 64
// inputs, two Q8_0 groups (GGUF_PREFILL_INT8_GROUP).
#define GGUF_TILE_COLUMNS 64u
// Decode tile of the staged kernels, which runs plans of up to
// GGUF_STAGED_ROWS rows, a decode step's or a prompt chunk's alike: two
// simdgroups, each staging its own GGUF_STAGED_COLUMNS columns
// GGUF_STAGED_STEP inputs at a time over 8, 16, 32, 48 or 64 rows
// (GGUF_ROW_GROUP_ROWS); a plan of more rows runs two row groups of the tile of
// half its rows in one dispatch, grid.z (ops::stagedRowGroups).
#define GGUF_ROW_GROUP_ROWS 64u
#define GGUF_STAGED_ROWS (2u * GGUF_ROW_GROUP_ROWS)
#define GGUF_STAGED_COLUMNS 32u
#define GGUF_STAGED_STEP 32u
#define GGUF_STAGED_THREADS (GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS * 32u)
// Decode tile of a gate/up pair of one format in one pass
// (LinearConfig::oneGateUpPass): one simdgroup staging the same
// GGUF_STAGED_COLUMNS columns of both tensors.
#define GGUF_GATE_UP_THREADS 32u
// Prefill tile of the staged kernels: GGUF_PREFILL_SIMDGROUPS simdgroups of
// GGUF_PREFILL_SIMDGROUP_ROWS rows share one stage of GGUF_PREFILL_STEP
// inputs of the tile's columns, which all threads dequantize.
#define GGUF_PREFILL_SIMDGROUP_ROWS 32u
#define GGUF_PREFILL_SIMDGROUPS 4u
#define GGUF_PREFILL_STEP 64u
#define GGUF_PREFILL_ROWS (GGUF_PREFILL_SIMDGROUP_ROWS * GGUF_PREFILL_SIMDGROUPS)
#define GGUF_PREFILL_THREADS (GGUF_PREFILL_SIMDGROUPS * 32u)
// Apple9 register tile: four simdgroups of GGUF_REGISTER_COLUMNS columns.
#define GGUF_REGISTER_COLUMNS 16u
#define GGUF_REGISTER_THREADS (GGUF_TILE_COLUMNS / GGUF_REGISTER_COLUMNS * 32u)
// Column tile (kernels/common/gguf_column_tile.h): a GEMV of at most
// GGUF_COLUMN_ROWS live rows in registers, which the MoE expert tiles run for
// an expert's only tile of that many rows and ops::Linear for a decode step of
// that many live rows (kernels/decode/linear_gguf_column.metal):
// GGUF_COLUMN_SPLITS simdgroups of GGUF_COLUMN_COLUMNS columns, one a lane,
// each summing its part of K.
#define GGUF_COLUMN_ROWS 2u
#define GGUF_COLUMN_COLUMNS 32u
#define GGUF_COLUMN_SPLITS 8u
#define GGUF_COLUMN_THREADS (GGUF_COLUMN_SPLITS * 32u)
struct GgufColumnParams {
  uint32_t rows;        // live rows, 1 to GGUF_COLUMN_ROWS
  uint32_t input_size;  // K
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column of the tensor
  uint32_t format;      // the tensor's GGUF_FMT_*, a one-pass gate/up's both tensors'
};
static_assert(sizeof(GgufColumnParams) == 20, "GGUF column parameters are 20 bytes on both sides");
// A fused projection's segments on the column tile (gguf_column_fused), as
// GgufDecodeFusedParams's on the staged tile.
struct GgufColumnFusedParams {
  uint32_t rows;        // live rows, 1 to GGUF_COLUMN_ROWS
  uint32_t input_size;  // K
  uint32_t out_stride;  // columns of a destination row
  uint32_t cols[3];     // columns per segment; 0 past the last
  uint32_t fmt[3];      // GGUF_FMT_* per segment
  uint32_t offset[3];   // first destination column per segment
};
static_assert(sizeof(GgufColumnFusedParams) == 48, "GGUF column fused parameters are 48 bytes on both sides");

// The register tile's activations, Table16 (kernels/common/gguf_sgmatrix.h):
// per eight-row tile of `width` inputs, the X^T table of width * 8 bfloat,
// then the fp32 chain seeds [width / 16][8 rows] and sums [width / 32][8
// rows]. Each 64-input span holds these values, seeds and sums per tile.
// A table tile holds GGUF_TABLE16_ROWS rows, the rows the register tile
// computes together; a plan's rows fill whole tiles, the last padded.
#define GGUF_TABLE16_ROWS 8u
#define GGUF_TABLE16_SPAN_INPUTS 64u
#define GGUF_TABLE16_SPAN_VALUES 512u
#define GGUF_TABLE16_SPAN_SEEDS 32u
#define GGUF_TABLE16_SPAN_SUMS 16u
inline constexpr uint64_t table16_sums32_offset(uint32_t width) { return uint64_t(width) / 16 * 8; }
inline constexpr uint64_t table16_sums_per_tile(uint32_t width) {
  return table16_sums32_offset(width) + uint64_t(width) / 32 * 8;
}

// Prefill tiles: the grid covers whole GGUF_PREFILL_ROWS-row tiles of the
// chunk; the simdgroups of the last tile whose rows start past `rows` skip
// their matmuls.
struct GgufPrefillParams {
  uint32_t input_size;  // K
  uint32_t rows;        // rows of the chunk
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column of this segment
};
static_assert(sizeof(GgufPrefillParams) == 16, "GGUF prefill parameters are 16 bytes on both sides");
// The residual prefill kernels over a view of the leading inputs of wider weight rows
// (gguf_prefill_<format>_r_leading_inputs): each row of the planes holds plane_input_size inputs, of which the
// projection reads the first prefill.input_size.
struct GgufPrefillLeadingParams {
  GgufPrefillParams prefill;
  uint32_t plane_input_size;
};
static_assert(sizeof(GgufPrefillLeadingParams) == 20,
              "GGUF leading-input prefill parameters are 20 bytes on both sides");
// The plain prefill kernels (gguf_prefill_<format>_a): up to three column segments of the kernel's format in one
// dispatch, tiles in segment order, as a fused decode dispatch takes them; a single tensor is one segment.
struct GgufPrefillSegmentsParams {
  uint32_t input_size;  // K
  uint32_t rows;        // rows of the chunk
  uint32_t out_stride;  // columns of a destination row
  uint32_t cols[3];     // columns per segment; 0 past the last
  uint32_t offset[3];   // first destination column per segment
};
static_assert(sizeof(GgufPrefillSegmentsParams) == 36, "GGUF segmented prefill parameters are 36 bytes on both sides");

// Int8 prefill tile (kernels/prefill/gguf_prefill_int8.metal), MLX affine
// weights in groups of 64 and Q8_0: gguf_prefill_int8_quantize writes each
// 64-input group of each chunk row as int8 codes and a float2 (scale, scale *
// sum of the codes); gguf_prefill_int8_<format>_<a|r|g> multiplies them by the
// weight codes in int8 and folds each group's affine correction (each Q8_0
// 32-group's d) in fp32. A tile is a column tile of
// GGUF_PREFILL_INT8_SIMDGROUPS simdgroups of 32 columns by
// GGUF_PREFILL_INT8_ROWS rows, for every format: af5g64 on 64-row tiles lost
// to the bf16 tile at 65-96 rows (0.95x, 16-core M5 Pro, measured).
#define GGUF_PREFILL_INT8_GROUP 64u
#define GGUF_PREFILL_INT8_SIMDGROUPS (GGUF_TILE_COLUMNS / 32u)
#define GGUF_PREFILL_INT8_ROWS 32u
#define GGUF_PREFILL_INT8_THREADS (GGUF_PREFILL_INT8_SIMDGROUPS * 32u)
// Threadgroups take this many row tiles down a column tile before the next
// column tile, so the column tile's weights stream once per 16 row tiles while
// those tiles' activations stay cached: walking every column tile of a row tile
// first re-streamed the weights per row tile, which left the M6's 2048-row
// af5g64 down projection (K 17408) at 0.95x the bf16 tile (measured).
#define GGUF_PREFILL_INT8_WALK 16u
// gguf_prefill_int8_quantize: one simdgroup per (row, group), this many per
// threadgroup; grid (group blocks, rows).
#define GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS 8u
struct GgufPrefillInt8QuantizeParams {
  uint32_t input_size;    // K
  uint32_t rows;          // rows of the chunk
  uint32_t storage_rows;  // row stride of the scales
};
static_assert(sizeof(GgufPrefillInt8QuantizeParams) == 12,
              "int8 prefill quantize parameters are 12 bytes on both sides");
// gguf_prefill_int8_<format>_<e>: grid (16, ceil(tiles / 16)), the walk of
// GGUF_PREFILL_INT8_WALK row tiles down each column tile.
struct GgufPrefillInt8Params {
  uint32_t input_size;        // K
  uint32_t rows;              // rows of the chunk
  uint32_t storage_rows;      // row stride of the scales
  uint32_t out_stride;        // columns of a destination row
  uint32_t out_offset;        // first destination column of the segment
  uint32_t plane_input_size;  // inputs a row of the planes holds (Projection::leadingInputs), K for none
  uint32_t column_tiles;      // the segment's tiles of GGUF_TILE_COLUMNS columns
};
static_assert(sizeof(GgufPrefillInt8Params) == 28, "int8 prefill parameters are 28 bytes on both sides");
// Scratch of `storage` rows of K inputs: the int8 codes in the table, and in
// the sums buffer float2 [K / 64][storage].
inline constexpr uint64_t gguf_prefill_int8_table_bytes(uint32_t k, uint64_t storage) {
  return uint64_t(k) * storage;
}
inline constexpr uint64_t gguf_prefill_int8_scale_bytes(uint32_t k, uint64_t storage) {
  return uint64_t(k / GGUF_PREFILL_INT8_GROUP) * storage * 8u;
}

// Decode tiles: the register tile, of plans of up to 32 rows where the
// device's calibration takes it (RegisterBoundary, ops/KernelCalibration.hpp),
// and the staged tile, of the other plans of up to 64 rows, a decode step's
// or a prompt chunk's, and of a decode step's of 65 to 128 rows in two row
// groups (ops::stagedRowGroups). One tensor per dispatch, over the dispatch's
// tiles (grid.x), K partitions (grid.y) and row groups (grid.z).
struct GgufDecodeParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions; 1 = no cross-threadgroup reduction
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column of the tensor
  uint32_t spread;      // staged tile: 1 = each column tile starts its K walk at its own step, 0 = lockstep
};
static_assert(sizeof(GgufDecodeParams) == 20, "GGUF decode parameters are 20 bytes on both sides");

// Decode of a fused projection: up to three column segments of any formats in
// one dispatch, tiles in segment order.
struct GgufDecodeFusedParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions of every segment
  uint32_t out_stride;  // columns of a destination row
  uint32_t cols[3];     // columns per segment; 0 past the last
  uint32_t fmt[3];      // GGUF_FMT_* per segment
  uint32_t offset[3];   // first destination column per segment
  uint32_t spread;      // as GgufDecodeParams', each segment's tiles from their index in it
};
static_assert(sizeof(GgufDecodeFusedParams) == 52, "GGUF fused decode parameters are 52 bytes on both sides");

struct GgufEmbedParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t hidden;
};
static_assert(sizeof(GgufEmbedParams) == 12, "GGUF embedding parameters are 12 bytes on both sides");

// Prism ML's input rotation (kernels/shared/gguf_rotation.metal): weights
// stored for rotated inputs multiply H (D x), H the normalized Walsh-Hadamard
// transform of every block of GGUF_ROTATION_BLOCK inputs and D their int8
// signs (+1 or -1); an embedding row stored rotated gathers as D (H r).
// One threadgroup of GGUF_ROTATION_THREADS threads per block and row.
#define GGUF_ROTATION_BLOCK 1024u
#define GGUF_ROTATION_THREADS 256u
struct GgufRotationParams {
  uint32_t width; // a multiple of GGUF_ROTATION_BLOCK
};
static_assert(sizeof(GgufRotationParams) == 4, "GGUF rotation parameters are 4 bytes on both sides");

// fp32 projections of GGUF float tensors (kernels/shared/gguf_float.metal),
// up to GGUF_FLOAT_PROJECTIONS of one input in one dispatch, their column
// tiles in projection order: out_p[r][out_offset[p] + n] = sum_k x[r][k] *
// W_p[n][k] for rows r < rows, W_p the values as stored
// ([output_size[p]][input_size]).
#define GGUF_FLOAT_PROJECTIONS 2u
struct GgufFloatParams {
  uint32_t rows;
  uint32_t input_size;                          // K, a multiple of 8
  uint32_t output_size[GGUF_FLOAT_PROJECTIONS]; // N, a multiple of 8; 0 past the last projection
  uint32_t out_stride[GGUF_FLOAT_PROJECTIONS];  // columns of a destination row
  uint32_t out_offset[GGUF_FLOAT_PROJECTIONS];  // first destination column
};
static_assert(sizeof(GgufFloatParams) == 32, "GGUF float parameters are 32 bytes on both sides");

// Int8-activation decode of rotated PQ2_0 projections (ops::LinearConfig::decodeInt8): gguf_rotate_int8
// (kernels/shared/gguf_rotation.metal) writes the rotated input as int8 rows [rows][width], each group of 32 inputs
// in the tile's stage order (gguf_decode_int8_position), then fp32 scales [rows][width / GGUF_ROTATION_BLOCK]:
// input k of row r is int8 * scale[r][k / GGUF_ROTATION_BLOCK]. The tiles (kernels/shared/gguf_linear.metal
// gguf_decode_int8_*) multiply them with the PQ2_0 codes in int8, taking the staged decode kernels' parameters
// (GgufDecodeParams, GgufDecodeFusedParams), each K partition of whole PQ2_0 blocks.
struct GgufRotationInt8Params {
  uint32_t width;  // K, the row stride of the input and of the int8 rows
  uint32_t rows;   // rows of int8 storage (the tile's height): the scales start at byte rows * K
};
static_assert(sizeof(GgufRotationInt8Params) == 8, "GGUF int8 rotation parameters are 8 bytes on both sides");
// The bytes of `rows` rows of int8 storage of `width` inputs: the int8 rows, then their scales.
inline constexpr uint64_t gguf_rotation_int8_bytes(uint32_t width, uint64_t rows) {
  return uint64_t(width) * rows + uint64_t(width / GGUF_ROTATION_BLOCK) * rows * 4u;
}
// The stage position, within its group of 32, of input t of the group: the PQ2_0 tile stages the codes of chunk
// word w (chunks 2w, 2w + 1) as four words (w >> 2e) & 0x03030303 (e = 0..3), so byte 16 w + 4 e + k holds code e
// of byte k of the word, which is input 16 (k & 1) + 4 (2 w + (k >> 1)) + e.
inline constexpr uint32_t gguf_decode_int8_position(uint32_t t) {
  return 16u * (((t & 15u) >> 2) >> 1) + 4u * (t & 3u) + ((t >> 4) | ((((t & 15u) >> 2) & 1u) << 1));
}
// ---- end runtime/metal/abi/Gguf.h ----
// ---- begin runtime/metal/kernels/common/gguf_prefill_int8_tile.h (Splash 2afc733, Apache-2.0) ----
// The int8-activation tile for MLX affine 4- and 5-bit weights in groups of 64 (af4g64, af5g64) and Q8_0, which the
// dense prefill GEMMs (kernels/prefill/gguf_prefill_int8.metal) and the MoE experts' prefill passes
// (kernels/shared/moe_int8.metal) run: see gguf_prefill_int8.metal for its arithmetic and fragment schedules.
// ---- begin runtime/metal/kernels/common/gguf_tile.h (Splash 2afc733, Apache-2.0) ----
// ---- begin runtime/metal/kernels/common/activation.h (Splash 2afc733, Apache-2.0) ----
#include <metal_stdlib>
using namespace metal;

// SiLU and the logistic sigmoid of every kernel, e^-x taken as
// fast::exp2(-x * log2(e)). A gated product multiplies splash_silu(gate)
// into its other factor, so silu(g) * up evaluates g / (1 + e^-g) * up.
inline float splash_silu(float x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float2 splash_silu(float2 x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float splash_sigmoid(float x) {
  return 1.0f / (1.0f + fast::exp2(-1.44269504089f * x));
}

// GELU, x Phi(x), with the normal CDF Phi taken through tanh
// (gelu_pytorch_tanh) or through erf. erf is Abramowitz & Stegun 7.1.26,
// whose error of 1.5e-7 lies far below bf16 rounding.
inline float splash_gelu_tanh(float x) {
  return 0.5f * x *
      (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
inline float splash_erf(float x) {
  float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * x);
  float polynomial = t * (0.254829592f + t * (-0.284496736f +
      t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}
inline float splash_gelu_erf(float x) {
  return 0.5f * x * (1.0f + splash_erf(x * 0.7071067811865476f));
}
// ---- end runtime/metal/kernels/common/activation.h ----

// What both GGUF GEMM families (kernels/shared/gguf_linear.metal and
// kernels/decode/linear_gguf_sgmatrix.metal) do to a finished fp32 sum.
enum GgufEpilogue : ushort { EpNone, EpResidual, EpUpWithGate };

// The output of sum v at element `at`: v, v plus the residual aux[at], or the bf16-rounded up value v times silu of
// the gate aux[at], as the destination's type Out: bf16, or fp32 (the plain epilogue's logits).
template <GgufEpilogue Ep, class Out = bfloat> inline Out gguf_epilogue(float v, device const bfloat *aux, ulong at) {
  if constexpr (Ep == EpResidual) v += float(aux[at]);
  if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[at]));
  return Out(v);
}
// ---- end runtime/metal/kernels/common/gguf_tile.h ----

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// What a tile reads and writes: K inputs a row, `rows` rows of codes from the codes' start (the scales' row stride
// storage_rows), destination rows of out_stride columns from out_offset on, and planes of plane_input_size inputs a
// row.
struct PrefillInt8Shape {
  uint input_size, rows, storage_rows, out_stride, out_offset, plane_input_size;
};

constant constexpr int kInt8FragM = 16, kInt8FragN = 32, kInt8FragK = 16, kInt8RowFrags = 2, kInt8Elements = 16;

// This lane's (column, row) in matmul2d's int8 16x16 fragment layout, the operands' and the destination's, which
// gguf-projection checks on each family. Taking the destination's from get_multidimensional_index instead loads each
// element's scales alone: up to 25% more time (16-core M5 Pro, measured).
inline short2 prefill_int8_coord(ushort lane) {
  const short qid = lane >> 2;
  return short2(((qid & 2) | (lane & 1)) * 4, (qid & 4) | ((lane >> 1) & 3));
}

// Four 5-bit codes: the step's nibbles and the fifth bits of chunk byte `b` at slots {0, 4, 1, 5} (or, shifted by
// two, {2, 6, 3, 7}) moved to bit 4 of each byte.
inline uint prefill_int8_fifth(uint b) {
  return ((b & 1u) | ((b & 0x10u) << 4) | ((b & 2u) << 15) | ((b & 0x20u) << 19)) << 4;
}

// A tile of GGUF_PREFILL_INT8_SIMDGROUPS simdgroups, each 32 rows (two 16-row fragments) by 32 columns, at (column
// tile, row tile) `tile`, its sums times `scale` (an expert pass's expert scale, moe_int8.metal) before the epilogue.
// BITS is the weights' code width: 4 or 5 for af4g64 and af5g64, 8 for Q8_0.
template <int BITS, GgufEpilogue Ep>
inline void prefill_int8_tile(device const char *codes, device const float2 *scales, device const uchar *w0,
                              device const uchar *w1, device const uchar *meta, device bfloat *output,
                              device const bfloat *aux, PrefillInt8Shape p, uint2 tile, ushort lane,
                              ushort simd_group, float scale = 1.0f) {
  using namespace mpp::tensor_ops;
  constexpr int BM = kInt8RowFrags * kInt8FragM, BN = GGUF_TILE_COLUMNS;
  static_assert(BN == kInt8FragN * GGUF_PREFILL_INT8_SIMDGROUPS, "the simdgroups cover the column tile");
  static_assert(BM == GGUF_PREFILL_INT8_ROWS, "a simdgroup holds the tile's rows");
  constexpr uint T = QUANT_TILE_ROWS;
  const int K = int(p.input_size), M = int(p.rows), groups = K / GGUF_PREFILL_INT8_GROUP;
  const uint plane_groups32 = p.plane_input_size / 32, plane_units = p.plane_input_size / GGUF_PREFILL_INT8_GROUP;
  const int row_base = int(tile.y) * BM, col_base = int(tile.x) * BN + simd_group * kInt8FragN;
  const short2 coord = prefill_int8_coord(lane);
  const int cx = coord.x >> 2;

  constexpr auto desc = matmul2d_descriptor(kInt8FragM, kInt8FragN, kInt8FragK, false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = matmul2d_descriptor(kInt8FragM, kInt8FragN, kInt8FragK, false, true, false,
                                                matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<desc_set, execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  float cf[kInt8RowFrags][kInt8Elements];
#pragma unroll
  for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) cf[i][e] = 0.0f;

  // The lane's weight column n_lane + 8 (q & 1) + 16 (q >> 1) for right-operand run q, all in one plane tile (a
  // 64-column tile never straddles one), its record record0 of 32-group 0, a 32-group T records further on. MLX
  // affine: plane0 records of 16 bytes, plane1 of 4; the lane reads bytes 8 (x & 1) .. +7 of 32-group 2g + (x >> 1).
  const uint n_lane = uint(col_base + coord.y);
  const ulong record0 = ulong(n_lane / T) * plane_groups32 * T + n_lane % T;
  device const uchar *wl0 = w0 + record0 * 16 + 8 * (cx & 1);
  device const uchar *wl1 = w1 + record0 * 4 + 2 * (cx & 1);
  const int h32 = cx >> 1;
  // Meta (s, z as bf16 in one uint) of the lane's destination columns n_run + 16h + j: 4 consecutive records (an
  // affine format's; Q8_0's d are halves).
  const uint n_run = uint(col_base + coord.x);
  device const uint *ml = (device const uint *)(meta) + ulong(n_run / T) * plane_units * T + n_run % T;
  const int m_base = row_base + coord.y;

  if constexpr (BITS == 8) {
    // Q8_0: plane0 records of 32 int8 codes, chunk c at bytes 8c .. 8c + 7 (slots 8c .. 8c + 7), meta a half d per
    // 32-group. Steps 2h and 2h + 1 of a 64-input group take its 32-group h alone, lane column x its chunk x, chunk
    // slots {0, 4, 1, 5} in the even step and {2, 6, 3, 7} in the odd one, which the quantizer's order holds at
    // 32h + 8x + 4 (t & 1) (prefill_int8_position): the steps' int32 sums are one 32-group's, folded into fp32 with
    // its d.
    device const uchar *wq = w0 + record0 * 32 + 8 * cx;
    device const half *mq = (device const half *)(meta) + ulong(n_run / T) * plane_groups32 * T + n_run % T;
    for (int g = 0; g < groups; ++g) {
      float sa[kInt8RowFrags][2];
#pragma unroll
      for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          sa[i][r] = scales[ulong(g) * p.storage_rows + min(m_base + i * kInt8FragM + r * 8, M - 1)].x;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const ulong g32 = ulong(2 * g + h) * T;
        uint2 wv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          wv[q] = *((device const uint2 *)(wq + (g32 + (q & 1) * 8 + (q >> 1) * 16) * 32));
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const char4 lo = as_type<char4>(wv[q].x), hi = as_type<char4>(wv[q].y);
            const int base = (q >> 1) * 8 + (q & 1) * 4;
            ct_b[base + 0] = t ? lo.z : lo.x; ct_b[base + 1] = t ? hi.z : hi.x;
            ct_b[base + 2] = t ? lo.w : lo.y; ct_b[base + 3] = t ? hi.w : hi.y;
          }
#pragma unroll
          for (int hf = 0; hf < kInt8RowFrags; ++hf) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
              const int m = min(m_base + hf * kInt8FragM + r * 8, M - 1);
              const ulong at = ulong(m) * K + g * GGUF_PREFILL_INT8_GROUP + 32 * h + cx * 8 + t * 4;
              const char4 quad = as_type<char4>(*((device const uint *)(codes + at)));
              ct_a[r * 4 + 0] = quad.x; ct_a[r * 4 + 1] = quad.y; ct_a[r * 4 + 2] = quad.z; ct_a[r * 4 + 3] = quad.w;
            }
            if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
            else { if (hf == 0) op.run(ct_a, ct_b, acc0); else op.run(ct_a, ct_b, acc1); }
          }
        }
        // The 32-group's d of the lane's two runs of four columns.
        const half4 d0 = *((device const half4 *)(mq + g32)), d1 = *((device const half4 *)(mq + g32 + 16));
#pragma unroll
        for (int e = 0; e < kInt8Elements; ++e) {
          const int r = (e & 7) >> 2;
          const float d = float((e >> 3) ? d1[e & 3] : d0[e & 3]);
          cf[0][e] = fma(sa[0][r] * d, float(acc0[e]), cf[0][e]);
          cf[1][e] = fma(sa[1][r] * d, float(acc1[e]), cf[1][e]);
        }
      }
    }
  } else {
    for (int g = 0; g < groups; ++g) {
      const ulong g32 = ulong(2 * g + h32) * T;
      uint2 wv[4];
      uint hb[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int off = (q & 1) * 8 + (q >> 1) * 16;
        wv[q] = *((device const uint2 *)(wl0 + (g32 + off) * 16));
        if (BITS == 5) hb[q] = *((device const ushort *)(wl1 + (g32 + off) * 4));
      }
#pragma unroll
      for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint bits = (wv[q][t >> 1] >> (4 * (t & 1))) & 0x0f0f0f0fu;
          if (BITS == 5) bits |= prefill_int8_fifth((hb[q] >> (8 * (t >> 1) + 2 * (t & 1))) & 0xFFu);
          const char4 quad = as_type<char4>(bits);
          const int base = (q >> 1) * 8 + (q & 1) * 4;
          ct_b[base + 0] = quad.x; ct_b[base + 1] = quad.y; ct_b[base + 2] = quad.z; ct_b[base + 3] = quad.w;
        }
#pragma unroll
        for (int hf = 0; hf < kInt8RowFrags; ++hf) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int m = min(m_base + hf * kInt8FragM + r * 8, M - 1);
            const ulong at = ulong(m) * K + g * GGUF_PREFILL_INT8_GROUP + cx * 16 + t * 4;
            const char4 quad = as_type<char4>(*((device const uint *)(codes + at)));
            ct_a[r * 4 + 0] = quad.x; ct_a[r * 4 + 1] = quad.y; ct_a[r * 4 + 2] = quad.z; ct_a[r * 4 + 3] = quad.w;
          }
          if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
          else { if (hf == 0) op.run(ct_a, ct_b, acc0); else op.run(ct_a, ct_b, acc1); }
        }
      }
      // The group's correction: s, z of the lane's two runs of four columns, and the scales of its four rows.
      const uint4 mv0 = *((device const uint4 *)(ml + ulong(g) * T));
      const uint4 mv1 = *((device const uint4 *)(ml + ulong(g) * T + 16));
      float2 sa[kInt8RowFrags][2];
#pragma unroll
      for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          sa[i][r] = scales[ulong(g) * p.storage_rows + min(m_base + i * kInt8FragM + r * 8, M - 1)];
#pragma unroll
      for (int e = 0; e < kInt8Elements; ++e) {
        const int r = (e & 7) >> 2;
        const uint mt = (e >> 3) ? mv1[e & 3] : mv0[e & 3];
        const float s = float(as_type<bfloat>(ushort(mt & 0xFFFF))), z = float(as_type<bfloat>(ushort(mt >> 16)));
        cf[0][e] = fma(sa[0][r].x * s, float(acc0[e]), fma(z, sa[0][r].y, cf[0][e]));
        cf[1][e] = fma(sa[1][r].x * s, float(acc1[e]), fma(z, sa[1][r].y, cf[1][e]));
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) {
      const int m = m_base + i * kInt8FragM + ((e & 7) >> 2) * 8;
      if (m >= M) continue;
      const ulong o = ulong(m) * p.out_stride + p.out_offset + n_run + (e & 3) + 16 * (e >> 3);
      output[o] = gguf_epilogue<Ep>(cf[i][e] * scale, aux, o);
    }
}
// ---- end runtime/metal/kernels/common/gguf_prefill_int8_tile.h ----
// ---- begin runtime/metal/kernels/common/int8_activations.h (Splash 2afc733, Apache-2.0) ----
#include <metal_stdlib>
using namespace metal;

// The int8-activation quantizers' rule, shared by gguf_rotate_int8 (a row's rotation block),
// gguf_prefill_int8_quantize (a row's 64-input group) and the attention tiles' INT8 queries (a fused row), each of
// which reduces its block's largest magnitude m itself:
// - an input x codes as q = clamp(rint(x * (127 / m)), -127, 127), ties to even, and stands for q * s, s = m / 127;
//   both quotients are correctly rounded (precise::divide, where fast math's divide is within 2.5 ulp), so a host
//   reference computes the same codes and scales bit for bit;
// - a block whose m is 2^-100 or less, zero among them, codes as zeros with scale 0: its products, 2^-100 times the
//   weights at most, vanish beside those of any input of normal size, and 127 / m stays finite (it overflows below
//   2^-121);
// - a block holding an infinity or a NaN, whose codes alone would round to finite values, takes a NaN scale, made
//   from its bits so that fast math cannot fold it, and its row goes on in NaN as it would in bf16: a projection's
//   outputs are not finite; an attention row's scores and softmax statistics are NaN, which the reduce writes as a
//   row of zeros.
struct Int8Block {
  float scale;    // s
  float inverse;  // 127 / m
};
inline Int8Block int8_activation_block(float largest) {
  const bool coded = largest > 0x1p-100f;
  return {coded ? precise::divide(largest, 127.0f) : 0.0f, coded ? precise::divide(127.0f, largest) : 0.0f};
}
// The code of input x, or of four, in a block of inverse `inverse`.
inline int int8_activation_code(float x, float inverse) { return int(clamp(rint(x * inverse), -127.0f, 127.0f)); }
inline int4 int8_activation_code(float4 x, float inverse) { return int4(clamp(rint(x * inverse), -127.0f, 127.0f)); }
// A block's value, its scale or a product of it, as stored: its bits, or a NaN's where the block is not finite.
inline uint int8_activation_bits(float value, bool finite) { return finite ? as_type<uint>(value) : 0x7fc00000u; }
// A bf16 input is found non-finite by its exponent bits: fast math may fold isfinite() to true.
inline bool int8_activation_not_finite(bfloat value) { return (as_type<ushort>(value) & 0x7f80) == 0x7f80; }
inline bool int8_activation_not_finite(bfloat4 value) {
  return any((as_type<ushort4>(value) & ushort(0x7f80)) == ushort(0x7f80));
}
// ---- end runtime/metal/kernels/common/int8_activations.h ----

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// The schedule position, within its 32-group, of the input at slot s of the planes: chunk c = s >> 3 is lane half
// c >> 1 and step half c & 1; chunk slot u holds step bit (u >> 1) & 1 and lane input ((u & 1) << 1) | (u >> 2).
inline uint prefill_int8_position(uint slot) {
  const uint c = slot >> 3, u = slot & 7;
  return 16 * (c >> 1) + 4 * (2 * (c & 1) + ((u >> 1) & 1)) + (((u & 1) << 1) | (u >> 2));
}

// grid (groups / GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS, rows): simdgroup s of threadgroup (b, row) quantizes group
// b * GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS + s of the row by the int8-activation rule
// (kernels/common/int8_activations.h), lane l its inputs l and 32 + l, and stores the group's scale and the scale
// times its codes' sum.
kernel void gguf_prefill_int8_quantize(device const bfloat *input [[buffer(0)]], device char *codes [[buffer(1)]],
                                       device float2 *scales [[buffer(2)]],
                                       constant GgufPrefillInt8QuantizeParams &p [[buffer(3)]],
                                       uint2 tid [[threadgroup_position_in_grid]],
                                       ushort lane [[thread_index_in_simdgroup]],
                                       ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint g = tid.x * GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS + simd_group, row = tid.y;
  if (g >= p.input_size / GGUF_PREFILL_INT8_GROUP || row >= p.rows) return;
  const ulong at = ulong(row) * p.input_size + g * GGUF_PREFILL_INT8_GROUP;
  const bfloat b0 = input[at + lane], b1 = input[at + 32 + lane];
  const float x0 = float(b0), x1 = float(b1);
  const Int8Block block = int8_activation_block(simd_max(max(fabs(x0), fabs(x1))));
  const int q0 = int8_activation_code(x0, block.inverse), q1 = int8_activation_code(x1, block.inverse);
  const uint position = prefill_int8_position(quant_slot(lane));
  codes[at + position] = char(q0);
  codes[at + 32 + position] = char(q1);
  const int sum = simd_sum(q0 + q1);
  const bool finite = simd_all(!int8_activation_not_finite(b0) && !int8_activation_not_finite(b1));
  if (lane != 0) return;
  ((device uint2 *)scales)[ulong(g) * p.storage_rows + row] =
      uint2(int8_activation_bits(block.scale, finite), int8_activation_bits(block.scale * float(sum), finite));
}

// The (column tile, row tile) of tile index `at` of grid.x column tiles by grid.y row tiles:
// GGUF_PREFILL_INT8_WALK row tiles take each column tile in turn before the next column tile, the last block of row
// tiles its remainder.
inline uint2 prefill_int8_walk(uint at, uint2 grid) {
  constexpr uint S = GGUF_PREFILL_INT8_WALK;
  const uint block = at / (S * grid.x), height = min(S, grid.y - block * S), i = at - block * S * grid.x;
  return uint2(i / height, block * S + i % height);
}

// grid (GGUF_PREFILL_INT8_WALK, its tiles / GGUF_PREFILL_INT8_WALK rounded up), GGUF_PREFILL_INT8_THREADS threads:
// threadgroup (x, y) takes tile index y GGUF_PREFILL_INT8_WALK + x of the segment's column tiles by the chunk's row
// tiles of GGUF_PREFILL_INT8_ROWS (prefill_int8_walk), and those past the tiles none. A run of 16 threadgroups along
// x is so one column tile's row tiles of a block, whatever the column count; a grid of (column tiles, row tiles)
// kept them together only where the column tiles were a multiple of 16 (M6: 260 column tiles took 29% longer than
// 256 at 2048 rows). aux (buffer 6) is the residual or the gate, the output for the plain epilogue, which reads none.
#define GGUF_PREFILL_INT8_KERNEL(B, f, ep, Ep)                                                                         \
  kernel void gguf_prefill_int8_##f##_##ep(                                                                            \
      device const char *codes [[buffer(0)]], device const float2 *scales [[buffer(1)]],                               \
      device const uchar *w0 [[buffer(2)]], device const uchar *w1 [[buffer(3)]],                                      \
      device const uchar *meta [[buffer(4)]], device bfloat *output [[buffer(5)]],                                     \
      device const bfloat *aux [[buffer(6)]], constant GgufPrefillInt8Params &p [[buffer(7)]],                         \
      uint2 tid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],                           \
      ushort simd_group [[simdgroup_index_in_threadgroup]]) {                                                          \
    const uint rows = (p.rows + GGUF_PREFILL_INT8_ROWS - 1) / GGUF_PREFILL_INT8_ROWS,                                  \
               at = tid.y * GGUF_PREFILL_INT8_WALK + tid.x;                                                            \
    if (at >= p.column_tiles * rows) return;                                                                           \
    prefill_int8_tile<B, Ep>(codes, scales, w0, w1, meta, output, aux,                                                 \
                             {p.input_size, p.rows, p.storage_rows, p.out_stride, p.out_offset, p.plane_input_size},    \
                             prefill_int8_walk(at, uint2(p.column_tiles, rows)), lane, simd_group);                    \
  }
#define GGUF_PREFILL_INT8_FORMAT(B, f)                                                                                 \
  GGUF_PREFILL_INT8_KERNEL(B, f, a, EpNone) GGUF_PREFILL_INT8_KERNEL(B, f, r, EpResidual)                              \
  GGUF_PREFILL_INT8_KERNEL(B, f, g, EpUpWithGate)
GGUF_PREFILL_INT8_FORMAT(4, af4g64)
GGUF_PREFILL_INT8_FORMAT(5, af5g64)
GGUF_PREFILL_INT8_FORMAT(8, q80)
#undef GGUF_PREFILL_INT8_FORMAT
#undef GGUF_PREFILL_INT8_KERNEL
// ---- end runtime/metal/kernels/prefill/gguf_prefill_int8.metal ----
