// Vendored from Splash (https://github.com/incoai/splash), revision
// 2afc7337aacc044722fa95e4cb3da75c281a19a8, Apache License 2.0 (LICENSE-APACHE-2.0).
// Quantized GGUF decoding follows ggml (MIT, LICENSE-GGML). Each original
// header is inlined once at its first include, between begin/end markers; the
// code is otherwise unchanged. Source hashes: provenance.json.
// ---- begin runtime/metal/kernels/decode/linear_gguf_column.metal (Splash 2afc733, Apache-2.0) ----
// Decode projections of at most GGUF_COLUMN_ROWS live rows (ops::LinearTile::GgufColumn): a GEMV in registers on the
// column tile the MoE expert tiles of few rows run (kernels/common/gguf_column_tile.h), each lane one output column, in
// the format its tensor holds at run time. The staged tiles run a dequantize, barrier and matmul chain per 32-input
// step on padded 8-row MPP tiles, and the Apple9 register tile reads a table its producer must write, which bound a
// step of one or two live rows above DRAM: the column tile reads the live rows alone. One pipeline of each kernel
// decodes every format: created cold, each took 0.31-0.64 s on the M5 Max, a one-format staged tile 0.06-0.09 s.
//
// One threadgroup: GGUF_COLUMN_SPLITS simdgroups over the same 32 columns, simdgroup s summing groups
// [s G / S, (s + 1) G / S) of K (G = K / 32) for every live row (column_dot), then simdgroup 0 adding the parts in
// simdgroup order, so a column's value depends neither on the grid nor on the other columns, and storing it as the
// tiles do (kernels/common/gguf_tile.h): bf16, or fp32 (the plain epilogue's logits), plus the residual, or a one-pass
// gate/up's bf16(up) * silu(bf16(gate)).
#pragma clang fp reassociate(off)
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
// ---- begin runtime/metal/kernels/common/gguf_column_tile.h (Splash 2afc733, Apache-2.0) ----
// ---- begin runtime/metal/kernels/common/gguf_staged.h (Splash 2afc733, Apache-2.0) ----
// ---- begin runtime/metal/kernels/common/quant_formats.h (Splash 2afc733, Apache-2.0) ----
// ---- begin runtime/metal/abi/QuantTables.h (Splash 2afc733, Apache-2.0) ----

// Decode-only value tables of the GGUF formats (metal/abi/QuantFormat.h),
// shared by the GEMM kernels, the token gathers and the host reference decoder
// of the tests. Of the prepared bytes, only the F32 values the loader writes
// of an MLX mxfp4 tensor depend on one (kFP4Values, model/GgufPreparation.cpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define QUANT_CONSTANT constant constexpr
#else
#include <stdint.h>
#define QUANT_CONSTANT inline constexpr
#endif

// The IQ4_NL and IQ4_XS values by 4-bit index and the IQ3_S grid by 9-bit
// index, an entry holding the magnitudes of four consecutive elements in its
// bytes 0..3: llama.cpp's kvalues_iq4nl and iq3s_grid, whose MIT notice is in
// THIRD_PARTY_NOTICES.
QUANT_CONSTANT int8_t kIQ4NLValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
QUANT_CONSTANT uint32_t kIQ3SGrid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

// The IQ3_XXS grid by 8-bit index, an entry holding the magnitudes of four
// consecutive elements in its bytes 0..3; the IQ2 grids (IQ2_XXS by 8-bit,
// IQ2_XS by 9-bit and IQ2_S by 10-bit index), an entry holding the magnitudes
// of eight in its bytes 0..7; the IQ1 grid by 11-bit index, an entry holding
// grid value + 1 (0..2) of element j in the low nibble of its byte j and of
// element j + 4 in the high one (j < 4); and the MXFP4 values, twice the E2M1
// values, by 4-bit index: llama.cpp's iq3xxs_grid, iq2xxs_grid, iq2xs_grid,
// iq2s_grid, iq1s_grid_gpu and kvalues_mxfp4, whose MIT notice is in
// THIRD_PARTY_NOTICES.
QUANT_CONSTANT uint32_t kIQ3XXSGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
};
QUANT_CONSTANT uint64_t kIQ2XXSGrid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908
};
QUANT_CONSTANT uint64_t kIQ2XSGrid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b
};
QUANT_CONSTANT uint64_t kIQ2SGrid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b
};
QUANT_CONSTANT uint32_t kIQ1SGrid[2048] = {
    0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101, 0x00020000,
    0x00020002, 0x00020200, 0x00020202, 0x01000101, 0x01010001, 0x01010100, 0x01010102, 0x01020101,
    0x02000000, 0x02000002, 0x02000200, 0x02000202, 0x02010101, 0x02020000, 0x02020002, 0x02020200,
    0x02020202, 0x00000110, 0x00000111, 0x00010011, 0x00010110, 0x00010112, 0x00010211, 0x00010212,
    0x00020111, 0x01000011, 0x01000112, 0x01000211, 0x01010012, 0x01010111, 0x01010212, 0x01020011,
    0x01020110, 0x01020112, 0x01020210, 0x02000111, 0x02010011, 0x02010110, 0x02010112, 0x02020111,
    0x00000020, 0x00000022, 0x00000220, 0x00000222, 0x00010121, 0x00020020, 0x00020022, 0x00020220,
    0x00020222, 0x01000121, 0x01010021, 0x01010221, 0x01020120, 0x01020221, 0x02000020, 0x02000022,
    0x02000220, 0x02000222, 0x02010021, 0x02010121, 0x02010221, 0x02020020, 0x02020022, 0x02020220,
    0x02020222, 0x00011001, 0x00011100, 0x00011102, 0x00021101, 0x01001001, 0x01001201, 0x01011101,
    0x01011202, 0x01021100, 0x01021101, 0x02011001, 0x02011201, 0x02021101, 0x00001011, 0x00001110,
    0x00001111, 0x00001112, 0x00011111, 0x00011210, 0x00011212, 0x00021211, 0x01001010, 0x01001111,
    0x01001212, 0x01011010, 0x01011011, 0x01011110, 0x01011111, 0x01011112, 0x01011211, 0x01021010,
    0x01021012, 0x01021111, 0x01021210, 0x01021212, 0x02001011, 0x02011011, 0x02011111, 0x02011210,
    0x02011212, 0x02021011, 0x02021110, 0x02021111, 0x02021112, 0x02021211, 0x00011120, 0x00011221,
    0x01001021, 0x01001120, 0x01011020, 0x01011022, 0x01011121, 0x01011220, 0x01021020, 0x01021021,
    0x01021122, 0x01021221, 0x02001121, 0x02011021, 0x02011120, 0x02011221, 0x00002000, 0x00002002,
    0x00002200, 0x00002202, 0x00012101, 0x00022000, 0x00022002, 0x00022200, 0x00022202, 0x01002101,
    0x01012001, 0x01012102, 0x01022101, 0x02002000, 0x02002002, 0x02002200, 0x02002202, 0x02012101,
    0x02022000, 0x02022002, 0x02022200, 0x02022202, 0x00002111, 0x00012011, 0x00012110, 0x00012211,
    0x00022110, 0x00022111, 0x01002011, 0x01012010, 0x01012011, 0x01012111, 0x01022011, 0x01022110,
    0x01022211, 0x02012011, 0x02012110, 0x02012112, 0x02012211, 0x02022111, 0x00002020, 0x00002022,
    0x00002220, 0x00002222, 0x00012121, 0x00022020, 0x00022022, 0x00022220, 0x00022222, 0x01002121,
    0x01012021, 0x01012221, 0x01022021, 0x01022121, 0x02002020, 0x02002022, 0x02002121, 0x02002220,
    0x02002222, 0x02012121, 0x02022020, 0x02022022, 0x02022220, 0x02022222, 0x00110000, 0x00110001,
    0x00110100, 0x00110201, 0x00120100, 0x00120101, 0x01100001, 0x01100100, 0x01110000, 0x01110101,
    0x01110200, 0x01120001, 0x01120100, 0x01120101, 0x01120201, 0x02110001, 0x02110100, 0x02110102,
    0x02120001, 0x02120101, 0x00100011, 0x00100110, 0x00100112, 0x00100211, 0x00110010, 0x00110012,
    0x00110111, 0x00110210, 0x00120011, 0x00120110, 0x00120211, 0x01100111, 0x01100212, 0x01110010,
    0x01110011, 0x01110012, 0x01110110, 0x01110111, 0x01110112, 0x01110211, 0x01120010, 0x01120111,
    0x02100110, 0x02110012, 0x02110111, 0x02120011, 0x02120110, 0x00110021, 0x00110120, 0x00110122,
    0x00120121, 0x01100020, 0x01100122, 0x01100221, 0x01110022, 0x01110121, 0x01110220, 0x01110222,
    0x01120120, 0x01120122, 0x02100121, 0x02110021, 0x02110120, 0x02110122, 0x02120121, 0x00101001,
    0x00101102, 0x00101201, 0x00111100, 0x00111101, 0x00111200, 0x00111201, 0x00121001, 0x00121102,
    0x01101001, 0x01101101, 0x01101102, 0x01101200, 0x01101202, 0x01111001, 0x01111100, 0x01111101,
    0x01111102, 0x01111201, 0x01121002, 0x01121101, 0x01121200, 0x02101100, 0x02101201, 0x02111000,
    0x02111100, 0x02111101, 0x02111200, 0x02111201, 0x02111202, 0x02121001, 0x02121100, 0x02121101,
    0x02121201, 0x00101012, 0x00101111, 0x00101212, 0x00111011, 0x00111110, 0x00111111, 0x00111112,
    0x00111211, 0x00121010, 0x00121012, 0x00121111, 0x00121210, 0x00121212, 0x01101011, 0x01101110,
    0x01101111, 0x01101112, 0x01111011, 0x01111012, 0x01111110, 0x01111111, 0x01111112, 0x01111211,
    0x01111212, 0x01121011, 0x01121110, 0x01121111, 0x01121112, 0x01121211, 0x02101010, 0x02101012,
    0x02101110, 0x02101111, 0x02101210, 0x02101212, 0x02111010, 0x02111011, 0x02111110, 0x02111111,
    0x02111112, 0x02111211, 0x02111212, 0x02121010, 0x02121012, 0x02121111, 0x00101021, 0x00101120,
    0x00101121, 0x00101122, 0x00111121, 0x00111122, 0x00111220, 0x00111222, 0x00121021, 0x00121122,
    0x01101020, 0x01101022, 0x01101120, 0x01101121, 0x01101220, 0x01101222, 0x01111021, 0x01111121,
    0x01111122, 0x01111220, 0x01111221, 0x01121021, 0x01121120, 0x01121121, 0x01121220, 0x01121221,
    0x01121222, 0x02101122, 0x02101222, 0x02111022, 0x02111121, 0x02121120, 0x02121221, 0x00112001,
    0x00112102, 0x00122101, 0x01102001, 0x01102100, 0x01102102, 0x01102201, 0x01112000, 0x01112101,
    0x01112200, 0x01112202, 0x01122000, 0x01122001, 0x01122100, 0x01122102, 0x01122201, 0x02102101,
    0x02112001, 0x02112100, 0x02122101, 0x00112010, 0x00112012, 0x00112111, 0x00112212, 0x00122011,
    0x00122111, 0x01102012, 0x01102110, 0x01102111, 0x01102210, 0x01112011, 0x01112110, 0x01112111,
    0x01112112, 0x01112211, 0x01112212, 0x01122010, 0x01122111, 0x01122212, 0x02102211, 0x02112011,
    0x02112012, 0x02112111, 0x02112210, 0x02122011, 0x02122112, 0x02122211, 0x00102221, 0x00112122,
    0x00122120, 0x00122122, 0x01102120, 0x01102122, 0x01102221, 0x01112020, 0x01112022, 0x01112121,
    0x01112220, 0x01122021, 0x01122122, 0x01122221, 0x02102121, 0x02112021, 0x02112122, 0x02112222,
    0x00200000, 0x00200002, 0x00200200, 0x00200202, 0x00210101, 0x00220000, 0x00220002, 0x00220101,
    0x00220200, 0x00220202, 0x01200101, 0x01210001, 0x01210201, 0x01220001, 0x01220101, 0x02200000,
    0x02200002, 0x02200200, 0x02200202, 0x02210101, 0x02220000, 0x02220002, 0x02220101, 0x02220200,
    0x02220202, 0x00200111, 0x00210011, 0x00210110, 0x00210211, 0x00220111, 0x01200012, 0x01200110,
    0x01200211, 0x01210111, 0x01210210, 0x01210212, 0x01220011, 0x01220110, 0x01220111, 0x01220112,
    0x02200111, 0x02210010, 0x02210112, 0x02210211, 0x02220111, 0x00200021, 0x00200220, 0x00200222,
    0x00210021, 0x00210121, 0x00220020, 0x00220022, 0x00220220, 0x00220222, 0x01200121, 0x01210021,
    0x01210122, 0x01210221, 0x01220121, 0x02200021, 0x02200220, 0x02200222, 0x02210021, 0x02210121,
    0x02220020, 0x02220022, 0x02220220, 0x02220222, 0x00201101, 0x00211100, 0x00211102, 0x00211201,
    0x00221101, 0x01201100, 0x01201101, 0x01201102, 0x01201201, 0x01211002, 0x01211101, 0x01211200,
    0x01211202, 0x01221102, 0x02201101, 0x02211001, 0x02211100, 0x02211201, 0x02221001, 0x02221101,
    0x00201211, 0x00211111, 0x00221011, 0x00221211, 0x01201010, 0x01201111, 0x01201210, 0x01211011,
    0x01211110, 0x01211111, 0x01211211, 0x01221012, 0x01221111, 0x01221210, 0x02201211, 0x02211010,
    0x02211110, 0x02211111, 0x02211210, 0x02211212, 0x02221011, 0x02221110, 0x02221112, 0x02221211,
    0x00201121, 0x00211020, 0x00211022, 0x00211221, 0x00221121, 0x01201021, 0x01201221, 0x01211121,
    0x01221020, 0x01221021, 0x01221221, 0x02201120, 0x02201122, 0x02211020, 0x02211222, 0x00202000,
    0x00202002, 0x00202200, 0x00202202, 0x00212101, 0x00222000, 0x00222002, 0x00222200, 0x00222202,
    0x01202101, 0x01212001, 0x01212100, 0x01222101, 0x02202000, 0x02202002, 0x02202200, 0x02202202,
    0x02222000, 0x02222002, 0x02222200, 0x02222202, 0x00202211, 0x00212011, 0x00212110, 0x00212211,
    0x00222111, 0x01202112, 0x01202211, 0x01212012, 0x01212111, 0x01222011, 0x01222110, 0x01222112,
    0x01222211, 0x02202111, 0x02212010, 0x02212112, 0x02212211, 0x02222110, 0x02222111, 0x00202020,
    0x00202022, 0x00202220, 0x00202222, 0x00222020, 0x00222022, 0x00222220, 0x00222222, 0x01202121,
    0x01212021, 0x01212122, 0x01212221, 0x01222121, 0x02202020, 0x02202022, 0x02202220, 0x02202222,
    0x02212121, 0x02222020, 0x02222022, 0x02222220, 0x02222222, 0x10000101, 0x10010001, 0x10010102,
    0x10020101, 0x11000201, 0x11010002, 0x11010101, 0x11010200, 0x11010202, 0x11020001, 0x11020100,
    0x11020102, 0x12010100, 0x12010201, 0x12020001, 0x12020102, 0x10000010, 0x10000011, 0x10000110,
    0x10000112, 0x10000211, 0x10010012, 0x10010111, 0x10010112, 0x10010210, 0x10010212, 0x10020011,
    0x10020112, 0x10020211, 0x11000111, 0x11000210, 0x11000212, 0x11010011, 0x11010110, 0x11010111,
    0x11010112, 0x11010211, 0x11010212, 0x11020111, 0x11020210, 0x11020212, 0x12000011, 0x12000110,
    0x12000112, 0x12010010, 0x12010012, 0x12010111, 0x12020010, 0x12020011, 0x12020012, 0x10000121,
    0x10010021, 0x10010120, 0x10010122, 0x10020121, 0x11000021, 0x11010022, 0x11010121, 0x11010222,
    0x11020120, 0x11020221, 0x12000221, 0x12010120, 0x12020121, 0x10001001, 0x10011101, 0x10011201,
    0x10021201, 0x11001101, 0x11001200, 0x11001202, 0x11011001, 0x11011100, 0x11011101, 0x11011102,
    0x11021001, 0x11021002, 0x11021101, 0x11021200, 0x11021202, 0x12001001, 0x12001102, 0x12001201,
    0x12011000, 0x12011002, 0x12011101, 0x12021000, 0x12021001, 0x12021201, 0x10001011, 0x10001012,
    0x10001111, 0x10001212, 0x10011011, 0x10011110, 0x10011111, 0x10011112, 0x10011211, 0x10021010,
    0x10021111, 0x10021212, 0x11001011, 0x11001110, 0x11001111, 0x11001112, 0x11001211, 0x11011010,
    0x11011011, 0x11011110, 0x11011111, 0x11011112, 0x11011210, 0x11011211, 0x11021011, 0x11021110,
    0x11021111, 0x11021112, 0x11021211, 0x12001012, 0x12001110, 0x12001111, 0x12001210, 0x12011011,
    0x12011110, 0x12011111, 0x12011112, 0x12011211, 0x12011212, 0x12021111, 0x12021210, 0x12021212,
    0x10001021, 0x10001121, 0x10001221, 0x10011120, 0x10011121, 0x10011220, 0x10011222, 0x10021021,
    0x10021120, 0x10021221, 0x11001020, 0x11001022, 0x11001121, 0x11001220, 0x11011020, 0x11011021,
    0x11011022, 0x11011121, 0x11011122, 0x11011221, 0x11021022, 0x11021121, 0x11021220, 0x12001021,
    0x12001121, 0x12001222, 0x12011120, 0x12011121, 0x12021021, 0x12021120, 0x12021122, 0x10002101,
    0x10012001, 0x10012101, 0x10012202, 0x10022101, 0x11002002, 0x11002201, 0x11012000, 0x11012101,
    0x11012200, 0x11022001, 0x11022100, 0x11022102, 0x11022201, 0x12002101, 0x12012001, 0x12012100,
    0x12012102, 0x12012201, 0x12022101, 0x10002011, 0x10002111, 0x10002112, 0x10002212, 0x10012010,
    0x10012110, 0x10012111, 0x10012210, 0x10022011, 0x10022110, 0x10022112, 0x11002010, 0x11002111,
    0x11002212, 0x11012011, 0x11012012, 0x11012110, 0x11012111, 0x11012112, 0x11012211, 0x11022010,
    0x11022012, 0x11022111, 0x11022112, 0x11022212, 0x12002112, 0x12002211, 0x12012012, 0x12012111,
    0x12012112, 0x12012210, 0x12022011, 0x12022110, 0x12022112, 0x12022211, 0x10012122, 0x11002120,
    0x11002122, 0x11002221, 0x11012121, 0x11012220, 0x11012222, 0x11022120, 0x11022221, 0x12012120,
    0x12022121, 0x10100001, 0x10100100, 0x10100101, 0x10100102, 0x10100201, 0x10110002, 0x10110101,
    0x10110202, 0x10120001, 0x10120100, 0x10120201, 0x11100000, 0x11100101, 0x11100200, 0x11110001,
    0x11110100, 0x11110101, 0x11110102, 0x11110201, 0x11120101, 0x11120200, 0x12100102, 0x12100201,
    0x12110101, 0x12110200, 0x12120000, 0x12120001, 0x12120102, 0x12120201, 0x10100111, 0x10100210,
    0x10100211, 0x10100212, 0x10110011, 0x10110110, 0x10110111, 0x10110112, 0x10110210, 0x10110211,
    0x10120010, 0x10120111, 0x10120112, 0x10120210, 0x10120212, 0x11100011, 0x11100110, 0x11100111,
    0x11100112, 0x11100211, 0x11110010, 0x11110011, 0x11110012, 0x11110110, 0x11110111, 0x11110112,
    0x11110210, 0x11110211, 0x11110212, 0x11120011, 0x11120110, 0x11120111, 0x11120112, 0x11120211,
    0x12100012, 0x12100111, 0x12110011, 0x12110110, 0x12110111, 0x12110112, 0x12110211, 0x12120010,
    0x12120111, 0x12120212, 0x10100021, 0x10100122, 0x10110022, 0x10110121, 0x10110222, 0x10120021,
    0x10120120, 0x11100022, 0x11100121, 0x11100222, 0x11110021, 0x11110120, 0x11110121, 0x11110122,
    0x11110221, 0x11120022, 0x11120121, 0x12100121, 0x12110020, 0x12110022, 0x12110121, 0x12110221,
    0x12110222, 0x12120120, 0x10101100, 0x10101101, 0x10111001, 0x10111100, 0x10111101, 0x10111102,
    0x10111200, 0x10111201, 0x10121001, 0x10121101, 0x10121200, 0x10121202, 0x11101001, 0x11101100,
    0x11101101, 0x11101102, 0x11101201, 0x11101202, 0x11111000, 0x11111001, 0x11111100, 0x11111101,
    0x11111102, 0x11111200, 0x11111201, 0x11111202, 0x11121001, 0x11121002, 0x11121100, 0x11121101,
    0x11121102, 0x11121201, 0x12101000, 0x12101200, 0x12101202, 0x12111001, 0x12111100, 0x12111101,
    0x12111102, 0x12111201, 0x12121001, 0x12121100, 0x12121101, 0x12121202, 0x10101011, 0x10101012,
    0x10101110, 0x10101111, 0x10101112, 0x10101211, 0x10111010, 0x10111011, 0x10111012, 0x10111110,
    0x10111111, 0x10111112, 0x10111211, 0x10111212, 0x10121011, 0x10121110, 0x10121111, 0x10121112,
    0x10121211, 0x11101010, 0x11101011, 0x11101012, 0x11101110, 0x11101111, 0x11101112, 0x11101210,
    0x11101211, 0x11111010, 0x11111011, 0x11111012, 0x11111110, 0x11111111, 0x11111112, 0x11111210,
    0x11111211, 0x11111212, 0x11121010, 0x11121011, 0x11121110, 0x11121111, 0x11121112, 0x11121210,
    0x11121211, 0x11121212, 0x12101011, 0x12101110, 0x12101111, 0x12101211, 0x12101212, 0x12111010,
    0x12111011, 0x12111110, 0x12111111, 0x12111112, 0x12111210, 0x12111211, 0x12121011, 0x12121110,
    0x12121111, 0x12121112, 0x12121211, 0x10101020, 0x10101021, 0x10101022, 0x10101120, 0x10101122,
    0x10101220, 0x10101221, 0x10111021, 0x10111120, 0x10111121, 0x10111220, 0x10111221, 0x10121020,
    0x10121021, 0x10121022, 0x10121120, 0x10121121, 0x10121122, 0x10121220, 0x10121221, 0x11101021,
    0x11101121, 0x11101122, 0x11101220, 0x11101221, 0x11101222, 0x11111020, 0x11111021, 0x11111022,
    0x11111120, 0x11111121, 0x11111122, 0x11111220, 0x11111221, 0x11111222, 0x11121021, 0x11121120,
    0x11121121, 0x11121221, 0x12101022, 0x12101121, 0x12101122, 0x12101220, 0x12101221, 0x12101222,
    0x12111021, 0x12111121, 0x12111222, 0x12121022, 0x12121121, 0x12121122, 0x12121220, 0x12121221,
    0x10102100, 0x10102101, 0x10102102, 0x10102201, 0x10112000, 0x10112101, 0x10112200, 0x10122001,
    0x10122202, 0x11102101, 0x11102200, 0x11102202, 0x11112001, 0x11112100, 0x11112101, 0x11112102,
    0x11112200, 0x11112201, 0x11122000, 0x11122002, 0x11122100, 0x11122101, 0x12102002, 0x12102201,
    0x12112000, 0x12112002, 0x12112101, 0x12112200, 0x12122001, 0x12122201, 0x10102011, 0x10102012,
    0x10102111, 0x10102212, 0x10112011, 0x10112110, 0x10112111, 0x10112112, 0x10112211, 0x10122111,
    0x11102011, 0x11102110, 0x11102111, 0x11102112, 0x11102211, 0x11112010, 0x11112011, 0x11112012,
    0x11112110, 0x11112111, 0x11112112, 0x11112210, 0x11112211, 0x11112212, 0x11122011, 0x11122110,
    0x11122111, 0x11122112, 0x11122211, 0x12102011, 0x12102111, 0x12102211, 0x12112011, 0x12112110,
    0x12112111, 0x12112112, 0x12112210, 0x12112211, 0x12122111, 0x10102120, 0x10102220, 0x10112121,
    0x10112222, 0x10122020, 0x10122121, 0x10122122, 0x10122221, 0x11102121, 0x11102220, 0x11102221,
    0x11112021, 0x11112121, 0x11112122, 0x11112220, 0x11112221, 0x11122022, 0x11122121, 0x11122220,
    0x11122222, 0x12102021, 0x12102222, 0x12112022, 0x12112121, 0x12112122, 0x12112220, 0x12112222,
    0x12122021, 0x10200101, 0x10210100, 0x10210102, 0x10210201, 0x10220101, 0x11200100, 0x11210000,
    0x11210101, 0x11210102, 0x11210200, 0x11210202, 0x11220001, 0x11220100, 0x11220102, 0x11220201,
    0x12200001, 0x12210102, 0x12220101, 0x10200011, 0x10200110, 0x10200112, 0x10200211, 0x10210012,
    0x10210111, 0x10220011, 0x10220012, 0x10220112, 0x10220211, 0x11200111, 0x11200211, 0x11210011,
    0x11210111, 0x11210112, 0x11210211, 0x11220111, 0x11220112, 0x11220212, 0x12200110, 0x12200212,
    0x12210012, 0x12210111, 0x12220011, 0x12220112, 0x12220211, 0x10210021, 0x10210122, 0x10210221,
    0x11200020, 0x11200021, 0x11200122, 0x11210121, 0x11210122, 0x11210220, 0x11220020, 0x12200121,
    0x12210021, 0x12210122, 0x12220121, 0x10211001, 0x10211002, 0x10211101, 0x10211102, 0x10211202,
    0x10221001, 0x10221102, 0x10221201, 0x11201000, 0x11201002, 0x11201101, 0x11201200, 0x11201202,
    0x11211001, 0x11211100, 0x11211101, 0x11211102, 0x11211201, 0x11211202, 0x11221000, 0x11221002,
    0x11221101, 0x12201100, 0x12201101, 0x12201201, 0x12211000, 0x12211002, 0x12211100, 0x12211101,
    0x12211102, 0x12211200, 0x12211202, 0x12221001, 0x12221100, 0x12221201, 0x10201111, 0x10201210,
    0x10201212, 0x10211011, 0x10211111, 0x10211112, 0x10211211, 0x11201110, 0x11201111, 0x11201112,
    0x11201211, 0x11211010, 0x11211011, 0x11211110, 0x11211111, 0x11211112, 0x11211211, 0x11221011,
    0x11221110, 0x11221111, 0x11221112, 0x11221211, 0x12201112, 0x12201211, 0x12201212, 0x12211011,
    0x12211111, 0x12211112, 0x12211211, 0x12211212, 0x12221012, 0x12221111, 0x12221112, 0x12221210,
    0x10201022, 0x10201221, 0x10211121, 0x10221020, 0x10221122, 0x10221220, 0x10221221, 0x11201020,
    0x11201121, 0x11201220, 0x11201222, 0x11211021, 0x11211120, 0x11211121, 0x11211122, 0x11211220,
    0x11211222, 0x11221020, 0x11221121, 0x11221220, 0x12201020, 0x12201022, 0x12201121, 0x12201222,
    0x12211120, 0x12211122, 0x12211220, 0x12211221, 0x12221020, 0x12221120, 0x12221122, 0x12221222,
    0x10212102, 0x10212201, 0x10222101, 0x11202001, 0x11212002, 0x11212101, 0x11212202, 0x11222001,
    0x11222201, 0x12202101, 0x12212001, 0x12212200, 0x12222102, 0x10202011, 0x10202110, 0x10212010,
    0x10212111, 0x10222011, 0x10222110, 0x10222112, 0x10222211, 0x11202010, 0x11202011, 0x11202111,
    0x11202112, 0x11202210, 0x11212011, 0x11212110, 0x11212111, 0x11212112, 0x11212211, 0x11222010,
    0x11222111, 0x11222212, 0x12202012, 0x12202110, 0x12202212, 0x12212111, 0x12222011, 0x12222110,
    0x12222111, 0x12222211, 0x10212021, 0x10212122, 0x10212220, 0x11202021, 0x11202120, 0x11202221,
    0x11212020, 0x11212121, 0x11212220, 0x11212222, 0x11222120, 0x11222121, 0x11222221, 0x12202122,
    0x12212120, 0x12212220, 0x12212222, 0x12222122, 0x20000000, 0x20000002, 0x20000200, 0x20000202,
    0x20020000, 0x20020002, 0x20020200, 0x20020202, 0x21000101, 0x21010000, 0x21010001, 0x21010100,
    0x21010102, 0x21010201, 0x21020101, 0x22000000, 0x22000002, 0x22000200, 0x22000202, 0x22010101,
    0x22020000, 0x22020002, 0x22020200, 0x22020202, 0x20000111, 0x20010011, 0x20010110, 0x20010112,
    0x20010211, 0x20020111, 0x21000011, 0x21000110, 0x21000211, 0x21010010, 0x21010012, 0x21010111,
    0x21010112, 0x21010210, 0x21010211, 0x21020110, 0x21020112, 0x21020211, 0x22000111, 0x22000211,
    0x22010110, 0x22010112, 0x22010211, 0x22020111, 0x20000020, 0x20000022, 0x20000220, 0x20000222,
    0x20010121, 0x20020020, 0x20020022, 0x20020220, 0x20020222, 0x21010021, 0x21010120, 0x21010221,
    0x21020121, 0x22000020, 0x22000022, 0x22000220, 0x22000222, 0x22010121, 0x22020020, 0x22020022,
    0x22020220, 0x22020222, 0x20011100, 0x20011201, 0x21001001, 0x21001100, 0x21011001, 0x21011101,
    0x21011202, 0x21021001, 0x21021100, 0x21021201, 0x22011100, 0x22011201, 0x20001011, 0x20001211,
    0x20011012, 0x20011111, 0x20011212, 0x20021112, 0x20021211, 0x21001010, 0x21001011, 0x21001111,
    0x21001210, 0x21011011, 0x21011110, 0x21011111, 0x21011112, 0x21011211, 0x21011212, 0x21021111,
    0x21021112, 0x21021210, 0x21021212, 0x22001011, 0x22001110, 0x22001112, 0x22001211, 0x22011010,
    0x22011012, 0x22011111, 0x22011210, 0x22021112, 0x20011021, 0x20011122, 0x20011221, 0x20021121,
    0x21001021, 0x21001120, 0x21001221, 0x21001222, 0x21011020, 0x21011121, 0x21011221, 0x21011222,
    0x21021021, 0x21021122, 0x21021222, 0x22001121, 0x22011021, 0x22011222, 0x22021120, 0x20002000,
    0x20002002, 0x20002200, 0x20002202, 0x20012101, 0x20022000, 0x20022002, 0x20022200, 0x20022202,
    0x21002001, 0x21002101, 0x21012001, 0x21012100, 0x21012201, 0x21022101, 0x21022201, 0x22002000,
    0x22002002, 0x22002200, 0x22002202, 0x22012101, 0x22022000, 0x22022002, 0x22022200, 0x22022202,
    0x20002111, 0x20002112, 0x20012011, 0x20012110, 0x20012112, 0x20022111, 0x21002011, 0x21002110,
    0x21002112, 0x21002211, 0x21012010, 0x21012012, 0x21012111, 0x21012212, 0x21022011, 0x21022110,
    0x22002111, 0x22012112, 0x22012211, 0x22022111, 0x20002020, 0x20002022, 0x20002220, 0x20002222,
    0x20012121, 0x20022020, 0x20022022, 0x20022220, 0x20022222, 0x21002121, 0x21012021, 0x21012120,
    0x21012122, 0x22002020, 0x22002022, 0x22002220, 0x22002222, 0x22012121, 0x22022020, 0x22022022,
    0x22022220, 0x22022222, 0x20100101, 0x20110001, 0x20110102, 0x20110200, 0x20110201, 0x20120101,
    0x21100001, 0x21100102, 0x21100201, 0x21110101, 0x21110200, 0x21110202, 0x21120201, 0x21120202,
    0x22100101, 0x22110001, 0x22110100, 0x22110102, 0x22110201, 0x22120101, 0x20100011, 0x20100110,
    0x20100112, 0x20100211, 0x20110010, 0x20110111, 0x20110210, 0x20110212, 0x20120011, 0x20120110,
    0x20120112, 0x20120211, 0x21100010, 0x21100111, 0x21110010, 0x21110011, 0x21110110, 0x21110111,
    0x21110112, 0x21110211, 0x21120012, 0x21120111, 0x22100110, 0x22100112, 0x22110012, 0x22110111,
    0x22110210, 0x22120011, 0x22120110, 0x22120112, 0x22120211, 0x20100121, 0x20110021, 0x20110120,
    0x20110221, 0x20120121, 0x21100120, 0x21100122, 0x21100221, 0x21110020, 0x21110022, 0x21110121,
    0x21110220, 0x21120122, 0x21120221, 0x22100121, 0x22110120, 0x22110122, 0x22120221, 0x20101001,
    0x20101100, 0x20101102, 0x20111000, 0x20111101, 0x20111200, 0x20121102, 0x21101000, 0x21101202,
    0x21111001, 0x21111100, 0x21111101, 0x21111102, 0x21111200, 0x21111201, 0x21121000, 0x21121001,
    0x21121002, 0x21121101, 0x22101100, 0x22101102, 0x22111002, 0x22111100, 0x22111101, 0x22111200,
    0x22121001, 0x22121201, 0x20101010, 0x20101111, 0x20101210, 0x20101212, 0x20111010, 0x20111011,
    0x20111110, 0x20111111, 0x20111112, 0x20111211, 0x20121011, 0x20121111, 0x20121211, 0x20121212,
    0x21101011, 0x21101110, 0x21101111, 0x21101112, 0x21101211, 0x21111010, 0x21111011, 0x21111012,
    0x21111110, 0x21111111, 0x21111112, 0x21111210, 0x21111211, 0x21111212, 0x21121011, 0x21121110,
    0x21121111, 0x21121112, 0x21121211, 0x22101011, 0x22101111, 0x22101210, 0x22111011, 0x22111012,
    0x22111110, 0x22111111, 0x22111112, 0x22111211, 0x22111212, 0x22121010, 0x22121012, 0x22121111,
    0x22121210, 0x22121212, 0x20101021, 0x20101120, 0x20111020, 0x20111121, 0x20111221, 0x20121020,
    0x20121122, 0x20121221, 0x21101121, 0x21101220, 0x21101221, 0x21111021, 0x21111022, 0x21111121,
    0x21111122, 0x21111221, 0x21121121, 0x21121220, 0x22101022, 0x22101120, 0x22101221, 0x22101222,
    0x22111022, 0x22111120, 0x22111121, 0x22121120, 0x22121122, 0x22121221, 0x20102101, 0x20112102,
    0x20112201, 0x20122101, 0x21102001, 0x21102102, 0x21112000, 0x21112002, 0x21112101, 0x21112102,
    0x21112202, 0x21122100, 0x21122101, 0x22102101, 0x22112001, 0x22112102, 0x22112201, 0x22122101,
    0x20102110, 0x20102112, 0x20102211, 0x20112010, 0x20112012, 0x20112111, 0x20112210, 0x20112212,
    0x20122010, 0x20122011, 0x20122110, 0x20122112, 0x21102010, 0x21102012, 0x21102111, 0x21102210,
    0x21102212, 0x21112011, 0x21112110, 0x21112111, 0x21112112, 0x21112211, 0x21122012, 0x21122111,
    0x21122112, 0x21122212, 0x22102011, 0x22102110, 0x22112010, 0x22112012, 0x22112111, 0x22112212,
    0x22122011, 0x22122112, 0x20102121, 0x20112121, 0x20122121, 0x21102120, 0x21102122, 0x21102221,
    0x21112020, 0x21112121, 0x21112220, 0x21122021, 0x22102121, 0x22112021, 0x22112120, 0x22112121,
    0x22112122, 0x20200000, 0x20200002, 0x20200200, 0x20200202, 0x20210101, 0x20220000, 0x20220002,
    0x20220200, 0x20220202, 0x21200101, 0x21210001, 0x21210100, 0x21210102, 0x21210201, 0x22200000,
    0x22200002, 0x22200200, 0x22200202, 0x22210101, 0x22220000, 0x22220002, 0x22220200, 0x22220202,
    0x20200111, 0x20200211, 0x20210011, 0x20210110, 0x20210112, 0x20210211, 0x20210212, 0x21200112,
    0x21200211, 0x21210011, 0x21210111, 0x21210210, 0x21210212, 0x21220011, 0x21220110, 0x22200111,
    0x22210010, 0x22210012, 0x22210112, 0x22210211, 0x20200022, 0x20200220, 0x20200222, 0x20210020,
    0x20210221, 0x20220022, 0x20220220, 0x20220222, 0x21200121, 0x21210021, 0x21210122, 0x21210221,
    0x21220121, 0x22200020, 0x22200022, 0x22200220, 0x22200222, 0x22210121, 0x22220020, 0x22220022,
    0x22220220, 0x22220222, 0x20211201, 0x20221101, 0x21201001, 0x21201100, 0x21211000, 0x21211100,
    0x21211101, 0x21211200, 0x21211202, 0x21221001, 0x21221101, 0x21221102, 0x21221200, 0x21221201,
    0x22201101, 0x20201112, 0x20201211, 0x20211010, 0x20211012, 0x20211111, 0x20211210, 0x20221112,
    0x20221211, 0x21201012, 0x21201111, 0x21211011, 0x21211110, 0x21211111, 0x21211112, 0x21211211,
    0x21221111, 0x21221212, 0x22201011, 0x22201110, 0x22201111, 0x22201112, 0x22201211, 0x22211012,
    0x22211111, 0x22211210, 0x20201121, 0x20211021, 0x20211122, 0x20211222, 0x20221021, 0x20221121,
    0x21201120, 0x21201122, 0x21201222, 0x21211022, 0x21211121, 0x21211122, 0x21211220, 0x21221020,
    0x21221022, 0x22201122, 0x22211020, 0x22211121, 0x22211122, 0x22211221, 0x22221021, 0x22221120,
    0x22221122, 0x20202000, 0x20202002, 0x20202200, 0x20202202, 0x20222000, 0x20222002, 0x20222200,
    0x20222202, 0x21212001, 0x21212100, 0x21212102, 0x21212201, 0x22202000, 0x22202002, 0x22202200,
    0x22202202, 0x22212101, 0x22222000, 0x22222002, 0x22222200, 0x22222202, 0x20202111, 0x20212110,
    0x20212211, 0x20222011, 0x20222111, 0x21202011, 0x21212010, 0x21212111, 0x21212212, 0x21222011,
    0x21222112, 0x21222211, 0x22212010, 0x22212112, 0x20202020, 0x20202022, 0x20202220, 0x20202222,
    0x20222020, 0x20222022, 0x20222220, 0x20222222, 0x21212021, 0x21212120, 0x21212122, 0x22202020,
    0x22202022, 0x22202220, 0x22202222, 0x22212121, 0x22222020, 0x22222022, 0x22222220, 0x22222222
};
QUANT_CONSTANT int8_t kFP4Values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

#undef QUANT_CONSTANT
// ---- end runtime/metal/abi/QuantTables.h ----
#include <metal_stdlib>
using namespace metal;

// Per-format decoding of one (row, group of 32) of the MDGG0001 image, shared
// by the GEMM kernels, to the values of llama.cpp's dequantize.h (MIT notice
// in THIRD_PARTY_NOTICES). A group is read by chunk (metal/abi/QuantFormat.h):
// chunk c holds pairs p = 0..3; pairs 0, 1 are elements 4c..4c+3 of the first
// 16-group and pairs 2, 3 are elements 16+4c..16+4c+3 of the second. A format
// has its id, its sizes P0, P1, MetaBytes and MetaGroups from kQuantFormats,
// its code offset Zero (0 but for the linear formats with a zero point), the
// Group of elements one coefficient covers (32, 16 when the halves of a group
// have their own, or every group of a meta unit when they share one: MLX
// affine's 64 and 128, PQ2_0's 128, Q2_0's 64, FP8's and BF16's 256) and
//   load(plane0, plane1) -> Payload, loadMeta(meta) -> Meta
//   chunk(Payload, c) -> Chunk
//   loadChunk(plane0, plane1, c) -> Chunk, the same chunk read on its own
//   coef(Meta, j) -> QuantCoef of group j of the meta unit
//     (ScaleInChunk: coef(Meta, Chunk), the group's scale is in every chunk
//     of it rather than in the meta unit)
// and one element accessor, by Kind:
//   QuantLinear    codes(Chunk) -> uint4, pair p in component p with e0 at
//                  bit 0 and e1 at bit 16; value = s * (code - Zero) + m
//   QuantCodebook  indices(Chunk) -> uint, byte p indexes pair p in the
//                  format's pair table (quant_pair_table, of its value(i));
//                  value = s * table value
//   QuantInt8      values(Chunk) -> uint2, the int8 values of pairs 0, 1 (x)
//                  and 2, 3 (y); value = s * int8
//   (both with Scale, the narrowest type that holds s exactly)
//   QuantGrid      grid(Chunk) -> uint2, the grid magnitudes of pairs 0, 1
//                  (x) and 2, 3 (y); signs(Chunk) bit 2p + i negates element
//                  i of pair p; value = s * signed magnitude
//   QuantFloat8    values(Chunk) -> uint2, the E4M3 bytes of pairs 0, 1 (x)
//                  and 2, 3 (y); value = s * quant_e4m3 of the byte, which
//                  is 2^-8 times the E4M3 value (so s carries the 2^8)
//   QuantFloat16   values(Chunk) -> uint4, pair p in component p with e0 at
//                  bits 0..15 and e1 at 16..31, each of the format's 16-bit
//                  Value type (bfloat or half); value = s * the Value
enum QuantKind : ushort { QuantLinear, QuantCodebook, QuantInt8, QuantGrid, QuantFloat8, QuantFloat16 };

// The E4M3 bytes at bits 0..7 and 16..23 of `pair` (sign, four exponent bits of bias 7, three mantissa bits) as
// halves 2^-8 times their values, exactly: the byte's seven magnitude bits placed so that its exponent field is
// the low four of the half's, its subnormals the half's, as MLX's fp8.h also converts them. The NaN encodings
// (0x7F, 0xFF) come out finite; no weight or scale holds one.
inline half2 quant_e4m3_pair(uint pair) { return as_type<half2>(((pair & 0x007F007Fu) << 7) | ((pair & 0x00800080u) << 8)); }
// The four E4M3 bytes of a word, in byte order, as quant_e4m3_pair's halves.
inline half4 quant_e4m3_word(uint word) {
  const half2 even = quant_e4m3_pair(word & 0x00FF00FFu), odd = quant_e4m3_pair((word >> 8) & 0x00FF00FFu);
  return half4(even.x, odd.x, even.y, odd.y);
}
// The scale of quant_e4m3_pair's halves in an MXFP8 group of E8M0 exponent e: 2^(e - 127) 2^8 = 2^(e - 119), a
// normal float for every e below 247 and infinite from there (255 is E8M0's NaN; no weight's scale is near 2^120).
inline float quant_mxfp8_scale(uint e) { return as_type<float>(min(e + 8u, 255u) << 23); }

// s.x scales and m.x offsets pairs 0, 1, s.y and m.y pairs 2, 3 (equal where
// one coefficient covers 32 elements or more: the register tile then reads the
// x halves alone).
struct QuantCoef {
  float2 s;
  float2 m = float2(0.0f);
};

// The id, sizes and traits of format F of kind K with code offset Z, whose
// coefficients cover G elements (32 * MetaGroups where the groups of a meta
// unit share one) and, with InChunk, whose scales are in every chunk of a
// group.
#define QUANT_FORMAT(F, K, Z, G, InChunk)                                                                        \
  enum : uint { Id = F, P0 = kQuantFormats[F].plane0_bytes, P1 = kQuantFormats[F].plane1_bytes, MetaBytes = kQuantFormats[F].meta_bytes }; \
  enum : ushort { MetaGroups = kQuantFormats[F].meta_groups, Zero = Z, Group = G };                              \
  static constexpr constant QuantKind Kind = K;                                                                  \
  static constexpr constant bool ScaleInChunk = InChunk

// Entries of a codebook format's pair table: one per index byte.
constant constexpr uint kQuantPairTableEntries = 256;
// Fills the threadgroup table of format F's value pairs, as half2 (staged tiles) or bfloat2 (register tiles), entry
// b = (F::value(b & 15), F::value(b >> 4)) for the index byte b of a pair, when F is a codebook format; called by all
// threads of the threadgroup. quant_pair_entries writes the entries alone and returns whether F has a table, for a
// threadgroup that fills several tables before one barrier.
template <class F, class T>
inline bool quant_pair_entries(threadgroup T *table, uint thread_index, uint threads) {
  if constexpr (F::Kind == QuantCodebook)
    for (uint i = thread_index; i < kQuantPairTableEntries; i += threads)
      table[i] = T(float2(F::value(i & 15), F::value(i >> 4)));
  return F::Kind == QuantCodebook;
}
template <class F, class T>
inline void quant_pair_table(threadgroup T *table, uint thread_index, uint threads) {
  if constexpr (F::Kind == QuantCodebook) {
    quant_pair_entries<F>(table, thread_index, threads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The pair words of a word of 4-bit codes: pair p's e0 at bits 4p, e1 at 16 + 4p.
inline uint4 quant_nibble_pairs(uint word) { return (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu; }
// Pair-word order of bit fields: the 1-bit fields of the two chunk bytes of a halfword (bit 2p + i of byte b)
// go to bits 8b + 2p (e0) and 16 + 8b + 2p (e1); the 2-bit fields of a chunk halfword (bits 4p + 2i) to bits
// 4p (e0) and 16 + 4p (e1).
inline uint quant_spread1(uint bits) { return (bits & 0x5555u) | ((bits & 0xAAAAu) << 15); }
inline uint quant_spread2(uint bits) { return (bits & 0x3333u) | ((bits & 0xCCCCu) << 14); }

// block_q4_K / block_q5_K header: s = d * sc and m = -dmin * mn with the 6-bit sc, mn of group j. The 12 scale
// bytes are hdr.y (0-3), hdr.z (4-7) and hdr.w (8-11), taken with shifts: the group index is not a compile-time
// constant, and indexing a thread-local byte array or vector by it costs ~8% of the eight-row staged kernel on
// Apple10.
inline QuantCoef quant_k4_coef(uint4 hdr, ushort j) {
  uint sc, m;
  if (j < 4) {
    const uint sh = 8u * j;
    sc = (hdr.y >> sh) & 63u;
    m = (hdr.z >> sh) & 63u;
  } else {
    const uint sh = 8u * (j - 4), w = hdr.w >> sh;
    sc = (w & 0xFu) | (((hdr.y >> sh) >> 6) & 3u) << 4;
    m = ((w >> 4) & 0xFu) | (((hdr.z >> sh) >> 6) & 3u) << 4;
  }
  const half d = as_type<half>(ushort(hdr.x & 0xFFFF)), dmin = as_type<half>(ushort(hdr.x >> 16));
  return {float2(float(d) * float(sc)), float2(-float(dmin) * float(m))};
}

// Scale bytes 2j, 2j + 1 of 16 bytes: bytes 2(j & 1), 2(j & 1) + 1 of word j >> 1, in bits 0..15 (shifts, as for
// Q4_K).
inline uint quant_scale_pair(packed_uint4 bytes, ushort j) {
  const uint word = j < 2 ? bytes.x : j < 4 ? bytes.y : j < 6 ? bytes.z : bytes.w;
  return word >> (16u * (j & 1));
}

// Q4_K: plane0 4-bit codes; meta the 16-byte block header.
struct FmtQ4K {
  QUANT_FORMAT(GGUF_FMT_Q4K, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q); }
  static QuantCoef coef(Meta hdr, ushort j) { return quant_k4_coef(hdr, j); }
};
// Q5_K: plane0 low 4 bits, plane1 the fifth bits (byte c = chunk c); meta as Q4_K.
struct FmtQ5K {
  QUANT_FORMAT(GGUF_FMT_Q5K, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; uint b; }; typedef uint2 Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) { return uint2(w.a[c], quant_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1))); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(*((device uint *)(p0 + 4 * c)), quant_spread1(p1[c]));
  }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0, 2, 4, 6)) & 0x00010001u) << 4); }
  static QuantCoef coef(Meta hdr, ushort j) { return quant_k4_coef(hdr, j); }
};
// Q6_K: plane0 low 4 bits, plane1 the high 2 bits (halfword c = chunk c); meta 16 int8 scales, then half d.
// value = d * sc * (q - 32) with one scale per 16-group.
struct FmtQ6K {
  QUANT_FORMAT(GGUF_FMT_Q6K, QuantLinear, 32, 16, false);
  struct Payload { uint4 a; uint2 b; }; typedef uint2 Chunk; struct Meta { packed_uint4 sc; uint d; };
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint2 *)p1)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.sc = *((device packed_uint4 *)m); r.d = *((device uint *)(m + 16)); return r; }
  static Chunk chunk(Payload w, ushort c) { return uint2(w.a[c], quant_spread2(w.b[c >> 1] >> (16 * (c & 1)))); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(*((device uint *)(p0 + 4 * c)), quant_spread2(*((device ushort *)(p1 + 2 * c))));
  }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0, 4, 8, 12)) & 0x00030003u) << 4); }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.d & 0xFFFF)));
    const uint pair = quant_scale_pair(mt.sc, j);   // the int8 scales of both 16-groups
    return {float2(d * float(as_type<char>(uchar(pair & 0xFFu))), d * float(as_type<char>(uchar(pair >> 8))))};
  }
};
// Q3_K: plane0 low 2 bits (halfword c = chunk c), plane1 the hmask bits (byte c = chunk c); meta half d, 2 zero
// bytes, the 12 packed scale bytes. value = d * (sc - 32) * (q - 4) with one scale per 16-group.
struct FmtQ3K {
  QUANT_FORMAT(GGUF_FMT_Q3K, QuantLinear, 4, 16, false);
  struct Payload { uint2 a; uint b; }; typedef uint2 Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) {
    return uint2(quant_spread2(w.a[c >> 1] >> (16 * (c & 1))), quant_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1)));
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(quant_spread2(*((device ushort *)(p0 + 2 * c))), quant_spread1(p1[c]));
  }
  static uint4 codes(Chunk q) { return ((uint4(q.x) >> uint4(0, 4, 8, 12)) & 0x00030003u) | (((uint4(q.y) >> uint4(0, 2, 4, 6)) & 0x00010001u) << 2); }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.x & 0xFFFF)));
    const uint t0 = mt.y, t1 = mt.z, t2 = mt.w;   // scales[0..3], [4..7], [8..11]
    uint aux;
    switch (j >> 1) {
      case 0: aux = (t0 & 0x0f0f0f0fu) | (((t2 >> 0) & 0x03030303u) << 4); break;
      case 1: aux = (t1 & 0x0f0f0f0fu) | (((t2 >> 2) & 0x03030303u) << 4); break;
      case 2: aux = ((t0 >> 4) & 0x0f0f0f0fu) | (((t2 >> 4) & 0x03030303u) << 4); break;
      default: aux = ((t1 >> 4) & 0x0f0f0f0fu) | (((t2 >> 6) & 0x03030303u) << 4); break;
    }
    const uint pair = aux >> (16u * (j & 1));   // 6-bit scales 2j, 2j + 1 (shifts, as for Q4_K)
    return {float2(d * float(int(pair & 0xFFu) - 32), d * float(int((pair >> 8) & 0xFFu) - 32))};
  }
};
// IQ4_XS: plane0 codebook indices; meta half d, scales_h, scales_l[4]. value = d * (ls - 32) * codebook.
struct FmtIQ4XS {
  QUANT_FORMAT(GGUF_FMT_IQ4XS, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint2 Meta; typedef float Scale;
  static half value(uint i) { return half(kIQ4NLValues[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint2 *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort j) {
    const half d = as_type<half>(ushort(mt.x & 0xFFFF)); const uint sh = mt.x >> 16;
    const int ls = int((mt.y >> (4 * j)) & 0xF) | int(((sh >> (2 * j)) & 3) << 4);
    return {float2(float(d) * float(ls - 32))};
  }
};
// IQ4_NL: plane0 codebook indices; meta half d per group. value = d * codebook.
struct FmtIQ4NL {
  QUANT_FORMAT(GGUF_FMT_IQ4NL, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta; typedef half Scale;
  static half value(uint i) { return half(kIQ4NLValues[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// Q8_0: plane0 int8 values (bytes 8c..8c+7 = chunk c); meta half d per group.
struct FmtQ80 {
  QUANT_FORMAT(GGUF_FMT_Q80, QuantInt8, 0, 32, false);
  struct Payload { uint4 a; uint4 b; }; typedef uint2 Chunk; typedef ushort Meta; typedef half Scale;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0), *((device uint4 *)(p0 + 16))}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { const uint4 h = c < 2 ? w.a : w.b; return (c & 1) ? h.zw : h.xy; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint2 *)(p0 + 8 * c)); }
  static uint2 values(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// IQ3_S: plane0 word c = the 8-bit grid indices of pairs 0, 1 and 2, 3, chunk c's sign bits, the two ninth index
// bits and the group's 4-bit scale; meta half d per super-block. value = d * (1 + 2 * scale) * signed grid value.
struct FmtIQ3S {
  QUANT_FORMAT(GGUF_FMT_IQ3S, QuantGrid, 0, 32, true);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint2 grid(Chunk q) { return uint2(kIQ3SGrid[(q & 0xFF) | ((q >> 16) & 0x100)], kIQ3SGrid[((q >> 8) & 0xFF) | ((q >> 17) & 0x100)]); }
  static uint signs(Chunk q) { return (q >> 16) & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return {float2(float(as_type<half>(mt)) * float(1 + 2 * ((q >> 26) & 0xF)))}; }
};
// Q2_K: plane0 2-bit codes (halfword c = chunk c); meta half d, half dmin, the 16 scale bytes. value = d * sc * q -
// dmin * mn with the low (sc) and high (mn) nibble of one scale byte per 16-group.
struct FmtQ2K {
  QUANT_FORMAT(GGUF_FMT_Q2K, QuantLinear, 0, 16, false);
  struct Payload { uint2 a; }; typedef uint Chunk; struct Meta { uint dm; packed_uint4 sc; };
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint2 *)p0)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.dm = *((device uint *)m); r.sc = *((device packed_uint4 *)(m + 4)); return r; }
  static Chunk chunk(Payload w, ushort c) { return quant_spread2(w.a[c >> 1] >> (16 * (c & 1))); }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return quant_spread2(*((device ushort *)(p0 + 2 * c))); }
  static uint4 codes(Chunk q) { return (uint4(q) >> uint4(0, 4, 8, 12)) & 0x00030003u; }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.dm & 0xFFFF))), dmin = float(as_type<half>(ushort(mt.dm >> 16)));
    const uint pair = quant_scale_pair(mt.sc, j);   // the scale bytes of both 16-groups
    return {float2(d * float(pair & 0xF), d * float((pair >> 8) & 0xF)),
            float2(-dmin * float((pair >> 4) & 0xF), -dmin * float((pair >> 12) & 0xF))};
  }
};
// The eight signs of a 7-bit sign index (IQ2_XXS, IQ2_XS, IQ3_XXS): its bits, and as bit 7 their parity, which
// makes the count of negated elements even (llama.cpp's ksigns_iq2xs).
inline uint quant_signs7(uint index) { return index | (popcount(index) & 1) << 7; }
// The sign bits of chunk c from a word of four 7-bit sign indices, index l at bits 7l covering elements 8l..8l+7
// (IQ2_XXS, IQ3_XXS): elements 4c..4c+3 are half c & 1 of index c >> 1, 16+4c..16+4c+3 of index 2 + (c >> 1).
inline uint quant_chunk_signs7(uint word, ushort c) {
  const uint l = 7u * (c >> 1), h = 4u * (c & 1);
  return ((quant_signs7((word >> l) & 127) >> h) & 0xF) | ((quant_signs7((word >> (l + 14)) & 127) >> h) & 0xF) << 4;
}
// The magnitudes of half h of two eight-element grid entries (IQ2): elements 4h..4h+3 of each.
inline uint2 quant_grid8(uint64_t a, uint64_t b, uint h) {
  return h ? uint2(as_type<uint2>(a).y, as_type<uint2>(b).y) : uint2(as_type<uint2>(a).x, as_type<uint2>(b).x);
}
// The coefficients of an IQ2 group: d * (1 + 2 * scale) / 8, of the low (pairs 0, 1) and high (pairs 2, 3) nibble
// of `scales`, which are equal where one scale covers the group.
inline QuantCoef quant_iq2_coef(ushort d, uint scales) {
  const float s = float(as_type<half>(d));
  return {float2(s * float(1 + 2 * (scales & 0xF)) * 0.125f, s * float(1 + 2 * ((scales >> 4) & 0xF)) * 0.125f)};
}
// IQ3_XXS: plane0 the group's 8-bit grid indices (byte t: elements 4t..4t+3), plane1 its word of four 7-bit sign
// indices (index l at bit 7l: elements 8l..8l+7) and the 4-bit scale (bits 28..31); meta half d. value = d * (1 + 2 *
// scale) / 4 * signed grid value. A chunk is IQ3_S's word without ninth index bits.
struct FmtIQ3XXS {
  QUANT_FORMAT(GGUF_FMT_IQ3XXS, QuantGrid, 0, 32, true);
  struct Payload { uint2 a; uint b; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    return ((w.a.x >> (8 * c)) & 0xFF) | ((w.a.y >> (8 * c)) & 0xFF) << 8 | quant_chunk_signs7(w.b, c) << 16 | (w.b >> 28) << 26;
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return uint2(kIQ3XXSGrid[q & 0xFF], kIQ3XXSGrid[(q >> 8) & 0xFF]); }
  static uint signs(Chunk q) { return (q >> 16) & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return {float2(float(as_type<half>(mt)) * float(1 + 2 * ((q >> 26) & 0xF)) * 0.25f)}; }
};
// IQ2_XXS: plane0 the group's native 8 bytes, word 0 the 8-bit grid indices of its four eight-element entries (byte
// l: elements 8l..8l+7) and word 1 their 7-bit sign indices (bits 7l) and the 4-bit scale (bits 28..31); meta half
// d. value = d * (1 + 2 * scale) / 8 * signed grid value. A chunk holds half c & 1 of entries c >> 1 (pairs 0, 1) and
// 2 + (c >> 1) (pairs 2, 3): their indices at bits 0..7 and 8..15, the chunk's sign bits at 16..23, the half at 24
// and the scale at 28..31.
struct FmtIQ2XXS {
  QUANT_FORMAT(GGUF_FMT_IQ2XXS, QuantGrid, 0, 32, true);
  struct Payload { uint2 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint2 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = 8u * (c >> 1);
    return ((w.a.x >> l) & 0xFF) | ((w.a.x >> (l + 16)) & 0xFF) << 8 | quant_chunk_signs7(w.a.y, c) << 16 | uint(c & 1) << 24 |
           (w.a.y & 0xF0000000u);
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return quant_grid8(kIQ2XXSGrid[q & 0xFF], kIQ2XXSGrid[(q >> 8) & 0xFF], (q >> 24) & 1); }
  static uint signs(Chunk q) { return (q >> 16) & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return quant_iq2_coef(mt, (q >> 28) * 0x11u); }
};
// IQ2_XS: plane0 the group's four 16-bit entries, entry l (elements 8l..8l+7) a 9-bit grid index and a 7-bit sign
// index; plane1 its scales byte, the scale of the first 16 elements in bits 0..3 and of the others in 4..7; meta half
// d. value = d * (1 + 2 * scale) / 8 * signed grid value. A chunk: x the grid indices of half c & 1 of entries c >> 1
// (bits 0..15) and 2 + (c >> 1) (16..31), y the chunk's sign bits, the scales byte at bit 8 and the half at 16.
struct FmtIQ2XS {
  QUANT_FORMAT(GGUF_FMT_IQ2XS, QuantGrid, 0, 16, true);
  struct Payload { uint2 a; uchar b; }; typedef uint2 Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *p1}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = 16u * (c >> 1), h = 4u * (c & 1), e0 = (w.a.x >> l) & 0xFFFF, e1 = (w.a.y >> l) & 0xFFFF;
    const uint signs = ((quant_signs7(e0 >> 9) >> h) & 0xF) | ((quant_signs7(e1 >> 9) >> h) & 0xF) << 4;
    return uint2((e0 & 511) | (e1 & 511) << 16, signs | uint(w.b) << 8 | uint(c & 1) << 16);
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return quant_grid8(kIQ2XSGrid[q.x & 0xFFFF], kIQ2XSGrid[q.x >> 16], (q.y >> 16) & 1); }
  static uint signs(Chunk q) { return q.y & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return quant_iq2_coef(mt, q.y >> 8); }
};
// IQ2_S: plane0 the group's four low grid index bytes (word 0, byte l: entry l, elements 8l..8l+7) and their four
// sign bytes (word 1); plane1 the two high index bits of entry l at bits 2l, then the scales byte as IQ2_XS's; meta
// half d. value = d * (1 + 2 * scale) / 8 * signed grid value. A chunk is IQ2_XS's, with 10-bit grid indices.
struct FmtIQ2S {
  QUANT_FORMAT(GGUF_FMT_IQ2S, QuantGrid, 0, 16, true);
  struct Payload { uint2 a; ushort b; }; typedef uint2 Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device ushort *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = c >> 1, h = 4u * (c & 1), qh = w.b;
    const uint i0 = ((w.a.x >> (8 * l)) & 0xFF) | ((qh >> (2 * l)) & 3) << 8;
    const uint i1 = ((w.a.x >> (8 * l + 16)) & 0xFF) | ((qh >> (2 * l + 4)) & 3) << 8;
    const uint signs = ((w.a.y >> (8 * l + h)) & 0xF) | ((w.a.y >> (8 * l + 16 + h)) & 0xF) << 4;
    return uint2(i0 | i1 << 16, signs | (qh >> 8) << 8 | uint(c & 1) << 16);
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return quant_grid8(kIQ2SGrid[q.x & 0xFFFF], kIQ2SGrid[q.x >> 16], (q.y >> 16) & 1); }
  static uint signs(Chunk q) { return q.y & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return quant_iq2_coef(mt, q.y >> 8); }
};
// Bytes 0, 1 and 2, 3 of a word as the pair words of two pairs.
inline uint2 quant_byte_pairs(uint v) { return uint2((v & 0xFFu) | (v & 0xFF00u) << 8, ((v >> 16) & 0xFFu) | ((v >> 8) & 0xFF0000u)); }
// The linear codes of an IQ1 chunk, 8 (grid value + 1) + 1 + delta with delta = +-1 (Zero 9), from the kIQ1SGrid
// indices of pairs 0, 1 (bits 0..10) and pairs 2, 3 (11..21), the half of their entries (22) and their negative
// deltas (23, 24).
inline uint4 quant_iq1_codes(uint q) {
  const uint h = 4 * ((q >> 22) & 1);
  const uint2 lo = quant_byte_pairs((kIQ1SGrid[q & 0x7FF] >> h) & 0x0F0F0F0Fu);
  const uint2 hi = quant_byte_pairs((kIQ1SGrid[(q >> 11) & 0x7FF] >> h) & 0x0F0F0F0Fu);
  const uint2 delta = select(uint2(0x00020002u), uint2(0u), bool2((q >> 23) & 1, (q >> 24) & 1));
  return uint4(lo << 3 | delta.x, hi << 3 | delta.y);
}

// IQ1_S: plane0 the group's four low grid index bytes (entry l: elements 8l..8l+7), plane1 its qh, the three high
// index bits of entry l at bits 3l, the 3-bit scale at 12 and a negative delta at 15; meta half d. value = d * (2 *
// scale + 1) * (grid value + delta), delta = +-1/8, which is s * (code - 9) with s = d * (2 * scale + 1) / 8 and the
// codes of quant_iq1_codes. A chunk: the indices of half c & 1 of entries c >> 1 and 2 + (c >> 1), that half, the
// deltas and the scale at bits 25..27.
struct FmtIQ1S {
  QUANT_FORMAT(GGUF_FMT_IQ1S, QuantLinear, 9, 32, true);
  struct Payload { uint a; ushort b; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint *)p0), *((device ushort *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = c >> 1, qh = w.b;
    const uint i0 = ((w.a >> (8 * l)) & 0xFF) | ((qh >> (3 * l)) & 7) << 8;
    const uint i1 = ((w.a >> (8 * l + 16)) & 0xFF) | ((qh >> (3 * l + 6)) & 7) << 8;
    return i0 | i1 << 11 | uint(c & 1) << 22 | ((qh >> 15) & 1) * 3 << 23 | ((qh >> 12) & 7) << 25;
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint4 codes(Chunk q) { return quant_iq1_codes(q); }
  static QuantCoef coef(Meta mt, Chunk q) {
    return {float2(float(as_type<half>(mt)) * float(2 * ((q >> 25) & 7) + 1) * 0.125f)};
  }
};
// IQ1_M: plane0 the group's four low grid index bytes, plane1 its two qh bytes, entry l's three high index bits at
// bit 4l and its negative delta at 4l + 3; meta the block's 8 scale bytes, four halfwords, halfword k holding the
// 3-bit scales of the halves of groups 2k (bits 0..5) and 2k + 1 (6..11) and bits 4k..4k+3 of the half d at its
// bits 12..15. value = d * (2 * scale + 1) * (grid value + delta) as IQ1_S's, with a scale per 16 elements and a
// delta per 8.
struct FmtIQ1M {
  QUANT_FORMAT(GGUF_FMT_IQ1M, QuantLinear, 9, 16, false);
  struct Payload { uint a; ushort b; }; typedef uint Chunk; typedef uint2 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint *)p0), *((device ushort *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint2 *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = c >> 1, qh = w.b;
    const uint i0 = ((w.a >> (8 * l)) & 0xFF) | ((qh >> (4 * l)) & 7) << 8;
    const uint i1 = ((w.a >> (8 * l + 16)) & 0xFF) | ((qh >> (4 * l + 8)) & 7) << 8;
    return i0 | i1 << 11 | uint(c & 1) << 22 | ((qh >> (4 * l + 3)) & 1) << 23 | ((qh >> (4 * l + 11)) & 1) << 24;
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint4 codes(Chunk q) { return quant_iq1_codes(q); }
  static QuantCoef coef(Meta mt, ushort j) {
    const uint d = ((mt.x >> 12) & 0xF) | ((mt.x >> 24) & 0xF0) | ((mt.y >> 4) & 0xF00) | ((mt.y >> 16) & 0xF000);
    const uint scales = ((j < 4 ? mt.x : mt.y) >> (16 * ((j >> 1) & 1) + 6 * (j & 1)));   // shifts, as for Q4_K
    const float s = float(as_type<half>(ushort(d)));
    return {float2(s * float(2 * (scales & 7) + 1) * 0.125f, s * float(2 * ((scales >> 3) & 7) + 1) * 0.125f)};
  }
};
// Q4_0: plane0 4-bit codes; meta half d. value = d * (q - 8).
struct FmtQ40 {
  QUANT_FORMAT(GGUF_FMT_Q40, QuantLinear, 8, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q); }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// Q4_1: plane0 4-bit codes; meta half d, half m. value = d * q + m.
struct FmtQ41 {
  QUANT_FORMAT(GGUF_FMT_Q41, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q); }
  static QuantCoef coef(Meta mt, ushort) {
    return {float2(float(as_type<half>(ushort(mt & 0xFFFF)))), float2(float(as_type<half>(ushort(mt >> 16))))};
  }
};
// Q5_0 and Q5_1 (format F): Q5_K's planes, plane0 the low 4 bits and plane1 the fifth bits; meta and coefficients
// those of Q4_0 and Q4_1 (Scales), whose zero point the fifth bit doubles. value = d * (q - 16), or d * q + m.
template <uint F, class Scales> struct FmtQ5 {
  QUANT_FORMAT(F, QuantLinear, 2 * Scales::Zero, 32, false);
  typedef FmtQ5K::Payload Payload; typedef FmtQ5K::Chunk Chunk; typedef typename Scales::Meta Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return FmtQ5K::load(p0, p1); }
  static Meta loadMeta(device uchar *m) { return Scales::loadMeta(m); }
  static Chunk chunk(Payload w, ushort c) { return FmtQ5K::chunk(w, c); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return FmtQ5K::loadChunk(p0, p1, c); }
  static uint4 codes(Chunk q) { return FmtQ5K::codes(q); }
  static QuantCoef coef(Meta mt, ushort j) { return Scales::coef(mt, j); }
};
typedef FmtQ5<GGUF_FMT_Q50, FmtQ40> FmtQ50;
typedef FmtQ5<GGUF_FMT_Q51, FmtQ41> FmtQ51;
// MXFP4: plane0 codebook indices; meta the E8M0 exponent e per group. value = 2^(e - 128) * kFP4Values, whose
// entries are twice the E2M1 values (llama.cpp's GGML_E8M0_TO_FP32_HALF, 2^-128 and 2^-127 subnormal).
struct FmtMXFP4 {
  QUANT_FORMAT(GGUF_FMT_MXFP4, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uchar Meta; typedef float Scale;
  static half value(uint i) { return half(kFP4Values[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *m; }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta e, ushort) { return {float2(as_type<float>(e < 2 ? 0x00200000u << e : uint(e - 1) << 23))}; }
};
// NVFP4: plane0 codebook indices, the E2M1 codes; meta the 16 E4M3 scales of the unit's 16-groups (bytes 2j, 2j + 1
// of group j), then the tensor's FP32 scale g. value = g * e4m3 * E2M1 = kFP4Values (twice E2M1) * (g / 2 * e4m3),
// whose scale is (128 g) * quant_e4m3, rounded once to float.
struct FmtNVFP4 {
  QUANT_FORMAT(GGUF_FMT_NVFP4, QuantCodebook, 0, 16, false);
  struct Payload { uint4 a; }; typedef uint Chunk; struct Meta { packed_uint4 sc; float g; }; typedef float Scale;
  static half value(uint i) { return half(kFP4Values[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.sc = *((device packed_uint4 *)m); r.g = *((device float *)(m + 16)); return r; }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort j) {
    const uint pair = quant_scale_pair(mt.sc, j);   // the E4M3 scales of both 16-groups
    return {float2(quant_e4m3_pair(quant_byte_pairs(pair).x)) * (128.0f * mt.g)};
  }
};
// NVFP4_64: NVFP4's codes and value in 64-element blocks (a row that is not whole 256-element ones); meta the 4
// E4M3 scales of the unit's 16-groups (bytes 2j, 2j + 1 of group j), then g.
struct FmtNVFP464 {
  QUANT_FORMAT(GGUF_FMT_NVFP464, QuantCodebook, 0, 16, false);
  struct Payload { uint4 a; }; typedef uint Chunk; struct Meta { uint sc; float g; }; typedef float Scale;
  static half value(uint i) { return half(kFP4Values[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.sc = *((device uint *)m); r.g = *((device float *)(m + 4)); return r; }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort j) {
    const uint pair = mt.sc >> (16u * (j & 1));   // the E4M3 scales of both 16-groups
    return {float2(quant_e4m3_pair(quant_byte_pairs(pair).x)) * (128.0f * mt.g)};
  }
};
// FP8: plane0 E4M3 values (bytes 8c..8c+7 = chunk c, as Q8_0's); meta the tensor's FP32 scale g. value = g * e4m3,
// whose scale is 256 g.
struct FmtFP8 {
  QUANT_FORMAT(GGUF_FMT_FP8, QuantFloat8, 0, 32 * MetaGroups, false);
  struct Payload { uint4 a; uint4 b; }; typedef uint2 Chunk; typedef float Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0), *((device uint4 *)(p0 + 16))}; }
  static Meta loadMeta(device uchar *m) { return *((device float *)m); }
  static Chunk chunk(Payload w, ushort c) { const uint4 h = c < 2 ? w.a : w.b; return (c & 1) ? h.zw : h.xy; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint2 *)(p0 + 8 * c)); }
  static uint2 values(Chunk q) { return q; }
  static QuantCoef coef(Meta g, ushort) { return {float2(256.0f * g)}; }
};
// MXFP8: FP8's planes; meta the E8M0 exponent e of the group, as MXFP4's. value = 2^(e - 127) e4m3, whose scale is
// quant_mxfp8_scale(e).
struct FmtMXFP8 {
  QUANT_FORMAT(GGUF_FMT_MXFP8, QuantFloat8, 0, 32, false);
  typedef FmtFP8::Payload Payload; typedef FmtFP8::Chunk Chunk; typedef uchar Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return FmtFP8::load(p0, p1); }
  static Meta loadMeta(device uchar *m) { return *m; }
  static Chunk chunk(Payload w, ushort c) { return FmtFP8::chunk(w, c); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return FmtFP8::loadChunk(p0, p1, c); }
  static uint2 values(Chunk q) { return q; }
  static QuantCoef coef(Meta e, ushort) { return {float2(quant_mxfp8_scale(e))}; }
};
// BF16, F16: plane0 the group's 16-bit values (16 bytes 16c..16c+15 = chunk c); meta the FP32 scale g of 256
// elements (1 for a GGUF tensor). value = g * x.
template <uint F, class V> struct FmtFloat16 {
  QUANT_FORMAT(F, QuantFloat16, 0, 32 * MetaGroups, false);
  typedef V Value;
  struct Payload { uint4 a, b, c, d; }; typedef uint4 Chunk; typedef float Meta;
  static Payload load(device uchar *p0, device uchar *) {
    device const uint4 *w = (device const uint4 *)p0;
    return {w[0], w[1], w[2], w[3]};
  }
  static Meta loadMeta(device uchar *m) { return *((device float *)m); }
  static Chunk chunk(Payload w, ushort c) { return c < 2 ? (c ? w.b : w.a) : (c == 2 ? w.c : w.d); }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint4 *)(p0 + 16 * c)); }
  static uint4 values(Chunk q) { return q; }
  static QuantCoef coef(Meta g, ushort) { return {float2(g)}; }
};
typedef FmtFloat16<GGUF_FMT_BF16, bfloat> FmtBF16;
typedef FmtFloat16<GGUF_FMT_F16, half> FmtF16;
// PQ2_0 (GGUF tensor type 142) and Q2_0 (type 42): plane0 2-bit codes as Q2_K's; meta the
// half d of the native block, four groups (PQ2_0's 128 elements) or two (Q2_0's 64). value = d * (q - 1).
template <uint F> struct FmtTwoBit {
  QUANT_FORMAT(F, QuantLinear, 1, 32 * MetaGroups, false);
  struct Payload { uint2 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint2 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return quant_spread2(w.a[c >> 1] >> (16 * (c & 1))); }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return quant_spread2(*((device ushort *)(p0 + 2 * c))); }
  static uint4 codes(Chunk q) { return (uint4(q) >> uint4(0, 4, 8, 12)) & 0x00030003u; }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
typedef FmtTwoBit<GGUF_FMT_PQ20> FmtPQ20;
typedef FmtTwoBit<GGUF_FMT_Q20> FmtQ20;

// The GGUF format whose planes hold Bits-bit linear codes, which an MLX affine format keeps as it does
// (metal/abi/QuantFormat.h).
template <uint Bits> struct AffinePlanes;
template <> struct AffinePlanes<2> { typedef FmtQ2K F; };
template <> struct AffinePlanes<3> { typedef FmtQ3K F; };
template <> struct AffinePlanes<4> { typedef FmtQ4K F; };
template <> struct AffinePlanes<5> { typedef FmtQ5K F; };
template <> struct AffinePlanes<6> { typedef FmtQ6K F; };
template <> struct AffinePlanes<8> { typedef FmtQ80 F; };
// MLX affine, Bits-bit codes in groups of 32 * Groups elements: the planes of AffinePlanes<Bits>, the 8-bit codes
// unsigned; meta the group's bf16 scale s and bias z. value = s * code + z.
template <uint Bits, uint Groups> struct FmtAffine {
  typedef typename AffinePlanes<Bits>::F Planes;
  QUANT_FORMAT(quant_affine_format_of(Bits, 32 * Groups), QuantLinear, 0, 32 * MetaGroups, false);
  typedef typename Planes::Payload Payload; typedef typename Planes::Chunk Chunk; typedef uint Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return Planes::load(p0, p1); }
  static Meta loadMeta(device uchar *m) { return *((device uint *)m); }
  static Chunk chunk(Payload w, ushort c) { return Planes::chunk(w, c); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return Planes::loadChunk(p0, p1, c); }
  static uint4 codes(Chunk q) {
    if constexpr (Bits == 8) return uint4(quant_byte_pairs(q.x), quant_byte_pairs(q.y));
    else return Planes::codes(q);
  }
  static QuantCoef coef(Meta mt, ushort) {
    return {float2(float(as_type<bfloat>(ushort(mt & 0xFFFF)))), float2(float(as_type<bfloat>(ushort(mt >> 16))))};
  }
};
// Whether a linear format's codes reach 128, past the 7-bit codes a bf16 operand 128 + code holds exactly
// (kernels/decode/linear_gguf_sgmatrix.metal).
template <class F> struct QuantWideCodes { static constexpr constant bool value = false; };
template <uint Groups> struct QuantWideCodes<FmtAffine<8, Groups>> { static constexpr constant bool value = true; };
#define QUANT_AFFINE_GROUPS(B)                                                                                     \
  typedef FmtAffine<B, 1> FmtAF##B##G32; typedef FmtAffine<B, 2> FmtAF##B##G64; typedef FmtAffine<B, 4> FmtAF##B##G128;
QUANT_AFFINE_GROUPS(2) QUANT_AFFINE_GROUPS(3) QUANT_AFFINE_GROUPS(4) QUANT_AFFINE_GROUPS(5) QUANT_AFFINE_GROUPS(6)
QUANT_AFFINE_GROUPS(8)
#undef QUANT_AFFINE_GROUPS

#undef QUANT_FORMAT

// Every format as X(format type, kernel name token), the token being its kQuantFormats name: the formats of the
// Apple9 register tile (quant_register_format), then the others.
#define QUANT_REGISTER_FORMATS(X)                                                                                   \
  X(FmtQ4K, q4k) X(FmtIQ4XS, iq4xs) X(FmtIQ4NL, iq4nl) X(FmtQ5K, q5k) X(FmtQ6K, q6k) X(FmtQ3K, q3k) X(FmtQ80, q80) \
  X(FmtIQ3S, iq3s) X(FmtQ2K, q2k) X(FmtIQ3XXS, iq3xxs) X(FmtIQ2XXS, iq2xxs) X(FmtIQ2XS, iq2xs) X(FmtIQ2S, iq2s)      \
  X(FmtIQ1S, iq1s) X(FmtIQ1M, iq1m) X(FmtQ40, q40) X(FmtQ41, q41) X(FmtMXFP4, mxfp4) X(FmtPQ20, pq20)            \
  X(FmtAF2G32, af2g32) X(FmtAF2G64, af2g64) X(FmtAF2G128, af2g128) X(FmtAF3G32, af3g32) X(FmtAF3G64, af3g64)      \
  X(FmtAF3G128, af3g128) X(FmtAF4G32, af4g32) X(FmtAF4G64, af4g64) X(FmtAF4G128, af4g128) X(FmtAF5G32, af5g32)    \
  X(FmtAF5G64, af5g64) X(FmtAF5G128, af5g128) X(FmtAF6G32, af6g32) X(FmtAF6G64, af6g64) X(FmtAF6G128, af6g128)    \
  X(FmtAF8G32, af8g32) X(FmtAF8G64, af8g64) X(FmtAF8G128, af8g128) X(FmtNVFP4, nvfp4) X(FmtFP8, fp8)            \
  X(FmtQ50, q50) X(FmtQ51, q51) X(FmtBF16, bf16) X(FmtQ20, q20) X(FmtNVFP464, nvfp464) X(FmtMXFP8, mxfp8)
#define QUANT_FORMATS(X) QUANT_REGISTER_FORMATS(X) X(FmtF16, f16)

// Runs body(F()) with the format type of run-time format id `format` (GGUF_FMT_*), for kernels whose tiles pick
// their tensor, and so its format, at run time: any format (quant_format_switch), or one of the register tile's
// (quant_register_format_switch). The branch is uniform in a threadgroup. The host passes known ids only; any other
// decodes as the first format, so every id runs one body (with a path that runs none, the staged fused kernel failed
// its test under shader validation).
#define QUANT_FORMAT_CASE(F, f) case F::Id: body(F()); break;
template <class Body>
inline void quant_format_switch(uint format, Body body) {
  switch (format) {
  default:
    QUANT_FORMATS(QUANT_FORMAT_CASE)
  }
}
template <class Body>
inline void quant_register_format_switch(uint format, Body body) {
  switch (format) {
  default:
    QUANT_REGISTER_FORMATS(QUANT_FORMAT_CASE)
  }
}
#undef QUANT_FORMAT_CASE

// QUANT_FORMATS lists every format, by its kQuantFormats name, and so does the switch above; QUANT_REGISTER_FORMATS
// lists exactly those of quant_register_format.
#define QUANT_FORMAT_ONE(F, f) +1
#define QUANT_REGISTER_ONE(F, f) +quant_register_format(F::Id)
static_assert(0 QUANT_FORMATS(QUANT_FORMAT_ONE) == GGUF_FMT_COUNT, "QUANT_FORMATS lists every format");
static_assert(0 QUANT_REGISTER_FORMATS(QUANT_REGISTER_ONE) == 0 QUANT_REGISTER_FORMATS(QUANT_FORMAT_ONE) &&
                  0 QUANT_FORMATS(QUANT_REGISTER_ONE) == 0 QUANT_REGISTER_FORMATS(QUANT_FORMAT_ONE),
              "QUANT_REGISTER_FORMATS lists the register tile's formats");
#undef QUANT_REGISTER_ONE
#undef QUANT_FORMAT_ONE
template <uint N> constexpr bool quant_format_named(uint id, const constant char (&token)[N]) {
  for (uint i = 0; i < N; ++i)
    if (kQuantFormats[id].name[i] != token[i]) return false;
  return true;
}
#define QUANT_FORMAT_NAME(F, f) static_assert(quant_format_named(F::Id, #f), #f " is not its kQuantFormats name");
QUANT_FORMATS(QUANT_FORMAT_NAME)
#undef QUANT_FORMAT_NAME
// ---- end runtime/metal/kernels/common/quant_formats.h ----

// The GGUF tiles' dequantization (kernels/shared/gguf_linear.metal and kernels/shared/moe_gguf.metal, through
// kernels/common/gguf_staged_tile.h and kernels/common/gguf_column_tile.h), shared with the dequantization test.
// Includers set `#pragma clang fp reassociate(off)` first, so the source order of float operations holds here.
// One thread decodes one column's group of 32 (kernels/common/quant_formats.h) to half, each value rounded once, and
// hands it to sink(c, lo, hi) by chunk: lo holds chunk c's pairs 0, 1 (elements 4c..4c+3), hi its pairs 2, 3
// (elements 16+4c..16+4c+3). The staged tiles store the group into their stage (dequant32), the column tile keeps it
// in registers.
template <class F>
inline half2 staged_linear(uint pair, float s, float m) {
  // 0x6400 is half 1024, whose ulp is 1: or-ing a code into its mantissa makes 1024 + code, exactly.
  const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + F::Zero)));
  if constexpr (F::Zero) return half2(code * s);
  else return half2(fma(code, float2(s), float2(m)));
}
template <class F, class Sink>
inline void dequant32_chunks(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl, Sink sink) {
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(meta, F::chunk(w, 0)); else k = F::coef(meta, j);
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    const typename F::Chunk q = F::chunk(w, c);
    half4 lo, hi;
    if constexpr (F::Kind == QuantLinear) {
      const uint4 p = F::codes(q);
      lo = half4(staged_linear<F>(p.x, k.s.x, k.m.x), staged_linear<F>(p.y, k.s.x, k.m.x));
      hi = half4(staged_linear<F>(p.z, k.s.y, k.m.y), staged_linear<F>(p.w, k.s.y, k.m.y));
    } else if constexpr (F::Kind == QuantCodebook) {   // value * s in Scale, then to half, as the reference rounds it
      typedef typename F::Scale S;
      const uchar4 b = as_type<uchar4>(F::indices(q));
      lo = half4(half2(vec<S, 2>(tl[b.x]) * S(k.s.x)), half2(vec<S, 2>(tl[b.y]) * S(k.s.x)));
      hi = half4(half2(vec<S, 2>(tl[b.z]) * S(k.s.y)), half2(vec<S, 2>(tl[b.w]) * S(k.s.y)));
    } else if constexpr (F::Kind == QuantInt8) {
      typedef typename F::Scale S;
      const uint2 v = F::values(q);
      lo = half4(vec<S, 4>(as_type<char4>(v.x)) * S(k.s.x));
      hi = half4(vec<S, 4>(as_type<char4>(v.y)) * S(k.s.y));
    } else if constexpr (F::Kind == QuantFloat8) {   // exact halves times s in float, then to half, as the reference rounds it
      const uint2 v = F::values(q);
      lo = half4(float4(quant_e4m3_word(v.x)) * k.s.x);
      hi = half4(float4(quant_e4m3_word(v.y)) * k.s.y);
    } else if constexpr (F::Kind == QuantFloat16) {   // the values times s in float, then to half
      typedef vec<typename F::Value, 2> V2;
      const uint4 v = F::values(q);
      lo = half4(float4(float2(as_type<V2>(v.x)), float2(as_type<V2>(v.y))) * k.s.x);
      hi = half4(float4(float2(as_type<V2>(v.z)), float2(as_type<V2>(v.w))) * k.s.y);
    } else {
      const uint2 g = F::grid(q); const uint s = F::signs(q);
      lo = half4(float4(as_type<uchar4>(g.x)) * k.s.x);
      hi = half4(float4(as_type<uchar4>(g.y)) * k.s.y);
      lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
      hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
    }
    sink(c, lo, hi);
  }
}
// The group into a stage column: chunk c's pairs 0, 1 at dst + 4c, pairs 2, 3 at dst + 16 + 4c.
template <class F>
inline void dequant32(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl,
                      threadgroup half *dst) {
  dequant32_chunks<F>(w, meta, j, tl, [&](ushort c, half4 lo, half4 hi) __attribute__((always_inline)) {
    *((threadgroup half4 *)(dst + 4 * c)) = lo; *((threadgroup half4 *)(dst + 16 + 4 * c)) = hi;
  });
}
// ---- end runtime/metal/kernels/common/gguf_staged.h ----

#include <metal_stdlib>
using namespace metal;

// The column tile of kernels/shared/moe_gguf.metal and kernels/decode/linear_gguf_column.metal: a GEMV of few rows
// in registers. Lane l of a simdgroup owns output column `column`, decodes its column's groups of 32 into registers
// (dequant32_chunks) and multiplies each weight into the fp32 sums of its rows: no stage, no barrier, Rows FMAs per
// weight. The staged tile runs a dequantize, barrier and matmul chain per 32-input step on padded 8-row MPP tiles,
// which bounds it at few rows, not DRAM, so the column tile runs the MoE expert tiles and the decode projections of at
// most GGUF_COLUMN_ROWS live rows (moe_column_tile, ops::LinearTile::GgufColumn). Includers set
// `#pragma clang fp reassociate(off)` first.

// Each row's sum over groups [g_begin, g_end) of K, of its first `rows` rows (at most Rows): per group one fp32 chain
// of its 32 products in element order, then added to the row's sum. Every row sums alone, so a row's result does not
// depend on the other rows, on its place among them or on Rows. The chain is written as multiplies and adds, whose
// order `reassociate(off)` keeps: Metal's fma() is emitted with every fast-math flag, reassociation among them, which
// the pragma does not clear. Each product of a half weight and a bf16 input is exact in fp32, so fused or not, every
// step rounds once (and as fast as the fma() chain). Each thread loads the next group's payload while the group's FMAs
// run, and its meta unit when the next group enters a new one, and reads the activations 16 bytes at a time (their
// rows of K, a multiple of 64, start 16-byte aligned in page-aligned buffers).
template <class F, ushort Rows>
inline void column_dot(device const bfloat *x, uint rows, uint input_size, device uchar *w0, device uchar *w1,
                       device uchar *meta, uint column, uint g_begin, uint g_end, threadgroup half2 *tl,
                       thread float (&sums)[Rows]) {
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = column / QUANT_TILE_ROWS, plane_row = column % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
#pragma unroll
  for (ushort r = 0; r < Rows; ++r) sums[r] = 0.0f;
  typename F::Payload w =
      F::load(tw0 + ulong(g_begin) * QUANT_TILE_ROWS * F::P0, tw1 + ulong(g_begin) * QUANT_TILE_ROWS * F::P1);
  typename F::Meta m = F::loadMeta(tmeta + ulong(g_begin / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
  for (uint g = g_begin; g < g_end; ++g) {
    half4 v[8];   // v[i]: elements 4i..4i+3 of the group
    dequant32_chunks<F>(w, m, ushort(g % F::MetaGroups), tl,
                        [&](ushort c, half4 lo, half4 hi) __attribute__((always_inline)) {
                          v[c] = lo;
                          v[4 + c] = hi;
                        });
    if (g + 1 < g_end) {
      w = F::load(tw0 + ulong(g + 1) * QUANT_TILE_ROWS * F::P0, tw1 + ulong(g + 1) * QUANT_TILE_ROWS * F::P1);
      if ((g + 1) / F::MetaGroups != g / F::MetaGroups)
        m = F::loadMeta(tmeta + ulong((g + 1) / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
    }
#pragma unroll
    for (ushort r = 0; r < Rows; ++r) {
      if (r >= rows) break;
      device const uint4 *xr = reinterpret_cast<device const uint4 *>(x + ulong(r) * input_size + g * 32);
      float sum = 0.0f;
#pragma unroll
      for (ushort i = 0; i < 8; ++i) {
        const uint4 q = xr[i / 2];
        const float4 xv = float4(as_type<bfloat4>(i % 2 ? q.zw : q.xy)), wv = float4(v[i]);
        sum = sum + wv.x * xv.x;
        sum = sum + wv.y * xv.y;
        sum = sum + wv.z * xv.z;
        sum = sum + wv.w * xv.w;
      }
      sums[r] += sum;
    }
  }
}

// column_dot in the run-time format `format` (uniform per threadgroup), for each format `formats` switches to, after
// its pair table, which every thread of the threadgroup fills.
template <ushort Rows, class Formats>
inline void column_dot_any(uint format, device const bfloat *x, uint rows, uint input_size, device uchar *w0,
                           device uchar *w1, device uchar *meta, uint column, uint g_begin, uint g_end,
                           threadgroup half2 *tl, uint thread_index, uint threads, thread float (&sums)[Rows],
                           Formats formats) {
  formats(format, [&](auto f) {
    typedef decltype(f) F;
    quant_pair_table<F>(tl, thread_index, threads);
    column_dot<F, Rows>(x, rows, input_size, w0, w1, meta, column, g_begin, g_end, tl, sums);
  });
}
// ---- end runtime/metal/kernels/common/gguf_column_tile.h ----
// ---- begin runtime/metal/kernels/common/gguf_segments.h (Splash 2afc733, Apache-2.0) ----

// Fused projections (qkv|z|ab, q|k|v; ops::Linear): up to three column segments of any formats in one dispatch, so
// the small segments do not run as dispatches of their own, on the staged and prefill tiles
// (kernels/shared/gguf_linear.metal) and the column tile (kernels/decode/linear_gguf_column.metal). The segments'
// planes bind from buffer 1, three each (GGUF_SEGMENTS); the dispatch's threadgroups cover their columns in order,
// `width` columns a threadgroup, and each threadgroup decodes its segment's planes and format.
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_SEGMENTS GGUF_SEGMENT(1, w0a, w1a, ma), GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc)
// The segment s of threadgroup `tile` among the segments' threadgroups of `width` columns (cols, 0 past the last) and
// the threadgroup's index `local` within it; GGUF_SEGMENT_PLANE the segment's `plane` of x##a, x##b and x##c;
// GGUF_SEGMENT_OF both, with the segment's planes w0, w1 and meta.
#define GGUF_SEGMENT_INDEX(tile, cols, width)                                                                      \
  const uint t0 = (cols)[0] / (width), t1 = t0 + (cols)[1] / (width);                                              \
  const uint s = (tile) < t0 ? 0 : (tile) < t1 ? 1 : 2, local = (tile) - (s == 0 ? 0 : s == 1 ? t0 : t1)
#define GGUF_SEGMENT_PLANE(plane, x) device uchar *plane = s == 0 ? x##a : s == 1 ? x##b : x##c
#define GGUF_SEGMENT_OF(tile, cols, width)                                                                         \
  GGUF_SEGMENT_INDEX(tile, cols, width);                                                                           \
  GGUF_SEGMENT_PLANE(w0, w0);                                                                                      \
  GGUF_SEGMENT_PLANE(w1, w1);                                                                                      \
  GGUF_SEGMENT_PLANE(meta, m)
// ---- end runtime/metal/kernels/common/gguf_segments.h ----
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

namespace gguf_column {

// A tensor's planes, bound in slots (w0, w1, meta).
struct Planes {
  device uchar *w0, *w1, *meta;
};

// A threadgroup's memory: the codebook formats' pair table and each simdgroup's parts of the live rows' sums (a
// one-pass gate/up's of both tensors).
template <bool GateUp> struct Shared {
  half2 table[kQuantPairTableEntries];
  float parts[GGUF_COLUMN_SPLITS][GateUp ? 2 : 1][GGUF_COLUMN_ROWS][32];
};

template <GgufEpilogue Ep, bool GateUp, class Out>
inline void project(device const bfloat *input, Planes planes, Planes up, device const bfloat *aux,
                    device Out *output, const GgufColumnParams p, uint tile, uint sg, uint lane,
                    threadgroup Shared<GateUp> &shared) {
  const uint tid = sg * 32 + lane, K = p.input_size, groups = K / 32;
  const uint first = sg * groups / GGUF_COLUMN_SPLITS, last = (sg + 1) * groups / GGUF_COLUMN_SPLITS;
  const uint column = tile * GGUF_COLUMN_COLUMNS + lane;
  float sums[GGUF_COLUMN_ROWS], up_sums[GGUF_COLUMN_ROWS];
  quant_format_switch(p.format, [&](auto f) {
    typedef decltype(f) F;
    quant_pair_table<F>(shared.table, tid, GGUF_COLUMN_THREADS);
    column_dot<F, GGUF_COLUMN_ROWS>(input, p.rows, K, planes.w0, planes.w1, planes.meta, column, first, last,
                                    shared.table, sums);
    if constexpr (GateUp)
      column_dot<F, GGUF_COLUMN_ROWS>(input, p.rows, K, up.w0, up.w1, up.meta, column, first, last, shared.table,
                                      up_sums);
  });
  for (uint r = 0; r < GGUF_COLUMN_ROWS; ++r) {
    shared.parts[sg][0][r][lane] = sums[r];
    if constexpr (GateUp) shared.parts[sg][1][r][lane] = up_sums[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg) return;
  for (uint r = 0; r < p.rows; ++r) {
    float total = 0.0f, up_total = 0.0f;
    for (uint s = 0; s < GGUF_COLUMN_SPLITS; ++s) {
      total += shared.parts[s][0][r][lane];
      if constexpr (GateUp) up_total += shared.parts[s][1][r][lane];
    }
    const ulong at = ulong(r) * p.out_stride + p.out_offset + column;
    if constexpr (GateUp)
      output[at] = Out(float(bfloat(up_total)) * splash_silu(float(bfloat(total))));
    else
      output[at] = gguf_epilogue<Ep, Out>(total, aux, at);
  }
}

} // namespace gguf_column

// Grid (the tensor's columns / GGUF_COLUMN_COLUMNS), GGUF_COLUMN_THREADS threads. Bindings: input, the tensor's planes
// (w0, w1, meta), output; then the residual (_r), or a one-pass gate/up's up planes (_gate_up: the first are the
// gate's); then GgufColumnParams.
#define GGUF_COLUMN_ARGS                                                                                       \
  device const bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],   \
      device uchar *meta [[buffer(3)]]
#define GGUF_COLUMN_THREAD_ARGS                                                                                \
  uint tile [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]
#define GGUF_COLUMN_PLAIN(Name, Out)                                                                           \
  kernel void Name(GGUF_COLUMN_ARGS, device Out *output [[buffer(4)]],                                         \
                   constant GgufColumnParams &p [[buffer(5)]], GGUF_COLUMN_THREAD_ARGS) {                      \
    threadgroup gguf_column::Shared<false> shared;                                                             \
    gguf_column::project<EpNone, false, Out>(input, {w0, w1, meta}, {w0, w1, meta}, input, output, p, tile, sg, \
                                             lane, shared);                                                    \
  }
GGUF_COLUMN_PLAIN(gguf_column_a, bfloat)
GGUF_COLUMN_PLAIN(gguf_column_a_f32, float)
kernel void gguf_column_gate_up(GGUF_COLUMN_ARGS, device bfloat *output [[buffer(4)]],
                                device uchar *up_w0 [[buffer(5)]], device uchar *up_w1 [[buffer(6)]],
                                device uchar *up_meta [[buffer(7)]], constant GgufColumnParams &p [[buffer(8)]],
                                GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<true> shared;
  gguf_column::project<EpNone, true, bfloat>(input, {w0, w1, meta}, {up_w0, up_w1, up_meta}, input, output, p, tile,
                                             sg, lane, shared);
}
kernel void gguf_column_r(GGUF_COLUMN_ARGS, device bfloat *output [[buffer(4)]],
                          device const bfloat *residual [[buffer(5)]], constant GgufColumnParams &p [[buffer(6)]],
                          GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<false> shared;
  gguf_column::project<EpResidual, false, bfloat>(input, {w0, w1, meta}, {w0, w1, meta}, residual, output, p, tile,
                                                  sg, lane, shared);
}
// A fused projection's segments in one dispatch (kernels/common/gguf_segments.h), as gguf_decode_fused_m<rows> runs
// them on the staged tile: the threadgroup's columns pick their segment, whose planes and format they decode as the
// segment's own dispatch does, bit for bit. Grid (the segments' columns / GGUF_COLUMN_COLUMNS). Bindings: input, the
// segments' planes, output; then GgufColumnFusedParams.
kernel void gguf_column_fused(device const bfloat *input [[buffer(0)]], GGUF_SEGMENTS,
                              device bfloat *output [[buffer(10)]], constant GgufColumnFusedParams &p [[buffer(11)]],
                              GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<false> shared;
  GGUF_SEGMENT_OF(tile, p.cols, GGUF_COLUMN_COLUMNS);
  const GgufColumnParams segment{p.rows, p.input_size, p.out_stride, p.offset[s], p.fmt[s]};
  gguf_column::project<EpNone, false, bfloat>(input, {w0, w1, meta}, {w0, w1, meta}, input, output, segment, local,
                                              sg, lane, shared);
}
#undef GGUF_COLUMN_PLAIN
#undef GGUF_COLUMN_THREAD_ARGS
#undef GGUF_COLUMN_ARGS
#undef GGUF_SEGMENT_OF
#undef GGUF_SEGMENT_PLANE
#undef GGUF_SEGMENT_INDEX
#undef GGUF_SEGMENTS
#undef GGUF_SEGMENT
// ---- end runtime/metal/kernels/decode/linear_gguf_column.metal ----
