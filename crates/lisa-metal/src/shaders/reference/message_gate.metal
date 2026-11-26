// Vendored from Splash (https://github.com/incoai/splash), revision
// 2afc7337aacc044722fa95e4cb3da75c281a19a8, Apache License 2.0 (LICENSE-APACHE-2.0).
// Quantized GGUF decoding follows ggml (MIT, LICENSE-GGML). Each original
// header is inlined once at its first include, between begin/end markers; the
// code is otherwise unchanged. Source hashes: provenance.json.
// ---- begin runtime/metal/kernels/shared/message_gate.metal (Splash 2afc733, Apache-2.0) ----
#pragma clang fp reassociate(off)
// ---- begin runtime/metal/abi/KernelABI.h (Splash 2afc733, Apache-2.0) ----

// Shader prelude. The ABI headers below also compile as host C++; their
// shared parameter layouts use fixed-width scalars and explicit padding.

// ---- begin runtime/metal/abi/AneFfn.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts and constants of the GPU side of the ANE FFN split
// (ops/AneFfn.cpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The Hadamard rotation blocks: the inputs gate and up multiply rotate in
// blocks of ANE_FFN_INPUT_BLOCK values, the ANE's intermediate rows, which
// down multiplies, in blocks of ANE_FFN_INTERMEDIATE_BLOCK, whose larger span
// spreads their larger outliers further. A simdgroup rotates 128 values at a
// time, four per lane.
#define ANE_FFN_ROTATION_UNIT 128u
#define ANE_FFN_INPUT_BLOCK 128u
#define ANE_FFN_INTERMEDIATE_BLOCK 512u
// ane_ffn_rotate: ANE_FFN_ROTATE_THREADS threads per row, each simdgroup
// rotating at most ANE_FFN_ROTATE_BLOCKS input blocks of the row.
#define ANE_FFN_ROTATE_THREADS 256u
#define ANE_FFN_ROTATE_BLOCKS 8u
// ane_ffn_weights and ane_ffn_row_scale: one simdgroup per weight row,
// ANE_FFN_WEIGHT_ROWS rows per threadgroup.
#define ANE_FFN_WEIGHT_ROWS 8u
#define ANE_FFN_WEIGHT_THREADS (ANE_FFN_WEIGHT_ROWS * 32u)
// ane_ffn_pack and ane_ffn_join: tiles of ANE_FFN_TILE rows by ANE_FFN_TILE
// channels, ANE_FFN_TILE x ANE_FFN_TILE_ROWS threads each.
#define ANE_FFN_TILE 32u
#define ANE_FFN_TILE_ROWS 8u
// int8 values span +-ANE_FFN_INT8_PEAK. The ANE dequantizes them by
// 1 / ANE_FFN_INT8_UNIT, so that fp16 cannot overflow, which every scale
// carries back.
#define ANE_FFN_INT8_PEAK 127.0f
#define ANE_FFN_INT8_UNIT 128.0f
// The least peak a token's or a weight row's int8 scale is taken over:
// fp16's least normal value, so that every scale is a normal half, its
// inverse finite, and an all-zero row takes codes of 0.
#define ANE_FFN_PEAK_FLOOR 0x1p-14f

struct AneFfnRotateParams {
  uint32_t hidden;
};

// Rotated rows [rows][hidden] as the channel-major int8 inputs
// [channel, channel + 32 * grid.y) of the ANE, stride bytes per channel.
struct AneFfnPackParams {
  uint32_t hidden;
  uint32_t channel;
  uint32_t stride;
};

// Rotated int8 of a projection: its output rows [row, row + rows) and inputs
// [input, input + width), where groups is its input count / 32 (an image
// tensor of GGUF_FMT_* format), into rows of stride bytes and their scales
// scale_stride halves apart; the ANE's channels, gate's and up's rows or
// down's inputs, in descending units (ane_ffn_descending).
struct AneFfnWeightParams {
  uint32_t groups;
  uint32_t row;
  uint32_t input;
  uint32_t width;
  uint32_t stride;
  uint32_t scale_stride;
  uint32_t format;
  uint32_t rows;
};

// output[t][c] += partial[c][t] * partial[hidden][t] * token_scale[t], in
// fp32, for rows t < rows, setting a status word if one of those values is
// not finite; stride halves per channel of the partial.
struct AneFfnJoinParams {
  uint32_t hidden;
  uint32_t stride;
  uint32_t rows;
};

static_assert(sizeof(AneFfnRotateParams) == 4, "ANE FFN rotate parameters are 4 bytes");
static_assert(sizeof(AneFfnPackParams) == 12, "ANE FFN pack parameters are 12 bytes");
static_assert(sizeof(AneFfnWeightParams) == 32, "ANE FFN weight parameters are 32 bytes");
static_assert(sizeof(AneFfnJoinParams) == 12, "ANE FFN join parameters are 12 bytes");
// ---- end runtime/metal/abi/AneFfn.h ----
// ---- begin runtime/metal/abi/DraftAttention.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
// ---- begin runtime/metal/abi/ExecutionGeometry.h (Splash 2afc733, Apache-2.0) ----

// C/Metal ABI constants shared by host and shader compilation. Compile-time
// static assertions, beside the model's ExecutionLimits (model/Model.hpp)
// and the operators sized by these, check the contracts between them.
#define SPLASH_DRAFT_QUERY_ROWS 8u
#define SPLASH_DRAFT_PROPOSAL_TOKENS 7u
// The most rows of a lane's verify block: its anchor's and the rows proposed
// after it, a chain or a tree of them (metal/abi/VerifyTree.h), which may
// hold more rows than the draft's block: alternatives beside the draft's
// proposals, and chains deeper than them. Every target kernel of a verify
// takes blocks of 1 to this many rows a lane. Sixteen: a replay of 17,432
// served cycles estimated trees of up to 16 rows a lane, the draft's
// proposals with lookup branches, to decode 28% faster than the draft's
// eight rows on Qwen3.8-27B. A decode cycle's bounds follow it: the KV a
// lane's block writes past the context (SPLASH_SPECULATIVE_SCRATCH_TOKENS),
// the KV the engine holds for a lane's cycle and the decode arena's rows.
#define SPLASH_TARGET_VERIFY_ROWS 16u
// The rows of a lane's verify block and of its pending path that the narrow
// GDN decode kernels hold (kernels/decode/gdn.metal, those without _r16, which
// hold SPLASH_TARGET_VERIFY_ROWS): a draft's block, a chain or a tree. A step
// takes them when none of its lanes may hold more rows (ops::verifyLaneCap)
// and none of its pending paths holds more (ops::GDN::addDecode).
#define SPLASH_GDN_NARROW_ROWS 8u
// The KV ceiling: four times the targets' native window of 262,144 tokens,
// which holds the most context YaRN serves them
// (QwenTargetDimensions::yarnContextTokens).
#define SPLASH_MAXIMUM_CONTEXT_TOKENS 1048576u
// The KV slots past the context a decode cycle writes: a lane's block at the
// context's last position stores its rows from there, 15 past it.
#define SPLASH_SPECULATIVE_SCRATCH_TOKENS                                  \
  (SPLASH_TARGET_VERIFY_ROWS - 1u)
#define SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS                                  \
  (SPLASH_MAXIMUM_CONTEXT_TOKENS + SPLASH_SPECULATIVE_SCRATCH_TOKENS)
// The most lanes of a decode step, the most requests a runtime may be started
// to run at once (--max-concurrent-requests). A step is ragged, so its lanes
// share its rows: past a lane's eighth row its rows hold few tokens (a replay
// of chat cycles on Qwen3.6-35B-A3B: 64 rows hold 13.3 tokens over four
// lanes, 25.0 over eight), while a lane costs what about five rows do (1.6 to
// 2.7 ms a cycle on a 36 GB M5 Max, Qwen3.6-35B-A3B and Qwen3.8-27B). Eight
// is the most the kernels' lane bindings hold (kernels/common/lane_bindings.h).
#define SPLASH_MAXIMUM_BATCH_WIDTH 8u
// The lanes a runtime runs unless its start asks for others: Splash 1.7.0's
// four, whose memory and context a default start keeps. More
// lanes cost each a lane state, its arenas' share and, past four, context
// where memory is short; a start says what they cost (RuntimeResources).
#define SPLASH_DEFAULT_LANES 4u
#define SPLASH_PREFILL_TOKEN_BUDGET 2048u
#define SPLASH_DRAFT_SLIDING_WINDOW 2048u
#define SPLASH_TARGET_KV_BLOCK_TOKENS 32u
// Attention runs a sequence's rows in tiles of this many, each over at most
// SPLASH_ATTENTION_MAXIMUM_SPLITS history splits (ops::PagedAttention).
#define SPLASH_ATTENTION_TILE_ROWS 8u
#define SPLASH_ATTENTION_MAXIMUM_SPLITS 128u
// Draft attention deals the live ring tiles of one (lane, KV head)
// round-robin to this many groups, the last of which also attends the eight
// current rows. Fixed rather than derived from the GPU so the combine order,
// and with it the rounding, is the same on every machine and lane count.
#define SPLASH_DRAFT_ATTENTION_SPLITS 4u
#define SPLASH_TARGET_SAMPLING_SHARDS 16u
// Threads of each group that selects a sampled row over the whole
// vocabulary (decode_sample_vocabulary*); a bracket of at most this many
// tokens is ordered in threadgroup memory, one token per thread.
#define SPLASH_TARGET_VOCABULARY_THREADS 1024u
// Groups that share such a row's draw, each over its own slice of the
// vocabulary, so the eight rows of a lane spread over the GPU's cores.
#define SPLASH_TARGET_VOCABULARY_GROUPS 8u
// The ranges of the vocabulary such a row's draw sums: one per simdgroup of
// the row's groups.
#define SPLASH_TARGET_VOCABULARY_RANGES                                    \
  (SPLASH_TARGET_VOCABULARY_GROUPS * (SPLASH_TARGET_VOCABULARY_THREADS / 32u))
#define SPLASH_DRAFT_SAMPLING_SHARDS 8u
// Candidates the draft selector keeps per proposal position; acceptance and
// the sampled draw read them.
#define SPLASH_DRAFT_CANDIDATES 16u
// The rank of the draft selector's codebooks.
#define SPLASH_DRAFT_SELECTOR_RANK 256u
// Rows of one value head's recurrent state a prefill GDN scan threadgroup
// carries through the chunk: four simdgroups whose lanes each own sixteen key
// columns of one row. The thread count follows from the rows: 128 columns /
// 16 per lane = 8 lanes per row, times the 16 rows; a static_assert in
// prefill/gdn.metal ties the two literals together.
#define SPLASH_GDN_SCAN_STATE_ROWS 16u
#define SPLASH_GDN_SCAN_THREADS 128u
// The epsilon of every RMS norm the kernels compute, which a target's and a
// draft's configurations must state. Unsuffixed, so the host compares a
// configuration's double with it exactly and the kernels take it as a float.
#define SPLASH_RMS_EPSILON 1e-6
// Plain norms of at most SPLASH_STAGED_NORM_ROWS rows of at most
// SPLASH_STAGED_NORM_WIDTH columns run norm_rms_staged, whose 1024-thread
// groups hold a row in threadgroup memory: the region where it beats norm_rms
// (shared/normalization.metal), which covers every decode norm of a 2048-wide
// model and its short prefill chunks.
#define SPLASH_STAGED_NORM_WIDTH 2048u
#define SPLASH_STAGED_NORM_ROWS 64u
#define SPLASH_STAGED_NORM_THREADS 1024u
// ---- end runtime/metal/abi/ExecutionGeometry.h ----
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The draft attention the kernels are compiled for, the only one
// ops::DraftAttention accepts: 32 query heads over 8 KV heads of 128
// dimensions, so a q|k|v row holds 6144 values and an attention row 4096.
// The attention core runs a KV head's four query heads over the eight query
// rows together, as 32 attention rows.
#define SPLASH_DRAFT_QUERY_HEADS 32u
#define SPLASH_DRAFT_KV_HEADS 8u
#define SPLASH_DRAFT_HEAD_DIMENSION 128u
#define SPLASH_DRAFT_ATTENTION_WIDTH                                           \
  (SPLASH_DRAFT_QUERY_HEADS * SPLASH_DRAFT_HEAD_DIMENSION)
#define SPLASH_DRAFT_QKV_WIDTH                                                 \
  (SPLASH_DRAFT_ATTENTION_WIDTH +                                              \
   2u * SPLASH_DRAFT_KV_HEADS * SPLASH_DRAFT_HEAD_DIMENSION)
#define SPLASH_DRAFT_ATTENTION_ROWS                                            \
  (SPLASH_DRAFT_QUERY_HEADS / SPLASH_DRAFT_KV_HEADS * SPLASH_DRAFT_QUERY_ROWS)

// The dynamic convolutions of a draft layer (draft_conv), each run in two
// stages, before its attention or MLP and after it with the residual: a
// stage weighs a channel's row and the row before it, its two taps, each by
// a base weight plus the dynamic weight of the channel's group of 16
// channels. The draft projects every row's dynamic weights, for each stage,
// tap and group of a `hidden`-wide layer.
#define SPLASH_DRAFT_CONVOLUTION_STAGES 2u
#define SPLASH_DRAFT_CONVOLUTION_TAPS 2u
#define SPLASH_DRAFT_CONVOLUTION_GROUP 16u
inline constexpr uint32_t draft_dynamic_width(uint32_t hidden) {
  return SPLASH_DRAFT_CONVOLUTION_STAGES * SPLASH_DRAFT_CONVOLUTION_TAPS *
         (hidden / SPLASH_DRAFT_CONVOLUTION_GROUP);
}

// The batched draft kernels' grids cover exactly the dispatch's lanes:
// per-lane arrays hold those lanes, and entries past them are zero and unread.

// A convolution stage (finish: the second, with the residual) of a draft
// `hidden` wide, a whole number of convolution groups.
struct DraftConvBatchParams {
  uint32_t finish;
  uint32_t hidden;
};

static_assert(sizeof(DraftConvBatchParams) == 8,
              "Draft convolution parameters are 8 bytes on both sides");

// The attention core's split count (SPLASH_DRAFT_ATTENTION_SPLITS) and its
// rings' slots per KV head (SPLASH_DRAFT_SLIDING_WINDOW) are compiled in.
// value_stride, the values ring's stride between head dimensions, is the
// window too (the op passes nothing else) but stays a run-time value: with it
// compiled into the value tiles, the split kernel returns wrong rows at
// random, and reports no error, on an M5 Max under full
// MTL_SHADER_VALIDATION; without validation its output is bitwise that of
// the run-time stride.
struct DraftAttentionBatchParams {
  uint32_t value_stride;
  uint32_t lanes;
  uint32_t cache_length[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftAttentionBatchParams) == 8 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft attention parameters are the same size on both sides");

struct DraftContextParams {
  uint32_t tokens;
  uint32_t start_position;
};

static_assert(sizeof(DraftContextParams) == 8,
              "Draft context prefill parameters are 8 bytes on both sides");

// The context commit of each lane's retained rows (the mask acceptance
// writes, AcceptBatchParams) of its lane rows (SplashLaneRows): the context
// rows among the block's target rows, the path's i-th at position
// start_position[l] + i, rotated by the lane's RoPE row i (a lane's RoPE rows
// are its verify block's positions, SPLASH_TARGET_VERIFY_ROWS a lane).
struct DraftContextBatchParams {
  uint32_t start_position[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftContextBatchParams) == 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft context commit parameters are the same size on both sides");

// A context window holds, per ring slot, the draft's context row, from which
// every layer's ring keys and values are computed: 4-bit codes over groups of
// SPLASH_DRAFT_CONTEXT_GROUP values, all slots' codes first, then each
// group's fp16 scale and minimum. Its kernels run one simdgroup per row and
// group, SPLASH_DRAFT_CONTEXT_SIMDGROUPS to a threadgroup.
#define SPLASH_DRAFT_CONTEXT_GROUP 64u
#define SPLASH_DRAFT_CONTEXT_SIMDGROUPS 8u
inline constexpr uint64_t draft_context_codes_bytes(uint32_t width) {
  return uint64_t(SPLASH_DRAFT_SLIDING_WINDOW) * (width / 2);
}
inline constexpr uint64_t draft_context_window_bytes(uint32_t width) {
  return draft_context_codes_bytes(width) +
         uint64_t(SPLASH_DRAFT_SLIDING_WINDOW) * (width / SPLASH_DRAFT_CONTEXT_GROUP) * 4;
}

struct DraftContextWindowParams {
  uint32_t rows;
  uint32_t start_position;
  uint32_t width;
};

static_assert(sizeof(DraftContextWindowParams) == 12,
              "Draft context window parameters are 12 bytes on both sides");
// ---- end runtime/metal/abi/DraftAttention.h ----
// ---- begin runtime/metal/abi/Embedding.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct DraftMaskRowsParams {
  uint32_t width;
};

static_assert(sizeof(DraftMaskRowsParams) == 4,
              "Draft mask row parameters are 4 bytes on both sides");
// ---- end runtime/metal/abi/Embedding.h ----
// ---- begin runtime/metal/abi/GDN.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif


// The taps of the GDN short convolution: a channel's input at a token and at
// the SPLASH_GDN_CONVOLUTION_TAPS - 1 tokens before it, which a state cell
// carries as that many rows of every channel.
#define SPLASH_GDN_CONVOLUTION_TAPS 4u

// The packed width, the value heads and the packed row stride of the GDN
// kernels are constants of their compiled variant and the grids give the
// tasks; the state cells' strides and a decode step's tape offsets come from
// the host.

// The prefill prepare and scan kernels.
struct GDNPrefillParams {
  uint32_t tokens;
};

static_assert(sizeof(GDNPrefillParams) == 4,
              "GDN prefill parameters are 4 bytes on both sides");

// flags: GDN_TILED_HEADS selects the value-head order of the GDN output, the
// out_proj input columns: without it a key head's value heads stay adjacent
// (head h at h); with it, the tiled order of GGUFs puts head h at
// (h % heads per key) * key heads + h / heads per key. GDN_SIGMOID_GATE gates
// the normalized output by sigmoid(z) (Qwen3.8-Flash-Next) instead of silu(z).
#define GDN_TILED_HEADS 1u
#define GDN_SIGMOID_GATE 2u
struct GDNGatePrefillParams {
  uint32_t flags;
};

static_assert(sizeof(GDNGatePrefillParams) == 4,
              "GDN prefill gate parameters are 4 bytes on both sides");

// A decode step leaves its rows on the lane's tape: one layer's holds room
// for SPLASH_TARGET_VERIFY_ROWS rows and, each field for every row of the
// lane's block in turn, the rows' convolution inputs ([conv_dim] bf16), their
// k then v rows ([key_width + value_width] bf16), decay ([value_heads] fp32)
// and beta ([value_heads] bf16). The lane's next step folds the path of rows
// it accepted into the state before its own (GDN::addDecode).
inline constexpr uint64_t gdn_tape_kv_offset(uint32_t conv_dim) {
  return uint64_t(SPLASH_TARGET_VERIFY_ROWS) * conv_dim * 2;
}
inline constexpr uint64_t gdn_tape_decay_offset(uint32_t conv_dim,
                                                uint32_t kv_width) {
  return gdn_tape_kv_offset(conv_dim) +
         uint64_t(SPLASH_TARGET_VERIFY_ROWS) * kv_width * 2;
}
inline constexpr uint64_t gdn_tape_beta_offset(uint32_t conv_dim,
                                               uint32_t kv_width,
                                               uint32_t value_heads) {
  return gdn_tape_decay_offset(conv_dim, kv_width) +
         uint64_t(SPLASH_TARGET_VERIFY_ROWS) * value_heads * 4;
}
// One layer's tape, a multiple of 16 bytes.
inline constexpr uint64_t gdn_tape_layer_bytes(uint32_t conv_dim,
                                               uint32_t kv_width,
                                               uint32_t value_heads) {
  return (gdn_tape_beta_offset(conv_dim, kv_width, value_heads) +
          uint64_t(SPLASH_TARGET_VERIFY_ROWS) * value_heads * 2 + 15) /
         16 * 16;
}

struct GDNDecodeBatchParams {
  uint32_t flags; // As in GDNGatePrefillParams.
  uint32_t layer;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  // Per lane, byte offsets in the tape buffer of this layer's tape of the
  // lane's previous step (the pending rows) and of this step, and the path
  // of rows of the previous step the lane accepted and its state does not
  // hold yet: a row mask (metal/abi/VerifyTree.h), none after a prefill.
  uint64_t pending_tape[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint64_t step_tape[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t pending_path[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(GDNDecodeBatchParams) == 32 + 20 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "GDN decode parameters are the same size on both sides");

// A fold of the first `rows` rows of a lane's accepted path of its step alone
// (GDN::addFold): the state cells' strides, as in GDNDecodeBatchParams; the
// cached cell's recurrent layers' (codes_layer_bytes apart from its
// convolution_state_bytes, as ops::StateCodec stores them); and the byte
// offset in the tape buffer of the step's first layer's tape, which the next
// layers' follow gdn_tape_layer_bytes apart.
struct GDNFoldParams {
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  uint64_t codes_layer_bytes;
  uint64_t step_tape;
  uint32_t rows;
  uint32_t padding;
};

static_assert(sizeof(GDNFoldParams) == 48,
              "GDN fold parameters are 48 bytes on both sides");
// ---- end runtime/metal/abi/GDN.h ----
// ---- begin runtime/metal/abi/MoE.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif
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
// ---- begin runtime/metal/abi/ShortConvolution.h (Splash 2afc733, Apache-2.0) ----

// Parameters of the short convolution kernels (kernels/shared/short_convolution.metal,
// ops/ShortConvolution.cpp): a causal depthwise convolution of
// SHORT_CONVOLUTION_TAPS taps over each channel of a projection's output
// rows, in fp32. The taps are static, F32 [channels][taps], or static plus
// the row's own dynamic taps, bf16 [rows][heads][taps], which every channel
// of a head shares, each in the order a GGUF stores it, which the loader
// keeps: a channel's static taps in a conv1d weight's order, its last column
// weighing the row itself and column taps - 1 - k the input k rows back
// (llama.cpp's ggml_ssm_conv); a head's dynamic taps by that lag, k the input
// k rows back (the projection's output, index head * taps + k). The sum
// plus the row's input, over `divisor`, is rounded to bf16 (the source
// returns its input's type) before an output adds it to a residual
// (SHORT_CONVOLUTION_OUTPUT_*).
//
// A sequence's state is the SHORT_CONVOLUTION_TAPS - 1 input rows before its
// first, oldest first, in bf16 as the projection writes them: rows of a
// lane's state matrix of `state_stride` columns, of which a convolution owns
// `channels` from `state_column` (ops::ShortConvState), so that every
// convolution of a target binds the same per-lane buffers and one commit
// writes all of them.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif


// The taps of every short convolution of a model in the tree (its
// configurations' convolution kernel sizes).
#define SHORT_CONVOLUTION_TAPS 4u
#define SHORT_CONVOLUTION_STATE_ROWS (SHORT_CONVOLUTION_TAPS - 1u)

// The most states captured inside a command's sequences
// (ops::ShortConvCapture) that one dispatch writes; a command of more runs
// dispatches of the rest alone (captures_only).
#define SHORT_CONVOLUTION_CAPTURES 4u

// The outputs (ops::ShortConvOutput): bf16(y) as bf16 rows; bf16 of an fp32
// residual plus bf16(y), a layer's output (short_convolution); fp32 rows of a
// bf16 residual, or none, plus bf16(y) (short_convolution_sum).
#define SHORT_CONVOLUTION_OUTPUT_PLAIN 0u
#define SHORT_CONVOLUTION_OUTPUT_SUM 1u
#define SHORT_CONVOLUTION_OUTPUT_ROUNDED_SUM 2u

// The convolution of a command's `sequences` sequences of its `rows` rows
// (sequence s's rows its entry of the command's layout, SplashLaneRows,
// metal/abi/VerifyTree.h), as ngram_embedding_conv runs its own: with
// write_state the state after each sequence's rows goes to its state_out, and
// capture q (q < captures) the state after the first capture_rows[q] rows of
// sequence capture_sequences[q] to its own buffer; captures_only writes those
// alone. With `tree` the sequences are verify blocks of at most 32 rows whose
// rows' taps follow their paths, and each row's input goes to the tape (rows
// of state_stride columns, the convolution's from state_column), which the
// commit reads. heads: 0 for static taps, else the dynamic taps' heads, whose
// rows are dynamic_stride values apart (a projection's rows, whole tiles).
// residual: whether a sum output adds a residual (none adds zero). The input
// rows are input_stride values apart, the convolution's channels from column
// input_column of each (a view of a wider projection's rows). A pair
// (short_convolution_pair: a verify step's two convolutions of one shape in
// one dispatch, plain) runs a second convolution of its own buffers, its
// input's channels from column pair_input_column, its state's and the tape's
// from pair_state_column. The commit takes `channels` as the state matrix's
// columns, every one of which it writes; with captures (one) it is a decode
// capture's fold instead, of the first capture_rows[0] rows of each lane's
// accepted path alone.
struct ShortConvolutionParams {
  uint32_t rows;
  uint32_t channels;
  uint32_t heads;
  uint32_t output;
  uint32_t residual;
  uint32_t write_state;
  uint32_t tree;
  uint32_t sequences;
  uint32_t state_stride;
  uint32_t state_column;
  float divisor;
  uint32_t captures;
  uint32_t captures_only;
  uint32_t dynamic_stride;
  uint32_t capture_sequences[SHORT_CONVOLUTION_CAPTURES];
  uint32_t capture_rows[SHORT_CONVOLUTION_CAPTURES];
  uint32_t input_stride;
  uint32_t input_column;
  uint32_t pair_input_column;
  uint32_t pair_state_column;
};

static_assert(sizeof(ShortConvolutionParams) == 104,
              "short convolution parameters are 104 bytes on both sides");
// ---- end runtime/metal/abi/ShortConvolution.h ----
// ---- begin runtime/metal/abi/VerifyTree.h (Splash 2afc733, Apache-2.0) ----

// The rows of a decode lane's verify block as a tree, shared by host dispatch
// code and Metal kernels. One entry per row of a command's packed rows (lane
// l's rows from its first row, ops::LaneRows): row 0 of a block holds the
// lane's anchor, and every other row a proposal that follows its parent row.
// Rows are in topological order (a parent comes before its children), so the
// set bits of a row's ancestors, ascending, are its path from the anchor. A
// chain is the tree whose row r follows row r - 1.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The anchor's parent.
#define SPLASH_TREE_ROOT 0xffffu

// ancestors: bit j for each row j of the block on the row's path from the
// anchor, the row itself included; parent: its parent row in the block;
// depth: its steps from the anchor, so its position is the block's first
// position plus its depth.
struct SplashTreeRow {
  uint32_t ancestors;
  uint16_t parent;
  uint16_t depth;
};

static_assert(sizeof(SplashTreeRow) == 8,
              "Verify tree rows are 8 bytes on both sides");

// A row's K/V and its GDN tape row sit at its own row of the block (the
// block's first position plus the row), so that siblings never collide; its
// RoPE position is the block's first position plus its depth. A block's
// accepted path is a row mask of the same form as `ancestors`: the ancestors
// of the deepest row it accepted.

// A lane's rows of a command's packed rows (ops::LaneRows): `rows` rows from
// row `first`, as a step's proposer writes them on the device for the
// kernels after it.
struct SplashLaneRows {
  uint32_t first;
  uint32_t rows;
};

static_assert(sizeof(SplashLaneRows) == 8,
              "Lane rows are 8 bytes on both sides");
// ---- end runtime/metal/abi/VerifyTree.h ----

// The expert slots of the routing kernels (ops::MoE): a row's router scores
// take this many floats, the block's experts first, and the select and
// grouping kernels run one thread per slot. A MoE block of at most
// SPLASH_MOE_EXPERT_SLOTS experts routes on their instances of that many
// slots, one of more on the softmax routing's wide instances (_e512)
// (ops::kCompiledMoeRoutings), so a block routes at most
// SPLASH_MOE_WIDE_EXPERT_SLOTS experts.
#define SPLASH_MOE_EXPERT_SLOTS 256u
#define SPLASH_MOE_WIDE_EXPERT_SLOTS 512u

struct MoeRouteParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t experts;
  uint32_t top_k;
};

static_assert(sizeof(MoeRouteParams) == 16,
              "MoE routing parameters are 16 bytes on both sides");

// A biased selection's routing (ops::MoeShape::selectionBias): the scale of
// its routing weights beside the routing's parameters.
struct MoeBiasedRouteParams {
  MoeRouteParams route;
  float scale;
};

static_assert(sizeof(MoeBiasedRouteParams) == 20,
              "biased MoE routing parameters are 20 bytes on both sides");

// Every row carries top_k routed experts followed by the slices of the shared
// expert, ids `experts` on, whose routing weight is the sigmoid of the
// block's scalar gate or 1 (ops::MoeShape). Routes are sorted by expert into
// tiles of at most tile_rows routes of one expert: a tile covers grouped rows
// [first_row, first_row + rows), each tile's following the tile before it's,
// a grouped row per route.
//
// Written by grouping kernels; the host uses this layout to size storage.
struct MoeTileDescriptor {
  uint32_t expert;
  uint32_t rows;
  uint32_t first_row;
};

static_assert(sizeof(MoeTileDescriptor) == 12,
              "MoE tile descriptors are 12 bytes on both sides");

// The staged GGUF expert passes (kernels/shared/moe_gguf.metal) run an
// expert's only tile of at most GGUF_COLUMN_ROWS live rows on the column tile
// (metal/abi/Gguf.h), which reads those rows alone, and every other tile on
// the staged tile's matmuls (moe_column_tile): a route's output depends on its
// expert's rows in the step.

// Whether the staged expert passes run tile t of `tiles`, the grouping's
// MoeTileDescriptors (on the GPU in device memory), on the column tile: at
// most GGUF_COLUMN_ROWS live rows, its expert's only tile. The kernels,
// the MoE test's expected tile kinds and the MoE benchmark's share of column
// tiles all ask this.
template <typename Tiles>
inline bool moe_column_tile(Tiles tiles, uint32_t t) {
  return tiles[t].rows <= GGUF_COLUMN_ROWS && (t == 0 || tiles[t - 1].expert != tiles[t].expert);
}

// The column tiles of the gate/up and down passes, whose grids the grouping
// writes (MoeTileCount). A verified grouping (moe_group_routes_verified*) of a
// decode step of `lanes` lanes routes only its lanes' rows, lane_rows[l] as
// the draft selector laid them out on the device
// (ops::DraftSelector::addVerifyRows), which its lanes verify: the routed
// routes of a row of no lane stay out of every tile with the route row ~0u,
// which the combine skips, so that its hidden state, which nothing reads,
// costs no expert's work, nor on a streamed store an expert's load. The other
// groupings route every row. Only a streamed grouping, of a step whose passes
// run around its load, reads the MoE block's slot table (MoeTileCount) and
// copies it for the passes beside the load.
struct MoeGroupParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t tile_rows;
  uint32_t experts;
  uint32_t gate_up_columns;
  uint32_t down_columns;
  uint32_t streamed;
  uint32_t lanes;
};

static_assert(sizeof(MoeGroupParams) == 32,
              "MoE grouping parameters are 32 bytes on both sides");

// What the grouping writes besides the tiles: their count, which every pass
// over them reads first, the tiles of experts resident before the step's
// load, and the grids (column tiles, tiles, 1) of the gate/up and down
// passes over all of them, over the resident ones and over the others
// (MOE_TILES_*), which they read as an indirect dispatch
// (MTLDispatchThreadgroupsIndirectArguments) to launch the live tiles alone.
// Tiles [0, resident) are the resident experts' and then the shared slices',
// and tiles [resident, tiles) those of the experts whose slot-table entry is
// MOE_SLOT_ABSENT, which a load brings.
struct MoeTileCount {
  uint32_t tiles;
  uint32_t resident;
  uint32_t gate_up_grid[3];
  uint32_t down_grid[3];
  uint32_t resident_gate_up_grid[3];
  uint32_t resident_down_grid[3];
  uint32_t loaded_gate_up_grid[3];
  uint32_t loaded_down_grid[3];
};

static_assert(sizeof(MoeTileCount) == 80,
              "the MoE tile count is 80 bytes on both sides");

// The tiles an expert pass runs (MoeTileCount): all of them, the resident
// experts' and shared slices', or the others; tile moe_first_tile + y runs at
// y of the pass's grid, below moe_end_tile.
#define MOE_TILES_ALL 0u
#define MOE_TILES_RESIDENT 1u
#define MOE_TILES_LOADED 2u
// A branch, not a select: the other ranges' tile loads do not wait for a
// load of the count.
template <typename Count>
inline uint32_t moe_first_tile(Count count, uint32_t tiles) {
  uint32_t first = 0;
  if (tiles == MOE_TILES_LOADED) first = count[1];
  return first;
}
template <typename Count>
inline uint32_t moe_end_tile(Count count, uint32_t tiles) {
  return count[tiles == MOE_TILES_RESIDENT ? 1 : 0];
}

// A slot-table entry (ops::BlockMoeWeights::slotTable) of an expert no slot
// holds, which only the grouping reads.
#define MOE_SLOT_ABSENT 0x80000000u

struct MoeGatherParams {
  uint32_t input_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeGatherParams) == 8,
              "MoE gather parameters are 8 bytes on both sides");

// The routed experts of a MoE block sit in slots, which chunks hold: the
// floor chunk, chunk 0, holds the first floor_slots slots, a store size's
// floor, and each elastic chunk after it chunk_slots slots (the last may hold
// fewer). Slot s < floor_slots is local slot s of chunk 0; any other is local
// slot e - (c - 1) * chunk_slots of chunk c = 1 + e / chunk_slots, where
// e = s - floor_slots, and slots is the block's slot table
// (ops::BlockMoeWeights). A resident block, and a store size without an
// elastic chunk, is its floor chunk alone, of floor_slots = chunk_slots = its
// slots. The expert kernels reach the chunks through a table of their GPU
// addresses (metal::AddressTable), in the constant address space: chunk c's
// MOE_EXPERT_CHUNK_ADDRESSES entries from c * MOE_EXPERT_CHUNK_ADDRESSES on
// are the gate's (z = 0), up's (z = 1) and down's (z = 2) plane0, plane1 and
// meta regions, at z * 3 + 0, 1 and 2; a format without plane1 has its meta
// region's address in plane1's entry. A region holds its chunk's slots in
// order, each the projection's rows, so local slot l starts at row l * rows
// of it. A table holds at most MOE_EXPERT_MAX_CHUNKS chunks, which setBytes
// passes (4 KiB).
#define MOE_EXPERT_CHUNK_ADDRESSES 9u
#define MOE_EXPERT_MAX_CHUNKS 48u

// A GGUF expert pass of one projection (the down pass,
// moe_expert_gguf_m<rows>_down; kernels/common/moe_gguf_expert_tile.h):
// routed expert e's planes start at row local * output_size of the
// projection's regions in the chunk that holds slot slots[e] (local is the
// slot's place in it); the shared expert's slices (ids `experts` on) have a
// segment of their own, slice s's planes from tile
// s * output_size / QUANT_TILE_ROWS on. Formats are GGUF_FMT_*
// (metal/abi/QuantFormat.h).
struct MoeGgufExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
  uint32_t floor_slots;
  uint32_t chunk_slots;
  uint32_t tiles; // MOE_TILES_*
};

static_assert(sizeof(MoeGgufExpertParams) == 32,
              "MoE GGUF expert parameters are 32 bytes on both sides");

// An int8 expert pass of a prefill chunk (kernels/shared/moe_int8.metal,
// ops/MoE.cpp): the grouped rows' int8 codes and scales (the dense prefill's
// rule, per row and 64 inputs, kernels/common/int8_activations.h), a row per
// route, the scales' row stride, through the af4g64 or af5g64 planes of the
// tile's expert as MoeGgufExpertParams reaches them.
struct MoeInt8ExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
  uint32_t floor_slots;
  uint32_t chunk_slots;
  uint32_t tiles; // MOE_TILES_*
  uint32_t routes;
};

static_assert(sizeof(MoeInt8ExpertParams) == 36,
              "MoE int8 expert parameters are 36 bytes on both sides");

// moe_gather_int8: the grouped rows' codes and scales from the chunk's
// (rows of input_size codes, scales [input_size / 64][rows]).
struct MoeGatherInt8Params {
  uint32_t input_size;
  uint32_t routes_per_row;
  uint32_t rows;          // the chunk's, its scales' row stride
  uint32_t routes;        // the grouped rows, their scales' row stride
};

static_assert(sizeof(MoeGatherInt8Params) == 16,
              "MoE int8 gather parameters are 16 bytes on both sides");

// The gate and up passes in one (moe_expert_gguf_m<rows>_gate_up[_mixed]):
// their sizes, expert count, floor and chunk slots, as MoeGgufExpertParams
// holds them, and each pass's routed and shared formats. Gate reads
// projection 0 and up projection 1 of the same chunk.
struct MoeGgufGateUpParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t gate_routed_format;
  uint32_t gate_shared_format;
  uint32_t up_routed_format;
  uint32_t up_shared_format;
  uint32_t floor_slots;
  uint32_t chunk_slots;
  uint32_t tiles; // MOE_TILES_*
};

static_assert(sizeof(MoeGgufGateUpParams) == 40,
              "MoE GGUF gate/up parameters are 40 bytes on both sides");

// One projection (gate, up or down) of the experts a load writes into their
// slots (moe_expert_load, model/ExpertStore.cpp): rows x columns in `format`,
// job j's rows staged from wherever the job's sources name, and its plane0,
// plane1 and meta planes slot_bytes[i] bytes per slot in the regions of the
// slot's chunk. The rows are staged as native rows of row_bytes each, or,
// where stored_bytes[0] is not 0, as the planes an MLX affine or mxfp4
// tensor stores them in, stored_bytes[p] bytes a row of plane p: its codes,
// scales and biases (model/GgufPreparation.hpp storedPlanes).
struct MoeExpertLoadProjection {
  uint32_t format;
  uint32_t rows;
  uint32_t columns;
  uint32_t row_bytes;
  uint32_t stored_bytes[3];
  uint32_t reserved;
  uint64_t slot_bytes[3];
};

static_assert(sizeof(MoeExpertLoadProjection) == 56,
              "MoE expert load projections are 56 bytes on both sides");

// The gate, up and down projections of one load, the slots of the floor
// chunk and of each elastic chunk, and the staged planes a job's sources
// name of each projection (1, or 3 where a projection is staged as stored
// planes): source s of job j's projection z is sources[(3 j + z) *
// source_planes + s].
struct MoeExpertLoadParams {
  struct MoeExpertLoadProjection projections[3];
  uint32_t floor_slots;
  uint32_t chunk_slots;
  uint32_t source_planes;
  uint32_t reserved;
};

static_assert(sizeof(MoeExpertLoadParams) == 184,
              "MoE expert load parameters are 184 bytes on both sides");

// The threads of a moe_expert_load threadgroup, and the 32-wide K groups of a
// row it repacks: its QUANT_TILE_ROWS rows' pairs of a span of
// MOE_EXPERT_LOAD_SPAN groups, the threads of a row taking every
// (MOE_EXPERT_LOAD_THREADS / QUANT_TILE_ROWS)-th.
#define MOE_EXPERT_LOAD_THREADS 256u
#define MOE_EXPERT_LOAD_SPAN 32u

struct MoeCombineParams {
  uint32_t rows;
  uint32_t hidden_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeCombineParams) == 12,
              "MoE combine parameters are 12 bytes on both sides");

// A verify step's combine with the short convolution of its rows
// (moe_combine_convolution, ops::MoeCombineConvolution): the combine's, and
// the convolution's of static taps (heads 0) along the lanes' trees (tree 1)
// onto an fp32 residual (residual 1, SHORT_CONVOLUTION_OUTPUT_ROUNDED_SUM),
// its sequences the step's lanes. A threadgroup runs a lane's rows, at most
// MOE_COMBINE_CONVOLUTION_ROWS (a verify block's), of
// MOE_COMBINE_CONVOLUTION_CHANNELS channels, a thread a row's channel: the
// rows its convolution taps are the lane's own, which it combined, or its
// state's.
#define MOE_COMBINE_CONVOLUTION_ROWS 32u
#define MOE_COMBINE_CONVOLUTION_CHANNELS 32u
struct MoeCombineConvolutionParams {
  struct MoeCombineParams combine;
  struct ShortConvolutionParams convolution;
};

static_assert(sizeof(MoeCombineConvolutionParams) == 116,
              "MoE combine convolution parameters are 116 bytes on both sides");
// ---- end runtime/metal/abi/MoE.h ----
// ---- begin runtime/metal/abi/PagedAttention.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// ---- begin runtime/metal/abi/KvExtent.h (Splash 2afc733, Apache-2.0) ----

// Where KV pages live, shared by the host and the kernels. A pool's pages sit
// in extents: ordinary shared Metal buffers that all hold the same number of
// pages, allocated when the pool grows; an empty one is released by the next
// reclaim pass between commands. With a block index, an extent holds its
// pages' index after every layer's K/V region, as its index plane, and an
// extent of pages that hold only their index (kv::PageForm::Index) holds only
// that plane. Kernels reach extents only through the page entries of a
// request's tables, never through a bound buffer: a page table's entries reach
// the pages' K/V, an index table's their index plane.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif


// One KV page as kernels address it: the GPU address of the extent that holds
// it, or of its index plane, 16 KiB-aligned, with the page's index in the
// extent in the low bits.
typedef uint64_t SplashKvPage;
#define SPLASH_KV_PAGE_INDEX_BITS 14u
#define SPLASH_KV_PAGE_INDEX_MASK ((1u << SPLASH_KV_PAGE_INDEX_BITS) - 1u)
// The K/V entry of a page that holds only its blocks' index (kv::PageForm::
// Index): no K/V to reach. A kernel that reads through it faults, loudly, on
// the null address; the target's dense-history rule keeps every read that
// goes through a page table off such pages.
#define SPLASH_KV_PAGE_ABSENT ((SplashKvPage)0)

// The dimension of the KV heads of Qwen3.5's pages, the default of the
// element functions below. A layout holds its own (kv::Layout::
// headDimension); a kernel takes its own as a template value, HeadDim
// (kernels/common/kv_paging.h).
#define SPLASH_KV_HEAD_DIMENSION 256u

// The tensors of a layer, in the order they sit in its region of an extent.
// BF16 has no scales: their bytes are zero.
#define SPLASH_KV_KEYS 0u
#define SPLASH_KV_KEY_SCALES 1u
#define SPLASH_KV_VALUES 2u
#define SPLASH_KV_VALUE_SCALES 3u

// The bytes one page holds of one tensor of one layer: data_bytes for keys
// and values, scale_bytes for their scales.
inline uint32_t splash_kv_page_bytes(uint32_t data_bytes, uint32_t scale_bytes,
                                     uint32_t tensor) {
  return tensor % 2 ? scale_bytes : data_bytes;
}

// Where one element of one KV head sits in a page's slab of a tensor, in
// elements, for `token` of the page's SPLASH_TARGET_KV_BLOCK_TOKENS, in heads
// of head_dimension dimensions: keys token-major, values dimension-major, and
// one scale per (head, token) for either of them.
inline uint64_t splash_kv_key_element(uint32_t head, uint32_t token,
                                      uint32_t dimension,
                                      uint32_t head_dimension = SPLASH_KV_HEAD_DIMENSION) {
  return (uint64_t(head) * SPLASH_TARGET_KV_BLOCK_TOKENS + token) *
             head_dimension +
         dimension;
}

inline uint64_t splash_kv_value_element(uint32_t head, uint32_t token,
                                        uint32_t dimension,
                                        uint32_t head_dimension = SPLASH_KV_HEAD_DIMENSION) {
  return (uint64_t(head) * head_dimension + dimension) *
             SPLASH_TARGET_KV_BLOCK_TOKENS +
         token;
}

inline uint64_t splash_kv_scale_element(uint32_t head, uint32_t token) {
  return uint64_t(head) * SPLASH_TARGET_KV_BLOCK_TOKENS + token;
}

// Where one tensor of one page of one layer sits in its extent, in bytes.
// Every extent of a pool holds extent_pages pages. Per attention layer a
// region holds the keys of every page, then their key scales, values and
// value scales.
inline uint64_t splash_kv_offset(uint32_t extent_pages, uint32_t data_bytes,
                                 uint32_t scale_bytes, uint32_t layer,
                                 uint32_t tensor, uint32_t index) {
  const uint64_t region =
      uint64_t(layer) * extent_pages * 2 * (uint64_t(data_bytes) + scale_bytes);
  const uint64_t before =
      (tensor + 1) / 2 * uint64_t(data_bytes) + tensor / 2 * uint64_t(scale_bytes);
  return region + extent_pages * before +
         uint64_t(index) * splash_kv_page_bytes(data_bytes, scale_bytes, tensor);
}

// Where one attention layer sits in every extent of a pool: its region
// begins `offset` bytes into each extent.
struct SplashKvLayer {
  uint32_t extent_pages;
  uint32_t offset;
};

static_assert(sizeof(SplashKvLayer) == 8, "KV layer placement is 8 bytes on both sides");

// The placement of `layer` in a pool of extents of extent_pages pages.
inline SplashKvLayer splash_kv_layer(uint32_t extent_pages, uint32_t data_bytes,
                                     uint32_t scale_bytes, uint32_t layer) {
  return {extent_pages, uint32_t(splash_kv_offset(extent_pages, data_bytes, scale_bytes,
                                                  layer, SPLASH_KV_KEYS, 0))};
}

// The entry of the page at `index` of the extent whose GPU address is
// extent_address.
inline SplashKvPage splash_kv_page_entry(uint64_t extent_address, uint32_t index) {
  return extent_address | index;
}
// ---- end runtime/metal/abi/KvExtent.h ----

// One full-attention layer's paged KV. Every current row is written directly
// into a page slot before attention, and every row reads its visible history
// from the same paged representation. A decode step's rows sit at the slots
// of their rows of the lane's block; acceptance moves the accepted path's rows
// to the slots of their depths (SplashKvPathCommitParams), nothing for a
// chain's, and advances committed_tokens; the next command overwrites the
// slots past them.
// Page tables hold one SplashKvPage per logical page, and kv places the layer
// in the pool's extents; the host sets it for each layer it encodes.

// The parts a dense paged-attention tile of a GQA group of `group` query heads
// per KV head runs in (kernels/common/paged_attention_tile.h
// splash_attention_tile_part): the fewest that divide its eight rows and leave
// each at most 64 fused rows, the 256-thread tile's. A group of up to eight
// is one part, sixteen two, thirty-two four. Two 64-row parts of a group of
// sixteen run long prefill and verify 20-40% faster than one 128-row tile of
// 512 threads: both parts read the same pages, the second mostly from cache,
// and the smaller threadgroups keep more of the GPU busy. More parts are
// slower again at long histories.
static_assert(SPLASH_ATTENTION_TILE_ROWS == 8, "a tile's parts divide its eight rows");
#define SPLASH_ATTENTION_TILE_PARTS(group) \
  ((group) <= 8u ? 1u : (group) <= 16u ? 2u : (group) <= 32u ? 4u : 8u)

// Sparse attention (kernels/shared/sparse_attention.metal,
// ops::AttentionBlocks): a query row attends only to the blocks of
// SPLASH_ATTENTION_BLOCK_TOKENS tokens its block mask names, within its causal
// history; the mask is per row a bitmap over the history's blocks, block b at
// bit b % 32 of word b / 32. A tile's rows' masks become its block list: in
// ascending order each block of the tile's history that some row attends, as
// block << SPLASH_ATTENTION_BLOCK_ROW_BITS | a bit per row of the tile that
// attends it, in lists of block_stride entries with their counts apart. A
// sequence's sparse tile t reads list block_list + t; without a stride a tile
// attends to every block of its history.
#define SPLASH_ATTENTION_BLOCK_TOKENS 4u
#define SPLASH_ATTENTION_BLOCK_ROW_BITS SPLASH_ATTENTION_TILE_ROWS
// A sparse split takes at least SPLASH_SPARSE_SPLIT_BLOCKS entries of its
// tile's list, and a list spreads over at least SPLASH_SPARSE_MINIMUM_SPLITS
// splits while it gives each a partition unit, so that a lane's split count
// follows its own list, not its history.
#define SPLASH_SPARSE_SPLIT_BLOCKS 48u
#define SPLASH_SPARSE_MINIMUM_SPLITS 32u

static_assert(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS / SPLASH_ATTENTION_BLOCK_TOKENS <
                  1u << (32 - SPLASH_ATTENTION_BLOCK_ROW_BITS),
              "A list entry's bits above its rows' hold any block");

// A list's blocks may sit outside the tile's page table (ops::AttentionBlocks
// locations, the blocks a spilled request's hot block cache holds), which the
// located split reads: a location per entry, in an array of the lists'
// shape, names a page of the locations' own page table (location >>
// SPLASH_ATTENTION_LOCATION_BLOCK_BITS) and the block's place in it (the low
// bits), or is SPLASH_ATTENTION_PAGED for a block whose page the page table
// holds, at its own place. The locations' pages hold the layer where the
// dispatch's `kv` places it, as the pool's do.
#define SPLASH_ATTENTION_PAGED 0xffffffffu
#define SPLASH_ATTENTION_LOCATION_BLOCK_BITS 3u

static_assert(SPLASH_TARGET_KV_BLOCK_TOKENS / SPLASH_ATTENTION_BLOCK_TOKENS ==
                  1u << SPLASH_ATTENTION_LOCATION_BLOCK_BITS,
              "A location's low bits hold a block's place in its page");

// A command's rows (ops::AttentionRows): `rows` packed rows, sequence s's
// rows entry s of the command's layout (SplashLaneRows: its first row and its
// rows), which a prompt's host writes and a decode step's selector writes on
// the device. Every attention kernel reads a sequence's rows there, and an
// entry past its sequence's row bound or the command's rows holds none, so
// that its sequence runs nothing, as the host checks a layout it writes.

// The block lists of up to SPLASH_MAXIMUM_BATCH_WIDTH sequences of a
// command's `rows` rows, in tiles of `tile_rows` rows: sequence s's rows,
// entry `sequence` of the command's layout, at most row_bound of them, whose
// block masks are the masks' rows of its packed rows, `mask_words` words
// each, the first at position `first_position`, its last tile holding the
// rest; its tile t's entries go to list `list` + t, `stride` entries each,
// and a tile of its bound past its rows lists nothing. A threadgroup of
// SPLASH_ATTENTION_BLOCK_LIST_THREADS threads lists one tile.
#define SPLASH_ATTENTION_BLOCK_LIST_THREADS 256u
struct SplashAttentionBlockListSequence {
  uint32_t sequence;
  uint32_t row_bound;
  uint32_t first_position;
  uint32_t list;
};

struct SplashAttentionBlockListParams {
  uint32_t tile_rows;
  uint32_t mask_words;
  uint32_t stride;
  uint32_t sequences;
  uint32_t rows;
  SplashAttentionBlockListSequence sequence[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SplashAttentionBlockListParams) == 20 + 16 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Block list parameters are the same size on both sides");

// One sequence of an attention layer's dispatches (ops::PagedAttention): at
// most row_bound rows, at positions from committed_tokens, its entry of the
// command's layout from row layout_row of it on (0, or the first row of its
// last tile, ops::PagedAttention::lastTiles); staged from row stage_row on of
// each staging, a KV head's rows chunk_stride apart (queries and attention
// rows [KV head][row][query head in group][dimension], keys [KV head][row]
// [dimension], values [KV head][dimension][row]), and reading its history
// through the dispatch's page table `s` (its place among the dispatch's
// sequences), page_table_entries long. Its bound's tiles of
// SPLASH_ATTENTION_TILE_ROWS rows are the dispatch's from first_tile on, and
// a tile past its rows writes nothing; each splits its pages into
// split_count balanced runs, whose partials take the slots from first_slot
// (in a tile's parts) on, and a sparse one, tile t of the sequence, attends
// through list block_list + t of block_stride entries. Split and reduce use
// the same count and partition rule. The bound sizes everything the host
// lays out for a sequence (its staging, page table, tiles, partial slots and
// lists), so that rows the device chooses within it need nothing else.
struct SplashAttentionSequence {
  uint32_t committed_tokens;
  uint32_t row_bound;
  uint32_t layout_row;
  uint32_t stage_row;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t split_count;
  uint32_t first_tile;
  uint32_t first_slot;
  uint32_t block_stride;
  uint32_t block_list;
};

static_assert(sizeof(SplashAttentionSequence) == 44,
              "Attention sequences are 44 bytes on both sides");

// A dispatch's sequences of a command of `rows` packed rows, in tile order,
// the command's from first_sequence on (its sequence s is entry
// first_sequence + s of the command's layout): `tiles`, their tiles' sum,
// sizes the split and reduce grids, and the per-row kernels (q/k projection,
// store, gate) take a grid of (head or element group, row in sequence,
// sequence), each threadgroup reading its sequence's layout entry once. A
// threadgroup of a row past its sequence's rows returns. Their pages hold
// the layer where `kv` places it in their
// extents, the same for every sequence of a dispatch, which the GPU reads once
// for the whole dispatch: with a placement per sequence, the prompt tiles of
// two KV heads ran up to 7% slower (M5 Max).
// With `trees` nonzero each sequence is a decode lane's block of rows from
// its anchor, at committed_tokens, a tree (metal/abi/VerifyTree.h) whose
// row r's place is the tree rows' entry of its packed row: its K/V sit at the
// slot of its row, as a chain's, and its position is the anchor's plus its
// depth, so it attends the history of that position's window and, of the
// block's rows, its ancestors and itself; a tree's tiles read their pages
// from the anchor's window. Without, rows attend causally.
// The reduce writes each row as the out-projection's input row: with
// `last_rows` nonzero one row a sequence, its last (ops::AttentionOutput::
// Last), the command's sequence first_sequence + s's as output row
// first_sequence + s; without, every row as its packed row.
struct SplashAttentionParams {
  uint32_t sequences;
  uint32_t first_sequence;
  uint32_t rows;
  uint32_t tiles;
  uint32_t trees;
  uint32_t last_rows;
  SplashKvLayer kv;
  SplashAttentionSequence sequence[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SplashAttentionParams) == 32 + 44 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Attention parameters are the same size on both sides");

// The commit of each lane's accepted path after a decode step
// (ops::PagedAttention::addPathCommit): a step stores a lane's rows at the
// slots of their rows of its block, committed_tokens[lane] plus the row, so
// that siblings never collide, and the path's row at depth k then moves to
// the slot of its depth, committed_tokens[lane] plus k, in each of the
// pool's `regions` layer regions in extents of extent_pages pages.
struct SplashKvPathCommitParams {
  uint32_t extent_pages;
  uint32_t regions;
  uint32_t committed_tokens[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SplashKvPathCommitParams) == 8 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Path commit parameters are the same size on both sides");

// A prefill whose history has pages that hold only their index reads one
// layer's history staged (kernels/shared/kv_stage.metal, ops::PagedAttention::
// addStageLayer): pages of the pool's layer from first_page on, a threadgroup
// each, reached through a page table, copied to the same pages of a staging
// table, whose storage holds that one layer; a pool entry that is
// SPLASH_KV_PAGE_ABSENT is left to the reads that fill it. data_bytes and
// scale_bytes are a page's bytes of one tensor of the layer (zero scale bytes
// for BF16).
struct SplashKvStageParams {
  SplashKvLayer pool;
  SplashKvLayer staged;
  uint32_t first_page;
  uint32_t data_bytes;
  uint32_t scale_bytes;
};

static_assert(sizeof(SplashKvStageParams) == 28,
              "KV stage parameters are 28 bytes on both sides");
// ---- end runtime/metal/abi/PagedAttention.h ----
// ---- begin runtime/metal/abi/RoPE.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Rotary pairs in a row of the RoPE tables (ops::RoPE), which the attention
// kernels read: Qwen3.5's target rotates 64 of its 256 head dimensions (its
// partial rotary factor 1/4), the draft its whole head. A target attention
// geometry that rotates other pairs reads tables of its own pairs
// (ops::RoPE::addTables, ops::PagedAttention's compiled geometries).
#define SPLASH_TARGET_ROPE_PAIRS 32u
#define SPLASH_DRAFT_ROPE_PAIRS (SPLASH_DRAFT_HEAD_DIMENSION / 2u)

struct RopeTableParams {
  uint32_t target_rows;
  uint32_t draft_rows;
  // What the target table's cosines and sines are scaled by: YaRN's
  // attention factor, 1 without YaRN (model::RopeScaling).
  float target_scale;
};

static_assert(sizeof(RopeTableParams) == 12,
              "RoPE table parameters are 12 bytes on both sides");

// Draft rows of consecutive positions from start_position, the rows a
// restore computes its rings from again (ops::RoPE::addDraftRangeTables).
struct RopeRangeParams {
  uint32_t rows;
  uint32_t start_position;
};

static_assert(sizeof(RopeRangeParams) == 8,
              "RoPE range parameters are 8 bytes on both sides");

// The target rows of a decode step's `rows` packed rows over `lanes` lanes
// (ops::RoPE::addTreeTables), which its selector lays out on the device: lane
// l's rows (its SplashLaneRows) each at the RoPE position anchors[l], its
// anchor's, plus the row's depth in the lane's tree (its SplashTreeRow), the
// table scaled by target_scale as RopeTableParams' is.
struct RopeTreeParams {
  uint32_t lanes;
  uint32_t rows;
  float target_scale;
  uint32_t anchors[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(RopeTreeParams) == 12 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "RoPE tree parameters are the same size on both sides");
// ---- end runtime/metal/abi/RoPE.h ----
// ---- begin runtime/metal/abi/RowCopy.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// `rows` rows of `width` bf16 values, from row source_row of the source's
// rows of source_stride values, at source_column, to the destination's
// alike (ops::RowCopy).
struct RowCopyParams {
  uint32_t rows;
  uint32_t width;
  uint32_t source_row;
  uint32_t source_stride;
  uint32_t source_column;
  uint32_t destination_row;
  uint32_t destination_stride;
  uint32_t destination_column;
};

static_assert(sizeof(RowCopyParams) == 32,
              "Row copy parameters are 32 bytes on both sides");
// ---- end runtime/metal/abi/RowCopy.h ----
// ---- begin runtime/metal/abi/StateCodec.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A cached state stores a GDN layer's recurrent state as int8 codes, each
// value row's values in blocks of this many that share one bf16 scale, the
// block's largest magnitude over 127 (ops::StateCodec). Restoring 40 cached
// states this way changed the 27B's next-token distributions by a mean KL of
// 1.1e-4, below the 1.5e-4 a cold run of the same prompts differs by; a
// scale per value row of 128 had heavier tails on the 35B (KL max 0.15 nats
// against 0.07).
#define SPLASH_STATE_CODEC_BLOCK 32u
// The values one thread of the codec kernels holds: half a block, as one
// lane of the prefill scan holds sixteen key columns of its row, so a
// block's two threads are neighbours in their simdgroup.
#define SPLASH_STATE_CODEC_THREAD_VALUES 16u

// A cached layer of `rows` value rows of `columns` values: every value's
// code, row by row as the running state holds them, then each block's
// scale, in the same order.
inline constexpr uint64_t state_codec_scales_offset(uint32_t rows, uint32_t columns) {
  return uint64_t(rows) * columns;
}
inline constexpr uint64_t state_codec_layer_bytes(uint32_t rows, uint32_t columns) {
  return state_codec_scales_offset(rows, columns) + uint64_t(rows) * (columns / SPLASH_STATE_CODEC_BLOCK) * 2;
}

// One dispatch codes `layers` layers, the grid's y, each running layer
// state_layer_bytes and each cached one codes_layer_bytes after the last;
// threads_per_layer threads per layer, each holding
// SPLASH_STATE_CODEC_THREAD_VALUES consecutive values (ops::StateCodec).
struct StateCodecParams {
  uint64_t state_layer_bytes;
  uint64_t codes_layer_bytes;
  uint64_t scales_offset;
  uint32_t threads_per_layer;
  uint32_t padding;
};

static_assert(sizeof(StateCodecParams) == 32,
              "State codec parameters are 32 bytes on both sides");
// ---- end runtime/metal/abi/StateCodec.h ----
// ---- begin runtime/metal/abi/Sampling.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The uniforms one lane draws a decode cycle with, in [0, 1): row r of a
// lane's selection draws with its uniform r, the first token after a prompt
// (a selection of one row) with uniform 0, and a sampled verify row tests the
// children its draft drew (DraftTreeParams) with its uniforms
// SPLASH_TARGET_VERIFY_ROWS + r * SPLASH_TREE_CHILDREN on. Such a row selects
// by recursive rejection sampling without replacement: it tests its children
// in the order drawn, selecting child i, of probability q_i under the draft
// and p_i under the target, with probability min(1, p_i / q_i), where after
// each rejection the target's distribution becomes its residual
// max(p - q, 0), renormalized, and the draft's loses the rejected child,
// renormalized; one that selects none draws from the last residual. Its
// selection is so a draw from the target's distribution and a child it
// rejected only at fp32 rounding, and acceptance follows a row's child only
// when the child's token is that selection, so
// every token a lane keeps is a draw from the target; the rows of one path
// draw with uniforms of their own. A greedy row's argmax is the same rule at
// temperature 0, where the draft's children are its most likely candidates
// and every distribution falls on one token. Lane l's uniforms start at
// l * SPLASH_SAMPLING_UNIFORMS.
#define SPLASH_SAMPLING_UNIFORMS (SPLASH_TARGET_VERIFY_ROWS * (1u + SPLASH_TREE_CHILDREN))

// The stop tokens a target's selections and acceptance compare
// (ops::StopTokens): up to SPLASH_STOP_TOKENS ids, every place past them
// SPLASH_NO_TOKEN, which no vocabulary holds. It is no stop token: a row with
// no selectable logit selects it, and the runtime fails that lane
// (Runtime.cpp, invalidSelection) whatever its family's list.
#define SPLASH_STOP_TOKENS 4u
#define SPLASH_NO_TOKEN 0xFFFFFFFFu

// One target-policy dispatch over the selected rows of its lanes, a grid of a
// lane's rows by lanes: lane l selects every one of its lane rows
// (SplashLaneRows, metal/abi/VerifyTree.h), which a decode cycle's selector
// writes (VerifyRowsParams), at most SPLASH_TARGET_VERIFY_ROWS, and no row
// of no lane. Its row i, the logits row s = first + i, reads
// mask row l * (SPLASH_TARGET_VERIFY_ROWS + 1) + mask_row + i of the lane's
// SPLASH_TARGET_VERIFY_ROWS + 1 constraint-mask rows and draws with uniform
// l * SPLASH_SAMPLING_UNIFORMS + i; workspaces, vocabulary rows and output
// tokens are indexed by s.
struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t mask_row;
  uint32_t top_k[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  float top_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  float min_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  // Lanes that sample; the others take the argmax.
  uint32_t sampling_mask;
  uint32_t constrained_mask;
  // Lanes that ignore end-of-sequence: they never select a stop token.
  uint32_t exclude_stop_mask;
  uint32_t stop_tokens[SPLASH_STOP_TOKENS];
  // Whether the rows are a decode step's verify rows, whose sampled rows test
  // the children their draft drew (SPLASH_SAMPLING_UNIFORMS).
  uint32_t trees;
};

static_assert(sizeof(TargetSamplingParams) == 44 + 16 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Target sampling parameters are the same size on both sides");

// The selection of each lane's first token after its prompt from its one row
// of the logits, row[l]: decode_sample_first_rows writes the lane rows
// (TargetSamplingParams) and that row's tree row, a tree's root, that the
// selection's kernels read.
struct SampleFirstRowsParams {
  uint32_t row[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SampleFirstRowsParams) == 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "First-token row parameters are the same size on both sides");

// One shard's share of a sampled row's softmax denominator: the largest
// logit it admits, the sum of exp((logit - maximum) / temperature) over its
// admitted tokens, and how many it admits.
struct TargetShardMass {
  float maximum;
  float sum;
  uint32_t admitted;
};

static_assert(sizeof(TargetShardMass) == 12,
              "Target shard masses are 12 bytes on both sides");

// A sampled row's selection over the whole vocabulary. The search records
// the row's largest admitted logit and where its min-p/top-k/top-p
// distribution ends in the order of the logits (the key and id of its last
// token, metal/kernels/decode/sampling.metal); the token the draw takes from
// that distribution goes to the output tokens.
struct TargetVocabularyRow {
  float maximum;
  uint32_t end_key;
  uint32_t end_last;
};

static_assert(sizeof(TargetVocabularyRow) == 12,
              "Target vocabulary rows are 12 bytes on both sides");

// One range of the vocabulary in such a row's draw, which one simdgroup of
// the row's groups sums: the kept weight of its tokens, and one past the last
// of those with weight.
struct TargetVocabularyRange {
  float weight;
  uint32_t after;
};

static_assert(sizeof(TargetVocabularyRange) == 8,
              "Target vocabulary ranges are 8 bytes on both sides");

// A penalized request's word for each vocabulary token, in its state lane's
// row of the penalty table (ops::Sampling::rebuildPenaltyWords): the
// prompt bit marks a prompt token, and the count is how often the target
// selected it.
#define SPLASH_PENALTY_PROMPT_BIT 0x80000000u
#define SPLASH_PENALTY_COUNT_MASK 0x7fffffffu

// The penalized lanes of one penalty dispatch. Entry e penalizes the selected
// rows of the dispatch's lane lane[e] (TargetSamplingParams), from the
// penalty table row table_row[e]; the rows' verify input rows hold its draft
// tokens, of which a row's context adds its ancestors'
// (metal/abi/VerifyTree.h). repetition_inverse is 1 / repetition, saturated
// to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t entries;
  uint32_t lane[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[SPLASH_MAXIMUM_BATCH_WIDTH];
  float presence[SPLASH_MAXIMUM_BATCH_WIDTH];
  float frequency[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SamplingPenaltyParams) == 8 + 24 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Sampling penalty parameters are the same size on both sides");

// One selection's thinking cutoffs (ops::SamplingPolicy::thinkingCutoff) over
// its lanes' selected rows (TargetSamplingParams), each a row of its block's
// tree (metal/abi/VerifyTree.h). A lane whose row is below its rows cuts each
// row's path at the path's row of depth row: the path's row of depth cut + i
// selects its end token tokens[i], for each i below its count, unless a draft
// token on the path up to the cut is close or also_ends, which ended thinking
// first. The end of a lane in waiting_mask, a constrained one, waits for its
// mask: a path's cut is instead the first of its rows from depth row on whose
// constraint mask row (mask_row on, of mask_words words, as in
// TargetSamplingParams) admits tokens[0], the path's rows before it keeping
// their own selections, as do all the rows of a path without one. A row whose
// selection is the sentinel of a row with no selectable logit (a token outside
// the vocabulary) keeps it, so the lane still fails. A lane whose row is 0 and
// whose end does not wait proposes its end tokens as a host chain of its tree
// (DraftTreeParams), so acceptance keeps them all.
struct ThinkingCutoffParams {
  uint32_t vocabulary;
  uint32_t waiting_mask;
  uint32_t mask_words;
  uint32_t mask_row;
  uint32_t row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t count[SPLASH_MAXIMUM_BATCH_WIDTH];
  // The token that closes thinking, the last of the lane's end.
  uint32_t close[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t also_ends[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t tokens[SPLASH_MAXIMUM_BATCH_WIDTH][SPLASH_TARGET_VERIFY_ROWS];
};

static_assert(sizeof(ThinkingCutoffParams) == 16 + (16 + 4 * SPLASH_TARGET_VERIFY_ROWS) * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Thinking cutoff parameters are the same size on both sides");

// The batched selector and acceptance kernels' grids cover exactly the
// dispatch's lanes: per-lane arrays hold those lanes, and entries past them
// are zero and unread. The draft's logits rows hold `columns` tokens of the
// vocabulary's (ops::DraftVocabulary): column c is token c below `leading`
// and token c - leading + trailing from there; the codebooks hold a row of
// each of the `vocabulary` tokens.
struct SelectorBatchParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t vocabulary;
  uint32_t columns;
  uint32_t leading;
  uint32_t trailing;
};

static_assert(sizeof(SelectorBatchParams) == 20 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft selector parameters are the same size on both sides");

// The most tokens a host chain proposes: a verify block's rows after its
// anchor.
#define SPLASH_TREE_CHAIN_TOKENS (SPLASH_TARGET_VERIFY_ROWS - 1u)
// A host chain token's estimates, by its agreement with the draft at its
// depth (model::LookupAgreement): the draft's most likely candidate there,
// another of its candidates, none of them, or past the draft's positions.
#define SPLASH_TREE_AGREEMENTS 4u
// The most children a tree's draft row takes: a greedy lane's most likely
// candidates, a sampled lane's draws (DraftTreeParams). On Qwen3.8-27B's
// sampled cycles (temperature 1, top_k 20, top_p 0.95) a row kept 0.701 of
// the proposals at each position with one child, 0.785 with two and 0.841
// with four.
#define SPLASH_TREE_CHILDREN 4u

// The tree each lane grows (draft_select_tree), up to a verify block's rows
// from row l * SPLASH_TARGET_VERIFY_ROWS of the selector's trees, its
// anchor's first, in the order it takes them. A row's estimate is the
// probability that the target accepts the path to it:
// the product over the path of its steps' estimates. A step to a draft
// candidate at proposal position p, of probability q among the position's
// candidates after its parent's (the softmax at temperature 1 of their unary
// scores plus the edge from the parent's candidate, each sum softcapped,
// softcap tanh(sum / softcap), where the draft's head takes a softcap, as
// DFlash2 trains a draft of such a head; as it is at a softcap of 0), is
// estimated at
// sigmoid(slope[l] * logit(q) + intercept[l][p]), the lane's acceptance model
// (a greedy lane's or a sampled one's, model::VerifyBudget), at most its cap:
// what its parent's host chain child and the steps its parent drew before it
// left of 1, as the target selects one token after the parent, so that a
// row's children have estimates summing to at most its own; a step to host
// chain token d of the lane, chain[l] tokens from the anchor, at its rate for
// its agreement with the draft at depth d, beside its parent's drawn
// children, whose caps count the rate first (a step the parent drew to the
// chain's token is that child: its cap is what the others left, and it takes
// the more of its estimate and the rate). The tree grows best first by
// estimate, a draft row taking children[l] candidates and a chain token the
// next, one row for each token path (a step to a token its parent already
// has a child of joins that child), until it holds the lane's rows, or
// depth[l] + 1 rows, as many as the tokens a walk of it may retain, or none
// no deeper than depth[l] is left. A greedy lane's row (temperature[l] 0)
// takes its most
// likely candidates; a sampled lane's draws them without replacement from
// q_T, the candidates' probabilities at the lane's temperature T, its
// Gumbel-top-k: the largest log q_T + G, G a Gumbel of the row's key and the
// candidate's token, a row's key a hash of its parent's and its token, the
// anchor's the lane's noise[l] (a 64-bit seed, low word first), so that every
// row draws with noise of its own. A sampled lane's q is q_T. A row's draft
// probability is the q of the draft step that made it, or joined it where a
// host chain made it first; a chain row no draft step joined has none (-1).
struct DraftTreeParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t depth[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t chain[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t children[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  float softcap;
  float slope[SPLASH_MAXIMUM_BATCH_WIDTH];
  float intercept[SPLASH_MAXIMUM_BATCH_WIDTH][SPLASH_DRAFT_PROPOSAL_TOKENS];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t noise[SPLASH_MAXIMUM_BATCH_WIDTH][2];
};

static_assert(sizeof(DraftTreeParams) == 8 + (32 + 4 * SPLASH_DRAFT_PROPOSAL_TOKENS) * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft tree parameters are the same size on both sides");

// The rows of a decode cycle (draft_verify_rows): the `rows` rows of its
// step's plan (ops::StepRows), laid out over its lanes' trees
// (DraftTreeParams), 1 to lane_cap a lane, every row a lane holds verified.
// Every lane holds its anchor; the plan's other rows go to the largest
// estimates of the trees' rows, a lane's in the order its tree took them
// (each at most its parent's, which comes before it), the earlier lane first
// among equal ones, none past a lane's cap and none estimated below `floor`
// (at 0, every row the plan and the caps hold). A row is so taken by its own
// estimate and its rank among all rows: its estimate reads its ancestors'
// draws, which made it and the siblings drawn before it, and never its own;
// every row ranked above it is another lane's, or its own tree's that its
// draws did not make, and its own descendants rank below it. So no row is
// verified for its own draws, which would leave its selection no draw from
// the target: a rule may weigh a row by its own estimate and the rows ranked
// above it, as the rank and the floor do, and never by the rows ranked after
// it, its own descendants among them. A tree's estimates are finite (no row
// is made of a candidate whose probability is not above 0), so they sort. A
// sampled row tests every child its draft drew, whether or not the layout
// holds the child's row (decode_sample_vocabulary_draw), and acceptance
// follows its selection only to a row its lane holds (AcceptBatchParams): a
// child the plan, the cap or the floor left out ends the lane's path, its
// token the next cycle's anchor. A lane's rows are its anchor and the rows
// it took, numbered in preorder, its likeliest path on its first rows, and
// lanes' rows are packed in lane order (SplashLaneRows). Where the lanes
// take fewer rows than the plan (rows under the floor, a sampled lane that
// drew fewer children than its chances promised, or a tree shallower than
// its lane's share of the step), the plan's rows past the lanes' belong to
// no lane: no kernel selects, retains, attends or routes through the MoE
// experts a row of no lane, and the projections compute it unread, as they
// compute a tile's spare rows. Each carries lane 0's anchor, as the
// embedding gathers by token. The rows, the cap and the floor come from the
// host before this cycle's draws, so they admit no row by its own draws: the
// step costs' choice from the last cycle's chances
// (model::StepCosts::choose), a draft's block a lane under --verify-rows
// all, a step within that block at its cost line's floor for a target whose
// experts stream (model::CostLine::choose), or a step a test forces.
struct VerifyRowsParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t lane_cap;
  float floor;
};

static_assert(sizeof(VerifyRowsParams) == 16, "Verify rows parameters are 16 bytes on both sides");

// Acceptance of each lane's rows (SplashLaneRows) of the command's packed
// rows, as a tree (metal/abi/VerifyTree.h): from its anchor's row the lane
// follows the child whose token (its verify input row) is the current row's
// selection, until no child is. It retains the selections of that path's
// rows in order, the last the target's token after the path, at most
// remaining[l] of them and none after a stop token; the mask of the rows it
// retains (bit j for row j of its block); and the path's proposals among
// them, all of them where the cap or a stop token cuts the path. A greedy
// row selects its argmax and a sampled row by rejection sampling over the
// children its draft drew there (SPLASH_SAMPLING_UNIFORMS), a draw from the
// target's distribution, so a lane retains the target's own tokens under
// either policy.
struct AcceptBatchParams {
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_tokens[SPLASH_STOP_TOKENS];
};

static_assert(sizeof(AcceptBatchParams) == 16 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Batched acceptance parameters are the same size on both sides");
// ---- end runtime/metal/abi/Sampling.h ----
// ---- begin runtime/metal/abi/Vision.h (Splash 2afc733, Apache-2.0) ----

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The epsilon of the vision tower's LayerNorms, which a vision GGUF must
// state; unsuffixed, as SPLASH_RMS_EPSILON.
#define SPLASH_VISION_NORM_EPSILON 1e-6

struct VisionGridParams {
  uint32_t grid_height;
  uint32_t grid_width;
};

static_assert(sizeof(VisionGridParams) == 8,
              "Vision grid parameters are 8 bytes on both sides");

struct VisionQkvParams {
  uint32_t tokens;
  uint32_t padded_tokens;
};

static_assert(sizeof(VisionQkvParams) == 8,
              "Vision QKV parameters are 8 bytes on both sides");
// ---- end runtime/metal/abi/Vision.h ----
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;
// ---- end runtime/metal/abi/KernelABI.h ----
// ---- begin runtime/metal/abi/MessageGate.h (Splash 2afc733, Apache-2.0) ----

// Parameters of the message gate kernels (kernels/shared/message_gate.metal,
// ops/MessageGate.cpp): a row x takes a message m by gates of a low-rank
// projection of both, g = down(norm(x), norm(m)), as
// (1 - tanh(forget g)) x + tanh(gain g) m, in fp32; a weighted RMS norm then
// writes the projections' bf16 input. The kernels below read the gates'
// weights themselves; the reference path they are checked against, which
// rounds at the same points, runs in the tests
// (dev/tests/engine/message_gate_reference.h).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define MESSAGE_GATE_THREADS 256u

// The fused gates (ops::MessageGate, a decode step's rows): three kernels over
// up to MESSAGE_GATE_MAXIMUM_GATES gates of the same rows (a layer's q, k and
// v inputs: grid z), the last two over groups of MESSAGE_GATE_GROUP_ROWS rows,
// which read a gate's weights, F32 or BF16 rows as stored, once a group.
// message_gate_normalize normalizes and concatenates each row of a gate once,
// a threadgroup a row, into the bf16 rows that each of the down projection's
// threadgroups of the row's group reads, where each had normalized its
// group's rows itself: an eight-row decode step's gates took 2.13 ms where
// they took 2.64 (Inkling-2-Mini on a 16-core M5 Pro, separate dispatches),
// the step 0.47 ms less. message_gate_down writes MESSAGE_GATE_DOWN_OUTPUTS
// of the group's rank values a threadgroup (32 threadgroups a gate for a rank
// of 64; a simdgroup a value, 8 threadgroups, took 2.0 ms of a decode step's
// 96 gates against 1.25 ms on an M5 Pro, F32 weights a gate a dispatch);
// message_gate_apply takes MESSAGE_GATE_THREADS of a row's values a
// threadgroup (a slice), a thread each: forget, gain and the gated row, whose
// squares the slices of a group add, the last to arrive in slice order, which
// then writes the group's normalized rows (slices of 128 and 64 values, two
// and four threads a value, ran no faster). A rank is at most
// MESSAGE_GATE_MAXIMUM_RANK, a multiple of four.
#define MESSAGE_GATE_GROUP_ROWS 8u
#define MESSAGE_GATE_DOWN_OUTPUTS 2u
#define MESSAGE_GATE_MAXIMUM_RANK 64u
#define MESSAGE_GATE_MAXIMUM_GATES 3u

// The fused gates of more rows (ops::MessageGate, a prompt chunk's): one
// kernel, message_gate_rows, of a gate's MESSAGE_GATE_TILE_ROWS rows a
// threadgroup of MESSAGE_GATE_TILE_THREADS threads (grid z the gates, as
// above), which reads the tile's rows and the gate's weights itself and keeps
// its rank values in threadgroup memory: the tile's inverse RMS, its rank
// values, its gated rows, which it stores to the gate's scratch and reads
// back, and their norm, the projections on simdgroup float MMA, each
// simdgroup 8 columns of a 64-column block for the tile's rows, so that the
// tile reads each weight once, the down projection's normalized inputs staged
// in threadgroup memory. Its rows of whole 64-value steps, its rank a
// multiple of 8. A 2,583-token Inkling-2-Mini prefill's gates took 102 ms on
// it on a 16-core M5 Pro, against 141 ms as the float projections' four
// dispatches a gate and 128 ms with each simdgroup reading its own operands
// from device memory (2 x 4 tiles of 16 x 16 a threadgroup).
#define MESSAGE_GATE_TILE_ROWS 32u
#define MESSAGE_GATE_TILE_THREADS 256u

// The fused gates of a dispatch: `gates` gates of `rows` rows of `hidden`
// values, gate g's row x at column x_columns[g] of rows of x_stride values and
// its message at message_columns[g] of rows of message_stride values, both
// fp32, in whole float4s; `rank` values a row between each gate's projections.
struct MessageGateFusedParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t x_stride;
  uint32_t message_stride;
  uint32_t rank;
  uint32_t gates;
  uint32_t x_columns[MESSAGE_GATE_MAXIMUM_GATES];
  uint32_t message_columns[MESSAGE_GATE_MAXIMUM_GATES];
};

static_assert(sizeof(MessageGateFusedParams) == 48, "fused message gate parameters are 48 bytes on both sides");
// ---- end runtime/metal/abi/MessageGate.h ----
// ---- begin runtime/metal/kernels/common/rms_inverse.h (Splash 2afc733, Apache-2.0) ----


// The epsilon of every RMS norm.
constant constexpr float kRmsEpsilon = SPLASH_RMS_EPSILON;

// The inverse RMS of `width` values from each thread's sum of squares,
// reduced per simdgroup and then over the first Simdgroups simdgroups, in
// order, by thread 0, and returned to every thread. `reductions` holds a
// partial per simdgroup. This and row_squares keep the source order of their
// float additions (reassociate(off), scoped to their bodies), whichever file
// includes them.
template <ushort Simdgroups = 8>
inline float rms_inverse_of_sums(float sum, uint width, threadgroup float *reductions,
                                 uint thread_index, uint lane, uint simd_group) {
#pragma clang fp reassociate(off)
  sum = simd_sum(sum);
  if (lane == 0)
    reductions[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < Simdgroups; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / width + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return reductions[0];
}

// A thread's squares of columns thread_index + 256 i of a row in device or
// threadgroup memory, in column order.
template <class Row>
inline float row_squares(Row row, uint width, uint thread_index) {
#pragma clang fp reassociate(off)
  float sum = 0.0f;
  for (uint column = thread_index; column < width; column += 256) {
    float value = float(row[column]);
    sum += value * value;
  }
  return sum;
}

// Inverse RMS of one row, reduced across the 256-thread group and returned to
// every thread.
inline float rms_inverse(device const bfloat *row_input, uint width,
                         threadgroup float *reductions, uint thread_index,
                         uint lane, uint simd_group) {
  return rms_inverse_of_sums(row_squares(row_input, width, thread_index), width, reductions,
                             thread_index, lane, simd_group);
}
// ---- end runtime/metal/kernels/common/rms_inverse.h ----
// ---- begin runtime/metal/kernels/common/sgmatrix.h (Splash 2afc733, Apache-2.0) ----
#include <metal_stdlib>
using namespace metal;

// The 8x8 simdgroup_matrix MMA on operands held in plain registers, shared by
// the register matrix kernels (decode/linear_gguf_sgmatrix.metal,
// shared/gguf_float.metal).
namespace sgmatrix {
// simdgroup_matrix lane -> element mapping (verified by the driver's probe):
// a lane's thread_elements() are M[fm][fn] and M[fm][fn + 1].
struct Lane {
  ushort fm;
  ushort fn;
};
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  Lane l;
  l.fm = ushort((qid & 4) | ((lane >> 1) & 3));
  l.fn = ushort(((qid & 2) << 1) | ((lane & 1) << 1));
  return l;
}

template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

// One 8x8x8 MMA on plain-register operands: c += a x b. The simdgroup_matrix
// objects live only inside this call (the MLX steel pattern), so the
// persistent accumulator is an ordinary float2 the compiler keeps in
// registers; a persistent simdgroup_matrix read through thread_elements()
// each group was kept in thread memory instead.
template <typename T>
__attribute__((always_inline)) inline void
mma_acc(thread float2 &c, vec<T, 2> a, vec<T, 2> b) {
  simdgroup_matrix<T, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}
} // namespace sgmatrix
// ---- end runtime/metal/kernels/common/sgmatrix.h ----
// ---- begin runtime/metal/kernels/common/split_reduce.h (Splash 2afc733, Apache-2.0) ----
#include <metal_stdlib>
using namespace metal;

// Cross-threadgroup reduction of a K split, shared by the split decode
// kernels. Each of the `splits` threadgroups of one output tile stores its
// fp32 partials, in its kernel's own layout, through a device
// coherent(device) pointer and then arrives at the tile's counter. The last
// one to arrive adds the splits in split order, its own from registers, and
// returns the counter to zero for the next dispatch. The result does not
// depend on scheduling, and no threadgroup waits for another.

// True in every thread of the threadgroup that arrives last. A store reaches
// another threadgroup's load only between two device-scope fences, the
// storing thread's after it and the loading thread's before it: a fence
// orders the accesses of the thread that runs it, and a barrier's memory
// flags those of its own threadgroup alone. So every thread fences its
// partials, the first barrier holds thread zero's arrival until all have,
// and every thread of the threadgroup whose arrival counted all the others
// fences again before it reads theirs. With no fence, half of 8e10 reads of
// another split's partials were of an earlier dispatch's (32-core M5 Max,
// tiles of 32 to 128 threads in 2 and 8 splits); with these, none, as with
// thread zero's alone, which Metal does not promise the other threads.
inline bool split_arrive_last(device atomic_uint *counter, uint splits,
                              uint thread_index, threadgroup uint *arrival) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  threadgroup_barrier(mem_flags::mem_none);
  if (thread_index == 0)
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (*arrival != splits - 1)
    return false;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  return true;
}

// The splits' partials added in split order: `own` for split `self`,
// `partial(s)` for every other split s.
template <typename T, typename Partial>
inline T split_sum(T own, uint self, uint splits, Partial partial) {
#pragma clang fp reassociate(off)
  T total = T(0);
  for (uint s = 0; s < splits; ++s)
    total += s == self ? own : partial(s);
  return total;
}

// Returns the tile's counter to zero; called by the reducing threadgroup.
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0)
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
}
// ---- end runtime/metal/kernels/common/split_reduce.h ----

#include <metal_stdlib>

using namespace metal;

// The fused message gates (metal/abi/MessageGate.h) over fp32 rows: a decode
// step's of MESSAGE_GATE_THREADS threads a threadgroup, a prompt chunk's of
// MESSAGE_GATE_TILE_THREADS (message_gate_rows below). Their reference path,
// which rounds where they do, runs in the tests alone
// (dev/tests/engine/message_gate_reference.h).
constant constexpr uint kGateThreads = MESSAGE_GATE_THREADS;
constant constexpr uint kGateSimdgroups = kGateThreads / 32;

// ---- The fused gates (metal/abi/MessageGate.h): a decode step's rows --------
//
// Each kernel runs in phases, functions out of line, which hand over through
// threadgroup memory alone: every sum is reduced by simd_sum into its own slot
// a simdgroup and then by one thread over the simdgroups in order; loops over
// a group's rows run to MESSAGE_GATE_GROUP_ROWS, the rows past the group's
// storing nothing; no thread keeps an array across a barrier (as
// block_residual_mix.metal: an array indexed by a runtime bound and alive
// across a barrier lost one simdgroup's values now and then under shader
// validation). The sums are those of message_gate_test_prepare, the float
// projections' and message_gate_test_norm's, in their order within a thread and
// over the simdgroups. A dispatch runs MessageGateFusedParams::gates gates
// (grid z), gate g reading its weights, norm and output at slot g of their
// kernel's MESSAGE_GATE_MAXIMUM_GATES bindings and its rows of the scratch
// after gate g - 1's: normalized concatenations [gates][rows][2 hidden], rank
// values [gates][rows][rank], gated rows [gates][rows][hidden], partial sums
// [gates][groups][slices][rows of a group] and counters [gates][groups].
// Weights: the gates' float rows as stored, F32 or BF16 (the _f32 and _bf16
// instances), each value widened to fp32.
constant constexpr uint kGroupRows = MESSAGE_GATE_GROUP_ROWS;
constant constexpr uint kDownOutputs = MESSAGE_GATE_DOWN_OUTPUTS;

// Quad i of a row of weights, in fp32.
inline float4 message_gate_quad(device const float *row, uint i) {
  return reinterpret_cast<device const float4 *>(row)[i];
}
inline float4 message_gate_quad(device const bfloat *row, uint i) {
  return float4(reinterpret_cast<device const bfloat4 *>(row)[i]);
}

// Gate g's binding of three.
template <typename T> inline T message_gate_pick(uint g, T first, T second, T third) {
  return g == 0 ? first : g == 1 ? second : third;
}

// A group of `count` rows from row `first`: gate g's row r's x, or its
// message, the group's last row standing in for the rows past it (their
// scales are zero).
inline device const float *message_gate_row(device const float *x, device const float *message,
                                            constant MessageGateFusedParams &p, uint g, uint first, uint count,
                                            uint r, bool own) {
  const ulong row = first + min(r, count - 1);
  return own ? x + row * p.x_stride + p.x_columns[g] : message + row * p.message_stride + p.message_columns[g];
}

// What message_gate_normalize's phases share: a partial a simdgroup of the
// row's and of the message's squares, and their inverse RMS.
struct MessageGateNormalizeShared {
  float partials[2 * kGateSimdgroups];
  float scales[2];
};

// Phase one: the inverse RMS of the row and of its message, scales 0 and 1,
// from quads: a thread's quads tid + MESSAGE_GATE_THREADS i in order, each
// quad's squares added in order (a sum of the same terms as
// message_gate_test_prepare's in another order: it reads a value a thread at
// a time).
__attribute__((noinline)) void message_gate_normalize_scales(device const float4 *row, device const float4 *message,
                                                             uint hidden, uint tid, uint lane, uint sg,
                                                             threadgroup MessageGateNormalizeShared &shared) {
  for (uint i = 0; i < 2; ++i) {
    device const float4 *vector = i == 0 ? row : message;
    float squares = 0.0f;
    for (uint d = tid; d < hidden / 4; d += kGateThreads) {
      const float4 v = vector[d];
      squares += v.x * v.x;
      squares += v.y * v.y;
      squares += v.z * v.z;
      squares += v.w * v.w;
    }
    squares = simd_sum(squares);
    if (lane == 0) shared.partials[i * kGateSimdgroups + sg] = squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2) {
    float total = 0.0f;
    for (uint s = 0; s < kGateSimdgroups; ++s) total += shared.partials[tid * kGateSimdgroups + s];
    shared.scales[tid] = rsqrt(total / hidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The down projection's input of a row of each gate, a threadgroup each: the
// row's x and its message, each RMS-normalized without weights
// (message_gate_normalize_scales) and rounded to bf16 as
// message_gate_test_prepare rounds it, side by side into the row's 2 hidden
// values of `concat` [gates][rows][2 hidden], once for every threadgroup of
// message_gate_down that reads the row. Grid: (rows, 1, gates).
kernel void message_gate_normalize(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                                   device bfloat *concat [[buffer(2)]],
                                   constant MessageGateFusedParams &p [[buffer(3)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateNormalizeShared shared;
  const uint row = group.x, g = group.z;
  if (row >= p.rows || g >= p.gates) return;
  device const float4 *xr = reinterpret_cast<device const float4 *>(x + ulong(row) * p.x_stride + p.x_columns[g]);
  device const float4 *mr =
      reinterpret_cast<device const float4 *>(message + ulong(row) * p.message_stride + p.message_columns[g]);
  message_gate_normalize_scales(xr, mr, p.hidden, tid, lane, sg, shared);
  const uint quads = p.hidden / 4;
  device bfloat4 *out = reinterpret_cast<device bfloat4 *>(concat + (ulong(g) * p.rows + row) * 2 * p.hidden);
  for (uint i = tid; i < 2 * quads; i += kGateThreads)
    out[i] = i < quads ? bfloat4(xr[i] * shared.scales[0]) : bfloat4(mr[i - quads] * shared.scales[1]);
}

// The group's rank values k0 .. k0 + kDownOutputs - 1, each the dot of the
// weights' row (`down`, [rank][2 hidden]) with the normalized concatenation
// (message_gate_normalize's `concat` [rows][2 hidden], the gate's), the
// threads striding its quads, a partial a simdgroup of each output's value of
// each row; rounded to bf16 into `low` [rows][rank], the gate's.
template <typename Weight>
__attribute__((noinline)) void message_gate_down_values(device const bfloat *concat, device const Weight *down,
                                                        device bfloat *low, constant MessageGateFusedParams &p,
                                                        uint first, uint count, uint k0, uint tid, uint lane, uint sg,
                                                        threadgroup float *partials) {
  const uint quads = p.hidden / 2;
  float sums[kDownOutputs * kGroupRows];
  for (uint j = 0; j < kDownOutputs * kGroupRows; ++j) sums[j] = 0.0f;
  for (uint i = tid; i < quads; i += kGateThreads) {
    float4 w[kDownOutputs];
    for (uint o = 0; o < kDownOutputs; ++o)
      w[o] = k0 + o < p.rank ? message_gate_quad(down + ulong(k0 + o) * 2 * p.hidden, i) : float4(0.0f);
    for (uint r = 0; r < kGroupRows; ++r) {
      const float4 normalized = message_gate_quad(concat + ulong(first + min(r, count - 1)) * 2 * p.hidden, i);
      for (uint o = 0; o < kDownOutputs; ++o) sums[o * kGroupRows + r] += dot(w[o], normalized);
    }
  }
  for (uint j = 0; j < kDownOutputs * kGroupRows; ++j) {
    const float sum = simd_sum(sums[j]);
    if (lane == 0) partials[j * kGateSimdgroups + sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < kDownOutputs * kGroupRows) {
    float total = 0.0f;
    for (uint s = 0; s < kGateSimdgroups; ++s) total += partials[tid * kGateSimdgroups + s];
    const uint o = tid / kGroupRows, r = tid % kGroupRows;
    if (r < count && k0 + o < p.rank) low[ulong(first + r) * p.rank + k0 + o] = bfloat(total);
  }
}

// The rank values of a group of MESSAGE_GATE_GROUP_ROWS rows of each gate, of
// MESSAGE_GATE_DOWN_OUTPUTS of them a threadgroup, each output's weights' row
// read once for every row of the group, from the rows' normalized
// concatenations (message_gate_normalize). Grid: (rank /
// MESSAGE_GATE_DOWN_OUTPUTS, groups, gates).
template <typename Weight>
kernel void message_gate_down(device const bfloat *concat [[buffer(0)]], device const Weight *down0 [[buffer(1)]],
                              device const Weight *down1 [[buffer(2)]], device const Weight *down2 [[buffer(3)]],
                              device bfloat *low [[buffer(4)]], constant MessageGateFusedParams &p [[buffer(5)]],
                              uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[kDownOutputs * kGroupRows * kGateSimdgroups];
  const uint first = group.y * kGroupRows, g = group.z;
  if (first >= p.rows || g >= p.gates) return;
  const uint count = min(kGroupRows, p.rows - first);
  message_gate_down_values(concat + ulong(g) * p.rows * 2 * p.hidden, message_gate_pick(g, down0, down1, down2),
                           low + ulong(g) * p.rows * p.rank, p, first, count, group.x * kDownOutputs, tid, lane, sg,
                           partials);
}

// What message_gate_apply's phases share: the group's rank values (zero past
// its rows), a partial a simdgroup of each row's squares, the arrival of
// split_arrive_last, and the last slice's inverse RMS of each row.
struct MessageGateApplyShared {
  float4 lows[kGroupRows * MESSAGE_GATE_MAXIMUM_RANK / 4];
  float partials[kGroupRows * kGateSimdgroups];
  uint arrival;
  float inverses[kGroupRows];
};

// Phase one: value d of each row of the group, a thread each: forget and gain
// from the rank values and the weights' rows ([hidden][rank]), the gated
// value in fp32 (message_gated's) to `values` [rows][hidden], and the slice's
// squares of each row, by thread 0 to its slot of `partials`
// [slice][MESSAGE_GATE_GROUP_ROWS] (the group's).
template <typename Weight>
__attribute__((noinline)) void message_gate_apply_values(device const float *x, device const float *message,
                                                         device const Weight *forget, device const Weight *gain,
                                                         device coherent(device) float *values,
                                                         device coherent(device) float *partials,
                                                         constant MessageGateFusedParams &p, uint g, uint first,
                                                         uint count, uint slice, uint tid, uint lane, uint sg,
                                                         threadgroup MessageGateApplyShared &shared) {
  const uint d = slice * kGateThreads + tid, quarters = p.rank / 4;
  const bool active = d < p.hidden;
  float f[kGroupRows], e[kGroupRows];
  for (uint r = 0; r < kGroupRows; ++r) f[r] = e[r] = 0.0f;
  if (active) {
    device const Weight *f4 = forget + ulong(d) * p.rank, *e4 = gain + ulong(d) * p.rank;
    for (uint j = 0; j < quarters; ++j) {
      const float4 a = message_gate_quad(f4, j), b = message_gate_quad(e4, j);
      for (uint r = 0; r < kGroupRows; ++r) {
        const float4 l = shared.lows[r * quarters + j];
        f[r] += dot(a, l);
        e[r] += dot(b, l);
      }
    }
  }
  for (uint r = 0; r < kGroupRows; ++r) {
    float squares = 0.0f;
    if (active && r < count) {
      const float xv = x[ulong(first + r) * p.x_stride + p.x_columns[g] + d];
      const float mv = message[ulong(first + r) * p.message_stride + p.message_columns[g] + d];
      const float v = (1.0f - precise::tanh(f[r])) * xv + precise::tanh(e[r]) * mv;
      values[ulong(first + r) * p.hidden + d] = v;
      squares = v * v;
    }
    squares = simd_sum(squares);
    if (lane == 0) shared.partials[r * kGateSimdgroups + sg] = squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0)
    for (uint r = 0; r < count; ++r) {
      float total = 0.0f;
      for (uint s = 0; s < kGateSimdgroups; ++s) total += shared.partials[r * kGateSimdgroups + s];
      partials[slice * kGroupRows + r] = total;
    }
}

// Phase two, by the slice that arrives last: each row's squares over the
// slices in slice order, by a thread a row, then the group's rows through the
// weighted RMS norm into bf16, as message_gate_test_norm writes each value, the
// threads striding the rows' quads.
__attribute__((noinline)) void message_gate_apply_norm(device coherent(device) float *values,
                                                       device coherent(device) float *partials,
                                                       device const float *norm, device bfloat *output,
                                                       constant MessageGateFusedParams &p, uint first, uint count,
                                                       uint slices, uint tid,
                                                       threadgroup MessageGateApplyShared &shared) {
  if (tid < count) {
    float total = 0.0f;
    for (uint s = 0; s < slices; ++s) total += partials[s * kGroupRows + tid];
    shared.inverses[tid] = rsqrt(total / p.hidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint quads = p.hidden / 4;
  device coherent(device) float4 *rows =
      reinterpret_cast<device coherent(device) float4 *>(values + ulong(first) * p.hidden);
  device bfloat4 *out = reinterpret_cast<device bfloat4 *>(output + ulong(first) * p.hidden);
  for (uint q = tid; q < count * quads; q += kGateThreads) {
    const uint r = q / quads, e = 4 * (q % quads);
    const float4 weights = float4(norm[e], norm[e + 1], norm[e + 2], norm[e + 3]);
    out[q] = bfloat4((rows[q] * shared.inverses[r]) * weights);
  }
}

// The gated rows of a group of each gate from its rank values (`low`): a
// slice of MESSAGE_GATE_THREADS values a threadgroup, its forget and gain rows
// read once for the group's rows; the slice that arrives last
// (kernels/common/split_reduce.h, the group's counter, zero before and after)
// writes the group's normalized rows. Grid: (slices, groups, gates).
template <typename Weight>
kernel void message_gate_apply(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                               device const Weight *forget0 [[buffer(2)]], device const Weight *forget1 [[buffer(3)]],
                               device const Weight *forget2 [[buffer(4)]], device const Weight *gain0 [[buffer(5)]],
                               device const Weight *gain1 [[buffer(6)]], device const Weight *gain2 [[buffer(7)]],
                               device const bfloat *low [[buffer(8)]], device const float *norm0 [[buffer(9)]],
                               device const float *norm1 [[buffer(10)]], device const float *norm2 [[buffer(11)]],
                               device coherent(device) float *values [[buffer(12)]],
                               device coherent(device) float *partials [[buffer(13)]],
                               device atomic_uint *counters [[buffer(14)]], device bfloat *output0 [[buffer(15)]],
                               device bfloat *output1 [[buffer(16)]], device bfloat *output2 [[buffer(17)]],
                               constant MessageGateFusedParams &p [[buffer(18)]],
                               uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateApplyShared shared;
  const uint first = group.y * kGroupRows, g = group.z;
  if (first >= p.rows || g >= p.gates) return;
  const uint count = min(kGroupRows, p.rows - first), slices = (p.hidden + kGateThreads - 1) / kGateThreads;
  const uint groups = (p.rows + kGroupRows - 1) / kGroupRows;
  device const bfloat *gateLow = low + ulong(g) * p.rows * p.rank;
  threadgroup float *lows = reinterpret_cast<threadgroup float *>(shared.lows);
  for (uint i = tid; i < kGroupRows * p.rank; i += kGateThreads)
    lows[i] = i < count * p.rank ? float(gateLow[ulong(first) * p.rank + i]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device coherent(device) float *gateValues = values + ulong(g) * p.rows * p.hidden;
  device coherent(device) float *groupPartials = partials + (ulong(g) * groups + group.y) * slices * kGroupRows;
  device atomic_uint *counter = counters + g * groups + group.y;
  message_gate_apply_values(x, message, message_gate_pick(g, forget0, forget1, forget2),
                            message_gate_pick(g, gain0, gain1, gain2), gateValues, groupPartials, p, g, first, count,
                            group.x, tid, lane, sg, shared);
  if (!split_arrive_last(counter, slices, tid, &shared.arrival)) return;
  message_gate_apply_norm(gateValues, groupPartials, message_gate_pick(g, norm0, norm1, norm2),
                          message_gate_pick(g, output0, output1, output2), p, first, count, slices, tid, shared);
  split_release(counter, tid);
}

// ---- The fused gates of a prompt chunk's rows (metal/abi/MessageGate.h) ----
//
// message_gate_rows runs a gate of MESSAGE_GATE_TILE_ROWS rows a threadgroup
// from its rows to its output, in four phases that hand over through
// threadgroup memory and, for the gated rows, the threadgroup's own rows of
// device scratch: each row's and message's inverse RMS; the rank values, the
// down projection of the normalized concatenation, each normalized value
// rounded to bf16 as message_gate_test_prepare rounds it, each rank value rounded
// to bf16 as the float projection writes it; forget, gain and the gated row in
// fp32, a 64-value block of the rows after another, into the gated rows'
// scratch, with each row's squares; and the gated rows through the weighted
// RMS norm into bf16, as message_gate_test_norm writes each value. The products
// run on simdgroup float 8x8 MMA with exact operands (bf16 values widened, the
// weights as stored, BF16 ones widened, as shared/gguf_float.metal's), so a
// sum differs from the float projections' only by its fp32 order. Simdgroup s
// takes columns [8 s, 8 s + 8) of a 64-column block of either projection for
// the tile's 32 rows (four 8-row fragments), so the tile reads each weight
// once; the down projection's normalized inputs of a 64-input step, which
// every simdgroup multiplies, are staged in threadgroup memory, two steps'
// stages alternating. MMA j of a 64-input step multiplies inputs 8 i + j over
// its index i (gguf_float's order): lane (fm, fn) reads a row's inputs
// [8 fn, 8 fn + 16) and a weights' row's [8 fm, 8 fm + 8) of the step, whole
// runs of 8. Rows past the chunk read its last row with scales of zero and
// store nothing.
constant constexpr uint kTileRows = MESSAGE_GATE_TILE_ROWS;
constant constexpr uint kTileSimdgroups = MESSAGE_GATE_TILE_THREADS / 32;
constant constexpr uint kTileStep = 64;
constant constexpr uint kTileFragments = kTileRows / 8;
// A staged row of a step's 64 inputs, or of the rank values, bf16 values
// padded by 8 so that the fragments' rows start in other banks.
constant constexpr uint kTileStride = kTileStep + 8;
static_assert(kTileSimdgroups * 8 == kTileStep, "a tile's simdgroups take a block's 64 columns, 8 each");
static_assert(MESSAGE_GATE_MAXIMUM_RANK == kTileStep, "a rank's values are one step of the forget and gain sums");
static_assert(kTileRows * kTileStep / MESSAGE_GATE_TILE_THREADS == 8, "a thread stages a run of 8 inputs a step");
static_assert(kTileFragments == 4, "a lane's partial squares are a float4, a fragment each");

// Eight consecutive weights of a row from input k, widened to fp32: two
// quads, zero where `inside` is false.
struct MessageGateEight {
  float4 low, high;
  float operator[](uint i) const { return i < 4 ? low[i] : high[i - 4]; }
};
inline MessageGateEight message_gate_eight(device const float *row, uint k, bool inside) {
  device const float4 *quads = reinterpret_cast<device const float4 *>(row + k);
  return inside ? MessageGateEight{quads[0], quads[1]} : MessageGateEight{float4(0.0f), float4(0.0f)};
}
inline MessageGateEight message_gate_eight(device const bfloat *row, uint k, bool inside) {
  device const bfloat4 *quads = reinterpret_cast<device const bfloat4 *>(row + k);
  return inside ? MessageGateEight{float4(quads[0]), float4(quads[1])} : MessageGateEight{float4(0.0f), float4(0.0f)};
}

// What message_gate_rows' phases share: each row's and message's inverse RMS
// (zero past the chunk's rows); the down projection's two stages of a step's
// normalized inputs [row][kTileStride], bf16 as rounded, whose memory the rank
// values, bf16 as rounded [row][kTileStride] (zero past the rank), and a
// partial of each fragment's rows' squares a lane then take; and the rows'
// inverse RMS of the gated rows.
struct MessageGateRowsShared {
  float scales[2 * kTileRows];
  union {
    bfloat stages[2][kTileRows * kTileStride];
    struct {
      bfloat lows[kTileRows * kTileStride];
      float4 partials[kTileSimdgroups * 32];
    } gated;
  };
  float inverses[kTileRows];
};

// Phase one: vector i of the tile, row i / 2's x (i even) or message, a
// simdgroup a vector: a lane's quads lane + 32 j in order, each quad's squares
// added in order, then simd_sum.
__attribute__((noinline)) void message_gate_rows_scales(device const float *x, device const float *message,
                                                        constant MessageGateFusedParams &p, uint g, uint first,
                                                        uint count, uint lane, uint sg,
                                                        threadgroup MessageGateRowsShared &shared) {
  const uint quads = p.hidden / 4;
  for (uint i = sg; i < 2 * kTileRows; i += kTileSimdgroups) {
    device const float4 *row =
        reinterpret_cast<device const float4 *>(message_gate_row(x, message, p, g, first, count, i / 2, i % 2 == 0));
    float squares = 0.0f;
    if (i / 2 < count)
      for (uint d = lane; d < quads; d += 32) {
        const float4 v = row[d];
        squares += v.x * v.x;
        squares += v.y * v.y;
        squares += v.z * v.z;
        squares += v.w * v.w;
      }
    squares = simd_sum(squares);
    if (lane == 0) shared.scales[i] = i / 2 < count ? rsqrt(squares / p.hidden + kRmsEpsilon) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A thread's run of 8 of the concatenation's step `step` (x's half and then
// the message's): its row tid / 8's inputs from 8 (tid % 8), as read.
struct MessageGateRun {
  float4 low, high;
};
inline MessageGateRun message_gate_rows_load(device const float *x, device const float *message,
                                             constant MessageGateFusedParams &p, uint g, uint first, uint count,
                                             uint step, uint steps, uint tid) {
  const bool own = step < steps;
  device const float4 *run = reinterpret_cast<device const float4 *>(
      message_gate_row(x, message, p, g, first, count, tid / 8, own) + (own ? step : step - steps) * kTileStep +
      8 * (tid % 8));
  return {run[0], run[1]};
}
// The run normalized by its row's scale and rounded to bf16, into stage
// `stage`.
inline void message_gate_rows_stage(MessageGateRun run, bool own, uint tid, threadgroup bfloat *stage,
                                    threadgroup const float *scales) {
  const uint r = tid / 8;
  const float scale = scales[2 * r + (own ? 0 : 1)];
  threadgroup bfloat4 *out = reinterpret_cast<threadgroup bfloat4 *>(stage + r * kTileStride + 8 * (tid % 8));
  out[0] = bfloat4(run.low * scale);
  out[1] = bfloat4(run.high * scale);
}
// A lane's 16 staged values of a fragment's row from 8 fn, widened: the first
// 8 and the next 8.
inline void message_gate_rows_read(threadgroup const bfloat *row, thread MessageGateEight &low,
                                   thread MessageGateEight &high) {
  threadgroup const bfloat4 *a = reinterpret_cast<threadgroup const bfloat4 *>(row);
  low = {float4(a[0]), float4(a[1])};
  high = {float4(a[2]), float4(a[3])};
}

// A half's chains of a lane's four fragments: each fragment's even and odd
// MMAs apart, in fields, so that no array lives across phase two's barriers.
struct MessageGateRowsChains {
  float2 even0, odd0, even1, odd1, even2, odd2, even3, odd3;
};
// A fragment's MMAs of a step, from its row of the stage: MMA j, j = 0 to 7 in
// order, into `even` for even j and into `odd` for odd j.
inline void message_gate_rows_mma(threadgroup const bfloat *row, MessageGateEight b0, MessageGateEight b1,
                                  thread float2 &even, thread float2 &odd) {
  MessageGateEight low, high;
  message_gate_rows_read(row, low, high);
  for (uint j = 0; j < 8; j += 2) {
    sgmatrix::mma_acc<float>(even, float2(low[j], high[j]), float2(b0[j], b1[j]));
    sgmatrix::mma_acc<float>(odd, float2(low[j + 1], high[j + 1]), float2(b0[j + 1], b1[j + 1]));
  }
}

// Phase two: the tile's rank values, the down weights' rows ([rank][2
// hidden]) against the normalized concatenation, x's half and then the
// message's, a step of 64 inputs at a time, simdgroup s rank values
// [8 s, 8 s + 8), into `shared.gated.lows`. Each half's even and odd MMAs
// accumulate apart (four chains of 1,152 products at Inkling-2-Mini's 4,608
// inputs, as gguf_float splits its own), added at the end in order. The
// chains live across every step's barrier, so they are fields, not an array
// indexed at run time (which the decode kernels above keep none of across a
// barrier, and which the AIR held in memory here even with the loops over
// them marked to unroll): each half's steps run in a loop of their own,
// which takes that half's chains.
template <typename Weight>
__attribute__((noinline)) void message_gate_rows_down(device const float *x, device const float *message,
                                                      device const Weight *down, constant MessageGateFusedParams &p,
                                                      uint g, uint first, uint count, uint tid, uint lane, uint sg,
                                                      threadgroup MessageGateRowsShared &shared) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint steps = p.hidden / kTileStep, column = 8 * sg + l.fn;
  device const Weight *w0 = down + ulong(min(column, p.rank - 1)) * 2 * p.hidden,
                      *w1 = down + ulong(min(column + 1, p.rank - 1)) * 2 * p.hidden;
  // x's half's chains (own) and the message's (other).
  MessageGateRowsChains own{}, other{};
  message_gate_rows_stage(message_gate_rows_load(x, message, p, g, first, count, 0, steps, tid), true, tid,
                          shared.stages[0], shared.scales);
  // The weights load a step ahead.
  MessageGateEight next0 = message_gate_eight(w0, 8 * l.fm, column < p.rank),
                   next1 = message_gate_eight(w1, 8 * l.fm, column + 1 < p.rank);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Step `step` into its half's chains, the four fragments' in order; the next
  // step's inputs staged and its weights loaded.
  const auto run = [&](uint step, thread MessageGateRowsChains &chains) __attribute__((always_inline)) {
    const bool next = step + 1 < 2 * steps;
    const MessageGateRun ahead =
        next ? message_gate_rows_load(x, message, p, g, first, count, step + 1, steps, tid) : MessageGateRun{};
    const MessageGateEight b0 = next0, b1 = next1;
    if (next) {
      next0 = message_gate_eight(w0, (step + 1) * kTileStep + 8 * l.fm, column < p.rank);
      next1 = message_gate_eight(w1, (step + 1) * kTileStep + 8 * l.fm, column + 1 < p.rank);
    }
    threadgroup const bfloat *row = shared.stages[step % 2] + l.fm * kTileStride + 8 * l.fn;
    message_gate_rows_mma(row, b0, b1, chains.even0, chains.odd0);
    message_gate_rows_mma(row + 8 * kTileStride, b0, b1, chains.even1, chains.odd1);
    message_gate_rows_mma(row + 16 * kTileStride, b0, b1, chains.even2, chains.odd2);
    message_gate_rows_mma(row + 24 * kTileStride, b0, b1, chains.even3, chains.odd3);
    if (next) message_gate_rows_stage(ahead, step + 1 < steps, tid, shared.stages[(step + 1) % 2], shared.scales);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  };
  for (uint step = 0; step < steps; ++step) run(step, own);
  for (uint step = steps; step < 2 * steps; ++step) run(step, other);
  // Fragment f's rank values: (x's even + odd) + (the message's even + odd).
  const auto store = [&](uint f, float2 sum) {
    const uint r = 8 * f + l.fm;
    shared.gated.lows[r * kTileStride + column] = bfloat(sum.x);
    shared.gated.lows[r * kTileStride + column + 1] = bfloat(sum.y);
  };
  store(0, (own.even0 + own.odd0) + (other.even0 + other.odd0));
  store(1, (own.even1 + own.odd1) + (other.even1 + other.odd1));
  store(2, (own.even2 + own.odd2) + (other.even2 + other.odd2));
  store(3, (own.even3 + own.odd3) + (other.even3 + other.odd3));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase three: forget and gain of a block of 64 values of the tile's rows
// after another, simdgroup s values [8 s, 8 s + 8) of the block, the weights'
// rows ([hidden][rank]) against the rank values, the gated value
// (message_gated's) into `gated` [rows][hidden], and each row's squares, a
// lane's in block order, into `shared.gated.partials`.
template <typename Weight>
__attribute__((noinline)) void message_gate_rows_gated(device const float *x, device const float *message,
                                                       device const Weight *forget, device const Weight *gain,
                                                       device float *gated, constant MessageGateFusedParams &p,
                                                       uint g, uint first, uint count, uint lane, uint sg,
                                                       threadgroup MessageGateRowsShared &shared) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint blocks = p.hidden / kTileStep;
  const bool inside = 8 * l.fm < p.rank;
  const uint k = inside ? 8 * l.fm : 0;
  float squares[kTileFragments];
  for (uint f = 0; f < kTileFragments; ++f) squares[f] = 0.0f;
  // The weights load a block ahead.
  const auto load = [&](uint block, thread MessageGateEight(&w)[4]) {
    const uint column = block * kTileStep + 8 * sg + l.fn;
    w[0] = message_gate_eight(forget + ulong(column) * p.rank, k, inside);
    w[1] = message_gate_eight(forget + ulong(column + 1) * p.rank, k, inside);
    w[2] = message_gate_eight(gain + ulong(column) * p.rank, k, inside);
    w[3] = message_gate_eight(gain + ulong(column + 1) * p.rank, k, inside);
  };
  MessageGateEight next[4];
  load(0, next);
  for (uint block = 0; block < blocks; ++block) {
    const uint column = block * kTileStep + 8 * sg + l.fn;
    const MessageGateEight f0 = next[0], f1 = next[1], e0 = next[2], e1 = next[3];
    if (block + 1 < blocks) load(block + 1, next);
    // The block's x and message values of the lane's rows, read before the
    // products (a row past the chunk's reads its last).
    float2 xs[kTileFragments], ms[kTileFragments];
    for (uint f = 0; f < kTileFragments; ++f) {
      const ulong row = first + min(8 * f + l.fm, count - 1);
      xs[f] = *reinterpret_cast<device const float2 *>(x + row * p.x_stride + p.x_columns[g] + column);
      ms[f] = *reinterpret_cast<device const float2 *>(message + row * p.message_stride + p.message_columns[g] + column);
    }
    // Every fragment's MMAs run in every lane, the rows past the chunk's too:
    // a lane's row decides only its stores.
    float2 forgets[kTileFragments], gains[kTileFragments];
    for (uint f = 0; f < kTileFragments; ++f) {
      MessageGateEight low, high;
      message_gate_rows_read(shared.gated.lows + (8 * f + l.fm) * kTileStride + 8 * l.fn, low, high);
      forgets[f] = gains[f] = float2(0.0f);
      for (uint j = 0; j < 8; ++j) {
        sgmatrix::mma_acc<float>(forgets[f], float2(low[j], high[j]), float2(f0[j], f1[j]));
        sgmatrix::mma_acc<float>(gains[f], float2(low[j], high[j]), float2(e0[j], e1[j]));
      }
    }
    for (uint f = 0; f < kTileFragments; ++f) {
      const uint r = 8 * f + l.fm;
      if (r < count) {
        const ulong row = first + r;
        const float2 xv = xs[f], mv = ms[f];
        float2 v;
        v.x = (1.0f - precise::tanh(forgets[f].x)) * xv.x + precise::tanh(gains[f].x) * mv.x;
        v.y = (1.0f - precise::tanh(forgets[f].y)) * xv.y + precise::tanh(gains[f].y) * mv.y;
        *reinterpret_cast<device float2 *>(gated + row * p.hidden + column) = v;
        squares[f] += v.x * v.x;
        squares[f] += v.y * v.y;
      }
    }
  }
  shared.gated.partials[sg * 32 + lane] = float4(squares[0], squares[1], squares[2], squares[3]);
}

// Phase four: each row's squares, by a thread a row over the simdgroups and
// their lanes of its row in order, then the tile's gated rows through the
// weighted RMS norm into bf16, the threads striding the rows' quads.
__attribute__((noinline)) void message_gate_rows_norm(device const float *gated, device const float *norm,
                                                      device bfloat *output, constant MessageGateFusedParams &p,
                                                      uint first, uint count, uint tid,
                                                      threadgroup MessageGateRowsShared &shared) {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid < count) {
    const uint f = tid / 8, fm = tid % 8;
    float total = 0.0f;
    for (uint s = 0; s < kTileSimdgroups; ++s)
      for (uint lane = 0; lane < 32; ++lane)
        if (sgmatrix::lane_map(lane).fm == fm) total += shared.gated.partials[s * 32 + lane][f];
    shared.inverses[tid] = rsqrt(total / p.hidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint quads = p.hidden / 4;
  device const float4 *rows = reinterpret_cast<device const float4 *>(gated + ulong(first) * p.hidden);
  device bfloat4 *out = reinterpret_cast<device bfloat4 *>(output + ulong(first) * p.hidden);
  for (uint q = tid; q < count * quads; q += MESSAGE_GATE_TILE_THREADS) {
    const uint r = q / quads, e = 4 * (q % quads);
    const float4 weights = float4(norm[e], norm[e + 1], norm[e + 2], norm[e + 3]);
    out[q] = bfloat4((rows[q] * shared.inverses[r]) * weights);
  }
}

// A gate's gated rows of a tile of MESSAGE_GATE_TILE_ROWS rows a threadgroup,
// gate g's gated rows' scratch `gated` g. Grid: (tiles, 1, gates).
template <typename Weight>
kernel void message_gate_rows(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                              device const Weight *down0 [[buffer(2)]], device const Weight *down1 [[buffer(3)]],
                              device const Weight *down2 [[buffer(4)]], device const Weight *forget0 [[buffer(5)]],
                              device const Weight *forget1 [[buffer(6)]], device const Weight *forget2 [[buffer(7)]],
                              device const Weight *gain0 [[buffer(8)]], device const Weight *gain1 [[buffer(9)]],
                              device const Weight *gain2 [[buffer(10)]], device const float *norm0 [[buffer(11)]],
                              device const float *norm1 [[buffer(12)]], device const float *norm2 [[buffer(13)]],
                              device float *gated0 [[buffer(14)]], device float *gated1 [[buffer(15)]],
                              device float *gated2 [[buffer(16)]], device bfloat *output0 [[buffer(17)]],
                              device bfloat *output1 [[buffer(18)]], device bfloat *output2 [[buffer(19)]],
                              constant MessageGateFusedParams &p [[buffer(20)]],
                              uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateRowsShared shared;
  const uint first = group.x * kTileRows, g = group.z;
  if (first >= p.rows || g >= p.gates) return;
  const uint count = min(kTileRows, p.rows - first);
  device float *gated = message_gate_pick(g, gated0, gated1, gated2);
  message_gate_rows_scales(x, message, p, g, first, count, lane, sg, shared);
  message_gate_rows_down(x, message, message_gate_pick(g, down0, down1, down2), p, g, first, count, tid, lane, sg,
                         shared);
  message_gate_rows_gated(x, message, message_gate_pick(g, forget0, forget1, forget2),
                          message_gate_pick(g, gain0, gain1, gain2), gated, p, g, first, count, lane, sg, shared);
  message_gate_rows_norm(gated, message_gate_pick(g, norm0, norm1, norm2), message_gate_pick(g, output0, output1, output2),
                         p, first, count, tid, shared);
}

// The instances: F32 weights (_f32) and BF16 (_bf16), as a GGUF stores the
// gates' projections.
#define MESSAGE_GATE_INSTANCES(Weight, suffix)                                                                    \
  template [[host_name("message_gate_down" suffix)]] kernel void message_gate_down<Weight>(                       \
      device const bfloat *, device const Weight *, device const Weight *, device const Weight *, device bfloat *, \
      constant MessageGateFusedParams &, uint3, uint, uint, uint);                                                \
  template [[host_name("message_gate_apply" suffix)]] kernel void message_gate_apply<Weight>(                     \
      device const float *, device const float *, device const Weight *, device const Weight *,                   \
      device const Weight *, device const Weight *, device const Weight *, device const Weight *,                 \
      device const bfloat *, device const float *, device const float *, device const float *,                    \
      device coherent(device) float *, device coherent(device) float *, device atomic_uint *, device bfloat *,    \
      device bfloat *, device bfloat *, constant MessageGateFusedParams &, uint3, uint, uint, uint);              \
  template [[host_name("message_gate_rows" suffix)]] kernel void message_gate_rows<Weight>(                       \
      device const float *, device const float *, device const Weight *, device const Weight *,                   \
      device const Weight *, device const Weight *, device const Weight *, device const Weight *,                 \
      device const Weight *, device const Weight *, device const Weight *, device const float *,                  \
      device const float *, device const float *, device float *, device float *, device float *,                 \
      device bfloat *, device bfloat *, device bfloat *, constant MessageGateFusedParams &, uint3, uint, uint,     \
      uint);
MESSAGE_GATE_INSTANCES(float, "_f32")
MESSAGE_GATE_INSTANCES(bfloat, "_bf16")
#undef MESSAGE_GATE_INSTANCES
// ---- end runtime/metal/kernels/shared/message_gate.metal ----
