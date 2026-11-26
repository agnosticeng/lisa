// Vendored from Splash (https://github.com/incoai/splash), revision
// 2afc7337aacc044722fa95e4cb3da75c281a19a8, Apache License 2.0 (LICENSE-APACHE-2.0).
// Quantized GGUF decoding follows ggml (MIT, LICENSE-GGML). Each original
// header is inlined once at its first include, between begin/end markers; the
// code is otherwise unchanged. Source hashes: provenance.json.
// ---- begin runtime/metal/kernels/decode/sampling.metal (Splash 2afc733, Apache-2.0) ----
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

// The rows of the target policy. A greedy lane takes each row's argmax. A
// sampled lane draws from each row's min-p/top-k/top-p distribution over the
// whole vocabulary: a sharded scan sums the row's softmax denominator, one
// group per row finds the last token the distribution keeps in the order of
// the logits (logit descending, id ascending) without sorting the
// vocabulary, and the draw takes one pass over the kept tokens in id order,
// shared by the row's kVocabularyGroups groups: each sums ranges of the
// vocabulary, and the one that finishes last (split_arrive_last) draws.
// Every reduction runs in a fixed order, so a row selects the same token on
// every run. The same kernels select the first token after a prompt and the
// verify rows (TargetSamplingParams): each dispatch is a grid of a lane's rows
// by lanes, and the groups of the other policy's lanes, and those past a
// lane's rows, return at once.

// Whether `token` is one of a target's stop tokens (metal/abi/Sampling.h).
// SPLASH_NO_TOKEN, which pads them, is none, also where acceptance retains
// it as the selection of a row with no selectable logit.
inline bool stop_token(constant uint *stop_tokens, uint token) {
  bool stop = false;
  for (uint place = 0; place < SPLASH_STOP_TOKENS; ++place)
    stop |= token == stop_tokens[place];
  return stop && token != SPLASH_NO_TOKEN;
}

// The row a lane selects from: its logits, the tokens it admits (its
// constraint mask row, less the stop tokens when the lane ignores
// end-of-sequence) and, for a sampled row whose maximum is known, their
// softmax weights.
struct TargetRow {
  device const float *logits;
  device const uint *mask;
  bool constrained;
  bool exclude_stop;
  constant uint *stop_tokens;
  uint vocabulary;
  float temperature;
  float maximum;

  bool admits(uint token) const {
    if (constrained && (mask[token / 32] & (1u << (token % 32))) == 0)
      return false;
    return !exclude_stop || !stop_token(stop_tokens, token);
  }
  float weight(float value) const {
    return exp((value - maximum) / temperature);
  }
};

// Row `index` of lane `lane` of a dispatch (TargetSamplingParams): whether
// the lane selects it, one of its rows, and takes the policy the kernel runs
// (sampled or greedy), and its logits row s, which indexes the workspaces
// and outputs.
inline bool selects(constant TargetSamplingParams &params,
                    device const SplashLaneRows *lane_rows, uint lane, uint index,
                    bool sampled) {
  return index < lane_rows[lane].rows &&
         ((params.sampling_mask & (1u << lane)) != 0) == sampled;
}
inline uint selected_logits_row(device const SplashLaneRows *lane_rows,
                                uint lane, uint index) {
  return lane_rows[lane].first + index;
}

// Row `index` of lane `lane` of a dispatch (TargetSamplingParams).
inline TargetRow selected_row(device const float *logits,
                              device const uint *token_mask,
                              constant TargetSamplingParams &params,
                              device const SplashLaneRows *lane_rows, uint lane,
                              uint index) {
  return {logits + ulong(selected_logits_row(lane_rows, lane, index)) *
                       params.vocabulary,
          token_mask + (ulong(lane) * (SPLASH_TARGET_VERIFY_ROWS + 1) +
                        params.mask_row + index) *
                           params.mask_words,
          (params.constrained_mask & (1u << lane)) != 0,
          (params.exclude_stop_mask & (1u << lane)) != 0,
          params.stop_tokens,
          params.vocabulary,
          params.temperature[lane],
          0.0f};
}

// Each lane's lane rows and tree row for the selection of its first token
// after its prompt (SampleFirstRowsParams), one group a lane: its one row of
// the logits, a tree's root, which the selection's kernels read as a decode
// cycle's.
kernel void decode_sample_first_rows(
    device SplashLaneRows *lane_rows [[buffer(0)]],
    device SplashTreeRow *tree [[buffer(1)]],
    constant SampleFirstRowsParams &params [[buffer(2)]],
    uint lane [[threadgroup_position_in_grid]]) {
  lane_rows[lane] = {params.row[lane], 1};
  tree[params.row[lane]] = {1u, SPLASH_TREE_ROOT, 0};
}

// Shares of a row's softmax denominator combined in order: their largest
// maximum, their sums rescaled to it and their admitted counts. A group
// combines its simdgroups' shares, and a row's search its shards'.
template <class Shares>
inline TargetShardMass merge_masses(Shares shares, uint count,
                                    float temperature) {
  TargetShardMass total{-FLT_MAX, 0.0f, 0};
  for (uint index = 0; index < count; ++index)
    total.maximum = max(total.maximum, shares[index].maximum);
  for (uint index = 0; index < count; ++index) {
    const TargetShardMass share = shares[index];
    if (share.sum > 0.0f)
      total.sum +=
          share.sum * exp((share.maximum - total.maximum) / temperature);
    total.admitted += share.admitted;
  }
  return total;
}

// One shard's share of a sampled row's softmax denominator: each thread
// keeps a running maximum and sum of its admitted tokens' weights (an online
// softmax), combined over the group in a fixed order. The running maximum
// starts below every finite logit, so a -inf logit weighs nothing.
inline void shard_mass(TargetRow row, uint shard,
                       device TargetShardMass &partial,
                       threadgroup TargetShardMass *group_masses,
                       uint thread_index, uint lane, uint simd_group) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  TargetShardMass mass{-FLT_MAX, 0.0f, 0};
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    const float value = row.logits[token];
    if (value > mass.maximum) {
      mass.sum =
          mass.sum * exp((mass.maximum - value) / row.temperature) + 1.0f;
      mass.maximum = value;
    } else {
      mass.sum += exp((value - mass.maximum) / row.temperature);
    }
    ++mass.admitted;
  }
  const float simd_maximum = simd_max(mass.maximum);
  const float scaled =
      mass.sum > 0.0f
          ? mass.sum * exp((mass.maximum - simd_maximum) / row.temperature)
          : 0.0f;
  const float simd_total = simd_sum(scaled);
  const uint simd_admitted = simd_sum(mass.admitted);
  if (lane == 0)
    group_masses[simd_group] = {simd_maximum, simd_total, simd_admitted};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    partial = merge_masses(group_masses, 8, row.temperature);
}

// One shard of each selected row of the sampled lanes.
kernel void decode_sample_mass_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device TargetShardMass *partial_masses [[buffer(2)]],
    device const SplashLaneRows *lane_rows [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup TargetShardMass group_masses[8];
  const uint batch = group.y, index = group.x / SPLASH_TARGET_SAMPLING_SHARDS;
  if (!selects(params, lane_rows, batch, index, true))
    return;
  const uint shard = group.x % SPLASH_TARGET_SAMPLING_SHARDS;
  shard_mass(selected_row(logits, token_mask, params, lane_rows, batch, index), shard,
             partial_masses[ulong(selected_logits_row(lane_rows, batch, index)) *
                                SPLASH_TARGET_SAMPLING_SHARDS +
                            shard],
             group_masses, thread_index, lane, simd_group);
}

// The groups that search and draw a sampled row.
constant constexpr uint kVocabularyThreads = SPLASH_TARGET_VOCABULARY_THREADS;
constant constexpr uint kVocabularySimdgroups = kVocabularyThreads / 32;
constant constexpr uint kVocabularyGroups = SPLASH_TARGET_VOCABULARY_GROUPS;
// The draw's ranges of the vocabulary; the last group loads them one per
// thread.
constant constexpr uint kVocabularyRanges = SPLASH_TARGET_VOCABULARY_RANGES;
static_assert(kVocabularyRanges <= kVocabularyThreads,
              "a group holds one draw range per thread");
// Pivots per pass: each pass splits a bracket sixteen ways. In logit order
// kKeyPivots of them split its keys evenly and the others follow the logits'
// scale (place_pivots).
constant constexpr uint kPivots = 15;
constant constexpr uint kKeyPivots = 4;

// The order of the logits as an unsigned key: a larger logit has a larger
// key, and -0 and +0, which compare equal, share one. Every logit's key is
// above kNoKey.
inline uint logit_key(float value) {
  uint bits = as_type<uint>(value);
  bits = (bits << 1) ? bits : 0u;
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float key_logit(uint key) {
  return as_type<float>((key & 0x80000000u) ? key & 0x7fffffffu : ~key);
}
constant constexpr uint kNoKey = 0u;
// The key of -infinity, at or below that of every admitted logit.
constant constexpr uint kLowestKey = 0x007fffffu;

// A position in the order of the logits: the tokens at or before it have a
// larger key, or the same key and an id at most last. {kNoKey, 0} lies
// after every token.
struct OrderBoundary {
  uint key;
  uint last;
};

inline bool at_or_before(uint key, uint token, OrderBoundary boundary) {
  return key > boundary.key || (key == boundary.key && token <= boundary.last);
}

// The orders a search runs in. The logit order is the order of the logits.
// Among the tokens tied at one logit, the tie order puts a smaller id first;
// every other token has kNoKey.
struct LogitOrder {
  static constexpr constant bool by_logit = true;
  uint key(float value, uint) const { return logit_key(value); }
  float logit(uint key) const { return key_logit(key); }
};
struct TieOrder {
  uint tied_key;
  uint vocabulary;
  uint key(float value, uint token) const {
    return logit_key(value) == tied_key ? vocabulary - token : kNoKey;
  }
  static constexpr constant bool by_logit = false;
  float logit(uint) const { return key_logit(tied_key); }
};

// What a search keeps: the tokens in order until their count, or their mass
// (the sum of their weights), exceeds the target. A count search measures
// counts alone (top-k, and the tie order's first ids), a mass search counts
// and masses (top-p).
struct CountTarget {
  static constexpr constant bool by_mass = false;
  uint count;

  bool exceeded(uint measured_count, float) const {
    return measured_count > count;
  }
};
struct MassTarget {
  static constexpr constant bool by_mass = true;
  float mass;

  bool exceeded(uint, float measured_mass) const {
    return measured_mass > mass;
  }
};

// A key range [lo, hi) and the count and mass of the tokens with at least
// each end's key: lo's exceed the search target, hi's do not. A count
// search narrows the counts alone and leaves the masses as they came in.
struct Bracket {
  uint lo;
  uint hi;
  uint lo_count;
  uint hi_count;
  float lo_mass;
  float hi_mass;

  uint tokens() const { return lo_count - hi_count; }
};

// The last token a search keeps, and the count and mass of the tokens at or
// before it. A ranked selection leaves those tokens in the scratch keys and
// ids, in order (resolve).
struct Selection {
  OrderBoundary last;
  uint count;
  float mass;
  bool ranked;
};

struct VocabularyScratch {
  uint counts[kVocabularySimdgroups][kPivots];
  float masses[kVocabularySimdgroups][kPivots];
  uint total_counts[kPivots];
  float total_masses[kPivots];
  atomic_uint gathered;
  uint keys[kVocabularyThreads];
  uint ids[kVocabularyThreads];
  Selection selection;
};

struct DrawScratch {
  uint arrival;
  // Per range of the draw: the kept weight of its tokens and one past the
  // last of them with weight.
  float range_weights[kVocabularyRanges];
  uint range_last[kVocabularyRanges];
  uint drawn_range;
  float drawn_target;
  uint drawn_token;
  // Whether the row selected a candidate, and whether its draw excludes its
  // candidates, which it then holds (vocabulary_draw).
  bool selected;
  bool excluding;
  uint excluded[SPLASH_DRAFT_CANDIDATES];
};

// Pivots inside [lo, hi) that split the bracket. The tie order splits its
// keys evenly. In logit order the last kKeyPivots do, so a pass keeps at
// most a fifth of the bracket's keys, rounded up, and a search ends within
// fourteen passes whatever the logits, temperature or penalties; the others
// follow the logits' scale, which ends most searches within a few: geometric
// distances below the bracket's top, in units of the temperature, while it
// is open at the bottom, and even steps in logit value once it is not.
template <class Order>
inline void place_pivots(Order order, Bracket bracket, float temperature,
                         thread uint (&pivots)[kPivots]) {
  const ulong width = bracket.hi - bracket.lo;
  if (!order.by_logit) {
    for (uint pivot = 0; pivot < kPivots; ++pivot)
      pivots[pivot] = bracket.lo + uint(width * (pivot + 1) / (kPivots + 1));
    return;
  }
  constexpr uint kScaled = kPivots - kKeyPivots;
  const float top = key_logit(bracket.hi - 1);
  const float bottom = key_logit(bracket.lo);
  const bool open = bracket.lo == kLowestKey;
  for (uint pivot = 0; pivot < kScaled; ++pivot) {
    const float value =
        open ? top - temperature * 0.5f * exp2(float(kScaled - 1 - pivot))
             : bottom + (top - bottom) * float(pivot + 1) / float(kScaled + 1);
    pivots[pivot] = clamp(logit_key(value), bracket.lo, bracket.hi - 1);
  }
  for (uint pivot = 0; pivot < kKeyPivots; ++pivot)
    pivots[kScaled + pivot] =
        bracket.lo + uint(width * (pivot + 1) / (kKeyPivots + 1));
}

// The count and, with Masses, the mass of the admitted tokens at or before
// the floor whose keys are at least each pivot's, summed per thread in token
// order and then over the group in a fixed order. Counts alone weigh no
// token.
template <bool Masses, uint Pivots, class Order>
inline void measure_pivots(TargetRow row, Order order, OrderBoundary floor,
                           thread const uint (&pivots)[Pivots],
                           threadgroup VocabularyScratch &scratch,
                           uint thread_index, uint lane, uint simd_group) {
  uint counts[Pivots];
  float masses[Pivots];
  uint lowest = pivots[0];
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    counts[pivot] = 0;
    if constexpr (Masses)
      masses[pivot] = 0.0f;
    lowest = min(lowest, pivots[pivot]);
  }
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    // Tokens below every pivot add nothing.
    if (key < lowest || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if constexpr (Masses) {
      const float weight = row.weight(value);
      for (uint pivot = 0; pivot < Pivots; ++pivot) {
        const bool above = key >= pivots[pivot];
        counts[pivot] += above ? 1u : 0u;
        masses[pivot] += above ? weight : 0.0f;
      }
    } else {
      for (uint pivot = 0; pivot < Pivots; ++pivot)
        counts[pivot] += key >= pivots[pivot] ? 1u : 0u;
    }
  }
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    const uint count = simd_sum(counts[pivot]);
    if (lane == 0)
      scratch.counts[simd_group][pivot] = count;
    if constexpr (Masses) {
      const float mass = simd_sum(masses[pivot]);
      if (lane == 0)
        scratch.masses[simd_group][pivot] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < Pivots) {
    uint count = 0;
    for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
      count += scratch.counts[simd][thread_index];
    scratch.total_counts[thread_index] = count;
    if constexpr (Masses) {
      float mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][thread_index];
      scratch.total_masses[thread_index] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The count and mass of the admitted tokens at or before the floor whose
// logits' keys are at least key, summed as measure_pivots sums a pivot's.
struct Measure {
  uint count;
  float mass;
};
inline Measure measure_at_or_above(TargetRow row, OrderBoundary floor,
                                   uint key,
                                   threadgroup VocabularyScratch &scratch,
                                   uint thread_index, uint lane,
                                   uint simd_group) {
  const uint pivots[1] = {key};
  measure_pivots<true>(row, LogitOrder{}, floor, pivots, scratch,
                       thread_index, lane, simd_group);
  const Measure measure{scratch.total_counts[0], scratch.total_masses[0]};
  // Every thread reads the totals before the next pass rewrites them.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return measure;
}

// Narrows the bracket until it holds at most one token per thread or a
// single key.
template <class Order, class Target>
inline Bracket narrow(TargetRow row, Order order, OrderBoundary floor,
                      Bracket bracket, Target target,
                      threadgroup VocabularyScratch &scratch,
                      uint thread_index, uint lane, uint simd_group) {
  while (bracket.tokens() > kVocabularyThreads &&
         bracket.hi - bracket.lo > 1) {
    uint pivots[kPivots];
    place_pivots(order, bracket, row.temperature, pivots);
    measure_pivots<Target::by_mass>(row, order, floor, pivots, scratch,
                                    thread_index, lane, simd_group);
    // The highest pivot whose measure exceeds the target and the lowest one
    // whose measure does not bound the new bracket. A pivot at lo never
    // becomes hi: lo's measure may come from another sum (the shards' masses,
    // or the top-k selection's), and rounding must not empty the bracket.
    for (uint pivot = 0; pivot < kPivots; ++pivot) {
      const uint count = scratch.total_counts[pivot];
      const float mass = Target::by_mass ? scratch.total_masses[pivot] : 0.0f;
      if (target.exceeded(count, mass)) {
        if (pivots[pivot] >= bracket.lo) {
          bracket.lo = pivots[pivot];
          bracket.lo_count = count;
          if (Target::by_mass)
            bracket.lo_mass = mass;
        }
      } else if (pivots[pivot] > bracket.lo && pivots[pivot] < bracket.hi) {
        bracket.hi = pivots[pivot];
        bracket.hi_count = count;
        if (Target::by_mass)
          bracket.hi_mass = mass;
      }
    }
    // Every thread reads the totals before the next pass rewrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bracket;
}

// The last token the target keeps in a bracket of at most one token per
// thread: its tokens are gathered, put in order by rank, and added to the
// measure at hi one at a time until it exceeds the target. A count search in
// logit order measures no masses while it narrows: its gather sums the mass
// at hi as measure_pivots would have, or, when every token from lo up fits
// the group, gathers them all and adds them from the first, which leaves the
// tokens it keeps ranked in the scratch (Selection::ranked).
template <class Order, class Target>
inline Selection resolve(TargetRow row, Order order, OrderBoundary floor,
                         Bracket bracket, Target target,
                         threadgroup VocabularyScratch &scratch,
                         uint thread_index, uint lane, uint simd_group) {
  constexpr bool unmeasured = Order::by_logit && !Target::by_mass;
  const bool ranked = unmeasured && bracket.lo_count <= kVocabularyThreads;
  const bool measures = unmeasured && !ranked;
  if (thread_index == 0)
    atomic_store_explicit(&scratch.gathered, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float above = 0.0f;
  for (uint token = thread_index; token < row.vocabulary;
       token += kVocabularyThreads) {
    const float value = row.logits[token];
    const uint key = order.key(value, token);
    if (key < bracket.lo || !row.admits(token) ||
        !at_or_before(logit_key(value), token, floor))
      continue;
    if (!ranked && key >= bracket.hi) {
      if (measures)
        above += row.weight(value);
      continue;
    }
    const uint slot = atomic_fetch_add_explicit(&scratch.gathered, 1u,
                                                memory_order_relaxed);
    if (slot < kVocabularyThreads) {
      scratch.keys[slot] = key;
      scratch.ids[slot] = token;
    }
  }
  if (measures) {
    above = simd_sum(above);
    if (lane == 0)
      scratch.masses[simd_group][0] = above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gathered = min(atomic_load_explicit(&scratch.gathered,
                                                 memory_order_relaxed),
                            kVocabularyThreads);
  uint key = 0;
  uint id = 0;
  uint rank = 0;
  if (thread_index < gathered) {
    key = scratch.keys[thread_index];
    id = scratch.ids[thread_index];
    for (uint other = 0; other < gathered; ++other) {
      const uint other_key = scratch.keys[other];
      rank += other_key > key || (other_key == key && scratch.ids[other] < id)
                  ? 1u
                  : 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < gathered) {
    scratch.keys[rank] = key;
    scratch.ids[rank] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    uint count = ranked ? 0 : bracket.hi_count;
    float mass = ranked ? 0.0f : bracket.hi_mass;
    if (measures) {
      mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][0];
    }
    // Rounding may leave a mass target unreached: keep the whole bracket.
    Selection selection{{bracket.lo, 0xffffffffu},
                        bracket.lo_count,
                        Target::by_mass ? bracket.lo_mass : mass,
                        ranked};
    for (uint index = 0; index < gathered; ++index) {
      count += 1;
      mass += row.weight(order.logit(scratch.keys[index]));
      selection = {{scratch.keys[index], scratch.ids[index]}, count, mass,
                   ranked};
      if (target.exceeded(count, mass))
        break;
    }
    scratch.selection = selection;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch.selection;
}

// The last token the target keeps among the admitted tokens at or before
// the floor, starting from a bracket in logit order.
template <class Target>
inline Selection select_last(TargetRow row, OrderBoundary floor,
                             Bracket bracket, Target target,
                             threadgroup VocabularyScratch &scratch,
                             uint thread_index, uint lane, uint simd_group) {
  bracket = narrow(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  if (bracket.tokens() <= kVocabularyThreads)
    return resolve(row, LogitOrder{}, floor, bracket, target, scratch,
                   thread_index, lane, simd_group);
  // More tokens tie at one logit than a group gathers: the target keeps its
  // first ones by id, each of the same weight, after the mass above them,
  // which a count search measures in one more pass.
  const uint ties = bracket.tokens();
  const float weight = row.weight(key_logit(bracket.lo));
  float above;
  uint keep;
  if constexpr (Target::by_mass) {
    above = bracket.hi_mass;
    keep = 1u + uint(clamp((target.mass - above) / weight, 0.0f,
                           float(ties - 1)));
  } else {
    above = measure_at_or_above(row, floor, bracket.hi, scratch, thread_index,
                                lane, simd_group)
                .mass;
    keep = target.count + 1 - bracket.hi_count;
  }
  uint last = bracket.lo == floor.key ? floor.last : 0xffffffffu;
  if (keep < ties) {
    const TieOrder order{bracket.lo, row.vocabulary};
    const CountTarget first{keep - 1};
    Bracket tied{1, row.vocabulary + 1, ties, 0, 0.0f, 0.0f};
    tied = narrow(row, order, floor, tied, first, scratch, thread_index, lane,
                  simd_group);
    last = resolve(row, order, floor, tied, first, scratch, thread_index, lane,
                   simd_group)
               .last.last;
  }
  return {{bracket.lo, last}, bracket.hi_count + keep,
          above + float(keep) * weight, false};
}

// The last token of the row's distribution: the tokens that weigh at least
// min_p of the heaviest, then the top_k of those, then within those the top_p
// nucleus, whose mass is measured against the mass of what the two cuts
// before it keep. A top-k selection the scratch holds ranked is walked for
// the nucleus in that order, with the sums it was selected with; otherwise
// the nucleus is searched for. A row that keeps every admitted token ends at
// {kNoKey, 0}.
inline OrderBoundary distribution_end(TargetRow row, float min_p, uint top_k,
                                      float top_p, float mass, uint admitted,
                                      threadgroup VocabularyScratch &scratch,
                                      uint thread_index, uint lane,
                                      uint simd_group) {
  OrderBoundary end{kNoKey, 0};
  Bracket bracket{kLowestKey, logit_key(row.maximum) + 1, admitted, 0, mass,
                  0.0f};
  if (min_p > 0.0f) {
    // A token weighs min_p of the heaviest, which weighs 1, at the logit
    // -temperature * log(min_p) below the maximum. The tokens at or above
    // that logit stay, the heaviest always.
    const uint lowest = logit_key(
        min(row.maximum + row.temperature * log(min_p), row.maximum));
    const Measure kept = measure_at_or_above(row, end, lowest, scratch,
                                             thread_index, lane, simd_group);
    if (kept.count < admitted) {
      end = {lowest, 0xffffffffu};
      bracket.lo = lowest;
      bracket.lo_count = kept.count;
      bracket.lo_mass = kept.mass;
    }
  }
  if (top_k < bracket.lo_count) {
    // A min_p cut ends on a whole key, which the bracket bounds from below,
    // so this search needs no floor.
    const Selection top = select_last(row, {kNoKey, 0}, bracket,
                                      CountTarget{top_k - 1}, scratch,
                                      thread_index, lane, simd_group);
    end = top.last;
    if (top.ranked && top_p < 1.0f) {
      // Every thread reads the selection before the walk rewrites it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (thread_index == 0) {
        // Rounding may leave the nucleus unreached: keep the whole top-k.
        const float nucleus = top_p * top.mass;
        float walked = 0.0f;
        for (uint index = 0; index < top.count; ++index) {
          walked += row.weight(key_logit(scratch.keys[index]));
          if (walked > nucleus) {
            end = {scratch.keys[index], scratch.ids[index]};
            break;
          }
        }
        scratch.selection.last = end;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      return scratch.selection.last;
    }
    bracket.lo = end.key;
    bracket.lo_count = top.count;
    bracket.lo_mass = top.mass;
  }
  if (top_p < 1.0f)
    end = select_last(row, end, bracket, MassTarget{top_p * bracket.lo_mass},
                      scratch, thread_index, lane, simd_group)
              .last;
  return end;
}

// A token's weight in the row's distribution: its softmax weight if the
// row admits it at or before the distribution's end, else 0.
inline float kept_weight(TargetRow row, OrderBoundary end, uint token) {
  if (token >= row.vocabulary || !row.admits(token))
    return 0.0f;
  const float value = row.logits[token];
  return at_or_before(logit_key(value), token, end) ? row.weight(value) : 0.0f;
}

// A sampled verify row's draft children (TargetSamplingParams): the
// candidates of its next position (ids, kDrawCandidates of them), the draft's
// probabilities of them there (q), the candidate indices of the children its
// draft drew, in the order drawn (kNoDraw past them), and a uniform for each
// child's test. A row without children draws from its whole distribution.
constant constexpr uint kDrawCandidates = SPLASH_DRAFT_CANDIDATES;
constant constexpr uchar kNoDraw = 0xff;
struct DrawChildren {
  device const uint *ids;
  device const float *q;
  device const uchar *draws;
  device const float *uniforms;
};

// The row's selection, shared by its groups; returns true in the group that
// finishes last, with the selected token. Each kept token weighs its softmax
// weight. Each simdgroup of the row's groups sums one range of the
// vocabulary; the last group adds the ranges in order, then tests the row's
// children in the order drawn, recursive rejection sampling without
// replacement: child i, of probability p_i under the target and q_i under the
// draft, is selected with probability min(1, p_i / q_i), where after each
// rejection the target's distribution becomes its residual max(p - q, 0),
// renormalized, and the draft's loses the rejected child, renormalized. A
// row that selects none draws from the last residual: from its candidates'
// residual weights, or past them from the target's own weights of the tokens
// outside its candidates, whose range it walks with prefix sums. So the
// selection is a draw from the target's distribution whatever the children,
// and a child the row rejected only at fp32 rounding: where its residual
// rounds to nothing, which leaves the draw the target before the child's
// test, or where the candidates hold all the target keeps and their ranges a
// remainder of rounding, which a draw past them lands on. A range's weight
// and its walk add the same token weights, so the range the draw picks always
// has a token to draw; a row that keeps no token draws past the vocabulary's
// end, the sentinel of a row with no selectable logit.
inline bool vocabulary_draw(
    TargetRow row, OrderBoundary end, float uniform, DrawChildren children,
    device coherent(device) TargetVocabularyRange *ranges,
    device atomic_uint *arrivals, uint slice, threadgroup DrawScratch &scratch,
    uint thread_index, uint lane, uint simd_group, thread uint &drawn) {
  const uint range_tokens = (row.vocabulary + kVocabularyRanges * 32 - 1) /
                            (kVocabularyRanges * 32) * 32;
  const uint own = slice * kVocabularySimdgroups + simd_group;
  const uint begin = min(own * range_tokens, row.vocabulary);
  const uint finish = min(begin + range_tokens, row.vocabulary);
  float kept = 0.0f;
  uint after = 0;
  for (uint first = begin; first < finish; first += 32) {
    const uint token = first + lane;
    const float weight = token < finish ? kept_weight(row, end, token) : 0.0f;
    kept += weight;
    after = weight > 0.0f ? token + 1 : after;
  }
  kept = simd_sum(kept);
  after = simd_max(after);
  if (lane == 0)
    ranges[own] = {kept, after};
  if (!split_arrive_last(arrivals, kVocabularyGroups, thread_index,
                         &scratch.arrival))
    return false;
  if (thread_index < kVocabularyRanges) {
    const TargetVocabularyRange measured = ranges[thread_index];
    scratch.range_weights[thread_index] = measured.weight;
    scratch.range_last[thread_index] = measured.after;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      total += scratch.range_weights[range];
    scratch.excluding = false;
    scratch.selected = false;
    float target = uniform * total;
    if (children.ids && children.draws[0] != kNoDraw && total > 0.0f) {
      // The target's and the draft's probabilities of the candidates, and
      // the target's mass outside them: what the ranges weigh once the
      // candidates' weights leave them.
      float p[kDrawCandidates], q[kDrawCandidates];
      for (uint i = 0; i < kDrawCandidates; ++i) {
        const uint id = children.ids[i];
        const float weight = kept_weight(row, end, id);
        // A candidate past the vocabulary, as the draft's top 16 lists one
        // (0xffffffff) for a row of fewer comparable logits, weighs nothing
        // and lies in no range.
        if (id < row.vocabulary) {
          const uint range = id / range_tokens;
          scratch.range_weights[range] = max(scratch.range_weights[range] - weight, 0.0f);
        }
        scratch.excluded[i] = id;
        p[i] = weight / total;
        q[i] = children.q[i];
      }
      float outside_weight = 0.0f;
      for (uint range = 0; range < kVocabularyRanges; ++range)
        outside_weight += scratch.range_weights[range];
      float outside = outside_weight / total;
      for (uint child = 0; child < SPLASH_TREE_CHILDREN && !scratch.selected; ++child) {
        const uint drawn_child = children.draws[child];
        if (drawn_child == kNoDraw)
          break;
        if (children.uniforms[child] * q[drawn_child] < p[drawn_child]) {
          scratch.selected = true;
          scratch.drawn_token = children.ids[drawn_child];
          break;
        }
        // The target's residual, and the draft without the rejected child.
        float residual = outside;
        for (uint i = 0; i < kDrawCandidates; ++i)
          residual += max(p[i] - q[i], 0.0f);
        if (!(residual > 0.0f))
          break;
        for (uint i = 0; i < kDrawCandidates; ++i)
          p[i] = max(p[i] - q[i], 0.0f) / residual;
        outside /= residual;
        // The draft's rest, summed: one less a child that holds nearly all of
        // q_T would cancel to nothing in fp32.
        q[drawn_child] = 0.0f;
        float rest = 0.0f;
        for (uint i = 0; i < kDrawCandidates; ++i)
          rest += q[i];
        for (uint i = 0; rest > 0.0f && i < kDrawCandidates; ++i)
          q[i] /= rest;
      }
      if (!scratch.selected) {
        // The last residual's draw: a candidate, or past them a token of the
        // target's own weights outside them. Rounding may leave the point
        // unreached: the last candidate with weight, where none lies outside.
        float inside = 0.0f;
        for (uint i = 0; i < kDrawCandidates; ++i)
          inside += p[i];
        const float point = uniform * (inside + outside);
        float cumulative = 0.0f;
        uint last = kDrawCandidates;
        for (uint i = 0; i < kDrawCandidates && !(point < cumulative); ++i) {
          if (p[i] > 0.0f) {
            last = i;
            cumulative += p[i];
          }
        }
        if (last < kDrawCandidates && (point < cumulative || !(outside > 0.0f))) {
          scratch.selected = true;
          scratch.drawn_token = children.ids[last];
        } else {
          scratch.excluding = true;
          target = (point - inside) / outside * outside_weight;
        }
      }
    }
    if (!scratch.selected) {
      float cumulative = 0.0f;
      uint range_drawn = 0;
      float range_target = 0.0f;
      for (uint range = 0; range < kVocabularyRanges; ++range) {
        const float weight = scratch.range_weights[range];
        if (!(weight > 0.0f))
          continue;
        // Rounding may leave the target unreached: the last range with
        // weight.
        range_drawn = range;
        range_target = target - cumulative;
        cumulative += weight;
        if (cumulative > target)
          break;
      }
      scratch.drawn_range = range_drawn;
      scratch.drawn_target = range_target;
      // Rounding may leave the target unreached in the range too: its last
      // token with weight, which the walk finds.
      scratch.drawn_token = scratch.range_last[range_drawn] - 1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && !scratch.selected) {
    const uint walked = scratch.drawn_range;
    const uint walk_begin = min(walked * range_tokens, row.vocabulary);
    const uint walk_finish = min(walk_begin + range_tokens, row.vocabulary);
    const float target = scratch.drawn_target;
    float cumulative = 0.0f;
    uint last = 0;
    bool hit_found = false;
    for (uint first = walk_begin; first < walk_finish && !hit_found; first += 32) {
      const uint token = first + lane;
      bool excluded = false;
      for (uint i = 0; scratch.excluding && i < kDrawCandidates; ++i)
        excluded = excluded || scratch.excluded[i] == token;
      const float weight =
          token < walk_finish && !excluded ? kept_weight(row, end, token) : 0.0f;
      last = max(last, simd_max(weight > 0.0f ? token + 1 : 0u));
      const float prefix = simd_prefix_inclusive_sum(weight);
      const uint hit =
          simd_min(weight > 0.0f && cumulative + prefix > target ? lane : 32u);
      if (hit < 32) {
        if (lane == 0)
          scratch.drawn_token = first + hit;
        hit_found = true;
      }
      cumulative += simd_broadcast(prefix, 31);
    }
    // Excluding candidates, the range's last token with weight may be one of
    // them: the last the walk weighed.
    if (!hit_found && scratch.excluding && lane == 0 && last)
      scratch.drawn_token = last - 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  drawn = scratch.drawn_token;
  split_release(arrivals, thread_index);
  return true;
}

// Where a sampled row's distribution ends: its shards' masses, merged in
// shard order into the row's largest admitted logit, softmax denominator and
// admitted count, then the search. The record keeps the maximum and the
// end, which the draw reads.
inline void search_row(TargetRow row, device const TargetShardMass *masses,
                       float min_p, uint top_k, float top_p,
                       device TargetVocabularyRow &record,
                       threadgroup VocabularyScratch &scratch,
                       uint thread_index, uint lane, uint simd_group) {
  const TargetShardMass merged =
      merge_masses(masses, SPLASH_TARGET_SAMPLING_SHARDS, row.temperature);
  row.maximum = merged.maximum;
  const OrderBoundary end =
      distribution_end(row, min_p, top_k, top_p, merged.sum, merged.admitted,
                       scratch, thread_index, lane, simd_group);
  if (thread_index == 0)
    record = {merged.maximum, end.key, end.last};
}

// Where the distribution of each selected row of the sampled lanes ends, one
// group per row.
kernel void decode_sample_vocabulary_search(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetShardMass *partial_masses [[buffer(2)]],
    device TargetVocabularyRow *vocabulary_rows [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant TargetSamplingParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup VocabularyScratch scratch;
  const uint batch = group.y, index = group.x;
  if (!selects(params, lane_rows, batch, index, true))
    return;
  const uint s = selected_logits_row(lane_rows, batch, index);
  search_row(selected_row(logits, token_mask, params, lane_rows, batch, index),
             partial_masses + ulong(s) * SPLASH_TARGET_SAMPLING_SHARDS,
             params.min_p[batch], params.top_k[batch], params.top_p[batch],
             vocabulary_rows[s], scratch, thread_index, lane, simd_group);
}

// The selection of each selected row of the sampled lanes, kVocabularyGroups
// groups per row: row i of a lane draws with the lane's uniform i, and a
// decode step's verify row (params.trees) tests the children its lane's
// draft drew there (sources, its row of the lane's tree, whose draws and next
// position's probabilities the selector wrote, and that position's
// candidates) with the lane's uniforms SPLASH_TARGET_VERIFY_ROWS +
// i * SPLASH_TREE_CHILDREN on.
kernel void decode_sample_vocabulary_draw(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]],
    device coherent(device) TargetVocabularyRange *ranges [[buffer(5)]],
    device atomic_uint *arrivals [[buffer(6)]],
    device const SplashLaneRows *lane_rows [[buffer(7)]],
    device const SplashTreeRow *tree [[buffer(8)]],
    device const uint *sources [[buffer(9)]],
    device const uint *candidates [[buffer(10)]],
    device const uchar *draws [[buffer(11)]],
    device const float *next [[buffer(12)]],
    constant TargetSamplingParams &params [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup DrawScratch scratch;
  const uint batch = group.y, index = group.x / kVocabularyGroups;
  if (!selects(params, lane_rows, batch, index, true))
    return;
  const uint s = selected_logits_row(lane_rows, batch, index);
  const TargetVocabularyRow record = vocabulary_rows[s];
  TargetRow row = selected_row(logits, token_mask, params, lane_rows, batch, index);
  row.maximum = record.maximum;
  device const float *lane_uniforms = uniforms + ulong(batch) * SPLASH_SAMPLING_UNIFORMS;
  DrawChildren children{nullptr, nullptr, nullptr, nullptr};
  if (params.trees) {
    const uint source = sources[s], depth = tree[s].depth;
    if (source != SPLASH_NO_TOKEN && depth < SPLASH_DRAFT_PROPOSAL_TOKENS) {
      const ulong at = ulong(batch) * SPLASH_TARGET_VERIFY_ROWS + source;
      children = {candidates + (ulong(batch) * SPLASH_DRAFT_PROPOSAL_TOKENS + depth) * kDrawCandidates,
                  next + at * kDrawCandidates, draws + at * SPLASH_TREE_CHILDREN,
                  lane_uniforms + SPLASH_TARGET_VERIFY_ROWS + index * SPLASH_TREE_CHILDREN};
    }
  }
  uint token;
  if (vocabulary_draw(row, {record.end_key, record.end_last}, lane_uniforms[index], children,
                      ranges + ulong(s) * kVocabularyRanges, arrivals + s,
                      group.x % kVocabularyGroups, scratch, thread_index, lane,
                      simd_group, token) &&
      thread_index == 0)
    tokens[s] = token;
}

// Whether no token can be selected for a logit: NaN has no order and -inf
// no probability. A bit test, which Metal's fast math keeps.
inline bool unselectable_logit(float value) {
  const uint bits = as_type<uint>(value);
  return (bits & 0x7fffffffu) > 0x7f800000u || bits == 0xff800000u;
}

// The sampling penalties of one logit: repetition divides a positive
// logit and multiplies a negative one when the prompt or the output holds
// the token, and presence and frequency lower it by the output's count of
// the token. The result saturates, so no rewritten logit is infinite; an
// unselectable logit stays as it is, since saturating would make it
// selectable.
inline float penalize_logit(float value, uint count, float repetition,
                            float repetition_inverse, float presence,
                            float frequency) {
  if (unselectable_logit(value))
    return value;
  value *= value > 0.0f ? repetition_inverse : repetition;
  if (count)
    value -= frequency * float(count) + presence;
  return clamp(value, -FLT_MAX, FLT_MAX);
}

// The selected rows of each penalized entry, one thread per vocabulary token
// and entry, rewritten in place. Their contexts add the entry's draft tokens
// along their paths: bit r of drafted is set when the entry's verify input
// row r (from its first row) holds the token, a draft token that the rows
// below row r in its block's tree add to the output, so a row counts the bits
// of its ancestors but its anchor (metal/abi/VerifyTree.h) on top of the
// table's count. A selection of one row (the first token after a prompt) adds
// none.
kernel void decode_sample_penalize(
    device float *logits [[buffer(0)]], device const uint *words [[buffer(1)]],
    device const uint *input_tokens [[buffer(2)]],
    device const SplashTreeRow *tree [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant SamplingPenaltyParams &params [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint token = position.x;
  const uint entry = position.y;
  if (token >= params.vocabulary || entry >= params.entries)
    return;
  const SplashLaneRows lane = lane_rows[params.lane[entry]];
  const uint rows = lane.rows;
  device const uint *inputs = input_tokens + lane.first;
  uint drafted = 0;
  for (uint row = 1; row < rows; ++row)
    drafted |= uint(inputs[row] == token) << row;
  const uint word =
      words[ulong(params.table_row[entry]) * params.vocabulary + token];
  if (!word && !drafted)
    return;
  device float *column = logits + ulong(lane.first) * params.vocabulary + token;
  for (uint row = 0; row < rows; ++row) {
    const uint path = tree[lane.first + row].ancestors;
    const uint count = (word & SPLASH_PENALTY_COUNT_MASK) + popcount(drafted & path & ~1u);
    if (!count && !(word & SPLASH_PENALTY_PROMPT_BIT))
      continue;
    device float &logit = column[ulong(row) * params.vocabulary];
    logit = penalize_logit(logit, count, params.repetition[entry],
                           params.repetition_inverse[entry],
                           params.presence[entry], params.frequency[entry]);
  }
}

inline bool top_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert for an entry the caller has already
// checked against the last slot; the unrolled shift keeps every index static.
inline void top16_insert(thread float (&values)[16], thread uint (&ids)[16],
                         float value, uint token) {
#pragma clang loop unroll(full)
  for (uint slot = 15; slot > 0; --slot) {
    bool here = top_beats(value, token, values[slot], ids[slot]);
    bool above = top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// Pops the head of a register-resident sorted list of Count entries.
template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the
// empty sentinel (-inf, ~0u) resolve the same way everywhere.
inline void simd_best_head(float value, uint token, thread float &best,
                           thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

// One shard of one proposal row. Threads stream their share of the shard in
// 16-byte vectors, keep the chunk in registers, and first find the 16th
// largest of the per-thread maxima: at least sixteen tokens are that large,
// so nothing below it can be in the row's top-16 and the exact sorted insert
// only runs for the few survivors. The group then pops its best sixteen in
// rank order. The (value desc, id asc) order is total, so the partial is the
// same set in the same order whatever the thread partition; it ranks the
// row's columns and writes their tokens (SelectorBatchParams), whose order
// is theirs.
kernel void draft_select_top16_sharded(
    device const float *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant SelectorBatchParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint K = SPLASH_DRAFT_CANDIDATES;
  constexpr uint VectorTokens = 4, ChunkVectors = 16;
  const uint vocabulary = params.columns;
  uint batch = group / (Positions * Shards);
  uint local = group % (Positions * Shards);
  uint position = local / Shards;
  uint shard = local % Shards;
  ulong row_start = (ulong(batch) * Rows + position + 1) * vocabulary;
  uint shard_tokens = (vocabulary + Shards - 1) / Shards;
  uint begin = min(shard * shard_tokens, vocabulary);
  uint end = min(begin + shard_tokens, vocabulary);
  // Vector loads require 16-byte alignment. Handle the shard's unaligned head
  // and tail with scalar inserts; the logits binding must also be aligned.
  uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) %
                   VectorTokens);
  head = min(head, end - begin);
  uint vectors = (end - begin - head) / VectorTokens;
  uint vector_begin = begin + head;
  uint tail_begin = vector_begin + vectors * VectorTokens;
  device const float *row = logits + row_start;
  device const float4 *vector_row =
      reinterpret_cast<device const float4 *>(row + vector_begin);

  float values[K];
  uint ids[K];
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  if (thread_index < head) {
    uint token = begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }
  if (thread_index < end - tail_begin) {
    uint token = tail_begin + thread_index;
    float value = row[token];
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }

  threadgroup float maxima[256];
  threadgroup float thresholds[8];
  for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
    float4 loaded[ChunkVectors];
    float best = -INFINITY;
    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      loaded[i] = index < vectors ? vector_row[index] : float4(0.0f);
      if (index < vectors) {
        for (uint j = 0; j < VectorTokens; ++j)
          if (loaded[i][j] > best)
            best = loaded[i][j];
      }
    }
    // The 16th largest thread maximum: the minimum over the maxima that
    // fewer than sixteen others exceed.
    maxima[thread_index] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = 0;
    for (uint other = 0; other < 256; ++other)
      above += maxima[other] > best ? 1u : 0u;
    float candidate = simd_min(above < K ? best : INFINITY);
    if (lane == 0)
      thresholds[simd_group] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threshold = thresholds[0];
    for (uint other = 1; other < 8; ++other)
      threshold = min(threshold, thresholds[other]);

    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      if (index >= vectors)
        continue;
      uint token = vector_begin + index * VectorTokens;
      for (uint j = 0; j < VectorTokens; ++j) {
        float value = loaded[i][j];
        if (value >= threshold &&
            top_beats(value, token + j, values[K - 1], ids[K - 1]))
          top16_insert(values, ids, value, token + j);
      }
    }
  }

  // Sixteen rounds pop the group-wide best head; the simdgroup bests
  // alternate between two slots so one barrier per round suffices.
  threadgroup float round_values[2][8];
  threadgroup uint round_ids[2][8];
  for (uint rank = 0; rank < K; ++rank) {
    float head_value = values[0];
    uint head_id = ids[0];
    float simd_value;
    uint simd_id;
    simd_best_head(head_value, head_id, simd_value, simd_id);
    uint slot = rank & 1;
    if (lane == 0) {
      round_values[slot][simd_group] = simd_value;
      round_ids[slot][simd_group] = simd_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = round_values[slot][0];
    uint best_id = round_ids[slot][0];
    for (uint other = 1; other < 8; ++other) {
      float value = round_values[slot][other];
      uint token = round_ids[slot][other];
      if (top_beats(value, token, best, best_id)) {
        best = value;
        best_id = token;
      }
    }
    if (thread_index == rank) {
      // The empty sentinel stays one.
      partial_ids[group * K + rank] = best_id < params.leading || best_id == 0xffffffffu
                                          ? best_id
                                          : best_id - params.leading + params.trailing;
      partial_values[group * K + rank] = best;
    }
    if (head_value == best && head_id == best_id)
      top_pop<K>(values, ids);
  }
}

// Merges the eight shard partials of one row into the simdgroup's lanes: each
// lane holds four consecutive entries of one shard's sorted list and sixteen
// rounds pop the simdgroup-wide best into rank order, lane `rank` keeping it.
inline void top16_merge_shards(device const uint *partial_ids,
                               device const float *partial_values,
                               uint row, uint lane, thread float &value,
                               thread uint &token) {
  constexpr uint K = SPLASH_DRAFT_CANDIDATES;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    simd_best_head(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      top_pop<Entries>(values, ids);
  }
}

// One group per proposal row. Simdgroup 0 merges its candidates and unary
// scores; simdgroup 1 merges the preceding row (the anchor for row 0). The
// remaining simdgroups score the 16 x 16 predecessor/candidate edge table in
// fixed per-lane reduction order. The table follows the shard partials in
// partial-values scratch.
kernel void draft_select_edges(
    device const uint *partial_ids [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device uint *candidates [[buffer(2)]],
    device float *unary [[buffer(3)]],
    device const bfloat *hidden [[buffer(4)]],
    device const bfloat *predecessor_codebook [[buffer(5)]],
    device const bfloat *successor_codebook [[buffer(6)]],
    constant SelectorBatchParams &params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = SPLASH_DRAFT_CANDIDATES;
  constexpr uint Rank = SPLASH_DRAFT_SELECTOR_RANK;
  uint batch = row / Positions;
  uint position = row % Positions;
  threadgroup uint successors[Candidates];
  threadgroup uint predecessors[Candidates];
  if (simd_group == 0) {
    float value;
    uint token;
    top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
    if (lane < Candidates) {
      candidates[row * Candidates + lane] = token;
      unary[row * Candidates + lane] = value;
      successors[lane] = token;
    }
  } else if (simd_group == 1) {
    uint token = params.anchor[batch];
    if (position > 0) {
      float value;
      top16_merge_shards(partial_ids, partial_values, row - 1, lane, value,
                         token);
    }
    if (lane < Candidates)
      predecessors[lane] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One task is one predecessor and eight of the sixteen candidates, so a
  // simdgroup issues all its codebook loads at once instead of one cold row
  // per edge.
  constexpr uint TaskCandidates = 8, Dims = Rank / 32;
  uint tasks = (position > 0 ? Candidates : 1) * (Candidates / TaskCandidates);
  device float *table = partial_values +
                        ulong(params.lanes) * Positions * Shards * Candidates +
                        ulong(row) * Candidates * Candidates;
  device const bfloat *row_hidden =
      hidden + (ulong(batch) * Rows + position + 1) * Rank;
  for (uint task = simd_group; task < tasks; task += threads / 32) {
    uint predecessor_index = task / (Candidates / TaskCandidates);
    uint first_candidate =
        task % (Candidates / TaskCandidates) * TaskCandidates;
    // Ids are produced by the top-k selection and are always in range; the
    // clamp only keeps a corrupted id inside the codebooks.
    uint safe_predecessor =
        min(predecessors[predecessor_index], params.vocabulary - 1u);
    float context[Dims];
    float successor[TaskCandidates][Dims];
    for (uint i = 0; i < Dims; ++i) {
      uint dim = lane + i * 32;
      context[i] = float(predecessor_codebook[safe_predecessor * Rank + dim]) *
                   float(row_hidden[dim]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      uint safe_candidate =
          min(successors[first_candidate + j], params.vocabulary - 1u);
      for (uint i = 0; i < Dims; ++i)
        successor[j][i] =
            float(successor_codebook[safe_candidate * Rank + lane + i * 32]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      float score = 0.0f;
      for (uint i = 0; i < Dims; ++i)
        score += context[i] * successor[j][i];
      score = simd_sum(score);
      if (lane == 0)
        table[predecessor_index * Candidates + first_candidate + j] = score;
    }
  }
}

// The best-first tree of draft_select_tree (DraftTreeParams). An entry is a
// step the tree may take, from its parent row to a token one deeper, which
// grows the tree in the order of its score, the estimate of the path it ends,
// the earlier entry first among equal ones.
constant constexpr uint kTreeChildren = SPLASH_TREE_CHILDREN;
// A lane's rows, a verify block's at most, each a bit of its descendants'
// ancestors masks (SplashTreeRow).
constant constexpr uint kTreeRows = SPLASH_TARGET_VERIFY_ROWS;
static_assert(kTreeRows <= 32, "a lane's rows fit its ancestors' masks");
// An entry's info word: whether its token is a host chain's, its parent row,
// its proposal position (its depth less one) and its candidate's index.
constant constexpr uint kTreeChain = 1u << 31;
// A row, the anchor among them, pushes its draft children once, as a draft
// step makes it or joins it where a chain token made it, and one chain
// entry, so no more entries than these are ever pushed.
constant constexpr uint kTreeEntries = kTreeRows * (kTreeChildren + 1);
constant constexpr uchar kNoCandidate = 0xff;
// The token of a row's host chain child where it has none.
constant constexpr uint kNoChainToken = 0xffffffffu;

// A lane's draft candidates at each proposal position, the probability of
// each after each predecessor candidate (the anchor for position 0) and the
// predecessor's kTreeChildren most likely.
struct TreeLattice {
  uint ids[SPLASH_DRAFT_PROPOSAL_TOKENS][SPLASH_DRAFT_CANDIDATES];
  float probability[SPLASH_DRAFT_PROPOSAL_TOKENS][SPLASH_DRAFT_CANDIDATES]
                   [SPLASH_DRAFT_CANDIDATES];
  uchar likeliest[SPLASH_DRAFT_PROPOSAL_TOKENS][SPLASH_DRAFT_CANDIDATES]
                 [kTreeChildren];
};

struct TreeHeap {
  float score[kTreeEntries];
  uint order[kTreeEntries];
  uint info[kTreeEntries];
  uint size;
  uint pushed;
};

// A lane's tree as it grows: its rows' tokens, ancestors' masks, parents,
// depths, whether each lies on its host chain's path (its tokens from the
// anchor are the chain's first ones), draft probabilities, estimates, caps
// and keys, and of each row the children its draft drew (their candidate
// indices, in order; kNoCandidate past them) with each one's cap, and its
// next position's probabilities (DraftTreeParams).
struct TreeRows {
  uint token[kTreeRows];
  uint ancestors[kTreeRows];
  ushort parent[kTreeRows];
  ushort depth[kTreeRows];
  bool on_chain[kTreeRows];
  float q[kTreeRows];
  float estimate[kTreeRows];
  float cap[kTreeRows];
  ulong key[kTreeRows];
  uchar draw[kTreeRows][kTreeChildren];
  float left[kTreeRows][kTreeChildren];
  float next[kTreeRows][SPLASH_DRAFT_CANDIDATES];
  uint made;
};

// A new row's record of its children: none yet.
inline void tree_row_clear(threadgroup TreeRows &rows, uint row) {
  for (uint child = 0; child < kTreeChildren; ++child)
    rows.draw[row][child] = kNoCandidate;
  for (uint i = 0; i < SPLASH_DRAFT_CANDIDATES; ++i)
    rows.next[row][i] = 0.0f;
}

inline bool tree_entry_before(threadgroup TreeHeap &heap, uint a, uint b) {
  return heap.score[a] > heap.score[b] ||
         (heap.score[a] == heap.score[b] && heap.order[a] < heap.order[b]);
}

inline void tree_entry_swap(threadgroup TreeHeap &heap, uint a, uint b) {
  const float score = heap.score[a];
  const uint order = heap.order[a], info = heap.info[a];
  heap.score[a] = heap.score[b];
  heap.order[a] = heap.order[b];
  heap.info[a] = heap.info[b];
  heap.score[b] = score;
  heap.order[b] = order;
  heap.info[b] = info;
}

inline void tree_push(threadgroup TreeHeap &heap, float score, uint info) {
  uint at = heap.size++;
  heap.score[at] = score;
  heap.order[at] = heap.pushed++;
  heap.info[at] = info;
  while (at > 0 && tree_entry_before(heap, at, (at - 1) / 2)) {
    tree_entry_swap(heap, at, (at - 1) / 2);
    at = (at - 1) / 2;
  }
}

// Removes the first entry, which the caller has read.
inline void tree_pop(threadgroup TreeHeap &heap) {
  tree_entry_swap(heap, 0, --heap.size);
  for (uint at = 0;;) {
    uint first = at;
    for (uint child = 2 * at + 1; child <= 2 * at + 2; ++child)
      if (child < heap.size && tree_entry_before(heap, child, first))
        first = child;
    if (first == at)
      break;
    tree_entry_swap(heap, at, first);
    at = first;
  }
}

inline uint tree_info(bool chain, uint parent, uint position, uint index) {
  return (chain ? kTreeChain : 0u) | parent << 16 | position << 8 | index;
}

// The estimate of lane l's draft step of probability q at proposal position
// p (DraftTreeParams), q kept off 0 and 1 for its logit.
inline float tree_draft_estimate(constant DraftTreeParams &params, uint lane,
                                 float q, uint position) {
  q = clamp(q, 1.0e-6f, 1.0f - 1.0e-6f);
  const float z = params.slope[lane] * log(q / (1.0f - q)) +
                  params.intercept[lane][position];
  return 1.0f / (1.0f + exp(-z));
}

// A host chain token's agreement with the draft at its depth (0-based):
// the draft's most likely candidate there, another of its candidates, none,
// or past its positions.
inline uint tree_agreement(threadgroup TreeLattice &lattice, uint token,
                           uint depth) {
  if (depth >= SPLASH_DRAFT_PROPOSAL_TOKENS)
    return 3;
  if (token == lattice.ids[depth][0])
    return 0;
  for (uint i = 1; i < SPLASH_DRAFT_CANDIDATES; ++i)
    if (token == lattice.ids[depth][i])
      return 1;
  return 2;
}

// The rate of a lane's host chain token at depth `depth` (0-based), by its
// agreement with the draft there.
inline float tree_chain_rate(threadgroup TreeLattice &lattice,
                             device const uint *chain,
                             device const float *rates, uint depth) {
  return rates[depth * SPLASH_TREE_AGREEMENTS +
               tree_agreement(lattice, chain[depth], depth)];
}

// The splitmix64 finalizer.
inline ulong tree_hash(ulong value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
  return value ^ (value >> 31);
}

// A row's key from its parent's and its token (DraftTreeParams).
inline ulong tree_key(ulong parent, uint token) {
  return tree_hash(parent + 0x9e3779b97f4a7c15ul * (ulong(token) + 1));
}

// The standard Gumbel of candidate token `token` under a row of key `key`:
// of a uniform from the top 23 bits k of a hash of the two, (k + 0.5) / 2^23,
// which fp32 holds exactly, inside (0, 1): with 24 bits the largest rounds to
// 1, a Gumbel of infinity.
inline float tree_gumbel(ulong key, uint token) {
  const ulong value = tree_hash(key ^ 0xd1b54a32d192ed03ul * (ulong(token) + 1));
  const float uniform = (float(value >> 41) + 0.5f) * 0x1p-23f;
  return -log(-log(uniform));
}

// A draft row's entries for its children at the next position, and its
// record of them (TreeRows): a greedy lane's most likely candidates, q at
// temperature 1; a sampled lane's draws without replacement from q_T,
// proportional to q^(1/T), its Gumbel-top-k with the row's key. q_T weighs
// each candidate against the most likely, exp((log q - log q_max) / T), so
// that its mass is at least 1 at any temperature, and the draws take its
// stored values, which the target's tests read: a candidate whose weight
// underflows is never drawn. The target selects one token after the row, so
// its children's estimates sum to at most its own: each child's own estimate
// is at most its cap, what the row's host chain child (`chain_token` at
// `chain_rate`; none at a rate of 0) and the children drawn before it left of
// 1 (DraftTreeParams), and the child is estimated at most as the one before
// it, so that the tree takes a row's children in the order drawn and a cycle
// verifies a first few of them. A drawn child of the chain's token is the
// chain's child: its cap is what the others left, and it takes the more of
// its own estimate and the chain's rate. The rate is the host's, from before
// any draw, so no row is admitted by its own draws. Before the chain's child
// counted, on Qwen3.6-35B-A3B's 31K-token code prompts at one and two lanes,
// a chain's first rows of rate 0.9 or more were estimated 0.983 and 0.984 and
// retained 0.954 and 0.985 of the time (108 and 328 rows), and the draft rows
// beside them were estimated 0.202 and 0.193 and retained 0.088 and 0.018 (34
// and 56 rows): the anchor's children summed to 1.14 to 1.16.
inline void tree_expand(threadgroup TreeLattice &lattice,
                        threadgroup TreeHeap &heap, threadgroup TreeRows &rows,
                        constant DraftTreeParams &params, uint lane,
                        uint children, uint row, uint position,
                        uint predecessor, float score, uint chain_token,
                        float chain_rate) {
  threadgroup const float *q = lattice.probability[position][predecessor];
  const float temperature = params.temperature[lane];
  float top = 0.0f;
  for (uint i = 0; i < SPLASH_DRAFT_CANDIDATES; ++i)
    top = max(top, q[i]);
  float mass = 0.0f;
  for (uint i = 0; i < SPLASH_DRAFT_CANDIDATES; ++i) {
    const float weight = temperature == 0.0f || !(q[i] > 0.0f)
                             ? q[i]
                             : exp((log(q[i]) - log(top)) / temperature);
    rows.next[row][i] = weight;
    mass += weight;
  }
  for (uint i = 0; i < SPLASH_DRAFT_CANDIDATES; ++i)
    rows.next[row][i] /= mass;
  float keys[SPLASH_DRAFT_CANDIDATES];
  for (uint i = 0; temperature > 0.0f && i < SPLASH_DRAFT_CANDIDATES; ++i)
    keys[i] = rows.next[row][i] > 0.0f
                  ? log(rows.next[row][i]) + tree_gumbel(rows.key[row], lattice.ids[position][i])
                  : -INFINITY;
  uint taken = 0;
  float bound = score, left = 1.0f - chain_rate, reserved = chain_rate;
  for (uint child = 0; child < children; ++child) {
    uint drawn = SPLASH_DRAFT_CANDIDATES;
    if (temperature == 0.0f) {
      const uint likeliest = lattice.likeliest[position][predecessor][child];
      drawn = q[likeliest] > 0.0f ? likeliest : drawn;
    } else {
      for (uint i = 0; i < SPLASH_DRAFT_CANDIDATES; ++i)
        if (!(taken & (1u << i)) && keys[i] > -INFINITY &&
            (drawn == SPLASH_DRAFT_CANDIDATES || keys[i] > keys[drawn]))
          drawn = i;
    }
    if (drawn == SPLASH_DRAFT_CANDIDATES)
      return;
    taken |= 1u << drawn;
    const bool joins_chain = lattice.ids[position][drawn] == chain_token;
    const float cap = joins_chain ? left + reserved : left;
    rows.draw[row][child] = uchar(drawn);
    rows.left[row][child] = cap;
    const float own = min(tree_draft_estimate(params, lane, rows.next[row][drawn], position), cap);
    left = joins_chain ? cap - max(own, reserved) : left - own;
    reserved = joins_chain ? 0.0f : reserved;
    bound = min(bound, score * own);
    tree_push(heap, bound, tree_info(false, row, position, drawn));
  }
}

// The cap of the step row `parent` drew to its candidate `index`
// (tree_expand).
inline float tree_cap(threadgroup TreeRows &rows, uint parent, uint index) {
  for (uint child = 0; child < kTreeChildren; ++child)
    if (rows.draw[parent][child] == index)
      return rows.left[parent][child];
  return 1.0f;
}

// One group per lane builds its tree (DraftTreeParams). Its threads first
// take, for every proposal position and predecessor candidate (the anchor
// for position 0), the probabilities of the position's candidates (the
// softmax at temperature 1 of their unary scores plus the edges from the
// predecessor, the table draft_select_edges left in the partial-values
// scratch, each sum softcapped where the draft takes its head's softcap) and
// the predecessor's most likely, the lowest index first among
// equal scores, as the chain walk chose; thread 0 then grows the tree. Rows
// are written in the order the tree takes them, lane l's from row l *
// SPLASH_TARGET_VERIFY_ROWS, so a row's parent comes before it and its
// estimate is at most its parent's; sizes holds each lane's rows. A row's
// draft probability (scores' x) is that of the draft step that made it, or
// joined it where a lookup chain's token made it, its token's among the
// candidates after its parent's; -1 for a chain row no draft step joined;
// its estimate is scores' y, and the cap of that draft step its caps entry
// (1 for a row it has no draft probability of). Of each row it also writes the candidate indices of the
// children it drew, in order (draws, kNoCandidate past them), and its next
// position's probabilities (next), which a sampled verify row's selection
// tests (decode_sample_vocabulary_draw).
kernel void draft_select_tree(
    device const uint *candidates [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *partial_values [[buffer(2)]],
    device const uint *chain_tokens [[buffer(3)]],
    device const float *chain_rates [[buffer(4)]],
    device uint *tokens [[buffer(5)]],
    device SplashTreeRow *tree [[buffer(6)]],
    device float2 *scores [[buffer(7)]],
    device uint *sizes [[buffer(8)]],
    device uchar *draws [[buffer(9)]],
    device float *next [[buffer(10)]],
    device float *caps [[buffer(11)]],
    constant DraftTreeParams &params [[buffer(12)]],
    uint lane [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = SPLASH_DRAFT_CANDIDATES;
  threadgroup TreeLattice lattice;
  threadgroup TreeHeap heap;
  threadgroup TreeRows rows;

  const ulong origin = ulong(lane) * Positions * Candidates;
  if (thread_index < Positions * Candidates) {
    const uint position = thread_index / Candidates;
    const uint predecessor = thread_index % Candidates;
    lattice.ids[position][predecessor] = candidates[origin + thread_index];
    // Position 0 follows the anchor alone.
    if (position > 0 || predecessor == 0) {
      device const float *edges =
          partial_values +
          ulong(params.lanes) * Positions * Shards * Candidates +
          ((ulong(lane) * Positions + position) * Candidates + predecessor) *
              Candidates;
      // A softcapped head's draft takes the softcap on each sum, its tanh the
      // precise one, as the reference computes it, and a candidate of no
      // logit (-inf) stays none (DraftTreeParams).
      const float softcap = params.softcap;
      float score[Candidates];
      for (uint i = 0; i < Candidates; ++i) {
        const float sum = unary[origin + position * Candidates + i] + edges[i];
        score[i] = softcap > 0.0f && sum > -INFINITY ? softcap * precise::tanh(sum / softcap) : sum;
      }
      float best = score[0];
      for (uint i = 1; i < Candidates; ++i)
        best = max(best, score[i]);
      float weight[Candidates];
      float mass = 0.0f;
      for (uint i = 0; i < Candidates; ++i) {
        weight[i] = exp(score[i] - best);
        mass += weight[i];
      }
      for (uint i = 0; i < Candidates; ++i)
        lattice.probability[position][predecessor][i] = weight[i] / mass;
      uint taken = 0;
      for (uint child = 0; child < kTreeChildren; ++child) {
        uint selected = Candidates;
        for (uint i = 0; i < Candidates; ++i)
          if (!(taken & (1u << i)) &&
              (selected == Candidates || score[i] > score[selected]))
            selected = i;
        taken |= 1u << selected;
        lattice.likeliest[position][predecessor][child] = uchar(selected);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  if (thread_index == 0) {
    device const uint *chain =
        chain_tokens + ulong(lane) * SPLASH_TREE_CHAIN_TOKENS;
    device const float *rates = chain_rates + ulong(lane) *
                                                  SPLASH_TREE_CHAIN_TOKENS *
                                                  SPLASH_TREE_AGREEMENTS;
    const uint children = params.children[lane];
    const uint deepest = params.depth[lane], chained = params.chain[lane];
    heap.size = 0;
    heap.pushed = 0;
    rows.made = 1;
    rows.token[0] = params.anchor[lane];
    rows.ancestors[0] = 1;
    rows.parent[0] = SPLASH_TREE_ROOT;
    rows.depth[0] = 0;
    rows.on_chain[0] = true;
    rows.q[0] = 1.0f;
    rows.estimate[0] = 1.0f;
    rows.cap[0] = 1.0f;
    rows.key[0] = ulong(params.noise[lane][1]) << 32 | params.noise[lane][0];
    tree_row_clear(rows, 0);
    if (deepest > 0) {
      const float rate =
          chained ? tree_chain_rate(lattice, chain, rates, 0) : 0.0f;
      tree_expand(lattice, heap, rows, params, lane, children, 0, 0, 0, 1.0f,
                  chained ? chain[0] : kNoChainToken, rate);
      if (chained)
        tree_push(heap, rate, tree_info(true, 0, 0, 0));
    }
    // No more rows than the tokens a walk of the tree may retain.
    const uint most = min(Rows, deepest + 1);
    while (heap.size && rows.made < most) {
      const float score = heap.score[0];
      const uint info = heap.info[0];
      tree_pop(heap);
      const bool from_chain = (info & kTreeChain) != 0;
      const uint parent = (info >> 16) & 0x7fffu;
      const uint position = (info >> 8) & 0xffu, index = info & 0xffu;
      const uint token =
          from_chain ? chain[position] : lattice.ids[position][index];
      // A token path the tree holds already gains no row.
      uint row = 0;
      for (uint other = 1; other < rows.made && !row; ++other)
        row = rows.parent[other] == parent && rows.token[other] == token
                  ? other
                  : 0;
      if (!row) {
        row = rows.made++;
        rows.token[row] = token;
        rows.ancestors[row] = rows.ancestors[parent] | (1u << row);
        rows.parent[row] = ushort(parent);
        rows.depth[row] = ushort(position + 1);
        rows.on_chain[row] = rows.on_chain[parent] && position < chained &&
                             token == chain[position];
        // A row a host chain token made is no draft step: its estimate is
        // the chain's, and it has no draft probability.
        rows.q[row] = from_chain ? -1.0f : rows.next[parent][index];
        rows.estimate[row] = score;
        rows.cap[row] = from_chain ? 1.0f : tree_cap(rows, parent, index);
        rows.key[row] = tree_key(rows.key[parent], token);
        tree_row_clear(rows, row);
      } else if (!from_chain && rows.q[row] < 0.0f) {
        // A draft step that joins a lookup chain's row labels the draft's
        // calibration like any draft row: such rows, which the draft's top
        // candidate often is, the target accepted 97-100% of the time in
        // decode traces (model::LookupAcceptance::kPrior), and the fit
        // without them learns from the draft's other rows alone.
        rows.q[row] = rows.next[parent][index];
        rows.cap[row] = tree_cap(rows, parent, index);
      }
      // The row's children lie a step deeper.
      if (position + 1 >= deepest)
        continue;
      // A row on the chain's path has the chain's next token for a child.
      const bool chain_child = rows.on_chain[row] && position + 1 < chained;
      if (!from_chain && position + 1 < Positions)
        tree_expand(lattice, heap, rows, params, lane, children, row,
                    position + 1, index, score,
                    chain_child ? chain[position + 1] : kNoChainToken,
                    chain_child ? tree_chain_rate(lattice, chain, rates,
                                                  position + 1)
                                : 0.0f);
      else if (from_chain && position + 1 < chained)
        tree_push(heap,
                  score * tree_chain_rate(lattice, chain, rates, position + 1),
                  tree_info(true, row, position + 1, 0));
    }
    sizes[lane] = rows.made;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < rows.made) {
    const ulong at = ulong(lane) * Rows + thread_index;
    tokens[at] = rows.token[thread_index];
    tree[at] = {rows.ancestors[thread_index], rows.parent[thread_index],
                rows.depth[thread_index]};
    scores[at] = float2(rows.q[thread_index], rows.estimate[thread_index]);
    caps[at] = rows.cap[thread_index];
    for (uint child = 0; child < kTreeChildren; ++child)
      draws[at * kTreeChildren + child] = rows.draw[thread_index][child];
    for (uint i = 0; i < Candidates; ++i)
      next[at * Candidates + i] = rows.next[thread_index][i];
  }
}

// One thread lays out a decode cycle's rows (VerifyRowsParams) from the
// trees the selector grew, lane l's sizes[l] rows from row l *
// SPLASH_TARGET_VERIFY_ROWS of tree_tokens, trees and scores: each lane's
// rows, and its rows' tokens, tree rows and sources (the row of its lane's
// tree each is), packed in lane order, and after them the rows of no lane,
// each lane 0's anchor and no source (SPLASH_NO_TOKEN). A lane's rows are
// numbered in preorder, the earlier-taken (likelier) child first, so that
// the likeliest path of its tree lies on its first rows.
kernel void draft_verify_rows(device const uint *tree_tokens [[buffer(0)]],
                              device const SplashTreeRow *trees [[buffer(1)]],
                              device const float2 *scores [[buffer(2)]],
                              device const uint *sizes [[buffer(3)]],
                              device uint *tokens [[buffer(4)]],
                              device SplashTreeRow *tree [[buffer(5)]],
                              device SplashLaneRows *lane_rows [[buffer(6)]],
                              device uint *sources [[buffer(7)]],
                              constant VerifyRowsParams &params [[buffer(8)]],
                              uint index [[thread_position_in_grid]]) {
  constexpr uint Lanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  if (index != 0)
    return;
  // Every lane's rows past its anchor by estimate, a lane's earlier row first
  // among equal ones.
  float estimates[Lanes * Rows];
  uint owners[Lanes * Rows];
  uint count = 0;
  for (uint lane = 0; lane < params.lanes; ++lane) {
    for (uint row = 1; row < sizes[lane]; ++row) {
      const float estimate = scores[lane * Rows + row].y;
      uint at = count++;
      for (; at > 0 && estimates[at - 1] < estimate; --at) {
        estimates[at] = estimates[at - 1];
        owners[at] = owners[at - 1];
      }
      estimates[at] = estimate;
      owners[at] = lane;
    }
  }
  // The best of them that the cycle's rows and the lanes' cap hold, none
  // below the floor: they fall, so the first below it ends them.
  uint rows[Lanes];
  for (uint lane = 0; lane < params.lanes; ++lane)
    rows[lane] = 1;
  uint taken = params.lanes;
  for (uint i = 0; i < count && taken < params.rows; ++i) {
    if (estimates[i] < params.floor)
      break;
    if (rows[owners[i]] == params.lane_cap)
      continue;
    ++rows[owners[i]];
    ++taken;
  }
  uint first = 0;
  for (uint lane = 0; lane < params.lanes; ++lane) {
    device const uint *token = tree_tokens + lane * Rows;
    uint parent[Rows];
    for (uint row = 0; row < rows[lane]; ++row)
      parent[row] = trees[lane * Rows + row].parent;
    // Preorder: a row's children follow it, each with its subtree, in the
    // order the tree took them.
    uint order[Rows], placed[Rows], stack[Rows];
    uint size = 0, next = 0;
    stack[size++] = 0;
    while (size) {
      const uint row = stack[--size];
      placed[row] = next;
      order[next++] = row;
      for (uint child = rows[lane]; child-- > row + 1;)
        if (parent[child] == row)
          stack[size++] = child;
    }
    uint ancestors[Rows], depths[Rows];
    for (uint at = 0; at < rows[lane]; ++at) {
      const uint row = order[at];
      const uint up = at ? placed[parent[row]] : uint(SPLASH_TREE_ROOT);
      ancestors[at] = (at ? ancestors[up] : 0u) | (1u << at);
      depths[at] = at ? depths[up] + 1 : 0u;
      tokens[first + at] = token[row];
      tree[first + at] = {ancestors[at], ushort(up), ushort(depths[at])};
      sources[first + at] = row;
    }
    lane_rows[lane] = {first, rows[lane]};
    first += rows[lane];
  }
  // The rows of no lane (VerifyRowsParams), which the embedding gathers by
  // their token, on no path and of no parent: none of an earlier cycle's.
  for (uint row = first; row < params.rows; ++row) {
    tokens[row] = tree_tokens[0];
    tree[row] = {0u, ushort(SPLASH_TREE_ROOT), ushort(0)};
    sources[row] = SPLASH_NO_TOKEN;
  }
}

// One shard of a greedy row: its admitted token with the largest selectable
// logit, the lowest id among ties, per thread in token order and then over
// the group. A row with no selectable logit keeps the sentinel id.
inline void argmax_shard(TargetRow row, uint shard, device float &partial_value,
                         device uint &partial_index,
                         threadgroup float *group_values,
                         threadgroup uint *group_indices, uint thread_index) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  for (uint token = shard * 256 + thread_index; token < row.vocabulary;
       token += Shards * 256) {
    if (!row.admits(token))
      continue;
    float value = row.logits[token];
    if (unselectable_logit(value))
      continue;
    if (value > best || (value == best && token < best_index)) {
      best = value;
      best_index = token;
    }
  }
  uint lane = thread_index & 31;
  uint simd_group = thread_index >> 5;
  float simd_best = simd_max(best);
  uint simd_index = simd_min(best == simd_best ? best_index : 0xffffffffu);
  if (lane == 0) {
    group_values[simd_group] = simd_best;
    group_indices[simd_group] = simd_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float value = lane < 8 ? group_values[lane] : -INFINITY;
    float group_best = simd_max(value);
    uint index =
        lane < 8 && value == group_best ? group_indices[lane] : 0xffffffffu;
    index = simd_min(index);
    if (lane == 0) {
      partial_value = group_best;
      partial_index = index;
    }
  }
}

// One shard of each selected row of the greedy lanes.
kernel void decode_sample_argmax_sharded(
    device const float *logits [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    device uint *partial_indices [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant TargetSamplingParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  const uint batch = group.y, index = group.x / SPLASH_TARGET_SAMPLING_SHARDS;
  if (!selects(params, lane_rows, batch, index, false))
    return;
  const uint shard = group.x % SPLASH_TARGET_SAMPLING_SHARDS;
  const ulong partial =
      ulong(selected_logits_row(lane_rows, batch, index)) *
          SPLASH_TARGET_SAMPLING_SHARDS +
      shard;
  argmax_shard(selected_row(logits, token_mask, params, lane_rows, batch, index), shard,
               partial_values[partial], partial_indices[partial],
               group_values, group_indices, thread_index);
}

// A greedy row's token from its shards' partials.
inline uint argmax_reduce(device const float *partial_values,
                          device const uint *partial_indices, uint row,
                          uint lane) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float value = lane < Shards ? partial_values[row * Shards + lane] : -INFINITY;
  float best = simd_max(value);
  uint index = lane < Shards && value == best
                   ? partial_indices[row * Shards + lane]
                   : 0xffffffffu;
  return simd_min(index);
}

// The token of each selected row of the greedy lanes.
kernel void decode_sample_argmax_reduce(
    device const float *partial_values [[buffer(0)]],
    device const uint *partial_indices [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    device const SplashLaneRows *lane_rows [[buffer(3)]],
    constant TargetSamplingParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint batch = group.y, index = group.x;
  if (!selects(params, lane_rows, batch, index, false))
    return;
  const uint s = selected_logits_row(lane_rows, batch, index);
  const uint token = argmax_reduce(partial_values, partial_indices, s, lane);
  if (lane == 0)
    tokens[s] = token;
}

// After the selection, the thinking cutoff of each lane that has one
// (ThinkingCutoffParams): each row whose path reaches the path's cut, its row
// at the depth of the lane's row or, where the lane's end waits for its mask,
// the first from there whose mask admits the end's first token, selects the
// end token of its depth past the cut. A row's path up to its cut, its anchor
// excepted, holds the draft tokens before the cut.
kernel void decode_sample_thinking_cutoff(
    device const uint *input_tokens [[buffer(0)]],
    device const uint *token_mask [[buffer(1)]],
    device const SplashTreeRow *tree [[buffer(2)]],
    device uint *tokens [[buffer(3)]],
    device const SplashLaneRows *lane_rows [[buffer(4)]],
    constant ThinkingCutoffParams &params [[buffer(5)]],
    uint lane [[threadgroup_position_in_grid]]) {
  const uint rows = lane_rows[lane].rows, first = lane_rows[lane].first;
  const bool waits = (params.waiting_mask & (1u << lane)) != 0;
  // The lane's mask rows, as selected_row reads them.
  device const uint *masks =
      token_mask + (ulong(lane) * (SPLASH_TARGET_VERIFY_ROWS + 1) +
                    params.mask_row) *
                       params.mask_words;
  const uint start = params.tokens[lane][0];
  for (uint at = 0; at < rows; ++at) {
    const uint path = tree[first + at].ancestors;
    const uint depth = tree[first + at].depth;
    // The path's rows ascend from its anchor, one a depth; one past the
    // row's depth where the path holds no cut.
    uint cut = 0;
    for (uint walk = path; walk; walk &= walk - 1, ++cut)
      if (cut >= params.row[lane] &&
          (!waits || (masks[ctz(walk) * params.mask_words + start / 32] &
                      (1u << (start % 32)))))
        break;
    if (cut > depth || depth - cut >= params.count[lane])
      continue;
    bool ended = false;
    for (uint walk = path & ~1u; walk; walk &= walk - 1) {
      const uint before = ctz(walk);
      const uint draft = input_tokens[first + before];
      ended = ended || (tree[first + before].depth <= cut &&
                        (draft == params.close[lane] || draft == params.also_ends[lane]));
    }
    // A row with no selectable logit keeps its sentinel, so the lane fails.
    if (!ended && tokens[first + at] < params.vocabulary)
      tokens[first + at] = params.tokens[lane][depth - cut];
  }
}

// Each lane's acceptance (AcceptBatchParams), one thread a lane. Its retained
// tokens, the selections of the path's rows, go to the output tokens from its
// first row, each read before an earlier path row's is written there.
kernel void decode_accept_tree(
    device const uint *input_tokens [[buffer(0)]],
    device const SplashTreeRow *tree [[buffer(1)]],
    device uint *tokens [[buffer(2)]],
    device uint *retained [[buffer(3)]],
    device uint *accepted_count [[buffer(4)]],
    device uint *retained_rows [[buffer(5)]],
    device const SplashLaneRows *lane_rows [[buffer(6)]],
    constant AcceptBatchParams &params [[buffer(7)]],
    uint batch [[threadgroup_position_in_grid]]) {
  const uint first = lane_rows[batch].first, rows = lane_rows[batch].rows;
  device const uint *inputs = input_tokens + first;
  device const SplashTreeRow *nodes = tree + first;
  device uint *selected = tokens + first;
  uint path[SPLASH_TARGET_VERIFY_ROWS];
  path[0] = 0;
  uint accepted = 0;
  for (uint row = 0;;) {
    const uint token = selected[row];
    uint next = 0;
    for (uint child = row + 1; child < rows && !next; ++child)
      next = nodes[child].parent == row && inputs[child] == token ? child : 0;
    if (!next)
      break;
    path[++accepted] = next;
    row = next;
  }
  uint kept = min(accepted + 1, params.remaining[batch]);
  for (uint i = 0; i < kept; ++i) {
    const uint token = selected[path[i]];
    selected[i] = token;
    if (stop_token(params.stop_tokens, token))
      kept = i + 1;
  }
  // Where the lane's cap or a stop token cuts its path, every token it
  // retains is a proposal it accepted.
  accepted_count[batch] = min(accepted, kept);
  retained[batch] = kept;
  retained_rows[batch] = nodes[path[kept - 1]].ancestors;
}
// ---- end runtime/metal/kernels/decode/sampling.metal ----
