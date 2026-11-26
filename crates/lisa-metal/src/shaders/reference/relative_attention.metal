// Vendored from Splash (https://github.com/incoai/splash), revision
// 2afc7337aacc044722fa95e4cb3da75c281a19a8, Apache License 2.0 (LICENSE-APACHE-2.0).
// Quantized GGUF decoding follows ggml (MIT, LICENSE-GGML). Each original
// header is inlined once at its first include, between begin/end markers; the
// code is otherwise unchanged. Source hashes: provenance.json.
// ---- begin runtime/metal/kernels/shared/relative_attention.metal (Splash 2afc733, Apache-2.0) ----
#pragma clang fp reassociate(off)
// ---- begin runtime/metal/kernels/common/kv_path_commit.h (Splash 2afc733, Apache-2.0) ----

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
// ---- begin runtime/metal/kernels/common/kv_extent.h (Splash 2afc733, Apache-2.0) ----

// ---- begin runtime/metal/kernels/common/kv_paging.h (Splash 2afc733, Apache-2.0) ----

#include <metal_stdlib>

using namespace metal;

// KV page geometry of both formats, shared by the attention and store
// kernels. A page holds SplashKvPageTokens tokens of every KV head of HeadDim
// dimensions, a value each kernel is compiled for: keys token-major, values
// dimension-major, or token-major like the keys where a layout's are
// (ValuesTokenMajor, kv::Layout::valuesTokenMajor): a block of a head's
// tokens then holds its values contiguously. INT8 adds one fp32 scale per
// (KV head, token) for each tensor.
constant uint SplashKvPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;

// An element's index in a layer's region of one tensor: its page's slab,
// then its place in the slab (abi/KvExtent.h).
template <uint KVHeads, uint HeadDim>
inline ulong splash_kv_key_index(uint page, uint head, uint token,
                                   uint dimension) {
  constexpr ulong ElementsPerPage = ulong(KVHeads) * SplashKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         splash_kv_key_element(head, token, dimension, HeadDim);
}

template <uint KVHeads, uint HeadDim, bool ValuesTokenMajor = false>
inline ulong splash_kv_value_index(uint page, uint head, uint token,
                                     uint dimension) {
  constexpr ulong ElementsPerPage = ulong(KVHeads) * SplashKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         (ValuesTokenMajor ? splash_kv_key_element(head, token, dimension, HeadDim)
                           : splash_kv_value_element(head, token, dimension, HeadDim));
}

template <uint KVHeads>
inline ulong splash_q8_scale_index(uint page, uint head, uint token) {
  constexpr ulong ScalesPerPage = ulong(KVHeads) * SplashKvPageTokens;
  return ulong(page) * ScalesPerPage + splash_kv_scale_element(head, token);
}
// ---- end runtime/metal/kernels/common/kv_paging.h ----
#include <metal_stdlib>

using namespace metal;

// Kernels reach a KV page through its entry (abi/KvExtent.h): the GPU address
// of the extent that holds it, with the page's index in the low bits. Every
// kernel that touches KV pages forms the pointers to one KV head's slab of
// each of a page's tensors in one layer the same way, with
// SplashKvAddressing: splash_kv_offset places the layer's regions in the
// extent, whose 64-bit offsets it adds to the extent's address, and the page
// index functions of kv_paging.h place the page and the head in each region.

inline uint splash_kv_page_index(SplashKvPage page) {
  return uint(page) & SPLASH_KV_PAGE_INDEX_MASK;
}

inline device uchar *splash_kv_extent(SplashKvPage page, uint index) {
  return reinterpret_cast<device uchar *>(page - index);
}

// One page's bytes of keys (or values) and of their scales in one layer.
template <uint KVHeads, uint HeadDim, typename CacheElement> struct SplashKvPageBytes {
  static constexpr constant bool Quantized = is_same<CacheElement, int8_t>::value;
  static constexpr constant uint Data =
      KVHeads * SplashKvPageTokens * HeadDim * sizeof(CacheElement);
  static constexpr constant uint Scale =
      Quantized ? KVHeads * SplashKvPageTokens * sizeof(float) : 0;
};

// One KV head's slab of each tensor of a page; BF16 has no scales.
template <typename CacheElement> struct SplashKvPageTensors {
  device CacheElement *keys;
  device CacheElement *values;
  device float *key_scales;
  device float *value_scales;
};

template <uint KVHeads, uint HeadDim, typename CacheElement> struct SplashKvAddressing {
  using Bytes = SplashKvPageBytes<KVHeads, HeadDim, CacheElement>;
  uint layer_offset;
  uint head;
  ulong key_scales_offset;
  ulong values_offset;
  ulong value_scales_offset;

  SplashKvAddressing(SplashKvLayer kv, uint kv_head)
      : layer_offset(kv.offset), head(kv_head),
        key_scales_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                           0, SPLASH_KV_KEY_SCALES, 0)),
        values_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                       0, SPLASH_KV_VALUES, 0)),
        value_scales_offset(splash_kv_offset(kv.extent_pages, Bytes::Data,
                                             Bytes::Scale, 0,
                                             SPLASH_KV_VALUE_SCALES, 0)) {}

  SplashKvPageTensors<CacheElement> page(SplashKvPage entry) const {
    const uint index = splash_kv_page_index(entry);
    device uchar *region = splash_kv_extent(entry, index) + layer_offset;
    SplashKvPageTensors<CacheElement> tensors{};
    tensors.keys = reinterpret_cast<device CacheElement *>(region) +
                   splash_kv_key_index<KVHeads, HeadDim>(index, head, 0, 0);
    tensors.values = reinterpret_cast<device CacheElement *>(region + values_offset) +
                     splash_kv_value_index<KVHeads, HeadDim>(index, head, 0, 0);
    if constexpr (Bytes::Quantized) {
      const ulong scale_index = splash_q8_scale_index<KVHeads>(index, head, 0);
      tensors.key_scales =
          reinterpret_cast<device float *>(region + key_scales_offset) + scale_index;
      tensors.value_scales =
          reinterpret_cast<device float *>(region + value_scales_offset) + scale_index;
    }
    return tensors;
  }
};
// ---- end runtime/metal/kernels/common/kv_extent.h ----
// ---- begin runtime/metal/kernels/common/lane_bindings.h (Splash 2afc733, Apache-2.0) ----


// A batched kernel binds a per-lane tensor once per lane
// (ops::appendLaneBindings): lane `lane`'s binding of the eight. A macro, not
// a function: the compiler keeps the chain as selects only where it is
// written in place, and turns a function's into branches. Eight lanes are
// the most this binding form holds: the kernel that binds the most per-lane
// tensors (block_indexer_commit, three) reaches buffer index 28 of Metal's 31.
static_assert(SPLASH_MAXIMUM_BATCH_WIDTH == 8, "one binding per lane");
#define SPLASH_LANE_BINDING(lane, lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7)                      \
  ((lane) == 0   ? (lane0)                                                                                     \
   : (lane) == 1 ? (lane1)                                                                                     \
   : (lane) == 2 ? (lane2)                                                                                     \
   : (lane) == 3 ? (lane3)                                                                                     \
   : (lane) == 4 ? (lane4)                                                                                     \
   : (lane) == 5 ? (lane5)                                                                                     \
   : (lane) == 6 ? (lane6)                                                                                     \
                 : (lane7))
// ---- end runtime/metal/kernels/common/lane_bindings.h ----

// A decode step's q/k projection writes every row it holds into its page
// slot (splash_store_kv_slot, attention_qkv_prepare.h), a lane's at the
// slots of its rows of its block; its commit here moves the accepted path's
// rows to the slots of their depths, and the rows past them stay unreachable
// until the next command starting at the same logical position overwrites
// them.

// Moves a lane's accepted path's K/V of one KV head of one layer region
// (SplashKvPathCommitParams): its row at depth k from the slot of its row to
// that of its depth, each thread one dimension of every row it moves, a
// threadgroup of HeadDim threads per (region and KV head, lane), and thread 0
// an INT8 page's scales. Rows move in order of depth: a row's target is a
// slot only an earlier row read (rows 0, 2, 3 move 2 to 1, then 3 to 2), as
// every later row sits past it. A row at its depth stays: a chain's path
// moves nothing.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
inline void splash_path_commit_phase(device const SplashKvPage *page_table, uint path, uint committed,
                                     SplashKvLayer kv, uint head, uint dimension) {
  const SplashKvAddressing<KVHeads, HeadDim, CacheElement> addressing(kv, head);
  for (uint depth = 0, rows = path; rows; ++depth, rows &= rows - 1) {
    const uint row = ctz(rows);
    if (row == depth)
      continue;
    const uint from = committed + row, to = committed + depth;
    const uint source_place = from % SplashKvPageTokens, target_place = to % SplashKvPageTokens;
    const SplashKvPageTensors<CacheElement> source = addressing.page(page_table[from / SplashKvPageTokens]);
    const SplashKvPageTensors<CacheElement> target = addressing.page(page_table[to / SplashKvPageTokens]);
    target.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, target_place, dimension)] =
        source.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, source_place, dimension)];
    target.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, target_place, dimension)] =
        source.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, source_place, dimension)];
    if constexpr (is_same<CacheElement, int8_t>::value) {
      if (dimension == 0) {
        target.key_scales[splash_q8_scale_index<KVHeads>(0, 0, target_place)] =
            source.key_scales[splash_q8_scale_index<KVHeads>(0, 0, source_place)];
        target.value_scales[splash_q8_scale_index<KVHeads>(0, 0, target_place)] =
            source.value_scales[splash_q8_scale_index<KVHeads>(0, 0, source_place)];
      }
    }
  }
}

// Grid {regions x KV heads, lanes}; paths: each lane's accepted path, as
// acceptance wrote it.
#define PAGED_ATTENTION_PATH_COMMIT(Name, Heads, HeadDim, CacheElement, ValuesTokenMajor)                         \
  kernel void Name(device const SplashKvPage *page_table0 [[buffer(0)]],                                         \
                   device const SplashKvPage *page_table1 [[buffer(1)]],                                         \
                   device const SplashKvPage *page_table2 [[buffer(2)]],                                         \
                   device const SplashKvPage *page_table3 [[buffer(3)]], device const SplashKvPage *page_table4 [[buffer(4)]], device const SplashKvPage *page_table5 [[buffer(5)]], device const SplashKvPage *page_table6 [[buffer(6)]], device const SplashKvPage *page_table7 [[buffer(7)]],                                         \
                   device const uint *paths [[buffer(8)]],                                                       \
                   constant SplashKvPathCommitParams &params [[buffer(9)]],                                      \
                   uint2 group [[threadgroup_position_in_grid]],                                                 \
                   uint thread_index [[thread_index_in_threadgroup]]) {                                          \
    using Bytes = SplashKvPageBytes<Heads, HeadDim, CacheElement>;                                               \
    const uint lane = group.y;                                                                                   \
    splash_path_commit_phase<Heads, HeadDim, CacheElement, ValuesTokenMajor>(                                    \
        SPLASH_LANE_BINDING(lane, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7), paths[lane],              \
        params.committed_tokens[lane],                                                                           \
        splash_kv_layer(params.extent_pages, Bytes::Data, Bytes::Scale, group.x / Heads), group.x % Heads,       \
        thread_index);                                                                                           \
  }
// ---- end runtime/metal/kernels/common/kv_path_commit.h ----
// ---- begin runtime/metal/kernels/common/relative_attention_tile.h (Splash 2afc733, Apache-2.0) ----

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
// ---- begin runtime/metal/abi/RelativeAttention.h (Splash 2afc733, Apache-2.0) ----

// Constants of relative attention (ops::RelativeAttention, kernels/shared/
// relative_attention.metal), shared by host dispatch code and Metal kernels:
// Inkling-2-Mini's attention, whose kernels take SplashAttentionParams
// (metal/abi/PagedAttention.h) as the paged attention's do.

// Each query head of a row carries SPLASH_RELATIVE_ATTENTION_STATES relative
// states s. A key at backward distance d (the row's position less the key's)
// below SPLASH_RELATIVE_ATTENTION_EXTENT adds s . bias[d] to its score, a
// key past it nothing (the source's d_rel and rel_extent). The layer's bias
// table holds a row of as many states for each distance: [distance][state].
#define SPLASH_RELATIVE_ATTENTION_STATES 16u
#define SPLASH_RELATIVE_ATTENTION_EXTENT 1024u

// The prepare clips each head of the queries and keys to an L2 norm of at
// most SPLASH_RELATIVE_ATTENTION_QK_CLIP (sqrt(512), qk_l2_clip_threshold),
// and of the relative states to SPLASH_RELATIVE_ATTENTION_STATE_CLIP
// (rel_l2_clip_threshold): x min(1, clip / (|x| + epsilon)), the source's
// epsilon. A position-scaled layer's (the full layers') queries and states
// are scaled by 1 + alpha ln(max((position + 1) / floor, 1))
// (log_scaling_alpha, log_scaling_n_floor), which grows their logits past
// the floor's positions.
#define SPLASH_RELATIVE_ATTENTION_QK_CLIP 22.627416997969522f
#define SPLASH_RELATIVE_ATTENTION_STATE_CLIP 8.0f
#define SPLASH_RELATIVE_ATTENTION_CLIP_EPSILON 1e-6f
#define SPLASH_RELATIVE_ATTENTION_LOG_SCALE_ALPHA 0.1f
#define SPLASH_RELATIVE_ATTENTION_LOG_SCALE_FLOOR 8192.0f

// A decode step's single pass (ops::RelativeAttention::addDecode) splits each
// lane's history into at most SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS runs:
// the run that finishes last combines them, a simdgroup lane each.
#define SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS 32u

// A prompt chunk's dispatch whose tiles each read their history in one split
// (ops::RelativeAttention::addAttention) runs them as tiles of
// SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS rows, two of the paged attention's,
// each of which writes its output rows itself (the direct kernels).
#define SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS 16u
// ---- end runtime/metal/abi/RelativeAttention.h ----
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
// ---- begin runtime/metal/kernels/common/attention_sequence.h (Splash 2afc733, Apache-2.0) ----

#include <metal_stdlib>

using namespace metal;

// A dispatch's sequences (SplashAttentionParams) are in tile order: the one
// that holds the dispatch's tile `tile` is the last whose first tile is at
// most it. A sequence's place among them names its page table among the
// dispatch's bindings (SPLASH_LANE_BINDING) and its entry of the command's
// layout. The lookup is seven selects over every entry, not a loop to
// params.sequences: a tile whose sequence such a loop found ran its page loop
// up to a quarter slower (the 64-fused-row tiles of two KV heads).
inline uint splash_attention_sequence_of_tile(constant SplashAttentionParams &params, uint tile) {
  uint sequence = 0;
#pragma unroll
  for (uint s = 1; s < SPLASH_MAXIMUM_BATCH_WIDTH; ++s)
    sequence = s < params.sequences && tile >= params.sequence[s].first_tile ? s : sequence;
  return sequence;
}

// Whether a sequence's row bound, staging and pages, and the layer the
// dispatch's pages hold, are within the kernels' bounds, as the host checks
// them (ops::PagedAttention): the store's contract, and with its split count
// the page loop's.
inline bool splash_attention_rows_valid(constant SplashAttentionParams &params, constant SplashAttentionSequence &s) {
  return s.row_bound > 0 && s.row_bound <= SPLASH_PREFILL_TOKEN_BUDGET && s.chunk_stride >= s.row_bound &&
         s.chunk_stride <= SPLASH_PREFILL_TOKEN_BUDGET && s.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         ulong(s.committed_tokens) + s.row_bound <= ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         s.page_table_entries >= (s.committed_tokens + s.row_bound + SplashKvPageTokens - 1) / SplashKvPageTokens &&
         params.kv.extent_pages > 0;
}

// Dispatch sequence s's rows of the command (ops::AttentionRows): its first
// packed row and its rows, from its layout entry's row layout_row on. An
// entry past the sequence's row bound or the command's rows holds none.
struct SplashAttentionRows {
  uint first;
  uint rows;
};

inline SplashAttentionRows splash_attention_rows(device const SplashLaneRows *layout,
                                                 constant SplashAttentionParams &params, uint s) {
  const SplashLaneRows entry = layout[params.first_sequence + s];
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const uint rows = entry.rows - min(entry.rows, sequence.layout_row);
  if (rows > sequence.row_bound || ulong(entry.first) + entry.rows > params.rows)
    return {0, 0};
  return {entry.first + sequence.layout_row, rows};
}

// Where a reduce writes row `row` of dispatch sequence s (counted from its
// rows' first, SplashAttentionRows): its packed row, the output row it
// writes, its packed row or, with last_rows (SplashAttentionParams), its
// sequence's place among the command's for the sequence's last row alone,
// and whether it writes: a row the sequence holds, and with last_rows its
// last.
struct SplashAttentionOutput {
  ulong packed;
  ulong output;
  bool writes;
};

inline SplashAttentionOutput splash_attention_output(constant SplashAttentionParams &params, uint s,
                                                     SplashAttentionRows rows, uint row) {
  const ulong packed = ulong(rows.first) + row;
  return {packed, params.last_rows ? ulong(params.first_sequence) + s : packed,
          row < rows.rows && (!params.last_rows || row + 1 == rows.rows)};
}
// ---- end runtime/metal/kernels/common/attention_sequence.h ----
// ---- begin runtime/metal/kernels/common/attention_store.h (Splash 2afc733, Apache-2.0) ----

#include <metal_stdlib>

using namespace metal;

// Writes one row's key and value of one KV head into the row's page slot,
// through the slot's page entry (splash_store_page): a threadgroup of HeadDim
// threads, each one dimension of both. The slot is the row's own, at the
// sequence's committed tokens plus the row: a decode step
// writes all its rows, a lane's at the slots of their rows of its block; its
// commit moves the accepted path's rows to the slots of their depths
// (splash_path_commit_phase, kv_path_commit.metal), and the rows past
// them stay unreachable until the next command starting at the same logical
// position overwrites them. INT8 pages take a scale per row and head, the
// largest magnitude of the head's values over 127, and codes of the values
// over it rounded to nearest within +-127 (all zero for a zero row); BF16
// pages take the values' bits. Slots are addressed inside the head's slab of
// the page: the page index functions at page zero and head zero.

// The scratch of a row's scale reductions, by CacheElement: a maximum per
// simdgroup of the HeadDim threads for the keys and then the values, for
// INT8; BF16 has none.
#define SPLASH_STORE_MAXIMA_int8_t(HeadDim) threadgroup float maxima[2 * ((HeadDim) / 32)];
#define SPLASH_STORE_MAXIMA_bfloat(HeadDim) threadgroup float *maxima = nullptr;

// The largest magnitude of each of the threadgroup's two values, keys' first
// in `maxima`, values' after them: every thread's, through a maximum per
// simdgroup and then over the simdgroups'. A maximum is exact, so its order
// is any.
template <uint HeadDim>
inline float2 splash_store_maxima(float key, float value, threadgroup float *maxima, uint simd_lane,
                                  uint simd_group) {
  constexpr uint Groups = HeadDim / 32;
  const float key_maximum = simd_max(abs(key)), value_maximum = simd_max(abs(value));
  if (simd_lane == 0) {
    maxima[simd_group] = key_maximum;
    maxima[Groups + simd_group] = value_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float2 maximum = 0.0f;
#pragma unroll
  for (uint group = 0; group < Groups; ++group)
    maximum = max(maximum, float2(maxima[group], maxima[Groups + group]));
  return maximum;
}

// The INT8 code of `value` at a row's largest magnitude `maximum`.
inline char splash_store_code(float value, float maximum) {
  return maximum == 0.0f ? char(0) : char(clamp(int(rint(value * 127.0f / maximum)), -127, 127));
}

// The page entry of a sequence's row, which a store reads before its row's
// values are ready: the slot's place follows the parameters alone.
inline SplashKvPage splash_store_page(device const SplashKvPage *page_table, constant SplashAttentionSequence &sequence,
                                      uint row) {
  return page_table[(sequence.committed_tokens + row) / SplashKvPageTokens];
}

template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
__attribute__((always_inline)) inline void splash_store_kv_slot(
    SplashKvPage page, constant SplashAttentionSequence &sequence, SplashKvLayer kv, threadgroup float *maxima,
    uint head, uint row, bfloat key, bfloat value, uint dimension, uint simd_lane, uint simd_group) {
  const SplashKvPageTensors<CacheElement> slab =
      SplashKvAddressing<KVHeads, HeadDim, CacheElement>(kv, head).page(page);
  const uint page_token = (sequence.committed_tokens + row) % SplashKvPageTokens;
  const ulong key_index = splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension);
  const ulong value_index = splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, page_token, dimension);
  if constexpr (is_same<CacheElement, bfloat>::value) {
    slab.keys[key_index] = key;
    slab.values[value_index] = value;
  } else {
    const float2 maximum = splash_store_maxima<HeadDim>(float(key), float(value), maxima, simd_lane, simd_group);
    slab.keys[key_index] = splash_store_code(float(key), maximum.x);
    slab.values[value_index] = splash_store_code(float(value), maximum.y);
    if (dimension == 0) {
      const ulong scale_index = splash_q8_scale_index<KVHeads>(0, 0, page_token);
      slab.key_scales[scale_index] = maximum.x == 0.0f ? 0.0f : maximum.x / 127.0f;
      slab.value_scales[scale_index] = maximum.y == 0.0f ? 0.0f : maximum.y / 127.0f;
    }
  }
}
// ---- end runtime/metal/kernels/common/attention_store.h ----
// ---- begin runtime/metal/kernels/common/gguf_sgmatrix.h (Splash 2afc733, Apache-2.0) ----
#include <metal_stdlib>
using namespace metal;

// The eight-row X^T table the Apple9 GGUF register kernel reads
// (LinearInput::Table16, laid out as metal/abi/Gguf.h states). A 64-element
// span occupies 512 bfloat in the xt_offset order; its fragments follow the chunk order of the GGUF
// image (metal/abi/QuantFormat.h): fragment 4q + f holds pair f of every
// chunk of the span's 32-element group q, so fragments 4q + 2h and
// 4q + 2h + 1 cover its 16-element group h. Per row, the sum of every 32
// inputs (the min or zero-point term of formats whose scale covers 32 inputs
// or more) and the chain seed of every 16 inputs (zero-point formats of
// 16-input scales, and Q2_K's min) are stored row-minor, so a lane reads its
// two rows at once.
namespace gguf_sg {

// Formats with a zero point enter the MMA as 160 + code - zero, exact in bf16
// (160 = 128 + 32, Q6_K's zero point), so their chains start from -160 times
// the sum of their inputs: the seed the table stores per 16 inputs, or, where
// a scale covers 32 inputs or more, -160 times its sums of 32 once per
// coefficient group and 64-input span
// (kernels/decode/linear_gguf_sgmatrix.metal).
constant constexpr uint kZeroPointOffset = 160;

// The rows of a table tile.
constant constexpr uint kRows = GGUF_TABLE16_ROWS;
static_assert(GGUF_TABLE16_SPAN_VALUES == GGUF_TABLE16_SPAN_INPUTS * kRows, "a span holds every row's inputs");

// A span's values laid out as quads, so the lane (fm, fn) reads fragments
// 4 jq .. 4 jq + 3 for (k' = fm, m = fn, fn + 1) as one vec<T, 8>:
//   offset(j, k', m) = (((j >> 2) 8 + k') 4 + (m >> 1)) 8 + (j & 3) 2 + (m & 1)
inline uint xt_offset(uint j, uint kp, uint m) {
  return (((j >> 2) * 8 + kp) * 4 + (m >> 1)) * 8 + (j & 3) * 2 + (m & 1);
}

// Physical k inside a span -> (fragment j, row k').
inline uint2 klogical(uint k) {
  const uint q = k >> 5, h = (k >> 4) & 1, a = (k >> 2) & 3, b = (k >> 1) & 1, e = k & 1;
  return uint2(4 * q + 2 * h + b, 2 * a + e);
}

// One simdgroup writes one span of one row; the lane holds elements
// 2 lane, 2 lane + 1.
inline void write_input(device bfloat *table, device float *sums, uint width, uint span,
                        uint row, uint lane, bfloat a, bfloat b) {
  const uint2 l = klogical(2 * lane);
  table[span * GGUF_TABLE16_SPAN_VALUES + xt_offset(l.x, l.y, row)] = a;
  table[span * GGUF_TABLE16_SPAN_VALUES + xt_offset(l.x, l.y + 1, row)] = b;
  float s = float(a) + float(b);
  s += simd_shuffle_xor(s, 1u);
  s += simd_shuffle_xor(s, 2u);
  s += simd_shuffle_xor(s, 4u);
  if ((lane & 7) == 0) sums[(span * (GGUF_TABLE16_SPAN_SEEDS / kRows) + lane / 8) * kRows + row] = -float(kZeroPointOffset) * s;
  const float s32 = s + simd_shuffle_xor(s, 8u);
  if ((lane & 15) == 0) sums[table16_sums32_offset(width) + (span * (GGUF_TABLE16_SPAN_SUMS / kRows) + lane / 16) * kRows + row] = s32;
}

struct Table16 {
  static ulong sums_per_tile(uint width) { return table16_sums_per_tile(width); }
  static void write(device bfloat *table, device float *sums, uint width, uint span, uint row,
                    uint lane, bfloat a, bfloat b) {
    write_input(table, sums, width, span, row, lane, a, b);
  }
};

} // namespace gguf_sg
// ---- end runtime/metal/kernels/common/gguf_sgmatrix.h ----
// ---- begin runtime/metal/kernels/common/paged_attention_tile.h (Splash 2afc733, Apache-2.0) ----

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
using namespace mpp::tensor_ops;

// Paged attention over INT8 or BF16 KV. INT8 has one fp32 scale per
// (token, KV head); BF16 reads the stored values directly. The preceding
// store writes each row's K/V to its row's slot: a prompt row's is final, a
// decode tree row's is its own row of the block, whose accepted path a commit
// moves to the slots of its depths; committed_tokens controls visibility,
// and the next command overwrites a decode step's rejected rows.
//
// A threadgroup owns one KV head, query tile (or a part of one,
// splash_attention_tile_part) and history split. It processes the GQA group
// as M = rows x query heads per KV head, at most 64, of HeadDim dimensions
// each, the head dimension a kernel is compiled for. Each split writes fp32
// partials and statistics for a fixed-order reduce.
//
// A kernel compiled with a Window attends, for a row at position p, only the
// keys p - Window to p (Window + 1 with its own; a sliding-window layer's),
// and a reduce given a layer's learned sinks adds one per query head
// (splash_attention_combine_splits). Window 0 and no sinks is plain causal
// attention over the whole history.
//
// Every attention dispatch's tiles share this device-operand page loop.
constant uint SplashAttentionMaximumSplits = SPLASH_ATTENTION_MAXIMUM_SPLITS;
constant uint SplashAttentionTileRows = SPLASH_ATTENTION_TILE_ROWS;

// A sequence's tiles run while its rows and splits are within bounds
// (attention_sequence.h).
inline bool splash_attention_contract_valid(constant SplashAttentionParams &params,
                                            constant SplashAttentionSequence &s) {
  return splash_attention_rows_valid(params, s) && s.split_count > 0 && s.split_count <= SplashAttentionMaximumSplits;
}

// The query tile a threadgroup owns, of one KV head: the offset of its fused
// rows in the queries and the output, the position of its first row, its
// active rows, its history splits and the partial slot of its first split.
// A sequence's tile t holds its rows (`rows`, its layout entry's,
// attention_sequence.h) from t SplashAttentionTileRows on; a tile whose
// sequence fails the contract, or past its rows (a spare slot of its bound),
// is inactive. Split and reduce entries, dense and sparse, take their tiles
// from here, so they agree on every offset.
struct SplashAttentionTile {
  ulong offset;
  ulong first_slot;
  uint first_position;
  uint active_rows;
  uint splits;
  bool active;
};

template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim>
inline SplashAttentionTile splash_attention_tile(constant SplashAttentionParams &params,
                                                 constant SplashAttentionSequence &sequence, SplashAttentionRows rows,
                                                 uint kv_head, uint tile) {
  const uint start = tile * SplashAttentionTileRows;
  if (!splash_attention_contract_valid(params, sequence) || kv_head >= KVHeads || start >= rows.rows)
    return {};
  return {(ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + start) * QueryHeadsPerKVHead *
              HeadDim,
          ulong(sequence.first_slot) + (ulong(tile) * KVHeads + kv_head) * sequence.split_count,
          sequence.committed_tokens + start, min(SplashAttentionTileRows, rows.rows - start),
          sequence.split_count, true};
}

// Part `part` of a tile run as Parts parts of RowsPerTile / Parts rows each
// (SPLASH_ATTENTION_TILE_PARTS: at most 64 fused rows each, the 256-thread page
// loop's), each a threadgroup of the split over the same pages. A part is a
// tile of its own rows: their queries and output, positions and active rows,
// and partial slots of its own, slot_stride (the slot splits of each of the
// tile's slots) apart. A part past the tile's active rows is inactive. One
// part is the tile.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Parts, uint HeadDim>
inline SplashAttentionTile splash_attention_tile_part(SplashAttentionTile tile, uint part,
                                                      uint slot_stride) {
  static_assert(Parts > 0 && RowsPerTile % Parts == 0, "parts of whole rows");
  constexpr uint Rows = RowsPerTile / Parts;
  const uint first = part * Rows;
  if (!tile.active || first >= tile.active_rows)
    return {};
  return {tile.offset + ulong(first) * QueryHeadsPerKVHead * HeadDim,
          tile.first_slot * Parts + ulong(part) * slot_stride, tile.first_position + first,
          min(Rows, tile.active_rows - first), tile.splits, true};
}

// A tile of a decode lane's block as a tree (SplashAttentionParams::trees):
// the place in its tree of the tile's first row, and the slot of the block's
// anchor, from which the block's rows' K/V sit at the slots of their rows.
// A prompt's tile has none (null rows) and attends causally.
struct SplashAttentionTree {
  device const SplashTreeRow *rows;
  uint anchor;
};

// The tree of a sequence's tile whose first row is at `first_position`, of
// the command's tree rows (a row each of its packed rows).
inline SplashAttentionTree splash_attention_tree(constant SplashAttentionParams &params,
                                                 constant SplashAttentionSequence &sequence, SplashAttentionRows rows,
                                                 device const SplashTreeRow *tree, uint first_position) {
  if (!params.trees)
    return {nullptr, 0};
  return {tree + rows.first + (first_position - sequence.committed_tokens), sequence.committed_tokens};
}

// The position a tile's window starts from: its first row's, or a tree's
// anchor's, since a tree row's depth may be less than its row.
inline uint splash_attention_window_position(bool tree, uint anchor, uint first_position) {
  return tree ? anchor : first_position;
}

// Split and reduce derive the same balanced partition of each query tile's
// visible pages (splash_attention_first_page on). Causal and window masking
// remain per query row inside each split.
inline uint splash_attention_pages(uint visible_tokens) {
  return (visible_tokens + SplashKvPageTokens - 1) / SplashKvPageTokens;
}

inline uint splash_attention_pages_per_split(uint pages, uint splits) {
  return (pages + splits - 1) / splits;
}

// The first key a row at `position` attends: Window keys before it, or with
// no window (0) the history's first.
template <uint Window> inline uint splash_attention_window_start(uint position) {
  return Window && position > Window ? position - Window : 0;
}

// The first page a tile whose window position (splash_attention_window_position)
// is first_position reads: that position's window start's, the tile's rows
// starting their windows on it or after it and masking the keys before them.
// Its pages end with its last row's (splash_attention_pages), and with no
// window begin at page 0.
template <uint Window> inline uint splash_attention_first_page(uint first_position) {
  return splash_attention_window_start<Window>(first_position) / SplashKvPageTokens;
}

// The scale of a head's scores, 1/sqrt(HeadDim), as the literal of each head
// dimension the kernels are compiled for.
template <uint HeadDim> constexpr float splash_attention_scale() {
  static_assert(HeadDim == 128 || HeadDim == 256, "a score scale for each compiled head dimension");
  return HeadDim == 256 ? 0.0625f : 0.08838834764831845f;
}

// The pages [begin, end) that every row of a tile sees wholly, so that their
// softmax tests no token: from the first page at or past the window start of
// the tile's last row (every row's window starts at or before it), to the
// page holding the tile's first row, below which every row sees every key, or
// a tree's anchor, from which a block's rows are masked by path (a lane's
// second tile starts past its anchor). FlashInfer's prefill masks only its
// iterations from mask_iteration on, and llama.cpp's fa_tensor.metal skips
// the mask of blocks it classed unmasked. Measured on an M5 Pro with 16 GPU
// cores (attention-sweep, INT8 KV, 32K and 98K): the 35B's split 1.4-3.9%
// faster in prefill and verify, the 27B's prefill 2.3% and its 16-row lanes'
// verify 0.7-1.5%; output bits unchanged.
struct SplashAttentionWholePages {
  uint begin;
  uint end;
};

template <uint Window>
inline SplashAttentionWholePages splash_attention_whole_pages(uint first_position, uint active_rows, bool tree,
                                                              uint anchor) {
  constexpr uint N = SplashKvPageTokens;
  const uint last_start = splash_attention_window_start<Window>(first_position + active_rows - 1);
  return {(last_start + N - 1) / N, (tree ? anchor : first_position) / N};
}

// The lanes of one fused row in the page loop: the softmax's, eight of a
// page's tokens each, and the query quantization's, so that each lane holds
// its own row's query scale.
constant uint SplashAttentionLanesPerRow = 4;

// One Page32 block of one tile: key-scaled scores (fp32, or the int32
// products of INT8 queries) times score_scale, the row's, become value-scaled
// bf16 probabilities and the row statistics advance. Four lanes own one
// fused row, eight consecutive tokens each, so the row maximum and sum are
// two xor shuffles instead of a 32-lane reduction per row and the eight bf16
// probabilities leave as one 16-byte store. Threads past 4 x FusedRows
// (kv4_g6: simdgroups 6 and 7) only take part in the caller's barriers.
// The float4 loads need 16-byte aligned slabs (alignas in the entries) and
// the page's 32 key and value scales as float4 vectors, in device memory
// (splash_q8_scale_index is a multiple of 32 floats and every region of an
// extent starts 64 KiB-aligned).
// Rows whose running maximum grew atomically set the shared boolean rescale
// flag. Existing threadgroup barriers separate reset, concurrent set, and
// read; relaxed atomics make the same-value writes safe without changing
// arithmetic.
// A tree's row (SplashAttentionTree) attends the history of its position's
// window, its position the anchor's plus its `depth`, and of the block's
// rows those its `ancestors` hold, itself among them; only a page that
// holds rows of the block tests them.
// Masked false is a page every row of the tile sees wholly
// (splash_attention_whole_pages): its tokens are all seen, untested.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Window, bool Quantized, bool Masked, typename Score>
inline void splash_attention_page_softmax(
    threadgroup const Score *scores, float score_scale, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    bool tree, uint anchor, uint ancestors, uint depth,
    uint thread_index) {
  constexpr uint N = SplashKvPageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = SplashAttentionLanesPerRow;
  static_assert(N == LanesPerRow * TokensPerLane, "four lanes of eight tokens span one page");
  static_assert(LanesPerRow * FusedRows <= 256,
                "one softmax lane per thread of the 256-thread tile");
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  bool seen[TokensPerLane];
  if constexpr (Masked) {
    const uint query_row = fused_row / QueryHeadsPerKVHead;
    const uint causal_end =
        committed_tokens + min(query_row, active_rows - 1) + 1;
    // The row sees the tokens of [start, its causal end): its window's, or
    // with no window its whole history's. Tokens and limit count from start,
    // so one unsigned comparison tests both ends: a token before start wraps
    // past every limit.
    const uint start = splash_attention_window_start<Window>(tree ? anchor + depth : causal_end - 1);
    const uint limit = min(visible_tokens, causal_end) - start;
    const uint token = token_start + column - start;
    // Bit j: the lane's token j is a row of the tree's block off the row's
    // path, which it does not see.
    uint off_path = 0;
    if (tree && token_start + N > anchor) {
#pragma unroll
      for (uint j = 0; j < TokensPerLane; ++j) {
        const uint slot = token_start + column + j;
        off_path |= uint(slot >= anchor && !((ancestors >> min(slot - anchor, 31u)) & 1u)) << j;
      }
    }
#pragma unroll
    for (uint j = 0; j < TokensPerLane; ++j)
      seen[j] = token + j < limit && !((off_path >> j) & 1u);
  } else {
#pragma unroll
    for (uint j = 0; j < TokensPerLane; ++j)
      seen[j] = true;
  }
  threadgroup const vec<Score, 4> *scores4 =
      reinterpret_cast<threadgroup const vec<Score, 4> *>(scores + fused_row * N + column);
  const uint vector = column / 4;
  float score[TokensPerLane];
  {
    float4 low = float4(scores4[0]), high = float4(scores4[1]);
    if constexpr (Quantized) {
      low *= key_scales[vector];
      high *= key_scales[vector + 1];
    }
    low *= score_scale;
    high *= score_scale;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = seen[j] ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = seen[j] ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  // Masked tokens stay exactly zero whatever their stored value scale holds.
  float4 low_scales(1.0f), high_scales(1.0f);
  if constexpr (Quantized) {
    low_scales = value_scales[vector];
    high_scales = value_scales[vector + 1];
  }
  const float4 low(seen[0] ? probability[0] * low_scales.x : 0.0f,
                   seen[1] ? probability[1] * low_scales.y : 0.0f,
                   seen[2] ? probability[2] * low_scales.z : 0.0f,
                   seen[3] ? probability[3] * low_scales.w : 0.0f);
  const float4 high(seen[4] ? probability[4] * high_scales.x : 0.0f,
                    seen[5] ? probability[5] * high_scales.y : 0.0f,
                    seen[6] ? probability[6] * high_scales.z : 0.0f,
                    seen[7] ? probability[7] * high_scales.w : 0.0f);
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(
      probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}

// The tile's queries as INT8 codes with one scale per fused row, by the
// int8-activation rule (kernels/common/int8_activations.h), for the INT8 x
// INT8 score product. A row's softmax lanes quantize it, a quarter of its
// dimensions each, so each returns its own row's scale: NaN for a row holding
// an infinity or a NaN, whose scores are then NaN, as with bf16 queries.
// Lanes past those of M rows return 0.
template <uint M, uint HeadDim>
inline float splash_attention_query_codes(device const bfloat *queries, device int8_t *codes,
                                          uint thread_index) {
  constexpr uint LanesPerRow = SplashAttentionLanesPerRow;
  static_assert(LanesPerRow == 4, "two xor shuffles reduce a row's four lanes");
  constexpr uint Quarter = HeadDim / LanesPerRow;
  if (thread_index >= LanesPerRow * M)
    return 0.0f;
  const uint first = thread_index / LanesPerRow * HeadDim + thread_index % LanesPerRow * Quarter;
  device const bfloat4 *source = reinterpret_cast<device const bfloat4 *>(queries + first);
  float maximum = 0.0f;
  uint finite = 1;
#pragma unroll
  for (uint i = 0; i < Quarter / 4; ++i) {
    const bfloat4 query = source[i];
    finite &= !int8_activation_not_finite(query);
    const float4 value = abs(float4(query));
    maximum = max(maximum, max(max(value.x, value.y), max(value.z, value.w)));
  }
  maximum = max(maximum, simd_shuffle_xor(maximum, 1));
  maximum = max(maximum, simd_shuffle_xor(maximum, 2));
  finite &= simd_shuffle_xor(finite, 1);
  finite &= simd_shuffle_xor(finite, 2);
  const Int8Block block = int8_activation_block(maximum);
  device char4 *target = reinterpret_cast<device char4 *>(codes + first);
#pragma unroll
  for (uint i = 0; i < Quarter / 4; ++i)
    target[i] = char4(int8_activation_code(float4(source[i]), block.inverse));
  return as_type<float>(int8_activation_bits(block.scale, finite != 0));
}

// One tile over its split's pages, whose K/V (INT8 with their scales, or
// BF16) it consumes in place as device tensor operands: a tile (or a part of
// one) of RowsPerTile rows. Queries are KV-head-major
// [kv head][row][query head in group][dimension], so the tile's fused rows
// form one contiguous M x D tensor. Each page is reached through its table
// entry (kv_extent.h).
// Three barriers per page order the score store, the softmax and the
// probability reads of PV.
// Int8Queries (INT8 K/V only) scores the tile's queries as INT8 codes
// against the INT8 keys, exact in int32 (|sum| < 2^22 at 256 dimensions);
// the codes live in the threadgroup's own partials slot, which it writes
// only after its last page.
// A tree's tile (SplashAttentionTree) masks each row to its path's rows of
// the block; its K/V, the store's, sit at the slots of their rows before
// any tile runs, so a row of a later tile reads its ancestors' from them.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim,
          uint Window, typename CacheElement, bool Int8Queries>
inline void splash_paged_attention_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    SplashAttentionTree tree, uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = HeadDim;
  const bool in_tree = tree.rows != nullptr;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint first_page = 0;
  if constexpr (Window != 0)
    first_page = splash_attention_first_page<Window>(
        splash_attention_window_position(in_tree, tree.anchor, committed_tokens));
  uint per_split = splash_attention_pages_per_split(pages - first_page, splits);
  uint page_begin = first_page + split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  static_assert(Quantized || !Int8Queries, "INT8 queries take INT8 keys");
  using QueryElement = conditional_t<Int8Queries, int8_t, bfloat>;
  using Score = conditional_t<Int8Queries, int, float>;
  device QueryElement *queries;
  float score_scale = splash_attention_scale<HeadDim>();
  if constexpr (Int8Queries) {
    queries = reinterpret_cast<device int8_t *>(partials + slot * M * D);
    score_scale *= splash_attention_query_codes<M, HeadDim>(tile_queries, queries, thread_index);
  } else {
    queries = tile_queries;
  }
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  // A tree row's ancestors and depth, for its softmax lanes.
  uint ancestors = 0, depth = 0;
  if (in_tree && thread_index < SplashAttentionLanesPerRow * M) {
    const SplashTreeRow place =
        tree.rows[min(thread_index / SplashAttentionLanesPerRow / QueryHeadsPerKVHead, active_rows - 1)];
    ancestors = place.ancestors;
    depth = place.depth;
  }
  // The query codes are device writes the whole threadgroup reads.
  threadgroup_barrier(Int8Queries ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                  : mem_flags::mem_threadgroup);

  threadgroup Score *page_score_tile = reinterpret_cast<threadgroup Score *>(scores);
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(page_score_tile, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto key_type = tensor(static_cast<device CacheElement *>(nullptr),
                         dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device CacheElement *>(nullptr),
                           dextents<int, 2>{N, D}, array<int, 2>{1, N});
  const SplashKvAddressing<KVHeads, HeadDim, CacheElement> addressing(kv, kv_head);
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto v0 = value_type.template slice<N, D>(0, 0);
  // QK writes a complete page score tile, the softmax applies an INT8 page's
  // key scales and PV accumulates the running output.
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  // Uniform capacity partitions the logical tile across eight 32-lane groups.
  // Equality proves no padding; otherwise only valid entries may be accessed.
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  const SplashAttentionWholePages whole =
      splash_attention_whole_pages<Window>(committed_tokens, active_rows, in_tree, tree.anchor);
  for (uint page = page_begin; page < page_end; ++page) {
    const SplashKvPageTensors<CacheElement> tensors =
        addressing.page(page_table[page]);
    uint token_start = page * N;
    auto kt = tensor(tensors.keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(tensors.values, dextents<int, 2>{N, D}, array<int, 2>{1, N});
    device const float *key_scales = tensors.key_scales;
    device const float *value_scales = tensors.value_scales;

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), Score>();
    // One full-dimension product initializes the score CT through MPP.
    auto ks = kt.template slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.template slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (page >= whole.begin && page < whole.end)
      splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, false>(
          page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
          reinterpret_cast<device const float4 *>(key_scales),
          reinterpret_cast<device const float4 *>(value_scales), token_start,
          visible_tokens, committed_tokens, active_rows, in_tree, tree.anchor, ancestors, depth, thread_index);
    else
      splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, true>(
          page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
          reinterpret_cast<device const float4 *>(key_scales),
          reinterpret_cast<device const float4 *>(value_scales), token_start,
          visible_tokens, committed_tokens, active_rows, in_tree, tree.anchor, ancestors, depth, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto vs = vt.template slice<N, D>(0, 0);
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// The sink of a fused row's query head among its group's, or with none a
// logit below every score.
template <uint QueryHeadsPerKVHead> inline float splash_row_sink(uint) { return -INFINITY; }

template <uint QueryHeadsPerKVHead>
inline float splash_row_sink(uint fused_row, device const float *group_sinks) {
  return group_sinks[fused_row % QueryHeadsPerKVHead];
}

// Partials are [slot][fused row][dimension]; statistics are
// [slot][fused row]{max, sum}. The callers lay a sequence's slots out from its
// first_slot as [tile][KV head][part][split], a sparse tile being one part.
// Combines the first `written` splits of one fused row in split order, a
// threadgroup of HeadDim threads, one per output dimension. The statistics
// are shared by all output dimensions: their weights are computed once per
// row, the splits' maxima by the threads at their split's index (written
// never exceeds HeadDim), then each lane streams one dimension of the
// partials; the row's value comes back to each thread, its dimension's, and a
// row past the tile's active rows is zero. A threadgroup owns one fused row,
// so it returns uniformly.
// A layer's learned sinks come as group_sinks, a logit per query head of the
// tile's KV head (its GQA group), and none without: the row's query head's
// sink is one more key, with no value, of the softmax (the scores' scale
// already applied), so it enters the maximum and, after the splits in their
// order, the denominator.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim, class... GroupSinks>
inline bfloat splash_attention_combine_splits(
    device const float *partials, device const float *statistics, uint active_rows, uint written,
    ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values,
    GroupSinks... group_sinks) {
  constexpr bool Sinks = sizeof...(GroupSinks) != 0;
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = HeadDim;
  static_assert(D >= SPLASH_ATTENTION_MAXIMUM_SPLITS, "a thread per split");
  if (fused_row / QueryHeadsPerKVHead >= active_rows)
    return bfloat(0.0f);
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const ulong stat = ((head_slot + thread_index) * M + fused_row) * 2;
  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0) group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float split_maximum = simd_max(lane < D / 32 ? group_values[lane] : -INFINITY);
  const float sink = splash_row_sink<QueryHeadsPerKVHead>(fused_row, group_sinks...);
  const float row_maximum = Sinks ? max(split_maximum, sink) : split_maximum;
  if (thread_index < written)
    weights[thread_index] = fast::exp(maximum - row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Keep both accumulations in the original split order. A parallel sum of
  // the denominator changes speculative acceptance on real model prompts.
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    numerator += weight *
        partials[((head_slot + split) * M + fused_row) * D + thread_index];
    denominator += weight * statistics[stat + 1];
  }
  if constexpr (Sinks)
    denominator += fast::exp(sink - row_maximum);
  return bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

// The sinks of KV head kv_head's query heads (its GQA group) among a
// layer's, one per query head.
template <uint QueryHeadsPerKVHead>
inline device const float *splash_group_sinks(uint kv_head, device const float *sinks) {
  return sinks + kv_head * QueryHeadsPerKVHead;
}

// Fused row `fused_row` of a tile run as Parts parts
// (splash_attention_tile_part) combines the splits of its part's page loop:
// those that own at least one page wrote their partials, and the partition
// is recomputed here, of the same Window and, for a tree's tile, from the
// same anchor. A part past the tile's active rows ran no split; its rows are
// zero.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Parts, uint HeadDim,
          uint Window, class... GroupSinks>
inline bfloat splash_attention_reduce_row(
    device const float *partials, device const float *statistics, SplashAttentionTile whole, uint slot_stride,
    bool tree, uint anchor, uint fused_row, uint thread_index, threadgroup float *weights,
    threadgroup float *group_values, GroupSinks... group_sinks) {
  constexpr uint PartRows = RowsPerTile / Parts;
  constexpr uint PartM = PartRows * QueryHeadsPerKVHead;
  const SplashAttentionTile tile =
      splash_attention_tile_part<QueryHeadsPerKVHead, RowsPerTile, Parts, HeadDim>(
          whole, fused_row / PartM, slot_stride);
  if (!tile.active)
    return bfloat(0.0f);
  uint pages = splash_attention_pages(tile.first_position + tile.active_rows);
  if constexpr (Window != 0)
    pages -= splash_attention_first_page<Window>(splash_attention_window_position(tree, anchor, tile.first_position));
  const uint per_split = splash_attention_pages_per_split(pages, tile.splits);
  const uint written = (pages + per_split - 1) / per_split;
  return splash_attention_combine_splits<QueryHeadsPerKVHead, PartRows, HeadDim>(
      partials, statistics, tile.active_rows, written, tile.first_slot, fused_row % PartM, thread_index, weights,
      group_values, group_sinks...);
}
// ---- end runtime/metal/kernels/common/paged_attention_tile.h ----
// ---- begin runtime/metal/kernels/common/rms_inverse.h (Splash 2afc733, Apache-2.0) ----

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

// Relative attention (ops::RelativeAttention, metal/abi/RelativeAttention.h):
// the paged attention of kernels/common/paged_attention_tile.h, over the same
// pages, tiles, splits and reduce, whose scores also take a bias of each
// key's backward distance mixed by the row's relative states, and whose
// softmax takes a sink of the row's own for each query head. Its page
// softmax, page loop, combine and output row are this file's, each written
// as the shared one is (the same order and names) with these terms added, so
// that the two read side by side; every other helper is the shared one.

// A projection row of relative attention, a row each of the command's packed
// rows (ops::RelativeAttentionRows): every query head's queries, then their
// output gates, their relative states and their sinks, each a block. The
// projection writing them has whole tiles of GGUF_TILE_COLUMNS outputs
// (ops::Linear), so its rows are Stride apart, the blocks' Width rounded up
// to a tile; the columns past Width are padding, which nothing reads.
template <uint QHeads, uint HeadDim> struct RelativeAttentionRow {
  static constexpr constant uint States = SPLASH_RELATIVE_ATTENTION_STATES;
  static constexpr constant uint GateOffset = QHeads * HeadDim;
  static constexpr constant uint StateOffset = 2 * QHeads * HeadDim;
  static constexpr constant uint SinkOffset = StateOffset + QHeads * States;
  static constexpr constant uint Width = SinkOffset + QHeads;
  static constexpr constant uint Stride = (Width + GGUF_TILE_COLUMNS - 1) / GGUF_TILE_COLUMNS * GGUF_TILE_COLUMNS;
};

// ---- Prepare --------------------------------------------------------------

// The sum of each thread's `value` over a threadgroup of Simdgroups
// simdgroups, in simdgroup order, returned to every thread; `reductions`
// holds a partial per simdgroup, which the threadgroup does not reuse.
template <ushort Simdgroups>
inline float relative_attention_sum(float value, threadgroup float *reductions, uint lane, uint simd_group) {
#pragma clang fp reassociate(off)
  value = simd_sum(value);
  if (lane == 0)
    reductions[simd_group] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < Simdgroups; ++i)
    total += reductions[i];
  return total;
}

// The factor that clips a head whose squares sum to `squares` to an L2 norm
// of at most `clip`, in fp32 as the source's: min(1, clip / (|x| + epsilon)).
inline float relative_attention_clip(float squares, float clip) {
  return min(clip / (sqrt(squares) + SPLASH_RELATIVE_ATTENTION_CLIP_EPSILON), 1.0f);
}

// A position-scaled layer's factor of a row at `position`, in fp32 as the
// source's.
inline float relative_attention_log_scale(uint position) {
  return 1.0f + SPLASH_RELATIVE_ATTENTION_LOG_SCALE_ALPHA *
                    precise::log(max(float(position + 1) / SPLASH_RELATIVE_ATTENTION_LOG_SCALE_FLOOR, 1.0f));
}

// A threadgroup of HeadDim threads per (head, row, sequence) of a dispatch's
// sequences (SplashAttentionParams), as full_qkv_storage_phase
// (attention_qkv_prepare.h): query heads, then K heads, whose threadgroups
// also take the V head of their index; a row past its sequence's rows
// returns. A row's projection row is `projection` (RelativeAttentionRow),
// its convolved K and V rows `keys` and `values` (KHeads x HeadDim each), at
// its packed row.
// A query head's queries are scaled by query_scale ([query head][dimension]),
// clipped and rounded to bf16, as the source's projection returns them, and,
// PositionScaled, scaled again by the row's log scale and rounded; staged as
// the paged attention's ([KV head][row][query head in group][dimension]).
// Its relative states, the first simdgroup's lanes, are scaled by
// relative_scale ([query head][state]), clipped and scaled alike, and staged
// as F32 in the same order (States a fused row); its sink is staged as F32
// (one a fused row). A row's position is its sequence's committed tokens plus
// its row, or a tree row's (SplashAttentionParams::trees) its anchor's plus
// its depth.
// A K head's key is clipped, its V head's value normalized (times the inverse
// RMS of the head, unweighted), each rounded to bf16, and both stored in the
// row's page slot (splash_store_kv_slot) through its sequence's page table,
// before any tile reads them.
template <uint QHeads, uint KHeads, uint HeadDim, bool PositionScaled, typename CacheElement>
inline void relative_attention_prepare_phase(
    device const bfloat *projection, device const bfloat *keys, device const bfloat *values,
    device const float *query_scale, device const float *relative_scale, device bfloat *queries,
    device float *relative_states, device float *sinks, device const SplashKvPage *page_table,
    device const SplashTreeRow *tree, device const SplashLaneRows *layout, constant SplashAttentionParams &params,
    threadgroup float *reductions, threadgroup float *maxima, uint3 group, uint thread_index, uint lane,
    uint simd_group) {
  static_assert(QHeads % KHeads == 0);
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Group = QHeads / KHeads, States = Row::States;
  constexpr ushort Simdgroups = HeadDim / 32;
  static_assert(States <= 32, "one simdgroup clips a head's relative states");
  const uint sequence_row = group.y;
  constant SplashAttentionSequence &sequence = params.sequence[group.z];
  const bool query = group.x < QHeads;
  const bool stores =
      !query && splash_attention_rows_valid(params, sequence) && sequence_row < sequence.row_bound;
  const SplashKvPage page = stores ? splash_store_page(page_table, sequence, sequence_row) : SPLASH_KV_PAGE_ABSENT;
  const SplashAttentionRows rows = splash_attention_rows(layout, params, group.z);
  if (sequence_row >= rows.rows || (!query && !stores))
    return;
  const ulong packed = ulong(rows.first) + sequence_row;
  if (!query) {
    const uint head = group.x - QHeads;
    const ulong element = packed * (KHeads * HeadDim) + head * HeadDim + thread_index;
    const float key = float(keys[element]), value = float(values[element]);
    const float clip = relative_attention_clip(relative_attention_sum<Simdgroups>(key * key, reductions, lane,
                                                                                 simd_group),
                                               SPLASH_RELATIVE_ATTENTION_QK_CLIP);
    const float inverse = rms_inverse_of_sums<Simdgroups>(value * value, HeadDim, reductions + Simdgroups,
                                                          thread_index, lane, simd_group);
    splash_store_kv_slot<KHeads, HeadDim, CacheElement, false>(page, sequence, params.kv, maxima, head,
                                                               sequence_row, bfloat(key * clip),
                                                               bfloat(value * inverse), thread_index, lane,
                                                               simd_group);
    return;
  }
  const uint head = group.x;
  float log_scale = 1.0f;
  if constexpr (PositionScaled)
    log_scale = relative_attention_log_scale(sequence.committed_tokens +
                                             (params.trees ? uint(tree[packed].depth) : sequence_row));
  device const bfloat *row = projection + packed * Row::Stride;
  const float element = float(row[head * HeadDim + thread_index]) * query_scale[head * HeadDim + thread_index];
  const float clip = relative_attention_clip(
      relative_attention_sum<Simdgroups>(element * element, reductions, lane, simd_group),
      SPLASH_RELATIVE_ATTENTION_QK_CLIP);
  bfloat result = bfloat(element * clip);
  if constexpr (PositionScaled)
    result = bfloat(float(result) * log_scale);
  const uint kv_head = head / Group, local_head = head % Group;
  const ulong fused =
      (ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + sequence_row) * Group + local_head;
  queries[fused * HeadDim + thread_index] = result;
  if (simd_group != 0)
    return;
  const float state =
      lane < States ? float(row[Row::StateOffset + head * States + lane]) * relative_scale[head * States + lane]
                    : 0.0f;
  bfloat clipped =
      bfloat(state * relative_attention_clip(simd_sum(state * state), SPLASH_RELATIVE_ATTENTION_STATE_CLIP));
  if constexpr (PositionScaled)
    clipped = bfloat(float(clipped) * log_scale);
  if (lane < States)
    relative_states[fused * States + lane] = float(clipped);
  if (lane == 0)
    sinks[fused] = float(row[Row::SinkOffset + head]);
}

// ---- Page loop ------------------------------------------------------------

// A softmax lane's bias: its fused row's relative states, in registers for
// the tile's whole page loop, the layer's table, and the row's position. Each
// of the lane's eight keys at a backward distance d below the extent adds the
// states' dot with table row d: 16 products a key in the lane's registers,
// against a table that stays in cache (64 KiB). A table of each row's 1024
// biases a head, written before the tiles, would cost 96 KiB a row (200 MB a
// 2048-row chunk) and a dispatch more an application. A key past the row
// wraps past the extent: it is masked anyway.
// A tree row's keys of its block sit at the slots of their rows, not of
// their positions: an ancestor at block row j is at depth j's place on the
// row's path, the count of its path's rows before j (metal/abi/VerifyTree.h:
// a path's rows ascend), so its distance is the row's depth less that count.
// `block` is a tree's anchor slot, and past every slot for any other tile.
struct RelativeAttentionLane {
  device const float4 *table;
  float4 state[SPLASH_RELATIVE_ATTENTION_STATES / 4];
  uint position;
  uint block;
  uint ancestors;
  uint depth;
};

inline void relative_attention_add_bias(thread float *score, uint first_key, RelativeAttentionLane lane) {
  static_assert(SPLASH_RELATIVE_ATTENTION_STATES == 16, "four float4 states a row");
#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    const uint key = first_key + j;
    const uint distance =
        key >= lane.block ? lane.depth - popcount(lane.ancestors & ((1u << min(key - lane.block, 31u)) - 1u))
                          : lane.position - key;
    if (distance < SPLASH_RELATIVE_ATTENTION_EXTENT) {
      device const float4 *bias = lane.table + ulong(distance) * 4;
      score[j] += dot(lane.state[0], bias[0]) + dot(lane.state[1], bias[1]) + dot(lane.state[2], bias[2]) +
                  dot(lane.state[3], bias[3]);
    }
  }
}

// splash_attention_page_softmax with the bias: Biased, a page that holds a
// key within the extent of some row of the tile, adds the lane's to its
// scaled scores (after an INT8 page's key scales and the query codes' scale);
// a page below every row's extent takes none.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Window, bool Quantized, bool Masked, bool Biased,
          typename Score>
inline void relative_attention_page_softmax(
    threadgroup const Score *scores, float score_scale, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    bool tree, uint anchor, uint ancestors, uint depth,
    RelativeAttentionLane relative, uint thread_index) {
  constexpr uint N = SplashKvPageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = SplashAttentionLanesPerRow;
  static_assert(N == LanesPerRow * TokensPerLane, "four lanes of eight tokens span one page");
  static_assert(LanesPerRow * FusedRows <= 256,
                "one softmax lane per thread of the 256-thread tile");
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  bool seen[TokensPerLane];
  if constexpr (Masked) {
    const uint query_row = fused_row / QueryHeadsPerKVHead;
    const uint causal_end =
        committed_tokens + min(query_row, active_rows - 1) + 1;
    const uint start = splash_attention_window_start<Window>(tree ? anchor + depth : causal_end - 1);
    const uint limit = min(visible_tokens, causal_end) - start;
    const uint token = token_start + column - start;
    uint off_path = 0;
    if (tree && token_start + N > anchor) {
#pragma unroll
      for (uint j = 0; j < TokensPerLane; ++j) {
        const uint slot = token_start + column + j;
        off_path |= uint(slot >= anchor && !((ancestors >> min(slot - anchor, 31u)) & 1u)) << j;
      }
    }
#pragma unroll
    for (uint j = 0; j < TokensPerLane; ++j)
      seen[j] = token + j < limit && !((off_path >> j) & 1u);
  } else {
#pragma unroll
    for (uint j = 0; j < TokensPerLane; ++j)
      seen[j] = true;
  }
  threadgroup const vec<Score, 4> *scores4 =
      reinterpret_cast<threadgroup const vec<Score, 4> *>(scores + fused_row * N + column);
  const uint vector = column / 4;
  float score[TokensPerLane];
  {
    float4 low = float4(scores4[0]), high = float4(scores4[1]);
    if constexpr (Quantized) {
      low *= key_scales[vector];
      high *= key_scales[vector + 1];
    }
    low *= score_scale;
    high *= score_scale;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  if constexpr (Biased)
    relative_attention_add_bias(score, token_start + column, relative);
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = seen[j] ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = seen[j] ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  float4 low_scales(1.0f), high_scales(1.0f);
  if constexpr (Quantized) {
    low_scales = value_scales[vector];
    high_scales = value_scales[vector + 1];
  }
  const float4 low(seen[0] ? probability[0] * low_scales.x : 0.0f,
                   seen[1] ? probability[1] * low_scales.y : 0.0f,
                   seen[2] ? probability[2] * low_scales.z : 0.0f,
                   seen[3] ? probability[3] * low_scales.w : 0.0f);
  const float4 high(seen[4] ? probability[4] * high_scales.x : 0.0f,
                    seen[5] ? probability[5] * high_scales.y : 0.0f,
                    seen[6] ? probability[6] * high_scales.z : 0.0f,
                    seen[7] ? probability[7] * high_scales.w : 0.0f);
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(
      probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}

// The page-softmax call of one page: Masked unless every row of the tile
// sees it wholly, Biased from the tile's first biased page on.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Window, bool Quantized, bool Biased, typename Score>
inline void relative_attention_page(
    bool masked, threadgroup const Score *scores, float score_scale, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, device const float *key_scales, device const float *value_scales,
    uint token_start, uint visible_tokens, uint committed_tokens, uint active_rows, bool tree, uint anchor,
    uint ancestors, uint depth, RelativeAttentionLane relative, uint thread_index) {
  if (masked)
    relative_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, true, Biased>(
        scores, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales), reinterpret_cast<device const float4 *>(value_scales),
        token_start, visible_tokens, committed_tokens, active_rows, tree, anchor, ancestors, depth, relative,
        thread_index);
  else
    relative_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, false, Biased>(
        scores, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales), reinterpret_cast<device const float4 *>(value_scales),
        token_start, visible_tokens, committed_tokens, active_rows, tree, anchor, ancestors, depth, relative,
        thread_index);
}

// The pages a tile's history split `split` of `splits` reads, as the split,
// the reduce and the single pass derive them: the balanced partition of the
// pages from its window's first (splash_attention_first_page, from its first
// row's position or a tree's anchor's) to its last row's, the split's run
// [begin, end), empty past the pages, and the runs that hold pages
// (`written`), whose partials a combine reads.
struct RelativeAttentionPages {
  uint begin;
  uint end;
  uint written;
};

template <uint Window>
inline RelativeAttentionPages relative_attention_pages(uint first_position, uint active_rows, bool tree, uint anchor,
                                                       uint splits, uint split) {
  const uint pages = splash_attention_pages(first_position + active_rows);
  uint first_page = 0;
  if constexpr (Window != 0)
    first_page = splash_attention_first_page<Window>(splash_attention_window_position(tree, anchor, first_position));
  const uint per_split = splash_attention_pages_per_split(pages - first_page, splits);
  const uint begin = first_page + split * per_split;
  return {begin, min(pages, begin + per_split), (pages - first_page + per_split - 1) / per_split};
}

// splash_paged_attention_tile with the bias, over a split's nonempty run of
// pages (relative_attention_pages): the tile's fused rows' relative states
// (`states`, staged as their queries) and the layer's bias table (`table`).
// Its softmax lanes load their rows' states and positions once; the pages
// from the first that holds a key within the extent of the tile's first row's
// position (a tree's anchor's, as its window's) take the bias, the pages
// below it, past every row's extent, none. Int8Queries writes the tile's
// query codes to `query_codes` (M x HeadDim bytes the threadgroup owns).
// Then `store` stores the running output of the tile's fused rows (an
// M x HeadDim cooperative tensor, its argument, every element of which
// running_full says is valid) and their statistics, which row_max and row_sum
// hold.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim,
          uint Window, typename CacheElement, bool Int8Queries, class Store>
inline void relative_attention_page_loop(
    device bfloat *tile_queries, device int8_t *query_codes, device const float *states,
    device const float *table, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, RelativeAttentionPages run,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    SplashAttentionTree tree, uint thread_index, Store store) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = HeadDim;
  constexpr uint States = SPLASH_RELATIVE_ATTENTION_STATES;
  const bool in_tree = tree.rows != nullptr;
  const uint visible_tokens = committed_tokens + active_rows;
  static_assert(Quantized || !Int8Queries, "INT8 queries take INT8 keys");
  using QueryElement = conditional_t<Int8Queries, int8_t, bfloat>;
  using Score = conditional_t<Int8Queries, int, float>;
  device QueryElement *queries;
  float score_scale = splash_attention_scale<HeadDim>();
  if constexpr (Int8Queries) {
    queries = query_codes;
    score_scale *= splash_attention_query_codes<M, HeadDim>(tile_queries, queries, thread_index);
  } else {
    queries = tile_queries;
  }
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  uint ancestors = 0, depth = 0;
  if (in_tree && thread_index < SplashAttentionLanesPerRow * M) {
    const SplashTreeRow place =
        tree.rows[min(thread_index / SplashAttentionLanesPerRow / QueryHeadsPerKVHead, active_rows - 1)];
    ancestors = place.ancestors;
    depth = place.depth;
  }
  // The lane's fused row's states and position, a row past the tile's active
  // rows its last's, as its depth.
  RelativeAttentionLane lane{};
  if (thread_index < SplashAttentionLanesPerRow * M) {
    const uint fused_row = thread_index / SplashAttentionLanesPerRow;
    const uint row = min(fused_row / QueryHeadsPerKVHead, active_rows - 1);
    device const float4 *row_states = reinterpret_cast<device const float4 *>(
        states + (row * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead) * States);
#pragma unroll
    for (uint i = 0; i < States / 4; ++i)
      lane.state[i] = row_states[i];
    lane.table = reinterpret_cast<device const float4 *>(table);
    lane.position = in_tree ? tree.anchor + depth : committed_tokens + row;
    lane.block = in_tree ? tree.anchor : 0xffffffffu;
    lane.ancestors = ancestors;
    lane.depth = depth;
  }
  // The first page that holds a key within the extent of the tile's first
  // row: keys from its position less Extent - 1 on.
  const uint biased_page = splash_attention_first_page<SPLASH_RELATIVE_ATTENTION_EXTENT - 1>(
      splash_attention_window_position(in_tree, tree.anchor, committed_tokens));
  threadgroup_barrier(Int8Queries ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                  : mem_flags::mem_threadgroup);

  threadgroup Score *page_score_tile = reinterpret_cast<threadgroup Score *>(scores);
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(page_score_tile, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto key_type = tensor(static_cast<device CacheElement *>(nullptr),
                         dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device CacheElement *>(nullptr),
                           dextents<int, 2>{N, D}, array<int, 2>{1, N});
  const SplashKvAddressing<KVHeads, HeadDim, CacheElement> addressing(kv, kv_head);
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto v0 = value_type.template slice<N, D>(0, 0);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  const SplashAttentionWholePages whole =
      splash_attention_whole_pages<Window>(committed_tokens, active_rows, in_tree, tree.anchor);
  for (uint page = run.begin; page < run.end; ++page) {
    const SplashKvPageTensors<CacheElement> tensors =
        addressing.page(page_table[page]);
    uint token_start = page * N;
    auto kt = tensor(tensors.keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(tensors.values, dextents<int, 2>{N, D}, array<int, 2>{1, N});

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), Score>();
    auto ks = kt.template slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.template slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool masked = !(page >= whole.begin && page < whole.end);
    if (page >= biased_page)
      relative_attention_page<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, true>(
          masked, page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
          tensors.key_scales, tensors.value_scales, token_start, visible_tokens, committed_tokens, active_rows,
          in_tree, tree.anchor, ancestors, depth, lane, thread_index);
    else
      relative_attention_page<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, false>(
          masked, page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
          tensors.key_scales, tensors.value_scales, token_start, visible_tokens, committed_tokens, active_rows,
          in_tree, tree.anchor, ancestors, depth, lane, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto vs = vt.template slice<N, D>(0, 0);
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  store(running, running_full);
}

// ---- Reduce ---------------------------------------------------------------

// A fused row's combine of its splits and its sink, which the reduce
// (relative_attention_combine_splits, relative_attention_output_row), a
// decode step's single pass (relative_attention_combine_weights, _row,
// relative_attention_output_quad) and the direct kernels, of one split
// (relative_attention_direct_phase), each compute in the same order and
// rounding: each split's weight, from its maximum and the row's over its
// splits' and its sink (relative_attention_split_weight); the denominator,
// the splits' weighted sums in their order and then the sink's term
// (relative_attention_denominator); each element's quotient of its splits'
// weighted outputs and the denominator, 0 without one, rounded to bf16; and
// its gated output (relative_attention_gated). The weight and the
// denominator are computed uncontracted (contract(off)): a row of one split
// rounds its split's product before it adds the sink's term, as the loop
// over splits does, where one fused multiply-add of the two gave other bits.
// Each is always inlined, so that its callers compile as they did with its
// expression in place.

// The weight of a split of maximum `maximum` in a row of maximum
// `row_maximum`.
__attribute__((always_inline)) inline float relative_attention_split_weight(float maximum, float row_maximum) {
#pragma clang fp contract(off)
  return fast::exp(maximum - row_maximum);
}

// A row's denominator: its splits' weighted sums, `splits`, then its sink's
// term.
__attribute__((always_inline)) inline float relative_attention_denominator(float splits, float sink,
                                                                           float row_maximum) {
#pragma clang fp contract(off)
  return splits + fast::exp(sink - row_maximum);
}

// An element's gated output: its value as the combine rounds it, `value`,
// times the sigmoid of its query head's `gate`, rounded to bf16.
__attribute__((always_inline)) inline bfloat relative_attention_gated(float value, float gate) {
  return bfloat(value * splash_sigmoid(gate));
}

// splash_attention_combine_splits with the row's own sink, `sink` (the
// scores' scale is not applied to it): one more key with no value, in the
// maximum and, after the splits in their order, the denominator.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim>
inline bfloat relative_attention_combine_splits(
    device const float *partials, device const float *statistics, uint active_rows, uint written,
    ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values, float sink) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = HeadDim;
  static_assert(D >= SPLASH_ATTENTION_MAXIMUM_SPLITS, "a thread per split");
  if (fused_row / QueryHeadsPerKVHead >= active_rows)
    return bfloat(0.0f);
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const ulong stat = ((head_slot + thread_index) * M + fused_row) * 2;
  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0) group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float split_maximum = simd_max(lane < D / 32 ? group_values[lane] : -INFINITY);
  const float row_maximum = max(split_maximum, sink);
  if (thread_index < written)
    weights[thread_index] = relative_attention_split_weight(maximum, row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Both accumulations in the original split order, as the shared combine's.
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    numerator += weight *
        partials[((head_slot + split) * M + fused_row) * D + thread_index];
    denominator += weight * statistics[stat + 1];
  }
  denominator = relative_attention_denominator(denominator, sink, row_maximum);
  return bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

// The attention's output of one fused row, as splash_attention_output_row
// (attention_gate.h) writes a gated one: `value` times the sigmoid of its
// query head's gate in its projection row (RelativeAttentionRow's gate
// block), at row `output` of `hidden` and, with Table, of the
// out-projection's Table16 input and its sums.
template <uint QHeads, uint HeadDim, bool Table>
inline void relative_attention_output_row(bfloat value, device const bfloat *projection, device bfloat *hidden,
                                          device bfloat *table, device float *sums, ulong packed, ulong output,
                                          uint query_head, uint dimension, bool writes, threadgroup bfloat *pairs) {
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Width = QHeads * HeadDim;
  static_assert(HeadDim % 64 == 0, "a head's elements fill whole simdgroups of pairs");
  bfloat element = value;
  if (writes) {
    const float gate = float(projection[packed * Row::Stride + Row::GateOffset + query_head * HeadDim + dimension]);
    element = relative_attention_gated(float(value), gate);
    hidden[output * Width + query_head * HeadDim + dimension] = element;
  }
  if constexpr (Table) {
    pairs[dimension] = element;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (writes && dimension < HeadDim / 2) {
      const uint first = query_head * HeadDim + 2 * dimension;
      gguf_sg::Table16::write(table + output / 8 * Width * 8,
                              sums + output / 8 * gguf_sg::Table16::sums_per_tile(Width), Width, first / 64,
                              output % 8, dimension % 32, pairs[2 * dimension], pairs[2 * dimension + 1]);
    }
  }
}

// ---- Single pass ----------------------------------------------------------

// A decode step's attention of a layer in one dispatch (ops::RelativeAttention
// ::addDecode, relative_attention_decode_phase): each threadgroup of a lane's
// tile prepares what its split reads itself, then runs the split's page loop,
// and the split that finishes last combines the tile's splits as the reduce
// does. These compute every value as the prepare and the reduce do, in the same
// order, a simdgroup a head where they spread a head over a threadgroup: lane l
// holds elements l + 32 i of a head, whose squares it adds a block of 32 at a
// time (simd_sum, the prepare's per-simdgroup partial), block after block.

// The prepare of a query head's row (relative_attention_prepare_phase's query
// threadgroup) by one simdgroup: its queries, scaled, clipped and rounded,
// and, PositionScaled, scaled by the log scale of the row's `position` and
// rounded again, to `queries` (HeadDim bf16), and its relative states to
// `states` (F32, the first States lanes'), from the row's projection row
// `row` (RelativeAttentionRow).
template <uint QHeads, uint HeadDim, bool PositionScaled>
inline void relative_attention_stage_head(device const bfloat *row, device const float *query_scale,
                                          device const float *relative_scale, uint head, uint position,
                                          device bfloat *queries, device float *states, uint lane) {
#pragma clang fp reassociate(off)
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Blocks = HeadDim / 32, States = Row::States;
  float element[Blocks];
  float squares = 0.0f;
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    const uint dimension = head * HeadDim + 32 * i + lane;
    element[i] = float(row[dimension]) * query_scale[dimension];
    squares += simd_sum(element[i] * element[i]);
  }
  const float clip = relative_attention_clip(squares, SPLASH_RELATIVE_ATTENTION_QK_CLIP);
  float log_scale = 1.0f;
  if constexpr (PositionScaled)
    log_scale = relative_attention_log_scale(position);
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    bfloat result = bfloat(element[i] * clip);
    if constexpr (PositionScaled)
      result = bfloat(float(result) * log_scale);
    queries[32 * i + lane] = result;
  }
  const float state =
      lane < States ? float(row[Row::StateOffset + head * States + lane]) * relative_scale[head * States + lane]
                    : 0.0f;
  bfloat clipped =
      bfloat(state * relative_attention_clip(simd_sum(state * state), SPLASH_RELATIVE_ATTENTION_STATE_CLIP));
  if constexpr (PositionScaled)
    clipped = bfloat(float(clipped) * log_scale);
  if (lane < States)
    states[lane] = float(clipped);
}

// The prepare of a K head's row and its V head's (relative_attention_prepare_
// phase's K threadgroup) by one simdgroup: the key clipped and the value
// normalized, each rounded to bf16, from the row's convolved K and V heads at
// `element` of `keys` and `values`, stored in slot `token` of the page entry
// `page` as splash_store_kv_slot stores them (INT8 codes at the row's largest
// magnitude of each, and their scales).
template <uint KHeads, uint HeadDim, typename CacheElement>
inline void relative_attention_store_head(device const bfloat *keys, device const bfloat *values, ulong element,
                                          SplashKvPage page, SplashKvLayer kv, uint head, uint token, uint lane) {
#pragma clang fp reassociate(off)
  constexpr uint Blocks = HeadDim / 32;
  float key[Blocks], value[Blocks];
  float key_squares = 0.0f, value_squares = 0.0f;
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    key[i] = float(keys[element + 32 * i + lane]);
    value[i] = float(values[element + 32 * i + lane]);
    key_squares += simd_sum(key[i] * key[i]);
    value_squares += simd_sum(value[i] * value[i]);
  }
  const float clip = relative_attention_clip(key_squares, SPLASH_RELATIVE_ATTENTION_QK_CLIP);
  const float inverse = rsqrt(value_squares / HeadDim + kRmsEpsilon);
  bfloat clipped[Blocks], normalized[Blocks];
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    clipped[i] = bfloat(key[i] * clip);
    normalized[i] = bfloat(value[i] * inverse);
  }
  const SplashKvPageTensors<CacheElement> slab = SplashKvAddressing<KHeads, HeadDim, CacheElement>(kv, head).page(page);
  if constexpr (is_same<CacheElement, bfloat>::value) {
#pragma unroll
    for (uint i = 0; i < Blocks; ++i) {
      slab.keys[splash_kv_key_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] = clipped[i];
      slab.values[splash_kv_value_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] = normalized[i];
    }
  } else {
    float2 maximum = 0.0f;
#pragma unroll
    for (uint i = 0; i < Blocks; ++i)
      maximum = max(maximum, abs(float2(float(clipped[i]), float(normalized[i]))));
    maximum = float2(simd_max(maximum.x), simd_max(maximum.y));
#pragma unroll
    for (uint i = 0; i < Blocks; ++i) {
      slab.keys[splash_kv_key_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] =
          splash_store_code(float(clipped[i]), maximum.x);
      slab.values[splash_kv_value_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] =
          splash_store_code(float(normalized[i]), maximum.y);
    }
    if (lane == 0) {
      const ulong scale_index = splash_q8_scale_index<KHeads>(0, 0, token);
      slab.key_scales[scale_index] = maximum.x == 0.0f ? 0.0f : maximum.x / 127.0f;
      slab.value_scales[scale_index] = maximum.y == 0.0f ? 0.0f : maximum.y / 127.0f;
    }
  }
}

// relative_attention_combine_splits of a tile's fused rows by the split that
// arrives last, in two phases over the published splits' partials and
// statistics (split_reduce.h), read coherently. First a thread a fused row:
// its maximum over its splits' and its sink, each split's weight into
// `weights` ([fused row][split]) and its denominator, the splits' in their
// order and then the sink's, into `denominators`. Then, past a barrier, a
// simdgroup a fused row, a lane each four of its dimensions: their
// numerators, the splits' in their order, whose loads a lane issues
// CombineSplits splits at a time before it adds them, and their quotients,
// as the reduce's. A simdgroup that combined two dimensions of a row at a
// time, its lanes holding the splits' weights, had waited on each split's
// load of each of its six rows of an 8-row tile in turn: a verify step's
// layer took 10% longer (8 rows) and 12% (16), a 1-row step's 6% less
// (20-core M5 Pro).
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim>
inline void relative_attention_combine_weights(device coherent(device) const float *statistics, uint written,
                                               ulong head_slot, uint fused_row, float sink, threadgroup float *weights,
                                               threadgroup float *denominators) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  float maximum = sink;
  for (uint split = 0; split < written; ++split)
    maximum = max(maximum, statistics[((head_slot + split) * M + fused_row) * 2]);
  float denominator = 0.0f;
  for (uint split = 0; split < written; ++split) {
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    const float weight = relative_attention_split_weight(statistics[stat], maximum);
    weights[fused_row * written + split] = weight;
    denominator += weight * statistics[stat + 1];
  }
  denominators[fused_row] = relative_attention_denominator(denominator, sink, maximum);
}

template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim>
inline float4 relative_attention_combine_row(device coherent(device) const float *partials, uint written,
                                             ulong head_slot, uint fused_row, uint dimension,
                                             threadgroup const float *weights, float denominator) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = HeadDim;
  constexpr uint CombineSplits = 4;
  float4 numerator = 0.0f;
  for (uint first = 0; first < written; first += CombineSplits) {
    float4 partial[CombineSplits];
#pragma unroll
    for (uint i = 0; i < CombineSplits; ++i)
      partial[i] = first + i < written ? *reinterpret_cast<device coherent(device) const float4 *>(
                                             partials + ((head_slot + first + i) * M + fused_row) * D + dimension)
                                       : float4(0.0f);
#pragma unroll
    for (uint i = 0; i < CombineSplits && first + i < written; ++i) {
      const float weight = weights[fused_row * written + first + i];
      numerator.x += weight * partial[i].x;
      numerator.y += weight * partial[i].y;
      numerator.z += weight * partial[i].z;
      numerator.w += weight * partial[i].w;
    }
  }
  return denominator > 0.0f ? float4(numerator.x / denominator, numerator.y / denominator,
                                     numerator.z / denominator, numerator.w / denominator)
                            : float4(0.0f);
}

// relative_attention_output_row of a fused row's four dimensions from
// `dimension` on, a lane of the row's simdgroup each: their values (the
// combine's quotients) rounded, times the sigmoids of their gates, at the
// row's output place `place` of `hidden` and, with Table, of the
// out-projection's Table16 input, which the simdgroup writes a span of 64 at
// a time, lane l the span's elements 2l and 2l + 1, from the lanes that hold
// them.
template <uint QHeads, uint HeadDim, bool Table>
inline void relative_attention_output_quad(float4 value, device const bfloat *projection, device bfloat *hidden,
                                           device bfloat *table, device float *sums, SplashAttentionOutput place,
                                           uint query_head, uint dimension, uint lane) {
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Width = QHeads * HeadDim;
  static_assert(HeadDim == 128, "a lane's four dimensions of a simdgroup's row");
  if (!place.writes)
    return;
  const bfloat4 gates = *reinterpret_cast<device const bfloat4 *>(projection + place.packed * Row::Stride +
                                                                   Row::GateOffset + query_head * HeadDim + dimension);
  const bfloat4 element(relative_attention_gated(float(bfloat(value.x)), float(gates.x)),
                        relative_attention_gated(float(bfloat(value.y)), float(gates.y)),
                        relative_attention_gated(float(bfloat(value.z)), float(gates.z)),
                        relative_attention_gated(float(bfloat(value.w)), float(gates.w)));
  *reinterpret_cast<device bfloat4 *>(hidden + place.output * Width + query_head * HeadDim + dimension) = element;
  if constexpr (Table)
#pragma unroll
    for (uint span = 0; span < 2; ++span) {
      const bfloat4 held = as_type<bfloat4>(simd_shuffle(as_type<uint2>(element), ushort(span * 16 + lane / 2)));
      gguf_sg::Table16::write(table + place.output / 8 * Width * 8,
                              sums + place.output / 8 * gguf_sg::Table16::sums_per_tile(Width), Width,
                              (query_head * HeadDim) / 64 + span, place.output % 8, lane,
                              lane % 2 ? held.z : held.x, lane % 2 ? held.w : held.y);
    }
}
// ---- end runtime/metal/kernels/common/relative_attention_tile.h ----
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

// Relative attention (ops::RelativeAttention, kernels/common/
// relative_attention_tile.h): its prepare, then the page loop over the tiles
// of a dispatch's sequences and the reduce, laid out and bound as the paged
// attention's (kernels/shared/paged_attention.metal), or, a dispatch whose
// tiles each read their history in one split, the direct kernels, which
// write their tiles' rows themselves (Direct below); each tile
// SPLASH_ATTENTION_TILE_ROWS query rows of one KV head (a direct kernel's
// SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS), every sequence of a command in one
// dispatch, its rows its entry of the command's layout. The geometry is
// Inkling-2-Mini's: 24 query heads over 8 KV heads of 128 dimensions, three a
// group, a tile one part of 24 fused rows. Every kernel here computes in its
// source's order (no reassociation, the pragma above): the split, the direct
// kernels and a decode step's single pass share the page loop, and the direct
// kernels' and the single pass's combines take the reduce's order, so that
// they write the prepare, split and reduce's bits, where with reassociation
// the compiler arranged the page loop of INT8 queries in the split and the
// single pass otherwise.

static_assert(SPLASH_ATTENTION_TILE_PARTS(3u) == 1, "a group of three runs whole tiles");

// Prepare: a threadgroup of HeadDim threads per (head, row, sequence),
// relative_attention_prepare_phase. Bindings: the projection rows, the
// convolved K and V rows, the learned query and relative-state scales (F32),
// the stagings of the queries, relative states and sinks, the page tables,
// the command's tree rows (another buffer of the dispatch without), its
// layout and the parameters. _scaled scales the queries and states by their
// rows' log scales (the full layers').
#define RELATIVE_ATTENTION_PREPARE(Name, QHeads, KHeads, HeadDim, PositionScaled, CacheElement)                 \
  kernel void Name(                                                                                          \
      device const bfloat *projection [[buffer(0)]],                                                         \
      device const bfloat *keys [[buffer(1)]],                                                               \
      device const bfloat *values [[buffer(2)]],                                                             \
      device const float *query_scale [[buffer(3)]],                                                         \
      device const float *relative_scale [[buffer(4)]],                                                      \
      device bfloat *queries [[buffer(5)]],                                                                  \
      device float *relative_states [[buffer(6)]],                                                           \
      device float *sinks [[buffer(7)]],                                                                     \
      device const SplashKvPage *page_table0 [[buffer(8)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(9)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(10)]],                                                 \
      device const SplashKvPage *page_table3 [[buffer(11)]],                                                 \
      device const SplashKvPage *page_table4 [[buffer(12)]],                                                 \
      device const SplashKvPage *page_table5 [[buffer(13)]],                                                 \
      device const SplashKvPage *page_table6 [[buffer(14)]],                                                 \
      device const SplashKvPage *page_table7 [[buffer(15)]],                                                 \
      device const SplashTreeRow *tree [[buffer(16)]],                                                       \
      device const SplashLaneRows *layout [[buffer(17)]],                                                    \
      constant SplashAttentionParams &params [[buffer(18)]],                                                 \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[2 * (HeadDim / 32)];                                                        \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    relative_attention_prepare_phase<QHeads, KHeads, HeadDim, PositionScaled, CacheElement>(                 \
        projection, keys, values, query_scale, relative_scale, queries, relative_states, sinks,              \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4,        \
                            page_table5, page_table6, page_table7),                                          \
        tree, layout, params, reductions, maxima, group, thread_index, lane, simd_group);                    \
  }
RELATIVE_ATTENTION_PREPARE(relative_attention_q8_prepare, 24, 8, 128, false, int8_t)
RELATIVE_ATTENTION_PREPARE(relative_attention_bf16_prepare, 24, 8, 128, false, bfloat)
RELATIVE_ATTENTION_PREPARE(relative_attention_q8_prepare_scaled, 24, 8, 128, true, int8_t)
RELATIVE_ATTENTION_PREPARE(relative_attention_bf16_prepare_scaled, 24, 8, 128, true, bfloat)
#undef RELATIVE_ATTENTION_PREPARE

// Split: KV heads vary first, then the dispatch's tiles, then history splits,
// as splash_attention_split_phase's; a tile reads its fused rows' relative
// states from the states' staging at its queries' place.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, typename CacheElement,
          bool Int8Queries>
inline void relative_attention_split_phase(
    device bfloat *queries, device float *partials, device float *statistics,
    device const SplashKvPage *page_table, device const SplashTreeRow *tree, device const SplashLaneRows *layout,
    device const float *relative_states, device const float *relative_bias,
    constant SplashAttentionParams &params, threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, uint3 group, uint thread_index, uint s) {
  constexpr uint M = QueryHeadsPerKVHead * SPLASH_ATTENTION_TILE_ROWS;
  const uint split = group.z;
  const uint kv_head = group.x;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile = splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(
      params, sequence, rows, kv_head, group.y - sequence.first_tile);
  if (!tile.active || split >= tile.splits)
    return;
  const RelativeAttentionPages run = relative_attention_pages<Window>(
      tile.first_position, tile.active_rows, params.trees != 0, sequence.committed_tokens, tile.splits, split);
  if (run.begin >= run.end)
    return;
  // The INT8 query codes live in the threadgroup's own partials slot, which
  // it writes only after its last page.
  const ulong slot = tile.first_slot + split;
  device float *own = partials + slot * M * HeadDim;
  relative_attention_page_loop<KVHeads, QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, HeadDim, Window,
                               CacheElement, Int8Queries>(
      queries + tile.offset, reinterpret_cast<device int8_t *>(own),
      relative_states + tile.offset / HeadDim * SPLASH_RELATIVE_ATTENTION_STATES, relative_bias, page_table,
      params.kv, kv_head, tile.first_position, tile.active_rows, run, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, splash_attention_tree(params, sequence, rows, tree, tile.first_position),
      thread_index,
      [&](thread auto &running, bool) {
        auto target = tensor(own, dextents<int, 2>{HeadDim, M}, array<int, 2>{1, HeadDim});
        running.store(target.slice<HeadDim, M>(0, 0));
        if (thread_index < M) {
          statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
          statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
        }
      });
}

// The paged attention's split bindings, then the relative states' staging
// and the layer's bias table (F32, [distance][state]) before the parameters.
#define RELATIVE_ATTENTION_SPLIT(Name, Heads, Group, HeadDim, Window, CacheElement, Int8Queries)                \
  kernel void Name(                                                                                           \
      device bfloat *queries [[buffer(0)]],                                                                   \
      device float *partials [[buffer(1)]],                                                                   \
      device float *statistics [[buffer(2)]],                                                                 \
      device const SplashKvPage *page_table0 [[buffer(3)]],                                                   \
      device const SplashKvPage *page_table1 [[buffer(4)]],                                                   \
      device const SplashKvPage *page_table2 [[buffer(5)]],                                                   \
      device const SplashKvPage *page_table3 [[buffer(6)]],                                                   \
      device const SplashKvPage *page_table4 [[buffer(7)]],                                                   \
      device const SplashKvPage *page_table5 [[buffer(8)]],                                                   \
      device const SplashKvPage *page_table6 [[buffer(9)]],                                                   \
      device const SplashKvPage *page_table7 [[buffer(10)]],                                                  \
      device const SplashTreeRow *tree [[buffer(11)]],                                                        \
      device const SplashLaneRows *layout [[buffer(12)]],                                                     \
      device const float *relative_states [[buffer(13)]],                                                     \
      device const float *relative_bias [[buffer(14)]],                                                       \
      constant SplashAttentionParams &params [[buffer(15)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    constexpr uint M = Group * SPLASH_ATTENTION_TILE_ROWS;                                                    \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    const uint s = splash_attention_sequence_of_tile(params, group.y);                                        \
    relative_attention_split_phase<Heads, Group, HeadDim, Window, CacheElement, Int8Queries>(                 \
        queries, partials, statistics,                                                                        \
        SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5,  \
                            page_table6, page_table7),                                                        \
        tree, layout, relative_states, relative_bias, params, scores, probabilities, row_max, row_sum,        \
        previous_scale, &rescale, group, thread_index, s);                                                    \
  }
// Over the whole history (the full layers) and over a window of the 1023
// keys before each row (the sliding layers' 1024 keys, the row's own
// included); INT8 pages with bf16 or INT8 queries, and BF16 pages.
#define RELATIVE_ATTENTION_SPLITS(Suffix, Window)                                                             \
  RELATIVE_ATTENTION_SPLIT(relative_attention_q8_split##Suffix, 8, 3, 128, Window, int8_t, false)             \
  RELATIVE_ATTENTION_SPLIT(relative_attention_q8q_split##Suffix, 8, 3, 128, Window, int8_t, true)             \
  RELATIVE_ATTENTION_SPLIT(relative_attention_bf16_split##Suffix, 8, 3, 128, Window, bfloat, false)
RELATIVE_ATTENTION_SPLITS(, 0)
RELATIVE_ATTENTION_SPLITS(_w1023, 1023)
#undef RELATIVE_ATTENTION_SPLITS
#undef RELATIVE_ATTENTION_SPLIT

// Reduce: a threadgroup of HeadDim threads per fused row of a tile, group.z
// the dispatch's tile, as splash_attention_reduce_phase's: the row's splits
// with its own sink, staged as its queries, then its gated output row
// (relative_attention_output_row) at its output row. The splits that wrote
// are those splash_attention_reduce_row recomputes: the partition of the same
// Window from the same first page.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, bool Table>
inline void relative_attention_reduce_phase(
    device const float *partials, device const float *statistics, device const bfloat *projection,
    device bfloat *hidden, device bfloat *table, device float *sums, device const SplashLaneRows *layout,
    device const float *sinks, constant SplashAttentionParams &params, uint3 group, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values, threadgroup bfloat *pairs) {
  constexpr ushort M = SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  const uint fused_row = group.y;
  const uint s = splash_attention_sequence_of_tile(params, group.z);
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile = splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(
      params, sequence, rows, group.x, group.z - sequence.first_tile);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  uint pages = splash_attention_pages(tile.first_position + tile.active_rows);
  if constexpr (Window != 0)
    pages -= splash_attention_first_page<Window>(
        splash_attention_window_position(params.trees != 0, sequence.committed_tokens, tile.first_position));
  const uint per_split = splash_attention_pages_per_split(pages, tile.splits);
  const uint written = (pages + per_split - 1) / per_split;
  const bfloat value = relative_attention_combine_splits<QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, HeadDim>(
      partials, statistics, tile.active_rows, written, tile.first_slot, fused_row, thread_index, weights,
      group_values, sinks[tile.offset / HeadDim + fused_row]);
  const SplashAttentionOutput place = splash_attention_output(
      params, s, rows, tile.first_position - sequence.committed_tokens + fused_row / QueryHeadsPerKVHead);
  relative_attention_output_row<KVHeads * QueryHeadsPerKVHead, HeadDim, Table>(
      value, projection, hidden, table, sums, place.packed, place.output,
      group.x * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead, thread_index, place.writes, pairs);
}

// Bindings: the partials and statistics, the projection rows (their gates),
// the out-projection's input rows, its table and the table's sums (other
// buffers of the dispatch without a table), the command's layout, the rows'
// sinks' staging, then the parameters. Each is compiled plain and,
// _table16, writing the out-projection's Table16 input too.
#define RELATIVE_ATTENTION_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Table)                            \
  kernel void Name(                                                                                           \
      device const float *partials [[buffer(0)]],                                                             \
      device const float *statistics [[buffer(1)]],                                                           \
      device const bfloat *projection [[buffer(2)]],                                                          \
      device bfloat *hidden [[buffer(3)]],                                                                    \
      device bfloat *table [[buffer(4)]],                                                                     \
      device float *sums [[buffer(5)]],                                                                       \
      device const SplashLaneRows *layout [[buffer(6)]],                                                      \
      device const float *sinks [[buffer(7)]],                                                                \
      constant SplashAttentionParams &params [[buffer(8)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    threadgroup float weights[SplashAttentionMaximumSplits];                                                  \
    threadgroup float group_values[HeadDim / 32];                                                             \
    threadgroup bfloat pairs[HeadDim];                                                                        \
    relative_attention_reduce_phase<Heads, Group, HeadDim, Window, Table>(                                    \
        partials, statistics, projection, hidden, table, sums, layout, sinks, params, group, thread_index,    \
        weights, group_values, pairs);                                                                        \
  }
#define RELATIVE_ATTENTION_REDUCE(Name, Window)                                                               \
  RELATIVE_ATTENTION_REDUCE_KERNEL(Name, 8, 3, 128, Window, false)                                            \
  RELATIVE_ATTENTION_REDUCE_KERNEL(Name##_table16, 8, 3, 128, Window, true)
RELATIVE_ATTENTION_REDUCE(relative_attention_reduce, 0)
RELATIVE_ATTENTION_REDUCE(relative_attention_reduce_w1023, 1023)
#undef RELATIVE_ATTENTION_REDUCE
#undef RELATIVE_ATTENTION_REDUCE_KERNEL

// Direct (ops::RelativeAttention::addAttention, a dispatch whose sequences
// each read their history in one split): a threadgroup of 256 threads per (KV
// head, tile), each tile SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS rows of one KV
// head, a sequence's tiles the dispatch's from its first_tile on in tiles of
// that many rows. A tile runs the split's page loop over its one run of pages,
// those of its first row's window to its last row's: the union of the runs of
// its two paged-attention tiles of the same rows, whose other pages none of a
// row's keys sit on, so that their probabilities are zero and leave the row's
// statistics and output as they were. It then writes its rows' output as the
// reduce does from that one split: a thread a fused row its weight and
// denominator, each thread the gated outputs of the running output's elements
// it holds. It writes no out-projection table: its chunks, of more than 128
// rows, run the out-projection's prefill tile, which reads plain rows (the
// table, its sums and the simdgroup indices it is given, as the reduce is,
// go unused). The INT8 query codes take the slot of partials at the tile's
// first_slot plus its tile and KV head. Measured on a 2,048-row chunk's layer
// (20-core M5 Pro): the reduce and the partials it reads took a sliding
// layer's attention 10% of its time after 2,048 tokens (4.55 ms against 4.10
// without) and a full layer's 1.7% after 8,192; a tile of twice the rows,
// which reads each page once for both, ran 5-11% faster than two tiles at
// short histories (0 to 2,048 tokens) and 0.6-4% at 8,192 and 30,720.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, typename CacheElement, bool Int8Queries>
inline void relative_attention_direct_phase(
    device bfloat *queries, device float *partials, device const SplashKvPage *page_table,
    device const SplashTreeRow *tree, device const SplashLaneRows *layout, device const float *relative_states,
    device const float *relative_bias, device const bfloat *projection, device bfloat *hidden, device bfloat *,
    device float *, device const float *sinks, constant SplashAttentionParams &params, threadgroup float *scores,
    threadgroup bfloat *probabilities, threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group, uint thread_index, uint, uint,
    uint s) {
  constexpr uint Rows = SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS;
  constexpr uint M = QueryHeadsPerKVHead * Rows, QHeads = KVHeads * QueryHeadsPerKVHead;
  constexpr uint Width = QHeads * HeadDim;
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  const uint kv_head = group.x;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const uint tile = group.y - sequence.first_tile, start = tile * Rows;
  if (!splash_attention_contract_valid(params, sequence) || sequence.split_count != 1 || kv_head >= KVHeads ||
      start >= rows.rows)
    return;
  const uint first_position = sequence.committed_tokens + start, active_rows = min(Rows, rows.rows - start);
  const ulong offset =
      (ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + start) * QueryHeadsPerKVHead * HeadDim;
  const RelativeAttentionPages run =
      relative_attention_pages<Window>(first_position, active_rows, params.trees != 0, sequence.committed_tokens, 1, 0);
  device float *codes = partials + (ulong(sequence.first_slot) + ulong(tile) * KVHeads + kv_head) *
                                       (SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead) * HeadDim;
  static_assert(M * HeadDim <= SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead * HeadDim * sizeof(float),
                "a tile's query codes fit a slot of the paged attention's partials");
  relative_attention_page_loop<KVHeads, QueryHeadsPerKVHead, Rows, HeadDim, Window, CacheElement, Int8Queries>(
      queries + offset, reinterpret_cast<device int8_t *>(codes),
      relative_states + offset / HeadDim * SPLASH_RELATIVE_ATTENTION_STATES, relative_bias, page_table, params.kv,
      kv_head, first_position, active_rows, run, scores, probabilities, row_max, row_sum, previous_scale, rescale,
      splash_attention_tree(params, sequence, rows, tree, first_position), thread_index,
      [&](thread auto &running, bool running_full) {
        // The combine of the one split (relative_attention_tile.h's
        // Reduce): each active fused row's weight into previous_scale and
        // its denominator into row_sum, from its statistics and its sink,
        // the split's product rounded before the sink's term is added
        // (contract(off)), as the reduce's loop over splits rounds a row's
        // one product: contracted into one fused multiply-add, the two gave
        // other bits.
        if (thread_index < M && thread_index / QueryHeadsPerKVHead < active_rows) {
#pragma clang fp contract(off)
          const float sink = sinks[offset / HeadDim + thread_index];
          const float maximum = row_max[thread_index];
          const float row_maximum = max(maximum, sink);
          const float weight = relative_attention_split_weight(maximum, row_maximum);
          const float denominator =
              relative_attention_denominator(weight * row_sum[thread_index], sink, row_maximum);
          previous_scale[thread_index] = weight;
          row_sum[thread_index] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (ushort index = 0; index < running.get_capacity(); ++index) {
          if (!running_full && !running.is_valid_element(index))
            continue;
          const auto coordinates = running.get_multidimensional_index(index);
          const uint fused_row = uint(coordinates[1]), dimension = uint(coordinates[0]);
          const uint row = fused_row / QueryHeadsPerKVHead;
          if (row >= active_rows)
            continue;
          const SplashAttentionOutput place = splash_attention_output(params, s, rows, start + row);
          if (!place.writes)
            continue;
          float numerator = 0.0f;
          numerator += previous_scale[fused_row] * running[index];
          const float denominator = row_sum[fused_row];
          const bfloat value = bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
          const uint query_head = kv_head * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead;
          const float gate =
              float(projection[place.packed * Row::Stride + Row::GateOffset + query_head * HeadDim + dimension]);
          hidden[place.output * Width + query_head * HeadDim + dimension] =
              relative_attention_gated(float(value), gate);
        }
      });
}

// The split's bindings without its statistics, then the plain reduce's: the
// projection rows (their gates), the out-projection's input rows, a table
// and its sums, which it does not write (other buffers of the dispatch), and
// the rows' sinks' staging, before the parameters. Over the whole history and
// a sliding layer's window, for INT8 pages with bf16 or INT8 queries and BF16
// pages.
#define RELATIVE_ATTENTION_DIRECT_KERNEL(Name, Window, CacheElement, Int8Queries)                                \
  kernel void Name(                                                                                           \
      device bfloat *queries [[buffer(0)]],                                                                   \
      device float *partials [[buffer(1)]],                                                                   \
      device const SplashKvPage *page_table0 [[buffer(2)]],                                                   \
      device const SplashKvPage *page_table1 [[buffer(3)]],                                                   \
      device const SplashKvPage *page_table2 [[buffer(4)]],                                                   \
      device const SplashKvPage *page_table3 [[buffer(5)]],                                                   \
      device const SplashKvPage *page_table4 [[buffer(6)]],                                                   \
      device const SplashKvPage *page_table5 [[buffer(7)]],                                                   \
      device const SplashKvPage *page_table6 [[buffer(8)]],                                                   \
      device const SplashKvPage *page_table7 [[buffer(9)]],                                                   \
      device const SplashTreeRow *tree [[buffer(10)]],                                                        \
      device const SplashLaneRows *layout [[buffer(11)]],                                                     \
      device const float *relative_states [[buffer(12)]],                                                     \
      device const float *relative_bias [[buffer(13)]],                                                       \
      device const bfloat *projection [[buffer(14)]],                                                         \
      device bfloat *hidden [[buffer(15)]],                                                                   \
      device bfloat *table [[buffer(16)]],                                                                    \
      device float *sums [[buffer(17)]],                                                                      \
      device const float *sinks [[buffer(18)]],                                                               \
      constant SplashAttentionParams &params [[buffer(19)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]],                                                      \
      uint lane [[thread_index_in_simdgroup]],                                                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                   \
    constexpr uint M = 3 * SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS;                                             \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    const uint s = splash_attention_sequence_of_tile(params, group.y);                                        \
    relative_attention_direct_phase<8, 3, 128, Window, CacheElement, Int8Queries>(                            \
        queries, partials,                                                                                    \
        SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5,  \
                            page_table6, page_table7),                                                        \
        tree, layout, relative_states, relative_bias, projection, hidden, table, sums, sinks, params, scores,  \
        probabilities, row_max, row_sum, previous_scale, &rescale, group, thread_index, lane, simd_group, s);   \
  }
#define RELATIVE_ATTENTION_DIRECTS(Suffix, Window)                                                            \
  RELATIVE_ATTENTION_DIRECT_KERNEL(relative_attention_q8_direct##Suffix, Window, int8_t, false)               \
  RELATIVE_ATTENTION_DIRECT_KERNEL(relative_attention_q8q_direct##Suffix, Window, int8_t, true)               \
  RELATIVE_ATTENTION_DIRECT_KERNEL(relative_attention_bf16_direct##Suffix, Window, bfloat, false)
RELATIVE_ATTENTION_DIRECTS(, 0)
RELATIVE_ATTENTION_DIRECTS(_w1023, 1023)
#undef RELATIVE_ATTENTION_DIRECTS
#undef RELATIVE_ATTENTION_DIRECT_KERNEL

// Single pass (ops::RelativeAttention::addDecode): a decode step's prepare,
// split and reduce in one dispatch, a threadgroup of 256 threads per (KV
// head, dispatch sequence, history split), each sequence a lane's block of at
// most RowsPerTile rows run as one tile (relative_attention_tile.h, Single
// pass). A split's threadgroup stages its tile's queries and relative states
// in a staging slot of its own, a simdgroup a (row, query head), and stores
// the K/V of the block's rows that sit on its pages, a simdgroup a row, before
// its page loop reads them; no other threadgroup reads those pages' slots of
// them. It then runs the split's page loop and publishes its partials and
// statistics of the tile's rows, and the split that arrives last at its
// tile's counter (kernels/common/split_reduce.h; zero before and after)
// combines them with each row's sink into the gated output rows, its weights
// a thread a fused row and then its numerators a simdgroup a fused row
// (relative_attention_combine_weights, _row). A sequence's slots, from its
// first_slot in tiles of SPLASH_ATTENTION_TILE_ROWS rows (RowsPerTile / tile
// rows of them a slot), hold each KV head's splits' partials, then each KV
// head's splits' staging: the INT8 query codes, the relative states and the
// bf16 queries, in that order.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim, uint Window, bool PositionScaled,
          typename CacheElement, bool Int8Queries, bool Table>
inline void relative_attention_decode_phase(
    device const bfloat *projection, device const bfloat *keys, device const bfloat *values,
    device const float *query_scale, device const float *relative_scale, device const float *relative_bias,
    device float *partials, device float *statistics, device atomic_uint *counters, device bfloat *hidden,
    device bfloat *table, device float *sums, device const SplashKvPage *page_table, device const SplashTreeRow *tree,
    device const SplashLaneRows *layout, constant SplashAttentionParams &params, threadgroup float *scores,
    threadgroup bfloat *probabilities, threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, threadgroup uint *arrival, uint3 group,
    uint thread_index, uint lane, uint simd_group) {
  using Row = RelativeAttentionRow<KVHeads * QueryHeadsPerKVHead, HeadDim>;
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead, D = HeadDim, States = Row::States;
  constexpr uint Simdgroups = 8;
  constexpr uint Tiles = RowsPerTile / SPLASH_ATTENTION_TILE_ROWS;
  static_assert(Tiles * SPLASH_ATTENTION_TILE_ROWS == RowsPerTile && M <= 64, "whole tiles of a 256-thread loop");
  const uint kv_head = group.x, s = group.y, split = group.z;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  if (!splash_attention_contract_valid(params, sequence) || sequence.row_bound > RowsPerTile ||
      sequence.split_count > SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS || kv_head >= KVHeads || !rows.rows ||
      split >= sequence.split_count)
    return;
  const uint first = sequence.committed_tokens, fused_rows = rows.rows * QueryHeadsPerKVHead;
  const bool in_tree = params.trees != 0;
  const RelativeAttentionPages run =
      relative_attention_pages<Window>(first, rows.rows, in_tree, first, sequence.split_count, split);
  if (run.begin >= run.end)
    return;
  const ulong head_slot = ulong(sequence.first_slot) / Tiles + ulong(kv_head) * sequence.split_count;
  device float *staging =
      partials + (ulong(sequence.first_slot) / Tiles + ulong(KVHeads + kv_head) * sequence.split_count + split) * M * D;
  device float *states = staging + M * D / 4;
  device bfloat *queries = reinterpret_cast<device bfloat *>(staging) + M * D;
  for (uint fused = simd_group; fused < fused_rows; fused += Simdgroups) {
    const uint row = fused / QueryHeadsPerKVHead;
    const ulong packed = ulong(rows.first) + row;
    relative_attention_stage_head<KVHeads * QueryHeadsPerKVHead, D, PositionScaled>(
        projection + packed * Row::Stride, query_scale, relative_scale,
        kv_head * QueryHeadsPerKVHead + fused % QueryHeadsPerKVHead,
        first + (in_tree ? uint(tree[packed].depth) : row), queries + fused * D, states + fused * States, lane);
  }
  for (uint row = simd_group; row < rows.rows; row += Simdgroups) {
    const uint page = (first + row) / SplashKvPageTokens;
    if (page >= run.begin && page < run.end)
      relative_attention_store_head<KVHeads, D, CacheElement>(
          keys, values, (ulong(rows.first) + row) * (KVHeads * D) + kv_head * D, page_table[page], params.kv,
          kv_head, (first + row) % SplashKvPageTokens, lane);
  }
  threadgroup_barrier(mem_flags::mem_device);
  // Publish the tile's rows' partials and statistics, then arrive.
  const ulong slot = head_slot + split;
  relative_attention_page_loop<KVHeads, QueryHeadsPerKVHead, RowsPerTile, D, Window, CacheElement, Int8Queries>(
      queries, reinterpret_cast<device int8_t *>(staging), states, relative_bias, page_table, params.kv, kv_head,
      first, rows.rows, run, scores, probabilities, row_max, row_sum, previous_scale, rescale,
      splash_attention_tree(params, sequence, rows, tree, first), thread_index,
      [&](thread auto &running, bool running_full) {
        device coherent(device) float *published = partials + slot * M * D;
#pragma unroll
        for (ushort index = 0; index < running.get_capacity(); ++index) {
          if (!running_full && !running.is_valid_element(index))
            continue;
          const auto coordinates = running.get_multidimensional_index(index);
          if (uint(coordinates[1]) < fused_rows)
            published[uint(coordinates[1]) * D + uint(coordinates[0])] = running[index];
        }
        if (thread_index < fused_rows) {
          device coherent(device) float *row_statistics = statistics + (slot * M + thread_index) * 2;
          row_statistics[0] = row_max[thread_index];
          row_statistics[1] = row_sum[thread_index];
        }
      });
  device atomic_uint *counter = counters + (params.first_sequence + s) * KVHeads + kv_head;
  if (!split_arrive_last(counter, run.written, thread_index, arrival))
    return;
  // The tile's statistics are published: the combine's weights reuse the
  // scores' storage (fused rows x splits, at most M x SplashKvPageTokens)
  // and its denominators the row sums'.
  static_assert(SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS <= SplashKvPageTokens, "weights in the scores' storage");
  if (thread_index < fused_rows) {
    const uint query_head = kv_head * QueryHeadsPerKVHead + thread_index % QueryHeadsPerKVHead;
    const SplashAttentionOutput place = splash_attention_output(params, s, rows, thread_index / QueryHeadsPerKVHead);
    relative_attention_combine_weights<QueryHeadsPerKVHead, RowsPerTile, D>(
        statistics, run.written, head_slot, thread_index,
        float(projection[place.packed * Row::Stride + Row::SinkOffset + query_head]), scores, row_sum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint fused = simd_group; fused < fused_rows; fused += Simdgroups) {
    const uint query_head = kv_head * QueryHeadsPerKVHead + fused % QueryHeadsPerKVHead;
    relative_attention_output_quad<KVHeads * QueryHeadsPerKVHead, D, Table>(
        relative_attention_combine_row<QueryHeadsPerKVHead, RowsPerTile, D>(partials, run.written, head_slot, fused,
                                                                           4 * lane, scores, row_sum[fused]),
        projection, hidden, table, sums, splash_attention_output(params, s, rows, fused / QueryHeadsPerKVHead),
        query_head, 4 * lane, lane);
  }
  split_release(counter, thread_index);
}

// Bindings: the projection rows, the convolved K and V rows, the learned
// query and relative-state scales, the layer's bias table, the partials and
// statistics, the tiles' counters, the out-projection's input rows, its table
// and the table's sums (other buffers of the dispatch without a table), the
// page tables, the command's tree rows (another buffer of the dispatch
// without), its layout and the parameters. Each is compiled for blocks of up
// to 8 rows and, _r16, a verify block's 16; for the layers' two variants
// alone, a full layer's (the whole history, position-scaled: _scaled) and a
// sliding one's (windowed, unscaled: _w1023); plain and writing the
// out-projection's Table16 input too (_table16); for INT8 pages with bf16 or
// INT8 queries, and BF16 pages.
#define RELATIVE_ATTENTION_DECODE_KERNEL(Name, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries, Table) \
  kernel void Name(                                                                                           \
      device const bfloat *projection [[buffer(0)]],                                                          \
      device const bfloat *keys [[buffer(1)]],                                                                \
      device const bfloat *values [[buffer(2)]],                                                              \
      device const float *query_scale [[buffer(3)]],                                                          \
      device const float *relative_scale [[buffer(4)]],                                                       \
      device const float *relative_bias [[buffer(5)]],                                                        \
      device float *partials [[buffer(6)]],                                                                   \
      device float *statistics [[buffer(7)]],                                                                 \
      device atomic_uint *counters [[buffer(8)]],                                                             \
      device bfloat *hidden [[buffer(9)]],                                                                    \
      device bfloat *table [[buffer(10)]],                                                                    \
      device float *sums [[buffer(11)]],                                                                      \
      device const SplashKvPage *page_table0 [[buffer(12)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(13)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(14)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(15)]],                                                  \
      device const SplashKvPage *page_table4 [[buffer(16)]],                                                  \
      device const SplashKvPage *page_table5 [[buffer(17)]],                                                  \
      device const SplashKvPage *page_table6 [[buffer(18)]],                                                  \
      device const SplashKvPage *page_table7 [[buffer(19)]],                                                  \
      device const SplashTreeRow *tree [[buffer(20)]],                                                        \
      device const SplashLaneRows *layout [[buffer(21)]],                                                     \
      constant SplashAttentionParams &params [[buffer(22)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]],                                                      \
      uint lane [[thread_index_in_simdgroup]],                                                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                   \
    constexpr uint M = 3 * RowsPerTile;                                                                       \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    threadgroup uint arrival;                                                                                 \
    relative_attention_decode_phase<8, 3, RowsPerTile, 128, Window, PositionScaled, CacheElement, Int8Queries,  \
                                    Table>(                                                                   \
        projection, keys, values, query_scale, relative_scale, relative_bias, partials, statistics, counters,  \
        hidden, table, sums,                                                                                  \
        SPLASH_LANE_BINDING(group.y, page_table0, page_table1, page_table2, page_table3, page_table4,          \
                            page_table5, page_table6, page_table7),                                           \
        tree, layout, params, scores, probabilities, row_max, row_sum, previous_scale, &rescale, &arrival,     \
        group, thread_index, lane, simd_group);                                                               \
  }
#define RELATIVE_ATTENTION_DECODE_OUTPUTS(Name, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries)  \
  RELATIVE_ATTENTION_DECODE_KERNEL(Name, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries, false) \
  RELATIVE_ATTENTION_DECODE_KERNEL(Name##_table16, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries, \
                                   true)
#define RELATIVE_ATTENTION_DECODE_FORMATS(Suffix, RowsPerTile, Window, PositionScaled)                         \
  RELATIVE_ATTENTION_DECODE_OUTPUTS(relative_attention_q8_decode##Suffix, RowsPerTile, Window, PositionScaled,  \
                                    int8_t, false)                                                            \
  RELATIVE_ATTENTION_DECODE_OUTPUTS(relative_attention_q8q_decode##Suffix, RowsPerTile, Window, PositionScaled, \
                                    int8_t, true)                                                             \
  RELATIVE_ATTENTION_DECODE_OUTPUTS(relative_attention_bf16_decode##Suffix, RowsPerTile, Window, PositionScaled, \
                                    bfloat, false)
#define RELATIVE_ATTENTION_DECODE_VARIANTS(Suffix, RowsPerTile)                                                \
  RELATIVE_ATTENTION_DECODE_FORMATS(Suffix##_scaled, RowsPerTile, 0, true)                                     \
  RELATIVE_ATTENTION_DECODE_FORMATS(Suffix##_w1023, RowsPerTile, 1023, false)
RELATIVE_ATTENTION_DECODE_VARIANTS(, 8)
RELATIVE_ATTENTION_DECODE_VARIANTS(_r16, 16)
#undef RELATIVE_ATTENTION_DECODE_VARIANTS
#undef RELATIVE_ATTENTION_DECODE_FORMATS
#undef RELATIVE_ATTENTION_DECODE_OUTPUTS
#undef RELATIVE_ATTENTION_DECODE_KERNEL

// The path commit (kernels/common/kv_path_commit.h) of the pages relative
// attention stores: 8 KV heads of 128 dimensions, values dimension-major
// (ops::RelativeAttention::addPathCommit).
PAGED_ATTENTION_PATH_COMMIT(relative_attention_q8_commit, 8, 128, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(relative_attention_bf16_commit, 8, 128, bfloat, false)
#undef PAGED_ATTENTION_PATH_COMMIT
// ---- end runtime/metal/kernels/shared/relative_attention.metal ----
