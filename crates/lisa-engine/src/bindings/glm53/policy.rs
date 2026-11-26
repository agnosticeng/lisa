//! Explicitly scoped diagnostic scheduling controls.
/// The reference's grouped routed-MoE prefill shape gate starts at 96 rows. Keep the
/// production route aligned with that execution path; low-row throughput is
/// an open optimization target and must not silently select a different MoE
/// arithmetic schedule.
pub use lisa_models::glm53::routed_moe::GROUPED_MOE_MIN_ROWS as GLM53_GROUPED_MOE_MIN_ROWS;
use std::cell::Cell;

thread_local! {
    static GLM53_Q8_MPP_PREFILL_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_PREFILL_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
    static GLM53_INDEXER_DECODE_ROWS4_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_KDA_NSG_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Scoped override used by the in-process prefill A/B harness. `true` selects
/// the native reference Q8 schedule; `false` selects stock QMM. Production uses the reference's
/// tiny-row and matrix schedules with the original MLX affine storage.
pub struct Glm53Q8PrefillScope {
    previous: Option<bool>,
}

impl Glm53Q8PrefillScope {
    pub fn new(use_ds4_mpp: bool) -> Self {
        let previous = GLM53_Q8_MPP_PREFILL_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_ds4_mpp));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53Q8PrefillScope {
    fn drop(&mut self) {
        GLM53_Q8_MPP_PREFILL_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_q8_mpp_prefill_override() -> Option<bool> {
    GLM53_Q8_MPP_PREFILL_OVERRIDE.with(Cell::get)
}

/// Keep model-specific projection shortcuts on the same explicitly selected
/// Q8 arithmetic schedule as ordinary GLM linear projections.
pub fn glm53_q8_native_prefill_enabled() -> bool {
    glm53_q8_mpp_prefill_override() != Some(false)
}

/// Scoped override for the reference-shaped grouped routed-MoE prefill schedule.
/// The in-process prefill A/B harness uses this to compare the grouped path
/// with the production per-slot schedule on identical model state and rows.
pub struct Glm53GroupedMoePrefillScope {
    previous: Option<bool>,
}

impl Glm53GroupedMoePrefillScope {
    pub fn new(use_grouped: bool) -> Self {
        let previous = GLM53_GROUPED_MOE_PREFILL_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_grouped));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoePrefillScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_PREFILL_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_grouped_moe_prefill_override() -> Option<bool> {
    GLM53_GROUPED_MOE_PREFILL_OVERRIDE.with(Cell::get)
}

/// Scoped override used to interleave the fused and reference-split grouped gate/up
/// schedules on the same loaded model and prompt.
pub struct Glm53GroupedMoePairScope {
    previous: Option<bool>,
}

impl Glm53GroupedMoePairScope {
    pub fn new(use_split: bool) -> Self {
        let previous = GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_split));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoePairScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_grouped_moe_pair_split_override() -> Option<bool> {
    GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE.with(Cell::get)
}

/// Scoped load-mode switch used to interleave scalar byte loads with the
/// four-byte affine-Q2 unpack path in the grouped MoE prefill benchmark.
pub struct Glm53GroupedMoeQ2LoadScope {
    previous: Option<bool>,
}

impl Glm53GroupedMoeQ2LoadScope {
    pub fn new(use_vector_load: bool) -> Self {
        let previous = GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_vector_load));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoeQ2LoadScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_grouped_moe_q2_vector_load_override() -> Option<bool> {
    GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE.with(Cell::get)
}

/// Scoped output-row tile override for the grouped affine-Q2 down projection.
/// The 128-row candidate computes two independent reference 64-row tiles per
/// threadgroup while preserving each output's K accumulation order.
pub struct Glm53GroupedMoeDownTileScope {
    previous: Option<usize>,
}

impl Glm53GroupedMoeDownTileScope {
    pub fn new(output_tile: usize) -> Self {
        assert!(
            matches!(output_tile, 64 | 128),
            "unsupported grouped down tile {output_tile}"
        );
        let previous = GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(output_tile));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoeDownTileScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_grouped_moe_down_tile_override() -> Option<usize> {
    GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE.with(Cell::get)
}

/// Scoped experimental schedule for end-to-end serial-audit A/B runs. `None`
/// selects the device schedule matching the reference; a value selects a different
/// simdgroup count for the fused BF16 KDA input projection only.
pub struct Glm53KdaNsgScope {
    previous: Option<usize>,
}

impl Glm53KdaNsgScope {
    pub fn new(nsg: usize) -> Self {
        assert!(
            matches!(nsg, 4 | 8 | 16),
            "unsupported KDA simdgroup count {nsg}"
        );
        let previous = GLM53_KDA_NSG_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(nsg));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53KdaNsgScope {
    fn drop(&mut self) {
        GLM53_KDA_NSG_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_kda_nsg_override() -> Option<usize> {
    GLM53_KDA_NSG_OVERRIDE.with(Cell::get)
}

/// Scoped switch for interleaved comparisons of the reference's serial and four-row
/// one-token indexer-score schedules.
pub struct Glm53IndexerDecodeRows4Scope {
    previous: Option<bool>,
}

impl Glm53IndexerDecodeRows4Scope {
    pub fn new(use_rows4: bool) -> Self {
        let previous = GLM53_INDEXER_DECODE_ROWS4_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_rows4));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53IndexerDecodeRows4Scope {
    fn drop(&mut self) {
        GLM53_INDEXER_DECODE_ROWS4_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_indexer_decode_rows4_override() -> Option<bool> {
    GLM53_INDEXER_DECODE_ROWS4_OVERRIDE.with(Cell::get)
}

/// Scoped switch for interleaved comparisons of the reference's threadgroup and SIMD
/// shuffle top-k sort networks.
pub struct Glm53IndexerTopkShuffleScope {
    previous: Option<bool>,
}

impl Glm53IndexerTopkShuffleScope {
    pub fn new(use_shuffle: bool) -> Self {
        let previous = GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_shuffle));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53IndexerTopkShuffleScope {
    fn drop(&mut self) {
        GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub fn glm53_indexer_topk_shuffle_override() -> Option<bool> {
    GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE.with(Cell::get)
}

pub static INC_STEP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    static GLM53_Q8_GATE_UP_NATIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Scoped override for the native reference fused Q8 gate/up/SwiGLU schedule.
/// Production enables fusion for paired weights; diagnostics may disable it.
pub struct Glm53Q8GateUpNativeScope {
    previous: bool,
    _thread_local: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Glm53Q8GateUpNativeScope {
    pub fn new(enabled: bool) -> Self {
        Self {
            previous: GLM53_Q8_GATE_UP_NATIVE.with(|slot| slot.replace(enabled)),
            _thread_local: std::marker::PhantomData,
        }
    }
}

impl Drop for Glm53Q8GateUpNativeScope {
    fn drop(&mut self) {
        GLM53_Q8_GATE_UP_NATIVE.with(|slot| slot.set(self.previous));
    }
}

pub fn native_gate_up_enabled() -> bool {
    GLM53_Q8_GATE_UP_NATIVE.with(std::cell::Cell::get)
}
