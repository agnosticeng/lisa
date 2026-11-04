//! MoE and dense MLP for `glm5_next`, following ds4's F32 inference graph.
//!
//! Router in F32 (sigmoid, no
//! aux loss): selection scores add `e_score_correction_bias`, group by
//! `n_group=8` (top-2 within group summed for the group score), pick
//! `topk_group=4` groups, then top-`num_experts_per_tok=8` experts by the
//! SELECTION score; the WEIGHTS are the raw sigmoid scores renormalized
//! (`norm_topk_prob`) and scaled by `routed_scaling_factor=2.5`.
//! Activation: clamped SwiGLU (`swiglu_limit=10`): gate clipped ABOVE only,
//! up clipped both sides — for routed, shared AND dense experts.

use super::mhc::narrow;
use crate::core::capture::Capture;
use crate::core::loader::{LazyTensorStore, TensorSource, Weights};
use crate::core::quant::QuantizedLinear;
use lisa_mlx::{Array, Dtype, ops};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

thread_local! {
    static GLM53_Q8_GATE_UP_NATIVE: Cell<bool> = const { Cell::new(true) };
}

/// Scoped override for the native DS4 fused Q8 gate/up/SwiGLU schedule.
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
    GLM53_Q8_GATE_UP_NATIVE.with(Cell::get)
}

/// Concatenate gate and up output rows so one affine-quant projection computes
/// both results. Real layer-0 weights match the separate DS4 production path
/// for decode and prefill shapes; the layer-45 shared expert has the same
/// checkpoint-backed exactness check.
pub fn fuse_gate_up(
    gate: &QuantizedLinear,
    up: &QuantizedLinear,
) -> anyhow::Result<QuantizedLinear> {
    debug_assert_eq!(gate.group_size, up.group_size);
    debug_assert_eq!(gate.bits, up.bits);
    let w = ops::concatenate(&[&gate.weight, &up.weight], 0)?;
    let s = ops::concatenate(&[&gate.scales, &up.scales], 0)?;
    let b = ops::concatenate(&[&gate.biases, &up.biases], 0)?;
    Ok(QuantizedLinear {
        weight: w,
        scales: s,
        biases: b,
        group_size: gate.group_size,
        bits: gate.bits,
    })
}

/// Split a fused gate_up projection output [.., 2*inter] into (gate, up).
pub fn split_gate_up(fused_out: &Array, inter: usize) -> anyhow::Result<(Array, Array)> {
    let n = fused_out.dim(-1) as usize;
    debug_assert_eq!(n, 2 * inter);
    let g = narrow(fused_out, fused_out.rank() - 1, 0, inter)?;
    let u = narrow(fused_out, fused_out.rank() - 1, inter, inter)?;
    Ok((g, u))
}

/// Clamped SwiGLU: `silu(min(gate, limit)) * clip(up, -limit, limit)`.
pub fn clamped_swiglu(up: &Array, gate: &Array, limit: f32) -> anyhow::Result<Array> {
    let gc = gate.contiguous()?;
    let uc = up.contiguous()?;
    let stream = lisa_mlx::Stream::thread_local_or_default();
    lisa_mlx::models::glm53::kernels::swiglu_fused(&gc, &uc, limit, &stream)
        .ok_or_else(|| anyhow::anyhow!("GLM fused clamped-SwiGLU Metal kernel failed"))
}

/// The paired Q8 projection is the production path. The separate mode keeps
/// the original two-dispatch schedule available for controlled end-to-end
/// measurements; it is selected once while loading the model.
enum GateUpProjection {
    Paired {
        projection: QuantizedLinear,
        gate: QuantizedLinear,
        up: QuantizedLinear,
    },
    Separate {
        gate: QuantizedLinear,
        up: QuantizedLinear,
    },
}

struct ActivatedProjection {
    projections: Option<(Array, Array)>,
    activated: Array,
}

impl GateUpProjection {
    fn load_mode() -> anyhow::Result<bool> {
        match std::env::var("LISA_GLM53_Q8_GATE_UP").as_deref() {
            Ok("separate") => Ok(false),
            Ok("paired") | Err(std::env::VarError::NotPresent) => Ok(true),
            Ok(value) => {
                anyhow::bail!("LISA_GLM53_Q8_GATE_UP must be `paired` or `separate`, got {value:?}")
            }
            Err(error) => anyhow::bail!("cannot read LISA_GLM53_Q8_GATE_UP: {error}"),
        }
    }

    fn from_parts(
        gate: QuantizedLinear,
        up: QuantizedLinear,
        paired: bool,
    ) -> anyhow::Result<Self> {
        if paired {
            let projection = fuse_gate_up(&gate, &up)?;
            let (gate, up) = split_gate_up_linear(&projection, projection.dims_out() / 2)?;
            Ok(Self::Paired {
                projection,
                gate,
                up,
            })
        } else {
            Ok(Self::Separate { gate, up })
        }
    }

    fn inter(&self) -> usize {
        match self {
            Self::Paired { projection, .. } => projection.dims_out() / 2,
            Self::Separate { gate, .. } => gate.dims_out(),
        }
    }

    fn split_linears(&self) -> (&QuantizedLinear, &QuantizedLinear) {
        match self {
            Self::Paired { gate, up, .. } | Self::Separate { gate, up } => (gate, up),
        }
    }

    fn forward(&self, x: &Array) -> anyhow::Result<(Array, Array)> {
        match self {
            Self::Paired { projection, .. } => {
                let output = projection.forward_glm53_prefill(x)?;
                let inter = projection.dims_out() / 2;
                let n = output.dim(-1) as usize;
                let (gate, up) = split_gate_up(&output, inter)?;
                debug_assert_eq!(up.dim(-1) as usize, n - inter);
                Ok((gate, up))
            }
            Self::Separate { gate, up } => {
                Ok((gate.forward_glm53_prefill(x)?, up.forward_glm53_prefill(x)?))
            }
        }
    }

    fn forward_native_activated(
        &self,
        x: &Array,
        limit: f32,
        capture_projections: bool,
    ) -> anyhow::Result<Option<ActivatedProjection>> {
        if !GLM53_Q8_GATE_UP_NATIVE.with(Cell::get) || matches!(self, Self::Separate { .. }) {
            return Ok(None);
        }
        let shape = x.shape();
        let rows = shape[..shape.len() - 1].iter().product::<i32>() as usize;
        let k = x.dim(-1) as usize;
        if !(1..=16).contains(&rows)
            || k % 128 != 0
            || self.inter() % 2 != 0
            || x.dtype() != Dtype::Float32
        {
            return Ok(None);
        }
        let (gate, up) = self.split_linears();
        if [gate, up]
            .iter()
            .any(|projection| projection.bits != 8 || projection.group_size != 64)
        {
            return Ok(None);
        }
        let flat = x.reshape(&[rows as i32, k as i32])?.contiguous()?;
        // Gate/up are contiguous output-row views of the already-owned
        // projection storage. No repack or new weight allocation occurs here.
        for tensor in [
            &flat,
            &gate.weight,
            &gate.scales,
            &gate.biases,
            &up.weight,
            &up.scales,
            &up.biases,
        ] {
            tensor.eval_enqueue()?;
        }
        let output_shape = [&shape[..shape.len() - 1], &[self.inter() as i32]].concat();
        if !capture_projections {
            let activated = lisa_mlx::jit::glm53_affine_q8_swiglu_native(
                flat.t.device(),
                &flat.t,
                &gate.weight.t,
                &gate.scales.t,
                &gate.biases.t,
                &up.weight.t,
                &up.scales.t,
                &up.biases.t,
                limit,
            )?;
            let activated = Array::new(activated).reshape(&output_shape)?;
            return Ok(Some(ActivatedProjection {
                projections: None,
                activated,
            }));
        }
        let (g, u, activated) = lisa_mlx::jit::glm53_affine_q8_gate_up_native(
            flat.t.device(),
            &flat.t,
            &gate.weight.t,
            &gate.scales.t,
            &gate.biases.t,
            &up.weight.t,
            &up.scales.t,
            &up.biases.t,
            limit,
        )?;
        Ok(Some(ActivatedProjection {
            projections: Some((
                Array::new(g).reshape(&output_shape)?,
                Array::new(u).reshape(&output_shape)?,
            )),
            activated: Array::new(activated).reshape(&output_shape)?,
        }))
    }
}

/// Clamped GLM MLP. Leading dense layers use a concatenated gate/up Q8
/// projection. Shared experts use the same exact paired-projection schedule.
pub struct DenseMlp {
    gate_up: GateUpProjection,
    down: QuantizedLinear,
    limit: f32,
}

impl DenseMlp {
    /// The projection mode captured at load, independent of later environment changes.
    pub(super) fn paired_storage(&self) -> bool {
        matches!(&self.gate_up, GateUpProjection::Paired { .. })
    }

    pub fn load<S: TensorSource>(w: &mut S, prefix: &str, limit: f32) -> anyhow::Result<Self> {
        let (gate, up, down) = Self::load_parts(w, prefix)?;
        Ok(Self {
            gate_up: GateUpProjection::from_parts(gate, up, GateUpProjection::load_mode()?)?,
            down,
            limit,
        })
    }

    fn load_shared<S: TensorSource>(w: &mut S, prefix: &str, limit: f32) -> anyhow::Result<Self> {
        Self::load::<S>(w, prefix, limit)
    }

    fn load_parts<S: TensorSource>(
        w: &mut S,
        prefix: &str,
    ) -> anyhow::Result<(QuantizedLinear, QuantizedLinear, QuantizedLinear)> {
        let mut gate = QuantizedLinear::load(w, &format!("{prefix}.gate_proj"), "")?;
        let mut up = QuantizedLinear::load(w, &format!("{prefix}.up_proj"), "")?;
        let mut down = QuantizedLinear::load(w, &format!("{prefix}.down_proj"), "")?;
        for q in [&mut gate, &mut up, &mut down] {
            q.set_quant(64, 8);
            if q.scales.dtype() != Dtype::Bfloat16 {
                q.scales = q.scales.as_dtype(Dtype::Float32)?;
                q.biases = q.biases.as_dtype(Dtype::Float32)?;
            }
        }
        Ok((gate, up, down))
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        self.forward_with_capture(x, None, 0)
    }

    pub fn forward_with_capture(
        &self,
        x: &Array,
        capture: Option<&Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dense_mlp_input"), x);
        }
        // Logit-only captures must use the activation-only production path.
        let projection_capture =
            capture.filter(|capture| capture.accepts_key(&format!("L{layer}.dense_gate_up")));
        let native =
            self.gate_up
                .forward_native_activated(x, self.limit, projection_capture.is_some())?;
        let ActivatedProjection {
            projections,
            activated,
        } = if let Some(native) = native {
            native
        } else {
            let (gate, up) = self.gate_up.forward(x)?;
            let activated = clamped_swiglu(&up, &gate, self.limit)?;
            ActivatedProjection {
                projections: Some((gate, up)),
                activated,
            }
        };
        if let Some(capture) = projection_capture {
            let (gate, up) = projections
                .as_ref()
                .expect("capture requested projection outputs");
            let gate_up = ops::concatenate(&[gate, up], -1)?;
            capture.store(&format!("L{layer}.dense_gate_up"), &gate_up);
        }
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dense_swiglu"), &activated);
        }
        let out = self.forward_projection(&self.down, &activated)?;
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dense_out"), &out);
        }
        Ok(out)
    }

    /// Use DS4's row-count-specific Q8 projection schedule over the original
    /// affine storage: M=1, small-row and matrix paths keep their own arithmetic.
    fn forward_projection(
        &self,
        projection: &QuantizedLinear,
        x: &Array,
    ) -> lisa_mlx::error::Result<Array> {
        projection.forward_glm53_prefill(x)
    }
}

fn split_gate_up_linear(
    gate_up: &QuantizedLinear,
    inter: usize,
) -> anyhow::Result<(QuantizedLinear, QuantizedLinear)> {
    let split = |array: &Array, start: usize| -> anyhow::Result<Array> {
        Ok(narrow(array, 0, start, inter)?.contiguous()?)
    };
    let make = |start: usize| -> anyhow::Result<QuantizedLinear> {
        Ok(QuantizedLinear {
            weight: split(&gate_up.weight, start)?,
            scales: split(&gate_up.scales, start)?,
            biases: split(&gate_up.biases, start)?,
            group_size: gate_up.group_size,
            bits: gate_up.bits,
        })
    };
    Ok((make(0)?, make(inter)?))
}

#[derive(Clone)]
struct ExpertParameters {
    gate: QuantizedLinear,
    up: QuantizedLinear,
    down: QuantizedLinear,
}

impl ExpertParameters {
    fn byte_size(&self) -> usize {
        ExpertPools::arrays(self)
            .iter()
            .map(|array| array.size().saturating_mul(array.dtype().size_of()))
            .sum()
    }

    fn copied(&self) -> anyhow::Result<Self> {
        let copy_linear = |linear: &QuantizedLinear| -> anyhow::Result<QuantizedLinear> {
            let copy =
                |array: &Array| -> anyhow::Result<Array> { Ok(Array::new(array.t.copied()?)) };
            Ok(QuantizedLinear {
                weight: copy(&linear.weight)?,
                scales: copy(&linear.scales)?,
                biases: copy(&linear.biases)?,
                group_size: linear.group_size,
                bits: linear.bits,
            })
        };
        Ok(Self {
            gate: copy_linear(&self.gate)?,
            up: copy_linear(&self.up)?,
            down: copy_linear(&self.down)?,
        })
    }
}

struct CachedExpert {
    parameters: ExpertParameters,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct ExpertCacheState {
    entries: HashMap<usize, CachedExpert>,
    bytes: usize,
    clock: u64,
    address_table: Option<Array>,
}

enum ExpertStageMessage {
    Stage {
        pools: Weak<ExpertPools>,
        selected: Array,
        result: SyncSender<anyhow::Result<(Array, Vec<Array>)>>,
    },
}

struct ExpertPools {
    experts: Vec<ExpertParameters>,
    expert_cache_bytes: Vec<usize>,
    lazy_store: Option<Arc<LazyTensorStore>>,
    expert_prefix: Option<String>,
    inter: usize,
    address_table: Array,
    metadata_dtype: Dtype,
    residency_set: std::sync::OnceLock<std::sync::Arc<lisa_mlx::runtime::ResidencySet>>,
    gpu_address_all_experts: bool,
    cache_capacity_bytes: AtomicUsize,
    cache: Mutex<ExpertCacheState>,
}

fn selected_prefetch_bytes(expert_ids: &[usize], expert_cache_bytes: &[usize]) -> Option<usize> {
    expert_ids.iter().try_fold(0usize, |total, &expert_id| {
        total.checked_add(*expert_cache_bytes.get(expert_id)?)
    })
}

impl ExpertPools {
    fn arrays(expert: &ExpertParameters) -> [&Array; 9] {
        [
            &expert.gate.weight,
            &expert.gate.scales,
            &expert.gate.biases,
            &expert.up.weight,
            &expert.up.scales,
            &expert.up.biases,
            &expert.down.weight,
            &expert.down.scales,
            &expert.down.biases,
        ]
    }

    fn indirect_resources(
        experts: &[ExpertParameters],
        selected_ids: &[i32],
    ) -> anyhow::Result<Vec<Array>> {
        let mut seen = HashSet::new();
        let mut resources = Vec::new();
        for &selected_id in selected_ids {
            let expert_id = usize::try_from(selected_id).map_err(|_| {
                anyhow::anyhow!("glm53 moe: negative selected expert id {selected_id}")
            })?;
            let expert = experts.get(expert_id).ok_or_else(|| {
                anyhow::anyhow!("glm53 moe: selected expert id {expert_id} is out of range")
            })?;
            for array in Self::arrays(expert) {
                if seen.insert(array.backing_buffer_id()) {
                    resources.push(array.clone());
                }
            }
        }
        Ok(resources)
    }

    fn load_lazy_expert_weights(&self, expert_id: usize) -> anyhow::Result<ExpertParameters> {
        let store = self
            .lazy_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("glm53 expert lazy store is unavailable"))?;
        let prefix = self
            .expert_prefix
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("glm53 expert prefix is unavailable"))?;
        let mut parameters = self.experts[expert_id].clone();
        for (projection, linear) in [
            ("gate_proj", &mut parameters.gate),
            ("up_proj", &mut parameters.up),
            ("down_proj", &mut parameters.down),
        ] {
            // Keep weights and affine metadata page-scoped. File-backed typed
            // views avoid copying six selected metadata payloads into a new
            // Metal buffer for every cache miss.
            let weight_name = format!("{prefix}.experts.{expert_id}.{projection}.weight");
            linear.weight = {
                let _trace = lisa_mlx::trace::span("glm53.moe.expert_weight_mapping");
                store.get_mapped_weight_bytes_individual(&weight_name)?
            };
            let scales_name = format!("{prefix}.experts.{expert_id}.{projection}.scales");
            linear.scales = {
                let _trace = lisa_mlx::trace::span("glm53.moe.expert_metadata_mapping");
                store.get_mapped_individual(&scales_name)?
            };
            let biases_name = format!("{prefix}.experts.{expert_id}.{projection}.biases");
            linear.biases = {
                let _trace = lisa_mlx::trace::span("glm53.moe.expert_metadata_mapping");
                store.get_mapped_individual(&biases_name)?
            };
        }
        anyhow::ensure!(
            parameters.byte_size() == self.expert_cache_bytes[expert_id],
            "glm53 expert {expert_id} loaded payload size differs from safetensors metadata"
        );
        Ok(parameters)
    }

    fn load_lazy_experts(
        &self,
        expert_ids: &[usize],
    ) -> anyhow::Result<Vec<(usize, ExpertParameters)>> {
        if expert_ids.is_empty() {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            self.lazy_store.is_some(),
            "glm53 expert lazy store is unavailable"
        );
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(4)
            .min(expert_ids.len());
        if workers == 1 {
            return expert_ids
                .iter()
                .copied()
                .map(|expert_id| Ok((expert_id, self.load_lazy_expert_weights(expert_id)?)))
                .collect();
        }

        let chunk_size = expert_ids.len().div_ceil(workers);
        std::thread::scope(|scope| -> anyhow::Result<Vec<(usize, ExpertParameters)>> {
            let handles = expert_ids
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .copied()
                            .map(|expert_id| {
                                Ok((expert_id, self.load_lazy_expert_weights(expert_id)?))
                            })
                            .collect::<anyhow::Result<Vec<_>>>()
                    })
                })
                .collect::<Vec<_>>();
            let mut loaded = Vec::with_capacity(expert_ids.len());
            for handle in handles {
                loaded.extend(
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("glm53 expert staging thread panicked"))??,
                );
            }
            Ok(loaded)
        })
    }

    fn prefetch_selected_experts(&self, expert_ids: &[usize]) -> anyhow::Result<()> {
        if expert_ids.is_empty() {
            return Ok(());
        }
        // Read ahead only when the complete selected set fits this pool's
        // bounded cache. A fixed expert-count cutoff skipped useful prefill
        // selections even when their actual payload was small enough to fit.
        // Oversized selections stay demand-driven so a wide batch cannot
        // trigger unbounded page-in pressure.
        let Some(selected_bytes) = selected_prefetch_bytes(expert_ids, &self.expert_cache_bytes)
        else {
            anyhow::bail!("glm53 moe: selected prefetch byte count overflow or invalid expert id");
        };
        let cache_capacity = self.cache_capacity_bytes.load(Ordering::Relaxed);
        if selected_bytes > cache_capacity {
            if lisa_mlx::trace::enabled_for("glm53.moe.expert_prefetch") {
                eprintln!(
                    "[glm53-moe] skipped read-ahead for {} experts ({selected_bytes} bytes exceeds {cache_capacity}-byte pool cache)",
                    expert_ids.len()
                );
            }
            return Ok(());
        }
        let Some(store) = &self.lazy_store else {
            return Ok(());
        };
        let Some(prefix) = self.expert_prefix.as_deref() else {
            return Ok(());
        };
        let names = expert_ids
            .iter()
            .flat_map(|expert_id| {
                ["gate_proj", "up_proj", "down_proj"]
                    .into_iter()
                    .flat_map(move |projection| {
                        ["weight", "scales", "biases"]
                            .into_iter()
                            .map(move |field| {
                                format!("{prefix}.experts.{expert_id}.{projection}.{field}")
                            })
                    })
            })
            .collect::<Vec<_>>();
        let (ranges, bytes) = {
            let _trace = lisa_mlx::trace::span("glm53.moe.expert_prefetch");
            store.prefetch_mapped_tensors(&names)?
        };
        if lisa_mlx::trace::enabled_for("glm53.moe.expert_prefetch") {
            eprintln!(
                "[glm53-moe] prefetched {} selected tensor regions ({} bytes) for {} experts",
                ranges,
                bytes,
                expert_ids.len()
            );
        }
        Ok(())
    }

    fn transient_selected(
        &self,
        requested: &[usize],
    ) -> anyhow::Result<(Array, Vec<(usize, ExpertParameters)>)> {
        self.prefetch_selected_experts(requested)?;
        let mut experts = self.experts.clone();
        let active = {
            let _trace = lisa_mlx::trace::span("glm53.moe.transient_expert_map");
            self.load_lazy_experts(requested)?
        };
        for (expert_id, parameters) in &active {
            experts[*expert_id] = parameters.clone();
        }
        Ok((Self::address_table(&experts)?.0, active))
    }

    fn stage_selected(
        &self,
        selected_ids: &[i32],
    ) -> anyhow::Result<Option<(Array, Vec<(usize, ExpertParameters)>)>> {
        let capacity = self.cache_capacity_bytes.load(Ordering::Relaxed);

        let mut requested = Vec::with_capacity(selected_ids.len());
        let mut requested_set = HashSet::with_capacity(selected_ids.len());
        for &selected_id in selected_ids {
            let expert_id = usize::try_from(selected_id).map_err(|_| {
                anyhow::anyhow!("glm53 moe: negative selected expert id {selected_id}")
            })?;
            anyhow::ensure!(
                expert_id < self.experts.len(),
                "glm53 moe: selected expert id {expert_id} is out of range"
            );
            if requested_set.insert(expert_id) {
                requested.push(expert_id);
            }
        }

        let active_bytes = requested.iter().try_fold(0usize, |total, &expert_id| {
            total.checked_add(self.expert_cache_bytes[expert_id])
        });
        let Some(active_bytes) = active_bytes else {
            return Ok(None);
        };
        if capacity == 0 || active_bytes > capacity {
            return if self.lazy_store.is_some() {
                self.transient_selected(&requested).map(Some)
            } else {
                Ok(None)
            };
        }

        let mut cache = self.cache.lock().unwrap();
        let missing_bytes = requested
            .iter()
            .filter(|&&expert_id| !cache.entries.contains_key(&expert_id))
            .try_fold(0usize, |total, &expert_id| {
                total.checked_add(self.expert_cache_bytes[expert_id])
            });
        let Some(missing_bytes) = missing_bytes else {
            return Ok(None);
        };
        let completed = lisa_mlx::Stream::thread_local_or_default()
            .runtime()
            .commands
            .completed_watermark();
        let mut residency_changed = false;
        let mut victims = cache
            .entries
            .iter()
            .filter(|(expert_id, entry)| {
                !requested_set.contains(expert_id)
                    && ExpertPools::arrays(&entry.parameters)
                        .iter()
                        .all(|array| array.t.buffer().reusable_after(completed))
            })
            .map(|(&expert_id, entry)| (expert_id, entry.last_used, entry.bytes))
            .collect::<Vec<_>>();
        victims.sort_by_key(|&(_, last_used, _)| last_used);

        let mut freed = 0usize;
        let mut evict = Vec::new();
        for (expert_id, _, bytes) in victims {
            if cache
                .bytes
                .saturating_sub(freed)
                .saturating_add(missing_bytes)
                <= capacity
            {
                break;
            }
            freed = freed.saturating_add(bytes);
            evict.push(expert_id);
        }
        if cache
            .bytes
            .saturating_sub(freed)
            .saturating_add(missing_bytes)
            > capacity
        {
            if self.lazy_store.is_some() {
                drop(cache);
                return self.transient_selected(&requested).map(Some);
            }
            return Ok(None);
        }
        let mut address_table_changed = !evict.is_empty();
        for expert_id in evict {
            if let Some(entry) = cache.entries.remove(&expert_id) {
                if let Some(set) = self.residency_set.get().map(Arc::as_ref) {
                    for array in ExpertPools::arrays(&entry.parameters) {
                        residency_changed |= set.remove(array.t.buffer());
                    }
                }
                cache.bytes = cache.bytes.saturating_sub(entry.bytes);
            }
        }

        let missing_experts = requested
            .iter()
            .copied()
            .filter(|expert_id| !cache.entries.contains_key(expert_id))
            .collect::<Vec<_>>();
        self.prefetch_selected_experts(&missing_experts)?;
        let mut loaded_experts = if self.lazy_store.is_some() {
            self.load_lazy_experts(&missing_experts)?
                .into_iter()
                .collect::<HashMap<_, _>>()
        } else {
            HashMap::new()
        };

        for &expert_id in &requested {
            cache.clock = cache.clock.wrapping_add(1).max(1);
            let clock = cache.clock;
            if let Some(entry) = cache.entries.get_mut(&expert_id) {
                entry.last_used = clock;
                continue;
            }
            let parameters = if self.lazy_store.is_some() {
                loaded_experts
                    .remove(&expert_id)
                    .expect("missing lazy expert was staged")
            } else {
                self.experts[expert_id].copied()?
            };
            let bytes = self.expert_cache_bytes[expert_id];
            anyhow::ensure!(
                parameters.byte_size() == bytes,
                "glm53 expert cache entry changed expert byte size"
            );
            if let Some(set) = self.residency_set.get().map(Arc::as_ref) {
                for array in ExpertPools::arrays(&parameters) {
                    residency_changed |= set.add(array.t.buffer());
                }
            }
            cache.bytes = cache.bytes.saturating_add(bytes);
            cache.entries.insert(
                expert_id,
                CachedExpert {
                    parameters,
                    bytes,
                    last_used: clock,
                },
            );
            address_table_changed = true;
        }

        if residency_changed {
            let set = self
                .residency_set
                .get()
                .map(Arc::as_ref)
                .expect("residency set changed");
            {
                let _trace = lisa_mlx::trace::span("glm53.moe.residency_commit");
                set.commit();
            }
            {
                let _trace = lisa_mlx::trace::span("glm53.moe.residency_request");
                set.request_residency();
            }
        }

        if address_table_changed || cache.address_table.is_none() {
            let _trace = lisa_mlx::trace::span("glm53.moe.address_table_rebuild");
            let mut experts = self.experts.clone();
            for (&expert_id, entry) in &cache.entries {
                experts[expert_id] = entry.parameters.clone();
            }
            cache.address_table = Some(Self::address_table(&experts)?.0);
        }
        let address_table = cache
            .address_table
            .as_ref()
            .expect("selected expert cache has an address table")
            .clone();
        let active = requested
            .into_iter()
            .map(|expert_id| (expert_id, cache.entries[&expert_id].parameters.clone()))
            .collect();
        Ok(Some((address_table, active)))
    }

    fn dispatch_resources(&self, selected_ids: &[i32]) -> anyhow::Result<(Array, Vec<Array>)> {
        let staged = {
            let _trace = lisa_mlx::trace::span("glm53.moe.expert_cache_stage");
            self.stage_selected(selected_ids)?
        };
        let _resource_trace = lisa_mlx::trace::span("glm53.moe.resource_list_build");
        let Some((address_table, staged)) = staged else {
            let mut resources = Self::indirect_resources(&self.experts, selected_ids)?;
            let all_resident = self
                .residency_set
                .get()
                .map(Arc::as_ref)
                .is_some_and(|set| {
                    resources
                        .iter()
                        .all(|array| set.contains(array.backing_buffer_id()))
                });
            if all_resident {
                resources.clear();
            }
            return Ok((self.address_table.clone(), resources));
        };
        let mut seen = HashSet::new();
        let mut resources = Vec::new();
        for (_, expert) in staged {
            for array in Self::arrays(&expert) {
                if seen.insert(array.backing_buffer_id()) {
                    resources.push(array.clone());
                }
            }
        }
        // Keep explicit declarations even when the bounded residency set also
        // contains these buffers. Membership records the committed resource
        // list; it does not prove that a cold requestResidency() has completed
        // before this command buffer starts using indirect GPU addresses.
        Ok((address_table, resources))
    }

    /// Queue selected-ID readback and expert staging on one persistent worker.
    /// The caller can run the shared-expert branch while the worker waits for
    /// the router command buffer and updates this bounded cache.
    fn stage_selected_async(
        self: &Arc<Self>,
        selected: &Array,
    ) -> Option<Receiver<anyhow::Result<(Array, Vec<Array>)>>> {
        static WORKER: std::sync::OnceLock<Option<mpsc::Sender<ExpertStageMessage>>> =
            std::sync::OnceLock::new();
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let sender = WORKER
            .get_or_init(|| {
                let (sender, receiver) = mpsc::channel();
                let thread = std::thread::Builder::new()
                    .name("lisa-glm53-expert-stage".to_string())
                    .spawn(move || {
                        while let Ok(ExpertStageMessage::Stage {
                            pools,
                            selected,
                            result,
                        }) = receiver.recv()
                        {
                            let staged = match pools.upgrade() {
                                Some(pools) => (|| {
                                    let selected_ids = {
                                        let _trace = lisa_mlx::trace::span(
                                            "glm53.moe.async_selected_id_readback",
                                        );
                                        selected.to_vec1_wait_through::<i32>()?
                                    };
                                    let _trace = lisa_mlx::trace::span(
                                        "glm53.moe.async_expert_dispatch_resources",
                                    );
                                    pools.dispatch_resources(&selected_ids)
                                })(),
                                None => Err(anyhow::anyhow!(
                                    "GLM expert cache was dropped before staging"
                                )),
                            };
                            let _ = result.send(staged);
                        }
                    });
                match thread {
                    Ok(_thread) => Some(sender),
                    Err(error) => {
                        eprintln!("[glm53-moe] async expert staging unavailable: {error}");
                        None
                    }
                }
            })
            .as_ref()?;
        sender
            .send(ExpertStageMessage::Stage {
                pools: Arc::downgrade(self),
                selected: selected.clone(),
                result: result_tx,
            })
            .ok()?;
        Some(result_rx)
    }

    fn load_experts(
        store: &LazyTensorStore,
        prefix: &str,
        n_experts: usize,
        map_all_weights: bool,
        share_metadata_shards: bool,
    ) -> anyhow::Result<(Vec<ExpertParameters>, Vec<usize>)> {
        anyhow::ensure!(n_experts > 0, "glm53 moe: no routed experts");
        let metadata_dtype = store.tensor_dtype(&format!("{prefix}.experts.0.gate_proj.scales"))?;
        anyhow::ensure!(
            matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32),
            "glm53 expert metadata dtype {metadata_dtype:?} is unsupported by address kernels"
        );
        // Full-tower diagnostics map metadata through its source shard so it
        // shares backing storage with the model. The isolated mapped-kernel
        // test packs metadata to keep its small resource table bounded. The
        // production path retains typed placeholders and loads selected data
        // through the bounded cache.
        let mut metadata_views = HashMap::<(String, String), Vec<Array>>::new();
        let mut metadata_name_build = std::time::Duration::ZERO;
        let mut metadata_views_build = std::time::Duration::ZERO;
        if map_all_weights {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for field in ["scales", "biases"] {
                    let name_started = Instant::now();
                    let names = (0..n_experts)
                        .map(|expert| format!("{prefix}.experts.{expert}.{projection}.{field}"))
                        .collect::<Vec<_>>();
                    metadata_name_build += name_started.elapsed();
                    let view_started = Instant::now();
                    let views = if share_metadata_shards {
                        let mut views = Vec::with_capacity(n_experts);
                        let mut expert_shape: Option<Vec<i32>> = None;
                        for name in &names {
                            let view = store.get_mapped(name)?;
                            anyhow::ensure!(
                                view.dtype() == metadata_dtype && !view.shape().is_empty(),
                                "GLM {projection} {field} metadata shape or dtype is invalid for {name}"
                            );
                            if let Some(expected_shape) = &expert_shape {
                                anyhow::ensure!(
                                    view.shape() == expected_shape,
                                    "GLM {projection} {field} metadata shape changed for {name}"
                                );
                            } else {
                                expert_shape = Some(view.shape().to_vec());
                            }
                            let expected_bytes = store.tensor_nbytes(name)?;
                            anyhow::ensure!(
                                view.size() * metadata_dtype.size_of() == expected_bytes,
                                "GLM {projection} {field} metadata view changed byte length for {name}"
                            );
                            views.push(view);
                        }
                        views
                    } else {
                        let packed = store.get_packed(&names)?;
                        anyhow::ensure!(
                            packed.dtype() == metadata_dtype && !packed.shape().is_empty(),
                            "GLM {projection} {field} metadata shape or dtype is invalid"
                        );
                        let mut expert_shape = packed.shape().to_vec();
                        anyhow::ensure!(
                            expert_shape[0] % n_experts as i32 == 0,
                            "GLM {projection} {field} metadata is not evenly packed by expert"
                        );
                        expert_shape[0] /= n_experts as i32;
                        let expert_bytes = store.tensor_nbytes(&names[0])?;
                        anyhow::ensure!(
                            packed.size() * metadata_dtype.size_of() == expert_bytes * n_experts,
                            "GLM {projection} {field} metadata pack changed byte length"
                        );
                        (0..n_experts)
                            .map(|expert| {
                                Array::from_buffer_offset(
                                    packed.t.buffer().clone(),
                                    expert * expert_bytes,
                                    &expert_shape,
                                    metadata_dtype,
                                )
                                .map_err(anyhow::Error::from)
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?
                    };
                    metadata_views_build += view_started.elapsed();
                    metadata_views.insert((projection.to_string(), field.to_string()), views);
                }
            }
        }

        // Inactive experts need valid pointer slots in the address table. The
        // selected-expert path replaces both weight and metadata placeholders
        // before dispatch; the full-bank path supplies mapped weights and
        // mapped metadata for every expert.
        let placeholder_weight = ops::zeros::<u8>(&[1])?;
        let placeholder_metadata = Array::from_slice(&[0.0f32], &[1]).as_dtype(metadata_dtype)?;
        let load_linear = |expert: usize, projection: &str| -> anyhow::Result<QuantizedLinear> {
            let weight = if map_all_weights {
                store.get_mapped_weight_bytes(&format!(
                    "{prefix}.experts.{expert}.{projection}.weight"
                ))?
            } else {
                placeholder_weight.clone()
            };
            let (scales, biases) = if map_all_weights {
                (
                    metadata_views[&(projection.to_string(), "scales".to_string())][expert].clone(),
                    metadata_views[&(projection.to_string(), "biases".to_string())][expert].clone(),
                )
            } else {
                (placeholder_metadata.clone(), placeholder_metadata.clone())
            };
            Ok(QuantizedLinear {
                weight,
                scales,
                biases,
                group_size: 128,
                bits: 2,
            })
        };
        let experts = (0..n_experts)
            .map(|expert| {
                Ok(ExpertParameters {
                    gate: load_linear(expert, "gate_proj")?,
                    up: load_linear(expert, "up_proj")?,
                    down: load_linear(expert, "down_proj")?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let expert_bytes_started = Instant::now();
        let expert_bytes = (0..n_experts)
            .map(|expert| -> anyhow::Result<usize> {
                ["gate_proj", "up_proj", "down_proj"]
                    .into_iter()
                    .flat_map(|projection| {
                        ["weight", "scales", "biases"]
                            .into_iter()
                            .map(move |field| (projection, field))
                    })
                    .try_fold(0usize, |total, (projection, field)| {
                        let size = store.tensor_nbytes(&format!(
                            "{prefix}.experts.{expert}.{projection}.{field}"
                        ))?;
                        total.checked_add(size).ok_or_else(|| {
                            anyhow::anyhow!("glm53 expert {expert} byte count overflow")
                        })
                    })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if lisa_mlx::trace::enabled_for("glm53-loader") {
            eprintln!(
                "[glm53-loader] metadata phases for {prefix}: names={:.1}ms views={:.1}ms expert_sizes={:.1}ms",
                metadata_name_build.as_secs_f64() * 1000.0,
                metadata_views_build.as_secs_f64() * 1000.0,
                expert_bytes_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok((experts, expert_bytes))
    }

    fn address_table(experts: &[ExpertParameters]) -> anyhow::Result<(Array, Dtype)> {
        anyhow::ensure!(!experts.is_empty(), "glm53 moe: no routed experts");
        let mut addresses = Vec::<i64>::with_capacity(experts.len() * 9);
        let mut metadata_dtype = None;
        for (expert_id, expert) in experts.iter().enumerate() {
            for (projection, linear) in [
                ("gate", &expert.gate),
                ("up", &expert.up),
                ("down", &expert.down),
            ] {
                anyhow::ensure!(
                    linear.scales.dtype() == linear.biases.dtype(),
                    "glm53 expert {expert_id} {projection} scale/bias dtype mismatch"
                );
                match metadata_dtype {
                    Some(dtype) => anyhow::ensure!(
                        dtype == linear.scales.dtype(),
                        "glm53 expert metadata dtype changed at expert {expert_id}"
                    ),
                    None => metadata_dtype = Some(linear.scales.dtype()),
                }
                for array in [&linear.weight, &linear.scales, &linear.biases] {
                    let address = array.gpu_address().ok_or_else(|| {
                        anyhow::anyhow!(
                            "glm53 expert {expert_id} {projection} has no Metal GPU address"
                        )
                    })?;
                    addresses.push(address as i64);
                }
            }
        }
        let dtype = metadata_dtype.expect("nonempty expert set has metadata dtype");
        anyhow::ensure!(
            matches!(dtype, Dtype::Bfloat16 | Dtype::Float32),
            "glm53 routed metadata dtype {dtype:?} is unsupported by address kernels"
        );
        let table = Array::from_slice(&addresses, &[addresses.len() as i32]);
        Ok((table, dtype))
    }
}

/// The four large expert intermediates are shared by every GLM layer.
/// Serial Metal dispatch and command-buffer queue order make the next write
/// follow the previous sum8 read. The mutex holds that producer/consumer
/// sequence together during encoding, including concurrent host callers.
/// Captures bypass this workspace because they retain intermediate arrays.
#[derive(Default)]
pub(crate) struct GroupedMoeWorkspace {
    buffers: Mutex<GroupedMoeBuffers>,
}

#[derive(Default)]
struct GroupedMoeBuffers {
    device: Option<Weak<lisa_mlx::runtime::MetalRuntime>>,
    gate: Option<Arc<lisa_mlx::runtime::Buffer>>,
    up: Option<Arc<lisa_mlx::runtime::Buffer>>,
    act: Option<Arc<lisa_mlx::runtime::Buffer>>,
    down: Option<Arc<lisa_mlx::runtime::Buffer>>,
}

struct GroupedMoeViews {
    gate: Option<lisa_mlx::jit::Tensor>,
    up: Option<lisa_mlx::jit::Tensor>,
    act: lisa_mlx::jit::Tensor,
    down: lisa_mlx::jit::Tensor,
}

impl GroupedMoeBuffers {
    fn output(
        slot: &mut Option<Arc<lisa_mlx::runtime::Buffer>>,
        device: &lisa_mlx::jit::Device,
        rows: usize,
        width: usize,
        dtype: lisa_mlx::jit::DType,
        label: &str,
    ) -> anyhow::Result<lisa_mlx::jit::Tensor> {
        let bytes = rows
            .checked_mul(width)
            .and_then(|elements| elements.checked_mul(dtype.size_of()))
            .ok_or_else(|| anyhow::anyhow!("{label}: workspace size overflow"))?;
        if slot.as_ref().is_none_or(|buffer| buffer.length() < bytes) {
            // Every active output element is written by the grouped kernels.
            // Never clear a reused buffer on the host: older GPU work may
            // still be consuming it while the next layer is being encoded.
            *slot = Some(device.buffer_for_full_write(bytes, label)?);
        }
        Ok(lisa_mlx::jit::Tensor::from_parts(
            device,
            slot.as_ref().expect("workspace buffer allocated").clone(),
            &[rows, width],
            dtype,
        ))
    }

    fn prepare(
        &mut self,
        device: &lisa_mlx::jit::Device,
        rows: usize,
        intermediate: usize,
        hidden: usize,
        split_pair: bool,
    ) -> anyhow::Result<GroupedMoeViews> {
        if let Some(owner) = &self.device {
            anyhow::ensure!(
                owner.ptr_eq(&Arc::downgrade(device)),
                "grouped MoE workspace cannot be shared across Metal runtimes"
            );
        } else {
            self.device = Some(Arc::downgrade(device));
        }
        use lisa_mlx::jit::DType;
        let (gate, up) = if split_pair {
            (
                Some(Self::output(
                    &mut self.gate,
                    device,
                    rows,
                    intermediate,
                    DType::F32,
                    "glm53.workspace.gate",
                )?),
                Some(Self::output(
                    &mut self.up,
                    device,
                    rows,
                    intermediate,
                    DType::F32,
                    "glm53.workspace.up",
                )?),
            )
        } else {
            (None, None)
        };
        Ok(GroupedMoeViews {
            gate,
            up,
            act: Self::output(
                &mut self.act,
                device,
                rows,
                intermediate,
                DType::F16,
                "glm53.workspace.act",
            )?,
            down: Self::output(
                &mut self.down,
                device,
                rows,
                hidden,
                DType::F32,
                "glm53.workspace.down",
            )?,
        })
    }
}

/// Sparse MoE (layers >= 3 and the MTP block).
pub struct MoE {
    gate_w: Array, // router projection in F32
    bias: Array,   // router correction bias in F32
    pools: Arc<ExpertPools>,
    shared: Option<DenseMlp>,
    grouped_workspace: Option<Arc<GroupedMoeWorkspace>>,
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling: f32,
    norm_topk: bool,
    limit: f32,
    n_experts: usize,
}

impl MoE {
    pub(super) fn shared_paired_storage(&self) -> Option<bool> {
        self.shared.as_ref().map(DenseMlp::paired_storage)
    }

    pub(crate) fn set_grouped_workspace(&mut self, workspace: Arc<GroupedMoeWorkspace>) {
        self.grouped_workspace = Some(workspace);
    }

    #[cfg(test)]
    pub(crate) fn set_expert_cache_capacity(&self, bytes: usize) {
        self.pools
            .cache_capacity_bytes
            .store(bytes, Ordering::Relaxed);
    }

    pub(crate) fn expert_residency_capacity(&self) -> usize {
        if self.pools.lazy_store.is_some() {
            // Lazy pools contain placeholders until selected experts enter
            // the LRU; they cannot be registered as model-wide resident data.
            return 0;
        }
        self.pools
            .experts
            .iter()
            .flat_map(ExpertPools::arrays)
            .map(Array::backing_buffer_id)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    #[cfg(test)]
    pub(crate) fn cache_residency_capacity(&self) -> usize {
        let capacity = self.pools.cache_capacity_bytes.load(Ordering::Relaxed);
        let smallest_expert = self
            .pools
            .expert_cache_bytes
            .iter()
            .copied()
            .min()
            .unwrap_or(0);
        let cached_experts = if smallest_expert == 0 {
            0
        } else {
            (capacity / smallest_expert).min(self.pools.experts.len())
        };
        cached_experts.saturating_mul(4)
    }

    #[cfg(test)]
    pub(crate) fn new_cache_residency_set(
        &self,
    ) -> Option<std::sync::Arc<lisa_mlx::runtime::ResidencySet>> {
        let capacity = self.cache_residency_capacity();
        if capacity == 0 {
            return None;
        }
        let set = Array::new_residency_set(capacity)?;
        set.commit();
        Some(std::sync::Arc::new(set))
    }

    pub(crate) fn add_expert_resources_to_residency_set(
        &self,
        set: &lisa_mlx::runtime::ResidencySet,
    ) -> bool {
        if self.pools.lazy_store.is_some() {
            return false;
        }
        self.pools
            .experts
            .iter()
            .flat_map(ExpertPools::arrays)
            .all(|array| array.add_to_residency_set(set))
    }

    pub(crate) fn set_model_residency_set(
        &mut self,
        set: std::sync::Arc<lisa_mlx::runtime::ResidencySet>,
    ) {
        let _ = self.pools.residency_set.set(set);
    }

    /// One expert's fused gate_up (host fallback path).
    fn expert_gu(&self, ei: usize) -> anyhow::Result<QuantizedLinear> {
        let expert = self
            .pools
            .experts
            .get(ei)
            .ok_or_else(|| anyhow::anyhow!("glm53 moe: expert id {ei} out of range"))?;
        fuse_gate_up(&expert.gate, &expert.up)
    }

    fn expert_down(&self, ei: usize) -> anyhow::Result<QuantizedLinear> {
        self.pools
            .experts
            .get(ei)
            .map(|expert| expert.down.clone())
            .ok_or_else(|| anyhow::anyhow!("glm53 moe: expert id {ei} out of range"))
    }

    /// GPU-routed expert dispatch. The GPU reads original router expert ids
    /// and resolves each projection through the full checkpoint address table.
    fn forward_address_gpu(
        &self,
        x: &Array,
        sel: &Array,
        wts: &Array,
        b: usize,
        t: usize,
        h: usize,
        shared_for_sum: Option<&Array>,
        prepared_resources: Option<(Array, Vec<Array>)>,
        capture: Option<&Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let n = b * t;
        let k = self.top_k;
        let rows = n * k;
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.moe_input"), x);
        }
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let selected = sel.reshape(&[(n * k) as i32])?;
        let weights = wts.reshape(&[rows as i32])?;
        let x_rows = x.as_dtype(Dtype::Float32)?.reshape(&[n as i32, h as i32])?;
        if self.pools.gpu_address_all_experts {
            // The selector and address-based expert kernels run on the same
            // serial Metal queue. Materialize the producer without waiting:
            // queue order makes its selected IDs visible to the consumer,
            // while a host wait here serializes every routed layer.
            selected.eval_enqueue()?;
        }
        let (address_table, selected_resources) = if self.pools.gpu_address_all_experts {
            (self.pools.address_table.clone(), Vec::new())
        } else if let Some(prepared_resources) = prepared_resources {
            prepared_resources
        } else {
            let selected_ids = {
                let _trace =
                    lisa_mlx::trace::span_detail("glm53.moe.selected_id_readback", layer as u64);
                selected.to_vec1_wait_through::<i32>()?
            };
            self.pools.dispatch_resources(&selected_ids)?
        };
        // A bounded cache uses the residency set only when every routed
        // buffer is already in it. Oversized prefill batches stage transient
        // experts and keep the explicit-resource path instead.
        let residency_set = self
            .pools
            .residency_set
            .get()
            .map(Arc::as_ref)
            .filter(|_| selected_resources.is_empty());
        let indirect_expert_buffers: Vec<&Array> = selected_resources.iter().collect();
        let keepalive: &[&Array] = &indirect_expert_buffers;
        let grouped_prefill = n >= 96
            && k == 8
            && self.pools.inter % 64 == 0
            && h % 128 == 0
            && crate::core::quant::glm53_grouped_moe_prefill_override()
                .unwrap_or(n >= crate::core::quant::GLM53_GROUPED_MOE_MIN_ROWS);
        let grouped_route_map = if grouped_prefill {
            let _trace = lisa_mlx::trace::span_detail("glm53.moe.grouped_route_map", layer as u64);
            Some(
                lisa_mlx::models::glm53::kernels::moe_expert_route_map(
                    &selected,
                    n,
                    k,
                    self.n_experts,
                    32,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe grouped: expert route map declined"))?,
            )
        } else {
            None
        };
        let use_vector_load =
            crate::core::quant::glm53_grouped_moe_q2_vector_load_override().unwrap_or(false);
        let use_split_pair =
            crate::core::quant::glm53_grouped_moe_pair_split_override().unwrap_or(true);
        // Keep the lease until sum8 has been encoded. The returned routed
        // tensor is independently allocated; no workspace view escapes.
        // Any capture, including preserve-dispatch capture, keeps private
        // outputs so its delayed snapshots cannot observe the next layer.
        let mut workspace_lease = if grouped_prefill && capture.is_none() {
            self.grouped_workspace
                .as_ref()
                .map(|workspace| {
                    workspace
                        .buffers
                        .lock()
                        .map_err(|_| anyhow::anyhow!("grouped MoE workspace mutex was poisoned"))
                })
                .transpose()?
        } else {
            None
        };
        let workspace = workspace_lease
            .as_mut()
            .map(|buffers| {
                buffers.prepare(x_rows.t.device(), rows, self.pools.inter, h, use_split_pair)
            })
            .transpose()?;
        let act = {
            if grouped_prefill {
                let (route_ids, counts, work_count, work_items) = grouped_route_map
                    .as_ref()
                    .expect("grouped route map was built");
                let mpp_keepalive = keepalive.iter().map(|array| &array.t).collect::<Vec<_>>();
                let metadata_dtype = match self.pools.metadata_dtype {
                    Dtype::Float32 => lisa_mlx::jit::DType::F32,
                    Dtype::Bfloat16 => lisa_mlx::jit::DType::BF16,
                    dtype => anyhow::bail!("moe grouped: unsupported metadata dtype {dtype:?}"),
                };
                let grouped = if use_split_pair {
                    let gate = {
                        let _trace = lisa_mlx::trace::span_detail(
                            "glm53.moe.grouped_gate_ds4",
                            layer as u64,
                        );
                        lisa_mlx::jit::glm53_moe_q2_mm_grouped_ds4_prefill_with_output(
                            x_rows.t.device(),
                            &address_table.t,
                            &x_rows.t,
                            &route_ids.t,
                            &counts.t,
                            &work_count.t,
                            &work_items.t,
                            &mpp_keepalive,
                            residency_set,
                            metadata_dtype,
                            n,
                            k,
                            self.n_experts,
                            self.pools.inter,
                            false,
                            use_vector_load,
                            workspace.as_ref().and_then(|views| views.gate.as_ref()),
                        )?
                    };
                    let up = {
                        let _trace =
                            lisa_mlx::trace::span_detail("glm53.moe.grouped_up_ds4", layer as u64);
                        lisa_mlx::jit::glm53_moe_q2_mm_grouped_ds4_prefill_with_output(
                            x_rows.t.device(),
                            &address_table.t,
                            &x_rows.t,
                            &route_ids.t,
                            &counts.t,
                            &work_count.t,
                            &work_items.t,
                            &mpp_keepalive,
                            residency_set,
                            metadata_dtype,
                            n,
                            k,
                            self.n_experts,
                            self.pools.inter,
                            true,
                            use_vector_load,
                            workspace.as_ref().and_then(|views| views.up.as_ref()),
                        )?
                    };
                    let _trace =
                        lisa_mlx::trace::span_detail("glm53.moe.grouped_swiglu_ds4", layer as u64);
                    lisa_mlx::jit::glm53_moe_grouped_swiglu_weight_prefill_with_output(
                        x_rows.t.device(),
                        &gate,
                        &up,
                        &weights.t,
                        n * k,
                        self.pools.inter,
                        self.limit,
                        workspace.as_ref().map(|views| &views.act),
                    )?
                } else {
                    let _trace = lisa_mlx::trace::span_detail(
                        "glm53.moe.grouped_pair_fused_ds4",
                        layer as u64,
                    );
                    lisa_mlx::jit::glm53_moe_q2_pair_grouped_ds4_prefill_with_output(
                        x_rows.t.device(),
                        &address_table.t,
                        &x_rows.t,
                        &route_ids.t,
                        &counts.t,
                        &work_count.t,
                        &work_items.t,
                        &weights.t,
                        &mpp_keepalive,
                        residency_set,
                        metadata_dtype,
                        n,
                        k,
                        self.n_experts,
                        self.pools.inter,
                        self.limit,
                        workspace.as_ref().map(|views| &views.act),
                    )?
                };
                Array::new(grouped)
            } else {
                let _trace = lisa_mlx::trace::span_detail("glm53.moe.pair_swiglu", layer as u64);
                lisa_mlx::models::glm53::kernels::moe_pair_swiglu_addr(
                    &address_table,
                    &x_rows,
                    &selected,
                    &weights,
                    keepalive,
                    self.pools.inter,
                    k,
                    self.limit,
                    self.pools.metadata_dtype,
                    residency_set,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe address: pair SwiGLU declined"))?
            }
        }; // [k, inter] F32
        if let Some(capture) = capture {
            capture.store_ephemeral(&format!("L{layer}.moe_pair_weighted"), &act);
        }

        let routed = if grouped_prefill {
            let (route_ids, counts, work_count, work_items) = grouped_route_map
                .as_ref()
                .expect("grouped route map was built");
            let metadata_dtype = match self.pools.metadata_dtype {
                Dtype::Float32 => lisa_mlx::jit::DType::F32,
                Dtype::Bfloat16 => lisa_mlx::jit::DType::BF16,
                dtype => anyhow::bail!("moe grouped down: unsupported metadata dtype {dtype:?}"),
            };
            let _trace = lisa_mlx::trace::span_detail("glm53.moe.down_grouped_ds4", layer as u64);
            let down_slots = lisa_mlx::jit::glm53_moe_q2_down_grouped_ds4_prefill_with_output(
                act.t.device(),
                &address_table.t,
                &act.t,
                &route_ids.t,
                &counts.t,
                &work_count.t,
                &work_items.t,
                &keepalive.iter().map(|array| &array.t).collect::<Vec<_>>(),
                residency_set,
                metadata_dtype,
                n,
                k,
                self.n_experts,
                h,
                use_vector_load,
                crate::core::quant::glm53_grouped_moe_down_tile_override().unwrap_or(64),
                workspace.as_ref().map(|views| &views.down),
            )?;
            let down_slots = Array::new(down_slots);
            if let Some(capture) = capture {
                capture.store_ephemeral(&format!("L{layer}.moe_down_slots"), &down_slots);
            }
            let acc = {
                let _trace =
                    lisa_mlx::trace::span_detail("glm53.moe.grouped_sum8_ds4", layer as u64);
                lisa_mlx::models::glm53::kernels::moe_sum8(&down_slots, n, h, &stream)
                    .ok_or_else(|| anyhow::anyhow!("glm53 grouped MoE: ordered sum8 declined"))?
            };
            let acc = if let Some(shared) = shared_for_sum {
                acc.add(shared)?
            } else {
                acc
            };
            acc
        } else if k == 8 && capture.is_none_or(|capture| capture.preserves_dispatch()) {
            let _trace = lisa_mlx::trace::span_detail("glm53.moe.down_sum8", layer as u64);
            // Keep DS4's per-expert reduction order, add routes in order, and
            // append the shared output in the same dispatch.
            let routed = lisa_mlx::models::glm53::kernels::moe_gather_qmv_sum8_addr_ds4(
                &address_table,
                &act,
                &selected,
                keepalive,
                shared_for_sum,
                h,
                self.pools.metadata_dtype,
                residency_set,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("moe address: DS4 fused down projection declined"))?;
            if let Some(capture) = capture {
                // In diagnostic captures, compare the fused result to the
                // standalone DS4 matvec plus ordered sum and retain each slot.
                let down_slots = lisa_mlx::models::glm53::kernels::moe_gather_qmv_addr_ds4(
                    &address_table,
                    &act,
                    &selected,
                    keepalive,
                    h,
                    true,
                    self.pools.metadata_dtype,
                    residency_set,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe address: DS4 slot capture declined"))?;
                capture.store_ephemeral(&format!("L{layer}.moe_down_slots"), &down_slots);
                let ordered =
                    lisa_mlx::models::glm53::kernels::moe_sum8(&down_slots, n, h, &stream)
                        .ok_or_else(|| anyhow::anyhow!("glm53 MoE ordered down sum declined"))?;
                let ordered = if let Some(shared) = shared_for_sum {
                    ordered.add(shared)?
                } else {
                    ordered
                };
                let fused_values = routed.to_vec1_wait_through::<f32>()?;
                let ordered_values = ordered.to_vec1_wait_through::<f32>()?;
                anyhow::ensure!(
                    fused_values
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(ordered_values.iter().map(|value| value.to_bits())),
                    "fused DS4 down projection differs from slotwise DS4 accumulation at layer {layer}"
                );
            }
            routed
        } else {
            let down_sorted = {
                let _trace = lisa_mlx::trace::span_detail("glm53.moe.down_slots", layer as u64);
                lisa_mlx::models::glm53::kernels::moe_gather_qmv_addr(
                    &address_table,
                    &act,
                    &selected,
                    keepalive,
                    h,
                    true,
                    self.pools.metadata_dtype,
                    residency_set,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe address: down declined"))?
            }; // [rows, h]
            if let Some(capture) = capture {
                capture.store_ephemeral(&format!("L{layer}.moe_down_slots"), &down_sorted);
            }
            let down = down_sorted.reshape(&[n as i32, k as i32, h as i32])?;
            let mut acc = narrow(&down, 1, 0, 1)?.contiguous()?;
            for slot in 1..k {
                acc = acc.add(&narrow(&down, 1, slot, 1)?.contiguous()?)?;
            }
            acc.reshape(&[n as i32, h as i32])?
        };
        // The eager sum8 dispatch above is now ordered before any later
        // layer can acquire and overwrite these four scratch buffers.
        drop(workspace_lease);
        Ok(routed.reshape(&[b as i32, t as i32, h as i32])?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load<S: TensorSource>(
        w: &mut Weights,
        prefix: &str,
        n_experts: usize,
        top_k: usize,
        n_group: usize,
        topk_group: usize,
        routed_scaling: f32,
        norm_topk: bool,
        n_shared: usize,
        moe_inter: usize,
        _hidden: usize,
        limit: f32,
        lazy_store: Option<Arc<LazyTensorStore>>,
        map_all_expert_weights: bool,
    ) -> anyhow::Result<Self> {
        let take = |w: &mut Weights, name: &str| -> anyhow::Result<Array> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53 moe: {name} missing"))
        };
        let gate_w = take(w, &format!("{prefix}.gate.weight"))?.as_dtype(Dtype::Float32)?;
        let bias =
            take(w, &format!("{prefix}.gate.e_score_correction_bias"))?.as_dtype(Dtype::Float32)?;
        // Resident inference resolves every immutable expert address at load.
        // The selected-expert cache remains a diagnostic reference path.
        let expert_load_started = Instant::now();
        let (experts, expert_cache_bytes) = if let Some(store) = lazy_store.as_ref() {
            ExpertPools::load_experts(
                store,
                prefix,
                n_experts,
                map_all_expert_weights,
                map_all_expert_weights,
            )?
        } else {
            let mut experts = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let ep = format!("{prefix}.experts.{e}");
                let mut gate = QuantizedLinear::load(w, &format!("{ep}.gate_proj"), "")?;
                gate.set_quant(128, 2);
                let mut up = QuantizedLinear::load(w, &format!("{ep}.up_proj"), "")?;
                up.set_quant(128, 2);
                let mut down = QuantizedLinear::load(w, &format!("{ep}.down_proj"), "")?;
                down.set_quant(128, 2);
                experts.push(ExpertParameters { gate, up, down });
            }
            let expert_cache_bytes = experts.iter().map(ExpertParameters::byte_size).collect();
            (experts, expert_cache_bytes)
        };
        if lisa_mlx::trace::enabled_for("glm53-loader") {
            eprintln!(
                "[glm53-loader] loaded {n_experts} routed experts for {prefix} in {:.1}s",
                expert_load_started.elapsed().as_secs_f64()
            );
        }
        let (address_table, metadata_dtype) = ExpertPools::address_table(&experts)?;
        let pools = Arc::new(ExpertPools {
            experts,
            expert_cache_bytes,
            lazy_store: if map_all_expert_weights {
                None
            } else {
                lazy_store.clone()
            },
            expert_prefix: if map_all_expert_weights {
                None
            } else {
                lazy_store.as_ref().map(|_| prefix.to_string())
            },
            inter: moe_inter,
            address_table,
            metadata_dtype,
            residency_set: std::sync::OnceLock::new(),
            gpu_address_all_experts: map_all_expert_weights,
            cache_capacity_bytes: AtomicUsize::new(0),
            cache: Mutex::new(ExpertCacheState::default()),
        });
        let shared = if n_shared > 0 {
            Some(DenseMlp::load_shared::<Weights>(
                w,
                &format!("{prefix}.shared_experts"),
                limit,
            )?)
        } else {
            None
        };
        Ok(Self {
            gate_w,
            bias,
            pools,
            shared,
            grouped_workspace: None,
            top_k,
            n_group,
            topk_group,
            routed_scaling,
            norm_topk,
            limit,
            n_experts,
        })
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        self.forward_with_capture(x, None, 0)
    }

    pub fn forward_with_capture(
        &self,
        x: &Array,
        capture: Option<&Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let (routed, shared) = self.forward_parts_with_capture_mode(x, capture, layer, true)?;
        match shared {
            Some(shared) => Ok(routed.add(&shared)?),
            None => Ok(routed),
        }
    }

    /// Return routed and shared expert outputs independently. DS4's GLM MTP
    /// residual combines both with the attention residual in one ordered add3.
    pub fn forward_parts_with_capture(
        &self,
        x: &Array,
        capture: Option<&Capture>,
        layer: usize,
    ) -> anyhow::Result<(Array, Option<Array>)> {
        self.forward_parts_with_capture_mode(x, capture, layer, false)
    }

    fn forward_parts_with_capture_mode(
        &self,
        x: &Array,
        capture: Option<&Capture>,
        layer: usize,
        fuse_shared_sum8: bool,
    ) -> anyhow::Result<(Array, Option<Array>)> {
        let (b, t, h) = (x.dim(0) as usize, x.dim(1) as usize, x.dim(2) as usize);
        let n = b * t;
        let xf = x
            .as_dtype(Dtype::Float32)?
            .reshape(&[(n) as i32, (h) as i32])?;

        // Router logits in F32. Keep ds4's token-count-specific matvec lane:
        // plain one-row, specialized two-to-eight-row, then plain per-token.
        let logits = {
            let _trace = lisa_mlx::trace::span_detail("glm53.moe.router_projection", layer as u64);
            lisa_mlx::models::glm53::kernels::router_matvec_f32(
                &self.gate_w,
                &xf,
                &lisa_mlx::Stream::thread_local_or_default(),
            )
            .ok_or_else(|| anyhow::anyhow!("glm53 moe: ds4 F32 router matvec declined"))?
        };
        let e = self.n_experts;
        let capture = capture.filter(|capture| capture.wants(layer));
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.moe_router_logits"), &logits);
        }

        // GLM-5.3's ungrouped router and expert dispatch stay on the GPU. A
        // declined selector/address kernel is an error for this model path;
        // silently falling back to per-expert host dispatch hides a severe
        // performance and execution-path mismatch.
        if self.n_group <= 1 && self.topk_group <= 1 {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let (selected, weights) = {
                let _trace = lisa_mlx::trace::span_detail("glm53.moe.router_select", layer as u64);
                lisa_mlx::models::glm53::kernels::router_select_fused(
                    &logits,
                    &self.bias,
                    self.top_k,
                    self.routed_scaling,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("glm53 moe: GPU router selector declined"))?
            };
            if let Some(capture) = capture {
                capture.store(&format!("L{layer}.moe_selected"), &selected);
                capture.store(&format!("L{layer}.moe_weights"), &weights);
            }

            let stage_result = if self.pools.gpu_address_all_experts {
                None
            } else {
                // Commit the selector output before the worker starts its
                // bounded readback. The shared branch can then execute in
                // later command buffers while the worker stages these experts.
                selected.eval_enqueue()?;
                self.pools.stage_selected_async(&selected)
            };

            let fuse_shared_candidate = fuse_shared_sum8
                && capture.is_none_or(|capture| capture.preserves_dispatch())
                && self.top_k == 8;
            let shared = if let Some(sh) = &self.shared {
                // Only the streaming reference needs a submission boundary
                // here. Resident execution follows the tower's DS4 cadence.
                if !self.pools.gpu_address_all_experts {
                    stream.runtime().commands.flush()?;
                }
                let shared = sh.forward_with_capture(x, capture, layer)?;
                if let Some(capture) = capture {
                    capture.store(&format!("L{layer}.moe_shared"), &shared);
                }
                Some(shared)
            } else {
                None
            };

            let fuse_shared = fuse_shared_candidate && shared.is_some();
            let shared_for_sum = if fuse_shared {
                shared
                    .as_ref()
                    .map(|shared| shared.as_dtype(Dtype::Float32))
                    .transpose()?
                    .map(|shared| shared.reshape(&[n as i32, h as i32]))
                    .transpose()?
            } else {
                None
            };
            let prepared_resources =
                if let Some(stage_result) = stage_result {
                    // Match DS4's streaming schedule: submit the shared-expert
                    // work before joining the selected-expert staging worker,
                    // so the GPU can execute it while the host prepares the
                    // routed experts. Without this flush, the join could wait
                    // on cache staging while the shared branch remained only
                    // encoded in the current command buffer.
                    stream.runtime().commands.flush()?;
                    let _trace = lisa_mlx::trace::span("glm53.moe.async_expert_stage_join");
                    Some(stage_result.recv().map_err(|_| {
                        anyhow::anyhow!("GLM async expert staging worker stopped")
                    })??)
                } else {
                    None
                };
            // The combined shared-down/sum8 kernel misses the stock M=1 QMV
            // arithmetic on real layer-45 weights; keep shared down separate
            // until that path can satisfy the strict serial numerics contract.
            let y = self
                .forward_address_gpu(
                    x,
                    &selected,
                    &weights,
                    b,
                    t,
                    h,
                    shared_for_sum.as_ref(),
                    prepared_resources,
                    capture,
                    layer,
                )
                .map_err(|e| anyhow::anyhow!("glm53 moe: GPU address route failed: {e}"))?;
            if let Some(capture) = capture {
                capture.store(&format!("L{layer}.moe_routed"), &y);
            }
            return Ok((y, if fuse_shared { None } else { shared }));
        }

        // Grouped routing is not used by the pinned GLM-5.3 artifact. Keep its
        // explicit reference implementation for configurations that request
        // it, while the production ungrouped route above remains GPU-only.
        let logits_h: Vec<f32> = logits.to_vec1()?;
        let bias_h: Vec<f32> = self.bias.as_dtype(Dtype::Float32)?.to_vec1()?;
        let mut rows = Vec::with_capacity(n);
        for row in 0..n {
            let lg = &logits_h[row * e..(row + 1) * e];
            let sel_sc: Vec<f32> = lg
                .iter()
                .zip(bias_h.iter())
                .map(|(l, b)| 1.0 / (1.0 + (-l).exp()) + b)
                .collect();
            let per_group = e / self.n_group;
            let mut group_sc: Vec<(f32, usize)> = Vec::with_capacity(self.n_group);
            for g in 0..self.n_group {
                let lo = g * per_group;
                let mut top = [0f32; 2];
                for si in &sel_sc[lo..lo + per_group] {
                    if *si > top[0] {
                        top[1] = top[0];
                        top[0] = *si;
                    } else if *si > top[1] {
                        top[1] = *si;
                    }
                }
                group_sc.push((top[0] + top[1], g));
            }
            group_sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut allowed = vec![false; e];
            for (_, g) in group_sc.iter().take(self.topk_group) {
                for i in (g * per_group)..((g + 1) * per_group) {
                    allowed[i] = true;
                }
            }
            let mut candidates: Vec<(f32, usize)> = (0..e)
                .filter(|&i| allowed[i])
                .map(|i| (sel_sc[i], i))
                .collect();
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let chosen: Vec<usize> = candidates
                .iter()
                .take(self.top_k)
                .map(|(_, i)| *i)
                .collect();
            let mut route_weights: Vec<f32> = chosen
                .iter()
                .map(|&i| 1.0 / (1.0 + (-lg[i]).exp()))
                .collect();
            if self.norm_topk {
                let sum: f32 = route_weights.iter().sum();
                if sum > 0.0 {
                    for weight in &mut route_weights {
                        *weight /= sum;
                    }
                }
            }
            for weight in &mut route_weights {
                *weight *= self.routed_scaling;
            }

            let xr = narrow(&xf, 0, row, 1)?;
            let mut acc: Option<Array> = None;
            for (slot, &expert_id) in chosen.iter().enumerate() {
                let gate_up = self.expert_gu(expert_id)?;
                let down = self.expert_down(expert_id)?;
                let fused = gate_up.forward(&xr)?;
                let (gate, up) = split_gate_up(&fused, self.pools.inter)?;
                let route_weight = Array::from_f32_as(route_weights[slot], Dtype::Float32);
                let hidden = clamped_swiglu(&up, &gate, self.limit)?.multiply(&route_weight)?;
                let term = down.forward(&hidden)?.as_dtype(Dtype::Float32)?;
                acc = Some(match acc {
                    Some(sum) => sum.add(&term)?,
                    None => term,
                });
            }
            rows.push(
                acc.ok_or_else(|| {
                    anyhow::anyhow!("glm53 moe: grouped router selected no experts")
                })?,
            );
        }
        let y = lisa_mlx::ops::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?
            .reshape(&[b as i32, t as i32, h as i32])?;
        let shared = self.shared.as_ref().map(|sh| sh.forward(x)).transpose()?;
        Ok((y, shared))
    }

    /// Re-run the GLM router for the MTP DS4 differential without changing
    /// the production MoE dispatch or its fused down-sum path.
    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(crate) fn capture_or_assert_mtp_router_ds4(&self, x: &Array) -> anyhow::Result<()> {
        const HIDDEN: usize = 4096;
        const EXPERTS: usize = 288;
        const TOP_K: usize = 8;
        let external_fixture: &[u8] =
            &crate::models::glm53::test_fixtures::read("mtp_router_layer45_ds4.bin");
        anyhow::ensure!(
            x.dim(x.rank() as i32 - 1) == HIDDEN as i32
                && self.gate_w.shape() == [EXPERTS as i32, HIDDEN as i32]
                && self.bias.size() == EXPERTS
                && self.top_k == TOP_K
                && self.n_group == 1
                && self.topk_group == 1,
            "layer-45 MTP router does not match the DS4 fixture geometry"
        );
        let input = x.as_dtype(Dtype::Float32)?.reshape(&[1, HIDDEN as i32])?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let logits =
            lisa_mlx::models::glm53::kernels::router_matvec_f32(&self.gate_w, &input, &stream)
                .ok_or_else(|| anyhow::anyhow!("GLM MTP router projection declined"))?;
        let (selected, weights) = lisa_mlx::models::glm53::kernels::router_select_fused(
            &logits,
            &self.bias,
            TOP_K,
            self.routed_scaling,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP router selection declined"))?;
        let input_values = input.to_vec1::<f32>()?;
        let router_weight_values = self.gate_w.to_vec1::<f32>()?;
        let router_bias_values = self.bias.to_vec1::<f32>()?;
        let logits_values = logits.to_vec1::<f32>()?;
        let selected_values = selected.to_vec1::<i32>()?;
        let weight_values = weights.to_vec1::<f32>()?;
        anyhow::ensure!(
            input_values.len() == HIDDEN
                && router_weight_values.len() == EXPERTS * HIDDEN
                && router_bias_values.len() == EXPERTS
                && logits_values.len() == EXPERTS
                && selected_values.len() == TOP_K
                && weight_values.len() == TOP_K,
            "layer-45 MTP router output shape mismatch"
        );

        if let Some(path) = std::env::var_os("LISA_GLM53_MTP_ROUTER_CAPTURE_DIR") {
            let path = std::path::Path::new(&path);
            std::fs::create_dir_all(path)?;
            for (name, data) in [
                ("input.f32", encode_f32(&input_values)),
                ("router_weight.f32", encode_f32(&router_weight_values)),
                ("router_bias.f32", encode_f32(&router_bias_values)),
                ("lisa_logits.f32", encode_f32(&logits_values)),
                ("lisa_weights.f32", encode_f32(&weight_values)),
            ] {
                std::fs::write(path.join(name), data)?;
            }
            std::fs::write(path.join("lisa_selected.i32"), encode_i32(&selected_values))?;
            return Ok(());
        }

        const F32_BYTES: usize = (HIDDEN + EXPERTS * HIDDEN + 2 * EXPERTS + TOP_K) * 4;
        const SELECTED_BYTES: usize = TOP_K * 4;
        anyhow::ensure!(
            external_fixture.len() == F32_BYTES + SELECTED_BYTES,
            "DS4 layer-45 MTP router fixture has an invalid size"
        );
        let mut offset = 0usize;
        let mut expected_f32 = Vec::with_capacity(5);
        for count in [HIDDEN, EXPERTS * HIDDEN, EXPERTS, EXPERTS, TOP_K] {
            let end = offset + count * 4;
            expected_f32.push(
                external_fixture[offset..end]
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 word")))
                    .collect::<Vec<_>>(),
            );
            offset = end;
        }
        let expected_selected = external_fixture[offset..]
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("I32 id")))
            .collect::<Vec<_>>();
        for (name, actual, expected) in [
            ("input", &input_values, &expected_f32[0]),
            ("router weight", &router_weight_values, &expected_f32[1]),
            ("router bias", &router_bias_values, &expected_f32[2]),
        ] {
            anyhow::ensure!(
                actual
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(expected.iter().map(|v| v.to_bits())),
                "layer-45 MTP DS4 {name} fixture input changed"
            );
        }
        anyhow::ensure!(
            selected_values == expected_selected,
            "layer-45 MTP router expert IDs differ from DS4"
        );
        let logits_max_abs = logits_values
            .iter()
            .zip(&expected_f32[3])
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let weight_max_abs = weight_values
            .iter()
            .zip(&expected_f32[4])
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(
            logits_max_abs <= 2e-5 && weight_max_abs <= 1e-6,
            "layer-45 MTP router differs from DS4: logits max_abs={logits_max_abs:e}, weights max_abs={weight_max_abs:e}"
        );
        eprintln!(
            "[glm53-mtp-router] selected IDs match DS4; logits max_abs={logits_max_abs:.3e}; weights max_abs={weight_max_abs:.3e}"
        );
        Ok(())
    }

    /// Compare the MTP shared expert against DS4 using the checkpoint's
    /// MLX-affine weights dequantized by the direct Metal reference harness.
    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(crate) fn capture_or_assert_mtp_shared_ds4(
        &self,
        x: &Array,
        actual: &Array,
    ) -> anyhow::Result<()> {
        const HIDDEN: usize = 4096;
        let external_fixture: &[u8] =
            &crate::models::glm53::test_fixtures::read("mtp_shared_layer45_ds4.bin");
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GLM MTP shared expert is missing"))?;
        let inter = shared.gate_up.inter();
        let (gate, up) = shared.gate_up.split_linears();
        let hidden = HIDDEN;
        let group = usize::try_from(gate.group_size)?;
        let bits = usize::try_from(gate.bits)?;
        anyhow::ensure!(
            hidden == x.dim(x.rank() as i32 - 1) as usize
                && inter > 0
                && group == 64
                && bits == 8
                && up.dims_out() == inter
                && shared.down.dims_out() == hidden
                && gate.group_size == up.group_size
                && gate.group_size == shared.down.group_size
                && gate.bits == up.bits
                && gate.bits == shared.down.bits,
            "layer-45 MTP shared expert geometry differs from the DS4 fixture"
        );

        let input = x.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        let output = actual.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        let gate_weight = gate.weight.to_vec1::<u32>()?;
        let gate_scales = gate.scales.to_vec1::<f32>()?;
        let gate_biases = gate.biases.to_vec1::<f32>()?;
        let up_weight = up.weight.to_vec1::<u32>()?;
        let up_scales = up.scales.to_vec1::<f32>()?;
        let up_biases = up.biases.to_vec1::<f32>()?;
        let down_weight = shared.down.weight.to_vec1::<u32>()?;
        let down_scales = shared.down.scales.to_vec1::<f32>()?;
        let down_biases = shared.down.biases.to_vec1::<f32>()?;
        anyhow::ensure!(
            input.len() == hidden
                && output.len() == hidden
                && gate_weight.len() == inter * hidden * bits / 32
                && up_weight.len() == gate_weight.len()
                && down_weight.len() == hidden * inter * bits / 32
                && gate_scales.len() == inter * hidden / group
                && gate_biases.len() == gate_scales.len()
                && up_scales.len() == gate_scales.len()
                && up_biases.len() == gate_scales.len()
                && down_scales.len() == hidden * inter / group
                && down_biases.len() == down_scales.len(),
            "layer-45 MTP shared expert tensor sizes do not match its geometry"
        );

        let mut prefix = Vec::new();
        for value in [hidden as u32, inter as u32, group as u32, bits as u32] {
            prefix.extend_from_slice(&value.to_le_bytes());
        }
        prefix.extend_from_slice(&encode_f32(&input));
        prefix.extend_from_slice(&encode_u32(&gate_weight));
        prefix.extend_from_slice(&encode_f32(&gate_scales));
        prefix.extend_from_slice(&encode_f32(&gate_biases));
        prefix.extend_from_slice(&encode_u32(&up_weight));
        prefix.extend_from_slice(&encode_f32(&up_scales));
        prefix.extend_from_slice(&encode_f32(&up_biases));
        prefix.extend_from_slice(&encode_u32(&down_weight));
        prefix.extend_from_slice(&encode_f32(&down_scales));
        prefix.extend_from_slice(&encode_f32(&down_biases));

        if let Some(path) = std::env::var_os("LISA_GLM53_MTP_SHARED_CAPTURE_DIR") {
            let path = std::path::Path::new(&path);
            std::fs::create_dir_all(path)?;
            std::fs::write(
                path.join("geometry.u32"),
                [hidden as u32, inter as u32, group as u32, bits as u32]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )?;
            std::fs::write(path.join("input.f32"), encode_f32(&input))?;
            std::fs::write(path.join("gate_weight.u32"), encode_u32(&gate_weight))?;
            std::fs::write(path.join("gate_scales.f32"), encode_f32(&gate_scales))?;
            std::fs::write(path.join("gate_biases.f32"), encode_f32(&gate_biases))?;
            std::fs::write(path.join("up_weight.u32"), encode_u32(&up_weight))?;
            std::fs::write(path.join("up_scales.f32"), encode_f32(&up_scales))?;
            std::fs::write(path.join("up_biases.f32"), encode_f32(&up_biases))?;
            std::fs::write(path.join("down_weight.u32"), encode_u32(&down_weight))?;
            std::fs::write(path.join("down_scales.f32"), encode_f32(&down_scales))?;
            std::fs::write(path.join("down_biases.f32"), encode_f32(&down_biases))?;
            std::fs::write(path.join("lisa_output.f32"), encode_f32(&output))?;
            return Ok(());
        }

        anyhow::ensure!(
            external_fixture.len() == prefix.len() + hidden * 4
                && external_fixture.starts_with(&prefix),
            "layer-45 MTP shared expert DS4 fixture inputs changed"
        );
        let expected = external_fixture[prefix.len()..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 output word")))
            .collect::<Vec<_>>();
        let expected_norm2 = expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();

        // Exercise the shared Q8 down projection inside the DS4-default sum8
        // kernel against the direct DS4 branch fixture. Zero routed activations
        // isolate this branch while retaining the production address-table and
        // dispatch path.
        let gate_output = gate.forward(x)?;
        let up_output = up.forward(x)?;
        let shared_activation =
            clamped_swiglu(&up_output, &gate_output, shared.limit)?.reshape(&[1, inter as i32])?;
        let selected_ids = [0i32; 8];
        let selected = Array::from_slice(&selected_ids, &[selected_ids.len() as i32]);
        let routed_activation = Array::from_slice(
            &vec![0.0f32; selected_ids.len() * inter],
            &[selected_ids.len() as i32, inter as i32],
        );
        let selected_resources = if self.pools.residency_set.get().is_some() {
            Vec::new()
        } else {
            ExpertPools::indirect_resources(&self.pools.experts, &selected_ids)?
        };
        let keepalive: Vec<&Array> = selected_resources.iter().collect();
        let fused_shared =
            lisa_mlx::models::glm53::kernels::moe_gather_qmv_sum8_addr_add_shared_down(
                &self.pools.address_table,
                &routed_activation,
                &selected,
                &keepalive,
                &shared_activation,
                &shared.down.weight,
                &shared.down.scales,
                &shared.down.biases,
                hidden,
                self.pools.metadata_dtype,
                self.pools.residency_set.get().map(Arc::as_ref),
                &lisa_mlx::Stream::thread_local_or_default(),
            )
            .ok_or_else(|| anyhow::anyhow!("shared Q8 down/sum8 DS4 fixture kernel declined"))?
            .to_vec1::<f32>()?;
        let fused_stock_mismatches = fused_shared
            .iter()
            .zip(&output)
            .filter(|(fused, stock)| fused.to_bits() != stock.to_bits())
            .count();
        let fused_stock_max_abs = fused_shared
            .iter()
            .zip(&output)
            .map(|(fused, stock)| (fused - stock).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[glm53-mtp-shared-down-exact] mismatches={fused_stock_mismatches}/{} max_abs={fused_stock_max_abs:e}",
            output.len()
        );
        let fused_max_abs = fused_shared
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let fused_delta2 = fused_shared
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let fused_rel_l2 = (fused_delta2 / expected_norm2.max(f64::MIN_POSITIVE)).sqrt();
        anyhow::ensure!(
            fused_max_abs <= 1e-4 && fused_rel_l2 <= 1e-5,
            "fused layer-45 MTP shared expert differs from DS4: max_abs={fused_max_abs:e}, rel_l2={fused_rel_l2:e}"
        );
        eprintln!(
            "[glm53-mtp-shared-fused-down] output max_abs={fused_max_abs:.3e} rel_l2={fused_rel_l2:.3e} against DS4"
        );
        let mut delta2 = 0.0f64;
        let max_abs = output
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        for (actual, expected) in output.iter().zip(&expected) {
            delta2 += f64::from(actual - expected).powi(2);
        }
        let rel_l2 = (delta2 / expected_norm2.max(f64::MIN_POSITIVE)).sqrt();
        anyhow::ensure!(
            max_abs <= 1e-4 && rel_l2 <= 1e-5,
            "layer-45 MTP shared expert differs from DS4: max_abs={max_abs:e}, rel_l2={rel_l2:e}"
        );
        eprintln!(
            "[glm53-mtp-shared] output max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} against DS4"
        );
        Ok(())
    }

    /// Capture only the eight selected MTP experts, then compare the routed
    /// output with DS4's F32 Metal graph replay.
    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(crate) fn capture_or_assert_mtp_routed_ds4(
        &self,
        x: &Array,
        actual: &Array,
    ) -> anyhow::Result<()> {
        const HIDDEN: usize = 4096;
        const INTER: usize = 2048;
        const TOP_K: usize = 8;
        let external_fixture: &[u8] =
            &crate::models::glm53::test_fixtures::read("mtp_routed_layer45_ds4.bin");
        anyhow::ensure!(
            x.dim(x.rank() as i32 - 1) == HIDDEN as i32
                && self.pools.inter == INTER
                && self.top_k == TOP_K
                && self.n_group == 1
                && self.topk_group == 1,
            "layer-45 MTP routed expert geometry differs from DS4 fixture"
        );
        let input = x.as_dtype(Dtype::Float32)?.reshape(&[1, HIDDEN as i32])?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let logits =
            lisa_mlx::models::glm53::kernels::router_matvec_f32(&self.gate_w, &input, &stream)
                .ok_or_else(|| {
                    anyhow::anyhow!("GLM MTP routed fixture router projection declined")
                })?;
        let (selected, route_weights) = lisa_mlx::models::glm53::kernels::router_select_fused(
            &logits,
            &self.bias,
            TOP_K,
            self.routed_scaling,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP routed fixture router selector declined"))?;
        let input_values = input.to_vec1::<f32>()?;
        let selected_values = selected.to_vec1::<i32>()?;
        let route_values = route_weights.to_vec1::<f32>()?;
        let output_values = actual.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        anyhow::ensure!(
            input_values.len() == HIDDEN
                && selected_values.len() == TOP_K
                && route_values.len() == TOP_K
                && output_values.len() == HIDDEN,
            "layer-45 MTP routed expert fixture shape mismatch"
        );

        if let Some(path) = std::env::var_os("LISA_GLM53_MTP_ROUTED_CAPTURE_DIR") {
            let path = std::path::Path::new(&path);
            std::fs::create_dir_all(path)?;
            std::fs::write(path.join("input.f32"), encode_f32(&input_values))?;
            std::fs::write(path.join("selected.i32"), encode_i32(&selected_values))?;
            std::fs::write(path.join("route_weights.f32"), encode_f32(&route_values))?;
            std::fs::write(path.join("lisa_output.f32"), encode_f32(&output_values))?;
            for (slot, expert_id) in selected_values.iter().enumerate() {
                let expert = self.pools.experts.get(*expert_id as usize).ok_or_else(|| {
                    anyhow::anyhow!("selected MTP expert id {expert_id} is invalid")
                })?;
                for (name, packed, scales, biases) in [
                    (
                        "gate",
                        &expert.gate.weight,
                        &expert.gate.scales,
                        &expert.gate.biases,
                    ),
                    (
                        "up",
                        &expert.up.weight,
                        &expert.up.scales,
                        &expert.up.biases,
                    ),
                    (
                        "down",
                        &expert.down.weight,
                        &expert.down.scales,
                        &expert.down.biases,
                    ),
                ] {
                    let packed = match packed.dtype() {
                        Dtype::Uint8 => packed.to_vec1::<u8>()?,
                        Dtype::Uint32 => encode_u32(&packed.to_vec1::<u32>()?),
                        dtype => anyhow::bail!(
                            "MTP expert {slot} {name} uses unsupported packed dtype {dtype:?}"
                        ),
                    };
                    let scales =
                        scales
                            .as_dtype(Dtype::Float32)?
                            .to_vec1::<f32>()
                            .map_err(|error| {
                                anyhow::anyhow!("read MTP expert {slot} {name} scales: {error}")
                            })?;
                    let biases =
                        biases
                            .as_dtype(Dtype::Float32)?
                            .to_vec1::<f32>()
                            .map_err(|error| {
                                anyhow::anyhow!("read MTP expert {slot} {name} biases: {error}")
                            })?;
                    let stem = format!("expert_{slot:02}_{name}");
                    std::fs::write(path.join(format!("{stem}_weight.u8")), packed)?;
                    std::fs::write(path.join(format!("{stem}_scales.f32")), encode_f32(&scales))?;
                    std::fs::write(path.join(format!("{stem}_biases.f32")), encode_f32(&biases))?;
                }
            }
            return Ok(());
        }

        const SELECTED_BYTES: usize = TOP_K * 4;
        anyhow::ensure!(
            external_fixture.len() == SELECTED_BYTES + HIDDEN * 8,
            "DS4 layer-45 MTP routed fixture has an invalid size"
        );
        let expected_selected = external_fixture[..SELECTED_BYTES]
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("I32 expert id")))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            selected_values == expected_selected,
            "layer-45 MTP routed expert ids differ from DS4 fixture"
        );
        let input_end = SELECTED_BYTES + HIDDEN * 4;
        let expected_input = external_fixture[SELECTED_BYTES..input_end]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 input word")))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            input_values
                .iter()
                .zip(&expected_input)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "layer-45 MTP routed fixture input changed"
        );
        let expected_output = external_fixture[input_end..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 output word")))
            .collect::<Vec<_>>();
        let max_abs = output_values
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let delta2 = output_values
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let ref2 = expected_output
            .iter()
            .map(|expected| f64::from(*expected).powi(2))
            .sum::<f64>();
        let rel_l2 = (delta2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
        anyhow::ensure!(
            max_abs <= 1e-4 && rel_l2 <= 1e-5,
            "layer-45 MTP routed output differs from DS4: max_abs={max_abs:e}, rel_l2={rel_l2:e}"
        );
        eprintln!(
            "[glm53-mtp-routed] selected IDs match DS4; output max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}"
        );
        Ok(())
    }
}

#[cfg(test)]
fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn encode_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn encode_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "moe_grouped_workspace_tests.rs"]
mod grouped_workspace_tests;

#[cfg(test)]
#[path = "moe_grouped_workspace_bounds_tests.rs"]
mod grouped_workspace_bounds_tests;
