//! GLM affine expert storage, cache binding and capture I/O. Numerical
//! plans and router arithmetic are owned by lisa-models::glm53::routed_moe.
//!
//! Router in F32 (sigmoid, no
//! aux loss): selection scores add `e_score_correction_bias`, group by
//! `n_group=8` (top-2 within group summed for the group score), pick
//! `topk_group=4` groups, then top-`num_experts_per_tok=8` experts by the
//! SELECTION score; the WEIGHTS are the raw sigmoid scores renormalized
//! (`norm_topk_prob`) and scaled by `routed_scaling_factor=2.5`.
//! Activation: clamped SwiGLU (`swiglu_limit=10`): gate clipped ABOVE only,
//! up clipped both sides — for routed, shared AND dense experts.

use crate::bindings::shared::tensor_io::CaptureTensor;
pub trait ExpertDevice: BindingDevice + ResourceDevice {}
impl<D: BindingDevice + ResourceDevice> ExpertDevice for D {}

use crate::bindings::glm53::affine::moe::ops::run as moe_run;

use crate::bindings::shared::capture::Capture;
use crate::weights::affine::QuantizedLinear;
use crate::weights::source::{TensorBank, TensorSource, Weights};
use crate::{
    bindings::BindingDevice,
    execution::{DeviceContext, Observer},
};
use lisa_backend::{ResourceDevice, ScalarType as Dtype, TensorAccess};
use lisa_models::glm53::routed_moe::{GroupedRouter, Request as MoeRequest};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

pub use crate::bindings::glm53::policy::{Glm53Q8GateUpNativeScope, native_gate_up_enabled};

pub use crate::bindings::glm53::affine::dense_mlp::{DenseMlp, fuse_gate_up};

#[derive(Clone)]
pub struct ExpertParameters<D: ExpertDevice> {
    pub gate: QuantizedLinear<D::Tensor>,
    pub up: QuantizedLinear<D::Tensor>,
    pub down: QuantizedLinear<D::Tensor>,
}

impl<D: ExpertDevice> ExpertParameters<D> {
    pub fn byte_size(&self, device: &D) -> usize {
        ExpertPools::<D>::arrays(self)
            .iter()
            .map(|array| {
                array.size().saturating_mul(
                    device
                        .dtype(array)
                        .expect("expert scalar storage")
                        .size_of(),
                )
            })
            .sum()
    }

    pub fn copied(&self) -> anyhow::Result<Self> {
        let copy_linear =
            |linear: &QuantizedLinear<D::Tensor>| -> anyhow::Result<QuantizedLinear<D::Tensor>> {
                let copy =
                    |array: &D::Tensor| -> anyhow::Result<D::Tensor> { array.independent_copy() };
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

pub struct CachedExpert<D: ExpertDevice> {
    pub parameters: ExpertParameters<D>,
    pub bytes: usize,
    pub last_used: u64,
}

pub struct ExpertCacheState<D: ExpertDevice> {
    pub entries: HashMap<usize, CachedExpert<D>>,
    pub bytes: usize,
    pub clock: u64,
    pub address_table: Option<D::Table>,
}

impl<D: ExpertDevice> Default for ExpertCacheState<D> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
            clock: 0,
            address_table: None,
        }
    }
}

pub enum ExpertStageMessage<D: ExpertDevice> {
    Stage {
        pools: Weak<ExpertPools<D>>,
        selected: D::Tensor,
        result: SyncSender<anyhow::Result<(D::Table, Vec<D::Tensor>)>>,
    },
}

pub struct ExpertPools<D: ExpertDevice> {
    pub device: D,
    pub observer: Observer,
    pub worker: Arc<std::sync::OnceLock<Option<mpsc::Sender<ExpertStageMessage<D>>>>>,
    pub experts: Vec<ExpertParameters<D>>,
    pub expert_cache_bytes: Vec<usize>,
    pub lazy_store: Option<Arc<dyn TensorBank<D::Tensor>>>,
    pub expert_prefix: Option<String>,
    pub inter: usize,
    pub address_table: D::Table,
    pub metadata_dtype: Dtype,
    pub residency_set: std::sync::OnceLock<D::Residency>,
    pub gpu_address_all_experts: bool,
    pub resources: std::sync::OnceLock<D::Resources>,
    pub cache_capacity_bytes: AtomicUsize,
    pub cache: Mutex<ExpertCacheState<D>>,
}

fn selected_prefetch_bytes(expert_ids: &[usize], expert_cache_bytes: &[usize]) -> Option<usize> {
    expert_ids.iter().try_fold(0usize, |total, &expert_id| {
        total.checked_add(*expert_cache_bytes.get(expert_id)?)
    })
}

impl<D: ExpertDevice> ExpertPools<D> {
    pub fn trace(&self, name: &'static str) -> Option<Box<dyn std::any::Any>> {
        (self.observer.span)(name, None)
    }
    pub fn trace_enabled(&self, name: &str) -> bool {
        (self.observer.enabled)(name)
    }

    pub fn arrays(expert: &ExpertParameters<D>) -> [&D::Tensor; 9] {
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

    pub fn indirect_resources(
        device: &D,
        experts: &[ExpertParameters<D>],
        selected_ids: &[i32],
    ) -> anyhow::Result<Vec<D::Tensor>> {
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
                if seen.insert(device.allocation_id(array)) {
                    resources.push(array.clone());
                }
            }
        }
        Ok(resources)
    }

    pub fn load_lazy_expert_weights(
        &self,
        expert_id: usize,
    ) -> anyhow::Result<ExpertParameters<D>> {
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
                let _trace = self.trace("glm53.moe.expert_weight_mapping");
                store.get_mapped_weight_bytes_individual(&weight_name)?
            };
            let scales_name = format!("{prefix}.experts.{expert_id}.{projection}.scales");
            linear.scales = {
                let _trace = self.trace("glm53.moe.expert_metadata_mapping");
                store.get_mapped_individual(&scales_name)?
            };
            let biases_name = format!("{prefix}.experts.{expert_id}.{projection}.biases");
            linear.biases = {
                let _trace = self.trace("glm53.moe.expert_metadata_mapping");
                store.get_mapped_individual(&biases_name)?
            };
        }
        anyhow::ensure!(
            parameters.byte_size(&self.device) == self.expert_cache_bytes[expert_id],
            "glm53 expert {expert_id} loaded payload size differs from safetensors metadata"
        );
        Ok(parameters)
    }

    pub fn load_lazy_experts(
        &self,
        expert_ids: &[usize],
    ) -> anyhow::Result<Vec<(usize, ExpertParameters<D>)>> {
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
        std::thread::scope(
            |scope| -> anyhow::Result<Vec<(usize, ExpertParameters<D>)>> {
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
                        handle.join().map_err(|_| {
                            anyhow::anyhow!("glm53 expert staging thread panicked")
                        })??,
                    );
                }
                Ok(loaded)
            },
        )
    }

    pub fn prefetch_selected_experts(&self, expert_ids: &[usize]) -> anyhow::Result<()> {
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
            if self.trace_enabled("glm53.moe.expert_prefetch") {
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
            let _trace = self.trace("glm53.moe.expert_prefetch");
            store.prefetch_mapped_tensors(&names)?
        };
        if self.trace_enabled("glm53.moe.expert_prefetch") {
            eprintln!(
                "[glm53-moe] prefetched {} selected tensor regions ({} bytes) for {} experts",
                ranges,
                bytes,
                expert_ids.len()
            );
        }
        Ok(())
    }

    pub fn transient_selected(
        &self,
        requested: &[usize],
    ) -> anyhow::Result<(D::Table, Vec<(usize, ExpertParameters<D>)>)> {
        self.prefetch_selected_experts(requested)?;
        let mut experts = self.experts.clone();
        let active = {
            let _trace = self.trace("glm53.moe.transient_expert_map");
            self.load_lazy_experts(requested)?
        };
        for (expert_id, parameters) in &active {
            experts[*expert_id] = parameters.clone();
        }
        Ok((Self::address_table(&self.device, &experts)?.0, active))
    }

    pub fn stage_selected(
        &self,
        selected_ids: &[i32],
    ) -> anyhow::Result<Option<(D::Table, Vec<(usize, ExpertParameters<D>)>)>> {
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
        let completed = self.device.completed_watermark();
        let mut residency_changed = false;
        let mut victims = cache
            .entries
            .iter()
            .filter(|(expert_id, entry)| {
                !requested_set.contains(expert_id)
                    && ExpertPools::<D>::arrays(&entry.parameters)
                        .iter()
                        .all(|array| self.device.reusable_after(array, completed))
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
                if let Some(set) = self.residency_set.get() {
                    for array in ExpertPools::<D>::arrays(&entry.parameters) {
                        residency_changed |= self.device.residency_remove(set, array);
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
                parameters.byte_size(&self.device) == bytes,
                "glm53 expert cache entry changed expert byte size"
            );
            if let Some(set) = self.residency_set.get() {
                for array in ExpertPools::<D>::arrays(&parameters) {
                    residency_changed |= self.device.residency_add(set, array);
                }
            }
            cache.bytes = cache.bytes.saturating_add(bytes);
            cache.entries.insert(
                expert_id,
                CachedExpert::<D> {
                    parameters,
                    bytes,
                    last_used: clock,
                },
            );
            address_table_changed = true;
        }

        if residency_changed {
            let set = self.residency_set.get().expect("residency set changed");
            {
                let _trace = self.trace("glm53.moe.residency_commit");
                self.device.residency_commit(set);
            }
            {
                let _trace = self.trace("glm53.moe.residency_request");
                self.device.residency_request(set);
            }
        }

        if address_table_changed || cache.address_table.is_none() {
            let _trace = self.trace("glm53.moe.address_table_rebuild");
            let mut experts = self.experts.clone();
            for (&expert_id, entry) in &cache.entries {
                experts[expert_id] = entry.parameters.clone();
            }
            cache.address_table = Some(Self::address_table(&self.device, &experts)?.0);
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

    pub fn dispatch_resources(
        &self,
        selected_ids: &[i32],
    ) -> anyhow::Result<(D::Table, Vec<D::Tensor>)> {
        let staged = {
            let _trace = self.trace("glm53.moe.expert_cache_stage");
            self.stage_selected(selected_ids)?
        };
        let _resource_trace = self.trace("glm53.moe.resource_list_build");
        let Some((address_table, staged)) = staged else {
            let resources = Self::indirect_resources(&self.device, &self.experts, selected_ids)?;
            return Ok((self.address_table.clone(), resources));
        };
        let mut seen = HashSet::new();
        let mut resources = Vec::new();
        for (_, expert) in staged {
            for array in Self::arrays(&expert) {
                if seen.insert(self.device.allocation_id(array)) {
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
    pub fn stage_selected_async(
        self: &Arc<Self>,
        selected: &D::Tensor,
    ) -> Option<Receiver<anyhow::Result<(D::Table, Vec<D::Tensor>)>>> {
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let sender = self
            .worker
            .get_or_init(|| {
                let (sender, receiver) = mpsc::channel::<ExpertStageMessage<D>>();
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
                                        let _trace =
                                            pools.trace("glm53.moe.async_selected_id_readback");
                                        pools.device.read_i32_through(&selected)?
                                    };
                                    let _trace =
                                        pools.trace("glm53.moe.async_expert_dispatch_resources");
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

    pub fn load_experts(
        context: &DeviceContext<D>,
        store: &dyn TensorBank<D::Tensor>,
        prefix: &str,
        n_experts: usize,
        map_all_weights: bool,
        share_metadata_shards: bool,
    ) -> anyhow::Result<(Vec<ExpertParameters<D>>, Vec<usize>)> {
        anyhow::ensure!(n_experts > 0, "glm53 moe: no routed experts");
        let metadata_dtype = store.tensor_dtype(&format!("{prefix}.experts.0.gate_proj.scales"))?;
        anyhow::ensure!(
            matches!(metadata_dtype, Dtype::BF16 | Dtype::F32),
            "glm53 expert metadata dtype {metadata_dtype:?} is unsupported by address kernels"
        );
        // Full-tower diagnostics map metadata through its source shard so it
        // shares backing storage with the model. The isolated mapped-kernel
        // test packs metadata to keep its small resource table bounded. The
        // production path retains typed placeholders and loads selected data
        // through the bounded cache.
        let mut metadata_views = HashMap::<(String, String), Vec<D::Tensor>>::new();
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
                                context.device().dtype(&view) == Some(metadata_dtype)
                                    && !view.shape().is_empty(),
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
                            context.device().dtype(&packed) == Some(metadata_dtype)
                                && !packed.shape().is_empty(),
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
                                context
                                    .device()
                                    .byte_view(
                                        &packed,
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
        let placeholder_weight = context.device().pull(
            lisa_backend::HostTensor {
                bytes: &[0u8],
                shape: &[1],
                dtype: Dtype::U8,
            }
            .into(),
            Default::default(),
        )?;
        let placeholder_metadata =
            crate::bindings::shared::tensor_io::pull_f32(context, &[0.0f32], &[1])?
                .cast_on(context, metadata_dtype)?;
        let load_linear = |expert: usize,
                           projection: &str|
         -> anyhow::Result<QuantizedLinear<D::Tensor>> {
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
                Ok(ExpertParameters::<D> {
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
        if context.trace_enabled("glm53-loader") {
            eprintln!(
                "[glm53-loader] metadata phases for {prefix}: names={:.1}ms views={:.1}ms expert_sizes={:.1}ms",
                metadata_name_build.as_secs_f64() * 1000.0,
                metadata_views_build.as_secs_f64() * 1000.0,
                expert_bytes_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok((experts, expert_bytes))
    }

    pub fn address_table(
        device: &D,
        experts: &[ExpertParameters<D>],
    ) -> anyhow::Result<(D::Table, Dtype)> {
        anyhow::ensure!(!experts.is_empty(), "glm53 moe: no routed experts");
        let mut slots = Vec::with_capacity(experts.len() * 9);
        let mut metadata_dtype = None;
        for (expert_id, expert) in experts.iter().enumerate() {
            for (projection, linear) in [
                ("gate", &expert.gate),
                ("up", &expert.up),
                ("down", &expert.down),
            ] {
                anyhow::ensure!(
                    device.dtype(&linear.scales).expect("expert scales storage")
                        == device.dtype(&linear.biases).expect("expert bias storage"),
                    "glm53 expert {expert_id} {projection} scale/bias dtype mismatch"
                );
                match metadata_dtype {
                    Some(dtype) => anyhow::ensure!(
                        dtype == device.dtype(&linear.scales).expect("expert scales storage"),
                        "glm53 expert metadata dtype changed at expert {expert_id}"
                    ),
                    None => {
                        metadata_dtype =
                            Some(device.dtype(&linear.scales).expect("expert scales storage"))
                    }
                }
                for array in [&linear.weight, &linear.scales, &linear.biases] {
                    slots.push(array);
                }
            }
        }
        let dtype = metadata_dtype.expect("nonempty expert set has metadata dtype");
        anyhow::ensure!(
            matches!(dtype, Dtype::BF16 | Dtype::F32),
            "glm53 routed metadata dtype {dtype:?} is unsupported by address kernels"
        );
        let table = ResourceDevice::indirect_table(device, &slots)?;
        Ok((table, dtype))
    }
}

/// The four large expert intermediates are shared by every GLM layer.
/// Serial Metal dispatch and command-buffer queue order make the next write
/// follow the previous sum8 read. The mutex holds that producer/consumer
/// sequence together during encoding, including concurrent host callers.
/// Captures bypass this workspace because they retain intermediate arrays.
pub struct GroupedMoeWorkspace<D: ExpertDevice> {
    pub buffers: Mutex<GroupedMoeBuffers<D>>,
}
impl<D: ExpertDevice> Default for GroupedMoeWorkspace<D> {
    fn default() -> Self {
        Self {
            buffers: Mutex::new(GroupedMoeBuffers::default()),
        }
    }
}
pub struct GroupedMoeBuffers<D: ExpertDevice> {
    pub gate: Option<D::Workspace>,
    pub up: Option<D::Workspace>,
    pub act: Option<D::Workspace>,
    pub down: Option<D::Workspace>,
}
impl<D: ExpertDevice> Default for GroupedMoeBuffers<D> {
    fn default() -> Self {
        Self {
            gate: None,
            up: None,
            act: None,
            down: None,
        }
    }
}
pub struct GroupedMoeViews<D: ExpertDevice> {
    pub gate: Option<D::Tensor>,
    pub up: Option<D::Tensor>,
    pub act: D::Tensor,
    pub down: D::Tensor,
}
impl<D: ExpertDevice> GroupedMoeBuffers<D> {
    fn output(
        slot: &mut Option<D::Workspace>,
        device: &D,
        rows: usize,
        width: usize,
        dtype: Dtype,
        label: &str,
    ) -> anyhow::Result<D::Tensor> {
        let shape = [i32::try_from(rows)?, i32::try_from(width)?];
        // SAFETY: The workspace mutex covers the entire producer/consumer sequence.
        // Each active element is fully overwritten before use on the same queue.
        unsafe { device.full_write_workspace(slot, &shape, dtype, label) }
    }
    pub fn prepare(
        &mut self,
        device: &D,
        rows: usize,
        intermediate: usize,
        hidden: usize,
        split_pair: bool,
    ) -> anyhow::Result<GroupedMoeViews<D>> {
        let (gate, up) = if split_pair {
            (
                Some(Self::output(
                    &mut self.gate,
                    device,
                    rows,
                    intermediate,
                    Dtype::F32,
                    "glm53.workspace.gate",
                )?),
                Some(Self::output(
                    &mut self.up,
                    device,
                    rows,
                    intermediate,
                    Dtype::F32,
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
                Dtype::F16,
                "glm53.workspace.act",
            )?,
            down: Self::output(
                &mut self.down,
                device,
                rows,
                hidden,
                Dtype::F32,
                "glm53.workspace.down",
            )?,
        })
    }
}

/// Sparse `MoE<D>` (layers >= 3 and the MTP block).
pub struct MoE<D: ExpertDevice> {
    pub context: DeviceContext<D>,
    pub gate_w: D::Tensor, // router projection in F32
    pub bias: D::Tensor,   // router correction bias in F32
    pub pools: Arc<ExpertPools<D>>,
    pub shared: Option<DenseMlp<D>>,
    pub grouped_workspace: Option<Arc<GroupedMoeWorkspace<D>>>,
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling: f32,
    pub norm_topk: bool,
    pub limit: f32,
    pub n_experts: usize,
}

impl<D: ExpertDevice> MoE<D> {
    pub fn shared_paired_storage(&self) -> Option<bool> {
        self.shared.as_ref().map(DenseMlp::paired_storage)
    }

    pub fn set_grouped_workspace(&mut self, workspace: Arc<GroupedMoeWorkspace<D>>) {
        self.grouped_workspace = Some(workspace);
    }

    pub fn expert_residency_capacity(&self) -> usize {
        if self.pools.lazy_store.is_some() {
            // Lazy pools contain placeholders until selected experts enter
            // the LRU; they cannot be registered as model-wide resident data.
            return 0;
        }
        self.pools
            .experts
            .iter()
            .flat_map(ExpertPools::<D>::arrays)
            .map(|a| self.context.device().allocation_id(a))
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    pub fn add_expert_resources_to_residency_set(&self, set: &D::Residency) -> bool {
        if self.pools.lazy_store.is_some() {
            return false;
        }
        self.pools
            .experts
            .iter()
            .flat_map(ExpertPools::<D>::arrays)
            .all(|array| self.context.device().residency_add(set, array))
    }

    pub fn set_model_residency_set(&mut self, set: D::Residency) {
        let _ = self.pools.residency_set.set(set);
    }

    /// One expert's fused gate_up (host fallback path).
    pub fn expert_gu(&self, ei: usize) -> anyhow::Result<QuantizedLinear<D::Tensor>> {
        let expert = self
            .pools
            .experts
            .get(ei)
            .ok_or_else(|| anyhow::anyhow!("glm53 moe: expert id {ei} out of range"))?;
        fuse_gate_up(&self.context, &expert.gate, &expert.up)
    }

    pub fn expert_down(&self, ei: usize) -> anyhow::Result<QuantizedLinear<D::Tensor>> {
        self.pools
            .experts
            .get(ei)
            .map(|expert| expert.down.clone())
            .ok_or_else(|| anyhow::anyhow!("glm53 moe: expert id {ei} out of range"))
    }

    /// Bind resources and scratch leases, then execute model-owned expert plans.
    pub fn forward_address_gpu(
        &self,
        x: &D::Tensor,
        sel: &D::Tensor,
        wts: &D::Tensor,
        b: usize,
        t: usize,
        h: usize,
        shared_for_sum: Option<&D::Tensor>,
        prepared_resources: Option<(D::Table, Vec<D::Tensor>)>,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        use crate::bindings::glm53::affine::moe::ops::run;
        use lisa_models::glm53::routed_moe::{Request as R, Spec};

        let device = self.context.device();
        let grouped = Spec::grouped(
            b * t,
            self.top_k,
            self.pools.inter,
            h,
            crate::bindings::glm53::policy::glm53_grouped_moe_prefill_override(),
        );
        let spec = Spec {
            batch: b,
            time: t,
            hidden: h,
            intermediate: self.pools.inter,
            top_k: self.top_k,
            experts: self.n_experts,
            metadata: match self.pools.metadata_dtype {
                Dtype::F32 => lisa_backend::ScalarType::F32,
                Dtype::BF16 => lisa_backend::ScalarType::BF16,
                _ => anyhow::bail!("unsupported expert metadata"),
            },
            limit: self.limit.to_bits(),
            grouped,
            split: crate::bindings::glm53::policy::glm53_grouped_moe_pair_split_override()
                .unwrap_or(true),
            vector: crate::bindings::glm53::policy::glm53_grouped_moe_q2_vector_load_override()
                .unwrap_or(false),
            down_tile: crate::bindings::glm53::policy::glm53_grouped_moe_down_tile_override()
                .unwrap_or(64),
            workspace: grouped && capture.is_none() && self.grouped_workspace.is_some(),
            shared: shared_for_sum.is_some(),
            capture: capture.is_some(),
            preserve: capture.is_some_and(Capture::preserves_dispatch),
        };
        if let Some(c) = capture {
            c.store(&format!("L{layer}.moe_input"), x);
        }
        let inputs = run(&self.context, R::Inputs(spec.clone()), &[x, sel, wts], None)?;
        let (selected, weights, x_rows) = (&inputs[0], &inputs[1], &inputs[2]);
        if self.pools.gpu_address_all_experts {
            self.context.device().evaluate(selected, false)?;
        }
        let transient;
        let resources = if self.pools.gpu_address_all_experts {
            if self.pools.resources.get().is_none() {
                let slots = self
                    .pools
                    .experts
                    .iter()
                    .flat_map(ExpertPools::<D>::arrays)
                    .collect::<Vec<_>>();
                let resource = device
                    .resources_with_residency(&slots, self.pools.residency_set.get().cloned())?;
                let _ = self.pools.resources.set(resource);
            }
            self.pools
                .resources
                .get()
                .expect("resident resources initialized")
        } else {
            let (table, owners) = if let Some(r) = prepared_resources {
                r
            } else {
                let _trace = self
                    .context
                    .trace_detail("glm53.moe.selected_id_readback", layer as u64);
                self.pools
                    .dispatch_resources(&device.read_i32_through(selected)?)?
            };
            let residency = self
                .pools
                .residency_set
                .get()
                .filter(|set| {
                    owners
                        .iter()
                        .all(|a| device.residency_contains(set, device.allocation_id(a)))
                })
                .cloned();
            // SAFETY: The streaming adapter constructs this table from checked
            // expert buffers. Every reachable selected allocation is retained
            // in owners; the device extends those leases through completion.
            transient = unsafe { device.bind_indirect_table(table, owners, residency)? };
            &transient
        };
        let routes = if grouped {
            run(&self.context, R::RouteMap(spec.clone()), &[selected], None)?
        } else {
            vec![]
        };
        let mut lease = if spec.workspace {
            Some(
                self.grouped_workspace
                    .as_ref()
                    .unwrap()
                    .buffers
                    .lock()
                    .map_err(|_| anyhow::anyhow!("grouped MoE<D> workspace mutex was poisoned"))?,
            )
        } else {
            None
        };
        let workspace = lease
            .as_mut()
            .map(|buffers| {
                buffers.prepare(device, b * t * self.top_k, self.pools.inter, h, spec.split)
            })
            .transpose()?;
        let mut scratch = vec![];
        if let Some(w) = &workspace {
            if let Some(gate) = &w.gate {
                scratch.push(gate.clone());
            }
            if let Some(up) = &w.up {
                scratch.push(up.clone());
            }
            scratch.push(w.act.clone());
        }
        let mut args = vec![x_rows, weights];
        if grouped {
            args.extend(routes.iter());
            args.extend(scratch.iter());
        } else {
            args.push(selected);
        }
        let act = run(
            &self.context,
            R::Activation(spec.clone()),
            &args,
            Some(resources),
        )?
        .remove(0);
        if let Some(c) = capture {
            c.store_ephemeral(&format!("L{layer}.moe_pair_weighted"), &act);
        }
        let down_workspace = workspace.as_ref().map(|w| w.down.clone());
        let mut args = vec![&act];
        if grouped {
            args.extend(routes.iter());
            if let Some(w) = &down_workspace {
                args.push(w);
            }
        } else {
            args.push(selected);
        }
        if let Some(shared) = shared_for_sum {
            args.push(shared);
        }
        let output = run(&self.context, R::Down(spec.clone()), &args, Some(resources))?;
        if let Some(c) = capture {
            if let Some(slots) = output.get(1) {
                c.store_ephemeral(&format!("L{layer}.moe_down_slots"), slots);
            }
            if spec.fused_down() {
                let mut args = vec![&act, selected];
                if let Some(shared) = shared_for_sum {
                    args.push(shared);
                }
                let shadow = run(
                    &self.context,
                    R::Shadow(spec.clone()),
                    &args,
                    Some(resources),
                )?;
                c.store_ephemeral(&format!("L{layer}.moe_down_slots"), &shadow[1]);
                let a = device.read_f32_through(&output[0])?;
                let b = device.read_f32_through(&shadow[0])?;
                anyhow::ensure!(
                    a.iter()
                        .map(|x| x.to_bits())
                        .eq(b.iter().map(|x| x.to_bits())),
                    "fused DS4 down projection differs from slotwise DS4 accumulation at layer {layer}"
                );
            }
        }
        // All scratch consumers have now been encoded on the same queue.
        drop(lease);
        Ok(run(
            &self.context,
            R::Output {
                shape: spec.output_shape().to_vec(),
            },
            &[&output[0]],
            None,
        )?
        .remove(0))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: &DeviceContext<D>,
        w: &mut Weights<D::Tensor>,
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
        lazy_store: Option<Arc<dyn TensorBank<D::Tensor>>>,
        map_all_expert_weights: bool,
    ) -> anyhow::Result<Self> {
        let take = |w: &mut Weights<D::Tensor>, name: &str| -> anyhow::Result<D::Tensor> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53 moe: {name} missing"))
        };
        let gate_w = moe_run(
            context,
            MoeRequest::Promote,
            &[&take(w, &format!("{prefix}.gate.weight"))?],
            None,
        )?
        .remove(0);
        let bias = moe_run(
            context,
            MoeRequest::Promote,
            &[&take(w, &format!("{prefix}.gate.e_score_correction_bias"))?],
            None,
        )?
        .remove(0);
        // Resident inference resolves every immutable expert address at load.
        // The selected-expert cache remains a diagnostic reference path.
        let expert_load_started = Instant::now();
        let (experts, expert_cache_bytes) = if let Some(store) = lazy_store.as_ref() {
            ExpertPools::load_experts(
                context,
                store.as_ref(),
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
                experts.push(ExpertParameters::<D> { gate, up, down });
            }
            let expert_cache_bytes = experts
                .iter()
                .map(|e| e.byte_size(context.device()))
                .collect();
            (experts, expert_cache_bytes)
        };
        if context.trace_enabled("glm53-loader") {
            eprintln!(
                "[glm53-loader] loaded {n_experts} routed experts for {prefix} in {:.1}s",
                expert_load_started.elapsed().as_secs_f64()
            );
        }
        let (address_table, metadata_dtype) =
            ExpertPools::address_table(context.device(), &experts)?;
        let pools = Arc::new(ExpertPools::<D> {
            device: context.device().clone(),
            observer: context.observer(),
            worker: context.with_slot(
                "glm53.expert.worker",
                || Arc::new(std::sync::OnceLock::new()),
                Arc::clone,
            ),
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
            resources: Default::default(),
            cache_capacity_bytes: AtomicUsize::new(0),
            cache: Mutex::new(ExpertCacheState::default()),
        });
        let shared = if n_shared > 0 {
            Some(DenseMlp::load_shared_on::<Weights<D::Tensor>>(
                context,
                w,
                &format!("{prefix}.shared_experts"),
                limit,
            )?)
        } else {
            None
        };
        Ok(Self {
            context: context.clone(),
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

    pub fn forward(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        self.forward_with_capture(x, None, 0)
    }

    pub fn forward_with_capture(
        &self,
        x: &D::Tensor,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        let (routed, shared) = self.forward_parts_with_capture_mode(x, capture, layer, true)?;
        match shared {
            Some(shared) => {
                Ok(moe_run(&self.context, MoeRequest::Add, &[&routed, &shared], None)?.remove(0))
            }
            None => Ok(routed),
        }
    }

    /// Return routed and shared expert outputs independently. The reference's GLM MTP
    /// residual combines both with the attention residual in one ordered add3.
    pub fn forward_parts_with_capture(
        &self,
        x: &D::Tensor,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<(D::Tensor, Option<D::Tensor>)> {
        self.forward_parts_with_capture_mode(x, capture, layer, false)
    }

    pub fn forward_parts_with_capture_mode(
        &self,
        x: &D::Tensor,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
        fuse_shared_sum8: bool,
    ) -> anyhow::Result<(D::Tensor, Option<D::Tensor>)> {
        let (b, t, h) = (x.dim(0) as usize, x.dim(1) as usize, x.dim(2) as usize);
        let n = b * t;
        let xf = moe_run(
            &self.context,
            MoeRequest::Rows { rows: n, hidden: h },
            &[x],
            None,
        )?
        .remove(0);
        let logits = {
            let _trace = self
                .context
                .trace_detail("glm53.moe.router_projection", layer as u64);
            moe_run(
                &self.context,
                MoeRequest::Router { rows: n, hidden: h },
                &[&xf, &self.gate_w],
                None,
            )?
            .remove(0)
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
            let (selected, weights) = {
                let _trace = self
                    .context
                    .trace_detail("glm53.moe.router_select", layer as u64);
                let mut values = moe_run(
                    &self.context,
                    MoeRequest::Select {
                        top_k: self.top_k,
                        scale: self.routed_scaling.to_bits(),
                    },
                    &[&logits, &self.bias],
                    None,
                )?
                .into_iter();
                (values.next().unwrap(), values.next().unwrap())
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
                self.context.device().evaluate(&selected, false)?;
                self.pools.stage_selected_async(&selected)
            };

            let fuse_shared_candidate = fuse_shared_sum8
                && capture.is_none_or(|capture| capture.preserves_dispatch())
                && self.top_k == 8;
            let shared = if let Some(sh) = &self.shared {
                // Only the streaming reference needs a submission boundary
                // here. Resident execution follows the tower's reference cadence.
                if !self.pools.gpu_address_all_experts {
                    self.context.device().submit(false)?;
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
                    .map(|shared| {
                        moe_run(
                            &self.context,
                            MoeRequest::Rows { rows: n, hidden: h },
                            &[shared],
                            None,
                        )
                        .map(|mut out| out.remove(0))
                    })
                    .transpose()?
            } else {
                None
            };
            let prepared_resources =
                if let Some(stage_result) = stage_result {
                    // Match the reference's streaming schedule: submit the shared-expert
                    // work before joining the selected-expert staging worker,
                    // so the GPU can execute it while the host prepares the
                    // routed experts. Without this flush, the join could wait
                    // on cache staging while the shared branch remained only
                    // encoded in the current command buffer.
                    self.context.device().submit(false)?;
                    let _trace = self.context.trace("glm53.moe.async_expert_stage_join");
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
        let logits_h: Vec<f32> = logits.read_f32()?;
        let bias_h: Vec<f32> = moe_run(&self.context, MoeRequest::Promote, &[&self.bias], None)?
            .remove(0)
            .read_f32()?;
        let mut rows = Vec::with_capacity(n);
        for row in 0..n {
            let lg = &logits_h[row * e..(row + 1) * e];
            let chosen = GroupedRouter {
                groups: self.n_group,
                selected_groups: self.topk_group,
                top_k: self.top_k,
                normalize: self.norm_topk,
                scaling: self.routed_scaling.to_bits(),
            }
            .select(lg, &bias_h)?;
            let mut acc: Option<D::Tensor> = None;
            for (expert_id, weight) in chosen {
                let gate_up = self.expert_gu(expert_id)?;
                let down = self.expert_down(expert_id)?;
                let term = moe_run(
                    &self.context,
                    MoeRequest::GroupedTerm {
                        row,
                        hidden: h,
                        intermediate: self.pools.inter,
                        gate_bits: gate_up.bits,
                        gate_group: gate_up.group_size,
                        down_bits: down.bits,
                        down_group: down.group_size,
                        weight: weight.to_bits(),
                        limit: self.limit.to_bits(),
                    },
                    &[
                        &xf,
                        &gate_up.weight,
                        &gate_up.scales,
                        &gate_up.biases,
                        &down.weight,
                        &down.scales,
                        &down.biases,
                    ],
                    None,
                )?
                .remove(0);
                acc = Some(if let Some(sum) = acc {
                    moe_run(&self.context, MoeRequest::Add, &[&sum, &term], None)?.remove(0)
                } else {
                    term
                });
            }

            rows.push(
                acc.ok_or_else(|| {
                    anyhow::anyhow!("glm53 moe: grouped router selected no experts")
                })?,
            );
        }
        let y = moe_run(
            &self.context,
            MoeRequest::JoinRows {
                rows: rows.len(),
                shape: vec![b as i32, t as i32, h as i32],
            },
            &rows.iter().collect::<Vec<_>>(),
            None,
        )?
        .remove(0);
        let shared = self.shared.as_ref().map(|sh| sh.forward(x)).transpose()?;
        Ok((y, shared))
    }
}

pub mod ops;
