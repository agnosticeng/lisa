//! DSpine block drafter for the Qwen 3.8 27B target.
//!
//! The checkpoint is separate from the target and uses the DSpine code bridge
//! over five target-layer features. The implementation follows Tencent's
//! Apache-2.0 DSpine inference reference (revision
//! `321b0fc1388c4cdf03483ab73856975f7c0c0be7`); see the repository NOTICE.

use crate::bindings::dspine_bridge as bound_bridge;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use anyhow::{Context, ensure};
use lisa_backend::{HostTensor, ManagedDevice, ScalarType as Dtype, TensorAccess};
use lisa_models::qwen3_5::dspine::bridge::Utility;
use lisa_models::qwen3_5::dspine::config::{DSpineConfig, DraftQuantization};
use std::path::Path;

use crate::{
    attention_cache::FullAttentionCache, bindings::normalization::RmsNorm, weights::mlx::Shard,
};

/// Shared Hub checkpoint for compatible Qwen 27B targets.
pub const REPO: &str = "tencent/Qwen3.8-27B-DSpine-b16";

/// Select by checkpoint geometry, since smaller Qwen models share model_type.
pub use lisa_models::qwen3_5::dspine::config::supports_target;

pub fn resolve_selection(explicit: Option<&Path>) -> anyhow::Result<Option<std::path::PathBuf>> {
    if let Some(path) = explicit {
        return Ok(Some(path.to_path_buf()));
    }
    match std::env::var("LISA_QWEN_DSPINE_MODEL") {
        Ok(model) => Ok(Some(crate::checkpoint::resolve_model_dir(&model)?)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub trait DSpineTarget<T> {
    fn embed_tokens(&self, ids: &T) -> anyhow::Result<T>;
    fn head_logits(&self, hidden: &T) -> anyhow::Result<T>;
    fn head_rows(&self, ids: &T) -> anyhow::Result<T>;
    fn shortlist(&self, _hidden: &T) -> anyhow::Result<Option<T>> {
        Ok(None)
    }
}

pub struct DenseLinear<T> {
    pub weight: T,
    pub bias: Option<T>,
    pub quantized: Option<(T, T, i32)>,
}

impl<T: TensorAccess> DenseLinear<T> {
    pub fn load<D: BindingDevice<Tensor = T>>(
        context: &DeviceContext<D>,
        shards: &[Shard<D>],
        prefix: &str,
        bias: bool,
        quant: Option<&DraftQuantization>,
    ) -> anyhow::Result<Self> {
        let weight = tensor(shards, &format!("{prefix}.weight"))?;
        let (weight, quantized) = if let Some(q) = quant {
            let scales = tensor(shards, &format!("{prefix}.scales"))?;
            let biases = tensor(shards, &format!("{prefix}.biases"))?;
            ensure!(
                context.device().dtype(&weight) == Some(Dtype::U32)
                    && weight.rank() == 2
                    && context.device().dtype(&scales) == Some(Dtype::BF16)
                    && context.device().dtype(&biases) == Some(Dtype::BF16)
                    && scales.shape() == biases.shape()
                    && scales.shape() == [weight.dim(0), weight.dim(1) * 8 / q.group_size]
                    && weight.dim(1) * 8 % q.group_size == 0,
                "invalid DSpine affine tensors for {prefix}"
            );
            (weight, Some((scales, biases, q.group_size)))
        } else {
            ensure!(
                context.device().dtype(&weight) == Some(Dtype::BF16),
                "DSpine tensor {prefix}.weight must be BF16"
            );
            // Prepare the original layout for NAX or one transposed copy for
            // legacy dense matmul. Quantized weights remain packed on device.
            (Self::prepare_weight(context, &weight)?, None)
        };
        let bias = if bias {
            Some(tensor(shards, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            quantized,
        })
    }

    pub fn prepare_weight<D: BindingDevice<Tensor = T>>(
        context: &DeviceContext<D>,
        weight: &T,
    ) -> anyhow::Result<T> {
        crate::bindings::dspine_layers::prepare(
            context,
            weight,
            crate::bindings::dspine_layers::small_rows(context),
        )
    }

    pub fn bindings(&self) -> Vec<&T> {
        let mut inputs = vec![&self.weight];
        if let Some((scales, biases, _)) = &self.quantized {
            inputs.extend([scales, biases]);
        }
        if let Some(bias) = &self.bias {
            inputs.push(bias);
        }
        inputs
    }

    pub fn shape(&self) -> [i32; 2] {
        [
            self.weight.dim(0),
            self.weight.dim(1) * if self.quantized.is_some() { 8 } else { 1 },
        ]
    }
}

pub struct DraftAttention<T> {
    pub q: DenseLinear<T>,
    pub k: DenseLinear<T>,
    pub v: DenseLinear<T>,
    pub o: DenseLinear<T>,
    pub q_norm: RmsNorm<T>,
    pub k_norm: RmsNorm<T>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub window: Option<usize>,
}

impl<T: TensorAccess> DraftAttention<T> {
    pub fn load<D: BindingDevice<Tensor = T>>(
        context: &DeviceContext<D>,
        shards: &[Shard<D>],
        cfg: &DSpineConfig,
        index: usize,
    ) -> anyhow::Result<Self> {
        let prefix = format!("layers.{index}.self_attn");
        let load_norm = |name: &str| -> anyhow::Result<RmsNorm<T>> {
            Ok(RmsNorm {
                weight: tensor(shards, &format!("{prefix}.{name}.weight"))?,
                eps: cfg.rms_norm_eps,
                group_size: None,
            })
        };
        let q = DenseLinear::load(
            context,
            shards,
            &format!("{prefix}.q_proj"),
            false,
            cfg.quantization.as_ref(),
        )?;
        let k = DenseLinear::load(
            context,
            shards,
            &format!("{prefix}.k_proj"),
            false,
            cfg.quantization.as_ref(),
        )?;
        let v = DenseLinear::load(
            context,
            shards,
            &format!("{prefix}.v_proj"),
            false,
            cfg.quantization.as_ref(),
        )?;
        let o = DenseLinear::load(
            context,
            shards,
            &format!("{prefix}.o_proj"),
            false,
            cfg.quantization.as_ref(),
        )?;
        let q_norm = load_norm("q_norm")?;
        let k_norm = load_norm("k_norm")?;
        ensure!(
            q.shape()
                == [
                    (cfg.num_attention_heads * cfg.head_dim) as i32,
                    cfg.hidden_size as i32,
                ]
                && k.shape()
                    == [
                        (cfg.num_key_value_heads * cfg.head_dim) as i32,
                        cfg.hidden_size as i32,
                    ]
                && v.shape() == k.shape()
                && o.shape()
                    == [
                        cfg.hidden_size as i32,
                        (cfg.num_attention_heads * cfg.head_dim) as i32,
                    ]
                && q_norm.weight.shape() == [cfg.head_dim as i32]
                && k_norm.weight.shape() == [cfg.head_dim as i32],
            "DSpine attention tensors do not match the configured geometry"
        );
        Ok(Self {
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            theta: cfg.rope_theta,
            window: cfg.attention_window(),
        })
    }

    pub fn description<D: BindingDevice<Tensor = T>>(
        &self,
        shape: &[i32],
        small_rows: bool,
        cache: &FullAttentionCache<D>,
        cache_start: usize,
        context_only: bool,
    ) -> lisa_models::qwen3_5::dspine::layers::Attention {
        lisa_models::qwen3_5::dspine::layers::Attention {
            hidden: shape[2],
            heads: self.heads as i32,
            theta: self.theta.to_bits(),
            epsilon: self.q_norm.eps.to_bits(),
            quantized: self.q.quantized.is_some(),
            small_rows,
            cache: lisa_models::attention_cache::Append {
                batch: shape[0],
                rows: shape[1],
                heads: self.kv_heads as i32,
                width: self.head_dim as i32,
                offset: cache.offset,
                capacity: cache.capacity,
                previous_batch: cache.keys.as_ref().map(|k| k.dim(0)),
                dtype: lisa_backend::ScalarType::BF16,
            },
            cache_start,
            window: self.window,
            context_only,
        }
    }
}

pub struct DraftLayer<T> {
    pub attention: DraftAttention<T>,
    pub gate: DenseLinear<T>,
    pub up: DenseLinear<T>,
    pub down: DenseLinear<T>,
    pub input_norm: RmsNorm<T>,
    pub post_norm: RmsNorm<T>,
}

impl<T: TensorAccess> DraftLayer<T> {
    pub fn load<D: BindingDevice<Tensor = T>>(
        context: &DeviceContext<D>,
        shards: &[Shard<D>],
        cfg: &DSpineConfig,
        index: usize,
    ) -> anyhow::Result<Self> {
        let prefix = format!("layers.{index}");
        let load_norm = |name: &str| -> anyhow::Result<RmsNorm<T>> {
            Ok(RmsNorm {
                weight: tensor(shards, &format!("{prefix}.{name}.weight"))?,
                eps: cfg.rms_norm_eps,
                group_size: None,
            })
        };
        let gate = DenseLinear::load(
            context,
            shards,
            &format!("{prefix}.mlp.gate_proj"),
            false,
            cfg.quantization.as_ref(),
        )?;
        let up = DenseLinear::load(
            context,
            shards,
            &format!("{prefix}.mlp.up_proj"),
            false,
            cfg.quantization.as_ref(),
        )?;
        let down = DenseLinear::load(
            context,
            shards,
            &format!("{prefix}.mlp.down_proj"),
            false,
            cfg.quantization.as_ref(),
        )?;
        ensure!(
            gate.shape() == [cfg.intermediate_size as i32, cfg.hidden_size as i32]
                && up.shape() == gate.shape()
                && down.shape() == [cfg.hidden_size as i32, cfg.intermediate_size as i32],
            "DSpine MLP tensors do not match the configured geometry"
        );
        Ok(Self {
            attention: DraftAttention::load(context, shards, cfg, index)?,
            gate,
            up,
            down,
            input_norm: load_norm("input_layernorm")?,
            post_norm: load_norm("post_attention_layernorm")?,
        })
    }
}

pub struct CodeBridge<T> {
    pub packed_weight: T,
    pub sender_weight: T,
    pub read_weight: T,
    pub early_read_weights: T,
    pub feature_write_weights: T,
    pub codebook: T,
    pub gate_bias: T,
    pub rank: usize,
    pub code_size: usize,
    pub eps: f32,
}

impl<T: TensorAccess> CodeBridge<T> {
    pub fn load<D: BindingDevice<Tensor = T>>(
        context: &DeviceContext<D>,
        shards: &[Shard<D>],
        cfg: &DSpineConfig,
    ) -> anyhow::Result<Self> {
        let p = "selected_prefix_bridge";
        let version = tensor(shards, "draft_block_causal_version")?;
        ensure!(
            scalar_i64(context, &version)? == 1,
            "unsupported DSpine causal version"
        );
        let refine = tensor(shards, "last_write_refine_version")?;
        ensure!(
            scalar_i64(context, &refine)? == 1,
            "unsupported DSpine refinement version"
        );
        let independent = tensor(shards, &format!("{p}.independent_write_version"))?;
        ensure!(
            scalar_i64(context, &independent)? == 1,
            "unsupported DSpine write version"
        );
        let terminal = tensor(shards, &format!("{p}.terminal_disabled_version"))?;
        ensure!(
            scalar_i64(context, &terminal)? == 1,
            "unsupported DSpine terminal version"
        );
        let read_weight = tensor(shards, &format!("{p}.read_weight"))?;
        let early_read_weights = tensor(shards, &format!("{p}.early_read_weights"))?;
        let feature_write_weights = tensor(shards, &format!("{p}.feature_write_weights"))?;
        let codebook = tensor(shards, &format!("{p}.codebook"))?;
        let bridge = &cfg.dflash.selected_prefix_code_bridge;
        ensure!(
            read_weight.shape() == [bridge.code_size as i32, cfg.hidden_size as i32]
                && early_read_weights.shape()
                    == [
                        (cfg.num_hidden_layers - 1) as i32,
                        bridge.code_size as i32,
                        cfg.hidden_size as i32,
                    ]
                && feature_write_weights.shape()
                    == [
                        cfg.num_hidden_layers as i32,
                        cfg.hidden_size as i32,
                        bridge.rank as i32,
                    ]
                && codebook.shape() == [cfg.vocab_size as i32, bridge.code_size as i32],
            "DSpine bridge tensor geometry does not match its config"
        );
        let out = tensor(shards, &format!("{p}.out.weight"))?;
        ensure!(
            context
                .device()
                .read_bytes(&out)?
                .iter()
                .all(|&value| value == 0),
            "DSpine checkpoint has a nonzero terminal bridge projection"
        );
        let prev = DenseLinear::load(context, shards, &format!("{p}.prev"), false, None)?;
        let gate = DenseLinear::load(context, shards, &format!("{p}.gate"), true, None)?;
        let predecessor_gate = DenseLinear::load(
            context,
            shards,
            &format!("{p}.predecessor_gate"),
            false,
            None,
        )?;
        // Match upstream prepare(): pack immutable bridge projections once.
        let packed_weight = DenseLinear::prepare_weight(
            context,
            &bound_bridge::one(
                context,
                Utility::Pack(3),
                &[&gate.weight, &predecessor_gate.weight, &prev.weight],
            )?,
        )?;
        let sender_weight = DenseLinear::prepare_weight(
            context,
            &bound_bridge::one(
                context,
                Utility::Pack(2),
                &[&predecessor_gate.weight, &prev.weight],
            )?,
        )?;
        let read_weight = DenseLinear::prepare_weight(context, &read_weight)?;
        let prepare_layers = |weights: &T| {
            crate::bindings::dspine_layers::prepare(
                context,
                weights,
                crate::bindings::dspine_layers::small_rows(context),
            )
        };
        let early_read_weights = prepare_layers(&early_read_weights)?;
        let feature_write_weights = prepare_layers(&feature_write_weights)?;
        Ok(Self {
            packed_weight,
            sender_weight,
            gate_bias: tensor(shards, &format!("{p}.gate.bias"))?,
            read_weight,
            early_read_weights,
            feature_write_weights,
            codebook,
            rank: bridge.rank,
            code_size: bridge.code_size,
            eps: bridge.eps,
        })
    }
}

pub struct DraftStash<T> {
    pub raw: T,
    pub rms: T,
    pub packed: T,
}

/// Stage every layer before replacing any cache, so allocation failures leave
/// the prior origin available for transaction rollback. Copies preserve bytes.
pub fn compact_draft_caches<D: BindingDevice>(
    context: &DeviceContext<D>,
    caches: &[FullAttentionCache<D>],
    keep: usize,
) -> anyhow::Result<Vec<FullAttentionCache<D>>> {
    caches
        .iter()
        .map(|cache| {
            ensure!(
                cache.offset >= keep,
                "draft cache shorter than retained window"
            );
            let keys = cache.keys.as_ref().context("missing draft keys")?;
            let values = cache.values.as_ref().context("missing draft values")?;
            let spec = lisa_models::qwen3_5::dspine::state::Plan::Compact {
                batch: keys.dim(0),
                heads: cache.kv_heads as i32,
                width: cache.head_dim as i32,
                offset: cache.offset,
                keep,
                dtype: if cache.dtype == lisa_backend::ScalarType::BF16 {
                    lisa_backend::ScalarType::BF16
                } else {
                    lisa_backend::ScalarType::F32
                },
            };
            let mut output = bound_bridge::state(context, spec, &[keys, values])?.into_iter();
            let mut next = FullAttentionCache::with_dtype(
                context.clone(),
                cache.kv_heads,
                cache.head_dim,
                cache.dtype,
            );
            next.keys = Some(output.next().context("missing compacted draft keys")?);
            next.values = Some(output.next().context("missing compacted draft values")?);
            next.capacity = next.keys.as_ref().unwrap().dim(2) as usize;
            next.offset = keep;
            Ok(next)
        })
        .collect()
}

/// Immutable BF16 vocabulary projections from Tencent's build_serial_tables.
/// Build using the selected-row matmul shape so cached and uncached paths obey
/// the same reduction law, including the padded final vocabulary chunk.
pub struct SerialTables<T> {
    pub tg: T,
    pub tsv: T,
    pub rows_per_projection: usize,
}

pub fn full_tables_enabled() -> anyhow::Result<bool> {
    match std::env::var("LISA_DSPINE_TABLES").as_deref() {
        Ok("full") => Ok(true),
        Err(std::env::VarError::NotPresent) | Ok("selected") => Ok(false),
        _ => anyhow::bail!("LISA_DSPINE_TABLES must be selected or full"),
    }
}

pub struct DSpineDraft<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    pub cfg: DSpineConfig,
    pub fc: DenseLinear<D::Tensor>,
    pub hidden_norm: RmsNorm<D::Tensor>,
    pub layers: Vec<DraftLayer<D::Tensor>>,
    pub final_norm: RmsNorm<D::Tensor>,
    pub bridge: CodeBridge<D::Tensor>,
    pub caches: Vec<FullAttentionCache<D>>,
    /// Absolute position represented by physical KV row zero.
    pub cache_start: usize,
    pub context_tokens: Vec<u32>,
    pub serial_tables: Option<SerialTables<D::Tensor>>,
    pub full_tables: bool,
    pub target_reserved_bytes: usize,
    pub block_size: usize,
    pub verify_width: usize,
    pub replay: bool,
}

impl<D: BindingDevice + ManagedDevice> DSpineDraft<D> {
    pub fn load(
        context: DeviceContext<D>,
        path: &Path,
        target_vocab: usize,
        target_hidden: usize,
        target_reserved_bytes: usize,
    ) -> anyhow::Result<Self> {
        let cfg = DSpineConfig::parse(
            &std::fs::read(path.join("config.json"))?,
            target_vocab,
            target_hidden,
        )?;
        let mut shard_paths: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|file| file.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shard_paths.sort();
        ensure!(
            !shard_paths.is_empty(),
            "DSpine checkpoint has no safetensors weights"
        );
        let draft_bytes = shard_paths.iter().try_fold(0usize, |total, file| {
            total
                .checked_add(std::fs::metadata(file)?.len() as usize)
                .ok_or_else(|| anyhow::anyhow!("DSpine checkpoint size overflow"))
        })?;
        let already_used = context
            .device()
            .memory_usage()?
            .private
            .checked_add(target_reserved_bytes)
            .ok_or_else(|| anyhow::anyhow!("DSpine admission size overflow"))?;
        let needed = already_used
            .checked_add(draft_bytes)
            .ok_or_else(|| anyhow::anyhow!("DSpine admission size overflow"))?;
        ensure!(
            needed <= crate::memory::ram_budget(),
            "DSpine draft needs {:.2} GiB in addition to the loaded target; RAM budget is {:.2} GiB",
            draft_bytes as f64 / (1u64 << 30) as f64,
            crate::memory::ram_budget() as f64 / (1u64 << 30) as f64
        );
        let shards = shard_paths
            .iter()
            .map(|file| Shard::open(context.device().clone(), file))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let hidden_norm = RmsNorm {
            weight: tensor(&shards, "hidden_norm.weight")?,
            eps: cfg.rms_norm_eps,
            group_size: None,
        };
        let final_norm = RmsNorm {
            weight: tensor(&shards, "norm.weight")?,
            eps: cfg.rms_norm_eps,
            group_size: None,
        };
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DraftLayer::load(&context, &shards, &cfg, i))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let bridge = CodeBridge::load(&context, &shards, &cfg)?;
        let concat_size = cfg.target_layers().len() * cfg.hidden_size;
        let fc = DenseLinear::load(&context, &shards, "fc", false, cfg.quantization.as_ref())?;
        ensure!(
            fc.shape() == [cfg.hidden_size as i32, concat_size as i32],
            "DSpine target-feature projection has an unexpected shape"
        );
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| {
                FullAttentionCache::with_dtype(
                    context.clone(),
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    Dtype::BF16,
                )
            })
            .collect();
        let block_size = cfg.served_block_size();
        Ok(Self {
            context: context.clone(),
            cfg,
            fc,
            hidden_norm,
            layers,
            final_norm,
            bridge,
            caches,
            cache_start: 0,
            context_tokens: Vec::new(),
            serial_tables: None,
            full_tables: full_tables_enabled()?,
            target_reserved_bytes,
            block_size,
            verify_width: block_size,
            replay: match std::env::var("LISA_DSPINE_SSM").as_deref() {
                Ok("replay") => true,
                Err(std::env::VarError::NotPresent) | Ok("capture") => false,
                _ => anyhow::bail!("LISA_DSPINE_SSM must be replay or capture"),
            },
        })
    }

    pub fn target_layers(&self) -> &[usize] {
        self.cfg.target_layers()
    }

    pub fn max_draft_depth(&self) -> usize {
        self.block_size.saturating_sub(1)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.restore_offset(0);
        }
        self.cache_start = 0;
        self.context_tokens.clear();
    }

    pub fn trim(&mut self, rows: usize) {
        self.restore_offset(self.offset().saturating_sub(rows));
    }

    pub fn offset(&self) -> usize {
        self.cache_start + self.caches.first().map_or(0, |cache| cache.offset)
    }

    pub fn restore_offset(&mut self, offset: usize) {
        let needed_start = lisa_models::qwen3_5::dspine::state::retained_start(
            offset,
            self.cfg.attention_window(),
        );
        if needed_start < self.cache_start {
            // An older target snapshot may need history that was physically
            // evicted. Invalidate the companion; the session's existing
            // context-mismatch path rebuilds it from the restored target prefix.
            self.reset();
            return;
        }
        for cache in &mut self.caches {
            cache.restore_offset(offset.saturating_sub(self.cache_start));
        }
        self.context_tokens.truncate(offset);
    }

    pub fn context_matches(&self, tokens: &[u32]) -> bool {
        self.context_tokens == tokens
    }

    pub fn append_target_hidden(
        &mut self,
        target_hidden: &D::Tensor,
        tokens: &[u32],
    ) -> anyhow::Result<()> {
        let _trace = self
            .context
            .trace_detail("dspine.context", tokens.len() as u64);
        ensure!(
            target_hidden.dim(1) as usize == tokens.len(),
            "DSpine target tap/token length mismatch"
        );
        if target_hidden.dim(1) == 0 {
            return Ok(());
        }
        let projection = lisa_models::qwen3_5::dspine::layers::Linear {
            shape: target_hidden.shape().to_vec(),
            output: self.fc.shape()[0],
            quantized: self.fc.quantized.is_some(),
            bias: false,
            small_rows: crate::bindings::dspine_layers::small_rows(&self.context),
        };
        let offset = self.offset();
        ensure!(
            self.caches
                .iter()
                .all(|cache| self.cache_start + cache.offset == offset),
            "DSpine layer context offsets diverged"
        );
        let result = (|| -> anyhow::Result<()> {
            let shape = [
                target_hidden.dim(0),
                target_hidden.dim(1),
                self.cfg.hidden_size as i32,
            ];
            let spec = lisa_models::qwen3_5::dspine::block::Context {
                projection,
                epsilon: self.hidden_norm.eps.to_bits(),
                attention: self
                    .layers
                    .iter()
                    .zip(&self.caches)
                    .map(|(l, c)| {
                        l.attention.description(
                            &shape,
                            crate::bindings::dspine_layers::small_rows(&self.context),
                            c,
                            self.cache_start,
                            true,
                        )
                    })
                    .collect(),
            };
            let layouts = spec
                .attention
                .iter()
                .map(|a| a.cache.layout().map_err(anyhow::Error::msg))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let mut inputs = vec![target_hidden];
            inputs.extend(self.fc.bindings());
            inputs.push(&self.hidden_norm.weight);
            for layer in &self.layers {
                for projection in [
                    &layer.attention.q,
                    &layer.attention.k,
                    &layer.attention.v,
                    &layer.attention.o,
                ] {
                    inputs.extend(projection.bindings());
                }
                inputs.extend([
                    &layer.attention.q_norm.weight,
                    &layer.attention.k_norm.weight,
                ]);
            }
            for cache in &self.caches {
                if let Some(keys) = &cache.keys {
                    inputs.push(keys);
                    inputs.push(cache.values.as_ref().context("missing draft values")?);
                }
            }
            let mut out = bound_bridge::context(&self.context, spec, &inputs)?.into_iter();
            for (cache, (end, capacity, _)) in self.caches.iter_mut().zip(layouts) {
                cache.keys = Some(out.next().context("missing context keys")?);
                cache.values = Some(out.next().context("missing context values")?);
                cache.offset = end;
                cache.capacity = capacity;
            }
            if let Some((keep, shift)) = lisa_models::qwen3_5::dspine::state::compaction(
                self.caches[0].offset,
                self.cfg.attention_window(),
            ) {
                self.caches = compact_draft_caches(&self.context, &self.caches, keep)?;
                self.cache_start += shift;
            }
            Ok(())
        })();
        if result.is_err() {
            self.restore_offset(offset);
            return result;
        }
        self.context_tokens.extend_from_slice(tokens);
        Ok(())
    }

    pub fn candidates(
        &self,
        rows: &D::Tensor,
        target: &dyn DSpineTarget<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        let _trace = self.context.trace("dspine.candidates");
        if let Some(ids) = target.shortlist(rows)? {
            return Ok(ids);
        }
        let logits = target.head_logits(rows)?;
        bound_bridge::one(&self.context, Utility::Candidates(16), &[&logits])
    }

    pub fn project_tg(&self, head_rows: &D::Tensor) -> anyhow::Result<D::Tensor> {
        bound_bridge::one(
            &self.context,
            Utility::Tg(self.cfg.hidden_size as i32),
            &[
                head_rows,
                &self.final_norm.weight,
                &self.bridge.feature_write_weights,
            ],
        )
    }

    pub fn project_tsv(&self, ids: &D::Tensor) -> anyhow::Result<D::Tensor> {
        bound_bridge::one(
            &self.context,
            Utility::Tsv(
                ids.size() as i32,
                self.bridge.code_size as i32,
                self.bridge.rank as i32,
                crate::bindings::dspine_layers::small_rows(&self.context),
            ),
            &[ids, &self.bridge.codebook, &self.bridge.sender_weight],
        )
    }

    pub fn build_serial_tables(
        &self,
        target: &dyn DSpineTarget<D::Tensor>,
        vocab: usize,
    ) -> anyhow::Result<SerialTables<D::Tensor>> {
        let _trace = self.context.trace("dspine.tables.build");
        let chunk =
            (self.block_size - 1) * self.cfg.dflash.selected_prefix_last_write_refine.head_topk;
        ensure!(
            chunk > 0 && vocab <= self.cfg.vocab_size,
            "invalid table geometry"
        );
        let r = self.bridge.rank;
        let table_bytes = vocab
            .checked_mul(3 * r * 2)
            .context("DSpine table size overflow")?;
        // Include mapped target bytes when a storage adapter supplies them and
        // reserve bounded staging space independently of vocabulary size.
        let staging = chunk * (self.cfg.hidden_size + self.bridge.code_size + 3 * r) * 64;
        let needed = self
            .context
            .device()
            .memory_usage()?
            .active
            .checked_add(self.target_reserved_bytes)
            .and_then(|n| n.checked_add(table_bytes))
            .and_then(|n| n.checked_add(staging))
            .context("DSpine table admission size overflow")?;
        ensure!(
            needed <= crate::memory::ram_budget(),
            "DSpine full tables exceed the RAM budget; use LISA_DSPINE_TABLES=selected"
        );
        let mut tables = bound_bridge::state(
            &self.context,
            lisa_models::qwen3_5::dspine::state::Plan::Tables {
                vocab: vocab as i32,
                rank: r as i32,
            },
            &[],
        )?
        .into_iter();
        let tg = tables.next().context("missing TG table storage")?;
        let tsv = tables.next().context("missing TSV table storage")?;
        for (iteration, start) in (0..vocab).step_by(chunk).enumerate() {
            let end = (start + chunk).min(vocab);
            let ids = lisa_models::qwen3_5::dspine::state::table_ids(start, chunk, vocab);
            let ids = pull_ids(&self.context, &ids, &[chunk as i32])?;
            let head_rows = target.head_rows(&ids)?;
            ensure!(
                head_rows.shape() == [chunk as i32, self.cfg.hidden_size as i32],
                "DSpine target head table shape mismatch"
            );
            let tg_rows = self.project_tg(&head_rows)?;
            let tsv_rows = self.project_tsv(&ids)?;
            bound_bridge::state(
                &self.context,
                lisa_models::qwen3_5::dspine::state::Plan::Fill {
                    start: start as i32,
                    end: end as i32,
                },
                &[&tg_rows, &tsv_rows, &tg, &tsv],
            )?;
            // Copies and projection scratch remain leased until completion.
            // Keep startup working memory bounded without allocator-lock waits.
            if iteration % 8 == 7 {
                self.context.device().submit(true)?;
            }
        }
        self.context.device().submit(true)?;
        Ok(SerialTables {
            tg,
            tsv,
            rows_per_projection: chunk,
        })
    }

    pub fn greedy_refine(
        &self,
        rows: &D::Tensor,
        stash: DraftStash<D::Tensor>,
        anchor: u32,
        target: &dyn DSpineTarget<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        let _trace = self.context.trace("dspine.refine");
        let n = rows.dim(0);
        let k = self.cfg.dflash.selected_prefix_last_write_refine.head_topk as i32;
        let candidates = self.candidates(rows, target)?;
        let flat_ids = bound_bridge::one(&self.context, Utility::FlatIds(n, k), &[&candidates])?;
        let head_rows = target.head_rows(&flat_ids)?;
        let anchor_ids = lisa_models::qwen3_5::dspine::bridge::anchor_ids(anchor, k as usize);
        let anchor_ids = pull_ids(&self.context, &anchor_ids, &[1, k])?;
        let mut inputs = vec![
            &candidates,
            &head_rows,
            &stash.raw,
            &stash.rms,
            &stash.packed,
            &self.final_norm.weight,
            &self.bridge.gate_bias,
            &self.bridge.feature_write_weights,
            &self.bridge.sender_weight,
            &self.bridge.codebook,
            &anchor_ids,
        ];
        if let Some(tables) = &self.serial_tables {
            ensure!(
                tables.rows_per_projection == (n * k) as usize,
                "DSpine table projection shape changed"
            );
            inputs.extend([&tables.tg, &tables.tsv]);
        }
        bound_bridge::refine(
            &self.context,
            lisa_models::qwen3_5::dspine::bridge::Refine {
                rows: n,
                count: k,
                hidden: self.cfg.hidden_size as i32,
                rank: self.bridge.rank as i32,
                codes: self.bridge.code_size as i32,
                tables: self.serial_tables.is_some(),
                small_rows: crate::bindings::dspine_layers::small_rows(&self.context),
            },
            &inputs,
        )
    }

    pub fn draft_block(
        &mut self,
        anchor: u32,
        target: &dyn DSpineTarget<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        let _trace = self.context.trace("dspine.draft");
        let offset = self.offset();
        let result = self.draft_block_inner(anchor, target);
        self.restore_offset(offset);
        result
    }

    pub fn draft_block_inner(
        &mut self,
        anchor: u32,
        target: &dyn DSpineTarget<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        if self.full_tables && self.serial_tables.is_none() {
            self.serial_tables = Some(self.build_serial_tables(target, self.cfg.vocab_size)?);
        }
        let (rows, stash) = self.draft_hidden(anchor, target)?;
        self.greedy_refine(&rows, stash, anchor, target)
    }

    pub fn draft_hidden(
        &mut self,
        anchor: u32,
        target: &dyn DSpineTarget<D::Tensor>,
    ) -> anyhow::Result<(D::Tensor, DraftStash<D::Tensor>)> {
        let ids = lisa_models::qwen3_5::dspine::block::input_ids(
            anchor,
            self.cfg.dflash.mask_token_id,
            self.block_size,
            self.cfg.block_size,
        )?;
        let ids = pull_i32(&self.context, &ids, &[1, self.block_size as i32])?;
        let embedding = target.embed_tokens(&ids)?;
        let spec = lisa_models::qwen3_5::dspine::block::Block {
            attention: self
                .layers
                .iter()
                .zip(&self.caches)
                .map(|(l, c)| {
                    l.attention.description(
                        embedding.shape(),
                        crate::bindings::dspine_layers::small_rows(&self.context),
                        c,
                        self.cache_start,
                        false,
                    )
                })
                .collect(),
            intermediate: self.cfg.intermediate_size as i32,
            epsilon: self.cfg.rms_norm_eps.to_bits(),
            bridge_epsilon: self.bridge.eps.to_bits(),
            rank: self.bridge.rank as i32,
            codes: self.bridge.code_size as i32,
        };
        let mut inputs = vec![&embedding, &ids];
        for layer in &self.layers {
            inputs.push(&layer.input_norm.weight);
            for projection in [
                &layer.attention.q,
                &layer.attention.k,
                &layer.attention.v,
                &layer.attention.o,
            ] {
                inputs.extend(projection.bindings());
            }
            inputs.extend([
                &layer.attention.q_norm.weight,
                &layer.attention.k_norm.weight,
                &layer.post_norm.weight,
            ]);
            for projection in [&layer.gate, &layer.up, &layer.down] {
                inputs.extend(projection.bindings());
            }
        }
        inputs.extend([
            &self.final_norm.weight,
            &self.bridge.read_weight,
            &self.bridge.early_read_weights,
            &self.bridge.codebook,
            &self.bridge.packed_weight,
            &self.bridge.gate_bias,
            &self.bridge.feature_write_weights,
        ]);
        for cache in &self.caches {
            if let Some(keys) = &cache.keys {
                inputs.push(keys);
                inputs.push(cache.values.as_ref().context("missing draft values")?);
            }
        }
        let layouts = spec
            .attention
            .iter()
            .map(|a| a.cache.layout().map_err(anyhow::Error::msg))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut output = bound_bridge::block(&self.context, spec, &inputs)?.into_iter();
        let rows = output.next().context("missing draft rows")?;
        let stash = DraftStash {
            raw: output.next().context("missing draft raw state")?,
            rms: output.next().context("missing draft RMS state")?,
            packed: output.next().context("missing draft packed state")?,
        };
        for (cache, (_, capacity, _)) in self.caches.iter_mut().zip(layouts) {
            cache.keys = Some(output.next().context("missing draft block keys")?);
            cache.values = Some(output.next().context("missing draft block values")?);
            cache.capacity = capacity;
        }
        Ok((rows, stash))
    }
}

pub fn tensor<D: BindingDevice>(shards: &[Shard<D>], name: &str) -> anyhow::Result<D::Tensor> {
    for candidate in [name.to_owned(), format!("model.{name}")] {
        if let Some(shard) = shards.iter().find(|s| s.tensor_bytes(&candidate).is_some()) {
            if let Some(tensor) = shard.tensor_array(&candidate)? {
                return Ok(tensor);
            }
            let (bytes, dtype, shape) = shard.tensor_bytes(&candidate).unwrap();
            return Ok(shard.device().pull(
                HostTensor {
                    bytes,
                    dtype,
                    shape,
                }
                .into(),
                Default::default(),
            )?);
        }
    }
    anyhow::bail!("missing DSpine tensor {name}")
}
fn pull_ids<D: BindingDevice>(
    context: &DeviceContext<D>,
    ids: &[u32],
    shape: &[i32],
) -> anyhow::Result<D::Tensor> {
    Ok(context.device().pull(
        HostTensor {
            bytes: &ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(),
            shape,
            dtype: Dtype::U32,
        }
        .into(),
        Default::default(),
    )?)
}
fn scalar_i64<D: BindingDevice>(context: &DeviceContext<D>, t: &D::Tensor) -> anyhow::Result<i64> {
    ensure!(t.size() == 1, "expected one checkpoint version");
    let bytes = context.device().read_bytes(t)?;
    macro_rules! scalar {
        ($ty:ty) => {
            <$ty>::from_ne_bytes(
                bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid checkpoint scalar width"))?,
            )
        };
    }
    Ok(
        match context.device().dtype(t).context("unknown version dtype")? {
            Dtype::I64 => scalar!(i64),
            Dtype::I32 => scalar!(i32) as i64,
            Dtype::U32 => scalar!(u32) as i64,
            Dtype::U16 => scalar!(u16) as i64,
            Dtype::I16 => scalar!(i16) as i64,
            Dtype::I8 => scalar!(i8) as i64,
            Dtype::Bool | Dtype::U8 => scalar!(u8) as i64,
            Dtype::F16 => half::f16::from_bits(scalar!(u16)).to_f32() as i64,
            Dtype::BF16 => half::bf16::from_bits(scalar!(u16)).to_f32() as i64,
            Dtype::F32 => scalar!(f32) as i64,
            Dtype::F64 => scalar!(f64) as i64,
        },
    )
}

fn pull_i32<D: BindingDevice>(
    context: &DeviceContext<D>,
    ids: &[i32],
    shape: &[i32],
) -> anyhow::Result<D::Tensor> {
    Ok(context.device().pull(
        HostTensor {
            bytes: &ids.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(),
            shape,
            dtype: Dtype::I32,
        }
        .into(),
        Default::default(),
    )?)
}
