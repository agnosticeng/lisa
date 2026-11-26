//! The PLE (per-layer embedding) block and the sharded n-gram table.

use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;

use crate::bindings::shared::normalization::{RmsNorm, load_bf16};
use crate::weights::affine::{BITS, GROUP_SIZE, QuantizedLinear};
use crate::weights::source::TensorSource;

// ---------------------------------------------------------------------------
// Host-side hash constants (rebuilt from configuration — the checkpoint
// copies are never used).
// ---------------------------------------------------------------------------

pub use lisa_models::qwen4::ngram::NgramConstants;

// ---------------------------------------------------------------------------
// Abstract row source; packed storage remains outside execution.
// ---------------------------------------------------------------------------

/// Lazily supplies original affine row tensors; storage remains loader-owned.
pub trait NgramRows<D: BindingDevice>: Send + Sync {
    fn set_rows_per_shard(&mut self, rows: usize) -> anyhow::Result<()>;
    fn gather(
        &self,
        context: &DeviceContext<D>,
        ids: &[i64],
    ) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)>;
}
pub type RowSource<D> = std::sync::Arc<std::sync::RwLock<dyn NgramRows<D>>>;

/// The n-gram embedding: host hash -> row gather -> dequantize.
pub struct NgramEmbedding<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub constants: NgramConstants,
    pub table: RowSource<D>,
    pub ngram_heads: usize,
    pub eos_token_id: i64,
    pub head_dimensions: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
}

impl<D: BindingDevice> NgramEmbedding<D> {
    fn embedding_ids(&self, ids: &[Vec<i64>], history: &[Vec<i64>]) -> anyhow::Result<Vec<i64>> {
        self.constants.batch_ids(
            ids,
            history,
            self.eos_token_id,
            self.ngram_size,
            self.heads_per_ngram,
        )
    }

    /// Forward: ids [B, S], previous_context [B, ctx] -> [B, S, 2560].
    pub fn forward(
        &self,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
    ) -> anyhow::Result<D::Tensor> {
        let b = ids.len();
        let s = ids.first().map_or(0, Vec::len);
        let all_ids = self.embedding_ids(ids, previous_context)?;
        let (packed, scales, biases) = {
            let table = self.table.read().unwrap();
            table.gather(&self.context, &all_ids)?
        };
        self.context.with_slot(
            "qwen4.ple.plans",
            || {
                std::cell::RefCell::new(crate::core::graph::PlanCache::new(
                    self.context.device().clone(),
                    64,
                ))
            },
            |cache| {
                cache
                    .borrow_mut()
                    .get_or_prepare(Key::Embedding(b as i32, s as i32), || {
                        lisa_models::qwen4::ple::embedding(b as i32, s as i32, GROUP_SIZE, BITS)
                    })?
                    .run(&[&packed, &scales, &biases], &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)
            },
        )
    }
}

/// PLE layer: n-gram embedding, gated against the hidden stream, plus a
/// dilated short convolution.
pub struct PleLayer<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub embedding: NgramEmbedding<D>,
    pub key_proj: QuantizedLinear<D::Tensor>,
    pub value_proj: QuantizedLinear<D::Tensor>,
    pub norm_key: RmsNorm<D::Tensor>,
    pub norm_query: RmsNorm<D::Tensor>,
    pub norm_conv: RmsNorm<D::Tensor>,
    pub conv1d_weight: D::Tensor, // [wide, 4, 1]
    pub hidden_size: usize,
    pub hc_count: usize,
    pub dilation: usize,
    pub short_conv_state_length: usize,
    /// Debug: last computed embedding rows (pre-projection).
    pub last_embed: Option<D::Tensor>,
    /// Debug: last computed gated stream.
    pub last_gated: Option<D::Tensor>,
}

impl<D: BindingDevice> PleLayer<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        eps: f32,
        hidden_size: usize,
        hc_count: usize,
        ngram_size: usize,
        heads_per_ngram: usize,
        ple_embed_dim: usize,
        ple_conv_kernel_size: usize,
        ple_layer_index: usize,
        seed: i64,
        vocab_size: usize,
        ngram_vocab_size_base: usize,
        divisible_by: usize,
        split_parts: usize,
        table: RowSource<D>,
    ) -> anyhow::Result<Self> {
        let names = lisa_models::qwen4::ple::names(prefix);
        let constants = NgramConstants::new(
            ngram_size,
            heads_per_ngram,
            vocab_size,
            ngram_vocab_size_base,
            divisible_by,
            split_parts,
            seed,
            ple_layer_index,
        );
        table
            .write()
            .unwrap()
            .set_rows_per_shard(constants.rows_per_shard)?;

        let conv1d_weight = load_bf16(&context, src, &names.convolution)?;
        let prepare = crate::core::graph::PreparedGraph::new(
            context.device().clone(),
            lisa_models::qwen4::ple::prepare_convolution(conv1d_weight.shape()),
            vec![],
        )?;
        let conv1d_weight = prepare
            .run(&[&conv1d_weight], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            context: context.clone(),
            embedding: NgramEmbedding {
                context: context.clone(),
                constants,
                table,
                ngram_heads: (ngram_size - 1) * heads_per_ngram,
                eos_token_id: 0, // set by the model from config
                head_dimensions: ple_embed_dim / ((ngram_size - 1) * heads_per_ngram),
                ngram_size,
                heads_per_ngram,
            },
            key_proj: QuantizedLinear::load(src, prefix, names.key)?,
            value_proj: QuantizedLinear::load(src, prefix, names.value)?,
            norm_key: RmsNorm::load_on(&context, src, &names.norm_key, eps, Some(hidden_size))?,
            norm_query: RmsNorm::load_on(&context, src, &names.norm_query, eps, Some(hidden_size))?,
            norm_conv: RmsNorm::load_on(
                &context,
                src,
                &names.norm_convolution,
                eps,
                Some(hidden_size),
            )?,
            conv1d_weight,
            hidden_size,
            hc_count,
            dilation: ngram_size,
            short_conv_state_length: (ple_conv_kernel_size - 1) * ngram_size,
            last_embed: None,
            last_gated: None,
        })
    }

    pub fn set_eos(&mut self, eos: i64) {
        self.embedding.eos_token_id = eos;
    }

    pub fn forward(
        &mut self,
        hidden: &D::Tensor,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
        conv_state: &mut Option<D::Tensor>,
        capture: bool,
        capture_full: &mut Option<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        self.forward_with_capture_law(
            hidden,
            ids,
            previous_context,
            conv_state,
            capture,
            capture_full,
            false,
        )
    }

    /// Capture a wide token window with the single-token projection law.
    pub fn forward_captured(
        &mut self,
        hidden: &D::Tensor,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
        conv_state: &mut Option<D::Tensor>,
        capture: bool,
        capture_full: &mut Option<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        self.forward_with_capture_law(
            hidden,
            ids,
            previous_context,
            conv_state,
            capture,
            capture_full,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_capture_law(
        &mut self,
        hidden: &D::Tensor,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
        conv_state: &mut Option<D::Tensor>,
        capture: bool,
        capture_full: &mut Option<D::Tensor>,
        serial_capture: bool,
    ) -> anyhow::Result<D::Tensor> {
        let embedded = self.embedding.forward(ids, previous_context)?;
        let spec = lisa_models::qwen4::ple::Spec {
            hidden: self.hidden_size as i32,
            streams: self.hc_count as i32,
            dilation: self.dilation as i32,
            state_rows: self.short_conv_state_length as i32,
            epsilon: self.norm_key.eps.to_bits(),
            key: (self.key_proj.group_size, self.key_proj.bits),
            value: (self.value_proj.group_size, self.value_proj.bits),
        };
        let mut inputs = vec![
            hidden,
            &embedded,
            &self.key_proj.weight,
            &self.key_proj.scales,
            &self.key_proj.biases,
            &self.value_proj.weight,
            &self.value_proj.scales,
            &self.value_proj.biases,
            &self.norm_key.weight,
            &self.norm_query.weight,
            &self.norm_conv.weight,
            &self.conv1d_weight,
        ];
        let has_state = conv_state.is_some();
        if let Some(state) = conv_state.as_ref() {
            inputs.push(state);
        }
        let mut output = self
            .context
            .with_slot(
                "qwen4.ple.plans",
                || {
                    std::cell::RefCell::new(crate::core::graph::PlanCache::new(
                        self.context.device().clone(),
                        64,
                    ))
                },
                |cache| {
                    cache
                        .borrow_mut()
                        .try_get_or_prepare(
                            Key::Forward(spec, hidden.shape().to_vec(), has_state, serial_capture),
                            || {
                                if serial_capture {
                                    spec.graph_captured(hidden.shape(), has_state)
                                } else {
                                    spec.graph(hidden.shape(), has_state)
                                }
                            },
                        )?
                        .run(&inputs, &mut [])
                },
            )?
            .into_iter();
        let result = output.next().unwrap();
        self.last_embed = output.next();
        self.last_gated = output.next();
        *conv_state = output.next();
        if capture {
            *capture_full = output.next();
        }
        Ok(result)
    }
}
#[derive(PartialEq)]
enum Key {
    Forward(lisa_models::qwen4::ple::Spec, Vec<i32>, bool, bool),
    Embedding(i32, i32),
}
