//! Native DS4 scalar graph and bounded layer-major prefill. Packed-weight
//! extensions are explicit; this does not certify whole-model DS4 parity.

use super::{
    attention::Attention,
    config::Config,
    engram::{self, History, Layout, Tables},
    moe::Moe,
    projection,
    state::{State, view},
    weights::WeightPlan,
};
use crate::core::{
    capture::Capture,
    gguf_resident::{PackedTensor, ResidentWeights},
};
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Package, Tensor, quant};
use lisa_mlx::models::deepseek41 as gpu;
use lisa_mlx::{Array, Stream};

struct Hc {
    mixer: PackedTensor,
    scale: Array,
    base: Array,
    norm: Array,
}

impl Hc {
    fn load(weights: &ResidentWeights, layer: usize, kind: &str) -> Result<Self> {
        let get = |suffix: &str| {
            weights
                .tensors
                .get(&format!("blk.{layer}.{suffix}"))
                .cloned()
                .with_context(|| format!("missing DeepSeek HC tensor {suffix}"))
        };
        Ok(Self {
            mixer: get(&format!("hc_{kind}_fn.weight"))?,
            scale: projection::plain_view(&get(&format!("hc_{kind}_scale.weight"))?)?,
            base: projection::plain_view(&get(&format!("hc_{kind}_base.weight"))?)?,
            norm: projection::plain_view(&get(&format!("{kind}_norm.weight"))?)?,
        })
    }

    fn forward(&self, residual: &Array, previous: &Array, stream: &Stream) -> Result<gpu::HcInput> {
        let flat = gpu::rms_norm(&residual.reshape(&[1, 20480])?, None, 1e-20, false, stream)?;
        let mix = projection::linear(&self.mixer, &flat, false, stream)?;
        Ok(gpu::hc_input(
            &mix,
            residual,
            previous,
            &self.scale,
            &self.base,
            &self.norm,
            1e-6,
            1e-20,
            stream,
        )?)
    }
}

struct Engram {
    kv: PackedTensor,
    q: Array,
    k: Array,
}
struct Layer {
    hc_attention: Hc,
    attention: Attention,
    hc_ffn: Hc,
    moe: Moe,
    engram: Option<Engram>,
}

pub struct Graph {
    // Declared first: drains queued weight reads before any borrowed view is
    // dropped, including load errors after creating only some layers.
    _resident: ResidentWeights,
    pub config: Config,
    layers: Vec<Layer>,
    embedding: Tensor,
    output_norm: Array,
    output: PackedTensor,
    engram_layout: Layout,
    engram_tables: Tables,
    attention_scratch: gpu::DecodeAttentionScratch,
    stream: Stream,
}

pub struct Step {
    pub hidden: Array,
    pub residual: Array,
    pub logits: Option<Array>,
}

impl Graph {
    /// Validate every shard and tensor, then admit resident weights before
    /// mapping them. Engram tables remain on disk with uncached random reads.
    pub fn load(package: &Package) -> Result<Self> {
        let _trace = lisa_mlx::trace::span("deepseek41.load");
        let config = Config::from_header(&package.header)?;
        let engram_layout = Layout::from_header(&package.header)?;
        let plan = WeightPlan::inspect(package, &config, &engram_layout)?;
        let engram_tables = Tables::open(package, &engram_layout)?;
        let stream = Stream::gpu();
        let names: Vec<&str> = plan.resident.iter().map(String::as_str).collect();
        let resident = ResidentWeights::load(package, &names, &stream)?;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let engram = if engram::LAYERS.contains(&layer) {
                let get = |suffix: &str| {
                    resident
                        .tensors
                        .get(&format!("blk.{layer}.{suffix}"))
                        .cloned()
                        .with_context(|| format!("missing DeepSeek Engram tensor {suffix}"))
                };
                let (q, k) = if engram_layout.combined {
                    // The standard converter stores the exact F32 product of
                    // the two BF16 norm weights. DS4 multiplies q*k before h*k.
                    (
                        projection::plain_view(&get("engram_gate.weight")?)?,
                        Array::from_slice(&vec![1f32; 4 * 5120], &[4, 5120]),
                    )
                } else {
                    (
                        projection::plain_view(&get("engram_q_norm.weight")?)?,
                        projection::plain_view(&get("engram_k_norm.weight")?)?,
                    )
                };
                Some(Engram {
                    kv: get("engram_kv.weight")?,
                    q,
                    k,
                })
            } else {
                None
            };
            layers.push(Layer {
                hc_attention: Hc::load(&resident, layer, "attn")?,
                attention: Attention::load(&resident, layer, &config)?,
                hc_ffn: Hc::load(&resident, layer, "ffn")?,
                moe: Moe::load(&resident, layer, &config)?,
                engram,
            });
        }
        let output_norm = projection::plain_view(&resident.tensors["output_norm.weight"])?;
        let output = resident.tensors["output.weight"].clone();
        let attention_scratch = gpu::DecodeAttentionScratch::new(&stream)?;
        Ok(Self {
            _resident: resident,
            config,
            layers,
            embedding: package.tensor("token_embd.weight")?.clone(),
            output_norm,
            output,
            engram_layout,
            engram_tables,
            attention_scratch,
            stream,
        })
    }

    pub fn mapped_weight_bytes(&self) -> usize {
        self._resident.mapped_bytes
    }
    pub(super) fn output_weight(&self) -> &PackedTensor {
        &self.output
    }

    pub fn diagnostic_capture(layers: Vec<usize>) -> Capture {
        Capture {
            layers,
            preserve_dispatch: true,
            ..Capture::default()
        }
    }

    /// One token with the original layer order and submission cadence.
    pub fn step(
        &mut self,
        token: u32,
        state: &mut State,
        logits: bool,
        capture: Option<&Capture>,
    ) -> Result<Step> {
        Ok(self.prefill(&[token], state, logits, capture)?.remove(0))
    }

    /// Bounded layer-major row scheduling. Each row keeps the scalar attention
    /// reduction and causal cache writes; shared/routed projections dispatch
    /// together over up to eight rows using their original matvec trees.
    /// This is the small-row fallback schedule, not DS4's large GEMM prefill.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        state: &mut State,
        logits: bool,
        capture: Option<&Capture>,
    ) -> Result<Vec<Step>> {
        ensure!(
            state.valid,
            "DeepSeek session was invalidated by an earlier failure"
        );
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= 8
                && state
                    .position
                    .checked_add(tokens.len())
                    .is_some_and(|end| end <= state.capacity)
                && tokens
                    .iter()
                    .all(|&token| (token as usize) < self.config.vocab),
            "DeepSeek row count, token or position is out of range"
        );
        let start = state.position;
        let _trace = lisa_mlx::trace::span_detail(
            if tokens.len() == 1 {
                "deepseek41.step"
            } else {
                "deepseek41.prefill"
            },
            start as u64,
        );
        let mut history = state.history;
        let ids = self.engram_layout.hash(&mut history, tokens, None)?;
        let mut reads = self.engram_tables.prefetch(&ids)?;
        let result = self.encode_rows(tokens, state, logits, capture, &mut reads);
        // Every exit joins both disk jobs and drains GPU reads before reuse.
        drop(reads);
        let completion = self.stream.runtime().commands.flush_and_wait();
        match (result, completion) {
            (Ok(result), Ok(())) => {
                state.position = start + tokens.len();
                state.history = history;
                Ok(result)
            }
            (Err(error), _) => {
                state.valid = false;
                Err(error)
            }
            (_, Err(error)) => {
                state.valid = false;
                Err(error.into())
            }
        }
    }

    fn encode_rows(
        &mut self,
        tokens: &[u32],
        state: &mut State,
        logits: bool,
        capture: Option<&Capture>,
        reads: &mut engram::ReadAhead,
    ) -> Result<Vec<Step>> {
        struct Row {
            residual: Array,
            previous: Array,
            rope: Array,
            selected: Option<Array>,
            candidates: Option<gpu::IndexCandidates>,
        }
        let stream = &self.stream;
        let start = state.position;
        let mut rows = Vec::with_capacity(tokens.len());
        {
            let _trace = lisa_mlx::trace::span("deepseek41.embedding");
            let row_bytes = self.embedding.info.row_bytes()? as usize;
            let mut packed = vec![0u8; row_bytes * tokens.len()];
            self.embedding.read_rows(
                &tokens.iter().map(|&t| t as u64).collect::<Vec<_>>(),
                &mut packed,
            )?;
            for (t, packed) in packed.chunks_exact(row_bytes).enumerate() {
                let mut embedding = vec![0f32; 5120];
                quant::decode(self.embedding.info.kind, packed, &mut embedding)?;
                let repeated: Vec<f32> = (0..4).flat_map(|_| embedding.iter().copied()).collect();
                rows.push(Row {
                    residual: Array::from_slice(&repeated, &[1, 4, 5120]),
                    previous: Array::from_slice(&[1f32, 0.0, 0.0, 0.0], &[4]),
                    rope: gpu::rope_coefficient_table((start + t) as u32, stream)?,
                    selected: None,
                    candidates: None,
                });
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let _trace = lisa_mlx::trace::span_detail("deepseek41.layer", index as u64);
            let record = |row: usize, key: &str, x: &Array| {
                if let Some(capture) = capture.filter(|capture| capture.wants(index)) {
                    capture.store_ephemeral(&format!("p{}.l{index}.{key}", start + row), x);
                }
            };
            if let Some(engram) = &layer.engram {
                let _trace = lisa_mlx::trace::span("deepseek41.engram");
                let values = reads.wait(if index == 1 { 0 } else { 1 })?;
                let values = Array::from_slice(&values, &[tokens.len() as i32, 6144]);
                let kv = projection::linear(&engram.kv, &values, true, stream)?;
                for (t, row) in rows.iter_mut().enumerate() {
                    let kv = view(&kv, t * 5 * 5120, &[1, 5, 5120])?;
                    gpu::engram_in_place(
                        &mut row.residual,
                        &kv,
                        &engram.q,
                        &engram.k,
                        None,
                        1e-20,
                        stream,
                    )?;
                    record(t, "engram", &row.residual);
                }
            }
            let mut attention_inputs = Vec::with_capacity(rows.len());
            for (t, row) in rows.iter().enumerate() {
                let attention = {
                    let _trace = lisa_mlx::trace::span("deepseek41.hc.attention");
                    layer
                        .hc_attention
                        .forward(&row.residual, &row.previous, stream)?
                };
                record(t, "attention.norm", &attention.norm);
                record(t, "attention.split", &attention.split);
                attention_inputs.push(attention);
            }
            let projected = {
                let _trace = lisa_mlx::trace::span("deepseek41.attention.project");
                let norms = if attention_inputs.len() == 1 {
                    attention_inputs[0].norm.clone()
                } else {
                    Array::concatenate(
                        &attention_inputs.iter().map(|a| &a.norm).collect::<Vec<_>>(),
                        0,
                    )?
                };
                layer.attention.project(&norms, stream)?
            };
            let mut before_ffn = Vec::with_capacity(rows.len());
            for (t, (row, attention)) in rows.iter_mut().zip(attention_inputs).enumerate() {
                state.position = start + t;
                let block = {
                    let _trace = lisa_mlx::trace::span("deepseek41.attention");
                    layer
                        .attention
                        .forward(
                            &attention.norm,
                            projected.row(t)?,
                            state,
                            &mut row.selected,
                            &mut row.candidates,
                            &mut self.attention_scratch,
                            &row.rope,
                            capture,
                            stream,
                        )
                        .with_context(|| {
                            format!(
                                "DeepSeek attention layer {index} position {}",
                                state.position
                            )
                        })?
                };
                record(t, "attention.output", &block);
                let after_attention =
                    gpu::hc_expand(&block, None, &row.residual, &attention.split, false, stream)?;
                let ffn = {
                    let _trace = lisa_mlx::trace::span("deepseek41.hc.ffn");
                    layer.hc_ffn.forward(
                        &after_attention,
                        &view(&attention.split, 0, &[4])?,
                        stream,
                    )?
                };
                record(t, "ffn.norm", &ffn.norm);
                record(t, "ffn.split", &ffn.split);
                before_ffn.push((after_attention, ffn));
            }
            let moe = {
                let _trace = lisa_mlx::trace::span("deepseek41.moe");
                let norms = if before_ffn.len() == 1 {
                    before_ffn[0].1.norm.clone()
                } else {
                    Array::concatenate(
                        &before_ffn
                            .iter()
                            .map(|(_, ffn)| &ffn.norm)
                            .collect::<Vec<_>>(),
                        0,
                    )?
                };
                layer
                    .moe
                    .forward(&norms, stream)
                    .with_context(|| format!("DeepSeek FFN layer {index} position {start}"))?
            };
            for (t, (row, (after_attention, ffn))) in rows.iter_mut().zip(before_ffn).enumerate() {
                let routed = view(&moe.routed, t * 5120, &[1, 5120])?;
                let shared = view(&moe.shared, t * 5120, &[1, 5120])?;
                record(
                    t,
                    "ffn.router",
                    &view(&moe.routing.selected, t * 6, &[1, 6])?,
                );
                record(
                    t,
                    "ffn.weights",
                    &view(&moe.routing.weights, t * 6, &[1, 6])?,
                );
                record(t, "ffn.routed", &routed);
                record(t, "ffn.shared", &shared);
                row.residual = gpu::hc_expand(
                    &routed,
                    Some(&shared),
                    &after_attention,
                    &ffn.split,
                    true,
                    stream,
                )?;
                row.previous = view(&ffn.split, 0, &[4])?;
                record(t, "residual", &row.residual);
            }
            if tokens.len() > 1 {
                // DS4's layer sweep completes its command batch per layer.
                // This also bounds transient allocation without ordered reuse.
                stream.runtime().commands.flush_and_wait()?;
            } else if index + 1 < self.layers.len() && (index == 0 || (index + 1) % 2 == 0) {
                stream.runtime().commands.flush()?;
            }
        }
        let _trace = lisa_mlx::trace::span("deepseek41.head");
        let mut results = Vec::with_capacity(rows.len());
        for (t, row) in rows.into_iter().enumerate() {
            let hidden = gpu::hc_collapse(&row.residual, &row.previous, stream)?;
            let hidden = gpu::rms_norm(&hidden, Some(&self.output_norm), 1e-20, true, stream)?;
            let logits = if logits && t + 1 == tokens.len() {
                Some(projection::linear(&self.output, &hidden, false, stream)?)
            } else {
                None
            };
            if let Some(capture) = capture {
                capture.store_ephemeral(&format!("p{}.hidden", start + t), &hidden);
                if let Some(logits) = &logits {
                    capture.store_ephemeral(&format!("p{}.logits", start + t), logits);
                }
            }
            results.push(Step {
                hidden,
                residual: row.residual,
                logits,
            });
        }
        Ok(results)
    }

    pub fn history(&self, state: &State) -> History {
        state.history
    }
}
