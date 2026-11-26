//! Native reference scalar graph and bounded layer-major prefill. Packed-weight
//! extensions are explicit; this does not certify whole-model reference parity.

use crate::bindings::shared::capture::Capture;
use crate::bindings::{
    BindingDevice,
    deepseek41::attention::{self as attention, Attention},
    deepseek41::engram::{self as engram, History, Layout, Tables},
    deepseek41::moe::Moe,
    deepseek41::projection,
    deepseek41::state::State,
};
use crate::{
    core::graph::{PlanCache, PreparedGraph},
    execution::DeviceContext,
    storage::view,
    weights::prepared::{PreparedTensor, PreparedWeights},
};
use anyhow::{Context, Result, ensure};
use lisa_backend::{HostTensor, ManagedDevice, ScalarType};
use lisa_models::deepseek41::config::Config;
use lisa_models::deepseek41::tower as model;

use crate::bindings::deepseek41::hyper::{Hc, ResidualPlans};

struct Engram<D: BindingDevice + ManagedDevice> {
    kv: PreparedTensor<D::Tensor>,
    q: D::Tensor,
    k: D::Tensor,
}
struct Layer<D: BindingDevice + ManagedDevice> {
    hc_attention: Hc<D>,
    attention: Attention<D>,
    hc_ffn: Hc<D>,
    moe: Moe<D>,
    engram: Option<Engram<D>>,
}

pub struct Graph<D: BindingDevice + ManagedDevice> {
    // Declared first: drains queued weight reads before any borrowed view is
    // dropped, including load errors after creating only some layers.
    _resident: PreparedWeights<D::Tensor, D::Reservation>,
    pub config: Config,
    layers: Vec<Layer<D>>,
    embedding: Box<dyn engram::EmbeddingRows>,
    output_norm: D::Tensor,
    output: PreparedTensor<D::Tensor>,
    engram_layout: Layout,
    engram_tables: Box<dyn Tables>,
    attention_scratch: attention::Scratch<D>,
    residual_plans: ResidualPlans<D>,
    engram_gate: PreparedGraph<D>,
    context: DeviceContext<D>,
}

pub struct Step<T> {
    pub hidden: T,
    pub residual: T,
    pub logits: Option<T>,
}

impl<D: BindingDevice + ManagedDevice> Graph<D> {
    /// Bind imported tensors and deferred rows without checkpoint access.
    pub fn from_prepared(
        context: DeviceContext<D>,
        config: Config,
        engram_layout: Layout,
        engram_tables: Box<dyn Tables>,
        embedding: Box<dyn engram::EmbeddingRows>,
        resident: PreparedWeights<D::Tensor, D::Reservation>,
    ) -> Result<Self> {
        let device = context.device();
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let engram = model::engram_weights(layer, engram_layout.combined)
                .map(|names| -> Result<_> {
                    let get = |name: &str| {
                        resident
                            .tensors
                            .get(name)
                            .cloned()
                            .with_context(|| format!("missing {name}"))
                    };
                    let q = projection::plain_view(&get(&names.query)?)?;
                    let k = if let Some(name) = names.key {
                        projection::plain_view(&get(&name)?)?
                    } else {
                        let (values, shape) = model::identity_gate();
                        pull_f32(device, &values, &shape)?
                    };
                    Ok(Engram {
                        kv: get(&names.kv)?,
                        q,
                        k,
                    })
                })
                .transpose()?;
            layers.push(Layer {
                hc_attention: Hc::load(&resident, layer, model::ATTENTION_HC, device)?,
                attention: Attention::load(&resident, layer, &config, device)?,
                hc_ffn: Hc::load(&resident, layer, model::FFN_HC, device)?,
                moe: Moe::load(&resident, layer, config.experts, device)?,
                engram,
            });
        }
        let output_norm = projection::plain_view(&resident.tensors[model::OUTPUT_NORM])?;
        let output = resident.tensors[model::OUTPUT].clone();
        let attention_scratch = attention::Scratch::new(device)?;
        Ok(Self {
            _resident: resident,
            config,
            layers,
            embedding,
            output_norm,
            output,
            engram_layout,
            engram_tables,
            attention_scratch,
            residual_plans: ResidualPlans::new(device)?,
            engram_gate: PreparedGraph::new(
                device.clone(),
                lisa_models::deepseek41::engram::gate(),
                vec![],
            )?,
            context,
        })
    }

    pub fn mapped_weight_bytes(&self) -> usize {
        self._resident.mapped_bytes
    }
    pub fn output_weight(&self) -> &PreparedTensor<D::Tensor> {
        &self.output
    }

    pub fn diagnostic_capture(layers: Vec<usize>) -> Capture<D::Tensor> {
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
        state: &mut State<D>,
        logits: bool,
        capture: Option<&Capture<D::Tensor>>,
    ) -> Result<Step<D::Tensor>> {
        Ok(self.prefill(&[token], state, logits, capture)?.remove(0))
    }

    /// Bounded layer-major row scheduling. Each row keeps the scalar attention
    /// reduction and causal cache writes; shared/routed projections dispatch
    /// together over up to eight rows using their original matvec trees.
    /// This is the small-row fallback schedule, not the reference's large GEMM prefill.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        state: &mut State<D>,
        logits: bool,
        capture: Option<&Capture<D::Tensor>>,
    ) -> Result<Vec<Step<D::Tensor>>> {
        ensure!(
            state.valid,
            "DeepSeek session was invalidated by an earlier failure"
        );
        ensure!(
            model::validate_tokens(tokens, self.config.vocab, state.position, state.capacity),
            "DeepSeek row count, token or position is out of range"
        );
        let start = state.position;
        let _trace = self.context.trace_detail(
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
        let result = self.encode_rows(tokens, state, logits, capture, reads.as_mut());
        // Every exit joins both disk jobs and drains GPU reads before reuse.
        drop(reads);
        let completion = self.context.device().submit(true);
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
                Err(error)
            }
        }
    }

    fn encode_rows(
        &mut self,
        tokens: &[u32],
        state: &mut State<D>,
        logits: bool,
        capture: Option<&Capture<D::Tensor>>,
        reads: &mut dyn engram::ReadAhead,
    ) -> Result<Vec<Step<D::Tensor>>> {
        use lisa_models::deepseek41::tower::{self as model, Step as Op};
        struct Row<T> {
            residual: T,
            previous: T,
            rope: T,
            selected: Option<T>,
            candidates: Option<T>,
        }
        let stream = &self.context;
        let start = state.position;
        let mut rows = Vec::with_capacity(tokens.len());
        {
            let _trace = self.context.trace("deepseek41.embedding");
            for (t, embedding) in self.embedding.rows(tokens)?.enumerate() {
                let embedding = embedding?;
                rows.push(Row {
                    residual: pull_f32(
                        self.context.device(),
                        &model::embedding(&embedding)?,
                        &[1, model::STREAMS as i32, model::WIDTH as i32],
                    )?,
                    previous: pull_f32(
                        self.context.device(),
                        &model::INITIAL_MIX,
                        &[model::STREAMS as i32],
                    )?,
                    rope: attention::rope_table((start + t) as u32, stream.device())?,
                    selected: None,
                    candidates: None,
                });
            }
        }
        let mut index = 0;
        let mut layer_trace = None;
        let mut engram_kv = None;
        let mut attention_inputs: Vec<
            Option<crate::bindings::deepseek41::hyper::Input<D::Tensor>>,
        > = Vec::new();
        let mut projected = None;
        let mut block = None;
        let mut after_attention = None;
        let mut before_ffn = Vec::new();
        let mut moe = None;
        let mut results = Vec::with_capacity(rows.len());
        let mut head_trace = None;
        for op in model::schedule(self.layers.len(), tokens.len(), logits)? {
            let record = |row: usize, key: &str, x: &D::Tensor| {
                if let Some(c) = capture.filter(|c| c.wants(index)) {
                    c.store_ephemeral(&format!("p{}.l{index}.{key}", start + row), x);
                }
            };
            match op {
                Op::Layer(i) => {
                    index = i;
                    layer_trace = Some(self.context.trace_detail("deepseek41.layer", index as u64));
                }
                Op::EngramProject(table) => {
                    let _trace = self.context.trace("deepseek41.engram");
                    let weights = self.layers[index]
                        .engram
                        .as_ref()
                        .context("missing bound Engram block")?;
                    let values = reads.wait(table)?;
                    let values = pull_f32(
                        self.context.device(),
                        &values,
                        &[tokens.len() as i32, (engram::HEADS * engram::WIDTH) as i32],
                    )?;
                    engram_kv = Some(projection::linear(&weights.kv, &values, true, stream)?);
                }
                Op::EngramRow(t) => {
                    let weights = self.layers[index]
                        .engram
                        .as_ref()
                        .context("missing bound Engram block")?;
                    let kv = view(
                        self.context.device(),
                        engram_kv.as_ref().context("missing Engram projection")?,
                        t * 5 * model::WIDTH,
                        &[1, 5, model::WIDTH as i32],
                    )?;
                    rows[t].residual = self
                        .engram_gate
                        .run(&[&rows[t].residual, &kv, &weights.q, &weights.k], &mut [])?
                        .into_one()
                        .map_err(anyhow::Error::msg)?;
                    record(t, "engram", &rows[t].residual);
                }
                Op::AttentionInput(t) => {
                    let _trace = self.context.trace("deepseek41.hc.attention");
                    let row = &rows[t];
                    let input = self.layers[index]
                        .hc_attention
                        .forward(&row.residual, &row.previous)?;
                    record(t, "attention.norm", &input.norm);
                    record(t, "attention.split", &input.split);
                    attention_inputs.push(Some(input));
                }
                Op::AttentionProject => {
                    let _trace = self.context.trace("deepseek41.attention.project");
                    let inputs = attention_inputs
                        .iter()
                        .map(|a| &a.as_ref().unwrap().norm)
                        .collect::<Vec<_>>();
                    let norms = assembly(Assembly::Join(inputs.len()), &inputs, stream)?;
                    projected = Some(
                        self.layers[index]
                            .attention
                            .project(&norms, stream.device())?,
                    );
                }
                Op::Attention(t) => {
                    state.position = start + t;
                    let _trace = self.context.trace("deepseek41.attention");
                    let row = &mut rows[t];
                    let value = self.layers[index]
                        .attention
                        .forward(
                            &attention_inputs[t].as_ref().unwrap().norm,
                            projected
                                .as_ref()
                                .context("missing attention projections")?
                                .row(t)?,
                            state,
                            &mut row.selected,
                            &mut row.candidates,
                            &mut self.attention_scratch,
                            &row.rope,
                            capture,
                            stream.device(),
                        )
                        .with_context(|| {
                            format!(
                                "DeepSeek attention layer {index} position {}",
                                state.position
                            )
                        })?;
                    record(t, "attention.output", &value);
                    block = Some(value);
                }
                Op::AttentionResidual(t) => {
                    after_attention = Some(self.residual_plans.attention(
                        &block.take().context("missing attention block")?,
                        &rows[t].residual,
                        &attention_inputs[t].as_ref().unwrap().split,
                    )?);
                }
                Op::FfnInput(t) => {
                    let _trace = self.context.trace("deepseek41.hc.ffn");
                    let attention = attention_inputs[t]
                        .take()
                        .context("missing attention state")?;
                    let after = after_attention
                        .take()
                        .context("missing attention residual")?;
                    let input = self.layers[index].hc_ffn.forward(
                        &after,
                        &view(
                            self.context.device(),
                            &attention.split,
                            0,
                            &[model::STREAMS as i32],
                        )?,
                    )?;
                    record(t, "ffn.norm", &input.norm);
                    record(t, "ffn.split", &input.split);
                    before_ffn.push(Some((after, input)));
                }
                Op::Experts => {
                    let _trace = self.context.trace("deepseek41.moe");
                    let inputs = before_ffn
                        .iter()
                        .map(|x| &x.as_ref().unwrap().1.norm)
                        .collect::<Vec<_>>();
                    let norms = assembly(Assembly::Join(inputs.len()), &inputs, stream)?;
                    moe =
                        Some(self.layers[index].moe.forward(&norms).with_context(|| {
                            format!("DeepSeek FFN layer {index} position {start}")
                        })?);
                }
                Op::FfnResidual(t) => {
                    let experts = moe.as_ref().context("missing expert output")?;
                    let (after, ffn) = before_ffn[t].take().context("missing FFN input")?;
                    let routed = view(
                        self.context.device(),
                        &experts.routed,
                        t * model::WIDTH,
                        &[1, model::WIDTH as i32],
                    )?;
                    let shared = view(
                        self.context.device(),
                        &experts.shared,
                        t * model::WIDTH,
                        &[1, model::WIDTH as i32],
                    )?;
                    record(
                        t,
                        "ffn.router",
                        &view(
                            self.context.device(),
                            &experts.routing.selected,
                            t * 6,
                            &[1, 6],
                        )?,
                    );
                    record(
                        t,
                        "ffn.weights",
                        &view(
                            self.context.device(),
                            &experts.routing.weights,
                            t * 6,
                            &[1, 6],
                        )?,
                    );
                    record(t, "ffn.routed", &routed);
                    record(t, "ffn.shared", &shared);
                    rows[t].residual = self
                        .residual_plans
                        .ffn(&routed, &shared, &after, &ffn.split)?;
                    rows[t].previous = view(
                        self.context.device(),
                        &ffn.split,
                        0,
                        &[model::STREAMS as i32],
                    )?;
                    record(t, "residual", &rows[t].residual);
                }
                Op::Submit { wait } => {
                    if wait {
                        stream.device().submit(true)?;
                    } else {
                        stream.device().submit(false)?;
                    }
                }
                Op::EndLayer => {
                    attention_inputs.clear();
                    before_ffn.clear();
                    projected = None;
                    moe = None;
                    engram_kv = None;
                    layer_trace = None;
                }
                Op::Head { row: t, logits } => {
                    if head_trace.is_none() {
                        head_trace = Some(self.context.trace("deepseek41.head"));
                    }
                    let row = &rows[t];
                    let hidden = self.residual_plans.head(
                        &row.residual,
                        &row.previous,
                        &self.output_norm,
                    )?;
                    let logits = if logits {
                        Some(projection::linear(&self.output, &hidden, false, stream)?)
                    } else {
                        None
                    };
                    if let Some(c) = capture {
                        c.store_ephemeral(&format!("p{}.hidden", start + t), &hidden);
                        if let Some(x) = &logits {
                            c.store_ephemeral(&format!("p{}.logits", start + t), x);
                        }
                    }
                    results.push(Step {
                        hidden,
                        residual: row.residual.clone(),
                        logits,
                    });
                }
            }
        }
        drop(layer_trace);
        drop(head_trace);
        Ok(results)
    }

    pub fn history(&self, state: &State<D>) -> History {
        state.history
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Assembly {
    Join(usize),
    Reshape(Vec<i32>),
    Output(usize, i32),
    Ids,
    Head(lisa_models::deepseek41::projection::Matrix, Vec<i32>),
}
pub fn assembly<D: BindingDevice>(
    key: Assembly,
    inputs: &[&D::Tensor],
    context: &DeviceContext<D>,
) -> Result<D::Tensor> {
    context.with_slot(
        "deepseek41.assembly",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 64)),
        |p| {
            let recipe = key.clone();
            p.borrow_mut()
                .try_get_or_prepare(key, || {
                    Ok(match recipe {
                        Assembly::Join(n) => model::join(n),
                        Assembly::Reshape(shape) => model::reshape(shape),
                        Assembly::Output(n, width) => model::concatenate_output(n, width),
                        Assembly::Ids => model::ids(),
                        Assembly::Head(matrix, shape) => model::head(matrix, &shape)?,
                    })
                })?
                .run(inputs, &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
fn pull_f32<D: lisa_backend::Device>(
    device: &D,
    values: &[f32],
    shape: &[i32],
) -> Result<D::Tensor> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    Ok(device.pull(
        HostTensor {
            bytes: &bytes,
            shape,
            dtype: ScalarType::F32,
        }
        .into(),
        Default::default(),
    )?)
}
