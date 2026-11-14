//! Full attention for `qwen3_5`: GQA with a per-head output gate and learned
//! Q/K RMSNorm (no QSA indexer).

use crate::core::graph::PreparedGraph;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_models::qwen3_5::attention::Plan;

use crate::attention_cache::FullAttentionCache;
use crate::bindings::normalization::RmsNorm;
use crate::weights::affine::QuantizedLinear;
use crate::weights::source::TensorSource;

pub fn join_qkv<D: BindingDevice>(
    context: &DeviceContext<D>,
    q: &mut QuantizedLinear<D::Tensor>,
    k: &mut QuantizedLinear<D::Tensor>,
    v: &mut QuantizedLinear<D::Tensor>,
) -> anyhow::Result<Option<QuantizedLinear<D::Tensor>>> {
    let ps = [&*q, &*k, &*v];
    let Some(graph) = lisa_models::qwen3_5::attention::join_projection(
        ps.map(|p| p.dims_out() as i32),
        ps.map(|p| p.weight.dim(1) * (32 / p.bits)),
        ps.map(|p| (p.group_size, p.bits)),
    ) else {
        return Ok(None);
    };
    let plan = PreparedGraph::new(context.device().clone(), graph, vec![])?;
    let inputs = ps
        .into_iter()
        .flat_map(|p| [&p.weight, &p.scales, &p.biases])
        .collect::<Vec<_>>();
    let mut outputs = plan.run(&inputs, &mut [])?.into_iter();
    let mut projection = || QuantizedLinear {
        weight: outputs.next().unwrap(),
        scales: outputs.next().unwrap(),
        biases: outputs.next().unwrap(),
        group_size: q.group_size,
        bits: q.bits,
    };
    let joined = projection();
    let views = [projection(), projection(), projection()];
    for (p, view) in [q, k, v].into_iter().zip(views) {
        *p = view;
    }
    Ok(Some(joined))
}
pub struct Qwen35Attention<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub q_proj: QuantizedLinear<D::Tensor>,
    /// Shared packed storage for Q/gate, K and V. Individual projections are
    /// row views, so grouping adds no second resident copy of the weights.
    pub qkv_proj: Option<QuantizedLinear<D::Tensor>>,
    /// Decode-only merged arm (specs/16 phase 2): q_proj's rows gathered and
    /// re-laid-out as `[Q | G]` (all q rows first, then all gate rows), so the
    /// decode qmv emits q and gate as two CONTIGUOUS row halves and the
    /// s==1 de-interleave `.contiguous()` copies disappear. Exact on the GEMV
    /// paths: each output row is the same dot over the same K values with the
    /// same packed weights/scales — the row gather preserves each row's
    /// packing verbatim (same argument as GDN's `in_proj_all` / MLP's
    /// `gate_up_proj`). `None` when ungated or gs/bits mismatch.
    pub q_gate_proj: Option<QuantizedLinear<D::Tensor>>,
    pub k_proj: QuantizedLinear<D::Tensor>,
    pub v_proj: QuantizedLinear<D::Tensor>,
    pub o_proj: QuantizedLinear<D::Tensor>,
    pub q_norm: RmsNorm<D::Tensor>,
    pub k_norm: RmsNorm<D::Tensor>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub gated: bool,
    pub plans: std::cell::RefCell<Vec<(Plan, PreparedGraph<D>)>>,
}

impl<D: BindingDevice> Qwen35Attention<D> {
    /// Whether to build the merged q|gate retile at load. **OFF by default: RAM
    /// wins.** The retile is a permuted COPY of `q_proj`'s rows resident across
    /// every full-attention layer, and the standing goal is to match the
    /// reference engine's footprint to the MB. The trade is measured, not
    /// guessed — removing it buys **0.77 GB** and costs **+3.56 ms on the serial
    /// step (38.69 -> 42.25, +9.2 %)** — and it is accepted deliberately: the
    /// fallback arm (`q_proj.forward` plus a per-step de-interleave) is exact, so
    /// nothing but speed moves. `LISA_QGATE_RETILE=1` restores it when a run is
    /// about latency instead of footprint.
    fn qgate_retile_enabled() -> bool {
        matches!(
            std::env::var("LISA_QGATE_RETILE").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
    }

    pub fn load<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        gated: bool,
        group_size: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        let mut q_proj = QuantizedLinear::load(src, prefix, "q_proj")?;
        q_proj.set_quant(group_size, bits);
        let mut k_proj = QuantizedLinear::load(src, prefix, "k_proj")?;
        k_proj.set_quant(group_size, bits);
        let mut v_proj = QuantizedLinear::load(src, prefix, "v_proj")?;
        v_proj.set_quant(group_size, bits);
        let qkv_proj = join_qkv(&context, &mut q_proj, &mut k_proj, &mut v_proj)?;
        let mut o_proj = QuantizedLinear::load(src, prefix, "o_proj")?;
        o_proj.set_quant(group_size, bits);
        // Merged q|gate projection (specs/16 phase 2): gather q_proj's rows
        // from the per-head [q|gate] interleave into a [Q | G] row-concat.
        // One-time at load; decode then needs ZERO de-interleave copies.
        let q_gate_proj = if gated && Self::qgate_retile_enabled() {
            let plan = PreparedGraph::new(
                context.device().clone(),
                lisa_models::qwen3_5::attention::retile_projection(heads as i32, head_dim as i32),
                vec![],
            )?;
            let mut out = plan
                .run(&[&q_proj.weight, &q_proj.scales, &q_proj.biases], &mut [])?
                .into_iter();
            Some(QuantizedLinear {
                weight: out.next().unwrap(),
                scales: out.next().unwrap(),
                biases: out.next().unwrap(),
                group_size: q_proj.group_size,
                bits: q_proj.bits,
            })
        } else {
            None
        };
        Ok(Self {
            context: context.clone(),
            q_proj,
            qkv_proj,
            q_gate_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RmsNorm::load_on(&context, src, &format!("{prefix}.q_norm"), eps, None)?,
            k_norm: RmsNorm::load_on(&context, src, &format!("{prefix}.k_norm"), eps, None)?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            gated,
            plans: Default::default(),
        })
    }

    pub fn description(
        &self,
        batch: i32,
        rows: i32,
        rotary: i32,
        frequencies_bf16: bool,
        cache: Option<&FullAttentionCache<D>>,
    ) -> Plan {
        let cache_layout = cache.as_deref().map(|c| c.append_plan(batch, rows));
        Plan {
            batch: batch,
            rows: rows,
            heads: self.heads as i32,
            key_heads: self.kv_heads as i32,
            width: self.head_dim as i32,
            rotary: rotary,
            frequencies_bf16: frequencies_bf16,
            epsilon: self.q_norm.eps.to_bits(),
            scale: self.scale.to_bits(),
            gated: self.gated,
            quant: [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
                .map(|p| (p.group_size, p.bits)),
            joined: self.qkv_proj.is_some(),
            retiled: self.q_gate_proj.is_some(),
            cache: cache_layout,
            mask_length: cache
                .as_deref()
                .and_then(|c| c.attn_mask.as_ref())
                .map(|m| m.dim(-1)),
            next_positions: cache.as_deref().and_then(|c| c.next_pos.clone()),
        }
    }
    pub fn weights(&self) -> Vec<D::Tensor> {
        let mut weights = Vec::new();
        for p in [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj] {
            weights.extend([p.weight.clone(), p.scales.clone(), p.biases.clone()]);
        }
        weights.extend([self.q_norm.weight.clone(), self.k_norm.weight.clone()]);
        for p in [self.qkv_proj.as_ref(), self.q_gate_proj.as_ref()]
            .into_iter()
            .flatten()
        {
            weights.extend([p.weight.clone(), p.scales.clone(), p.biases.clone()]);
        }
        weights
    }

    /// Bind model inputs and install the cache handles after successful execution.
    pub fn forward(
        &self,
        x: &D::Tensor,
        cos: &D::Tensor,
        sin: &D::Tensor,
        mut cache: Option<&mut FullAttentionCache<D>>,
        _offset: usize,
    ) -> anyhow::Result<D::Tensor> {
        let spec = self.description(
            x.dim(0),
            x.dim(1),
            cos.dim(-1),
            self.context.device().dtype(cos) == Some(lisa_backend::ScalarType::BF16)
                && self.context.device().dtype(sin) == Some(lisa_backend::ScalarType::BF16),
            cache.as_deref(),
        );
        let mut plans = self.plans.borrow_mut();
        let index = if let Some(i) = plans.iter().position(|(p, _)| p == &spec) {
            i
        } else {
            let weights = self.weights();
            let plan = PreparedGraph::new(
                self.context.device().clone(),
                spec.graph().map_err(anyhow::Error::msg)?,
                weights,
            )?;
            if plans.len() == 16 {
                plans.remove(0);
            }
            plans.push((spec.clone(), plan));
            plans.len() - 1
        };
        let mut inputs = vec![x, cos, sin];
        if let Some(c) = cache.as_deref() {
            if let (Some(k), Some(v)) = (&c.keys, &c.values) {
                inputs.extend([k, v]);
            }
            if let Some(mask) = &c.attn_mask {
                inputs.push(mask);
            }
        }
        let mut out = plans[index].1.run(&inputs, &mut [])?.into_iter();
        let activation = out.next().unwrap();
        if let (Some(c), Some(layout)) = (cache.as_deref_mut(), spec.cache.as_ref()) {
            let (offset, capacity, _) = layout.layout().map_err(anyhow::Error::msg)?;
            c.keys = out.next();
            c.values = out.next();
            c.offset = offset;
            c.capacity = capacity;
        }
        Ok(activation)
    }
}
