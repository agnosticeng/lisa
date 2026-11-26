use crate::bindings::shared::normalization::{RmsNorm, Rotary};
use crate::cache::attention::FullAttentionCache;
use crate::weights::{affine::QuantizedLinear, source::TensorSource};
// Qwen4 attention weight/cache bindings. Computation is defined in lisa-models.
use crate::bindings::qwen4::indexer::QsaIndexer;
use crate::core::graph::{PlanCache, PreparedGraph};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_backend::{BlockAttentionKind, KernelKind};
use lisa_models::qwen4::attention::{self as model, Plan, Selection};
pub struct Attention<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub q_proj: QuantizedLinear<D::Tensor>,
    pub k_proj: QuantizedLinear<D::Tensor>,
    pub v_proj: QuantizedLinear<D::Tensor>,
    pub o_proj: QuantizedLinear<D::Tensor>,
    pub q_norm: RmsNorm<D::Tensor>,
    pub k_norm: RmsNorm<D::Tensor>,
    pub indexer: QsaIndexer<D>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub block_kernel: bool,
}
fn error(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::Error::msg(e.to_string())
}
impl<D: BindingDevice> Attention<D> {
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<Self> {
        let n = model::names(prefix);
        let mut q = QuantizedLinear::load(src, prefix, n.projections[0])?;
        let prep = PreparedGraph::new(
            context.device().clone(),
            model::retile(heads as i32, head_dim as i32),
            vec![],
        )?;
        let mut out = prep
            .run(&[&q.weight, &q.scales, &q.biases], &mut [])?
            .into_iter();
        q.weight = out.next().unwrap();
        q.scales = out.next().unwrap();
        q.biases = out.next().unwrap();
        Ok(Self {
            context: context.clone(),
            q_proj: q,
            k_proj: QuantizedLinear::load(src, prefix, n.projections[1])?,
            v_proj: QuantizedLinear::load(src, prefix, n.projections[2])?,
            o_proj: QuantizedLinear::load(src, prefix, n.projections[3])?,
            q_norm: RmsNorm::load_on(&context, src, &n.q_norm, eps, None)?,
            k_norm: RmsNorm::load_on(&context, src, &n.k_norm, eps, None)?,
            indexer: QsaIndexer::load_on(context.clone(), src, &n.indexer, eps)?,
            heads,
            kv_heads,
            head_dim,
            scale: model::scale(head_dim),
            block_kernel: context
                .device()
                .clone()
                .capabilities()
                .iter()
                .any(|c| c.kind == KernelKind::BlockAttention(BlockAttentionKind::Attend)),
        })
    }
    pub fn forward(
        &self,
        x: &D::Tensor,
        rope: &Rotary,
        cache: Option<&mut FullAttentionCache<D>>,
        offset: usize,
        positions: &D::Tensor,
    ) -> anyhow::Result<D::Tensor> {
        self.forward_with_capture(x, rope, cache, offset, positions, false)
    }
    pub fn forward_with_capture(
        &self,
        x: &D::Tensor,
        rope: &Rotary,
        mut cache: Option<&mut FullAttentionCache<D>>,
        offset: usize,
        positions: &D::Tensor,
        serial_capture: bool,
    ) -> anyhow::Result<D::Tensor> {
        let tape = cache.as_deref_mut().and_then(|c| {
            model::use_indexer_tape(c.next_pos.is_some(), c.indexer_ok)
                .then_some(&mut c.indexer_tape)
        });
        let sparse = if serial_capture {
            self.indexer.forward_captured(x, rope, tape, offset)
        } else {
            self.indexer.forward(x, rope, tape, offset)
        }
        .map_err(error)?;
        let plan = Plan {
            batch: x.dim(0),
            rows: x.dim(1),
            heads: self.heads as i32,
            key_heads: self.kv_heads as i32,
            width: self.head_dim as i32,
            rotary: rope.dimensions,
            base: rope.base.to_bits(),
            epsilon: self.q_norm.eps.to_bits(),
            scale: self.scale.to_bits(),
            quant: [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
                .map(|p| (p.group_size, p.bits)),
            position_shape: positions.shape().to_vec(),
            offset,
            cache: cache.as_ref().map(|c| c.append_plan(x.dim(0), x.dim(1))),
            ragged: cache.as_ref().is_some_and(|c| c.attn_mask.is_some()),
            selection: sparse.as_ref().map(|s| Selection {
                width: s.block_ids.dim(2),
                ratio: s.cr,
                length: s.kv_len,
            }),
            block_kernel: self.block_kernel,
            serial_capture,
            selection_budget: self.indexer.token_budget,
        };
        let layout = plan
            .cache
            .as_ref()
            .map(|c| c.layout())
            .transpose()
            .map_err(error)?;
        let mut inputs = vec![x, positions];
        for p in [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj] {
            inputs.extend([&p.weight, &p.scales, &p.biases]);
        }
        inputs.extend([&self.q_norm.weight, &self.k_norm.weight]);
        if let Some(c) = cache.as_ref() {
            if let (Some(k), Some(v)) = (&c.keys, &c.values) {
                inputs.extend([k, v]);
            }
            if let Some(mask) = &c.attn_mask {
                inputs.push(mask);
            }
        }
        if let Some(s) = &sparse {
            inputs.extend([&s.block_ids, &s.block_valid, &s.complete, &s.q_pos]);
        }
        let mut out = self
            .context
            .with_slot(
                "qwen4.attention.plans",
                || std::cell::RefCell::new(PlanCache::new(self.context.device().clone(), 64)),
                |plans| {
                    plans
                        .borrow_mut()
                        .try_get_or_prepare(plan.clone(), || plan.graph())?
                        .run(&inputs, &mut [])
                },
            )
            .map_err(error)?
            .into_iter();
        let output = out.next().unwrap();
        if let (Some(c), Some((offset, capacity, _))) = (cache, layout) {
            c.keys = out.next();
            c.values = out.next();
            c.offset = offset;
            c.capacity = capacity;
        }
        Ok(output)
    }
}
