use crate::bindings::shared::normalization::{RmsNorm, Rotary};
use crate::cache::indexer::IndexerTape;
use crate::weights::{affine::QuantizedLinear, source::TensorSource};
// Indexer weight and tape storage bound to model-owned computation stages.
use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_backend::{BlockAttentionKind, KernelKind, Outputs};
use lisa_models::qwen4::indexer::{self as model, Spec};
pub struct QsaIndexer<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub index_qk_proj: QuantizedLinear<D::Tensor>,
    pub q_layer_norm: RmsNorm<D::Tensor>,
    pub k_layer_norm: RmsNorm<D::Tensor>,
    pub heads: usize,
    pub head_dim: usize,
    pub token_budget: usize,
    pub compress_ratio: usize,
    pub block_top_k: usize,
    pub select_available: bool,
}
#[derive(PartialEq)]
enum Key {
    Project(Spec, Vec<i32>, bool),
    Pool(Spec, i32, usize, usize, bool),
    Select(Spec, i32, i32, usize, usize, bool, bool),
}
fn execute<D: BindingDevice>(
    context: &DeviceContext<D>,
    key: Key,
    inputs: &[&D::Tensor],
    build: impl FnOnce() -> anyhow::Result<lisa_models::graph::Graph>,
) -> anyhow::Result<Outputs<D::Tensor>> {
    context.with_slot(
        "qwen4.indexer.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 64)),
        |cache| {
            cache
                .borrow_mut()
                .try_get_or_prepare(key, build)?
                .run(inputs, &mut [])
        },
    )
}
impl<D: BindingDevice> QsaIndexer<D> {
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        eps: f32,
    ) -> anyhow::Result<Self> {
        let (q, k, projection) = model::names(prefix);
        let (heads, head_dim, token_budget, compress_ratio, block_top_k) = model::GEOMETRY;
        Ok(Self {
            context: context.clone(),
            index_qk_proj: QuantizedLinear::load(src, prefix, projection)?,
            q_layer_norm: RmsNorm::load_on(&context, src, &q, eps, None)?,
            k_layer_norm: RmsNorm::load_on(&context, src, &k, eps, None)?,
            heads,
            head_dim,
            token_budget,
            compress_ratio,
            block_top_k,
            select_available: context
                .device()
                .clone()
                .capabilities()
                .iter()
                .any(|c| c.kind == KernelKind::BlockAttention(BlockAttentionKind::Select)),
        })
    }
    fn spec(&self, rope: &Rotary) -> Spec {
        Spec {
            heads: self.heads as i32,
            width: self.head_dim as i32,
            budget: self.token_budget,
            ratio: self.compress_ratio,
            top_k: self.block_top_k,
            quant: (self.index_qk_proj.group_size, self.index_qk_proj.bits),
            epsilon: self.q_layer_norm.eps.to_bits(),
            key_epsilon: self.k_layer_norm.eps.to_bits(),
            rotary: rope.dimensions,
            base: rope.base.to_bits(),
        }
    }
    pub fn forward(
        &self,
        x: &D::Tensor,
        rope: &Rotary,
        tape: Option<&mut IndexerTape<D>>,
        offset: usize,
    ) -> anyhow::Result<Option<QsaSelection<D::Tensor>>> {
        self.forward_with_capture_law(x, rope, tape, offset, false)
    }
    /// Preserve decode projections and per-prefix selection during capture.
    pub fn forward_captured(
        &self,
        x: &D::Tensor,
        rope: &Rotary,
        tape: Option<&mut IndexerTape<D>>,
        offset: usize,
    ) -> anyhow::Result<Option<QsaSelection<D::Tensor>>> {
        self.forward_with_capture_law(x, rope, tape, offset, true)
    }
    fn forward_with_capture_law(
        &self,
        x: &D::Tensor,
        rope: &Rotary,
        tape: Option<&mut IndexerTape<D>>,
        offset: usize,
        serial_capture: bool,
    ) -> anyhow::Result<Option<QsaSelection<D::Tensor>>> {
        let spec = self.spec(rope);
        let (b, s) = (x.dim(0), x.dim(1));
        let update = spec.update(s as usize, tape.as_ref().map(|t| (t.total, t.pooled_upto)))?;
        let mut out = execute(
            &self.context,
            Key::Project(spec, x.shape().to_vec(), serial_capture),
            &[
                x,
                &self.index_qk_proj.weight,
                &self.index_qk_proj.scales,
                &self.index_qk_proj.biases,
            ],
            || {
                if serial_capture {
                    spec.project_captured(x.shape())
                } else {
                    spec.project(x.shape())
                }
            },
        )?
        .into_iter();
        let q = out.next().unwrap();
        let raw = out.next().unwrap();
        let pooled = match tape {
            Some(t) => {
                t.append(&raw, self.compress_ratio)?;
                if !update.active {
                    return Ok(None);
                }
                if let Some(range) = update.pool {
                    let raw = t
                        .raw_slice(
                            range.start * self.compress_ratio,
                            range.end * self.compress_ratio,
                        )
                        .ok_or_else(|| anyhow::anyhow!("indexer tape raw missing"))?;
                    let blk = execute(
                        &self.context,
                        Key::Pool(spec, b, range.start, range.end, true),
                        &[&raw, &self.k_layer_norm.weight],
                        || spec.pool(b, range.start, range.end, true),
                    )?
                    .into_one()
                    .map_err(anyhow::Error::msg)?;
                    t.push_pooled(&blk)?;
                }
                t.pooled_view()
                    .ok_or_else(|| anyhow::anyhow!("indexer pooled missing"))?
            }
            None => {
                if !update.active {
                    return Ok(None);
                }
                let range = update
                    .pool
                    .ok_or_else(|| anyhow::anyhow!("indexer pooling interval missing"))?;
                // The raw binding may end in a partial block; the model plan truncates it.
                execute(
                    &self.context,
                    Key::Pool(spec, b, range.start, range.end, false),
                    &[&raw, &self.k_layer_norm.weight],
                    || model::pool_without_tape(spec, b, s, range.end),
                )?
                .into_one()
                .map_err(anyhow::Error::msg)?
            }
        };
        let native = spec.native_select(b, s, offset, update.blocks, self.select_available);
        let mut out = execute(
            &self.context,
            Key::Select(spec, b, s, offset, update.blocks, native, serial_capture),
            &[&q, &pooled, &self.q_layer_norm.weight],
            || {
                if serial_capture {
                    spec.select_captured(b, s, offset, update.blocks, native)
                } else {
                    spec.select(b, s, offset, update.blocks, native)
                }
            },
        )?
        .into_iter();
        Ok(Some(QsaSelection {
            block_ids: out.next().unwrap(),
            block_valid: out.next().unwrap(),
            complete: out.next().unwrap(),
            q_pos: out.next().unwrap(),
            cr: self.compress_ratio as i32,
            kv_len: if native {
                (offset + s as usize) as i32
            } else {
                update.total as i32
            },
        }))
    }
}
/// Bound sparse-selection outputs; their interpretation is defined by the model.
pub struct QsaSelection<T> {
    pub block_ids: T,
    pub block_valid: T,
    pub complete: T,
    pub q_pos: T,
    pub cr: i32,
    pub kv_len: i32,
}
