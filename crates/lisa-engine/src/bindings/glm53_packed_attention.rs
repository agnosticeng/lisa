use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
// Split K/V GGUF attention, using the shared DS4 indexer and cache contracts.
use crate::bindings::glm53_indexer::Indexer;
use crate::bindings::glm53_projection::{Projection, plain};
use crate::core::graph::PlanCache;
use crate::weights::gguf::{PackedTensor, ResidentWeights};
use crate::{attention_cache::FullAttentionCache, indexer_cache::IndexerTape};
use anyhow::{Context, Result, ensure};
use lisa_models::glm53::packed_attention as model;
pub fn run<D: BindingDevice, const N: usize>(
    context: &DeviceContext<D>,
    request: model::Request,
    inputs: &[&D::Tensor],
) -> Result<[D::Tensor; N]> {
    context.with_slot(
        "glm53.packed_attention.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 64)),
        |slot| {
            let mut p = slot.borrow_mut();
            let recipe = request.clone();
            p.try_get_or_prepare(request, || recipe.graph())?
                .run(inputs, &mut [])?
                .into_iter()
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("attention output arity differs"))
        },
    )
}

pub struct GgufDsa<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub qa: Projection<D>,
    pub qb: Projection<D>,
    pub kva: Projection<D>,
    pub output: Projection<D>,
    pub key: PackedTensor<D>,
    pub value: PackedTensor<D>,
    pub qnorm: D::Tensor,
    pub kvnorm: D::Tensor,
    pub indexer: Indexer<D>,
}
impl<D: BindingDevice> GgufDsa<D> {
    pub fn load_on<L>(
        context: DeviceContext<D>,
        w: &ResidentWeights<D, L>,
        layer: usize,
    ) -> Result<Self> {
        let p = |name: &str| model::weight_name(layer, name);
        let f = |name: &str| plain(w, &p(name));
        let projection = |name: &str| Projection::load(w, &p(name));
        let packed = |name: &str| {
            w.tensors
                .get(&p(name))
                .cloned()
                .context("missing GLM attention weight")
        };
        Ok(Self {
            context: context.clone(),
            qa: projection(model::WEIGHTS[0])?,
            qb: projection(model::WEIGHTS[1])?,
            kva: projection(model::WEIGHTS[2])?,
            output: projection(model::WEIGHTS[3])?,
            key: packed(model::WEIGHTS[4])?,
            value: packed(model::WEIGHTS[5])?,
            qnorm: f(model::WEIGHTS[6])?,
            kvnorm: f(model::WEIGHTS[7])?,
            indexer: Indexer {
                context: context.clone(),
                wq_b_w: f(model::WEIGHTS[8])?,
                wk_w: f(model::WEIGHTS[9])?,
                k_norm_w: f(model::WEIGHTS[10])?,
                k_norm_b: f(model::WEIGHTS[11])?,
                weights_w: f(model::WEIGHTS[12])?,
                gate_w: f(model::WEIGHTS[13])?,
                ape: f(model::WEIGHTS[14])?,
                heads: model::INDEXER.heads,
                head_dim: model::INDEXER.dimension,
                topk: model::INDEXER.topk,
                kpool: model::INDEXER.pool,
                always_tail: model::INDEXER.tail,
            },
        })
    }
    pub fn project(&self, x: &D::Tensor) -> Result<(D::Tensor, D::Tensor, D::Tensor)> {
        ensure!(
            model::valid_input(x.shape()),
            "GGUF GLM serial attention expects one row"
        );
        let [qr, q, latent] = run(
            &self.context,
            model::Request::Project([
                self.qa.descriptor_on(self.context.device())?,
                self.qb.descriptor_on(self.context.device())?,
                self.kva.descriptor_on(self.context.device())?,
            ]),
            &[x, &self.qa, &self.qb, &self.kva, &self.qnorm, &self.kvnorm],
        )?;
        Ok((qr, q, latent))
    }
    pub fn attend(
        &self,
        q: &D::Tensor,
        compact: &D::Tensor,
        ids: &D::Tensor,
        dense: bool,
    ) -> Result<D::Tensor> {
        let [out] = run(
            &self.context,
            model::Request::Attend {
                selected: ids.size(),
                dense,
                output: self.output.descriptor_on(self.context.device())?,
            },
            &[
                q,
                compact,
                ids,
                self.key.packed_bytes(),
                self.value.packed_bytes(),
                &self.output,
            ],
        )?;
        Ok(out)
    }

    pub fn forward(
        &self,
        x: &D::Tensor,
        cache: &mut FullAttentionCache<D>,
        layer: usize,
    ) -> Result<D::Tensor> {
        let (qr, q, latent) = self.project(x)?;
        let (key, gate) =
            self.indexer
                .project_pair(x, &self.indexer.wk_w, x, &self.indexer.gate_w)?;
        cache
            .indexer_tape
            .append_fields(&key, &gate, model::INDEXER.pool)?;
        cache.indexer_tape.append_compact_f16(&latent)?;
        self.indexer
            .update_pooled_cache(&mut cache.indexer_tape, None, layer)?;
        cache.offset = cache.indexer_tape.total;
        let total = cache.offset;
        let dense = model::dense(total);
        let ids = if dense {
            pull_ids(&self.context, &model::ids(0, total)?, &[1, 1, total as i32])?
        } else {
            self.indexer
                .select_ids_gpu(
                    &mut cache.indexer_tape,
                    x,
                    &qr,
                    total,
                    1,
                    None,
                    layer,
                    false,
                )?
                .0
        };
        let compact = cache
            .indexer_tape
            .compact_slice(0, total)
            .context("missing GLM compact cache")?;
        self.attend(&q, &compact, &ids, dense)
    }
    pub fn mtp(
        &self,
        x: &D::Tensor,
        tape: &mut IndexerTape<D>,
        position: usize,
        min_position: usize,
    ) -> Result<D::Tensor> {
        ensure!(min_position <= position, "invalid GLM MTP window");
        let (_, q, latent) = self.project(x)?;
        tape.write_compact_f16_at(&latent, position)?;
        let compact = tape
            .compact_slice(0, position + 1)
            .context("missing GLM MTP cache")?;
        let ids = pull_ids(
            &self.context,
            &model::ids(min_position, position + 1)?,
            &[1, 1, (position - min_position + 1) as i32],
        )?;
        self.attend(&q, &compact, &ids, false)
    }
}

fn pull_ids<D: BindingDevice>(
    context: &DeviceContext<D>,
    ids: &[i32],
    shape: &[i32],
) -> Result<D::Tensor> {
    Ok(context.device().pull(
        lisa_backend::HostTensor {
            bytes: &ids.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
            shape,
            dtype: lisa_backend::ScalarType::I32,
        }
        .into(),
        Default::default(),
    )?)
}
