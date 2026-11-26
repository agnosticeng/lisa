use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
// Weight/cache binding for model-owned sparse-indexer plans.
use crate::{
    bindings::shared::capture::Capture, cache::indexer::IndexerTape, core::graph::PlanCache,
};

use lisa_models::glm53::indexer::{self as model, Projection as P, Request as R, Spec};
pub fn run<D: BindingDevice, const N: usize>(
    context: &DeviceContext<D>,
    request: R,
    inputs: &[&D::Tensor],
) -> anyhow::Result<[D::Tensor; N]> {
    context.with_slot(
        "glm53.indexer.plans",
        || std::cell::RefCell::new(PlanCache::<D, R>::new(context.device().clone(), 64)),
        |slot| {
            let mut p = slot.borrow_mut();
            let recipe = request.clone();
            p.try_get_or_prepare(request, || recipe.graph())?
                .run(inputs, &mut [])?
                .into_iter()
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("indexer output arity differs"))
        },
    )
}
fn projection<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
) -> anyhow::Result<P> {
    Ok(P {
        shape: x.shape().to_vec(),
        weight: w.shape().to_vec(),
        dtype: context
            .device()
            .dtype(w)
            .ok_or_else(|| anyhow::anyhow!("unsupported indexer weight dtype"))?,
    })
}
pub struct Indexer<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub wq_b_w: D::Tensor,
    pub wk_w: D::Tensor,
    pub k_norm_w: D::Tensor,
    pub k_norm_b: D::Tensor,
    pub weights_w: D::Tensor,
    pub gate_w: D::Tensor,
    pub ape: D::Tensor,
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub kpool: usize,
    pub always_tail: bool,
}
impl<D: BindingDevice> Indexer<D> {
    fn spec(&self) -> Spec {
        Spec {
            heads: self.heads,
            dimension: self.head_dim,
            topk: self.topk,
            pool: self.kpool,
            tail: self.always_tail,
        }
    }
    pub fn project(&self, x: &D::Tensor, w: &D::Tensor) -> anyhow::Result<D::Tensor> {
        let [out] = run(
            &self.context,
            R::Project(projection(&self.context, x, w)?),
            &[x, w],
        )?;
        Ok(out)
    }
    pub fn project_pair(
        &self,
        a: &D::Tensor,
        wa: &D::Tensor,
        b: &D::Tensor,
        wb: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let [a, b] = run(
            &self.context,
            R::Pair(
                projection(&self.context, a, wa)?,
                projection(&self.context, b, wb)?,
            ),
            &[a, wa, b, wb],
        )?;
        Ok((a, b))
    }
    pub fn project_score_inputs(
        &self,
        x: &D::Tensor,
        qr: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let [q, w] = run(
            &self.context,
            R::ScoreInputs {
                spec: self.spec(),
                query: projection(&self.context, qr, &self.wq_b_w)?,
                weights: projection(&self.context, x, &self.weights_w)?,
            },
            &[qr, &self.wq_b_w, x, &self.weights_w],
        )?;
        Ok((q, w))
    }
    pub fn update_pooled_cache(
        &self,
        tape: &mut IndexerTape<D>,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<()> {
        let spec = self.spec();
        spec.validate()?;
        let Some((start, end)) = spec.pool_update(tape.total, tape.pooled_upto) else {
            return Ok(());
        };
        let raw = tape
            .raw_slice(start * spec.pool, end * spec.pool)
            .ok_or_else(|| anyhow::anyhow!("indexer raw cache missing"))?;
        let fused = spec.fused_pool(self.context.device().dtype(&self.ape));
        let [out] = run(
            &self.context,
            R::Pool {
                spec,
                batch: raw.dim(0) as usize,
                count: end - start,
                attempt: true,
                fused,
            },
            &[&raw, &self.ape, &self.k_norm_w, &self.k_norm_b],
        )?;
        if fused && let Some(c) = capture.filter(|c| c.wants(layer)) {
            c.store(&format!("L{layer}.dsa_pooled_kernel_output"), &out);
        }
        tape.push_pooled_f16(&out)?;
        tape.compact_raw_history(model::RAW_ROLLBACK_ROWS)?;
        Ok(())
    }
    pub fn index_scores(
        &self,
        tape: &mut IndexerTape<D>,
        iq: &D::Tensor,
        weights: &D::Tensor,
        offset: usize,
        t: usize,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
        prefill: bool,
    ) -> anyhow::Result<D::Tensor> {
        let b = iq.dim(0) as usize;
        let spec = self.spec();
        spec.validate()?;
        anyhow::ensure!(
            iq.shape() == [b as i32, self.heads as i32, t as i32, self.head_dim as i32]
                && weights.shape() == [b as i32, t as i32, self.heads as i32],
            "indexer score input shape mismatch"
        );
        let pools = tape.total / spec.pool;
        self.update_pooled_cache(tape, capture, layer)?;
        if pools == 0 {
            return pull_ids(
                &self.context,
                &spec.tail_ids(b, t, offset),
                &[b as i32, t as i32, (spec.pool - 1) as i32],
            );
        }
        let pooled = tape
            .pooled_view()
            .ok_or_else(|| anyhow::anyhow!("indexer pooled cache missing"))?;
        let position = offset.saturating_sub(t);
        let [scores] = run(
            &self.context,
            R::Scores {
                spec,
                position,
                prefill_tail: prefill && t == 1,
                rows4: crate::bindings::glm53::policy::glm53_indexer_decode_rows4_override()
                    .unwrap_or(true),
            },
            &[iq, weights, &pooled],
        )?;
        if let Some(c) = capture.filter(|c| c.wants(layer)) {
            c.store(&format!("L{layer}.dsa_indexer_q_p{position}_t{t}"), iq);
            c.store(
                &format!("L{layer}.dsa_indexer_weights_p{position}_t{t}"),
                weights,
            );
            c.store(&format!("L{layer}.dsa_pooled_keys"), &pooled);
            c.store(
                &format!("L{layer}.dsa_index_scores_p{position}_t{t}"),
                &scores,
            );
        }
        Ok(scores)
    }
    pub fn select_keep_gpu(
        &self,
        tape: &mut IndexerTape<D>,
        x: &D::Tensor,
        qr: &D::Tensor,
        offset: usize,
        t: usize,
        total: usize,
        _layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        let b = x.dim(0) as usize;
        let pools = tape.total / self.kpool;
        let (iq, w) = self.project_score_inputs(x, qr)?;
        let scores = self.index_scores(tape, &iq, &w, offset, t, None, 0, false)?;
        anyhow::ensure!(t == 1, "keep-mask path requires one query token");
        let [mask] = run(
            &self.context,
            R::Keep {
                spec: self.spec(),
                batch: b,
                pools,
                total,
                shuffle: crate::bindings::glm53::policy::glm53_indexer_topk_shuffle_override()
                    .unwrap_or(true),
            },
            &[&scores],
        )?;
        Ok(mask)
    }
    pub fn select_ids_gpu(
        &self,
        tape: &mut IndexerTape<D>,
        x: &D::Tensor,
        qr: &D::Tensor,
        offset: usize,
        t: usize,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
        prefill: bool,
    ) -> anyhow::Result<(D::Tensor, usize)> {
        let b = x.dim(0) as usize;
        let spec = self.spec();
        spec.validate()?;
        let pools = tape.total / spec.pool;
        let prefix = spec.valid_prefix(pools, t);
        let chunk = spec.score_chunk(b, pools, t);
        let start = offset.saturating_sub(t);
        let (iq, weights) = self.project_score_inputs(x, qr)?;
        let mut selections = vec![];
        for q0 in (0..t).step_by(chunk) {
            let n = chunk.min(t - q0);
            let [q, w] = run(
                &self.context,
                R::Chunk {
                    start: q0,
                    count: n,
                },
                &[&iq, &weights],
            )?;
            let offset = start + q0 + n;
            let scores = self.index_scores(tape, &q, &w, offset, n, capture, layer, prefill)?;
            let selected = self.expand_score_selection(&scores, b, n, pools, offset, layer)?;
            if let Some(c) = capture.filter(|c| c.wants(layer)) {
                c.store(
                    &format!("L{layer}.dsa_selected_p{}_t{n}", start + q0),
                    &selected,
                );
            }
            selections.push(selected);
        }
        if selections.len() == 1 {
            return Ok((selections.remove(0), prefix));
        }
        let [out] = run(
            &self.context,
            R::Join {
                count: selections.len(),
            },
            &selections.iter().collect::<Vec<_>>(),
        )?;
        Ok((out, prefix))
    }
    pub fn expand_score_selection(
        &self,
        scores: &D::Tensor,
        b: usize,
        t: usize,
        pools: usize,
        offset: usize,
        _layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        let [selected] = run(
            &self.context,
            R::Expand {
                spec: self.spec(),
                batch: b,
                tokens: t,
                pools,
                offset,
                shuffle: crate::bindings::glm53::policy::glm53_indexer_topk_shuffle_override()
                    .unwrap_or(true),
            },
            &[scores],
        )?;
        Ok(selected)
    }
    pub fn select(
        &self,
        x: &D::Tensor,
        tape: &mut IndexerTape<D>,
        qr: &D::Tensor,
        offset: usize,
        t: usize,
        total: usize,
        prefill: bool,
    ) -> anyhow::Result<D::Tensor> {
        let (iq, w) = self.project_score_inputs(x, qr)?;
        let scores = self.index_scores(tape, &iq, &w, offset, t, None, 0, prefill)?;
        let values = scores.read_f32()?;
        let b = x.dim(0) as usize;
        let spec = self.spec();
        let pools = tape.total / spec.pool;
        let selected = spec.select_host(&values, b, t, pools, offset, total);
        let width = spec.selected_pools(pools) * spec.pool + spec.pool - 1;
        pull_ids(
            &self.context,
            &selected,
            &[b as i32, t as i32, width as i32],
        )
    }
}
pub fn weighted_rms<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
    eps: f32,
    latent: bool,
) -> anyhow::Result<D::Tensor> {
    let [out] = run(
        context,
        R::WeightedRms {
            epsilon: eps.to_bits(),
            latent,
        },
        &[x, w],
    )?;
    Ok(out)
}
pub fn layer_norm<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
    b: &D::Tensor,
    eps: f32,
) -> anyhow::Result<D::Tensor> {
    let [out] = run(
        context,
        R::LayerNorm {
            epsilon: eps.to_bits(),
        },
        &[x, w, b],
    )?;
    Ok(out)
}

fn pull_ids<D: BindingDevice>(
    context: &DeviceContext<D>,
    ids: &[i32],
    shape: &[i32],
) -> anyhow::Result<D::Tensor> {
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
