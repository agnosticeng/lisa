//! Original residual weights bound to the model's portable computation graphs.
use crate::core::graph::{PlanCache, PreparedGraph};
use crate::{bindings::BindingDevice, execution::DeviceContext, weights::source::TensorSource};
use anyhow::{Context, Result};
use lisa_backend::TensorAccess;
use lisa_backend::{Outputs, ScalarType};
use lisa_models::glm53::hyper::{self as model, Mode, Spec};
use std::cell::RefCell;
pub type Key = (Spec, Mode, Vec<i32>, ScalarType);
pub fn plans<D: BindingDevice>(context: &DeviceContext<D>) -> RefCell<PlanCache<D, Key>> {
    RefCell::new(PlanCache::new(context.device().clone(), 64))
}
pub struct Hc<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub fn_w: D::Tensor,
    pub base: D::Tensor,
    pub scale: D::Tensor,
    pub hc_mult: usize,
    pub hidden: usize,
    pub eps: f32,
    pub sinkhorn_iters: usize,
    pub rms_eps: f32,
    pub plans: RefCell<PlanCache<D, Key>>,
}
impl<D: BindingDevice> Hc<D> {
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        hc_mult: usize,
        hidden: usize,
        hc_eps: f32,
        sinkhorn_iters: usize,
        rms_eps: f32,
    ) -> Result<Self> {
        let [matrix, scale, base] = model::requirements(prefix);
        let fn_w = src.get(&matrix)?;
        Spec {
            streams: hc_mult,
            hidden,
            epsilon: hc_eps.to_bits(),
            rms_epsilon: rms_eps.to_bits(),
            iterations: sinkhorn_iters,
        }
        .validate_matrix(fn_w.shape())?;
        let preparation =
            PreparedGraph::new(context.device().clone(), model::promote_parameter(), vec![])?;
        let promote = |x: &D::Tensor| -> Result<D::Tensor> {
            preparation
                .run(&[x], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        };
        let scale = promote(&src.get(&scale)?)?;
        let base = promote(&src.get(&base)?)?;
        Ok(Self {
            context: context.clone(),
            fn_w,
            scale,
            base,
            hc_mult,
            hidden,
            eps: hc_eps,
            sinkhorn_iters,
            rms_eps,
            plans: plans(&context),
        })
    }
    fn spec(&self) -> Spec {
        Spec {
            streams: self.hc_mult,
            hidden: self.hidden,
            epsilon: self.eps.to_bits(),
            rms_epsilon: self.rms_eps.to_bits(),
            iterations: self.sinkhorn_iters,
        }
    }
    fn run(&self, mode: Mode, shape: &[i32], inputs: &[&D::Tensor]) -> Result<Outputs<D::Tensor>> {
        let dtype = match self.context.device().dtype(&self.fn_w) {
            Some(ScalarType::BF16) => ScalarType::BF16,
            Some(ScalarType::F16) => ScalarType::F16,
            Some(ScalarType::F32) => ScalarType::F32,
            _ => anyhow::bail!("unsupported residual projection dtype"),
        };
        let spec = self.spec();
        self.plans
            .borrow_mut()
            .try_get_or_prepare((spec, mode, shape.to_vec(), dtype), || {
                spec.graph(shape, dtype, mode)
            })?
            .run(inputs, &mut [])
    }
    pub fn mix_rows(&self, state: &D::Tensor) -> Result<(usize, usize, usize, usize, D::Tensor)> {
        let mix = self
            .run(
                Mode::Mix,
                state.shape(),
                &[state, &self.fn_w, &self.scale, &self.base],
            )?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        Ok((
            state.dim(0) as usize,
            state.dim(1) as usize,
            state.dim(2) as usize,
            state.dim(3) as usize,
            mix,
        ))
    }
    pub fn split_normed(
        &self,
        state: &D::Tensor,
        norm_weight: &D::Tensor,
        norm_eps: f32,
    ) -> Result<Option<(D::Tensor, D::Tensor, D::Tensor, D::Tensor)>> {
        Ok(self
            .split_normed_with_mixes(state, norm_weight, norm_eps)?
            .map(|(a, b, c, d, _, _)| (a, b, c, d)))
    }
    pub fn split_normed_with_mixes(
        &self,
        state: &D::Tensor,
        norm_weight: &D::Tensor,
        norm_eps: f32,
    ) -> Result<
        Option<(
            D::Tensor,
            D::Tensor,
            D::Tensor,
            D::Tensor,
            D::Tensor,
            D::Tensor,
        )>,
    > {
        let Some(mode) = self.spec().normed_mode(state.shape(), norm_eps) else {
            self.mix_rows(state)?;
            return Ok(None);
        };
        let mut out = self
            .run(
                mode,
                state.shape(),
                &[state, &self.fn_w, &self.scale, &self.base, norm_weight],
            )?
            .into_iter();
        let mut next = || out.next().context("missing residual result");
        Ok(Some((next()?, next()?, next()?, next()?, next()?, next()?)))
    }
    pub fn split(&self, state: &D::Tensor) -> Result<(D::Tensor, D::Tensor, D::Tensor)> {
        let mut out = self
            .run(
                Mode::Split,
                state.shape(),
                &[state, &self.fn_w, &self.scale, &self.base],
            )?
            .into_iter();
        Ok((
            out.next().context("missing collapsed output")?,
            out.next().context("missing post weights")?,
            out.next().context("missing combination weights")?,
        ))
    }
    pub fn expand(
        &self,
        y: &D::Tensor,
        residual: &D::Tensor,
        post: &D::Tensor,
        comb: &D::Tensor,
    ) -> Result<D::Tensor> {
        self.run(Mode::Expand, residual.shape(), &[y, residual, post, comb])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
}
