use crate::bindings::shared::normalization::Rotary;
use crate::cache::LayerCache;
// Bound blocks for the model-owned hyper-stream decoder sequence.
use crate::bindings::{
    qwen4::{
        attention::Attention,
        hyper::{GatedResidual, inject_norm},
        moe::SparseMoeBlock,
        ple::PleLayer,
    },
    shared::gated_delta::GatedDeltaNet,
};
use crate::core::{
    graph::PreparedGraph,
    sequence::{Blocks, Prepared},
};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_models::qwen4::decoder::{self as model, Step};
pub enum Block<D: BindingDevice> {
    Linear(GatedDeltaNet<D>),
    Attention(Attention<D>),
}
pub struct DecoderLayer<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub block: Block<D>,
    pub mlp: SparseMoeBlock<D>,
    pub attn_hc: GatedResidual<D>,
    pub mlp_hc: GatedResidual<D>,
    pub ple: Option<PleLayer<D>>,
    pub has_ple: bool,
}
fn prepared(ple: bool) -> &'static Prepared<Step, 19> {
    static WITH: std::sync::OnceLock<Prepared<Step, 19>> = std::sync::OnceLock::new();
    static WITHOUT: std::sync::OnceLock<Prepared<Step, 19>> = std::sync::OnceLock::new();
    (if ple { &WITH } else { &WITHOUT })
        .get_or_init(|| Prepared::new(model::schedule(ple)).expect("valid static decoder schedule"))
}
fn add<D: BindingDevice>(
    context: &DeviceContext<D>,
    a: &D::Tensor,
    b: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "qwen4.decoder.add",
        std::cell::OnceCell::<PreparedGraph<D>>::new,
        |p| {
            if p.get().is_none() {
                let _ = p.set(PreparedGraph::new(
                    context.device().clone(),
                    model::add(),
                    vec![],
                )?);
            }
            p.get()
                .unwrap()
                .run(&[a, b], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
impl<D: BindingDevice> DecoderLayer<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        residual: &D::Tensor,
        pending_out: Option<&D::Tensor>,
        pending_inject: Option<&D::Tensor>,
        cache: Option<&mut LayerCache<D>>,
        rope: &Rotary,
        offset: usize,
        positions: &D::Tensor,
        ple_args: Option<(&[Vec<i64>], &[Vec<i64>])>,
        capture: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)> {
        self.forward_with_capture_law(
            residual,
            pending_out,
            pending_inject,
            cache,
            rope,
            offset,
            positions,
            ple_args,
            capture,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_capture_law(
        &mut self,
        residual: &D::Tensor,
        pending_out: Option<&D::Tensor>,
        pending_inject: Option<&D::Tensor>,
        cache: Option<&mut LayerCache<D>>,
        rope: &Rotary,
        offset: usize,
        positions: &D::Tensor,
        ple_args: Option<(&[Vec<i64>], &[Vec<i64>])>,
        capture: bool,
        serial_capture: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)> {
        struct Binding<'a, D: BindingDevice> {
            layer: &'a mut DecoderLayer<D>,
            cache: Option<&'a mut LayerCache<D>>,
            rope: &'a Rotary,
            offset: usize,
            positions: &'a D::Tensor,
            ple_args: Option<(&'a [Vec<i64>], &'a [Vec<i64>])>,
            capture: bool,
            serial_capture: bool,
        }
        impl<D: BindingDevice> Blocks<Step> for Binding<'_, D> {
            type Tensor = D::Tensor;
            fn call(
                &mut self,
                step: &Step,
                a: &[Option<&D::Tensor>],
            ) -> anyhow::Result<Vec<Option<D::Tensor>>> {
                let req = |i: usize| {
                    a[i].ok_or_else(|| anyhow::anyhow!("required decoder input is absent"))
                };
                let norm =
                    |layer: &DecoderLayer<D>, res: &D::Tensor, out, inj, scale: &D::Tensor| {
                        inject_norm(
                            &layer.context,
                            res,
                            out,
                            inj,
                            scale,
                            (layer.attn_hc.hidden / layer.attn_hc.hc_count) as i32,
                            layer.attn_hc.hc_count as i32,
                            layer.attn_hc.hc_norm.eps,
                        )
                    };
                let out = match *step {
                    Step::Scale(slot) => vec![
                        if slot == 0 {
                            &self.layer.attn_hc
                        } else {
                            &self.layer.mlp_hc
                        }
                        .norm_scale_q()?,
                    ],
                    Step::InjectNorm => {
                        let (x, y) = norm(self.layer, req(0)?, a[1], a[2], req(3)?)?;
                        vec![x, y]
                    }
                    Step::InjectNormPlain => {
                        let (x, y) = norm(self.layer, req(0)?, None, None, req(1)?)?;
                        vec![x, y]
                    }
                    Step::Ple => {
                        let ple = self
                            .layer
                            .ple
                            .as_mut()
                            .ok_or_else(|| anyhow::anyhow!("PLE block is not bound"))?;
                        let (tokens, history) = self
                            .ple_args
                            .ok_or_else(|| anyhow::anyhow!("PLE context is not bound"))?;
                        vec![if let Some(c) = self.cache.as_deref_mut() {
                            let (slot, cap) = c.ple_conv_mut();
                            if self.serial_capture {
                                ple.forward_captured(
                                    req(0)?,
                                    tokens,
                                    history,
                                    slot,
                                    self.capture,
                                    cap,
                                )?
                            } else {
                                ple.forward(req(0)?, tokens, history, slot, self.capture, cap)?
                            }
                        } else {
                            if self.serial_capture {
                                ple.forward_captured(
                                    req(0)?,
                                    tokens,
                                    history,
                                    &mut None,
                                    self.capture,
                                    &mut None,
                                )?
                            } else {
                                ple.forward(
                                    req(0)?,
                                    tokens,
                                    history,
                                    &mut None,
                                    self.capture,
                                    &mut None,
                                )?
                            }
                        }]
                    }
                    Step::Add => vec![add(&self.layer.context, req(0)?, req(1)?)?],
                    Step::HyperMix(slot) => {
                        let mixer = if slot == 0 {
                            &self.layer.attn_hc
                        } else {
                            &self.layer.mlp_hc
                        };
                        let (x, y) = if self.serial_capture {
                            mixer.mix_from_normed_captured(req(0)?, true)?
                        } else {
                            mixer.mix_from_normed(req(0)?, true)?
                        };
                        vec![x, y]
                    }
                    Step::Mixer => vec![match (&mut self.layer.block, self.cache.as_deref_mut()) {
                        (Block::Linear(g), Some(LayerCache::Linear(c))) => g
                            .forward_with_capture_law(
                                req(0)?,
                                Some(c),
                                self.capture,
                                self.serial_capture,
                            )?,
                        (Block::Linear(g), None) => g.forward_with_capture_law(
                            req(0)?,
                            None,
                            self.capture,
                            self.serial_capture,
                        )?,
                        (Block::Attention(g), Some(LayerCache::Full(c))) => g
                            .forward_with_capture(
                                req(0)?,
                                self.rope,
                                Some(c),
                                self.offset,
                                self.positions,
                                self.serial_capture,
                            )?,
                        (Block::Attention(g), None) => g.forward_with_capture(
                            req(0)?,
                            self.rope,
                            None,
                            self.offset,
                            self.positions,
                            self.serial_capture,
                        )?,
                        _ => anyhow::bail!("cache kind does not match bound block"),
                    }],
                    Step::Experts => vec![if self.serial_capture {
                        self.layer.mlp.forward_captured(req(0)?)?
                    } else {
                        self.layer.mlp.forward(req(0)?)?
                    }],
                };
                Ok(out.into_iter().map(Some).collect())
            }
        }
        let plan = prepared(model::has_ple(self.ple.is_some(), ple_args.is_some()));
        let mut binding = Binding {
            layer: self,
            cache,
            rope,
            offset,
            positions,
            ple_args,
            capture,
            serial_capture,
        };
        let mut out = plan
            .run(&mut binding, &[Some(residual), pending_out, pending_inject])?
            .into_iter();
        Ok((
            out.next().flatten().unwrap(),
            out.next().flatten().unwrap(),
            out.next().flatten().unwrap(),
        ))
    }
}
