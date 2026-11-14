//! Optional native counterpart to DSpine's FP8 coarse/BF16 rescore retrieval.
//! The target already uses affine weights, so a separate 2-bit proposal head
//! reads fewer bytes than FP8. Only candidate discovery is approximate; target
//! verification and selected-row rescoring retain the original packed weights.

use crate::bindings::dspine_retrieval_plans as bound;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use anyhow::{Context, ensure};
use lisa_backend::{ManagedDevice, TensorAccess};
use lisa_models::qwen3_5::dspine::retrieval::{Head, Rescore};

use crate::weights::affine::QuantizedLinear;

pub fn enabled() -> anyhow::Result<bool> {
    match std::env::var("LISA_DSPINE_RETRIEVAL").as_deref() {
        Ok("coarse2") => Ok(true),
        Err(std::env::VarError::NotPresent) | Ok("native") => Ok(false),
        _ => anyhow::bail!("LISA_DSPINE_RETRIEVAL must be native or coarse2"),
    }
}

pub struct Retrieval<D: BindingDevice + ManagedDevice> {
    context: DeviceContext<D>,
    pub head: QuantizedLinear<D::Tensor>,
}

impl<D: BindingDevice + ManagedDevice> Retrieval<D> {
    pub fn build(
        context: DeviceContext<D>,
        target: &QuantizedLinear<D::Tensor>,
    ) -> anyhow::Result<Option<Self>> {
        let spec = Head {
            vocab: target.weight.dim(0),
            hidden: target.weight.dim(1) * 32 / target.bits,
            source_group: target.group_size,
            source_bits: target.bits,
        };
        if !spec.useful() {
            return Ok(None);
        }
        let _trace = context.trace("dspine.retrieval.build");
        let start = std::time::Instant::now();
        let bytes = spec.bytes()?;
        let group = 64;
        let needed = context
            .device()
            .memory_usage()?
            .active
            .checked_add(bytes)
            .and_then(|n| n.checked_add(128 << 20))
            .context("coarse head admission overflow")?;
        ensure!(
            needed <= crate::memory::ram_budget(),
            "coarse head exceeds RAM budget"
        );
        let mut allocated = bound::allocate(&context, spec.clone())?.into_iter();
        let weight = allocated.next().context("missing coarse weight storage")?;
        let scales = allocated.next().context("missing coarse scale storage")?;
        let biases = allocated.next().context("missing coarse bias storage")?;
        ensure!(
            context.device().dtype(&target.scales) == context.device().dtype(&scales),
            "coarse head requires BF16 metadata"
        );
        for (first, end) in spec.chunks() {
            bound::chunk(
                &context,
                spec.clone(),
                first,
                end,
                &[
                    &target.weight,
                    &target.scales,
                    &target.biases,
                    &weight,
                    &scales,
                    &biases,
                ],
            )?;
            context.device().submit(true)?;
        }
        eprintln!(
            "[dspine] coarse2 retrieval: {} bytes built in {:.3}s",
            bytes,
            start.elapsed().as_secs_f64()
        );
        Ok(Some(Self {
            context,
            head: QuantizedLinear {
                weight,
                scales,
                biases,
                group_size: group,
                bits: 2,
            },
        }))
    }

    /// All selected columns run through the same row-width QMV family as the
    /// original head. Wide GEMM shapes have their own reduction law and keep
    /// full-head retrieval until independently admitted.
    pub fn candidates(
        &self,
        hidden: &D::Tensor,
        target: &QuantizedLinear<D::Tensor>,
    ) -> anyhow::Result<Option<D::Tensor>> {
        if !lisa_models::qwen3_5::dspine::retrieval::supported(hidden.shape()) {
            return Ok(None);
        }
        let _trace = self.context.trace("dspine.retrieval.coarse2");
        let spec = Rescore {
            rows: hidden.dim(0),
            count: 128,
            group: target.group_size,
            bits: target.bits,
        };
        Ok(Some(bound::rescore(
            &self.context,
            spec,
            &[
                hidden,
                &self.head.weight,
                &self.head.scales,
                &self.head.biases,
                &target.weight,
                &target.scales,
                &target.biases,
            ],
            true,
        )?))
    }
}
