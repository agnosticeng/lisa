//! Reader-to-compute encoding adapter. Payloads and quantization groups are unchanged.
use lisa_backend::PackedEncoding as E;
use lisa_gguf::TensorType as T;
use lisa_models::packed::Projection as P;
pub fn projection(kind: T) -> anyhow::Result<P> {
    Ok(match kind {
        T::F32 | T::F16 | T::BF16 => P::Plain,
        T::Q8_0 => P::Q8,
        T::IQ2_XXS => P::Block(E::Iq2Xxs),
        T::IQ2_XS => P::Block(E::Iq2Xs),
        T::IQ1_M => P::Block(E::Iq1M),
        T::Q5_K => P::Block(E::Q5K),
        T::Q6_K => P::Block(E::Q6K),
        T::Q2_K => P::Block(E::Q2K),
        T::Q4_K => P::Block(E::Q4K),
        _ => anyhow::bail!("unsupported packed compute encoding {kind:?}"),
    })
}
