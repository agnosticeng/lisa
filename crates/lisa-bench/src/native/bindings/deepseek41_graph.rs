//! Application aliases and native conformance checks for portable graph bindings.
#[cfg(test)]
use crate::core::graph::PreparedGraph;
use anyhow::Result;
use lisa_metal::{Array, Stream, device::MetalDevice};
#[cfg(test)]
use lisa_models::deepseek41::tower as model;
pub type Graph = lisa_engine::bindings::deepseek41_graph::Graph<MetalDevice>;
pub use lisa_engine::bindings::deepseek41_graph::Assembly;
pub(super) fn assembly(key: Assembly, inputs: &[&Array], stream: &Stream) -> Result<Array> {
    lisa_engine::bindings::deepseek41_graph::assembly(
        key,
        inputs,
        &super::deepseek41_projection::context_for(stream),
    )
}
#[cfg(test)]
mod tests {

    use super::*;
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    use lisa_metal::kernels::ggml as native;
    #[test]
    fn engram_gate_plan_preserves_ordered_bits_and_dispatches() -> Result<()> {
        let stream = Stream::gpu();
        let plan = PreparedGraph::new(
            MetalDevice::on_stream(&stream),
            lisa_models::deepseek41::engram::gate(),
            vec![],
        )?;
        let values = |shape: &[i32], seed: usize| {
            Array::from_slice(
                &(0..shape.iter().map(|&x| x as usize).product::<usize>())
                    .map(|i| ((i * 17 + seed) % 127) as f32 / 113. - 0.4)
                    .collect::<Vec<_>>(),
                shape,
            )
        };
        for rows in [1, 3, 8, 16, 17] {
            let a = values(&[rows, 4, 5120], 11);
            let mut old = Array::new(a.t.copied()?);
            let new = Array::new(a.t.copied()?);
            let kv = values(&[rows, 5, 5120], 31);
            let q = values(&[4, 5120], 91);
            let k = values(&[4, 5120], 103);
            let before = counts();
            native::engram_in_place(&mut old, &kv, &q, &k, None, 1e-20, &stream)?;
            let between = counts();
            let new = plan
                .run(&[&new, &kv, &q, &k], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            let after = counts();
            assert_eq!(old.to_vec1::<u32>()?, new.to_vec1::<u32>()?);
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(difference(&before, &between), difference(&between, &after));
            }
        }
        Ok(())
    }
}
