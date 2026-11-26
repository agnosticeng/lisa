//! Real-device coverage for bounded program-layout replacement, without a model.
use anyhow::{Result, ensure};
use lisa_backend::{Binary, Kernel, TensorAccess, TensorOperation, Unary};
use lisa_engine::core::graph::PreparedGraph;
use lisa_metal::{Array, device::MetalDevice};
use lisa_models::graph::{Builder, Value};

#[test]
fn recorded_graph_evicts_exact_layouts_without_overwriting_queued_outputs() -> Result<()> {
    let device = MetalDevice::system_default();
    let mut builder = Builder::new(1, 0, 1);
    let sum = builder.one(
        Kernel::Tensor(TensorOperation::Binary(Binary::Add)),
        vec![Value::Input(0), Value::State(0)],
    );
    let output = builder.one(
        Kernel::Tensor(TensorOperation::Unary(Unary::Negate)),
        vec![sum],
    );
    builder.commit(0, output);
    let plan = PreparedGraph::new(
        device,
        builder.finish(vec![output, Value::Input(0)]),
        vec![],
    )?
    .with_dispatch_program();
    let mut live = Vec::new();
    // More than eight genuine offset/stride views, followed by re-entry to an
    // evicted layout. No readback is allowed before all evictions are complete.
    for offset in (0..12).chain(std::iter::once(0)) {
        for replay in [false, true] {
            let salt = if replay { 100.0 } else { 10.0 };
            let storage = (0..64).map(|i| salt + i as f32).collect::<Vec<_>>();
            let input = Array::from_slice(&storage, &[64])
                .slice(0, offset, Some(offset + 4))?
                .reshape(&[2, 2])?
                .transpose_axes(&[1, 0])?;
            let logical = [
                storage[offset as usize],
                storage[offset as usize + 2],
                storage[offset as usize + 1],
                storage[offset as usize + 3],
            ];
            let mut states = [Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])];
            let before = lisa_metal::runtime::dispatch_program_replay_count();
            let values = plan
                .run(&[&input], &mut states)?
                .into_iter()
                .collect::<Vec<_>>();
            let after = lisa_metal::runtime::dispatch_program_replay_count();
            ensure!(
                if replay {
                    after > before
                } else {
                    after == before
                },
                "offset{offset}: capture and replay routes differ from the expected cache state"
            );
            let expected = logical
                .iter()
                .enumerate()
                .map(|(i, value)| (-(*value + i as f32 + 1.0)).to_bits())
                .collect::<Vec<_>>();
            live.push((
                values,
                states[0].clone(),
                expected,
                logical.map(f32::to_bits),
            ));
        }
    }
    // Commands retain encoded resources after every program has been evicted
    // or dropped; escaping output/state/input aliases remain independent.
    drop(plan);
    for (values, state, expected, input) in live {
        ensure!(
            values[0].t.to_vec::<u32>()? == expected,
            "queued output was overwritten"
        );
        ensure!(
            state.t.to_vec::<u32>()? == expected,
            "committed state was overwritten"
        );
        ensure!(
            values[1].t.to_vec::<u32>()? == input,
            "escaping input alias was overwritten"
        );
    }
    Ok(())
}
