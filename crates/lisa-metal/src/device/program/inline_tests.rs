//! Replay qualification for immutable inline metadata.
use super::*;
use anyhow::Context;
use half::bf16;

fn bits(a: &Array) -> Vec<u32> {
    match a.dtype() {
        Dtype::Bfloat16 => a
            .to_vec1::<bf16>()
            .unwrap()
            .into_iter()
            .map(|v| v.to_bits() as u32)
            .collect(),
        Dtype::Float32 => a
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect(),
        Dtype::Uint32 => a.to_vec1::<u32>().unwrap(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

#[test]
fn inline_i32_metadata_matches_original_tensor_abi_and_replays() -> anyhow::Result<()> {
    use crate::ffi::{KernelInput, MetalKernel, OutputArg};
    let device = MetalDevice::system_default();
    let stream = device.stream();
    let kernel = MetalKernel::new("inline_i32_array_abi", &["X", "P"], &["Y"],
        "uint i=thread_position_in_grid.x; if(i<37) Y[i]=X[i]+float(P[i%P_shape[0]]+P_ndim+int(P_strides[0]));",
        "", false, false)?.with_audited_full_write_outputs();
    for values in [
        &[31i32, -7][..],
        &[71, 3, -11][..],
        &[1, 2, 3, 4, 5, 6, 7][..],
    ] {
        let params = Array::from_slice(values, &[values.len() as i32]);
        let x = Array::from_slice(&(0..38).map(|i| i as f32 * 0.25).collect::<Vec<_>>(), &[38])
            .slice(0, 1, Some(38))?;
        let y = Array::from_slice(&(0..38).map(|i| i as f32 * -0.5).collect::<Vec<_>>(), &[38])
            .slice(0, 1, Some(38))?;
        let outputs = [OutputArg {
            shape: vec![37],
            dtype: Dtype::Float32,
        }];
        let ordinary = |a: &Array| {
            kernel.apply(
                &[a, &params],
                &[],
                (37, 1, 1),
                (32, 1, 1),
                &outputs,
                &stream,
            )
        };
        let inline = |a: &Array| {
            kernel.apply_with_inline_inputs(
                &[KernelInput::Tensor(&a.t), KernelInput::I32Slice(values)],
                &[],
                (37, 1, 1),
                (32, 1, 1),
                &outputs,
                &stream,
            )
        };
        let expected = ordinary(&x)?;
        let (captured, program) =
            device.capture_program(&[&x], &stream, &mut || Ok(inline(&x)?))?;
        assert_eq!(bits(&captured[0]), bits(&expected[0]));
        let program = program.context("inline metadata capture declined")?;
        let before = crate::runtime::dispatch_program_replay_count();
        let replay = program
            .replay(&[&y])?
            .context("inline metadata replay declined")?;
        assert!(crate::runtime::dispatch_program_replay_count() > before);
        assert_eq!(bits(&replay[0]), bits(&ordinary(&y)?[0]));
        assert_eq!(bits(&captured[0]), bits(&expected[0]));
        for bad in [&[][..], &[0; 8][..]] {
            assert!(
                kernel
                    .apply_with_inline_inputs(
                        &[KernelInput::Tensor(&x.t), KernelInput::I32Slice(bad)],
                        &[],
                        (37, 1, 1),
                        (32, 1, 1),
                        &outputs,
                        &stream
                    )
                    .is_err()
            );
        }
    }
    Ok(())
}
