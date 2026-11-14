//! Conformance of runtime-selected handles to the existing native provider.
use super::*;
use lisa_backend::erased::DeviceHandle;

#[test]
fn opaque_views_readback_and_indirect_dispatch_preserve_native_results() -> anyhow::Result<()> {
    let native = MetalDevice::system_default();
    let direct_only = DeviceHandle::new(native.clone());
    assert!(
        !direct_only
            .capabilities()
            .iter()
            .any(|cap| cap.kind == KernelKind::IndirectAffine(IndirectAffineKind::Pair))
    );
    assert!(
        direct_only
            .prepare(Kernel::IndirectAffine(IndirectAffine::Pair {
                intermediate: 512,
                top_k: 1,
                limit: 10f32.to_bits(),
                metadata: ScalarType::F32
            }))?
            .is_none()
    );
    let device = DeviceHandle::with_indirect(native.clone());
    let foreign = DeviceHandle::with_indirect(native.clone());
    let values = [99f32, 1.25, -2.5, 3.75, 8., 88.];
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let source = device.pull(
        HostTensor {
            bytes: &bytes,
            shape: &[6],
            dtype: ScalarType::F32,
        }
        .into(),
        ImportOptions::default(),
    )?;
    let view = device.byte_view(&source, 4, &[2, 2], ScalarType::F32)?;
    drop(source);
    assert_eq!(device.read_f32(&view)?, values[1..5]);
    assert!(
        device
            .byte_view(&view, 4, &[2, 2], ScalarType::F32)
            .is_err()
    );
    let transpose = device
        .prepare(Kernel::Tensor(TensorOperation::Transpose(vec![1, 0])))?
        .unwrap();
    let transposed = device
        .dispatch(&transpose, &[&view], &device.stream())?
        .unwrap()
        .output
        .into_one()
        .unwrap();
    assert!(!transposed.layout().is_contiguous(transposed.shape()));
    assert_eq!(device.read_f32(&transposed)?, [1.25, 3.75, -2.5, 8.]);
    let kernel = device
        .prepare(Kernel::Tensor(TensorOperation::Binary(Binary::Add)))?
        .unwrap();
    let resources = device.resources(&[&view])?;
    assert!(foreign.resources(&[&view]).is_err());
    let foreign_tensor = foreign.pull(
        HostTensor {
            bytes: &bytes,
            shape: &[6],
            dtype: ScalarType::F32,
        }
        .into(),
        ImportOptions::default(),
    )?;
    let foreign_resources = foreign.resources(&[&foreign_tensor])?;
    let stream = device.stream();
    assert!(
        device
            .dispatch_indirect(&kernel, &[&view, &view], &foreign_resources, &stream)
            .is_err()
    );
    assert!(
        device
            .dispatch_indirect(&kernel, &[&view, &view], &resources, &foreign.stream())
            .is_err()
    );
    let result = device
        .dispatch_indirect(&kernel, &[&view, &view], &resources, &stream)?
        .unwrap();
    drop(resources);
    drop(view);
    device.wait(&result.completion)?;
    assert!(foreign.wait(&result.completion).is_err());
    let output = result.output.into_one().unwrap();
    let oracle = Array::from_slice(&values[1..5], &[2, 2]);
    let oracle = oracle.add(&oracle)?.to_vec1::<f32>()?;
    assert_eq!(device.read_f32(&output)?, oracle);
    Ok(())
}
