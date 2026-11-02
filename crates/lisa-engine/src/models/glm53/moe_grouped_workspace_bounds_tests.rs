use lisa_mlx::jit::{self, DType, Tensor};

#[test]
fn grouped_moe_workspace_rejects_aliased_wrong_dtype_shape_and_bounds() {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-workspace-bounds] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 96 * 8;
    const WIDTH: usize = 128;
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let device = stream.runtime();
    let gate = Tensor::zeros(device, &[ROWS, WIDTH], DType::F32).unwrap();
    let up = Tensor::zeros(device, &[ROWS, WIDTH], DType::F32).unwrap();
    let weights = Tensor::zeros(device, &[ROWS], DType::F32).unwrap();
    let alias = Tensor::from_parts(
        device,
        gate.buffer_and_layout().0.clone(),
        &[ROWS, WIDTH],
        DType::F16,
    );
    let wrong_dtype = Tensor::zeros(device, &[ROWS, WIDTH], DType::F32).unwrap();
    let wrong_shape = Tensor::zeros(device, &[ROWS, WIDTH + 1], DType::F16).unwrap();
    let short = Tensor::from_parts(
        device,
        device.buffer(2, "short_workspace_test").unwrap(),
        &[ROWS, WIDTH],
        DType::F16,
    );
    let offset = Tensor::from_parts_offset(
        device,
        device
            .buffer((ROWS * WIDTH + 1) * 2, "offset_workspace_test")
            .unwrap(),
        &[ROWS, WIDTH],
        DType::F16,
        1,
    );
    for (reason, output) in [
        ("input alias", alias),
        ("dtype", wrong_dtype),
        ("shape", wrong_shape),
        ("bounds", short),
        ("offset", offset),
    ] {
        let result = jit::glm53_moe_grouped_swiglu_weight_prefill_with_output(
            device,
            &gate,
            &up,
            &weights,
            ROWS,
            WIDTH,
            10.0,
            Some(&output),
        );
        assert!(result.is_err(), "workspace must reject {reason}");
    }
}
