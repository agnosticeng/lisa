use super::*;
use crate::runtime::MetalRuntime;

#[test]
fn audited_custom_output_slots_preserve_padding_and_zero_unused_slots() {
    let rt = std::sync::Arc::new(MetalRuntime::new(1024).unwrap());
    let count = 259usize;
    // Best-fit reuse may return much larger storage than the logical output.
    // The audited slot must overwrite the logical range without clearing its
    // padding; an unused slot still receives its original initialized logical
    // range. A reused allocation's excess capacity is not tensor data.
    let first = rt.buffer(65536, "custom_output_poison_a").unwrap();
    let second = rt.buffer(65536, "custom_output_poison_b").unwrap();
    let ids = [first.id(), second.id()];
    // SAFETY: fresh leases with no GPU user; writes stay within each `length()`.
    unsafe {
        std::ptr::write_bytes(first.contents(), 0xa5, first.length());
        std::ptr::write_bytes(second.contents(), 0xa5, second.length());
    }
    drop(first);
    drop(second);
    let kernel = MetalKernel::new(
        "audited_slot_test",
        &[],
        &["O", "UNUSED"],
        "const uint i = thread_position_in_grid.x; O[i] = float(i) * 3.0f + 0.5f;",
        "",
        false,
        false,
    )
    .unwrap()
    .with_audited_full_write_output_indices(&[0]);
    let spec = OutputArg {
        shape: vec![count as i32],
        dtype: crate::Dtype::Float32,
    };
    let out = kernel
        .apply(
            &rt,
            &[],
            &[],
            (count as i32, 1, 1),
            (64, 1, 1),
            &[spec.clone(), spec],
            None,
        )
        .unwrap();
    assert!(ids.contains(&out[0].buffer().id()));
    assert!(ids.contains(&out[1].buffer().id()));
    assert_ne!(out[0].buffer().id(), out[1].buffer().id());
    rt.synchronize().unwrap();
    let a = out[0].buffer();
    let b = out[1].buffer();
    // SAFETY: synchronized above; each output buffer holds at least `count` f32 values.
    unsafe {
        assert_eq!(
            std::slice::from_raw_parts(a.contents().cast::<f32>(), count),
            (0..count).map(|i| i as f32 * 3. + 0.5).collect::<Vec<_>>()
        );
        assert!(
            std::slice::from_raw_parts(a.contents().add(count * 4), a.length() - count * 4)
                .iter()
                .all(|&v| v == 0xa5)
        );
        assert!(
            std::slice::from_raw_parts(b.contents(), count * 4)
                .iter()
                .all(|&v| v == 0)
        );
    }
}

#[test]
fn audited_custom_outputs_keep_completion_gating_without_ordered_scope() {
    let rt = std::sync::Arc::new(MetalRuntime::new(1024).unwrap());
    let kernel = MetalKernel::new(
        "audited_completion_test",
        &[],
        &["O"],
        "O[thread_position_in_grid.x] = 19.0f;",
        "",
        false,
        false,
    )
    .unwrap()
    .with_audited_full_write_outputs();
    let outputs = [OutputArg {
        shape: vec![259],
        dtype: crate::Dtype::Float32,
    }];
    let first = kernel
        .apply(&rt, &[], &[], (259, 1, 1), (64, 1, 1), &outputs, None)
        .unwrap();
    let id = first[0].buffer().id();
    drop(first);
    // Leave the command buffer unsubmitted so a fast completion cannot hide
    // unsafe reuse in the absence of an ordered-reuse scope.
    rt.commands.flush_encoder();
    let second = kernel
        .apply(&rt, &[], &[], (259, 1, 1), (64, 1, 1), &outputs, None)
        .unwrap();
    assert_ne!(second[0].buffer().id(), id);
    rt.synchronize().unwrap();
    // SAFETY: synchronized above; the output buffer holds at least 259 f32 values.
    unsafe {
        assert!(
            std::slice::from_raw_parts(second[0].buffer().contents().cast::<f32>(), 259)
                .iter()
                .all(|&v| v == 19.)
        );
    }
}

#[test]
#[should_panic(expected = "atomic outputs require initialization")]
fn audited_custom_atomic_outputs_are_rejected() {
    let _ = MetalKernel::new("audited_atomic_rejection", &[], &["O"], "", "", false, true)
        .unwrap()
        .with_audited_full_write_outputs();
}
