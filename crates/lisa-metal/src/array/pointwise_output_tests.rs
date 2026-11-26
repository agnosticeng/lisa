use super::*;
use crate::runtime::{
    ProgramInput, ProgramOutput, capture_dispatch_program, dispatch_program_replay_count,
};

fn input(rt: &Arc<MetalRuntime>, dtype: Dtype, shape: &[usize], offset: usize) -> Result<Array> {
    let n = shape.iter().product::<usize>();
    let count = n + offset + 3;
    let values = (0..count)
        .map(|i| match i % 11 {
            0 => -0.0,
            1 => f32::from_bits(0x7fc00005),
            2 => f32::INFINITY,
            3 => f32::NEG_INFINITY,
            4 => f32::from_bits(1),
            _ => (i % 31) as f32 / 7. - 2.,
        })
        .collect::<Vec<_>>();
    let flat = match dtype {
        Dtype::F32 => Array::from_slice_dt(rt, &values, &[count], dtype)?,
        Dtype::F16 => Array::from_slice_dt(
            rt,
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
            &[count],
            dtype,
        )?,
        Dtype::BF16 => Array::from_slice_dt(
            rt,
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
            &[count],
            dtype,
        )?,
        Dtype::I32 => Array::from_slice_dt(
            rt,
            &(0..count).map(|i| i as i32 % 251 - 125).collect::<Vec<_>>(),
            &[count],
            dtype,
        )?,
        Dtype::I64 => Array::from_slice_dt(
            rt,
            &(0..count).map(|i| i as i64 % 251 - 125).collect::<Vec<_>>(),
            &[count],
            dtype,
        )?,
        Dtype::U32 => Array::from_slice_dt(
            rt,
            &(0..count).map(|i| i as u32 % 251).collect::<Vec<_>>(),
            &[count],
            dtype,
        )?,
        Dtype::U8 | Dtype::Bool => Array::from_slice_dt(
            rt,
            &(0..count)
                .map(|i| {
                    if dtype == Dtype::Bool {
                        (i % 2) as u8
                    } else {
                        (i % 251) as u8
                    }
                })
                .collect::<Vec<_>>(),
            &[count],
            dtype,
        )?,
        _ => unreachable!(),
    };
    flat.narrow(0, offset, n)?.reshape(shape)
}

fn poison(rt: &Arc<MetalRuntime>, bytes: usize) -> Result<Vec<usize>> {
    rt.synchronize()?;
    // Two leases also cover unary's contiguous materialization of a strided
    // input. Keep both alive while allocating so they cannot alias each other.
    let buffers = (0..2)
        .map(|_| rt.buffer(bytes, "pointwise_poison"))
        .collect::<Result<Vec<_>>>()?;
    let ids = buffers.iter().map(|b| b.id()).collect();
    for buffer in buffers {
        // SAFETY: fresh lease with no GPU user; the write stays within `length()`.
        unsafe { std::ptr::write_bytes(buffer.contents(), 0xa5, buffer.length()) };
    }
    Ok(ids)
}

fn bytes(a: &Array) -> Vec<u8> {
    assert!(a.layout.is_contiguous());
    let start = a.layout.offset * a.dtype.size_of();
    // SAFETY: `check` synchronizes first; the contiguous logical range lies in the buffer.
    unsafe { std::slice::from_raw_parts(a.buf.contents().add(start), a.size() * a.dtype.size_of()) }
        .to_vec()
}

fn check(actual: &Array, oracle: &Array, poisoned: &[usize]) -> Result<()> {
    actual.rt.synchronize()?;
    assert!(
        poisoned.contains(&actual.buf.id()),
        "output must reuse a completed poisoned allocation"
    );
    assert_eq!(actual.shape(), oracle.shape());
    assert_eq!(actual.dtype(), oracle.dtype());
    assert_eq!(
        bytes(actual),
        bytes(oracle),
        "logical output differs from the original zeroed allocation path"
    );
    let logical = actual.size() * actual.dtype.size_of();
    assert!(
        // SAFETY: synchronized above; the padding after `logical` lies within the allocation.
        unsafe {
            std::slice::from_raw_parts(
                actual.buf.contents().add(logical),
                actual.buf.length() - logical,
            )
        }
        .iter()
        .all(|&b| b == 0xa5),
        "unused bucket padding must remain poisoned"
    );
    Ok(())
}

#[test]
fn unary_audited_output_matches_zeroed_shader_and_preserves_poisoned_padding() -> Result<()> {
    for dtype in [
        Dtype::F32,
        Dtype::F16,
        Dtype::BF16,
        Dtype::I32,
        Dtype::I64,
        Dtype::U32,
        Dtype::U8,
        Dtype::Bool,
    ] {
        let op = if dtype == Dtype::Bool {
            "LogicalNot"
        } else if dtype.is_float() {
            "Tanh"
        } else {
            "Negative"
        };
        for n in [1, 257, 65537] {
            let rt = Arc::new(MetalRuntime::new(1024)?);
            let x = input(&rt, dtype, &[n], 3)?;
            let oracle = x.unary_with_output_policy::<false>(op)?;
            let poisoned = poison(&rt, n * dtype.size_of())?;
            let actual = x.unary(op)?;
            check(&actual, &oracle, &poisoned)?;
        }
        let rt = Arc::new(MetalRuntime::new(1024)?);
        let x = input(&rt, dtype, &[17, 19], 3)?.permute(&[1, 0])?;
        let oracle = x.unary_with_output_policy::<false>(op)?;
        let poisoned = poison(&rt, x.size() * dtype.size_of())?;
        check(&x.unary(op)?, &oracle, &poisoned)?;
    }
    // Both sides of the work-per-thread switch and the real uncapped head tail.
    for n in [65535, 65536, 200058] {
        let rt = Arc::new(MetalRuntime::new(1024)?);
        let x = input(&rt, Dtype::F32, &[n], 3)?;
        let oracle = x.unary_with_output_policy::<false>("Tanh")?;
        let poisoned = poison(&rt, n * 4)?;
        check(&x.unary("Tanh")?, &oracle, &poisoned)?;
    }
    let rt = Arc::new(MetalRuntime::new(1024)?);
    let empty = input(&rt, Dtype::F32, &[0], 0)?;
    assert!(empty.unary("Tanh").is_err());
    assert!(empty.unary_with_output_policy::<false>("Tanh").is_err());
    Ok(())
}

#[test]
fn binary_audited_output_matches_zeroed_shader_for_broadcasts_strides_and_tails() -> Result<()> {
    for dtype in [
        Dtype::F32,
        Dtype::F16,
        Dtype::BF16,
        Dtype::I64,
        Dtype::U32,
        Dtype::U8,
    ] {
        for n in [0, 1, 257, 65537] {
            let rt = Arc::new(MetalRuntime::new(1024)?);
            let x = input(&rt, dtype, &[n], 3)?;
            let y = input(&rt, dtype, &[], 7)?;
            let oracle = x.binary_with_output_policy::<false>(&y, "bmul")?;
            let poisoned = poison(&rt, n * dtype.size_of())?;
            check(&x.binary(&y, "bmul")?, &oracle, &poisoned)?;
        }
        let rt = Arc::new(MetalRuntime::new(1024)?);
        let x = input(&rt, dtype, &[17, 19], 3)?.permute(&[1, 0])?;
        let y = input(&rt, dtype, &[1, 17], 5)?;
        let oracle = x.binary_with_output_policy::<false>(&y, "badd")?;
        let poisoned = poison(&rt, x.size() * dtype.size_of())?;
        check(&x.binary(&y, "badd")?, &oracle, &poisoned)?;
    }
    let rt = Arc::new(MetalRuntime::new(1024)?);
    for dtype in [Dtype::Bool, Dtype::I32] {
        let x = input(&rt, dtype, &[3], 0)?;
        assert!(
            x.binary(&x, "badd").is_err(),
            "preserve unsupported binary dtypes"
        );
        assert!(x.binary_with_output_policy::<false>(&x, "badd").is_err());
    }
    Ok(())
}

#[test]
fn audited_pointwise_dispatch_replay_preserves_dynamic_inputs_and_live_outputs() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(1)?);
    let first = input(&rt, Dtype::F32, &[257], 3)?;
    let changed = (0..263)
        .map(|i| (i % 29) as f32 / 8. - 1.75)
        .collect::<Vec<_>>();
    let second = Array::from_slice_dt(&rt, &changed, &[263], Dtype::F32)?.narrow(0, 3, 257)?;
    let scale = input(&rt, Dtype::F32, &[], 7)?;
    let slot = |a: &Array| ProgramInput {
        buffer: a.buf.clone(),
        required_bytes: a.buf.length(),
    };
    // Keep both origins fixed for replay while changing the input values.
    let run = |x: &Array| x.binary(&scale, "bmul")?.unary("Tanh");
    let (captured, program) = capture_dispatch_program(&rt, &[slot(&first), slot(&scale)], || {
        let out = run(&first)?;
        let view = ProgramOutput {
            buffer: out.buf.clone(),
            offset: 0,
        };
        Ok::<_, crate::error::Error>((out, vec![view]))
    })?;
    let program = program.expect("pointwise chain must be recordable");
    let before = dispatch_program_replay_count();
    let replayed = program
        .replay(&rt, &[slot(&second), slot(&scale)])?
        .expect("fixed geometry must replay");
    assert!(dispatch_program_replay_count() > before);
    let replayed = Array::from_parts(&rt, replayed[0].buffer.clone(), &[257], Dtype::F32);
    let oracle = second
        .binary_with_output_policy::<false>(&scale, "bmul")?
        .unary_with_output_policy::<false>("Tanh")?;
    let original = first
        .binary_with_output_policy::<false>(&scale, "bmul")?
        .unary_with_output_policy::<false>("Tanh")?;
    rt.synchronize()?;
    assert_ne!(
        captured.buf.id(),
        replayed.buf.id(),
        "live capture output cannot be reused"
    );
    assert_eq!(bytes(&replayed), bytes(&oracle));
    assert_eq!(bytes(&captured), bytes(&original));
    Ok(())
}
