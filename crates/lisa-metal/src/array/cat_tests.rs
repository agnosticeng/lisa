//! A concatenation collectively covers its output even though each individual
//! copy writes only one slice. Poisoned pool reuse makes missing writes visible.
use std::sync::Arc;

use super::{Array, Dtype, Layout};
use crate::error::Result;
use crate::runtime::{
    MetalRuntime, ProgramInput, ProgramOutput, capture_dispatch_program,
    dispatch_program_replay_count,
};

struct Fixture {
    array: Array,
    bytes: Vec<u8>,
}

fn fixture(
    rt: &Arc<MetalRuntime>,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    dtype: Dtype,
    salt: usize,
) -> Result<Fixture> {
    assert_eq!(shape.len(), strides.len());
    let end = if shape.contains(&0) {
        offset
    } else {
        offset
            + shape
                .iter()
                .zip(strides)
                .map(|(&d, &s)| (d - 1) * s)
                .sum::<usize>()
            + 1
    };
    // Include unused input storage before/after the view; the CPU oracle reads
    // the view's original coordinates rather than an already-materialized copy.
    let mut bytes = Vec::new();
    for i in 0..end + 7 {
        let integer = (i as u64).wrapping_mul(0x1020_3040_5060_7081)
            ^ (salt as u64).wrapping_mul(0xaabb_ccdd_eeff_0103);
        let float = if i % 17 == 0 {
            -0.0
        } else {
            (((i * 17 + salt * 29) % 1021) as f32 - 510.0) / 32.0
        };
        match dtype {
            Dtype::Bool => bytes.push((integer & 1) as u8),
            Dtype::U8 | Dtype::I8 => bytes.push(integer as u8),
            Dtype::U16 | Dtype::I16 => bytes.extend_from_slice(&(integer as u16).to_ne_bytes()),
            Dtype::U32 | Dtype::I32 => bytes.extend_from_slice(&(integer as u32).to_ne_bytes()),
            Dtype::I64 => bytes.extend_from_slice(&integer.to_ne_bytes()),
            Dtype::F16 => {
                bytes.extend_from_slice(&half::f16::from_f32(float).to_bits().to_ne_bytes())
            }
            Dtype::BF16 => {
                bytes.extend_from_slice(&half::bf16::from_f32(float).to_bits().to_ne_bytes())
            }
            Dtype::F32 => bytes.extend_from_slice(&float.to_ne_bytes()),
            _ => panic!("unsupported copy_gg test dtype {dtype:?}"),
        }
    }
    let buffer = rt.buffer_for_full_write(bytes.len(), "cat_test_input")?;
    // SAFETY: fresh full-write buffer of at least `bytes.len()` bytes with no GPU user.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents(), bytes.len()) };
    Ok(Fixture {
        array: Array::wrap(
            Arc::clone(rt),
            buffer,
            Layout {
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset,
            },
            dtype,
        ),
        bytes,
    })
}

fn contiguous_fixture(
    rt: &Arc<MetalRuntime>,
    shape: &[usize],
    dtype: Dtype,
    salt: usize,
) -> Result<Fixture> {
    fixture(
        rt,
        shape,
        &Layout::contiguous(shape).strides,
        3,
        dtype,
        salt,
    )
}

fn expected(fixtures: &[Fixture], axis: usize) -> (Vec<usize>, Vec<u8>) {
    let first = &fixtures[0].array;
    let mut shape = first.shape().to_vec();
    shape[axis] = fixtures.iter().map(|f| f.array.dim(axis)).sum();
    let outer = shape[..axis].iter().product::<usize>();
    let inner = shape[axis + 1..].iter().product::<usize>();
    let size = first.dtype().size_of();
    let mut bytes = Vec::with_capacity(shape.iter().product::<usize>() * size);
    for outer_index in 0..outer {
        for fixture in fixtures {
            let layout = fixture.array.layout();
            for axis_index in 0..fixture.array.dim(axis) {
                for inner_index in 0..inner {
                    let mut linear =
                        (outer_index * fixture.array.dim(axis) + axis_index) * inner + inner_index;
                    let mut address = layout.offset;
                    for (&dim, &stride) in layout.shape.iter().zip(&layout.strides).rev() {
                        address += (linear % dim) * stride;
                        linear /= dim;
                    }
                    bytes.extend_from_slice(&fixture.bytes[address * size..(address + 1) * size]);
                }
            }
        }
    }
    (shape, bytes)
}

fn poison(rt: &Arc<MetalRuntime>, bytes: usize) -> Result<usize> {
    let buffer = rt.buffer(bytes, "cat_test_poison")?;
    // SAFETY: fresh lease with no GPU user; the write stays within `length()`.
    unsafe { std::ptr::write_bytes(buffer.contents(), 0xa5, buffer.length()) };
    Ok(buffer.id())
}

fn payload(array: &Array) -> Vec<u8> {
    assert!(array.layout().is_contiguous());
    // SAFETY: callers synchronize first; the contiguous logical range lies in the buffer.
    unsafe {
        std::slice::from_raw_parts(
            array
                .buffer()
                .contents()
                .add(array.layout().offset * array.dtype().size_of()),
            array.size() * array.dtype().size_of(),
        )
        .to_vec()
    }
}

fn assert_padding(array: &Array, value: u8) {
    let bytes = array.size() * array.dtype().size_of();
    assert!(
        // SAFETY: the idle buffer's padding `[bytes, length())` lies within its allocation.
        unsafe {
            std::slice::from_raw_parts(
                array.buffer().contents().add(bytes),
                array.buffer().length() - bytes,
            )
        }
        .iter()
        .all(|&byte| byte == value),
        "cat wrote or cleared allocation padding"
    );
}

fn check(rt: &Arc<MetalRuntime>, fixtures: Vec<Fixture>, axis: usize) -> Result<()> {
    let (shape, bytes) = expected(&fixtures, axis);
    rt.synchronize()?;
    let seeded_id = poison(rt, bytes.len())?;
    let arrays = fixtures.iter().map(|f| f.array.clone()).collect::<Vec<_>>();
    let output = Array::cat(&arrays, axis)?;
    assert_eq!(
        output.buffer().id(),
        seeded_id,
        "test must actually reuse poison"
    );
    assert!(
        arrays
            .iter()
            .all(|a| a.buffer().id() != output.buffer().id())
    );
    rt.synchronize()?;
    assert_eq!(output.shape(), shape);
    assert_eq!(
        payload(&output),
        bytes,
        "dtype {:?}, axis {axis}, shape {shape:?}",
        output.dtype()
    );
    assert_padding(&output, if output.size() == 0 { 0 } else { 0xa5 });
    for fixture in &fixtures {
        assert_eq!(
            // SAFETY: the idle input buffer holds at least `fixture.bytes.len()` bytes.
            unsafe {
                std::slice::from_raw_parts(fixture.array.buffer().contents(), fixture.bytes.len())
            },
            fixture.bytes,
            "input storage must remain immutable"
        );
    }
    Ok(())
}

#[test]
fn cat_overwrites_poison_for_all_copy_dtypes_axes_and_strided_views() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(16)?);
    for dtype in [
        Dtype::Bool,
        Dtype::U8,
        Dtype::U16,
        Dtype::U32,
        Dtype::I8,
        Dtype::I16,
        Dtype::I32,
        Dtype::I64,
        Dtype::F16,
        Dtype::F32,
        Dtype::BF16,
    ] {
        check(
            &rt,
            vec![
                fixture(&rt, &[5], &[2], 3, dtype, 1)?,
                fixture(&rt, &[3], &[3], 5, dtype, 2)?,
            ],
            0,
        )?;
        check(
            &rt,
            vec![
                fixture(&rt, &[2, 5], &[1, 4], 3, dtype, 1)?,
                fixture(&rt, &[3, 5], &[12, 2], 5, dtype, 2)?,
            ],
            0,
        )?;
        check(
            &rt,
            vec![
                fixture(&rt, &[3, 2], &[7, 2], 3, dtype, 1)?,
                fixture(&rt, &[3, 5], &[1, 5], 5, dtype, 2)?,
            ],
            1,
        )?;
        check(
            &rt,
            vec![
                fixture(&rt, &[2, 3, 5], &[43, 12, 2], 3, dtype, 1)?,
                fixture(&rt, &[1, 3, 5], &[1, 7, 21], 5, dtype, 2)?,
            ],
            0,
        )?;
        check(
            &rt,
            vec![
                fixture(&rt, &[2, 2, 5], &[36, 12, 2], 3, dtype, 1)?,
                fixture(&rt, &[2, 3, 5], &[1, 10, 2], 5, dtype, 2)?,
            ],
            1,
        )?;
        check(
            &rt,
            vec![
                fixture(&rt, &[2, 3, 2], &[41, 11, 3], 3, dtype, 1)?,
                fixture(&rt, &[2, 3, 4], &[1, 8, 2], 5, dtype, 2)?,
            ],
            2,
        )?;
        // One input remains an independent copy, including broadcast strides.
        check(
            &rt,
            vec![fixture(&rt, &[2, 3, 4], &[0, 4, 1], 3, dtype, 1)?],
            1,
        )?;
    }
    Ok(())
}

#[test]
fn cat_preserves_empty_slices_and_validation() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(16)?);
    for dtype in [Dtype::U8, Dtype::F32, Dtype::BF16] {
        check(
            &rt,
            vec![
                contiguous_fixture(&rt, &[2, 0, 4], dtype, 0)?,
                contiguous_fixture(&rt, &[2, 3, 4], dtype, 1)?,
                contiguous_fixture(&rt, &[2, 0, 4], dtype, 2)?,
                contiguous_fixture(&rt, &[2, 1, 4], dtype, 3)?,
                contiguous_fixture(&rt, &[2, 0, 4], dtype, 4)?,
            ],
            1,
        )?;
        check(
            &rt,
            vec![
                contiguous_fixture(&rt, &[2, 0, 4], dtype, 0)?,
                contiguous_fixture(&rt, &[2, 0, 4], dtype, 1)?,
            ],
            1,
        )?;
        check(
            &rt,
            vec![
                contiguous_fixture(&rt, &[0, 2, 4], dtype, 0)?,
                contiguous_fixture(&rt, &[0, 3, 4], dtype, 1)?,
            ],
            1,
        )?;
    }
    let a = contiguous_fixture(&rt, &[2, 3], Dtype::F32, 0)?.array;
    let assert_error = |arrays: &[Array], axis: usize, message: &str| match Array::cat(arrays, axis)
    {
        Ok(_) => panic!("expected {message}"),
        Err(error) => assert!(error.to_string().contains(message), "{error}"),
    };
    assert_error(&[], 0, "cat: no arrays");
    assert_error(std::slice::from_ref(&a), 2, "cat: axis out of range");
    assert_error(
        &[
            a.clone(),
            contiguous_fixture(&rt, &[2, 3], Dtype::BF16, 1)?.array,
        ],
        0,
        "cat: rank/dtype mismatch",
    );
    assert_error(
        &[
            a.clone(),
            contiguous_fixture(&rt, &[3], Dtype::F32, 1)?.array,
        ],
        0,
        "cat: rank/dtype mismatch",
    );
    assert_error(
        &[a, contiguous_fixture(&rt, &[2, 4], Dtype::F32, 1)?.array],
        0,
        "cat: dim mismatch",
    );
    let unsupported = fixture(&rt, &[2, 2, 2, 2], &[100, 30, 8, 2], 3, Dtype::F32, 0)?;
    assert_error(&[unsupported.array], 0, "cat: collapsed rank 4 unsupported");
    Ok(())
}

#[test]
fn cat_overwrites_large_strided_kv_append_without_clearing_padding() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(16)?);
    check(
        &rt,
        vec![
            fixture(
                &rt,
                &[1, 8, 1024, 128],
                &[8 * 1033 * 128, 1033 * 128, 128, 1],
                3 * 128 + 17,
                Dtype::BF16,
                1,
            )?,
            fixture(
                &rt,
                &[1, 8, 7, 128],
                &[8 * 11 * 128, 11 * 128, 128, 1],
                128 + 17,
                Dtype::BF16,
                2,
            )?,
        ],
        2,
    )
}

#[test]
fn cat_keeps_pending_producers_and_live_outputs_ordered() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(2)?);
    rt.commands.set_per_buffer(1);
    let fixtures = [
        contiguous_fixture(&rt, &[259], Dtype::F32, 1)?,
        contiguous_fixture(&rt, &[269], Dtype::F32, 2)?,
        contiguous_fixture(&rt, &[257], Dtype::F32, 3)?,
    ];
    let (_, first_expected) = expected(&fixtures[..2], 0);
    let (_, second_expected) = expected(&fixtures, 0);
    let first_seed = poison(&rt, first_expected.len())?;
    // Retain this lease until the first allocation has taken its smaller seed.
    let second_seed = rt.buffer(second_expected.len(), "cat_test_next_poison")?;
    // SAFETY: fresh lease with no GPU user; the write stays within `length()`.
    unsafe { std::ptr::write_bytes(second_seed.contents(), 0xa5, second_seed.length()) };
    let second_id = second_seed.id();
    let first = Array::cat(&[fixtures[0].array.clone(), fixtures[1].array.clone()], 0)?;
    assert_eq!(first.buffer().id(), first_seed);
    drop(second_seed);
    let second = Array::cat(&[first.clone(), fixtures[2].array.clone()], 0)?;
    assert_eq!(second.buffer().id(), second_id);
    // There is deliberately no wait/readback between the producer and consumer.
    rt.synchronize()?;
    assert_eq!(payload(&first), first_expected);
    assert_eq!(payload(&second), second_expected);
    assert_padding(&first, 0xa5);
    assert_padding(&second, 0xa5);
    Ok(())
}

fn program_inputs(fixtures: &[Fixture]) -> Vec<ProgramInput> {
    fixtures
        .iter()
        .map(|fixture| ProgramInput {
            buffer: fixture.array.buffer().clone(),
            required_bytes: fixture.bytes.len(),
        })
        .collect()
}

#[test]
fn cat_program_replay_rebinds_strided_inputs_and_overwrites_poison() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(2)?);
    rt.commands.set_per_buffer(1);
    let fixtures = [
        fixture(&rt, &[2, 3, 4], &[37, 9, 2], 3, Dtype::BF16, 1)?,
        fixture(&rt, &[2, 2, 4], &[1, 16, 3], 5, Dtype::BF16, 2)?,
    ];
    let replacement = [
        fixture(&rt, &[2, 3, 4], &[37, 9, 2], 3, Dtype::BF16, 7)?,
        fixture(&rt, &[2, 2, 4], &[1, 16, 3], 5, Dtype::BF16, 8)?,
    ];
    let (shape, captured_expected) = expected(&fixtures, 1);
    let (_, replay_expected) = expected(&replacement, 1);
    let first_seed = poison(&rt, captured_expected.len())?;
    let (captured, program) =
        capture_dispatch_program(&rt, &program_inputs(&fixtures), || -> Result<_> {
            let array = Array::cat(&[fixtures[0].array.clone(), fixtures[1].array.clone()], 1)?;
            let output = ProgramOutput {
                buffer: array.buffer().clone(),
                offset: 0,
            };
            Ok((array, vec![output]))
        })?;
    assert_eq!(captured.buffer().id(), first_seed);
    let program = program.expect("cat must remain a GPU-only recordable operation");
    assert_eq!(program.dispatch_count(), 2);
    rt.synchronize()?;
    let replay_seed = poison(&rt, replay_expected.len())?;
    let before = dispatch_program_replay_count();
    let output = program
        .replay(&rt, &program_inputs(&replacement))?
        .expect("same-layout cat replay");
    assert!(
        dispatch_program_replay_count() > before,
        "fallback does not certify replay"
    );
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].offset, 0);
    assert_eq!(output[0].buffer.id(), replay_seed);
    assert_ne!(captured.buffer().id(), output[0].buffer.id());
    let replayed = Array::from_parts(&rt, output[0].buffer.clone(), &shape, Dtype::BF16);
    rt.synchronize()?;
    assert_eq!(payload(&replayed), replay_expected);
    assert_eq!(
        payload(&captured),
        captured_expected,
        "live capture output was overwritten"
    );
    assert_padding(&replayed, 0xa5);
    assert_padding(&captured, 0xa5);
    Ok(())
}
