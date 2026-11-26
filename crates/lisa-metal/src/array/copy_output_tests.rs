//! Full-write materialization must overwrite reused logical storage without
//! clearing padding or changing the source's offset/stride interpretation.
use std::sync::Arc;

use crate::array::{Array, Dtype, Layout};
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
            _ => panic!("copy_strided has no shader for {dtype:?}"),
        }
    }
    let buffer = rt.buffer_for_full_write(bytes.len(), "copy_test_input")?;
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

fn expected(f: &Fixture) -> Vec<u8> {
    let a = &f.array;
    let size = a.dtype().size_of();
    let mut bytes = Vec::with_capacity(a.size() * size);
    for mut linear in 0..a.size() {
        let mut address = a.layout().offset;
        for (&dim, &stride) in a.shape().iter().zip(a.stride()).rev() {
            address += (linear % dim) * stride;
            linear /= dim;
        }
        bytes.extend_from_slice(&f.bytes[address * size..(address + 1) * size]);
    }
    bytes
}

fn poison(rt: &Arc<MetalRuntime>, bytes: usize) -> Result<usize> {
    let buffer = rt.buffer(bytes, "copy_test_poison")?;
    // SAFETY: fresh lease with no GPU user; the write stays within `length()`.
    unsafe { std::ptr::write_bytes(buffer.contents(), 0xa5, buffer.length()) };
    Ok(buffer.id())
}

fn payload(a: &Array) -> Vec<u8> {
    assert!(a.layout().is_contiguous());
    // SAFETY: callers synchronize first; the contiguous logical range lies in the buffer.
    unsafe {
        std::slice::from_raw_parts(
            a.buffer()
                .contents()
                .add(a.layout().offset * a.dtype().size_of()),
            a.size() * a.dtype().size_of(),
        )
        .to_vec()
    }
}

fn assert_padding(a: &Array, value: u8) {
    let bytes = a.size() * a.dtype().size_of();
    assert!(
        // SAFETY: the idle buffer's padding `[bytes, length())` lies within its allocation.
        unsafe {
            std::slice::from_raw_parts(
                a.buffer().contents().add(bytes),
                a.buffer().length() - bytes,
            )
        }
        .iter()
        .all(|&byte| byte == value),
        "copy wrote or cleared allocation padding"
    );
}

fn assert_source(f: &Fixture) {
    assert_eq!(
        // SAFETY: the idle source buffer holds at least `f.bytes.len()` bytes.
        unsafe { std::slice::from_raw_parts(f.array.buffer().contents(), f.bytes.len()) },
        f.bytes,
        "copy changed original source storage"
    );
}

fn check(rt: &Arc<MetalRuntime>, f: Fixture) -> Result<()> {
    rt.synchronize()?;
    let wanted = expected(&f);
    let seeded = poison(rt, wanted.len())?;
    let output = f.array.copied()?;
    assert_eq!(output.buffer().id(), seeded, "must reuse poisoned storage");
    assert_ne!(output.buffer().id(), f.array.buffer().id());
    assert_eq!(output.layout(), &Layout::contiguous(f.array.shape()));
    assert_eq!(output.dtype(), f.array.dtype());
    rt.synchronize()?;
    assert_eq!(payload(&output), wanted);
    assert_padding(&output, if output.size() == 0 { 0 } else { 0xa5 });
    assert_source(&f);
    Ok(())
}

#[test]
fn copied_overwrites_poison_for_every_supported_dtype_layout_and_tail() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(16)?);
    // Every copy_strided instantiation, plus Bool's existing uint8 storage.
    // Float64 is intentionally absent: the shader has no double entry point.
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
        check(&rt, fixture(&rt, &[], &[], 3, dtype, 1)?)?;
        for n in [1, 255, 256, 257, 513] {
            check(&rt, fixture(&rt, &[n], &[2], 5, dtype, n)?)?;
        }
        check(&rt, fixture(&rt, &[2, 5], &[1, 4], 3, dtype, 2)?)?;
        check(&rt, fixture(&rt, &[2, 3, 4], &[37, 9, 2], 7, dtype, 3)?)?;
        check(&rt, fixture(&rt, &[3, 5], &[0, 2], 5, dtype, 4)?)?;
        check(
            &rt,
            fixture(
                &rt,
                &[2, 1, 2, 1, 2, 1, 2, 3],
                &[90, 0, 40, 0, 19, 0, 8, 2],
                3,
                dtype,
                5,
            )?,
        )?;
        // Preserve the old empty path, including its zeroed allocation.
        check(&rt, fixture(&rt, &[2, 0, 3], &[0, 3, 1], 5, dtype, 6)?)?;
    }
    Ok(())
}

#[test]
fn copied_keeps_pending_producers_and_live_aliases_independent() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(usize::MAX)?);
    let f = fixture(&rt, &[3, 257], &[1, 5], 7, Dtype::BF16, 3)?;
    let first = f.array.copied()?;
    let first_id = first.buffer().id();
    let tail = first.narrow(0, 1, 2)?;
    let second = tail.copied()?;
    assert_ne!(first_id, second.buffer().id());
    let expected_first = expected(&f);
    assert!(first.buffer().last_used_cb() > rt.commands.completed_watermark());
    drop(tail);
    drop(first);
    let third = f.array.copied()?;
    // The pool owns the old producer now, but its queued reads have not
    // completed. This audited allocation must not enable ordered reuse.
    assert_ne!(third.buffer().id(), first_id);
    assert_ne!(third.buffer().id(), second.buffer().id());
    rt.synchronize()?;
    assert_eq!(payload(&second), expected_first[257 * 2..]);
    assert_eq!(payload(&third), expected_first);
    assert_source(&f);
    Ok(())
}

fn inputs(f: &Fixture) -> [ProgramInput; 1] {
    [ProgramInput {
        buffer: f.array.buffer().clone(),
        required_bytes: f.bytes.len(),
    }]
}

#[test]
fn copied_program_replay_rebinds_offsets_and_strides_without_clearing() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(2)?);
    rt.commands.set_per_buffer(1);
    for dtype in [Dtype::BF16, Dtype::F32, Dtype::I64, Dtype::U8] {
        let f = fixture(&rt, &[3, 257], &[1, 5], 7, dtype, 1)?;
        let replacement = fixture(&rt, &[3, 257], &[1, 5], 7, dtype, 8)?;
        let wanted = expected(&f);
        let replay_wanted = expected(&replacement);
        let capture_seed = poison(&rt, wanted.len())?;
        let (captured, program) = capture_dispatch_program(&rt, &inputs(&f), || -> Result<_> {
            let array = f.array.copied()?;
            let output = ProgramOutput {
                buffer: array.buffer().clone(),
                offset: 0,
            };
            Ok((array, vec![output]))
        })?;
        assert_eq!(captured.buffer().id(), capture_seed);
        let program = program.expect("copy must seal without a host fallback");
        assert_eq!(program.dispatch_count(), 1);
        rt.synchronize()?;
        let replay_seed = poison(&rt, replay_wanted.len())?;
        let before = dispatch_program_replay_count();
        let outputs = program
            .replay(&rt, &inputs(&replacement))?
            .expect("same-layout copy replay");
        assert!(
            dispatch_program_replay_count() > before,
            "must actually replay"
        );
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].offset, 0);
        assert_eq!(outputs[0].buffer.id(), replay_seed);
        assert_ne!(outputs[0].buffer.id(), captured.buffer().id());
        let replayed = Array::from_parts(&rt, outputs[0].buffer.clone(), f.array.shape(), dtype);
        rt.synchronize()?;
        assert_eq!(payload(&replayed), replay_wanted);
        assert_eq!(payload(&captured), wanted, "live capture output changed");
        assert_padding(&captured, 0xa5);
        assert_padding(&replayed, 0xa5);
        assert_source(&f);
        assert_source(&replacement);
    }
    Ok(())
}

#[test]
fn contiguous_bytes_borrow_the_buffer_and_refuse_strided_views() -> Result<()> {
    let rt = Arc::new(MetalRuntime::new(16)?);
    let data: Vec<u8> = (0..24).collect();
    let a = Array::from_slice_dt(&rt, &data, &[2, 3, 4], Dtype::Uint8)?;
    assert_eq!(a.contiguous_bytes()?, &data[..]);
    assert_eq!(
        a.contiguous_bytes()?.as_ptr(),
        a.buffer_and_layout().0.contents() as *const u8
    );
    assert!(a.transpose(1, 2)?.contiguous_bytes().is_err());
    Ok(())
}
