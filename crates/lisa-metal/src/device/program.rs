//! Tensor metadata bridge for replay of existing fixed-geometry GPU dispatches.
use super::*;
use crate::runtime::{DispatchProgram, ProgramInput, ProgramOutput, capture_dispatch_program};
#[cfg(test)]
mod inline_tests;

#[derive(Clone, PartialEq, Eq)]
struct View {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    dtype: Dtype,
}
impl View {
    fn of(array: &Array) -> Self {
        Self {
            shape: array.t.shape().to_vec(),
            strides: array.t.stride().to_vec(),
            offset: array.t.layout().offset,
            dtype: array.dtype(),
        }
    }
    fn matches(&self, array: &Array) -> bool {
        self.dtype == array.dtype()
            && self.shape == array.t.shape()
            && self.strides == array.t.stride()
            && self.offset == array.t.layout().offset
    }
}
fn input_slot(array: &Array) -> anyhow::Result<ProgramInput> {
    let layout = array.t.layout();
    let elements = if layout.shape.contains(&0) {
        Some(layout.offset)
    } else {
        layout
            .shape
            .iter()
            .zip(&layout.strides)
            .try_fold(layout.offset, |end, (&dim, &stride)| {
                end.checked_add(dim.checked_sub(1)?.checked_mul(stride)?)
            })
            .and_then(|end| end.checked_add(1))
    };
    let required_bytes = elements
        .and_then(|n| n.checked_mul(array.dtype().size_of()))
        .ok_or_else(|| anyhow::anyhow!("program input layout span overflow"))?;
    anyhow::ensure!(
        required_bytes <= array.t.buffer().length(),
        "program input exceeds storage"
    );
    Ok(ProgramInput {
        buffer: array.t.buffer().clone(),
        required_bytes,
    })
}
struct TensorProgram {
    runtime: Arc<crate::runtime::MetalRuntime>,
    program: DispatchProgram,
    inputs: Vec<View>,
    outputs: Vec<View>,
}
impl RecordedProgram<Array> for TensorProgram {
    fn set_scheduling(&mut self, scheduling: ProgramScheduling) -> bool {
        match scheduling {
            ProgramScheduling::Ordered => {
                self.program.disable_parallel();
                true
            }
            ProgramScheduling::DependencyParallel => self.program.enable_parallel(),
        }
    }
    fn replay(&self, inputs: &[&Array]) -> anyhow::Result<Option<Vec<Array>>> {
        if inputs.len() != self.inputs.len()
            || inputs
                .iter()
                .zip(&self.inputs)
                .any(|(a, v)| !Arc::ptr_eq(a.t.runtime(), &self.runtime) || !v.matches(a))
        {
            crate::trace::event("program.replay.layout_declined");
            return Ok(None);
        }
        let slots = inputs
            .iter()
            .map(|a| input_slot(a))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let Some(outputs) = self.program.replay(&self.runtime, &slots)? else {
            crate::trace::event("program.replay.storage_declined");
            return Ok(None);
        };
        crate::trace::event("program.replay.hit");
        Ok(Some(
            outputs
                .into_iter()
                .zip(&self.outputs)
                .map(|(out, view)| {
                    debug_assert_eq!(out.offset, view.offset * view.dtype.size_of());
                    Array::new(crate::array::Array::from_program_parts(
                        &self.runtime,
                        out.buffer,
                        crate::array::Layout {
                            shape: view.shape.clone(),
                            strides: view.strides.clone(),
                            offset: view.offset,
                        },
                        view.dtype,
                    ))
                })
                .collect(),
        ))
    }
}
pub(super) fn capture(
    device: &MetalDevice,
    inputs: &[&Array],
    stream: &Stream,
    encode: &mut dyn FnMut() -> anyhow::Result<Vec<Array>>,
) -> anyhow::Result<ProgramCapture<Array>> {
    anyhow::ensure!(
        Arc::ptr_eq(stream.runtime(), &device.runtime)
            && inputs
                .iter()
                .all(|a| Arc::ptr_eq(a.t.runtime(), &device.runtime)),
        "program capture: foreign device context"
    );
    let slots = inputs
        .iter()
        .map(|a| input_slot(a))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (outputs, program) = capture_dispatch_program(&device.runtime, &slots, || {
        let output = encode()?;
        anyhow::ensure!(
            output
                .iter()
                .all(|a| Arc::ptr_eq(a.t.runtime(), &device.runtime)),
            "program capture: foreign output context"
        );
        let views = output
            .iter()
            .map(|a| ProgramOutput {
                buffer: a.t.buffer().clone(),
                offset: a.t.layout().offset * a.dtype().size_of(),
            })
            .collect();
        Ok::<_, anyhow::Error>((output, views))
    })?;
    crate::trace::event(if program.is_some() {
        "program.capture.sealed"
    } else {
        "program.capture.declined"
    });
    let program = program.map(|program| {
        Box::new(TensorProgram {
            runtime: device.runtime.clone(),
            program,
            inputs: inputs.iter().map(|a| View::of(a)).collect(),
            outputs: outputs.iter().map(View::of).collect(),
        }) as Box<dyn RecordedProgram<Array>>
    });
    Ok((outputs, program))
}

/// Audited wrappers whose values remain on the GPU and whose inline arguments
/// depend only on tensor metadata and the immutable prepared operation.
/// Each encoder acquisition also selects its pipeline and rebinds every used
/// operand, so independently locked program sections may interleave safely.
pub(super) fn supported(kernel: &Kernel) -> bool {
    match kernel {
        Kernel::PackedCompute(op) => matches!(
            op,
            PackedCompute::PlainLinear { .. }
                | PackedCompute::Q8Linear { .. }
                | PackedCompute::Q8Pair
                | PackedCompute::BlockLinear { .. }
                | PackedCompute::GroupedLinear { .. }
                | PackedCompute::ExpertLinear { .. }
                | PackedCompute::Q8ExpertLinear
                | PackedCompute::RmsNorm { .. }
                | PackedCompute::ConcatF32 { .. }
        ),
        Kernel::Tensor(op) => matches!(
            op,
            TensorOperation::Reshape(_)
                | TensorOperation::SplitLastAxis(_)
                | TensorOperation::Transpose(_)
                | TensorOperation::Broadcast(_)
                | TensorOperation::Slice { .. }
                | TensorOperation::Select { .. }
                | TensorOperation::Contiguous
                | TensorOperation::Concatenate { .. }
                | TensorOperation::Unary(_)
                | TensorOperation::Binary(_)
                | TensorOperation::Matmul
                | TensorOperation::Sum { .. }
                | TensorOperation::Repeat { .. }
                | TensorOperation::Softmax { .. }
                | TensorOperation::TakeAlong { .. }
                | TensorOperation::Take { .. }
                | TensorOperation::RmsNorm { .. }
                | TensorOperation::Cast(
                    ScalarType::F32
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::I64
                        | ScalarType::I32
                        | ScalarType::U32
                        | ScalarType::U8
                        | ScalarType::Bool
                )
        ),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_program_preserves_strided_views_and_live_snapshots() -> anyhow::Result<()> {
        let device = MetalDevice::system_default();
        let stream = device.stream();
        let add = device
            .prepare(Kernel::Tensor(TensorOperation::Binary(Binary::Add)))?
            .unwrap();
        let transpose = device
            .prepare(Kernel::Tensor(TensorOperation::Transpose(vec![1, 0])))?
            .unwrap();
        let x = Array::from_slice(&[99f32, 1., 2., 3., 4., 5., 6.], &[7])
            .slice(0, 1, Some(7))?
            .reshape(&[2, 3])?;
        let y = Array::from_slice(&[2f32; 6], &[2, 3]);
        let execute = |a: &Array| -> anyhow::Result<Vec<Array>> {
            let sum = device
                .dispatch(&add, &[a, &y], &stream)?
                .unwrap()
                .output
                .into_one()
                .map_err(anyhow::Error::msg)?;
            let view = device
                .dispatch(&transpose, &[&sum], &stream)?
                .unwrap()
                .output
                .into_one()
                .map_err(anyhow::Error::msg)?;
            Ok(vec![view, a.clone()])
        };
        let (first, program) = device.capture_program(&[&x, &y], &stream, &mut || execute(&x))?;
        let program = program.ok_or_else(|| anyhow::anyhow!("pure tensor region declined"))?;
        let second_x = Array::from_slice(&[88f32, 6., 5., 4., 3., 2., 1.], &[7])
            .slice(0, 1, Some(7))?
            .reshape(&[2, 3])?;
        let second = program.replay(&[&second_x, &y])?.unwrap();
        let oracle = execute(&second_x)?;
        assert_eq!(second[0].t.layout(), oracle[0].t.layout());
        assert_eq!(second[1].t.layout(), second_x.t.layout());
        assert_eq!(second[0].as_slice::<f32>(), oracle[0].as_slice::<f32>());
        assert_eq!(first[0].as_slice::<f32>(), &[3., 6., 4., 7., 5., 8.]);
        assert_eq!(first[1].as_slice::<f32>(), &[1., 2., 3., 4., 5., 6.]);
        let changed_offset = Array::from_slice(&[1f32; 6], &[2, 3]);
        assert!(program.replay(&[&changed_offset, &y])?.is_none());
        Ok(())
    }
}
