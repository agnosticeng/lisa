//! Provider-contract tests for the bounded exact-layout program cache.
use super::*;
use lisa_backend::*;
use lisa_models::graph::Builder;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    shape: [i32; 1],
    stride: usize,
    offset: usize,
    dtype: ScalarType,
}
#[derive(Clone)]
struct Tensor {
    layout: Layout,
    storage: Arc<Vec<i32>>,
}
impl Tensor {
    fn input(offset: usize, stride: usize, values: [i32; 2]) -> Self {
        let mut storage = vec![999; offset + stride + 1];
        storage[offset] = values[0];
        storage[offset + stride] = values[1];
        Self {
            layout: Layout {
                shape: [2],
                stride,
                offset,
                dtype: ScalarType::F32,
            },
            storage: Arc::new(storage),
        }
    }
    fn values(&self) -> [i32; 2] {
        [
            self.storage[self.layout.offset],
            self.storage[self.layout.offset + self.layout.stride],
        ]
    }
}
#[derive(Default)]
struct Counts {
    captures: AtomicUsize,
    replays: AtomicUsize,
    dispatched: AtomicUsize,
    fail_capture: AtomicBool,
    dropped: Mutex<Vec<usize>>,
    // Simulates a provider retaining a resource until its encoded users finish.
    queued: Mutex<Vec<Arc<()>>>,
    resources: Mutex<Vec<std::sync::Weak<()>>>,
}
#[derive(Clone, Default)]
struct RecordingDevice(Arc<Counts>);
struct Recorded {
    id: usize,
    output_count: usize,
    layouts: Vec<Layout>,
    aliases: bool,
    counts: Arc<Counts>,
    resource: Arc<()>,
}
impl Drop for Recorded {
    fn drop(&mut self) {
        self.counts.dropped.lock().unwrap().push(self.id);
    }
}
fn sum(a: &Tensor, b: &Tensor) -> Tensor {
    let a = a.values();
    let b = b.values();
    Tensor::input(0, 1, [a[0] + b[0], a[1] + b[1]])
}
impl RecordedProgram<Tensor> for Recorded {
    fn replay(&self, inputs: &[&Tensor]) -> anyhow::Result<Option<Vec<Tensor>>> {
        if inputs.len() != self.layouts.len()
            || inputs
                .iter()
                .zip(&self.layouts)
                .any(|(input, saved)| input.layout != *saved)
            || Arc::ptr_eq(&inputs[0].storage, &inputs[1].storage) != self.aliases
        {
            return Ok(None);
        }
        self.counts.replays.fetch_add(1, Ordering::Relaxed);
        self.counts
            .queued
            .lock()
            .unwrap()
            .push(self.resource.clone());
        let output = sum(inputs[0], inputs[1]);
        let mut values = vec![output.clone(), inputs[0].clone(), output];
        values.truncate(self.output_count);
        Ok(Some(values))
    }
}
impl Device for RecordingDevice {
    type Tensor = Tensor;
    type KernelHandle = Kernel;
    type Stream = ();
    type Completion = ();
    type Error = std::io::Error;
    fn capabilities(&self) -> &[KernelCapability] {
        &[]
    }
    fn pull(&self, _: TensorSource<'_>, _: ImportOptions) -> Result<Tensor, Self::Error> {
        unreachable!("fixture has no constants")
    }
    fn shape<'a>(&self, tensor: &'a Tensor) -> &'a [i32] {
        &tensor.layout.shape
    }
    fn dtype(&self, tensor: &Tensor) -> Option<ScalarType> {
        Some(tensor.layout.dtype)
    }
    fn stream(&self) {}
    fn prepare(&self, kernel: Kernel) -> Result<Option<Kernel>, Self::Error> {
        Ok(Some(kernel))
    }
    fn dispatch(
        &self,
        _: &Kernel,
        inputs: &[&Tensor],
        _: &(),
    ) -> Result<Option<Dispatch<Outputs<Tensor>, ()>>, Self::Error> {
        self.0.dispatched.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Dispatch {
            output: Outputs::One(sum(inputs[0], inputs[1])),
            completion: (),
        }))
    }
    fn wait(&self, _: &()) -> Result<(), Self::Error> {
        Ok(())
    }
    fn capture_program(
        &self,
        inputs: &[&Tensor],
        _: &(),
        encode: &mut dyn FnMut() -> anyhow::Result<Vec<Tensor>>,
    ) -> anyhow::Result<ProgramCapture<Tensor>> {
        let values = encode()?;
        anyhow::ensure!(
            !self.0.fail_capture.swap(false, Ordering::Relaxed),
            "capture failed after encoding"
        );
        let id = self.0.captures.fetch_add(1, Ordering::Relaxed);
        let resource = Arc::new(());
        self.0
            .resources
            .lock()
            .unwrap()
            .push(Arc::downgrade(&resource));
        let output_count = values.len();
        Ok((
            values,
            Some(Box::new(Recorded {
                id,
                output_count,
                layouts: inputs.iter().map(|input| input.layout.clone()).collect(),
                aliases: Arc::ptr_eq(&inputs[0].storage, &inputs[1].storage),
                counts: self.0.clone(),
                resource,
            })),
        ))
    }
}
fn plan(device: RecordingDevice) -> anyhow::Result<PreparedGraph<RecordingDevice>> {
    let mut b = Builder::new(1, 0, 1);
    let output = b.one(
        Kernel::Tensor(TensorOperation::Binary(Binary::Add)),
        vec![Value::Input(0), Value::State(0)],
    );
    b.commit(0, output);
    Ok(
        PreparedGraph::new(device, b.finish(vec![output, Value::Input(0)]), vec![])?
            .with_dispatch_program(),
    )
}
fn run(plan: &PreparedGraph<RecordingDevice>, input: &Tensor) -> anyhow::Result<Vec<Tensor>> {
    let mut states = [Tensor::input(0, 1, [7, 11])];
    let output = plan
        .run(&[input], &mut states)?
        .into_iter()
        .collect::<Vec<_>>();
    let values = input.values();
    assert_eq!(output[0].values(), [values[0] + 7, values[1] + 11]);
    assert_eq!(states[0].values(), output[0].values());
    assert!(Arc::ptr_eq(&states[0].storage, &output[0].storage));
    assert!(Arc::ptr_eq(&output[1].storage, &input.storage));
    Ok(output)
}

#[test]
fn recorded_layout_cache_admits_hot_layouts_after_eight_and_preserves_live_outputs()
-> anyhow::Result<()> {
    let device = RecordingDevice::default();
    let plan = plan(device.clone())?;
    let mut live = Vec::new();
    for offset in 0..12 {
        for value in [20, 40] {
            let input = Tensor::input(offset, 2, [value + offset as i32, -value]);
            let outputs = run(&plan, &input)?;
            live.push((
                outputs.iter().map(Tensor::values).collect::<Vec<_>>(),
                outputs,
            ));
        }
        assert!(
            plan.program.as_ref().unwrap().lock().unwrap().entries.len() <= PROGRAM_LAYOUT_CAPACITY
        );
    }
    assert_eq!(device.0.captures.load(Ordering::Relaxed), 12);
    assert_eq!(device.0.replays.load(Ordering::Relaxed), 12);
    assert_eq!(device.0.dispatched.load(Ordering::Relaxed), 12);
    assert_eq!(*device.0.dropped.lock().unwrap(), [0, 1, 2, 3]);
    // Descriptor eviction does not own a provider's in-flight resource lease.
    assert!(
        device
            .0
            .resources
            .lock()
            .unwrap()
            .iter()
            .all(|resource| resource.upgrade().is_some())
    );
    device.0.queued.lock().unwrap().clear();
    assert!(
        device.0.resources.lock().unwrap()[..4]
            .iter()
            .all(|resource| resource.upgrade().is_none())
    );
    drop(plan);
    for (expected, outputs) in live {
        assert_eq!(
            outputs.iter().map(Tensor::values).collect::<Vec<_>>(),
            expected
        );
    }
    Ok(())
}

#[test]
fn recorded_layout_cache_promotes_hits_before_evicting_the_oldest_layout() -> anyhow::Result<()> {
    let device = RecordingDevice::default();
    let plan = plan(device.clone())?;
    for offset in 0..8 {
        run(&plan, &Tensor::input(offset, 1, [1, 2]))?;
    }
    run(&plan, &Tensor::input(0, 1, [10, 20]))?;
    run(&plan, &Tensor::input(8, 1, [30, 40]))?;
    assert_eq!(*device.0.dropped.lock().unwrap(), [1]);
    run(&plan, &Tensor::input(0, 1, [50, 60]))?;
    assert_eq!(device.0.captures.load(Ordering::Relaxed), 9);
    run(&plan, &Tensor::input(1, 1, [70, 80]))?;
    assert_eq!(device.0.captures.load(Ordering::Relaxed), 10);
    assert_eq!(*device.0.dropped.lock().unwrap(), [1, 2]);
    Ok(())
}

#[test]
fn recorded_layout_cache_failed_seal_preserves_entries_and_state() -> anyhow::Result<()> {
    let device = RecordingDevice::default();
    let plan = plan(device.clone())?;
    for offset in 0..8 {
        run(&plan, &Tensor::input(offset, 1, [1, 2]))?;
    }
    let original = Tensor::input(0, 1, [7, 11]);
    let mut states = [original.clone()];
    device.0.fail_capture.store(true, Ordering::Relaxed);
    assert!(
        plan.run(&[&Tensor::input(8, 1, [3, 4])], &mut states)
            .is_err()
    );
    assert!(Arc::ptr_eq(&states[0].storage, &original.storage));
    assert!(device.0.dropped.lock().unwrap().is_empty());
    run(&plan, &Tensor::input(0, 1, [8, 9]))?;
    assert_eq!(device.0.captures.load(Ordering::Relaxed), 8);
    assert_eq!(device.0.replays.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn recorded_layout_cache_preserves_provider_stride_offset_dtype_and_alias_checks()
-> anyhow::Result<()> {
    let device = RecordingDevice::default();
    let plan = plan(device.clone())?;
    run(&plan, &Tensor::input(0, 1, [1, 2]))?;
    run(&plan, &Tensor::input(1, 1, [1, 2]))?;
    run(&plan, &Tensor::input(1, 2, [1, 2]))?;
    let mut changed_dtype = Tensor::input(1, 2, [3, 4]);
    changed_dtype.layout.dtype = ScalarType::BF16;
    run(&plan, &changed_dtype)?;
    let input = Tensor::input(0, 1, [5, 6]);
    let mut states = [input.clone()];
    let output = plan
        .run(&[&input], &mut states)?
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(output[0].values(), [10, 12]);
    assert_eq!(device.0.captures.load(Ordering::Relaxed), 5);
    run(&plan, &Tensor::input(0, 1, [9, 10]))?;
    assert_eq!(device.0.captures.load(Ordering::Relaxed), 5);
    assert_eq!(device.0.replays.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn region_replay_selection_skips_capture_without_discarding_a_hot_program() -> anyhow::Result<()> {
    let device = RecordingDevice::default();
    let mut builder = Builder::new(2, 0, 0);
    let output = builder.one(
        Kernel::Tensor(TensorOperation::Binary(Binary::Add)),
        vec![Value::Input(0), Value::Input(1)],
    );
    let graph = builder.finish(vec![output]);
    let regions = PreparedRegions::new(
        device.clone(),
        lisa_models::graph::regions::split(&graph, &[0, 1]).map_err(anyhow::Error::msg)?,
        vec![],
        &[true],
    )?;
    let state = Tensor::input(0, 1, [7, 11]);
    for (value, enabled, captures, replays) in [
        (1, false, 0, 0),
        (2, true, 1, 0),
        (3, false, 1, 0),
        (4, true, 1, 1),
    ] {
        let input = Tensor::input(2, 2, [value, value + 1]);
        let output = regions
            .run_with_replay(&[&input, &state], &mut [], &[enabled])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        assert_eq!(output.values(), [value + 7, value + 12]);
        assert_eq!(device.0.captures.load(Ordering::Relaxed), captures);
        assert_eq!(device.0.replays.load(Ordering::Relaxed), replays);
    }
    // The existing entry point keeps replay enabled without a new policy call.
    let input = Tensor::input(2, 2, [9, 10]);
    assert_eq!(
        regions
            .run(&[&input, &state], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?
            .values(),
        [16, 21]
    );
    assert_eq!(device.0.replays.load(Ordering::Relaxed), 2);
    let dispatched = device.0.dispatched.load(Ordering::Relaxed);
    assert!(
        regions
            .run_with_replay(&[&input, &state], &mut [], &[])
            .is_err()
    );
    assert_eq!(device.0.dispatched.load(Ordering::Relaxed), dispatched);
    Ok(())
}
