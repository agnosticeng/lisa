//! Generic preparation/execution of portable tensor graphs. No model-specific
//! traversal, weight names, kernel selection policy or device implementation.
use lisa_backend::{Device, HostTensor, ImportOptions, Outputs};
use lisa_models::graph::{Graph, Value};

#[derive(Clone, Copy)]
enum Binding {
    Input(usize),
    Weight(usize),
    State(usize),
    Constant(usize),
    Register(usize),
}
struct Call<H> {
    handle: H,
    inputs: Vec<Binding>,
    output: usize,
    count: usize,
    release: Vec<usize>,
}
pub struct PreparedGraph<D: Device> {
    device: D,
    stream: D::Stream,
    calls: Vec<Call<D::KernelHandle>>,
    weights: Vec<D::Tensor>,
    constants: Vec<D::Tensor>,
    outputs: Vec<Binding>,
    writes: Vec<(usize, Binding)>,
    inputs: usize,
    states: usize,
    registers: usize,
    direct: bool,
    resources: bool,
}
impl<D: Device> PreparedGraph<D>
where
    D::Tensor: Clone,
{
    pub fn new(device: D, graph: Graph, weights: Vec<D::Tensor>) -> anyhow::Result<Self> {
        graph.validate().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            graph.weights == weights.len(),
            "graph weight binding count differs"
        );
        let direct = graph.calls.len() == 1
            && graph.writes.is_empty()
            && graph.outputs.len() == graph.calls[0].kernel.outputs()
            && graph
                .outputs
                .iter()
                .enumerate()
                .all(|(output, value)| *value == Value::Result { call: 0, output });
        let mut offsets = Vec::with_capacity(graph.calls.len());
        let mut registers = 0usize;
        for call in &graph.calls {
            offsets.push(registers);
            registers = registers
                .checked_add(call.kernel.outputs())
                .ok_or_else(|| anyhow::anyhow!("graph register count overflow"))?;
        }
        let binding = |value| match value {
            Value::Input(i) => Binding::Input(i),
            Value::Weight(i) => Binding::Weight(i),
            Value::State(i) => Binding::State(i),
            Value::Constant(i) => Binding::Constant(i),
            Value::Result { call, output } => Binding::Register(offsets[call] + output),
        };
        // Calculate last uses once. Outputs and state commits stay live until
        // completion; all other intermediates are released after their last use.
        // Device leases, rather than these handles, protect queued GPU reads.
        let mut last_use = vec![0; registers];
        for (i, call) in graph.calls.iter().enumerate() {
            for slot in offsets[i]..offsets[i] + call.kernel.outputs() {
                last_use[slot] = i;
            }
            for &value in &call.inputs {
                if let Binding::Register(slot) = binding(value) {
                    last_use[slot] = i;
                }
            }
        }
        for value in graph
            .outputs
            .iter()
            .chain(graph.writes.iter().map(|w| &w.value))
        {
            if let Binding::Register(slot) = binding(*value) {
                last_use[slot] = graph.calls.len();
            }
        }
        let mut release = vec![vec![]; graph.calls.len()];
        for (slot, &last) in last_use.iter().enumerate() {
            if last < release.len() {
                release[last].push(slot);
            }
        }
        let resources = graph
            .calls
            .iter()
            .any(|call| call.kernel.requires_resources());
        let mut calls = Vec::with_capacity(graph.calls.len());
        for (i, call) in graph.calls.into_iter().enumerate() {
            let count = call.kernel.outputs();
            let handle = device
                .prepare(call.kernel)?
                .ok_or_else(|| anyhow::anyhow!("required graph operation {i} is unsupported"))?;
            calls.push(Call {
                handle,
                inputs: call.inputs.into_iter().map(binding).collect(),
                output: offsets[i],
                count,
                release: std::mem::take(&mut release[i]),
            });
        }
        let constants = graph
            .constants
            .iter()
            .map(|c| {
                device.pull(
                    HostTensor {
                        bytes: &c.bytes,
                        shape: &c.shape,
                        dtype: c.dtype,
                    }
                    .into(),
                    ImportOptions::default(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let stream = device.stream();
        Ok(Self {
            device,
            stream,
            calls,
            weights,
            constants,
            outputs: graph.outputs.into_iter().map(binding).collect(),
            writes: graph
                .writes
                .into_iter()
                .map(|w| (w.slot, binding(w.value)))
                .collect(),
            inputs: graph.inputs,
            states: graph.states,
            registers,
            direct,
            resources,
        })
    }
    pub fn run(
        &self,
        inputs: &[&D::Tensor],
        states: &mut [D::Tensor],
    ) -> anyhow::Result<Outputs<D::Tensor>> {
        anyhow::ensure!(!self.resources, "graph requires indirect resources");
        self.run_dispatch(inputs, states, |handle, args| {
            self.device.dispatch(handle, args, &self.stream)
        })
    }
    /// Bound in-flight allocations for memory-constrained application paths.
    /// Completion barriers change scheduling, never graph arithmetic or lifetimes.
    pub fn run_bounded(
        &self,
        inputs: &[&D::Tensor],
        states: &mut [D::Tensor],
        cadence: usize,
    ) -> anyhow::Result<Outputs<D::Tensor>>
    where
        D: lisa_backend::ManagedDevice,
    {
        anyhow::ensure!(
            !self.resources && cadence > 0,
            "invalid bounded graph execution"
        );
        let mut count = 0;
        self.run_dispatch(inputs, states, |handle, args| {
            let result = self.device.dispatch(handle, args, &self.stream)?;
            count += 1;
            if count % cadence == 0 {
                if let Some(result) = &result {
                    self.device.wait(&result.completion)?;
                }
                self.device.clear_unused();
            }
            Ok(result)
        })
    }

    /// Bind opaque resources for indirect operations. The provider validates the
    /// context and retains allocations until their encoded users complete.
    pub fn run_with_resources(
        &self,
        inputs: &[&D::Tensor],
        states: &mut [D::Tensor],
        resources: &D::Resources,
    ) -> anyhow::Result<Outputs<D::Tensor>>
    where
        D: lisa_backend::IndirectDevice,
    {
        self.run_dispatch(inputs, states, |handle, args| {
            self.device
                .dispatch_indirect(handle, args, resources, &self.stream)
        })
    }
    fn run_dispatch(
        &self,
        inputs: &[&D::Tensor],
        states: &mut [D::Tensor],
        mut dispatch: impl FnMut(
            &D::KernelHandle,
            &[&D::Tensor],
        ) -> Result<
            Option<lisa_backend::Dispatch<Outputs<D::Tensor>, D::Completion>>,
            D::Error,
        >,
    ) -> anyhow::Result<Outputs<D::Tensor>> {
        anyhow::ensure!(
            inputs.len() == self.inputs && states.len() == self.states,
            "graph input/state binding count differs"
        );
        // A single operation with identity outputs needs no register file,
        // output clones or heap collections (the common normalization path).
        if self.direct {
            let call = &self.calls[0];
            let resolve = |b| match b {
                Binding::Input(i) => inputs[i],
                Binding::Weight(i) => &self.weights[i],
                Binding::State(i) => &states[i],
                Binding::Constant(i) => &self.constants[i],
                Binding::Register(_) => unreachable!("validated single-call dependency"),
            };
            let dispatched = if call.inputs.is_empty() {
                dispatch(&call.handle, &[])
            } else {
                let mut stack = [resolve(call.inputs[0]); 64];
                let mut heap = Vec::new();
                let args: &[&D::Tensor] = if call.inputs.len() <= stack.len() {
                    for (i, &b) in call.inputs.iter().enumerate() {
                        stack[i] = resolve(b);
                    }
                    &stack[..call.inputs.len()]
                } else {
                    heap.extend(call.inputs.iter().map(|&b| resolve(b)));
                    &heap
                };
                dispatch(&call.handle, args)
            };
            let result = dispatched?.ok_or_else(|| {
                anyhow::anyhow!("required graph operation declined its input shape")
            })?;
            anyhow::ensure!(
                result.output.len() == call.count,
                "device returned an incorrect graph output count"
            );
            return Ok(result.output);
        }
        let mut stack: [Option<D::Tensor>; 16] = std::array::from_fn(|_| None);
        let mut heap = if self.registers > stack.len() {
            (0..self.registers).map(|_| None).collect::<Vec<_>>()
        } else {
            vec![]
        };
        let registers: &mut [Option<D::Tensor>] = if self.registers <= stack.len() {
            &mut stack[..self.registers]
        } else {
            &mut heap
        };
        let get = |binding, registers: &[Option<D::Tensor>]| -> D::Tensor {
            match binding {
                Binding::Input(i) => inputs[i].clone(),
                Binding::Weight(i) => self.weights[i].clone(),
                Binding::State(i) => states[i].clone(),
                Binding::Constant(i) => self.constants[i].clone(),
                Binding::Register(i) => registers[i]
                    .as_ref()
                    .expect("validated graph dependency")
                    .clone(),
            }
        };
        // References are gathered without cloning tensor storage/metadata per
        // call. Dead producer handles are released after their final consumer.
        for call in &self.calls {
            let resolve = |b| match b {
                Binding::Input(i) => inputs[i],
                Binding::Weight(i) => &self.weights[i],
                Binding::State(i) => &states[i],
                Binding::Constant(i) => &self.constants[i],
                Binding::Register(i) => registers[i].as_ref().expect("validated graph dependency"),
            };
            let dispatched = if call.inputs.is_empty() {
                dispatch(&call.handle, &[])
            } else {
                let mut stack = [resolve(call.inputs[0]); 64];
                let mut heap = Vec::new();
                let args: &[&D::Tensor] = if call.inputs.len() <= stack.len() {
                    for (i, &b) in call.inputs.iter().enumerate() {
                        stack[i] = resolve(b);
                    }
                    &stack[..call.inputs.len()]
                } else {
                    heap.extend(call.inputs.iter().map(|&b| resolve(b)));
                    &heap
                };
                dispatch(&call.handle, args)
            };
            let result = dispatched?.ok_or_else(|| {
                anyhow::anyhow!("required graph operation declined its input shape")
            })?;
            anyhow::ensure!(
                result.output.len() == call.count,
                "device returned an incorrect graph output count"
            );
            for (i, tensor) in result.output.into_iter().enumerate() {
                registers[call.output + i] = Some(tensor);
            }
            for &slot in &call.release {
                registers[slot] = None;
            }
        }
        let output = match self.outputs.as_slice() {
            [one] => Outputs::One(get(*one, registers)),
            [one, two] => Outputs::Two([get(*one, registers), get(*two, registers)]),
            values => Outputs::Many(values.iter().map(|&v| get(v, registers)).collect()),
        };
        let writes: Vec<_> = self
            .writes
            .iter()
            .map(|&(slot, v)| (slot, get(v, &registers)))
            .collect();
        // No state handle is changed on an unsupported operation or device error.
        for (slot, value) in writes {
            states[slot] = value;
        }
        Ok(output)
    }
}

/// A bounded cache of prepared, unbound plans. The engine chooses the cache
/// lifetime and key; models only supply the graph. Eviction drops kernel handles
/// and constants, while the device retains any resources used by queued work.
pub struct PlanCache<D: Device, K> {
    device: D,
    weights: Vec<D::Tensor>,
    capacity: usize,
    entries: Vec<(K, PreparedGraph<D>)>,
}
impl<D: Device + Clone, K: PartialEq> PlanCache<D, K>
where
    D::Tensor: Clone,
{
    pub fn new(device: D, capacity: usize) -> Self {
        Self::bound(device, capacity, vec![])
    }
    /// The binding owner fixes weights and device for the lifetime of this cache.
    pub fn bound(device: D, capacity: usize, weights: Vec<D::Tensor>) -> Self {
        Self {
            device,
            weights,
            capacity: capacity.max(1),
            entries: vec![],
        }
    }
    pub fn get_or_prepare(
        &mut self,
        key: K,
        build: impl FnOnce() -> Graph,
    ) -> anyhow::Result<&PreparedGraph<D>> {
        self.try_get_or_prepare(key, || Ok(build()))
    }
    pub fn try_get_or_prepare(
        &mut self,
        key: K,
        build: impl FnOnce() -> anyhow::Result<Graph>,
    ) -> anyhow::Result<&PreparedGraph<D>> {
        if let Some(i) = self.entries.iter().position(|(k, _)| *k == key) {
            return Ok(&self.entries[i].1);
        }
        let prepared = PreparedGraph::new(self.device.clone(), build()?, self.weights.clone())?;
        if self.entries.len() == self.capacity {
            self.entries.remove(0);
        }
        self.entries.push((key, prepared));
        Ok(&self.entries.last().unwrap().1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lisa_backend::*;
    use lisa_models::graph::{Builder, Value};
    use std::sync::{Arc, Mutex, Weak};
    #[derive(Clone, Default)]
    struct Tracking {
        outputs: Arc<Mutex<Vec<Weak<()>>>>,
    }
    impl Device for Tracking {
        type Tensor = Arc<()>;
        type KernelHandle = Kernel;
        type Stream = ();
        type Completion = ();
        type Error = std::io::Error;
        fn capabilities(&self) -> &[KernelCapability] {
            &[]
        }
        fn pull(&self, _: TensorSource<'_>, _: ImportOptions) -> Result<Arc<()>, Self::Error> {
            Ok(Arc::new(()))
        }
        fn shape<'a>(&self, _: &'a Arc<()>) -> &'a [i32] {
            &[]
        }
        fn dtype(&self, _: &Arc<()>) -> Option<ScalarType> {
            Some(ScalarType::BF16)
        }
        fn stream(&self) {}
        fn prepare(&self, k: Kernel) -> Result<Option<Kernel>, Self::Error> {
            Ok(Some(k))
        }
        fn dispatch(
            &self,
            k: &Kernel,
            _: &[&Arc<()>],
            _: &(),
        ) -> Result<Option<Dispatch<Outputs<Arc<()>>, ()>>, Self::Error> {
            let mut outputs = self.outputs.lock().unwrap();
            // A chain must retain at most its immediately preceding result.
            assert!(outputs.iter().filter(|w| w.strong_count() > 0).count() <= 1);
            let out: Vec<_> = (0..k.outputs())
                .map(|_| {
                    let t = Arc::new(());
                    outputs.push(Arc::downgrade(&t));
                    t
                })
                .collect();
            Ok(Some(Dispatch {
                output: Outputs::Many(out),
                completion: (),
            }))
        }
        fn wait(&self, _: &()) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    #[test]
    fn bound_cache_preserves_weights_across_eviction_and_failed_preparation() -> anyhow::Result<()>
    {
        let weight = lisa_cpu::Tensor::from_f32(vec![3.], &[1])?;
        let input = lisa_cpu::Tensor::from_f32(vec![2.], &[1])?;
        let mut cache = PlanCache::bound(lisa_cpu::CpuDevice, 1, vec![weight]);
        let graph = || {
            let mut b = Builder::new(1, 1, 0);
            let y = b.one(
                Kernel::HostTensor(TensorOperation::Binary(Binary::Add)),
                vec![Value::Input(0), Value::Weight(0)],
            );
            b.finish(vec![y])
        };
        for key in [0, 1, 0] {
            let output = cache
                .get_or_prepare(key, graph)?
                .run(&[&input], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            assert_eq!(output.f32()?, &[5.]);
        }
        assert!(
            cache
                .try_get_or_prepare(2, || anyhow::bail!("invalid plan"))
                .is_err()
        );
        let output = cache
            .get_or_prepare(0, || panic!("existing plan was evicted on failure"))?
            .run(&[&input], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        assert_eq!(output.f32()?, &[5.]);
        Ok(())
    }
    #[test]
    fn composed_graph_rebases_constants_results_weights_and_state_commits() -> anyhow::Result<()> {
        let mut child = Builder::new(1, 1, 1);
        let ten = child.scalar(10.);
        let a = child.one(
            Kernel::HostTensor(TensorOperation::Binary(Binary::Add)),
            vec![Value::Input(0), ten],
        );
        let a = child.one(
            Kernel::HostTensor(TensorOperation::Binary(Binary::Multiply)),
            vec![a, Value::Weight(0)],
        );
        child.commit(0, a);
        let child = child.finish(vec![a]);
        let mut root = Builder::new(1, 1, 1);
        let five = root.scalar(5.);
        let first = root.one(
            Kernel::HostTensor(TensorOperation::Binary(Binary::Add)),
            vec![Value::Input(0), five],
        );
        let out = root
            .append(child, &[first], &[Value::Weight(0)], &[Value::State(0)])
            .map_err(anyhow::Error::msg)?;
        let device = lisa_cpu::CpuDevice;
        let weight = lisa_cpu::Tensor::from_f32(vec![3.], &[1])?;
        let input = lisa_cpu::Tensor::from_f32(vec![2.], &[1])?;
        let mut state = [lisa_cpu::Tensor::from_f32(vec![0.], &[1])?];
        let plan = PreparedGraph::new(device, root.finish(out), vec![weight])?;
        let output = plan
            .run(&[&input], &mut state)?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        assert_eq!(output.f32()?, &[51.]);
        assert_eq!(state[0].f32()?, &[51.]);
        Ok(())
    }
    #[test]
    fn nullary_calls_work_in_direct_and_multi_call_graphs() {
        for chained in [false, true] {
            let mut b = Builder::new(0, 0, 0);
            let mut x = b.one(
                Kernel::Zeros {
                    shape: vec![2, 3],
                    dtype: ScalarType::F32,
                },
                vec![],
            );
            if chained {
                x = b.one(
                    Kernel::Tensor(TensorOperation::Unary(Unary::Negate)),
                    vec![x],
                );
            }
            let device = Tracking::default();
            let prepared = PreparedGraph::new(device.clone(), b.finish(vec![x]), vec![]).unwrap();
            let output = prepared.run(&[], &mut []).unwrap();
            assert_eq!(output.len(), 1);
            assert_eq!(
                device.outputs.lock().unwrap().len(),
                if chained { 2 } else { 1 }
            );
        }
    }
    #[test]
    fn long_graph_releases_intermediates_after_their_last_consumer() {
        let mut b = Builder::new(1, 0, 0);
        let mut x = Value::Input(0);
        for _ in 0..256 {
            x = b.one(
                Kernel::Tensor(TensorOperation::Unary(Unary::Negate)),
                vec![x],
            );
        }
        let device = Tracking::default();
        let prepared = PreparedGraph::new(device.clone(), b.finish(vec![x]), vec![]).unwrap();
        let input = Arc::new(());
        let output = prepared.run(&[&input], &mut []).unwrap();
        assert_eq!(
            device
                .outputs
                .lock()
                .unwrap()
                .iter()
                .filter(|w| w.strong_count() > 0)
                .count(),
            1
        );
        drop(output);
        assert!(
            device
                .outputs
                .lock()
                .unwrap()
                .iter()
                .all(|w| w.strong_count() == 0)
        );
    }
    #[test]
    fn failed_call_does_not_commit_state_handles() {
        let mut b = Builder::new(1, 0, 1);
        let next = b.one(
            Kernel::HostTensor(TensorOperation::Unary(Unary::Negate)),
            vec![Value::Input(0)],
        );
        b.commit(0, next);
        // Preparation succeeds, but this request violates reshape element count.
        let invalid = b.one(
            Kernel::HostTensor(TensorOperation::Reshape(vec![3])),
            vec![next],
        );
        let plan =
            PreparedGraph::new(lisa_cpu::CpuDevice, b.finish(vec![invalid]), vec![]).unwrap();
        let input = lisa_cpu::Tensor::from_f32(vec![1., 2.], &[2]).unwrap();
        let mut states = [lisa_cpu::Tensor::from_f32(vec![42.], &[1]).unwrap()];
        assert!(plan.run(&[&input], &mut states).is_err());
        assert_eq!(states[0].f32().unwrap(), &[42.]);
    }
    #[test]
    fn variadic_calls_preserve_one_concat_above_stack_capacity() -> anyhow::Result<()> {
        use lisa_backend::{Kernel, TensorOperation};
        for extra_call in [false, true] {
            let mut b = Builder::new(1, 0, 0);
            let x = if extra_call {
                b.one(
                    Kernel::HostTensor(TensorOperation::Unary(Unary::Negate)),
                    vec![Value::Input(0)],
                )
            } else {
                Value::Input(0)
            };
            let out = b.one(
                Kernel::HostTensor(TensorOperation::Concatenate {
                    axis: 0,
                    inputs: 129,
                }),
                vec![x; 129],
            );
            let graph = b.finish(vec![out]);
            assert_eq!(graph.calls.len(), 1 + usize::from(extra_call));
            let prepared = PreparedGraph::new(lisa_cpu::CpuDevice, graph, vec![])?;
            let input = lisa_cpu::Tensor::from_f32(vec![1.], &[1])?;
            let output = prepared
                .run(&[&input], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            assert_eq!(output.f32()?, vec![if extra_call { -1. } else { 1. }; 129]);
        }
        Ok(())
    }
}
