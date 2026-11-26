//! Runtime-selected devices with context-checked opaque handles.
//!
//! Erasure changes only Rust ownership and dispatch. Kernels, tensor layouts,
//! submission and completion retain the concrete provider's implementation.
use crate::*;
use std::{any::Any, sync::Arc};

#[derive(Debug)]
pub struct DeviceError(String);
impl DeviceError {
    pub fn msg(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DeviceError {}
type Result<T> = std::result::Result<T, DeviceError>;
fn error(e: impl std::fmt::Display) -> DeviceError {
    DeviceError(e.to_string())
}

type Object = dyn Any + Send + Sync;

#[derive(Clone)]
pub struct DeviceHandle(Arc<dyn Operations>);
#[derive(Clone)]
pub struct TensorHandle {
    owner: DeviceHandle,
    value: Arc<Object>,
    shape: Vec<i32>,
    dtype: Option<ScalarType>,
}
pub struct PreparedHandle {
    owner: DeviceHandle,
    value: Box<Object>,
}
#[derive(Clone)]
pub struct StreamHandle {
    owner: DeviceHandle,
    value: Arc<Object>,
}
pub struct CompletionHandle {
    owner: DeviceHandle,
    value: Box<Object>,
}
/// Retains provider-owned indirect tables and their source storage leases.
pub struct ResourcesHandle {
    owner: DeviceHandle,
    value: Box<Object>,
}

trait ErasedProgram: Send + Sync {
    fn set_scheduling(&mut self, scheduling: ProgramScheduling) -> bool;
    fn replay(&self, inputs: &[&TensorHandle]) -> anyhow::Result<Option<Vec<Arc<Object>>>>;
}
struct TypedProgram<T> {
    program: Box<dyn RecordedProgram<T>>,
}
impl<T: Any + Send + Sync> ErasedProgram for TypedProgram<T> {
    fn set_scheduling(&mut self, scheduling: ProgramScheduling) -> bool {
        self.program.set_scheduling(scheduling)
    }
    fn replay(&self, inputs: &[&TensorHandle]) -> anyhow::Result<Option<Vec<Arc<Object>>>> {
        let inputs = inputs
            .iter()
            .map(|t| {
                t.value
                    .downcast_ref::<T>()
                    .expect("checked recorded-program tensor owner")
            })
            .collect::<smallvec::SmallVec<[&T; 16]>>();
        Ok(self.program.replay(&inputs)?.map(|outputs| {
            outputs
                .into_iter()
                .map(|t| Arc::new(t) as Arc<Object>)
                .collect()
        }))
    }
}
struct ProgramHandle {
    owner: DeviceHandle,
    value: Box<dyn ErasedProgram>,
}
impl RecordedProgram<TensorHandle> for ProgramHandle {
    fn set_scheduling(&mut self, scheduling: ProgramScheduling) -> bool {
        self.value.set_scheduling(scheduling)
    }
    fn replay(&self, inputs: &[&TensorHandle]) -> anyhow::Result<Option<Vec<TensorHandle>>> {
        for tensor in inputs {
            self.owner.check(&tensor.owner)?;
        }
        Ok(self
            .value
            .replay(inputs)?
            .map(|outputs| outputs.into_iter().map(|t| self.owner.tensor(t)).collect()))
    }
}
type ErasedCapture = (Vec<Arc<Object>>, Option<Box<dyn ErasedProgram>>);

trait Operations: Send + Sync {
    fn same_storage(&self, a: &Object, b: &Object) -> bool;
    fn capabilities(&self) -> &[KernelCapability];
    fn pull(&self, source: TensorSource<'_>, options: ImportOptions) -> Result<Arc<Object>>;
    fn shape<'a>(&self, tensor: &'a Object) -> &'a [i32];
    fn dtype(&self, tensor: &Object) -> Option<ScalarType>;
    fn stream(&self) -> Arc<Object>;
    fn prepare(&self, kernel: Kernel) -> Result<Option<Box<Object>>>;
    fn dispatch(
        &self,
        handle: &Object,
        inputs: &[&TensorHandle],
        stream: &Object,
    ) -> Result<Option<(Outputs<Arc<Object>>, Box<Object>)>>;
    fn capture_program(
        &self,
        inputs: &[&TensorHandle],
        stream: &Object,
        encode: &mut dyn FnMut() -> anyhow::Result<Vec<TensorHandle>>,
    ) -> anyhow::Result<ErasedCapture>;
    fn resources(&self, slots: &[&TensorHandle]) -> Result<Box<Object>>;
    fn dispatch_indirect(
        &self,
        handle: &Object,
        inputs: &[&TensorHandle],
        resources: &Object,
        stream: &Object,
    ) -> Result<Option<(Outputs<Arc<Object>>, Box<Object>)>>;
    fn wait(&self, completion: &Object) -> Result<()>;
    fn max_allocation_bytes(&self) -> usize;
    fn byte_view(
        &self,
        source: &Object,
        offset: usize,
        shape: &[i32],
        dtype: ScalarType,
    ) -> Result<Arc<Object>>;
    fn read_bytes(&self, tensor: &Object) -> Result<Vec<u8>>;
    fn copy_tensor(&self, tensor: &Object) -> Result<Arc<Object>>;
    fn read_i32_through(&self, tensor: &Object) -> Result<Vec<i32>>;
    fn read_f32_through(&self, tensor: &Object) -> Result<Vec<f32>>;
    fn layout(&self, tensor: &Object) -> TensorLayout;
    fn evaluate(&self, tensor: &Object, wait: bool) -> Result<()>;
}
type ErasedDispatch = Option<(Outputs<Arc<Object>>, Box<Object>)>;
struct IndirectBridge<D: Device> {
    resources: fn(&D, &[&D::Tensor]) -> Result<Box<Object>>,
    dispatch:
        fn(&D, &D::KernelHandle, &[&D::Tensor], &Object, &D::Stream) -> Result<ErasedDispatch>,
}
struct Typed<D: Device>(D, Option<IndirectBridge<D>>, Option<Vec<KernelCapability>>);
fn needs_resources(kind: KernelKind) -> bool {
    matches!(kind, KernelKind::IndirectAffine(op) if !matches!(op, IndirectAffineKind::RouteMap | IndirectAffineKind::Sum8 | IndirectAffineKind::GroupedSwiGlu(_)))
}
impl<D> Operations for Typed<D>
where
    D: HostAccessDevice + Send + Sync + 'static,
    D::Tensor: Clone + Send + Sync + 'static,
    D::KernelHandle: Send + Sync + 'static,
    D::Stream: Send + Sync + 'static,
    D::Completion: Send + Sync + 'static,
{
    fn same_storage(&self, a: &Object, b: &Object) -> bool {
        self.0.same_storage(
            a.downcast_ref().expect("checked tensor owner"),
            b.downcast_ref().expect("checked tensor owner"),
        )
    }
    fn capabilities(&self) -> &[KernelCapability] {
        self.2.as_deref().unwrap_or_else(|| self.0.capabilities())
    }
    fn pull(&self, source: TensorSource<'_>, options: ImportOptions) -> Result<Arc<Object>> {
        Ok(Arc::new(self.0.pull(source, options).map_err(error)?))
    }
    fn shape<'a>(&self, tensor: &'a Object) -> &'a [i32] {
        self.0.shape(
            tensor
                .downcast_ref::<D::Tensor>()
                .expect("provider-owned tensor"),
        )
    }
    fn dtype(&self, tensor: &Object) -> Option<ScalarType> {
        self.0.dtype(
            tensor
                .downcast_ref::<D::Tensor>()
                .expect("provider-owned tensor"),
        )
    }
    fn stream(&self) -> Arc<Object> {
        Arc::new(self.0.stream())
    }
    fn prepare(&self, kernel: Kernel) -> Result<Option<Box<Object>>> {
        if self.1.is_none() && kernel.requires_resources() {
            return Ok(None);
        }
        Ok(self
            .0
            .prepare(kernel)
            .map_err(error)?
            .map(|h| Box::new(h) as Box<Object>))
    }
    fn dispatch(
        &self,
        handle: &Object,
        inputs: &[&TensorHandle],
        stream: &Object,
    ) -> Result<Option<(Outputs<Arc<Object>>, Box<Object>)>> {
        let inputs: smallvec::SmallVec<[&D::Tensor; 16]> = inputs
            .iter()
            .map(|t| {
                t.value
                    .downcast_ref::<D::Tensor>()
                    .expect("checked tensor owner")
            })
            .collect();
        let out = self
            .0
            .dispatch(
                handle.downcast_ref().expect("checked kernel owner"),
                &inputs,
                stream.downcast_ref().expect("checked stream owner"),
            )
            .map_err(error)?;
        Ok(out.map(|out| {
            (
                out.output.map(|t| Arc::new(t) as Arc<Object>),
                Box::new(out.completion) as Box<Object>,
            )
        }))
    }
    fn capture_program(
        &self,
        inputs: &[&TensorHandle],
        stream: &Object,
        encode: &mut dyn FnMut() -> anyhow::Result<Vec<TensorHandle>>,
    ) -> anyhow::Result<ErasedCapture> {
        let inputs = inputs
            .iter()
            .map(|t| {
                t.value
                    .downcast_ref::<D::Tensor>()
                    .expect("checked tensor owner")
            })
            .collect::<smallvec::SmallVec<[&D::Tensor; 16]>>();
        let (outputs, program) = self.0.capture_program(
            &inputs,
            stream.downcast_ref().expect("checked stream owner"),
            &mut || {
                Ok(encode()?
                    .iter()
                    .map(|tensor| {
                        tensor
                            .value
                            .downcast_ref::<D::Tensor>()
                            .expect("checked capture-output owner")
                            .clone()
                    })
                    .collect())
            },
        )?;
        Ok((
            outputs
                .into_iter()
                .map(|tensor| Arc::new(tensor) as Arc<Object>)
                .collect(),
            program.map(|program| Box::new(TypedProgram { program }) as Box<dyn ErasedProgram>),
        ))
    }
    fn resources(&self, slots: &[&TensorHandle]) -> Result<Box<Object>> {
        let bridge = self
            .1
            .as_ref()
            .ok_or_else(|| DeviceError::msg("device does not provide indirect resources"))?;
        let slots: smallvec::SmallVec<[&D::Tensor; 16]> = slots
            .iter()
            .map(|t| {
                t.value
                    .downcast_ref::<D::Tensor>()
                    .expect("checked tensor owner")
            })
            .collect();
        (bridge.resources)(&self.0, &slots)
    }
    fn dispatch_indirect(
        &self,
        handle: &Object,
        inputs: &[&TensorHandle],
        resources: &Object,
        stream: &Object,
    ) -> Result<ErasedDispatch> {
        let bridge = self
            .1
            .as_ref()
            .ok_or_else(|| DeviceError::msg("device does not provide indirect resources"))?;
        let inputs: smallvec::SmallVec<[&D::Tensor; 16]> = inputs
            .iter()
            .map(|t| {
                t.value
                    .downcast_ref::<D::Tensor>()
                    .expect("checked tensor owner")
            })
            .collect();
        (bridge.dispatch)(
            &self.0,
            handle.downcast_ref().expect("checked kernel owner"),
            &inputs,
            resources,
            stream.downcast_ref().expect("checked stream owner"),
        )
    }
    fn wait(&self, completion: &Object) -> Result<()> {
        self.0
            .wait(completion.downcast_ref().expect("checked completion owner"))
            .map_err(error)
    }
    fn max_allocation_bytes(&self) -> usize {
        self.0.max_allocation_bytes()
    }
    fn byte_view(
        &self,
        source: &Object,
        offset: usize,
        shape: &[i32],
        dtype: ScalarType,
    ) -> Result<Arc<Object>> {
        Ok(Arc::new(
            self.0
                .byte_view(
                    source.downcast_ref().expect("checked tensor owner"),
                    offset,
                    shape,
                    dtype,
                )
                .map_err(error)?,
        ))
    }
    fn copy_tensor(&self, tensor: &Object) -> Result<Arc<Object>> {
        Ok(Arc::new(
            self.0
                .copy_tensor(tensor.downcast_ref().expect("checked tensor owner"))
                .map_err(error)?,
        ))
    }
    fn read_bytes(&self, tensor: &Object) -> Result<Vec<u8>> {
        self.0
            .read_bytes(tensor.downcast_ref().expect("checked tensor owner"))
            .map_err(error)
    }
    fn read_i32_through(&self, tensor: &Object) -> Result<Vec<i32>> {
        self.0
            .read_i32_through(tensor.downcast_ref().expect("checked tensor owner"))
            .map_err(error)
    }
    fn read_f32_through(&self, tensor: &Object) -> Result<Vec<f32>> {
        self.0
            .read_f32_through(tensor.downcast_ref().expect("checked tensor owner"))
            .map_err(error)
    }
    fn layout(&self, tensor: &Object) -> TensorLayout {
        self.0
            .layout(tensor.downcast_ref().expect("checked tensor owner"))
    }
    fn evaluate(&self, tensor: &Object, wait: bool) -> Result<()> {
        self.0
            .evaluate(tensor.downcast_ref().expect("checked tensor owner"), wait)
            .map_err(error)
    }
}
impl DeviceHandle {
    pub fn new<D>(device: D) -> Self
    where
        D: HostAccessDevice + Send + Sync + 'static,
        D::Tensor: Clone + Send + Sync + 'static,
        D::KernelHandle: Send + Sync + 'static,
        D::Stream: Send + Sync + 'static,
        D::Completion: Send + Sync + 'static,
    {
        let filtered = device
            .capabilities()
            .iter()
            .any(|c| needs_resources(c.kind))
            .then(|| {
                device
                    .capabilities()
                    .iter()
                    .filter(|c| !needs_resources(c.kind))
                    .cloned()
                    .collect()
            });
        Self(Arc::new(Typed(device, None, filtered)))
    }
    /// Preserve a provider's indirect-resource support without exposing its
    /// address tables, residency sets or concrete tensor type to the engine.
    pub fn with_indirect<D>(device: D) -> Self
    where
        D: HostAccessDevice + IndirectDevice + Send + Sync + 'static,
        D::Tensor: Clone + Send + Sync + 'static,
        D::KernelHandle: Send + Sync + 'static,
        D::Stream: Send + Sync + 'static,
        D::Completion: Send + Sync + 'static,
        D::Resources: Send + Sync + 'static,
    {
        let bridge = IndirectBridge {
            resources: |d: &D, slots: &[&D::Tensor]| {
                Ok(Box::new(d.resources(slots).map_err(error)?) as Box<Object>)
            },
            dispatch: |d: &D,
                       handle: &D::KernelHandle,
                       inputs: &[&D::Tensor],
                       resources: &Object,
                       stream: &D::Stream| {
                let out = d
                    .dispatch_indirect(
                        handle,
                        inputs,
                        resources
                            .downcast_ref::<D::Resources>()
                            .expect("checked resources owner"),
                        stream,
                    )
                    .map_err(error)?;
                Ok(out.map(|out| {
                    (
                        out.output.map(|t| Arc::new(t) as Arc<Object>),
                        Box::new(out.completion) as Box<Object>,
                    )
                }))
            },
        };
        Self(Arc::new(Typed(device, Some(bridge), None)))
    }
    pub fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    fn check(&self, other: &Self) -> Result<()> {
        if self.same_device(other) {
            Ok(())
        } else {
            Err(DeviceError::msg(
                "handle belongs to a different device context",
            ))
        }
    }
    fn tensor(&self, value: Arc<Object>) -> TensorHandle {
        TensorHandle {
            shape: self.0.shape(value.as_ref()).to_vec(),
            dtype: self.0.dtype(value.as_ref()),
            owner: self.clone(),
            value,
        }
    }
    pub fn read_bytes(&self, tensor: &TensorHandle) -> Result<Vec<u8>> {
        self.check(&tensor.owner)?;
        self.0.read_bytes(tensor.value.as_ref())
    }
    pub fn evaluate(&self, tensor: &TensorHandle, wait: bool) -> Result<()> {
        self.check(&tensor.owner)?;
        self.0.evaluate(tensor.value.as_ref(), wait)
    }
}
impl TensorHandle {
    pub fn device(&self) -> &DeviceHandle {
        &self.owner
    }
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub fn dtype(&self) -> Option<ScalarType> {
        self.dtype
    }
    pub fn layout(&self) -> TensorLayout {
        self.owner.0.layout(self.value.as_ref())
    }
}
impl Device for DeviceHandle {
    type Tensor = TensorHandle;
    type KernelHandle = PreparedHandle;
    type Stream = StreamHandle;
    type Completion = CompletionHandle;
    type Error = DeviceError;
    fn capabilities(&self) -> &[KernelCapability] {
        self.0.capabilities()
    }
    fn pull(&self, source: TensorSource<'_>, options: ImportOptions) -> Result<TensorHandle> {
        Ok(self.tensor(self.0.pull(source, options)?))
    }
    fn shape<'a>(&self, tensor: &'a TensorHandle) -> &'a [i32] {
        tensor.shape()
    }
    fn dtype(&self, tensor: &TensorHandle) -> Option<ScalarType> {
        tensor.dtype()
    }
    fn stream(&self) -> StreamHandle {
        StreamHandle {
            owner: self.clone(),
            value: self.0.stream(),
        }
    }
    fn prepare(&self, kernel: Kernel) -> Result<Option<PreparedHandle>> {
        Ok(self.0.prepare(kernel)?.map(|value| PreparedHandle {
            owner: self.clone(),
            value,
        }))
    }
    fn dispatch(
        &self,
        handle: &PreparedHandle,
        inputs: &[&TensorHandle],
        stream: &StreamHandle,
    ) -> Result<Option<Dispatch<Outputs<TensorHandle>, CompletionHandle>>> {
        self.check(&handle.owner)?;
        self.check(&stream.owner)?;
        for tensor in inputs {
            self.check(&tensor.owner)?;
        }
        Ok(self
            .0
            .dispatch(handle.value.as_ref(), inputs, stream.value.as_ref())?
            .map(|(output, completion)| Dispatch {
                output: output.map(|t| self.tensor(t)),
                completion: CompletionHandle {
                    owner: self.clone(),
                    value: completion,
                },
            }))
    }
    fn capture_program(
        &self,
        inputs: &[&TensorHandle],
        stream: &StreamHandle,
        encode: &mut dyn FnMut() -> anyhow::Result<Vec<TensorHandle>>,
    ) -> anyhow::Result<ProgramCapture<TensorHandle>> {
        self.check(&stream.owner)?;
        for tensor in inputs {
            self.check(&tensor.owner)?;
        }
        let (outputs, program) =
            self.0
                .capture_program(inputs, stream.value.as_ref(), &mut || {
                    let outputs = encode()?;
                    for tensor in &outputs {
                        self.check(&tensor.owner)?;
                    }
                    Ok(outputs)
                })?;
        Ok((
            outputs.into_iter().map(|t| self.tensor(t)).collect(),
            program.map(|value| {
                Box::new(ProgramHandle {
                    owner: self.clone(),
                    value,
                }) as Box<dyn RecordedProgram<TensorHandle>>
            }),
        ))
    }
    fn wait(&self, completion: &CompletionHandle) -> Result<()> {
        self.check(&completion.owner)?;
        self.0.wait(completion.value.as_ref())
    }
}
impl StorageDevice for DeviceHandle {
    fn max_allocation_bytes(&self) -> usize {
        self.0.max_allocation_bytes()
    }
    fn byte_view(
        &self,
        source: &TensorHandle,
        offset: usize,
        shape: &[i32],
        dtype: ScalarType,
    ) -> Result<TensorHandle> {
        self.check(&source.owner)?;
        Ok(self.tensor(
            self.0
                .byte_view(source.value.as_ref(), offset, shape, dtype)?,
        ))
    }
}
impl ReadbackDevice for DeviceHandle {
    fn read_f32(&self, tensor: &TensorHandle) -> Result<Vec<f32>> {
        if tensor.dtype != Some(ScalarType::F32) {
            return Err(DeviceError::msg("read_f32 requires F32 storage"));
        }
        Ok(self
            .read_bytes(tensor)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect())
    }
}

impl IndirectDevice for DeviceHandle {
    type Resources = ResourcesHandle;
    fn resources(&self, slots: &[&TensorHandle]) -> Result<ResourcesHandle> {
        for tensor in slots {
            self.check(&tensor.owner)?;
        }
        Ok(ResourcesHandle {
            owner: self.clone(),
            value: self.0.resources(slots)?,
        })
    }
    fn dispatch_indirect(
        &self,
        handle: &PreparedHandle,
        inputs: &[&TensorHandle],
        resources: &ResourcesHandle,
        stream: &StreamHandle,
    ) -> Result<Option<Dispatch<Outputs<TensorHandle>, CompletionHandle>>> {
        self.check(&handle.owner)?;
        self.check(&stream.owner)?;
        self.check(&resources.owner)?;
        for tensor in inputs {
            self.check(&tensor.owner)?;
        }
        Ok(self
            .0
            .dispatch_indirect(
                handle.value.as_ref(),
                inputs,
                resources.value.as_ref(),
                stream.value.as_ref(),
            )?
            .map(|(output, completion)| Dispatch {
                output: output.map(|t| self.tensor(t)),
                completion: CompletionHandle {
                    owner: self.clone(),
                    value: completion,
                },
            }))
    }
}

impl HostAccessDevice for DeviceHandle {
    fn same_storage(&self, a: &TensorHandle, b: &TensorHandle) -> bool {
        self.same_device(&a.owner)
            && self.same_device(&b.owner)
            && self.0.same_storage(a.value.as_ref(), b.value.as_ref())
    }
    fn copy_tensor(&self, tensor: &TensorHandle) -> anyhow::Result<TensorHandle> {
        self.check(&tensor.owner)?;
        Ok(self.tensor(self.0.copy_tensor(tensor.value.as_ref())?))
    }
    fn read_bytes(&self, tensor: &TensorHandle) -> anyhow::Result<Vec<u8>> {
        Ok(DeviceHandle::read_bytes(self, tensor)?)
    }
    fn read_i32_through(&self, tensor: &TensorHandle) -> anyhow::Result<Vec<i32>> {
        self.check(&tensor.owner)?;
        Ok(self.0.read_i32_through(tensor.value.as_ref())?)
    }
    fn read_f32_through(&self, tensor: &TensorHandle) -> anyhow::Result<Vec<f32>> {
        self.check(&tensor.owner)?;
        Ok(self.0.read_f32_through(tensor.value.as_ref())?)
    }
    fn layout(&self, tensor: &TensorHandle) -> TensorLayout {
        tensor.layout()
    }
    fn evaluate(&self, tensor: &TensorHandle, wait: bool) -> anyhow::Result<()> {
        Ok(DeviceHandle::evaluate(self, tensor, wait)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct Tensor {
        bytes: Arc<Vec<u8>>,
        shape: Vec<i32>,
        dtype: ScalarType,
    }
    struct Provider {
        record: bool,
        replays: Arc<AtomicUsize>,
    }
    struct IdentityProgram(Arc<AtomicUsize>);
    impl RecordedProgram<Tensor> for IdentityProgram {
        fn set_scheduling(&mut self, _: ProgramScheduling) -> bool {
            true
        }
        fn replay(&self, inputs: &[&Tensor]) -> anyhow::Result<Option<Vec<Tensor>>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            if inputs.len() != 1 || inputs[0].shape != [1] {
                return Ok(None);
            }
            Ok(Some(vec![inputs[0].clone()]))
        }
    }
    impl Device for Provider {
        type Tensor = Tensor;
        type KernelHandle = ();
        type Stream = ();
        type Completion = ();
        type Error = DeviceError;
        fn capabilities(&self) -> &[KernelCapability] {
            &[]
        }
        fn pull(&self, source: TensorSource<'_>, _: ImportOptions) -> Result<Tensor> {
            let source = source.host_tensor().unwrap();
            Ok(Tensor {
                bytes: Arc::new(source.bytes.to_vec()),
                shape: source.shape.to_vec(),
                dtype: source.dtype,
            })
        }
        fn shape<'a>(&self, tensor: &'a Tensor) -> &'a [i32] {
            &tensor.shape
        }
        fn dtype(&self, tensor: &Tensor) -> Option<ScalarType> {
            Some(tensor.dtype)
        }
        fn stream(&self) {}
        fn prepare(&self, _: Kernel) -> Result<Option<()>> {
            Ok(None)
        }
        fn dispatch(
            &self,
            _: &(),
            _: &[&Tensor],
            _: &(),
        ) -> Result<Option<Dispatch<Outputs<Tensor>, ()>>> {
            Ok(None)
        }
        fn wait(&self, _: &()) -> Result<()> {
            Ok(())
        }
        fn capture_program(
            &self,
            _: &[&Tensor],
            _: &(),
            encode: &mut dyn FnMut() -> anyhow::Result<Vec<Tensor>>,
        ) -> anyhow::Result<ProgramCapture<Tensor>> {
            Ok((
                encode()?,
                self.record.then(|| {
                    Box::new(IdentityProgram(self.replays.clone()))
                        as Box<dyn RecordedProgram<Tensor>>
                }),
            ))
        }
    }
    impl StorageDevice for Provider {
        fn max_allocation_bytes(&self) -> usize {
            usize::MAX
        }
        fn byte_view(&self, _: &Tensor, _: usize, _: &[i32], _: ScalarType) -> Result<Tensor> {
            Err(DeviceError::msg("unused test operation"))
        }
    }
    impl HostAccessDevice for Provider {
        fn same_storage(&self, a: &Tensor, b: &Tensor) -> bool {
            Arc::ptr_eq(&a.bytes, &b.bytes)
        }
        fn read_bytes(&self, tensor: &Tensor) -> anyhow::Result<Vec<u8>> {
            Ok(tensor.bytes.as_ref().clone())
        }
        fn read_i32_through(&self, tensor: &Tensor) -> anyhow::Result<Vec<i32>> {
            anyhow::ensure!(tensor.dtype == ScalarType::I32, "expected I32 test storage");
            self.replays.fetch_add(1, Ordering::Relaxed);
            Ok(tensor
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_ne_bytes(*b))
                .collect())
        }
        fn layout(&self, _: &Tensor) -> TensorLayout {
            TensorLayout {
                strides: vec![1],
                offset: 0,
            }
        }
        fn evaluate(&self, _: &Tensor, _: bool) -> anyhow::Result<()> {
            Ok(())
        }
    }
    fn new_device(record: bool) -> (DeviceHandle, Arc<AtomicUsize>) {
        let replays = Arc::new(AtomicUsize::new(0));
        (
            DeviceHandle::new(Provider {
                record,
                replays: replays.clone(),
            }),
            replays,
        )
    }
    fn tensor(device: &DeviceHandle, values: &[u8]) -> TensorHandle {
        device
            .pull(
                HostTensor {
                    bytes: values,
                    shape: &[values.len() as i32],
                    dtype: ScalarType::U8,
                }
                .into(),
                ImportOptions::default(),
            )
            .unwrap()
    }

    #[test]
    fn recorded_program_scheduling_hint_reaches_erased_provider() -> anyhow::Result<()> {
        let (device, _) = new_device(true);
        let a = tensor(&device, &[7]);
        let stream = device.stream();
        let (_, program) = device.capture_program(&[&a], &stream, &mut || Ok(vec![a.clone()]))?;
        let mut program = program.unwrap();
        assert!(program.set_scheduling(ProgramScheduling::DependencyParallel));
        assert!(program.set_scheduling(ProgramScheduling::Ordered));
        Ok(())
    }

    #[test]
    fn recorded_program_preserves_owner_storage_and_declines_shapes() -> anyhow::Result<()> {
        let (device, calls) = new_device(true);
        let owner = Arc::downgrade(&device.0);
        let a = tensor(&device, &[7]);
        let stream = device.stream();
        let (out, program) = device.capture_program(&[&a], &stream, &mut || Ok(vec![a.clone()]))?;
        assert!(device.same_storage(&a, &out[0]));
        let program = program.unwrap();
        let b = tensor(&device, &[9]);
        let replay = program.replay(&[&b])?.unwrap();
        assert!(device.same_storage(&b, &replay[0]));
        assert_eq!(device.read_bytes(&replay[0])?, vec![9]);
        let wide = tensor(&device, &[1, 2]);
        assert!(program.replay(&[&wide])?.is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        drop((a, b, wide, out, replay, stream, device));
        assert!(owner.upgrade().is_some(), "program must retain its device");
        drop(program);
        assert!(owner.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn capture_and_replay_reject_foreign_owners_before_provider_calls() -> anyhow::Result<()> {
        let (device, calls) = new_device(true);
        let (other, _) = new_device(false);
        let a = tensor(&device, &[7]);
        let foreign = tensor(&other, &[8]);
        let stream = device.stream();
        let mut encoded = 0;
        let mut encode = || {
            encoded += 1;
            Ok(vec![a.clone()])
        };
        assert!(
            device
                .capture_program(&[&foreign], &stream, &mut encode)
                .is_err()
        );
        assert!(
            device
                .capture_program(&[&a], &other.stream(), &mut encode)
                .is_err()
        );
        assert_eq!(encoded, 0);
        assert!(
            device
                .capture_program(&[&a], &stream, &mut || Ok(vec![foreign.clone()]))
                .is_err()
        );
        let (_, program) = device.capture_program(&[&a], &stream, &mut || Ok(vec![a.clone()]))?;
        assert!(program.unwrap().replay(&[&foreign]).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn declined_capture_preserves_outputs_and_errors() -> anyhow::Result<()> {
        let (device, _) = new_device(false);
        let a = tensor(&device, &[7]);
        let stream = device.stream();
        let (out, program) = device.capture_program(&[&a], &stream, &mut || Ok(vec![a.clone()]))?;
        assert!(program.is_none());
        assert!(device.same_storage(&a, &out[0]));
        let failed = device.capture_program(&[&a], &stream, &mut || {
            anyhow::bail!("capture encoding failed")
        });
        assert_eq!(failed.err().unwrap().to_string(), "capture encoding failed");
        Ok(())
    }
    #[test]
    fn integer_partial_readback_preserves_provider_fence_and_owner_check() -> anyhow::Result<()> {
        let (device, calls) = new_device(false);
        let (other, _) = new_device(false);
        let bytes = 42i32.to_ne_bytes();
        let token = device.pull(
            HostTensor {
                bytes: &bytes,
                shape: &[1, 1],
                dtype: ScalarType::I32,
            }
            .into(),
            ImportOptions::default(),
        )?;
        assert_eq!(HostAccessDevice::read_i32_through(&device, &token)?, [42]);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "erasure fell back to a full byte read instead of the provider fence"
        );
        assert!(HostAccessDevice::read_i32_through(&other, &token).is_err());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "foreign read reached the original provider"
        );
        Ok(())
    }
}
