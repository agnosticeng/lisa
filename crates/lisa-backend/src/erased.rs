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
    D::Tensor: Send + Sync + 'static,
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
        D::Tensor: Send + Sync + 'static,
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
        D::Tensor: Send + Sync + 'static,
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
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
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
