//! Device-independent bindings for model-declared native session storage.
use anyhow::{Result, ensure};
use lisa_backend::HostAccessDevice;
use lisa_models::deepseek41::{engram::History, state as model};
pub use model::MAX_CONTEXT;
pub trait StateDevice:
    HostAccessDevice<Tensor: Clone + Send + Sync> + Clone + Send + Sync + 'static
{
}
impl<D> StateDevice for D
where
    D: HostAccessDevice + Clone + Send + Sync + 'static,
    D::Tensor: Clone + Send + Sync,
{
}
pub struct Owner<T> {
    pub compressed: T,
    pub index: T,
    pub index_bf16: Option<T>,
    pub previous_kv: T,
    pub previous_score: T,
}

pub struct State<D: StateDevice> {
    device: D,
    pub position: usize,
    pub capacity: usize,
    pub valid: bool,
    pub history: History,
    pub raw: Vec<D::Tensor>,
    pub owners: Vec<Owner<D::Tensor>>,
}

impl<D: StateDevice> State<D> {
    /// Native caches reserve their full configured capacity on first use.
    /// Admission must count this even for a short request or restored prefix.
    pub fn storage_bytes(capacity: usize) -> usize {
        model::storage_bytes(capacity)
    }

    /// Infallible empty session descriptor; allocation is deferred until the
    /// model's first forward, where allocation failures can be returned.
    pub fn empty(device: D, capacity: usize) -> Self {
        Self {
            device: device.clone(),
            position: 0,
            capacity,
            valid: true,
            history: History::default(),
            raw: Vec::new(),
            owners: Vec::new(),
        }
    }

    pub fn initialize(&mut self) -> Result<()> {
        if self.raw.is_empty() {
            *self = Self::new(self.device.clone(), self.capacity)?;
        }
        Ok(())
    }
    /// Reserve the configured context under the runtime's RAM budget. The
    /// BF16 mirror is needed only if a compressed owner can reach 8192 keys.
    pub fn new(device: D, capacity: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT).contains(&capacity),
            "DeepSeek context capacity must be between 1 and {MAX_CONTEXT}"
        );
        let plan = crate::core::graph::PreparedGraph::new(
            device.clone(),
            model::allocate(capacity)?,
            vec![],
        )?;
        let mut out = plan.run(&[], &mut [])?.into_iter();
        let raw = out.by_ref().take(model::LAYERS).collect();
        let owners = model::layout(capacity)
            .iter()
            .map(|owner| Owner {
                compressed: out.next().expect("validated cache layout"),
                index: out.next().expect("validated cache layout"),
                index_bf16: if owner.mirror {
                    Some(out.next().expect("validated cache layout"))
                } else {
                    None
                },
                previous_kv: out.next().expect("validated cache layout"),
                previous_score: out.next().expect("validated cache layout"),
            })
            .collect();
        Ok(Self {
            device: device.clone(),
            position: 0,
            capacity,
            valid: true,
            history: History::default(),
            raw,
            owners,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// An independent GPU copy, including raw ring contents and odd-token
    /// compressor carry. Cloning handles alone would corrupt rollback at wrap.
    pub fn snapshot(&self) -> Result<Self> {
        ensure!(self.valid, "cannot snapshot an invalid DeepSeek session");
        let copy = |x: &D::Tensor| self.device.copy_tensor(x);
        let raw = self.raw.iter().map(copy).collect::<Result<Vec<_>>>()?;
        let owners = self
            .owners
            .iter()
            .map(|owner| -> Result<_> {
                Ok(Owner {
                    compressed: copy(&owner.compressed)?,
                    index: copy(&owner.index)?,
                    index_bf16: owner.index_bf16.as_ref().map(copy).transpose()?,
                    previous_kv: copy(&owner.previous_kv)?,
                    previous_score: copy(&owner.previous_score)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device: self.device.clone(),
            position: self.position,
            capacity: self.capacity,
            valid: true,
            history: self.history,
            raw,
            owners,
        })
    }
}

impl<D: StateDevice> crate::cache::NativeState for State<D> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn copy_for_write(&self) -> Result<std::sync::Arc<dyn crate::cache::NativeState>> {
        Ok(std::sync::Arc::new(self.snapshot()?))
    }
    fn position(&self) -> usize {
        self.position
    }
}
