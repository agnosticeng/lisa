use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice,
    MTLFunction, MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode, MTLResourceOptions,
};

use crate::error::Result;

use super::buffer::{Buffer, BufferPool};
use super::commands::Commands;
use super::env::{ALLOC_COUNT, LABEL_COUNTS, POOLHIT_COUNT, err, ns};
use super::library_cache::LibraryCache;
use super::pipeline::ComputePipeline;
use super::program::{self, AllocationPolicy};

thread_local! {
    static ORDERED_GPU_REUSE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Enables reuse only at explicitly marked GPU-output allocation sites.
/// The guard is thread-bound; normal and host-written allocations never use
/// this policy. Dropping it restores the previous nested scope without a wait:
/// each GPU bind updates the buffer's completion fence for later CPU leases.
pub struct OrderedGpuReuseScope {
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

pub fn enter_ordered_gpu_reuse_scope() -> OrderedGpuReuseScope {
    ORDERED_GPU_REUSE_DEPTH.with(|depth| depth.set(depth.get() + 1));
    OrderedGpuReuseScope {
        _thread_bound: std::marker::PhantomData,
    }
}

impl Drop for OrderedGpuReuseScope {
    fn drop(&mut self) {
        ORDERED_GPU_REUSE_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

pub(super) fn ordered_gpu_reuse_enabled() -> bool {
    ORDERED_GPU_REUSE_DEPTH.with(|depth| depth.get() != 0)
}

const ORDERED_GPU_ZERO_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("common/data/ordered_zero.metal");

// ───────────────────────────── runtime ─────────────────────────────

// SAFETY: MTLDevice and MTLCommandQueue are thread-safe Metal objects; every other
// field is an `Arc`, `Mutex`, atomic or immutable `bool`, so shared access is synchronised.
unsafe impl Send for MetalRuntime {}
unsafe impl Sync for MetalRuntime {}

/// Owns the device, queue, pool, pipeline cache and command batching.
/// A process-wide mirror of the last-created runtime's NAX capability, for
/// engine code that has no easy handle on the device.
static NAX_AVAILABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the MPP tensor ops (M5/NAX) are available to the active runtime.
pub fn nax_available() -> bool {
    NAX_AVAILABLE.load(Ordering::Relaxed)
}

/// Total number of Metal pipeline compilations performed (cache misses in
/// `MetalRuntime::compile_full`). A stable value after model load means no
/// JIT compilation happens at runtime.
pub static JIT_COMPILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ─────────────── O2: kernel identity / numerical-law fingerprint ───────────────
//
// A compiled kernel's arithmetic is a *numerical law* — the same projection
// computes different bits under a different math mode or tensor-unit path — so
// `law_id` is bound into "any cache identity that stores state computed under
// it".
//
// The law is a CONSTANT identity, not a per-call source hash: both `compile_full`
// and the JIT fast path are hit once PER DISPATCH, so anything expensive in the
// key is a per-dispatch tax (an earlier revision hashed the source there and cost
// 2x decode — the golden gate caught it). Runtime-varying axes go in the key;
// the source body is a build-time artifact, covered by the twin-check below
// (slow path only) plus "kernels compile once at launch".

/// FNV-1a over bytes — the same 64-bit hash the JIT fast path already uses.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// name -> source fingerprint of every pipeline compiled in this process.
static PIPELINE_HASHES: std::sync::OnceLock<Mutex<HashMap<String, u64>>> =
    std::sync::OnceLock::new();

fn pipeline_hashes() -> &'static Mutex<HashMap<String, u64>> {
    PIPELINE_HASHES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The process-wide numerical-law fingerprint: folds the sorted (name, source)
/// set of every compiled pipeline. Order-independent, stable across a run.
/// Bind this into any cache key that stores KV/GDN state (O2).
pub fn kernel_fingerprint() -> u64 {
    let reg = pipeline_hashes().lock().unwrap();
    let mut names: Vec<&String> = reg.keys().collect();
    names.sort();
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    for n in names {
        for b in n.as_bytes() {
            acc ^= *b as u64;
            acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
        }
        acc ^= reg[n];
        acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
    acc
}

/// The FROZEN numerical law — what caches must key on, NOT the live
/// [`kernel_fingerprint`].
///
/// `kernel_fingerprint` is a *growing* set: every lazily-compiled shape adds a
/// pipeline, so a fingerprint recomputed at lookup time differs from the one
/// captured at insert as soon as one new context length is seen. Keying a cache
/// on the live value therefore **empties the cache mid-session** — measured: a
/// multi-turn pair resumed 683/710 tokens, and the identical pair after a single
/// long request resumed 0. The law is an identity of the *build*, so it is
/// captured once and frozen.
pub fn kernel_law() -> u64 {
    static LAW: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LAW.get_or_init(|| {
        // Binding arithmetic is part of the numerical law even when the
        // shader source is unchanged. Reject persisted states produced before
        // normalization and rotary views used byte offsets at the Metal API.
        kernel_fingerprint() ^ fnv1a(b"lisa.tensor-byte-offset-bindings.v2")
    })
}

/// Record a pipeline's source fingerprint — SLOW PATH ONLY (once per compile).
/// A same-name compile with a different body is a warm-cache hazard and fails
/// immediately. Cannot see a source edit that never
/// recompiles; that axis is covered by "kernels compile once at launch" + the
/// goldens (documented limitation, not a silent gap).
fn record_pipeline(name: &str, source_hash: u64) {
    let mut reg = pipeline_hashes().lock().unwrap();
    if let Some(prev) = reg.get(name)
        && *prev != source_hash
    {
        panic!(
            "[jit] twin-mismatch on `{name}`: src {prev:016x} -> {source_hash:016x} \
             under a warm pipeline cache"
        );
    }
    reg.insert(name.to_string(), source_hash);
}

/// Pipelines compiled so far in this process.
pub fn jit_compiles() -> u64 {
    JIT_COMPILES.load(Ordering::Relaxed)
}

pub struct MetalRuntime {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: Arc<BufferPool>,
    pub commands: Arc<Commands>,
    /// Compiled pipelines by key hash; entries keep the full key for equality.
    pipelines: Mutex<HashMap<u64, Vec<(PipelineKey, ComputePipeline)>>>,
    libraries: Mutex<LibraryCache<Retained<ProtocolObject<dyn MTLLibrary>>>>,
    allocation_limit: AtomicUsize,
    /// Whether the MetalPerformancePrimitives tensor ops (the M5/Apple10
    /// "NAX" path) are usable on this GPU. False on M1–M4, where the runtime
    /// routes to the non-NAX steel/vector kernels instead.
    nax: bool,
}

/// Which compile options a kernel needs. The two paths in the tree disagree,
/// and the disagreement is load-bearing for bit-exactness:
///
/// - `Safe` matches MLX's `device.cpp::set_compile_options` (and the runtime's
///   `compile_builtin`): `MathModeSafe` + `Precise` + an explicit language
///   version. Leaving `Precise` off shifted `metal::exp` by 1 ulp on the MLX
///   unary kernels; `fastMathEnabled(false)` with a default language version
///   moved `silu_head` by ~6e-5 on large inputs.
/// - `Fast` matches the kernels crate' `get_compile_options`, which compiles
///   *own* kernels (binary, reduce, indexing, ...) with
///   `MathModeFast` + `Fast` and the default language version. Without it,
///   `bdiv` differs by 1 ulp (fast math turns division into a reciprocal
///   multiply).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Math {
    /// Leave Metal's compile options at their platform defaults. The reference creates
    /// one library with a fresh `MTLCompileOptions` and does not override its
    /// math or language settings on the normal path; this mode lets a ported
    /// kernel match that compiler contract instead of approximating it with
    /// `Fast`.
    CompilerDefault,
    /// mlx_rt's JIT `MetalKernel` path: Safe + an explicit language version,
    /// but **no** `Precise` floating-point functions (its own `compile_options`).
    Jit,
    /// MLX's `device.cpp::set_compile_options`: Safe + Precise + an explicit
    /// language version.
    Safe,
    /// `compile_builtin` path (the MLX builtin/quantized kernels):
    /// Safe + Precise, but **no** language version — setting one changes the
    /// generated code for these kernels.
    SafeNoLang,
    Fast,
}

/// A specialised Metal function constant.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ConstVal {
    Bool(bool),
    Int(i32),
    UInt(u32),
    Short(i16),
}

/// Full identity of a cached pipeline.
struct PipelineKey {
    name: Box<str>,
    math: Math,
    nax: bool,
    consts: Vec<(usize, ConstVal)>,
}

impl PipelineKey {
    fn matches(&self, name: &str, math: Math, nax: bool, consts: &[(usize, ConstVal)]) -> bool {
        &*self.name == name && self.math == math && self.nax == nax && self.consts == consts
    }
}

/// Check before creating a runtime or loading weights. The compiler's separate
/// math controls and MSL 3.2 require macOS 15 or iOS 18; older Objective-C
/// runtimes throw for these selectors rather than returning a Metal error.
/// The default device's Apple GPU family (7 for M1-class GPUs, 9 for M3/M4,
/// 10 for M5, ...), or 0 without an Apple family. Families nest, so this is
/// the highest one the device supports.
pub fn apple_gpu_family() -> u32 {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return 0;
    };
    let mut family = 0;
    for next in 1..=20 {
        if device.supportsFamily(MTLGPUFamily(1000 + next)) {
            family = next as u32;
        } else if next >= 7 {
            break;
        }
    }
    family
}

/// Whether the default device runs the Metal 4 language and its tensor
/// operations: an Apple7 (M1-class) or newer GPU on macOS or iOS 26 or
/// later. Kernels compiled at the OS's default language version need it when
/// they use Metal 4 features; it is distinct from [`nax_available`], the
/// Apple10 neural-accelerator path.
pub fn metal4_supported() -> bool {
    if !objc2::available!(macos = 26.0, ios = 26.0) {
        return false;
    }
    MTLCreateSystemDefaultDevice().is_some_and(|device| device.supportsFamily(MTLGPUFamily::Metal4))
}

pub fn check_platform_support() -> Result<()> {
    let version = objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersion();
    #[cfg(target_os = "ios")]
    {
        if version.majorVersion < 18 {
            return Err(err(
                "LISA requires iOS/iPadOS 18 or later for Metal inference",
            ));
        }
        return Ok(());
    }
    #[cfg(not(target_os = "ios"))]
    check_macos_version(
        version.majorVersion,
        version.minorVersion,
        version.patchVersion,
    )
}

#[cfg(any(test, not(target_os = "ios")))]
fn check_macos_version(major: isize, minor: isize, patch: isize) -> Result<()> {
    if major < 15 {
        return Err(err(format!(
            "LISA requires macOS 15.0 or later for Metal inference. \
             This Mac runs macOS {major}.{minor}.{patch}. Update macOS to load models."
        )));
    }
    Ok(())
}

fn compile_options(math: Math, nax: bool) -> Retained<MTLCompileOptions> {
    use objc2_metal::MTLLanguageVersion;
    let opts = MTLCompileOptions::new();
    match math {
        Math::CompilerDefault => {}
        Math::SafeNoLang => {
            opts.setMathMode(MTLMathMode::Safe);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        }
        Math::Jit | Math::Safe => {
            opts.setMathMode(MTLMathMode::Safe);
            if math == Math::Safe {
                opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            }
            // Metal 4 language features (MPP tensor ops) are only legal on the
            // GPUs that support them; older GPUs stay on MSL 3.2.
            let lang = if !nax {
                (3 << 16) + 2
            } else if objc2::available!(macos = 27.0, ios = 27.0) {
                (4 << 16) + 1
            } else if objc2::available!(macos = 26.0, ios = 26.0) {
                4 << 16
            } else {
                (3 << 16) + 2
            };
            opts.setLanguageVersion(MTLLanguageVersion(lang));
        }
        Math::Fast => {
            opts.setMathMode(MTLMathMode::Fast);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
        }
    }
    opts
}

impl MetalRuntime {
    pub fn new(per_buffer: usize) -> Result<Self> {
        check_platform_support()?;
        crate::shader_pack::initialize()?;
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            MTLCreateSystemDefaultDevice().ok_or_else(|| err("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| err("newCommandQueue"))?;
        let pool = Arc::new(BufferPool::new());
        // Miss-time retention threshold when no resident hard budget is active.
        // Reuse adds no storage; the hard allocation limits stay independent.
        pool.set_default_limit(512 << 20);
        let commands = Arc::new(Commands::new(queue.clone(), device.clone(), per_buffer)?);
        // MPP tensor ops are the Apple10 (M5) "NAX" path; M1–M4 fall back.
        let nax = device.supportsFamily(MTLGPUFamily::Apple10);
        NAX_AVAILABLE.store(nax, Ordering::Relaxed);
        Ok(Self {
            device,
            queue,
            pool,
            commands,
            pipelines: Mutex::new(HashMap::new()),
            libraries: Mutex::new(LibraryCache::default()),
            allocation_limit: AtomicUsize::new(0),
            nax,
        })
    }

    /// Native allocations include resources still retained by submitted work,
    /// even after their Array and pool leases have been released.
    pub fn allocated_bytes(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    /// Application-supplied total native allocation limit; zero disables it.
    /// No platform or model admission policy belongs in the device runtime.
    pub fn swap_allocation_limit(&self, bytes: usize) -> usize {
        self.allocation_limit.swap(bytes, Ordering::AcqRel)
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Whether the MPP tensor ops (the M5/NAX path) are available. When false,
    /// callers must use the non-NAX fallbacks.
    pub fn nax(&self) -> bool {
        self.nax
    }

    /// The raw `MTLDevice` (for callers that need it directly).
    pub fn metal_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Device identity for measured, hardware-specific execution policies.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// GPU architecture name used to pick NAX tiles and bind measured profiles.
    pub fn architecture_name(&self) -> String {
        self.device.architecture().name().to_string()
    }

    /// Commit and wait (`Device::synchronize`).
    pub fn synchronize(&self) -> Result<()> {
        self.commands.flush_and_wait()
    }

    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Allocate (or reuse) shared storage with the requested size bucket zeroed.
    /// The buffer may have larger capacity; that unused tail is not initialized.
    pub fn buffer(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        let buffer = self.buffer_inner(bytes, label, true, false)?;
        program::allocation(
            self.commands.program_identity(),
            &buffer,
            bytes,
            AllocationPolicy::Zeroed,
        );
        Ok(buffer)
    }

    /// Allocate shared storage for a caller that will overwrite every logical
    /// byte before exposing the resulting array. This avoids clearing large
    /// imported weight tensors immediately before copying their full payload.
    pub fn buffer_for_full_write(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        program::unsupported(self.commands.program_identity());
        self.buffer_inner(bytes, label, false, false)
    }

    /// Completion-gated CPU import storage with exactly the requested 1 KiB
    /// capacity bucket. The caller overwrites every logical byte; unused bucket
    /// padding remains unspecified. Even an ordered GPU scope cannot permit a
    /// CPU writer to reuse in-flight storage. Host imports cannot be replayed.
    pub fn buffer_for_compact_full_write(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        program::unsupported(self.commands.program_identity());
        self.buffer_inner_with_capacity(bytes, label, false, false, true)
    }

    /// Allocate an output that a GPU kernel overwrites completely. Within an
    /// explicit ordered-reuse scope, unleased storage may still have earlier
    /// GPU readers; the new write is encoded after them on our one serial queue.
    /// Never expose the returned storage to a CPU writer before synchronization.
    pub fn buffer_for_full_gpu_write(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        if !ordered_gpu_reuse_enabled() {
            return self.buffer(bytes, label);
        }
        let buffer = self.buffer_inner(bytes, label, false, true)?;
        program::allocation(
            self.commands.program_identity(),
            &buffer,
            bytes,
            AllocationPolicy::FullGpuWrite,
        );
        Ok(buffer)
    }

    /// Storage for an audited GPU producer that writes every logical byte
    /// before any consumer reads it. No CPU clear is performed, including on
    /// completion-gated reuse outside an ordered scope. Bucket padding remains
    /// unspecified and must not be read. Ordered scopes retain their existing
    /// serial-queue reuse law; this method never enables that policy itself.
    pub fn buffer_for_audited_full_gpu_write(
        &self,
        bytes: usize,
        label: &str,
    ) -> Result<Arc<Buffer>> {
        let buffer = self.buffer_inner(bytes, label, false, ordered_gpu_reuse_enabled())?;
        program::allocation(
            self.commands.program_identity(),
            &buffer,
            bytes,
            AllocationPolicy::FullGpuWrite,
        );
        Ok(buffer)
    }

    /// Audited full GPU output whose retained capacity is exactly the requested
    /// 1 KiB bucket. A small persistent state must not inherit a large scratch
    /// allocation through best-fit reuse. Completion and ordered-scope rules
    /// are unchanged; this method does not enable ordered reuse.
    pub fn buffer_for_audited_compact_gpu_write(
        &self,
        bytes: usize,
        label: &str,
    ) -> Result<Arc<Buffer>> {
        let buffer = self.buffer_inner_with_capacity(
            bytes,
            label,
            false,
            ordered_gpu_reuse_enabled(),
            true,
        )?;
        program::allocation(
            self.commands.program_identity(),
            &buffer,
            bytes,
            AllocationPolicy::CompactFullGpuWrite,
        );
        Ok(buffer)
    }

    /// Queue-private, exact-capacity storage for a sealed program's disjoint
    /// nonescaping full-write slots. Only replay owns these leases. Dropping
    /// its last Arc proves every old access has already been encoded, allowing
    /// the next full GPU producer on this Commands queue to reuse the storage.
    /// No general allocator or CPU import can draw from this private lane.
    pub(super) fn buffer_for_recorded_scratch(&self, bytes: usize) -> Result<Arc<Buffer>> {
        let identity = self.commands.program_identity();
        if program::is_capturing(identity) || ordered_gpu_reuse_enabled() {
            // Captures retain ordinary allocation identities; this mechanism
            // neither nests nor enables the independent GLM reuse scope.
            return self.buffer_for_audited_compact_gpu_write(bytes, "dispatch_program_scratch");
        }
        self.buffer_inner_in_lane(
            bytes,
            "dispatch_program_scratch",
            false,
            false,
            true,
            Some(identity),
        )
    }

    /// Allocate a GPU output with the same zero-initialized contract as
    /// `buffer`. In the ordered scope its clear is a GPU command, so partial
    /// writes and atomic accumulations cannot race an earlier GPU reader.
    pub fn buffer_for_gpu_write(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        if !ordered_gpu_reuse_enabled() {
            return self.buffer(bytes, label);
        }
        self.buffer_cleared_by_gpu(bytes, label, true)
    }

    /// Zeroed storage whose clear is always a GPU command on this queue, in
    /// or out of the ordered scope: large state skips the host clear. Outside
    /// the scope the lease is completion-gated, so the clear has no earlier
    /// GPU user to race. A CPU reader must synchronize first, as for any
    /// kernel output.
    pub fn buffer_zeroed_on_gpu(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        self.buffer_cleared_by_gpu(bytes, label, ordered_gpu_reuse_enabled())
    }

    fn buffer_cleared_by_gpu(
        &self,
        bytes: usize,
        label: &str,
        ordered_gpu_write: bool,
    ) -> Result<Arc<Buffer>> {
        let buffer = self.buffer_inner(bytes, label, false, ordered_gpu_write)?;
        // Its zeroing dispatch is recorded below, so replay allocates the
        // same uncleared storage and replays that clear exactly once.
        program::allocation(
            self.commands.program_identity(),
            &buffer,
            bytes,
            AllocationPolicy::FullGpuWrite,
        );
        // Only the requested output is published. A best-fit lease may be
        // much larger; clearing its unused tail would waste GPU bandwidth.
        let words = bytes.div_ceil(std::mem::size_of::<u32>());
        if words == 0 {
            return Ok(buffer);
        }
        let pipeline = self.compile(ORDERED_GPU_ZERO_SOURCE, "lisa_ordered_zero_u32")?;
        let width = words.min(1 << 20);
        let guard = self.commands.encoder()?;
        let enc = guard.encoder();
        enc.set_pipeline(&pipeline);
        enc.set_output(0, Some(&buffer), 0);
        enc.set_bytes(1, &(words as u64));
        enc.set_bytes(2, &(width as u32));
        enc.dispatch_threads((width, words.div_ceil(width), 1), (256.min(width), 1, 1));
        Ok(buffer)
    }

    /// Arm a lifetime-scoped RAM budget for an immutable mapped model.
    pub fn reserve_model_memory(
        &self,
        bytes: usize,
        budget: usize,
    ) -> Result<super::buffer::ResidentMemoryReservation> {
        self.pool
            .reserve_model_memory(bytes, budget, self.commands.clone())
    }

    /// Wrap a page-aligned, externally owned read-only region in a Metal
    /// buffer without copying it. The deallocator retains `owner` until Metal
    /// releases the buffer, including while a command buffer is in flight.
    pub fn buffer_from_external(
        &self,
        pointer: NonNull<c_void>,
        length: usize,
        owner: Arc<dyn std::any::Any + Send + Sync>,
        label: &str,
    ) -> Result<Arc<Buffer>> {
        program::unsupported(self.commands.program_identity());
        if length == 0 || length > self.device.maxBufferLength() {
            return Err(err(format!(
                "external Metal buffer length {length} is invalid"
            )));
        }
        // SAFETY: `getpagesize` takes no arguments and only reports the VM page size.
        let page_size = unsafe { getpagesize() };
        if page_size <= 0
            || !(pointer.as_ptr() as usize).is_multiple_of(page_size as usize)
            || !length.is_multiple_of(page_size as usize)
        {
            return Err(err("external Metal buffer must be page-aligned"));
        }
        let deallocator = RcBlock::new(move |_pointer: NonNull<c_void>, _length: usize| {
            // Capturing the owner retains the mapping until Metal releases this block.
            let _ = &owner;
        });
        // SAFETY: the region is non-empty, page-aligned and within `maxBufferLength` (checked
        // above); the deallocator captures `owner`, keeping the mapping alive until Metal releases it.
        let raw = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    length,
                    MTLResourceOptions::StorageModeShared,
                    Some(&deallocator),
                )
        }
        .ok_or_else(|| err(format!("newBufferWithBytesNoCopy({length})")))?;
        let buf = Arc::new(Buffer::wrap(raw));
        buf.set_label(label);
        Ok(buf)
    }

    fn buffer_inner(
        &self,
        bytes: usize,
        label: &str,
        clear: bool,
        ordered_gpu_write: bool,
    ) -> Result<Arc<Buffer>> {
        self.buffer_inner_with_capacity(bytes, label, clear, ordered_gpu_write, false)
    }

    fn buffer_inner_with_capacity(
        &self,
        bytes: usize,
        label: &str,
        clear: bool,
        ordered_gpu_write: bool,
        exact_bucket: bool,
    ) -> Result<Arc<Buffer>> {
        self.buffer_inner_in_lane(bytes, label, clear, ordered_gpu_write, exact_bucket, None)
    }

    fn buffer_inner_in_lane(
        &self,
        bytes: usize,
        label: &str,
        clear: bool,
        ordered_gpu_write: bool,
        exact_bucket: bool,
        program_queue: Option<usize>,
    ) -> Result<Arc<Buffer>> {
        debug_assert!(
            !clear || !ordered_gpu_write,
            "in-flight GPU storage must not be cleared on the CPU"
        );
        debug_assert!(program_queue.is_none() || (!clear && !ordered_gpu_write && exact_bucket));
        let allocation_bytes = bytes
            .max(1)
            .checked_add(1023)
            .map(|rounded| rounded / 1024 * 1024)
            .ok_or_else(|| err("Metal allocation size overflow"))?;
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        if super::env::ALLOCATION_LABELS.load(Ordering::Relaxed) || crate::trace::enabled() {
            let mut c = LABEL_COUNTS.lock().unwrap();
            match c.iter_mut().find(|(l, _)| l == label) {
                Some(e) => e.1 += 1,
                None => c.push((label.to_string(), 1)),
            }
        }
        let completed = self.commands.completed_watermark();
        let reusable = if let Some(queue) = program_queue {
            self.pool.take_program_scratch(bytes, queue)
        } else if exact_bucket {
            self.pool.take_exact(bytes, completed, ordered_gpu_write)
        } else if ordered_gpu_write {
            self.pool.take_for_ordered_gpu_write(bytes)
        } else {
            self.pool.take(bytes, completed)
        };
        if let Some(b) = reusable {
            POOLHIT_COUNT.fetch_add(1, Ordering::Relaxed);
            b.set_label(label);
            if clear {
                // SAFETY: completion-gated `take*` leases have no GPU user; pool lengths are 1 KiB multiples
                // >= `bytes`. Spare best-fit capacity stays uncleared to avoid stalling context growth.
                unsafe { std::ptr::write_bytes(b.contents(), 0, allocation_bytes) };
            }
            return Ok(b);
        }
        // Soft reclamation is independent of hard admission. The default
        // threshold yields to a live resident budget; explicit caller limits
        // retain their existing sweep policy. New allocations still pass both
        // unchanged hard-budget checks below, and hits add no storage.
        self.pool
            .sweep_on_allocation_miss(completed, program_queue.is_some());
        let limit = self.allocation_limit.load(Ordering::Acquire);
        let mut completed = completed;
        if limit > 0 && self.allocated_bytes().saturating_add(allocation_bytes) > limit {
            // End pending GPU users before reclaiming their buffers. This is
            // outside both allocator locks and keeps every kernel/shape intact.
            self.synchronize()?;
            completed = self.commands.completed_watermark();
            self.pool.sweep_completed(completed);
            let allocated = self.allocated_bytes();
            if allocated.saturating_add(allocation_bytes) > limit {
                return Err(crate::error::Error::AllocationUnavailable(format!(
                    "Metal memory budget exceeded: {} MiB allocated + {} MiB requested > {} MiB limit",
                    allocated >> 20,
                    allocation_bytes >> 20,
                    limit >> 20
                )));
            }
        }
        let create = || {
            let raw = self
                .device
                .newBufferWithLength_options(
                    allocation_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| {
                    crate::error::Error::AllocationUnavailable(format!("newBuffer({bytes})"))
                })?;
            let buf = Arc::new(Buffer::wrap(raw));
            buf.set_label(label);
            // General-purpose buffers start at zero so a kernel that reads a region
            // its producer never wrote cannot observe uninitialized storage. Full
            // host-write imports skip this; their caller overwrites the logical
            // tensor payload before making the Array visible.
            if clear {
                // SAFETY: `buf` was just created with `allocation_bytes` of shared storage and has
                // not been bound to any command buffer.
                unsafe { std::ptr::write_bytes(buf.contents(), 0, buf.length()) };
            }
            Ok(buf)
        };
        if let Some(queue) = program_queue {
            self.pool
                .allocate_program_scratch(allocation_bytes, completed, queue, create)
        } else {
            self.pool.allocate_new(allocation_bytes, completed, create)
        }
    }

    /// Compile `source`'s kernel `name` with the MLX (`Safe`) options, cached.
    pub fn compile(&self, source: &str, name: &str) -> Result<ComputePipeline> {
        self.compile_with(source, name, Math::Safe)
    }

    /// Compile `source`'s kernel `name` with explicit math options, cached by
    /// `(name, math)`.
    pub fn compile_with(&self, source: &str, name: &str, math: Math) -> Result<ComputePipeline> {
        self.compile_full(source, name, math, &[])
    }

    /// Like [`Self::compile_with`], specialising Metal bool function constants
    /// (`[[function_constant(n)]]`), which ternary kernel uses to pick
    /// per-operand indexers.
    pub fn compile_with_constants(
        &self,
        source: &str,
        name: &str,
        math: Math,
        consts: &[(usize, ConstVal)],
    ) -> Result<ComputePipeline> {
        self.compile_full(source, name, math, consts)
    }

    /// Like [`Self::compile_with_constants`], building the source only when
    /// the pipeline is not cached. Dispatch paths use this so a cache hit
    /// costs one hash and one lookup, with no source assembly.
    pub fn compile_lazy(
        &self,
        name: &str,
        math: Math,
        consts: &[(usize, ConstVal)],
        source: impl FnOnce() -> String,
    ) -> Result<ComputePipeline> {
        if let Some(pipeline) = self.cached_pipeline(name, math, consts) {
            return Ok(pipeline);
        }
        self.compile_full(&source(), name, math, consts)
    }

    fn pipeline_hash(&self, name: &str, math: Math, consts: &[(usize, ConstVal)]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (name, math, self.nax, consts).hash(&mut hasher);
        hasher.finish()
    }

    fn cached_pipeline(
        &self,
        name: &str,
        math: Math,
        consts: &[(usize, ConstVal)],
    ) -> Option<ComputePipeline> {
        let hash = self.pipeline_hash(name, math, consts);
        self.pipelines
            .lock()
            .unwrap()
            .get(&hash)
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|(key, _)| key.matches(name, math, self.nax, consts))
                    .map(|(_, pipeline)| pipeline.clone())
            })
    }

    fn compile_full(
        &self,
        source: &str,
        name: &str,
        math: Math,
        consts: &[(usize, ConstVal)],
    ) -> Result<ComputePipeline> {
        // The key encodes every run-time behaviour axis: kernel name, math
        // mode, NAX language target and function constants. Source hashing
        // stays on the compile path below.
        if let Some(pipeline) = self.cached_pipeline(name, math, consts) {
            return Ok(pipeline);
        }
        JIT_COMPILES.fetch_add(1, Ordering::Relaxed);
        crate::trace::event("jit.compile");
        if crate::trace::enabled() {
            eprintln!("[jit] compile {name}");
        }
        // Source lookup is deliberately below the unchanged pipeline fast path.
        // One library can provide several entry points and constant-specialized
        // functions. Serialize library creation independently of pipeline and
        // allocator locks; compilation does not submit or wait for GPU work.
        let lib = self.libraries.lock().unwrap().get_or_try_insert_with(
            source,
            math,
            self.nax,
            || {
                let _span = crate::trace::span("jit.library.compile");
                if crate::trace::enabled() {
                    eprintln!("[jit] library {name}|{math:?}|nax={}", self.nax as u8);
                }
                let opts = compile_options(math, self.nax);
                self.device
                    .newLibraryWithSource_options_error(&ns(source), Some(&opts))
                    .map_err(|e| err(format!("library `{name}`: {}", e.localizedDescription())))
            },
        )?;
        // The retained library remains alive if another compiler thread evicts
        // its cache entry while this function or pipeline is being created.
        let specialize = crate::trace::span("jit.pipeline.specialize");
        let f: Retained<ProtocolObject<dyn MTLFunction>> = if consts.is_empty() {
            lib.newFunctionWithName(&ns(name))
                .ok_or_else(|| err(format!("function `{name}`")))?
        } else {
            let cvs = MTLFunctionConstantValues::new();
            for (i, v) in consts {
                let (ptr, ty) = match v {
                    ConstVal::Bool(b) => (NonNull::from(b).cast(), MTLDataType::Bool),
                    ConstVal::Int(x) => (NonNull::from(x).cast(), MTLDataType::Int),
                    ConstVal::UInt(x) => (NonNull::from(x).cast(), MTLDataType::UInt),
                    ConstVal::Short(x) => (NonNull::from(x).cast(), MTLDataType::Short),
                };
                // SAFETY: `ptr` borrows the `ConstVal` payload in `consts`, which outlives this call;
                // `ty` matches its Rust type (bool/i32/u32/i16) and Metal copies the value.
                unsafe { cvs.setConstantValue_type_atIndex(ptr, ty, *i) };
            }
            lib.newFunctionWithName_constantValues_error(&ns(name), &cvs)
                .map_err(|e| err(format!("function `{name}`: {}", e.localizedDescription())))?
        };
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| err(format!("pipeline `{name}`: {}", e.localizedDescription())))?;
        drop(specialize);
        let p = ComputePipeline::new(state, name);
        let key = PipelineKey {
            name: name.into(),
            math,
            nax: self.nax,
            consts: consts.iter().map(|(i, v)| (*i, *v)).collect(),
        };
        self.pipelines
            .lock()
            .unwrap()
            .entry(self.pipeline_hash(name, math, consts))
            .or_default()
            .push((key, p.clone()));
        // Twin-check on the SLOW path only (one source hash per real compile).
        record_pipeline(name, fnv1a(source.as_bytes()));
        Ok(p)
    }
}

unsafe extern "C" {
    fn getpagesize() -> i32;
}

#[cfg(test)]
mod ordered_gpu_reuse_scope_tests {
    use super::*;

    #[test]
    fn ordered_gpu_reuse_scope_is_nested_and_thread_local() {
        assert!(!ordered_gpu_reuse_enabled());
        let outer = enter_ordered_gpu_reuse_scope();
        assert!(ordered_gpu_reuse_enabled());
        {
            let _inner = enter_ordered_gpu_reuse_scope();
            assert!(ordered_gpu_reuse_enabled());
            assert!(
                !std::thread::spawn(ordered_gpu_reuse_enabled)
                    .join()
                    .unwrap()
            );
        }
        assert!(ordered_gpu_reuse_enabled());
        drop(outer);
        assert!(!ordered_gpu_reuse_enabled());
    }
}

#[cfg(test)]
mod reuse_first_tests {
    use super::*;

    #[test]
    fn soft_cap_preserves_hot_best_fit_leases_and_sweeps_only_on_a_miss() {
        let runtime = MetalRuntime::new(4).unwrap();
        runtime.pool.set_limit(0);
        let held = runtime.buffer(4096, "reuse_first_held").unwrap();
        let medium = runtime.buffer(2048, "reuse_first_medium").unwrap();
        let large = runtime.buffer(4096, "reuse_first_large").unwrap();
        let cold = runtime.buffer(1024, "reuse_first_cold").unwrap();
        let medium_id = medium.id();
        let large_id = large.id();
        drop((medium, large, cold));
        runtime.pool.set_limit(1024);
        let before = runtime.pool.bytes();

        let first = runtime.buffer(1500, "reuse_first_hit").unwrap();
        let second = runtime.buffer(3000, "reuse_first_next_hit").unwrap();
        assert_eq!(
            first.id(),
            medium_id,
            "smallest eligible capacity remains first"
        );
        assert_eq!(second.id(), large_id);
        assert_eq!(
            runtime.pool.bytes(),
            before,
            "hits add no storage or trim cold entries"
        );

        let miss = runtime.buffer(8192, "reuse_first_miss").unwrap();
        assert_eq!(
            runtime.pool.bytes(),
            before - 1024 + 8192,
            "a miss releases the cold entry"
        );
        assert_eq!(held.length(), 4096, "active leases survive the sweep");
        assert_eq!(miss.length(), 8192);

        // Reordering the soft trim does not bypass the separate hard native
        // allocation limit on a miss. All existing leases remain live here.
        runtime.swap_allocation_limit(runtime.allocated_bytes());
        let error = runtime
            .buffer(32 * 1024, "reuse_first_over_budget")
            .err()
            .unwrap();
        assert!(matches!(
            error,
            crate::error::Error::AllocationUnavailable(_)
        ));
        assert_eq!(runtime.pool.bytes(), before - 1024 + 8192);
    }

    #[test]
    fn soft_cap_cannot_reuse_or_clear_pending_cpu_storage_under_resident_admission() {
        // No commands are submitted: explicit watermarks exercise eligibility
        // independently of GPU scheduling or completion timing.
        let runtime = MetalRuntime::new(4).unwrap();
        runtime.pool.set_limit(0);
        let held = runtime.buffer(4096, "reuse_first_resident_held").unwrap();
        let pending = runtime.buffer(2048, "reuse_first_pending").unwrap();
        let pending_id = pending.id();
        // SAFETY: `pending` is a fresh shared buffer never bound to a submitted command buffer.
        unsafe { std::ptr::write_bytes(pending.contents(), 0xa5, pending.length()) };
        pending.mark_used_cb(100);
        drop(pending);
        let _reservation = runtime.reserve_model_memory(1024, 7168).unwrap();
        runtime.pool.set_limit(1024);
        let error = runtime
            .buffer(2048, "reuse_first_pending_rejected")
            .err()
            .unwrap();
        assert!(matches!(
            error,
            crate::error::Error::AllocationUnavailable(_)
        ));
        assert_eq!(runtime.pool.total_bytes(), 7168);
        assert_eq!(held.length(), 4096);
        let completed = runtime.pool.take(2048, 100).unwrap();
        assert_eq!(completed.id(), pending_id);
        assert!(
            // SAFETY: no command was submitted (watermarks are synthetic), so these `length()`
            // bytes are initialised CPU-only storage.
            unsafe { std::slice::from_raw_parts(completed.contents(), completed.length()) }
                .iter()
                .all(|&byte| byte == 0xa5),
            "an ineligible CPU destination must not be cleared"
        );
    }

    #[test]
    fn soft_cap_hits_preserve_exact_capacity_and_private_queue_isolation() {
        let runtime = MetalRuntime::new(4).unwrap();
        runtime.pool.set_limit(0);
        let private = runtime.buffer_for_recorded_scratch(1024).unwrap();
        let private_id = private.id();
        // Simulated unfinished work: only the same queue's private lane may
        // lease this after its former host owner has encoded all accesses.
        private.mark_used_cb(100);
        let general = runtime.buffer(4096, "reuse_first_general").unwrap();
        let general_id = general.id();
        drop((private, general));
        runtime.pool.set_limit(1024);

        let private = runtime.buffer_for_recorded_scratch(1024).unwrap();
        assert_eq!(private.id(), private_id);
        assert_eq!(private.length(), 1024);
        let general = runtime
            .buffer_for_audited_compact_gpu_write(4096, "reuse_first_exact")
            .unwrap();
        assert_eq!(general.id(), general_id);
        assert_eq!(runtime.pool.bytes(), 5120);
        drop(private);
        let queue = runtime.commands.program_identity();
        assert!(
            runtime
                .pool
                .take_program_scratch(1024, queue.wrapping_add(1))
                .is_none()
        );
        assert!(runtime.pool.take_program_scratch(2048, queue).is_none());

        let cpu = runtime
            .buffer_for_compact_full_write(1024, "reuse_first_cpu")
            .unwrap();
        assert_ne!(
            cpu.id(),
            private_id,
            "a CPU import cannot draw from private scratch"
        );
        assert_eq!(cpu.length(), 1024);
        let private = runtime.buffer_for_recorded_scratch(1024).unwrap();
        assert_eq!(private.id(), private_id);
        assert_ne!(private.id(), general.id());
        assert_ne!(private.id(), cpu.id());
    }
}

#[cfg(test)]
mod platform_tests {
    use super::check_macos_version;

    #[test]
    fn rejects_sonoma_with_an_actionable_error() {
        for (major, minor, patch) in [(13, 7, 0), (14, 1, 0), (14, 7, 6)] {
            let error = check_macos_version(major, minor, patch)
                .unwrap_err()
                .to_string();
            assert!(error.contains("requires macOS 15.0 or later"));
            assert!(error.contains(&format!("macOS {major}.{minor}.{patch}")));
            assert!(error.contains("Update macOS"));
        }
    }

    #[test]
    fn accepts_the_metal_compiler_api_boundary_and_newer_versions() {
        for (major, minor, patch) in [(15, 0, 0), (15, 1, 1), (26, 0, 0), (27, 0, 0)] {
            check_macos_version(major, minor, patch).unwrap();
        }
    }
}

#[cfg(test)]
mod retention_tests;

#[cfg(test)]
mod library_cache_tests;

#[cfg(test)]
mod family_tests {
    #[test]
    fn apple_silicon_reports_its_gpu_family() {
        // Every supported Mac has at least an M1-class (Apple7) GPU.
        assert!(super::apple_gpu_family() >= 7);
    }

    #[test]
    fn metal4_follows_the_os() {
        // Every Apple silicon GPU runs Metal 4 from macOS 26.
        assert_eq!(
            super::metal4_supported(),
            objc2::available!(macos = 26.0, ios = 26.0)
        );
    }
}
