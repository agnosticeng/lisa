use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
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
use super::pipeline::ComputePipeline;

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

fn ordered_gpu_reuse_enabled() -> bool {
    ORDERED_GPU_REUSE_DEPTH.with(|depth| depth.get() != 0)
}

const ORDERED_GPU_ZERO_SOURCE: &str = include_str!("../shaders/common/data/ordered_zero.metal");

// ───────────────────────────── runtime ─────────────────────────────

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
/// [`MetalRuntime::compile_full`]). A stable value after model load means no
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
    *LAW.get_or_init(kernel_fingerprint)
}

/// Record a pipeline's source fingerprint — SLOW PATH ONLY (once per compile).
/// A same-name compile with a different body is a warm-cache hazard: warn, and
/// bail under `LISA_STRICT_KERNELS`. Cannot see a source edit that never
/// recompiles; that axis is covered by "kernels compile once at launch" + the
/// goldens (documented limitation, not a silent gap).
fn record_pipeline(name: &str, source_hash: u64) {
    let mut reg = pipeline_hashes().lock().unwrap();
    if let Some(prev) = reg.get(name) {
        if *prev != source_hash {
            if std::env::var_os("LISA_STRICT_KERNELS").is_some() {
                panic!(
                    "[jit] twin-mismatch on `{name}`: body changed under a warm \
                     pipeline cache (LISA_STRICT_KERNELS)"
                );
            }
            eprintln!(
                "[jit] twin-mismatch {name}: src {prev:016x} -> {source_hash:016x} (rerouted)"
            );
        }
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
    pipelines: Mutex<HashMap<String, ComputePipeline>>,
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Math {
    /// Leave Metal's compile options at their platform defaults. ds4 creates
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConstVal {
    Bool(bool),
    Int(i32),
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
            } else if objc2::available!(macos = 27.0) {
                (4 << 16) + 1
            } else if objc2::available!(macos = 26.0) {
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
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            MTLCreateSystemDefaultDevice().ok_or_else(|| err("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| err("newCommandQueue"))?;
        let pool = Arc::new(BufferPool::new());
        // Bound retained native buffers to 512 MiB by default. LISA_POOL_CAP_MB
        // overrides this pool-only budget; zero leaves it uncapped. Resident
        // model reservations and admission are accounted separately.
        let cap_mb = std::env::var("LISA_POOL_CAP_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(512);
        pool.set_limit(cap_mb << 20);
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
            nax,
        })
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

    /// The GPU architecture name (e.g. `"applegpu_g17s"`), used to pick NAX
    /// tiles.
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

    /// Allocate (or reuse) a shared-storage buffer of at least `bytes`.
    pub fn buffer(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        self.buffer_inner(bytes, label, true, false)
    }

    /// Allocate shared storage for a caller that will overwrite every logical
    /// byte before exposing the resulting array. This avoids clearing large
    /// imported weight tensors immediately before copying their full payload.
    pub fn buffer_for_full_write(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        self.buffer_inner(bytes, label, false, false)
    }

    /// Allocate an output that a GPU kernel overwrites completely. Within an
    /// explicit ordered-reuse scope, unleased storage may still have earlier
    /// GPU readers; the new write is encoded after them on our one serial queue.
    /// Never expose the returned storage to a CPU writer before synchronization.
    pub fn buffer_for_full_gpu_write(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        if !ordered_gpu_reuse_enabled() {
            return self.buffer(bytes, label);
        }
        self.buffer_inner(bytes, label, false, true)
    }

    /// Allocate a GPU output with the same zero-initialized contract as
    /// `buffer`. In the ordered scope its clear is a GPU command, so partial
    /// writes and atomic accumulations cannot race an earlier GPU reader.
    pub fn buffer_for_gpu_write(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        if !ordered_gpu_reuse_enabled() {
            return self.buffer(bytes, label);
        }
        let buffer = self.buffer_inner(bytes, label, false, true)?;
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
        if length == 0 || length > self.device.maxBufferLength() {
            return Err(err(format!(
                "external Metal buffer length {length} is invalid"
            )));
        }
        let page_size = unsafe { getpagesize() };
        if page_size <= 0
            || pointer.as_ptr() as usize % page_size as usize != 0
            || length % page_size as usize != 0
        {
            return Err(err("external Metal buffer must be page-aligned"));
        }
        let deallocator = RcBlock::new(move |_pointer: NonNull<c_void>, _length: usize| {
            // Capturing the owner retains the mapping until Metal releases this block.
            let _ = &owner;
        });
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
        debug_assert!(
            !clear || !ordered_gpu_write,
            "in-flight GPU storage must not be cleared on the CPU"
        );
        let allocation_bytes = bytes
            .max(1)
            .checked_add(1023)
            .map(|rounded| rounded / 1024 * 1024)
            .ok_or_else(|| err("Metal allocation size overflow"))?;
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        {
            let mut c = LABEL_COUNTS.lock().unwrap();
            match c.iter_mut().find(|(l, _)| l == label) {
                Some(e) => e.1 += 1,
                None => c.push((label.to_string(), 1)),
            }
        }
        let completed = self.commands.completed_watermark();
        if self.pool.reserved_model_bytes() > 0 {
            self.pool.sweep_if_over_completed(completed);
        } else {
            self.pool.sweep_if_over();
        }
        let reusable = if ordered_gpu_write {
            self.pool.take_for_ordered_gpu_write(bytes)
        } else {
            self.pool.take(bytes, completed)
        };
        if let Some(b) = reusable {
            POOLHIT_COUNT.fetch_add(1, Ordering::Relaxed);
            b.set_label(label);
            if clear {
                // `take` only returns buffers whose last command buffer has
                // completed, so clearing shared storage here cannot race a GPU
                // reader. Fresh buffers receive the same initial contents below.
                unsafe { std::ptr::write_bytes(b.contents(), 0, b.length()) };
            }
            return Ok(b);
        }
        self.pool.allocate_new(allocation_bytes, completed, || {
            let raw = self
                .device
                .newBufferWithLength_options(
                    allocation_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| err(format!("newBuffer({bytes})")))?;
            let buf = Arc::new(Buffer::wrap(raw));
            buf.set_label(label);
            // General-purpose buffers start at zero so a kernel that reads a region
            // its producer never wrote cannot observe uninitialized storage. Full
            // host-write imports skip this; their caller overwrites the logical
            // tensor payload before making the Array visible.
            if clear {
                unsafe { std::ptr::write_bytes(buf.contents(), 0, buf.length()) };
            }
            Ok(buf)
        })
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

    /// Like [`compile_with`], specialising Metal bool function constants
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

    fn compile_full(
        &self,
        source: &str,
        name: &str,
        math: Math,
        consts: &[(usize, ConstVal)],
    ) -> Result<ComputePipeline> {
        // O2: the key must encode every RUN-time behaviour axis, not only the
        // caller-built name. The NAX language version (baked by
        // `compile_options`) changes the arithmetic and was absent from the key
        // entirely — on an M5 a `nax=true` pipeline could collide with a
        // `nax=false` one. CHEAP FIELDS ONLY: no hashing here (this function is
        // the per-dispatch cache lookup).
        let key = format!(
            "{name}|{math:?}|nax={}|{}",
            self.nax as u8,
            consts
                .iter()
                .map(|(i, v)| format!("{i}{v:?}"))
                .collect::<String>()
        );
        if let Some(p) = self.pipelines.lock().unwrap().get(&key) {
            return Ok(p.clone());
        }
        let opts = compile_options(math, self.nax);
        JIT_COMPILES.fetch_add(1, Ordering::Relaxed);
        crate::trace::event("jit.compile");
        if crate::trace::enabled() {
            eprintln!("[jit] compile {name}");
        }
        let lib = self
            .device
            .newLibraryWithSource_options_error(&ns(source), Some(&opts))
            .map_err(|e| err(format!("library `{name}`: {}", e.localizedDescription())))?;
        let f: Retained<ProtocolObject<dyn MTLFunction>> = if consts.is_empty() {
            lib.newFunctionWithName(&ns(name))
                .ok_or_else(|| err(format!("function `{name}`")))?
        } else {
            let cvs = MTLFunctionConstantValues::new();
            for (i, v) in consts {
                let (ptr, ty) = match v {
                    ConstVal::Bool(b) => (NonNull::from(b).cast(), MTLDataType::Bool),
                    ConstVal::Int(x) => (NonNull::from(x).cast(), MTLDataType::Int),
                };
                unsafe { cvs.setConstantValue_type_atIndex(ptr, ty, *i) };
            }
            lib.newFunctionWithName_constantValues_error(&ns(name), &cvs)
                .map_err(|e| err(format!("function `{name}`: {}", e.localizedDescription())))?
        };
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| err(format!("pipeline `{name}`: {}", e.localizedDescription())))?;
        let p = ComputePipeline {
            raw: state,
            name: name.to_string(),
            label: std::cell::OnceCell::new(),
        };
        self.pipelines.lock().unwrap().insert(key, p.clone());
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
