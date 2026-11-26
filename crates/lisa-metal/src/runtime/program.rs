//! Replay of an audited, fixed-geometry sequence of existing Metal dispatches.
//!
//! This records compiled pipelines, bindings, constants and launch geometry;
//! it does not compile different shaders or change their arithmetic. Each
//! invocation gets exclusive writable leases. Read-only external buffers are
//! retained, and dynamic inputs are rebound without writing to them. Encoder
//! acquisitions and explicit encoder boundaries retain their original order.
//! Each audited encoder section binds its own pipeline and every used operand;
//! another invocation may encode work between separately locked sections.
//!
//! The design was informed by MTPLX's Apache-2.0 native executor (revision
//! bebe21b5ba36af3688335ab66d365509fd042fad). This implementation uses LISA's
//! allocator and completion bookkeeping. Ordered replay keeps the serial
//! encoder; explicitly admitted dependency-parallel replay uses fenced
//! concurrent encoders with precomputed physical-buffer hazards.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2_metal::MTLDevice;

use super::{Buffer, ComputePipeline, MetalRuntime, Result};

mod concurrent;
use concurrent::ParallelPlan;
#[cfg(test)]
mod parallel_tests;

static REPLAYS: AtomicU64 = AtomicU64::new(0);
static PARALLEL_REPLAYS: AtomicU64 = AtomicU64::new(0);

/// Successful program encodings, excluding capture and declined replay. This
/// monotonic diagnostic lets numerical tests reject a vacuous fallback path.
pub fn dispatch_program_replay_count() -> u64 {
    REPLAYS.load(Ordering::Relaxed)
}

/// Successful dependency-parallel program encodings, excluding serial capture
/// and declined replay. Diagnostics use this to reject unexercised schedules.
pub fn dispatch_program_parallel_replay_count() -> u64 {
    PARALLEL_REPLAYS.load(Ordering::Relaxed)
}

/// A mutable binding slot, whose contents are read but never written by a
/// program. The caller must include every input whose contents can change.
#[derive(Clone)]
pub struct ProgramInput {
    pub buffer: Arc<Buffer>,
    /// Checked end byte of the fixed-layout view, including its offset and
    /// strides. Unused capacity of a best-fit pool allocation is not required.
    pub required_bytes: usize,
}

/// Output storage and its byte offset. Shape, dtype and strides remain the
/// caller's responsibility; replay preserves this exact buffer-relative view.
#[derive(Clone)]
pub struct ProgramOutput {
    pub buffer: Arc<Buffer>,
    pub offset: usize,
}

#[derive(Clone)]
enum RecordedCommand {
    Pipeline(ComputePipeline),
    Buffer {
        index: usize,
        buffer: Option<Arc<Buffer>>,
        offset: usize,
        write: bool,
    },
    Bytes {
        index: usize,
        data: Vec<MaybeUninit<u8>>,
    },
    Threadgroup {
        index: usize,
        length: usize,
    },
    Dispatch {
        groups: bool,
        grid: Size,
        group: Size,
    },
}

type Size = (usize, usize, usize);

enum RecordedSection {
    Allocate(usize),
    Encoder(Vec<RecordedCommand>),
    Boundary,
}

struct Recording {
    identity: usize,
    sections: Vec<RecordedSection>,
    open: bool,
    declined: bool,
    allocations: HashMap<usize, Allocation>,
    ordered_reuse: bool,
}

impl Recording {
    fn decline(&mut self) {
        self.declined = true;
        // An unsupported operation can be a budget-triggered wait. Release
        // capture-only leases promptly so the ordinary path can reclaim them.
        self.sections.clear();
        self.allocations.clear();
    }
}

#[derive(Clone, Copy)]
pub(super) enum AllocationPolicy {
    Zeroed,
    FullGpuWrite,
    CompactFullGpuWrite,
}

struct Allocation {
    // Retaining the capture allocation prevents pool reuse from conflating
    // distinct lifetimes into one recorded resource identity.
    _buffer: Arc<Buffer>,
    bytes: usize,
    policy: AllocationPolicy,
}

thread_local! {
    static RECORDING: RefCell<Option<Recording>> = const { RefCell::new(None) };
}

fn with_recording(identity: usize, f: impl FnOnce(&mut Recording)) {
    RECORDING.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(recording) = slot.as_mut()
            && recording.identity == identity
            && !recording.declined
        {
            f(recording);
        }
    });
}

pub(super) fn is_capturing(identity: usize) -> bool {
    RECORDING.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|recording| recording.identity == identity && !recording.declined)
    })
}

fn record(identity: usize, command: impl FnOnce() -> RecordedCommand) {
    with_recording(identity, |recording| {
        if !recording.open {
            recording.decline();
            return;
        }
        match recording.sections.last_mut() {
            Some(RecordedSection::Encoder(commands)) => commands.push(command()),
            _ => recording.decline(),
        }
    });
}

pub(super) fn begin_encoder(identity: usize) {
    with_recording(identity, |recording| {
        if recording.open {
            recording.decline();
        } else {
            recording.open = true;
            recording
                .sections
                .push(RecordedSection::Encoder(Vec::new()));
        }
    });
}

pub(super) fn end_encoder(identity: usize) {
    with_recording(identity, |recording| recording.open = false);
}

pub(super) fn encoder_boundary(identity: usize) {
    with_recording(identity, |recording| {
        if recording.open {
            recording.decline();
        } else {
            recording.sections.push(RecordedSection::Boundary);
        }
    });
}

pub(super) fn unsupported(identity: usize) {
    with_recording(identity, Recording::decline);
}

pub(super) fn allocation(
    identity: usize,
    buffer: &Arc<Buffer>,
    bytes: usize,
    policy: AllocationPolicy,
) {
    with_recording(identity, |recording| {
        if recording.open || recording.ordered_reuse != super::device::ordered_gpu_reuse_enabled() {
            // Allocations can wait for a memory budget. Replaying one while
            // an EncoderGuard owns Commands' mutex would deadlock that wait.
            recording.decline();
            return;
        }
        let id = resource_id(buffer);
        if recording
            .allocations
            .insert(
                id,
                Allocation {
                    _buffer: buffer.clone(),
                    bytes,
                    policy,
                },
            )
            .is_some()
        {
            recording.decline();
        } else {
            recording.sections.push(RecordedSection::Allocate(id));
        }
    });
}

pub(super) fn pipeline(identity: usize, pipeline: &ComputePipeline) {
    record(identity, || RecordedCommand::Pipeline(pipeline.clone()));
}

pub(super) fn buffer(
    identity: usize,
    index: usize,
    buffer: Option<&Arc<Buffer>>,
    offset: usize,
    write: bool,
) {
    record(identity, || RecordedCommand::Buffer {
        index,
        buffer: buffer.cloned(),
        offset,
        write,
    });
}

/// The encoder's setBytes contract already requires a readable byte range.
pub(super) fn bytes(identity: usize, index: usize, length: usize, data: *const u8) {
    record(identity, || RecordedCommand::Bytes {
        index,
        data: if length == 0 {
            Vec::new()
        } else {
            // Preserve padding without interpreting it as initialized Rust
            // bytes. Generic setBytes structs may contain padding that Metal
            // copies but the shader never reads.
            let mut copied = vec![MaybeUninit::uninit(); length];
            // SAFETY: identical range to the immediately following Metal copy;
            // MaybeUninit permits copying any padding in that range.
            unsafe { std::ptr::copy_nonoverlapping(data.cast(), copied.as_mut_ptr(), length) };
            copied
        },
    });
}

pub(super) fn threadgroup(identity: usize, index: usize, length: usize) {
    record(identity, || RecordedCommand::Threadgroup { index, length });
}

pub(super) fn dispatch(identity: usize, groups: bool, grid: Size, group: Size) {
    record(identity, || RecordedCommand::Dispatch {
        groups,
        grid,
        group,
    });
}

// Buffer::id identifies one wrapper. Several wrappers can share the same Metal
// allocation, so alias checks must use the underlying resource identity.
fn resource_id(buffer: &Buffer) -> usize {
    std::ptr::from_ref(buffer.raw_buffer()).cast::<()>() as usize
}

enum Storage {
    Input {
        index: usize,
        minimum_length: usize,
    },
    Pinned(Arc<Buffer>),
    Fresh {
        bytes: usize,
        policy: AllocationPolicy,
    },
}

enum Command {
    Pipeline(ComputePipeline),
    Buffer {
        index: usize,
        slot: Option<usize>,
        offset: usize,
        write: bool,
    },
    Bytes {
        index: usize,
        data: Vec<MaybeUninit<u8>>,
    },
    Threadgroup {
        index: usize,
        length: usize,
    },
    Dispatch {
        groups: bool,
        grid: Size,
        group: Size,
    },
}

enum Section {
    Allocate(usize),
    Encoder(Vec<Command>),
    Boundary,
}

/// Closed dispatch intervals: a read and a write in the same dispatch may not
/// alias, even when the shader's individual accesses could be ordered safely.
#[derive(Clone, Copy, Debug)]
struct ScratchLifetime {
    slot: usize,
    capacity: usize,
    first: usize,
    last: usize,
}

/// Private scratch ranges, shared only by nonoverlapping dispatch lifetimes.
/// One KiB preserves the alignment of admitted GPU operands (including native
/// affine vector accesses). Escaping aliases retain standalone allocations.
struct ScratchArena {
    bytes: usize,
    unaliased_bytes: usize,
    first_slot: usize,
    offsets: Vec<Option<usize>>,
}

impl ScratchArena {
    fn pack(
        mut lifetimes: Vec<ScratchLifetime>,
        slots: usize,
        max_bytes: usize,
    ) -> Option<(usize, Vec<Option<usize>>)> {
        lifetimes.sort_by_key(|lifetime| (lifetime.first, lifetime.slot));
        let mut active = Vec::<(ScratchLifetime, usize)>::new();
        let mut free = Vec::<(usize, usize)>::new();
        let mut offsets = vec![None; slots];
        let mut bytes = 0;
        for lifetime in lifetimes {
            if lifetime.capacity == 0
                || lifetime.capacity % 1024 != 0
                || lifetime.first > lifetime.last
                || lifetime.slot >= slots
                || offsets[lifetime.slot].is_some()
            {
                return None;
            }
            active.retain(|(prior, offset)| {
                if prior.last < lifetime.first {
                    free.push((*offset, prior.capacity));
                    false
                } else {
                    true
                }
            });
            // Coalesce adjacent released ranges before deterministic best fit.
            // Only planning uses these vectors; replay has no lifetime scan.
            free.sort_unstable_by_key(|&(offset, _)| offset);
            let mut merged = Vec::<(usize, usize)>::with_capacity(free.len());
            for (offset, capacity) in free.drain(..) {
                if let Some((prior, size)) = merged.last_mut()
                    && prior.checked_add(*size)? == offset
                {
                    *size = size.checked_add(capacity)?;
                    continue;
                }
                merged.push((offset, capacity));
            }
            free = merged;
            let candidate = free
                .iter()
                .enumerate()
                .filter(|(_, (_, capacity))| *capacity >= lifetime.capacity)
                .min_by_key(|(_, (offset, capacity))| (*capacity, *offset))
                .map(|(index, _)| index);
            let offset = if let Some(index) = candidate {
                let (offset, capacity) = free.remove(index);
                if capacity > lifetime.capacity {
                    free.push((
                        offset.checked_add(lifetime.capacity)?,
                        capacity - lifetime.capacity,
                    ));
                }
                offset
            } else {
                // Extend a released tail instead of leaving an unusable hole
                // below a new larger allocation.
                let tail = free
                    .iter()
                    .position(|(offset, capacity)| offset.checked_add(*capacity) == Some(bytes));
                let offset = tail.map_or(bytes, |index| free.remove(index).0);
                bytes = offset.checked_add(lifetime.capacity)?;
                if bytes > max_bytes {
                    return None;
                }
                offset
            };
            offsets[lifetime.slot] = Some(offset);
            active.push((lifetime, offset));
        }
        Some((bytes, offsets))
    }

    fn plan(
        storage: &[Storage],
        sections: &[Section],
        outputs: &[(usize, usize)],
        max_bytes: usize,
    ) -> Option<Self> {
        let mut eligible = storage
            .iter()
            .map(|storage| {
                matches!(
                    storage,
                    Storage::Fresh {
                        bytes: 1..,
                        policy: AllocationPolicy::FullGpuWrite,
                    }
                )
            })
            .collect::<Vec<_>>();
        // Any escaping view excludes its whole allocation, including aliases
        // returned at different offsets or under multiple output names.
        for &(slot, _) in outputs {
            eligible[slot] = false;
        }
        let mut intervals = vec![None::<(usize, usize)>; storage.len()];
        let mut bound = HashMap::<usize, (usize, bool)>::new();
        let mut dispatch = 0usize;
        for section in sections {
            match section {
                Section::Boundary => bound.clear(),
                Section::Allocate(_) => {}
                Section::Encoder(commands) => {
                    for command in commands {
                        match command {
                            Command::Buffer {
                                index,
                                slot,
                                offset,
                                write,
                            } => {
                                if let Some(slot) = slot {
                                    let storage = storage.get(*slot)?;
                                    if let Storage::Fresh { bytes, .. } = storage {
                                        // Captured excess capacity is not part
                                        // of the audited logical allocation.
                                        if *offset >= *bytes {
                                            eligible[*slot] = false;
                                        }
                                    }
                                    bound.insert(*index, (*slot, *write));
                                } else {
                                    bound.remove(index);
                                }
                            }
                            Command::Bytes { index, .. } => {
                                bound.remove(index);
                            }
                            Command::Dispatch { .. } => {
                                // Pipeline switches and EncoderGuard drops do
                                // not reset Metal bindings. Count inherited
                                // bindings conservatively, including every
                                // alias and output; no pipeline reflection or
                                // unproved unused-argument assumption is used.
                                for &(slot, write) in bound.values() {
                                    if !write && intervals[slot].is_none() {
                                        eligible[slot] = false;
                                    }
                                }
                                for &(slot, _) in bound.values() {
                                    intervals[slot] = Some((
                                        intervals[slot].map_or(dispatch, |(first, _)| first),
                                        dispatch,
                                    ));
                                }
                                dispatch = dispatch.checked_add(1)?;
                            }
                            Command::Pipeline(_) | Command::Threadgroup { .. } => {}
                        }
                    }
                }
            }
        }
        let mut lifetimes = Vec::new();
        let mut unaliased_bytes = 0usize;
        let mut first_slot = None;
        for section in sections {
            if let Section::Allocate(slot) = section {
                let Some((first, last)) = intervals[*slot] else {
                    continue;
                };
                if !eligible[*slot] {
                    continue;
                }
                let Storage::Fresh {
                    bytes: requested, ..
                } = storage[*slot]
                else {
                    unreachable!("only fresh full-write scratch is eligible");
                };
                let capacity = requested.checked_add(1023)? / 1024 * 1024;
                unaliased_bytes = unaliased_bytes.checked_add(capacity)?;
                first_slot.get_or_insert(*slot);
                lifetimes.push(ScratchLifetime {
                    slot: *slot,
                    capacity,
                    first,
                    last,
                });
            }
        }
        if lifetimes.len() < 2 {
            return None;
        }
        let (bytes, offsets) = Self::pack(lifetimes, storage.len(), max_bytes)?;
        if bytes > unaliased_bytes {
            return None;
        }
        Some(Self {
            bytes,
            unaliased_bytes,
            first_slot: first_slot?,
            offsets,
        })
    }
}

/// Immutable replay descriptors for one audited fixed-geometry GPU region.
/// Writable leases are never shared by concurrent calls or captured outputs.
/// Private scratch may be reused after all earlier accesses have been encoded
/// on this runtime's serial queue; escaping outputs still require completion.
pub struct DispatchProgram {
    identity: usize,
    input_aliases: Vec<usize>,
    input_requirements: Vec<usize>,
    storage: Vec<Storage>,
    sections: Vec<Section>,
    outputs: Vec<(usize, usize)>,
    dispatches: usize,
    fresh_bytes: usize,
    ordered_reuse: bool,
    scratch: Option<ScratchArena>,
    parallel: Option<ParallelPlan>,
}

struct CaptureGuard;

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        RECORDING.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Execute normally once and, when representable, seal its dispatch program.
/// A declined capture still returns the ordinary closure result unchanged.
/// Nested captures are declined without disturbing the outer recording.
///
/// # Capture contract
///
/// The closure must be an audited GPU-only region on the calling thread: every changing external
/// value belongs in `inputs`; constants copied with setBytes must be stable
/// for the caller's cache key. Host reads/writes, indirect pointer tables and
/// CPU-initialized writable scratch are unsupported. Input shape, dtype,
/// strides and byte offsets must match on replay. Only known operation
/// wrappers with these properties may be admitted by a higher-level planner.
/// Higher-level mapped-storage leases must remain owned by that planner.
/// Internal full-write storage must be relocatable at 1 KiB-aligned offsets:
/// kernels cannot inspect allocation capacity or encode absolute scratch
/// addresses in inline arguments. Every read stays within the requested
/// logical allocation; unused bucket padding is unspecified.
///
/// Each `EncoderGuard` section must select its pipeline and bind every buffer,
/// inline argument and threadgroup-memory range used by its dispatches before
/// use. Inheritance within one guard is supported, but a section must not rely
/// on bindings from an earlier guard: another caller can acquire the same
/// encoder between them. Sealing does not reflect shader arguments to verify
/// this prerequisite. Scratch planning conservatively counts inherited buffer
/// bindings for lifetime safety; that does not admit partial-rebinding wrappers.
pub fn capture_dispatch_program<T, E>(
    runtime: &Arc<MetalRuntime>,
    inputs: &[ProgramInput],
    encode: impl FnOnce() -> std::result::Result<(T, Vec<ProgramOutput>), E>,
) -> std::result::Result<(T, Option<DispatchProgram>), E> {
    let identity = runtime.commands.program_identity();
    let started = RECORDING.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return false;
        }
        *slot = Some(Recording {
            identity,
            sections: Vec::new(),
            open: false,
            declined: false,
            allocations: HashMap::new(),
            ordered_reuse: super::device::ordered_gpu_reuse_enabled(),
        });
        true
    });
    if !started {
        let (result, _) = encode()?;
        return Ok((result, None));
    }
    let guard = CaptureGuard;
    let (result, outputs) = encode()?;
    let recording = RECORDING.with(|slot| slot.borrow_mut().take()).unwrap();
    drop(guard);
    let program = DispatchProgram::seal(
        recording,
        inputs,
        &outputs,
        runtime.device().maxBufferLength(),
    );
    Ok((result, program))
}

impl DispatchProgram {
    fn seal(
        recording: Recording,
        inputs: &[ProgramInput],
        outputs: &[ProgramOutput],
        max_buffer_bytes: usize,
    ) -> Option<Self> {
        if recording.declined || recording.open {
            return None;
        }
        let mut external = HashMap::new();
        let mut input_aliases = Vec::with_capacity(inputs.len());
        let mut group_requirements = vec![0; inputs.len()];
        for (index, input) in inputs.iter().enumerate() {
            if input.required_bytes > input.buffer.length() {
                return None;
            }
            let first = *external.entry(resource_id(&input.buffer)).or_insert(index);
            input_aliases.push(first);
            group_requirements[first] = group_requirements[first].max(input.required_bytes);
        }
        let mut buffers: HashMap<usize, Arc<Buffer>> = HashMap::new();
        let mut resource_order = Vec::new();
        let mut remember = |buffer: &Arc<Buffer>| {
            let id = resource_id(buffer);
            if let std::collections::hash_map::Entry::Vacant(entry) = buffers.entry(id) {
                entry.insert(buffer.clone());
                resource_order.push(id);
            }
        };
        let mut writes = HashSet::new();
        for section in &recording.sections {
            match section {
                RecordedSection::Allocate(id) => {
                    remember(&recording.allocations.get(id)?._buffer);
                }
                RecordedSection::Encoder(commands) => {
                    for command in commands {
                        if let RecordedCommand::Buffer {
                            buffer: Some(buffer),
                            offset,
                            write,
                            ..
                        } = command
                        {
                            if *offset > buffer.length() {
                                return None;
                            }
                            remember(buffer);
                            if *write {
                                writes.insert(resource_id(buffer));
                            }
                        }
                    }
                }
                RecordedSection::Boundary => {}
            }
        }
        for output in outputs {
            if output.offset > output.buffer.length() {
                return None;
            }
            remember(&output.buffer);
        }
        drop(remember);
        if writes.iter().any(|id| external.contains_key(id)) {
            return None;
        }

        let mut slots = HashMap::new();
        let mut storage = Vec::with_capacity(buffers.len());
        let mut fresh_bytes = 0usize;
        // First occurrence follows the capture's ordered allocation/command
        // sections. HashMap iteration must not choose allocator request order:
        // a small request could otherwise consume a later large request's only
        // reusable bucket and unnecessarily grow resident storage.
        for id in resource_order {
            let buffer = buffers.remove(&id)?;
            slots.insert(id, storage.len());
            storage.push(if let Some(&index) = external.get(&id) {
                Storage::Input {
                    index,
                    minimum_length: group_requirements[index],
                }
            } else if let Some(allocation) = recording.allocations.get(&id) {
                fresh_bytes = fresh_bytes
                    .checked_add(allocation.bytes.max(1).checked_add(1023)? / 1024 * 1024)?;
                Storage::Fresh {
                    bytes: allocation.bytes,
                    policy: allocation.policy,
                }
            } else if writes.contains(&id) {
                // In-place external state and CPU-initialized scratch must
                // not silently become newly allocated zero-filled storage.
                return None;
            } else {
                Storage::Pinned(buffer)
            });
        }

        let mut sections = Vec::with_capacity(recording.sections.len());
        let mut bound: HashMap<usize, (usize, bool)> = HashMap::new();
        let mut initialized = HashSet::new();
        let mut allocated = HashSet::new();
        let mut has_pipeline = false;
        let mut dispatches = 0;
        for section in recording.sections {
            match section {
                RecordedSection::Allocate(id) => {
                    let slot = slots[&id];
                    if !matches!(storage[slot], Storage::Fresh { .. }) || !allocated.insert(slot) {
                        return None;
                    }
                    sections.push(Section::Allocate(slot));
                }
                RecordedSection::Boundary => {
                    bound.clear();
                    has_pipeline = false;
                    sections.push(Section::Boundary);
                }
                RecordedSection::Encoder(recorded) => {
                    let mut commands = Vec::with_capacity(recorded.len());
                    for command in recorded {
                        commands.push(match command {
                            RecordedCommand::Pipeline(pipeline) => {
                                has_pipeline = true;
                                Command::Pipeline(pipeline)
                            }
                            RecordedCommand::Buffer {
                                index,
                                buffer,
                                offset,
                                write,
                            } => {
                                let slot =
                                    buffer.as_ref().map(|buffer| slots[&resource_id(buffer)]);
                                if let Some(slot) = slot {
                                    if matches!(storage[slot], Storage::Fresh { .. })
                                        && !allocated.contains(&slot)
                                    {
                                        return None;
                                    }
                                    bound.insert(index, (slot, write));
                                } else {
                                    bound.remove(&index);
                                }
                                Command::Buffer {
                                    index,
                                    slot,
                                    offset,
                                    write,
                                }
                            }
                            RecordedCommand::Bytes { index, data } => {
                                bound.remove(&index);
                                Command::Bytes { index, data }
                            }
                            RecordedCommand::Threadgroup { index, length } => {
                                Command::Threadgroup { index, length }
                            }
                            RecordedCommand::Dispatch {
                                groups,
                                grid,
                                group,
                            } => {
                                if !has_pipeline {
                                    return None;
                                }
                                // Inspect all reads before accepting this dispatch's
                                // writes, including aliases in another argument slot.
                                if bound.values().any(|(slot, write)| {
                                    !write
                                        && matches!(
                                            storage[*slot],
                                            Storage::Fresh {
                                                policy: AllocationPolicy::FullGpuWrite
                                                    | AllocationPolicy::CompactFullGpuWrite,
                                                ..
                                            }
                                        )
                                        && !initialized.contains(slot)
                                }) {
                                    return None;
                                }
                                initialized.extend(
                                    bound
                                        .values()
                                        .filter_map(|(slot, write)| write.then_some(*slot)),
                                );
                                dispatches += 1;
                                Command::Dispatch {
                                    groups,
                                    grid,
                                    group,
                                }
                            }
                        });
                    }
                    sections.push(Section::Encoder(commands));
                }
            }
        }
        if dispatches == 0 {
            return None;
        }
        for output in outputs {
            let slot = slots[&resource_id(&output.buffer)];
            if matches!(storage[slot], Storage::Fresh { .. }) && !allocated.contains(&slot) {
                return None;
            }
            if matches!(
                storage[slot],
                Storage::Fresh {
                    policy: AllocationPolicy::FullGpuWrite | AllocationPolicy::CompactFullGpuWrite,
                    ..
                }
            ) && !initialized.contains(&slot)
            {
                return None;
            }
        }
        let outputs = outputs
            .iter()
            .map(|output| (slots[&resource_id(&output.buffer)], output.offset))
            .collect::<Vec<_>>();
        // This optimization never enables or broadens the GLM reuse scope.
        // Replay may use a private same-queue scratch lease; capture and all
        // escaping storage retain their ordinary allocation identities.
        let scratch = (!recording.ordered_reuse)
            .then(|| ScratchArena::plan(&storage, &sections, &outputs, max_buffer_bytes))
            .flatten();
        Some(Self {
            identity: recording.identity,
            input_aliases,
            input_requirements: inputs.iter().map(|input| input.required_bytes).collect(),
            storage,
            sections,
            outputs,
            dispatches,
            fresh_bytes,
            ordered_reuse: recording.ordered_reuse,
            scratch,
            parallel: None,
        })
    }

    /// Explicit backend scheduling preference; no model or shape is inferred.
    /// The original capture remains serial. Scoped ordered reuse and probe
    /// mode are not admitted to this independent dispatch schedule.
    pub fn enable_parallel(&mut self) -> bool {
        if self.ordered_reuse || super::env::gpu_probe_enabled() {
            return false;
        }
        self.parallel = ParallelPlan::plan(&self.storage, &self.sections, self.scratch.as_ref());
        self.parallel.is_some()
    }

    pub fn disable_parallel(&mut self) {
        self.parallel = None;
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatches
    }

    /// Sum of ordinary allocator bucket bytes before optional scratch packing.
    /// Trace events report the smaller arena actually acquired on replay.
    /// Independent best-fit allocations may retain larger capacities.
    pub fn fresh_bytes(&self) -> usize {
        self.fresh_bytes
    }

    /// Rebind fixed-layout inputs and enqueue the same dispatch sequence.
    /// A runtime, alias or capacity mismatch returns `None` before encoding.
    /// Sections can interleave with other callers; the capture contract requires
    /// a self-contained pipeline and used-operand binding set in each section.
    pub fn replay(
        &self,
        runtime: &Arc<MetalRuntime>,
        inputs: &[ProgramInput],
    ) -> Result<Option<Vec<ProgramOutput>>> {
        if runtime.commands.program_identity() != self.identity
            || inputs.len() != self.input_aliases.len()
            || super::device::ordered_gpu_reuse_enabled() != self.ordered_reuse
        {
            return Ok(None);
        }
        let mut aliases = HashMap::new();
        for (index, input) in inputs.iter().enumerate() {
            if input.required_bytes != self.input_requirements[index]
                || input.buffer.length() < input.required_bytes
            {
                return Ok(None);
            }
            if *aliases.entry(resource_id(&input.buffer)).or_insert(index)
                != self.input_aliases[index]
            {
                return Ok(None);
            }
        }
        for storage in &self.storage {
            match storage {
                Storage::Input {
                    index,
                    minimum_length,
                } if inputs[*index].buffer.length() < *minimum_length => return Ok(None),
                // A formerly independent pinned value cannot become aliased
                // to a dynamic input, even if every current binding is read-only.
                Storage::Pinned(buffer) if aliases.contains_key(&resource_id(buffer)) => {
                    return Ok(None);
                }
                _ => {}
            }
        }
        let mut buffers = Vec::with_capacity(self.storage.len());
        for storage in &self.storage {
            buffers.push(match storage {
                Storage::Input { index, .. } => Some((inputs[*index].buffer.clone(), 0)),
                Storage::Pinned(buffer) => Some((buffer.clone(), 0)),
                Storage::Fresh { .. } => None,
            });
        }
        let mut arena = None;
        let mut scratch = self.scratch.as_ref();
        // Declare this after every writable lease container: on all exits its
        // Drop closes/fences the last owned encoder BEFORE scratch is released.
        // Nested capture executes serially and records its normal operations.
        let parallel = self
            .parallel
            .as_ref()
            .filter(|_| !runtime.commands.is_capturing_program());
        let parallel_scope = parallel.map(|_| runtime.commands.recorded_encoder_scope());
        let mut dispatch_index = 0usize;
        for section in &self.sections {
            match section {
                Section::Allocate(slot) => {
                    // Each section begins after the preceding EncoderGuard
                    // dropped. Acquire scratch at its first allocation marker,
                    // never while holding Commands' mutex. All original fresh
                    // leases already lived until replay returned. The arena
                    // reserves a fixed layout; only allocations whose closed
                    // dispatch lifetimes do not overlap share a byte range.
                    if let Some(plan) = scratch
                        && *slot == plan.first_slot
                    {
                        // Exact capacity cannot consume a larger cached
                        // bucket that a later escaping output needs under
                        // the same peak-memory budget.
                        match runtime.buffer_for_recorded_scratch(plan.bytes) {
                            Ok(buffer) => {
                                crate::trace::event("program.scratch.packed");
                                crate::trace::record_elapsed(
                                    "program.scratch.bytes",
                                    plan.bytes as u64,
                                    std::time::Duration::ZERO,
                                );
                                crate::trace::record_elapsed(
                                    "program.scratch.saved_bytes",
                                    (plan.unaliased_bytes - plan.bytes) as u64,
                                    std::time::Duration::ZERO,
                                );
                                arena = Some(buffer);
                            }
                            // Separate cached buckets can fit when one
                            // contiguous allocation cannot. Nothing used
                            // arena storage yet, so keep the original path.
                            Err(crate::error::Error::AllocationUnavailable(_)) => {
                                crate::trace::event("program.scratch.fallback");
                                scratch = None;
                            }
                            Err(error) => return Err(error),
                        }
                    }
                    if let Some(base) = scratch.and_then(|plan| plan.offsets[*slot]) {
                        debug_assert!(buffers[*slot].is_none());
                        buffers[*slot] = Some((
                            arena.as_ref().expect("first scratch allocation").clone(),
                            base,
                        ));
                        continue;
                    }
                    let Storage::Fresh { bytes, policy } = &self.storage[*slot] else {
                        unreachable!("sealed allocation slot");
                    };
                    debug_assert!(buffers[*slot].is_none());
                    buffers[*slot] = Some((
                        match policy {
                            AllocationPolicy::Zeroed => {
                                runtime.buffer(*bytes, "dispatch_program")?
                            }
                            AllocationPolicy::FullGpuWrite => runtime
                                .buffer_for_audited_full_gpu_write(*bytes, "dispatch_program")?,
                            AllocationPolicy::CompactFullGpuWrite => runtime
                                .buffer_for_audited_compact_gpu_write(*bytes, "dispatch_program")?,
                        },
                        0,
                    ));
                }
                Section::Boundary => runtime.commands.flush_encoder(),
                Section::Encoder(commands) => {
                    let guard = if let Some(scope) = &parallel_scope {
                        scope.encoder()?
                    } else {
                        runtime.commands.encoder()?
                    };
                    let encoder = guard.encoder();
                    for command in commands {
                        match command {
                            Command::Pipeline(pipeline) => encoder.set_pipeline(pipeline),
                            Command::Buffer {
                                index,
                                slot,
                                offset,
                                write,
                            } => {
                                let binding = slot.map(|slot| {
                                    buffers[slot].as_ref().expect("sealed binding allocation")
                                });
                                // Planning checked each range, offset and
                                // closed lifetime against concurrent aliases.
                                let offset = *offset + binding.map_or(0, |(_, base)| *base);
                                let buffer = binding.map(|(buffer, _)| buffer);
                                if *write {
                                    encoder.set_output(*index, buffer, offset);
                                } else {
                                    encoder.set_input(*index, buffer, offset);
                                }
                            }
                            Command::Bytes { index, data } => {
                                encoder.set_bytes_directly(*index, data.len(), data.as_ptr().cast())
                            }
                            Command::Threadgroup { index, length } => {
                                encoder.set_threadgroup_memory_length(*index, *length)
                            }
                            Command::Dispatch {
                                groups,
                                grid,
                                group,
                            } => {
                                if parallel.is_some_and(|plan| plan.barriers[dispatch_index]) {
                                    encoder.buffer_barrier();
                                }
                                dispatch_index += 1;
                                if *groups {
                                    encoder.dispatch_groups(*grid, *group);
                                } else {
                                    encoder.dispatch_threads(*grid, *group);
                                }
                            }
                        }
                    }
                }
            }
        }
        let outputs = self
            .outputs
            .iter()
            .map(|&(slot, offset)| {
                buffers[slot]
                    .as_ref()
                    .map(|(buffer, base)| {
                        debug_assert_eq!(*base, 0, "escaping storage is never packed");
                        ProgramOutput {
                            buffer: buffer.clone(),
                            offset,
                        }
                    })
                    .ok_or_else(|| {
                        crate::error::Error::msg("program output storage was not allocated")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        drop(parallel_scope);
        if let Some(plan) = parallel {
            debug_assert_eq!(dispatch_index, plan.barriers.len());
            debug_assert!(plan.independent_dispatches > 0);
            crate::trace::event("program.concurrent.replay");
            PARALLEL_REPLAYS.fetch_add(1, Ordering::Relaxed);
        }
        REPLAYS.fetch_add(1, Ordering::Relaxed);
        Ok(Some(outputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void program_affine(const device float* x [[buffer(0)]],
                           const device float* bias [[buffer(1)]],
                           device float* y [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < count) y[i] = x[i] * 2.0f + bias[i];
}
"#;

    fn values(runtime: &Arc<MetalRuntime>, data: &[f32]) -> Arc<Buffer> {
        let buffer = runtime
            .buffer(data.len() * 4, "program_test_input")
            .unwrap();
        // SAFETY: fresh completion-gated shared storage, with sufficient size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.contents(),
                data.len() * 4,
            );
        }
        buffer
    }

    fn read(buffer: &Buffer, offset: usize, count: usize) -> Vec<f32> {
        // SAFETY: callers synchronize the runtime first and keep `offset + 4 * count` in bounds.
        unsafe {
            std::slice::from_raw_parts(buffer.contents().add(offset).cast::<f32>(), count).to_vec()
        }
    }

    fn launch(
        runtime: &Arc<MetalRuntime>,
        pipeline: &ComputePipeline,
        input: &Arc<Buffer>,
        bias: &Arc<Buffer>,
        output: &Arc<Buffer>,
        offset: usize,
    ) {
        launch_view(runtime, pipeline, input, offset, bias, output, offset, 4);
    }

    fn launch_view(
        runtime: &Arc<MetalRuntime>,
        pipeline: &ComputePipeline,
        input: &Arc<Buffer>,
        input_offset: usize,
        bias: &Arc<Buffer>,
        output: &Arc<Buffer>,
        output_offset: usize,
        count: u32,
    ) {
        let guard = runtime.commands.encoder().unwrap();
        let encoder = guard.encoder();
        encoder.set_pipeline(pipeline);
        encoder.set_input(0, Some(input), input_offset);
        encoder.set_input(1, Some(bias), 0);
        encoder.set_output(2, Some(output), output_offset);
        encoder.set_bytes(3, &count);
        encoder.dispatch_threads((count as usize, 1, 1), (count as usize, 1, 1));
    }

    fn scratch_dispatch() -> Command {
        Command::Dispatch {
            groups: false,
            grid: (1, 1, 1),
            group: (1, 1, 1),
        }
    }

    #[test]
    fn scratch_closed_intervals_conserve_live_ranges_and_merge_released_tails() {
        let lifetimes = [
            (2, 0, 2),
            (1, 0, 0),
            (1, 1, 1),
            (1, 2, 3),
            (2, 3, 4),
            (1, 4, 4),
            (4, 5, 5),
        ]
        .into_iter()
        .enumerate()
        .map(|(slot, (capacity, first, last))| ScratchLifetime {
            slot,
            capacity: capacity * 1024,
            first,
            last,
        })
        .collect::<Vec<_>>();
        let (bytes, offsets) =
            ScratchArena::pack(lifetimes.clone(), lifetimes.len(), 4096).unwrap();
        assert_eq!(
            bytes, 4096,
            "adjacent released ranges must coalesce and extend"
        );
        assert_eq!(offsets[0], offsets[4]);
        assert_eq!(offsets[1], offsets[2]);
        assert_ne!(
            offsets[0], offsets[3],
            "closed intervals overlap at dispatch2"
        );
        assert_eq!(offsets[6], Some(0));
        assert!(ScratchArena::pack(lifetimes, 7, 4095).is_none());

        // Cover varied lengths, same-dispatch producers, gaps and nested
        // intervals without relying on one hand-picked packing arrangement.
        let mut seed = 0x7363_7261_7463_6831u64;
        for _ in 0..128 {
            let mut draw = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 32) as usize
            };
            let lifetimes = (0..20)
                .map(|slot| {
                    let first = draw() % 16;
                    ScratchLifetime {
                        slot,
                        capacity: (1 + draw() % 4) * 1024,
                        first,
                        last: first + draw() % 6,
                    }
                })
                .collect::<Vec<_>>();
            let sum = lifetimes
                .iter()
                .map(|lifetime| lifetime.capacity)
                .sum::<usize>();
            let (bytes, offsets) =
                ScratchArena::pack(lifetimes.clone(), lifetimes.len(), sum).unwrap();
            assert!(bytes <= sum);
            for (i, a) in lifetimes.iter().enumerate() {
                let ao = offsets[a.slot].unwrap();
                assert_eq!(ao % 1024, 0);
                assert!(ao + a.capacity <= bytes);
                for b in &lifetimes[i + 1..] {
                    let bo = offsets[b.slot].unwrap();
                    assert!(
                        a.last < b.first
                            || b.last < a.first
                            || ao + a.capacity <= bo
                            || bo + b.capacity <= ao,
                        "overlapping closed lifetimes share storage: {a:?} {b:?}"
                    );
                }
            }
            assert!(ScratchArena::pack(lifetimes, 20, bytes - 1).is_none());
        }
    }

    #[test]
    fn scratch_intervals_retain_inherited_aliases_until_replaced() {
        // Exercise conservative lifetime metadata, not an executable wrapper's
        // right to omit bindings. Real admitted sections must rebind every
        // argument their selected pipeline actually uses.
        let storage = (0..5)
            .map(|slot| Storage::Fresh {
                bytes: if slot == 4 { 2048 } else { 1024 },
                policy: AllocationPolicy::FullGpuWrite,
            })
            .collect::<Vec<_>>();
        let bind = |index, slot, write| Command::Buffer {
            index,
            slot: Some(slot),
            offset: 0,
            write,
        };
        let mut sections = (0..5).map(Section::Allocate).collect::<Vec<_>>();
        sections.push(Section::Encoder(vec![
            bind(0, 0, true),
            scratch_dispatch(),
            bind(0, 0, false),
            bind(1, 1, true),
            scratch_dispatch(),
        ]));
        // The second encoder acquisition does not itself clear bindings.
        // Slot0 remains bound through argument2 while slot1 is rebound at0.
        sections.push(Section::Encoder(vec![
            bind(2, 0, false),
            bind(0, 1, false),
            bind(1, 2, true),
            scratch_dispatch(),
            Command::Bytes {
                index: 2,
                data: vec![],
            },
            bind(0, 2, false),
            bind(1, 3, true),
            scratch_dispatch(),
        ]));
        sections.push(Section::Boundary);
        sections.push(Section::Encoder(vec![bind(0, 4, true), scratch_dispatch()]));
        let plan = ScratchArena::plan(&storage, &sections, &[(3, 0), (3, 8)], 3072).unwrap();
        assert_eq!(plan.unaliased_bytes, 5120);
        assert_eq!(plan.bytes, 3072);
        assert_eq!(
            plan.offsets[3], None,
            "every escaping alias stays independent"
        );
        assert_ne!(plan.offsets[0], plan.offsets[1]);
        assert_ne!(
            plan.offsets[0], plan.offsets[2],
            "inherited alias is still read"
        );
        assert_ne!(
            plan.offsets[1], plan.offsets[2],
            "read and write share a dispatch"
        );
        assert_eq!(
            plan.offsets[4],
            Some(0),
            "boundary permits reuse of dead ranges"
        );
    }

    #[test]
    fn scratch_plan_respects_escape_policy_alignment_and_capacity() {
        let fresh = |bytes, policy| Storage::Fresh { bytes, policy };
        let storage = vec![
            fresh(17, AllocationPolicy::FullGpuWrite),
            fresh(1025, AllocationPolicy::FullGpuWrite),
            fresh(33, AllocationPolicy::FullGpuWrite),
            fresh(34, AllocationPolicy::Zeroed),
            fresh(35, AllocationPolicy::CompactFullGpuWrite),
            fresh(0, AllocationPolicy::FullGpuWrite),
            fresh(36, AllocationPolicy::FullGpuWrite),
            fresh(37, AllocationPolicy::FullGpuWrite),
        ];
        let mut sections = storage
            .iter()
            .enumerate()
            .map(|(slot, _)| Section::Allocate(slot))
            .collect::<Vec<_>>();
        sections.push(Section::Encoder(
            (0..storage.len() - 1)
                .map(|slot| Command::Buffer {
                    index: slot,
                    slot: Some(slot),
                    // Slot 6 addresses excess captured capacity, so keep it
                    // independent; slot 7 has no recorded producer at all.
                    offset: if slot == 6 { 40 } else { 0 },
                    write: true,
                })
                .chain(std::iter::once(scratch_dispatch()))
                .collect(),
        ));
        let outputs = [(2, 4), (2, 12)];
        let plan = ScratchArena::plan(&storage, &sections, &outputs, 3072).unwrap();
        assert_eq!(plan.bytes, 3072);
        assert_eq!(plan.first_slot, 0);
        assert_eq!(
            plan.offsets,
            [Some(0), Some(1024), None, None, None, None, None, None]
        );
        assert!(ScratchArena::plan(&storage, &sections, &outputs, 3071).is_none());
        assert!(
            ScratchArena::plan(&storage, &sections, &[(0, 0), (1, 0), (2, 0)], usize::MAX)
                .is_none()
        );
        assert!(ScratchArena::plan(&[], &[], &[], usize::MAX).is_none());
        let overflow = [
            fresh(usize::MAX, AllocationPolicy::FullGpuWrite),
            fresh(16, AllocationPolicy::FullGpuWrite),
        ];
        let overflow_sections = [
            Section::Allocate(0),
            Section::Allocate(1),
            Section::Encoder(
                (0..2)
                    .map(|slot| Command::Buffer {
                        index: slot,
                        slot: Some(slot),
                        offset: 0,
                        write: true,
                    })
                    .chain(std::iter::once(scratch_dispatch()))
                    .collect(),
            ),
        ];
        assert!(ScratchArena::plan(&overflow, &overflow_sections, &[], usize::MAX).is_none());
    }

    #[test]
    fn scratch_replay_is_bitwise_with_views_aliases_and_concurrent_live_outputs() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        runtime.commands.set_per_buffer(1);
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let data = [1., 2., 3., 4., 5., 6., 7., 8.];
        let input = values(&runtime, &data);
        let bias = values(&runtime, &[0.25, -0.5, 1., -2., 0.5, 2., -1., 0.]);
        let inputs = [ProgramInput {
            buffer: input.clone(),
            required_bytes: 32,
        }];
        let (captured, program) = capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
            let a = runtime.buffer_for_audited_full_gpu_write(32, "scratch_a")?;
            launch_view(&runtime, &pipeline, &input, 0, &bias, &a, 0, 8);
            let b = runtime.buffer_for_audited_full_gpu_write(16, "scratch_b")?;
            launch_view(&runtime, &pipeline, &a, 8, &bias, &b, 0, 4);
            runtime.commands.flush_encoder();
            let c = runtime.buffer_for_audited_full_gpu_write(16, "scratch_c")?;
            launch(&runtime, &pipeline, &b, &bias, &c, 0);
            let output = runtime.buffer_for_audited_full_gpu_write(32, "scratch_output")?;
            launch_view(&runtime, &pipeline, &a, 0, &bias, &output, 0, 8);
            let compact = runtime.buffer_for_audited_compact_gpu_write(16, "scratch_compact")?;
            launch(&runtime, &pipeline, &c, &bias, &compact, 0);
            let zero = runtime.buffer(32, "scratch_zero")?;
            let outputs = vec![
                ProgramOutput {
                    buffer: output.clone(),
                    offset: 4,
                },
                ProgramOutput {
                    buffer: Arc::new(output.retain_shared()),
                    offset: 8,
                },
                ProgramOutput {
                    buffer: compact,
                    offset: 0,
                },
                ProgramOutput {
                    buffer: zero,
                    offset: 0,
                },
                ProgramOutput {
                    buffer: input.clone(),
                    offset: 4,
                },
            ];
            Ok((outputs.clone(), outputs))
        })
        .unwrap();
        let mut program = program.unwrap();
        assert_eq!(program.fresh_bytes(), 6 * 1024);
        assert_eq!(program.scratch.as_ref().unwrap().bytes, 3 * 1024);
        assert_eq!(
            program
                .scratch
                .as_ref()
                .unwrap()
                .offsets
                .iter()
                .flatten()
                .count(),
            3
        );
        let arena = program.scratch.take();
        let before = super::super::ALLOC_COUNT.load(Ordering::Relaxed);
        let ordinary = program.replay(&runtime, &inputs).unwrap().unwrap();
        assert_eq!(
            super::super::ALLOC_COUNT.load(Ordering::Relaxed) - before,
            6
        );
        program.scratch = arena;
        let before = super::super::ALLOC_COUNT.load(Ordering::Relaxed);
        let packed = program.replay(&runtime, &inputs).unwrap().unwrap();
        assert_eq!(
            super::super::ALLOC_COUNT.load(Ordering::Relaxed) - before,
            4
        );
        runtime.synchronize().unwrap();
        for index in 0..captured.len() {
            let count = if index == 3 { 8 } else { 4 };
            let bits = |output: &ProgramOutput| {
                read(&output.buffer, output.offset, count)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&ordinary[index]), bits(&captured[index]));
            assert_eq!(bits(&packed[index]), bits(&ordinary[index]));
        }
        assert_eq!(
            resource_id(&packed[0].buffer),
            resource_id(&packed[1].buffer)
        );
        assert_eq!(resource_id(&packed[4].buffer), resource_id(&input));
        assert_eq!(packed[2].buffer.length(), 1024);
        assert_eq!(read(&packed[3].buffer, 0, 8), [0.; 8]);
        assert_ne!(
            resource_id(&packed[0].buffer),
            resource_id(&ordinary[0].buffer)
        );
        assert_ne!(
            resource_id(&packed[0].buffer),
            resource_id(&captured[0].buffer)
        );

        let program = Arc::new(program);
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let workers = (0..4)
            .map(|worker| {
                let runtime = runtime.clone();
                let program = program.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let data = data.map(|value| value + worker as f32 * 10.);
                    let input = values(&runtime, &data);
                    barrier.wait();
                    let outputs = program
                        .replay(
                            &runtime,
                            &[ProgramInput {
                                buffer: input,
                                required_bytes: 32,
                            }],
                        )
                        .unwrap()
                        .unwrap();
                    (worker, outputs)
                })
            })
            .collect::<Vec<_>>();
        let results = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        runtime.synchronize().unwrap();
        let mut identities = HashSet::new();
        for (worker, outputs) in results {
            assert!(identities.insert(resource_id(&outputs[0].buffer)));
            let delta = worker as f32 * 40.;
            let expected = read(&ordinary[0].buffer, ordinary[0].offset, 4)
                .into_iter()
                .map(|v| (v + delta).to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                read(&outputs[0].buffer, outputs[0].offset, 4)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                expected
            );
            let expected = read(&ordinary[2].buffer, 0, 4)
                .into_iter()
                .map(|v| (v + worker as f32 * 160.).to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                read(&outputs[2].buffer, 0, 4)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        assert_eq!(
            read(&captured[0].buffer, 4, 4),
            read(&ordinary[0].buffer, 4, 4)
        );
    }

    #[test]
    fn scratch_interval_replay_overwrites_poison_and_preserves_live_outputs() {
        const COUNT: usize = 257;
        let runtime = Arc::new(MetalRuntime::new(usize::MAX).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let second_source = SOURCE.replace("program_affine(", "program_affine_second(");
        let second = runtime
            .compile(&second_source, "program_affine_second")
            .unwrap();
        let data = (0..COUNT)
            .map(|i| (i as f32 - 100.) * 0.125)
            .collect::<Vec<_>>();
        let input = values(&runtime, &data);
        let changed = values(&runtime, &data.iter().map(|x| x + 3.).collect::<Vec<_>>());
        let bias = values(&runtime, &vec![0.5; COUNT]);
        let input_view = |buffer: &Arc<Buffer>| {
            [ProgramInput {
                buffer: buffer.clone(),
                required_bytes: COUNT * 4,
            }]
        };
        let (captured, mut program) =
            capture_dispatch_program(&runtime, &input_view(&input), || -> Result<_> {
                let mut previous = input.clone();
                for step in 0..5 {
                    let next =
                        runtime.buffer_for_audited_full_gpu_write(COUNT * 4, "interval_chain")?;
                    // Two live wrapper aliases of a real input must extend
                    // the same resource's interval, not create two storages.
                    let other = if step == 1 {
                        Arc::new(previous.retain_shared())
                    } else {
                        bias.clone()
                    };
                    launch_view(
                        &runtime,
                        if step % 2 == 0 { &pipeline } else { &second },
                        &previous,
                        0,
                        &other,
                        &next,
                        0,
                        COUNT as u32,
                    );
                    previous = next;
                    if step == 2 {
                        runtime.commands.flush_encoder();
                    }
                }
                let output =
                    runtime.buffer_for_audited_full_gpu_write(COUNT * 4, "interval_output")?;
                launch_view(
                    &runtime,
                    &pipeline,
                    &previous,
                    0,
                    &bias,
                    &output,
                    0,
                    COUNT as u32,
                );
                let outputs = vec![
                    ProgramOutput {
                        buffer: output.clone(),
                        offset: 0,
                    },
                    ProgramOutput {
                        buffer: Arc::new(output.retain_shared()),
                        offset: 4,
                    },
                    ProgramOutput {
                        buffer: input.clone(),
                        offset: 4,
                    },
                ];
                Ok((outputs.clone(), outputs))
            })
            .unwrap();
        let mut program = program.take().unwrap();
        let plan = program.scratch.as_ref().unwrap();
        assert_eq!(plan.unaliased_bytes, 10 * 1024);
        assert_eq!(plan.bytes, 4 * 1024);
        assert_eq!(
            plan.offsets
                .iter()
                .flatten()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            2
        );

        // Compare against the same recorded commands without any arena. Keep
        // all outputs live, including the capture's aliases, through replay.
        let arena = program.scratch.take();
        let ordinary = program
            .replay(&runtime, &input_view(&input))
            .unwrap()
            .unwrap();
        let ordinary_changed = program
            .replay(&runtime, &input_view(&changed))
            .unwrap()
            .unwrap();
        program.scratch = arena;
        runtime.synchronize().unwrap();
        let poison = runtime.buffer_for_recorded_scratch(4096).unwrap();
        // SAFETY: all prior queue work is complete and this is an exclusive
        // private lease. Every logical new result must overwrite NaN poison.
        unsafe {
            std::ptr::write_bytes(poison.contents(), 0xff, poison.length());
        }
        let arena_id = poison.id();
        drop(poison);
        runtime.commands.set_per_buffer(1);
        let before = super::super::ALLOC_COUNT.load(Ordering::Relaxed);
        let packed = program
            .replay(&runtime, &input_view(&input))
            .unwrap()
            .unwrap();
        let packed_changed = program
            .replay(&runtime, &input_view(&changed))
            .unwrap()
            .unwrap();
        assert_eq!(
            super::super::ALLOC_COUNT.load(Ordering::Relaxed) - before,
            4,
            "each invocation needs only one scratch lease and one escaping output"
        );
        let arena = runtime
            .pool
            .take_program_scratch(4096, runtime.commands.program_identity())
            .unwrap();
        assert_eq!(arena.id(), arena_id);
        drop(arena);
        runtime.synchronize().unwrap();
        let bits = |output: &ProgramOutput, count| {
            read(&output.buffer, output.offset, count)
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&captured[0], COUNT), bits(&ordinary[0], COUNT));
        for (packed, ordinary) in [(&packed, &ordinary), (&packed_changed, &ordinary_changed)] {
            for index in 0..3 {
                assert_eq!(
                    bits(&packed[index], COUNT - usize::from(index > 0)),
                    bits(&ordinary[index], COUNT - usize::from(index > 0))
                );
            }
            assert_eq!(
                resource_id(&packed[0].buffer),
                resource_id(&packed[1].buffer)
            );
            assert_ne!(
                packed[0].buffer.id(),
                arena_id,
                "escaping aliases never share scratch"
            );
        }
        assert_ne!(
            resource_id(&packed[0].buffer),
            resource_id(&packed_changed[0].buffer)
        );
        assert_eq!(
            read(&input, 0, COUNT),
            data,
            "source input remains unchanged"
        );
    }

    #[test]
    fn scratch_budget_failure_remains_an_error_before_encoding() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.; 4]);
        let bias = values(&runtime, &[2.; 4]);
        let inputs = [ProgramInput {
            buffer: input.clone(),
            required_bytes: 16,
        }];
        let (captured_leases, program) =
            capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
                let a = runtime.buffer_for_audited_full_gpu_write(16, "budget_a")?;
                let b = runtime.buffer_for_audited_full_gpu_write(16, "budget_b")?;
                let out = runtime.buffer_for_audited_full_gpu_write(16, "budget_out")?;
                launch(&runtime, &pipeline, &input, &bias, &a, 0);
                launch(&runtime, &pipeline, &a, &bias, &b, 0);
                launch(&runtime, &pipeline, &b, &bias, &out, 0);
                Ok((
                    vec![a, b, out.clone()],
                    vec![ProgramOutput {
                        buffer: out,
                        offset: 0,
                    }],
                ))
            })
            .unwrap();
        let program = program.unwrap();
        assert!(program.scratch.is_some());
        runtime.synchronize().unwrap();
        runtime.swap_allocation_limit(runtime.allocated_bytes());
        let dispatches = super::super::DISPATCH_COUNT.load(Ordering::Relaxed);
        let result = program.replay(&runtime, &inputs);
        runtime.swap_allocation_limit(0);
        assert!(matches!(
            result,
            Err(crate::error::Error::AllocationUnavailable(_))
        ));
        assert_eq!(
            super::super::DISPATCH_COUNT.load(Ordering::Relaxed),
            dispatches
        );
        assert_eq!(read(&captured_leases[2], 0, 4), [22.; 4]);
        let outputs = program.replay(&runtime, &inputs).unwrap().unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(read(&outputs[0].buffer, 0, 4), [22.; 4]);
    }

    #[test]
    fn scratch_exact_capacity_preserves_larger_output_budget() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.; 4]);
        let bias = values(&runtime, &[2.; 4]);
        let pinned_bytes = runtime.pool.bytes();
        let inputs = [ProgramInput {
            buffer: input.clone(),
            required_bytes: 16,
        }];
        let (_, program) = capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
            // The two complete 16-byte results each occupy a 1 KiB bucket.
            // The final output is a partial writer with a 4 KiB zeroed view.
            let a = runtime.buffer_for_audited_full_gpu_write(16, "exact_budget_a")?;
            let b = runtime.buffer_for_audited_full_gpu_write(16, "exact_budget_b")?;
            let output = runtime.buffer(4096, "exact_budget_output")?;
            launch(&runtime, &pipeline, &input, &bias, &a, 0);
            launch(&runtime, &pipeline, &a, &bias, &b, 0);
            launch(&runtime, &pipeline, &b, &bias, &output, 0);
            Ok((
                (),
                vec![ProgramOutput {
                    buffer: output,
                    offset: 0,
                }],
            ))
        })
        .unwrap();
        let mut program = program.unwrap();
        assert_eq!(program.fresh_bytes(), 6144);
        assert_eq!(program.scratch.as_ref().unwrap().bytes, 2048);
        runtime.synchronize().unwrap();
        assert_eq!(runtime.pool.bytes(), pinned_bytes + 6144);
        // Account for pinned operands separately: the replay itself has an
        // exact 6 KiB allowance and free buckets [1 KiB, 1 KiB, 4 KiB].
        let _reservation = runtime
            .reserve_model_memory(1, pinned_bytes + 6144 + 1)
            .unwrap();
        let arena = program.scratch.take();
        let ordinary = program.replay(&runtime, &inputs).unwrap().unwrap();
        runtime.synchronize().unwrap();
        let expected = read(&ordinary[0].buffer, 0, 4);
        assert_eq!(expected, [22.; 4]);
        assert_eq!(ordinary[0].buffer.length(), 4096);
        drop(ordinary);
        assert_eq!(runtime.pool.bytes(), pinned_bytes + 6144);
        program.scratch = arena;
        // A best-fit 2 KiB arena would steal the 4 KiB bucket, leaving only
        // 2 KiB for this 4 KiB output and fail admission. Exact arena capacity
        // keeps both logical peaks at 6 KiB, without retrying encoded work.
        let packed = program.replay(&runtime, &inputs).unwrap().unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(packed[0].buffer.length(), 4096);
        assert_eq!(read(&packed[0].buffer, 0, 4), expected);
        assert_eq!(read(&packed[0].buffer, 16, 1020), vec![0.; 1020]);
        assert_eq!(runtime.pool.bytes(), pinned_bytes + 6144);
    }

    #[test]
    fn scratch_reuses_only_after_last_read_encoding_and_survives_pool_drop() {
        let mut runtime = Arc::new(MetalRuntime::new(usize::MAX).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.; 4]);
        let input2 = values(&runtime, &[3.; 4]);
        let bias = values(&runtime, &[2.; 4]);
        let inputs = [ProgramInput {
            buffer: input.clone(),
            required_bytes: 16,
        }];
        let (captured, program) = capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
            let a = runtime.buffer_for_audited_full_gpu_write(16, "reuse_a")?;
            launch(&runtime, &pipeline, &input, &bias, &a, 0);
            let b = runtime.buffer_for_audited_full_gpu_write(16, "reuse_b")?;
            launch(&runtime, &pipeline, &a, &bias, &b, 0);
            let output = runtime.buffer_for_audited_full_gpu_write(16, "reuse_output")?;
            launch(&runtime, &pipeline, &b, &bias, &output, 0);
            Ok((
                output.clone(),
                vec![ProgramOutput {
                    buffer: output,
                    offset: 0,
                }],
            ))
        })
        .unwrap();
        let program = program.unwrap();
        let queue = runtime.commands.program_identity();
        assert!(
            runtime.pool.take_program_scratch(2048, queue).is_none(),
            "capture uses ordinary storage"
        );
        let first = program.replay(&runtime, &inputs).unwrap().unwrap();
        let arena = runtime.pool.take_program_scratch(2048, queue).unwrap();
        let arena_id = arena.id();
        assert!(
            arena.last_used_cb() > runtime.commands.completed_watermark(),
            "test must reuse uncompleted work"
        );
        drop(arena);
        let second = program
            .replay(
                &runtime,
                &[ProgramInput {
                    buffer: input2,
                    required_bytes: 16,
                }],
            )
            .unwrap()
            .unwrap();
        let arena = runtime.pool.take_program_scratch(2048, queue).unwrap();
        assert_eq!(arena.id(), arena_id);
        assert!(arena.last_used_cb() > runtime.commands.completed_watermark());
        let weak = Arc::downgrade(&arena);
        drop(arena);
        // Removing the host pool cannot release the bound Metal resource:
        // Commands uses retaining command buffers, and every read/write was
        // bound before this private lease became available again.
        let old_pool = std::mem::replace(
            &mut Arc::get_mut(&mut runtime).unwrap().pool,
            Arc::new(super::super::BufferPool::new()),
        );
        drop(old_pool);
        assert!(
            weak.upgrade().is_none(),
            "the Metal command buffer owns the remaining resource lifetime"
        );
        runtime.synchronize().unwrap();
        assert_eq!(read(&captured, 0, 4), [22.; 4]);
        assert_eq!(read(&first[0].buffer, 0, 4), [22.; 4]);
        assert_eq!(read(&second[0].buffer, 0, 4), [38.; 4]);
        assert_ne!(
            resource_id(&first[0].buffer),
            resource_id(&second[0].buffer)
        );
    }

    #[test]
    fn scratch_capture_and_partial_failure_keep_private_leases_coherent() {
        let runtime = Arc::new(MetalRuntime::new(usize::MAX).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.; 4]);
        let bias = values(&runtime, &[2.; 4]);
        let inputs = [ProgramInput {
            buffer: input.clone(),
            required_bytes: 16,
        }];
        let (mut captured, program) =
            capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
                let a = runtime.buffer_for_audited_full_gpu_write(16, "failed_reuse_a")?;
                launch(&runtime, &pipeline, &input, &bias, &a, 0);
                let b = runtime.buffer_for_audited_full_gpu_write(16, "failed_reuse_b")?;
                launch(&runtime, &pipeline, &a, &bias, &b, 0);
                let output = runtime.buffer(4096, "failed_reuse_output")?;
                launch(&runtime, &pipeline, &b, &bias, &output, 0);
                Ok((
                    vec![a, b, output.clone()],
                    vec![ProgramOutput {
                        buffer: output,
                        offset: 0,
                    }],
                ))
            })
            .unwrap();
        let program = program.unwrap();
        runtime.synchronize().unwrap();
        let reservation = runtime
            .reserve_model_memory(1, runtime.pool.bytes() + 2048 + 1)
            .unwrap();
        let before = super::super::DISPATCH_COUNT.load(Ordering::Relaxed);
        let failed = program.replay(&runtime, &inputs);
        assert!(matches!(
            failed,
            Err(crate::error::Error::AllocationUnavailable(_))
        ));
        assert_eq!(
            super::super::DISPATCH_COUNT.load(Ordering::Relaxed) - before,
            2,
            "failure follows both scratch producers"
        );
        let queue = runtime.commands.program_identity();
        let arena = runtime.pool.take_program_scratch(2048, queue).unwrap();
        let arena_id = arena.id();
        assert!(arena.last_used_cb() > runtime.commands.completed_watermark());
        drop(arena);
        // Release the captured, already-completed 4 KiB output so the ordinary
        // output path can fit. The failed attempt is never silently retried.
        drop(captured.pop());
        let recovered = program.replay(&runtime, &inputs).unwrap().unwrap();
        let arena = runtime.pool.take_program_scratch(2048, queue).unwrap();
        assert_eq!(arena.id(), arena_id);
        drop(arena);
        runtime.synchronize().unwrap();
        assert_eq!(read(&recovered[0].buffer, 0, 4), [22.; 4]);
        drop(reservation);
        let (_, outer) = capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
            let output = program.replay(&runtime, &inputs)?.unwrap();
            Ok(((), output))
        })
        .unwrap();
        let outer = outer.expect("nested replay records completion-gated arena storage");
        let output = outer.replay(&runtime, &inputs).unwrap().unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(read(&output[0].buffer, 0, 4), [22.; 4]);
    }

    #[test]
    fn scratch_capture_handles_empty_regions_and_no_escaping_outputs() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let (_, empty) =
            capture_dispatch_program(&runtime, &[], || -> Result<_> { Ok(((), Vec::new())) })
                .unwrap();
        assert!(empty.is_none());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.; 4]);
        let bias = values(&runtime, &[2.; 4]);
        let (_, program) = capture_dispatch_program(&runtime, &[], || -> Result<_> {
            let a = runtime.buffer_for_audited_full_gpu_write(16, "empty_outputs_a")?;
            let b = runtime.buffer_for_audited_full_gpu_write(16, "empty_outputs_b")?;
            launch(&runtime, &pipeline, &input, &bias, &a, 0);
            launch(&runtime, &pipeline, &a, &bias, &b, 0);
            Ok(((), Vec::new()))
        })
        .unwrap();
        let program = program.unwrap();
        assert_eq!(program.scratch.as_ref().unwrap().bytes, 2048);
        let allocations = super::super::ALLOC_COUNT.load(Ordering::Relaxed);
        let outputs = program.replay(&runtime, &[]).unwrap().unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            super::super::ALLOC_COUNT.load(Ordering::Relaxed) - allocations,
            1
        );
        runtime.synchronize().unwrap();
    }

    #[test]
    fn replay_rebinds_inputs_preserves_views_and_live_outputs() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        // Force ordinary command-buffer rotations between some acquisitions.
        runtime.commands.set_per_buffer(1);
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[99.0, 1.0, 2.0, 3.0, 4.0]);
        let bias = values(&runtime, &[0.25, -0.5, 1.0, -2.0]);
        let inputs = [ProgramInput {
            buffer: input.clone(),
            required_bytes: 20,
        }];
        let (captured, program) = capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
            let middle = runtime.buffer(20, "program_test_middle")?;
            let output = runtime.buffer(20, "program_test_output")?;
            launch(&runtime, &pipeline, &input, &bias, &middle, 4);
            runtime.commands.flush_encoder();
            launch(&runtime, &pipeline, &middle, &bias, &output, 4);
            Ok((
                output.clone(),
                vec![ProgramOutput {
                    buffer: output,
                    offset: 4,
                }],
            ))
        })
        .unwrap();
        let program = program.expect("pure GPU chain is recordable");
        assert_eq!(program.dispatch_count(), 2);
        let input2 = values(&runtime, &[88.0, -1.0, -2.0, -3.0, -4.0]);
        let first = program
            .replay(
                &runtime,
                &[ProgramInput {
                    buffer: input2,
                    required_bytes: 20,
                }],
            )
            .unwrap()
            .unwrap();
        let first_cb = first[0].buffer.last_used_cb();
        let input3 = values(&runtime, &[77.0, 10.0, 20.0, 30.0, 40.0]);
        let second = program
            .replay(
                &runtime,
                &[ProgramInput {
                    buffer: input3,
                    required_bytes: 20,
                }],
            )
            .unwrap()
            .unwrap();
        assert_eq!(first[0].offset, 4);
        assert_ne!(resource_id(&captured), resource_id(&first[0].buffer));
        assert_ne!(
            resource_id(&first[0].buffer),
            resource_id(&second[0].buffer)
        );
        assert!(first_cb > 0);
        assert!(second[0].buffer.last_used_cb() >= first_cb);
        runtime.synchronize().unwrap();
        assert_eq!(read(&captured, 4, 4), vec![4.75, 6.5, 15.0, 10.0]);
        assert_eq!(read(&first[0].buffer, 4, 4), vec![-3.25, -9.5, -9.0, -22.0]);
        assert_eq!(
            read(&second[0].buffer, 4, 4),
            vec![40.75, 78.5, 123.0, 154.0]
        );
        // Pinned constants outlive their caller's local handle.
        drop(bias);
        let third = program.replay(&runtime, &inputs).unwrap().unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(read(&third[0].buffer, 4, 4), read(&captured, 4, 4));
    }

    #[test]
    fn capture_declines_input_writes_and_read_before_write_scratch() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0, 2.0, 3.0, 4.0]);
        let bias = values(&runtime, &[0.0; 4]);
        let (_, in_place) = capture_dispatch_program(
            &runtime,
            &[ProgramInput {
                buffer: input.clone(),
                required_bytes: 16,
            }],
            || -> Result<_> {
                launch(&runtime, &pipeline, &input, &bias, &input, 0);
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: input.clone(),
                        offset: 0,
                    }],
                ))
            },
        )
        .unwrap();
        assert!(in_place.is_none());
        let seed = values(&runtime, &[2.0; 4]);
        let (_, read_before_write) = capture_dispatch_program(&runtime, &[], || -> Result<_> {
            let temporary = runtime.buffer(16, "program_seed_consumer")?;
            launch(&runtime, &pipeline, &seed, &bias, &temporary, 0);
            launch(&runtime, &pipeline, &temporary, &bias, &seed, 0);
            Ok((
                (),
                vec![ProgramOutput {
                    buffer: seed.clone(),
                    offset: 0,
                }],
            ))
        })
        .unwrap();
        assert!(read_before_write.is_none());
        runtime.synchronize().unwrap();
    }

    #[test]
    fn replay_declines_alias_changes_before_encoding() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0; 4]);
        let bias = values(&runtime, &[2.0; 4]);
        let (_, program) = capture_dispatch_program(
            &runtime,
            &[
                ProgramInput {
                    buffer: input.clone(),
                    required_bytes: 16,
                },
                ProgramInput {
                    buffer: bias.clone(),
                    required_bytes: 16,
                },
            ],
            || -> Result<_> {
                let output = runtime.buffer(16, "program_alias_output")?;
                launch(&runtime, &pipeline, &input, &bias, &output, 0);
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: output,
                        offset: 0,
                    }],
                ))
            },
        )
        .unwrap();
        let program = program.unwrap();
        let before = super::super::DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            program
                .replay(
                    &runtime,
                    &[
                        ProgramInput {
                            buffer: input.clone(),
                            required_bytes: 16
                        },
                        ProgramInput {
                            buffer: input.clone(),
                            required_bytes: 16
                        },
                    ]
                )
                .unwrap()
                .is_none()
        );
        assert_eq!(
            super::super::DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed),
            before
        );
        // A second Buffer wrapper for the same Metal allocation is an alias too.
        let alias = Arc::new(input.retain_shared());
        assert!(
            program
                .replay(
                    &runtime,
                    &[
                        ProgramInput {
                            buffer: input,
                            required_bytes: 16
                        },
                        ProgramInput {
                            buffer: alias,
                            required_bytes: 16
                        },
                    ]
                )
                .unwrap()
                .is_none()
        );
        runtime.synchronize().unwrap();
    }

    #[test]
    fn capture_declines_full_write_host_imports() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let bias = values(&runtime, &[0.25, -0.5, 1.0, -2.0]);
        for compact in [false, true] {
            let (output, program) = capture_dispatch_program(&runtime, &[], || -> Result<_> {
                let input = if compact {
                    runtime.buffer_for_compact_full_write(16, "program_host_import")?
                } else {
                    runtime.buffer_for_full_write(16, "program_host_import")?
                };
                let data = [1.0f32, 2.0, 3.0, 4.0];
                // SAFETY: fresh 16-byte host-import lease with no GPU user.
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), input.contents(), 16);
                }
                let output = runtime.buffer(16, "program_host_import_output")?;
                launch(&runtime, &pipeline, &input, &bias, &output, 0);
                Ok((
                    output.clone(),
                    vec![ProgramOutput {
                        buffer: output,
                        offset: 0,
                    }],
                ))
            })
            .unwrap();
            assert!(program.is_none(), "CPU-written imports cannot be replayed");
            runtime.synchronize().unwrap();
            assert_eq!(read(&output, 0, 4), vec![2.25, 3.5, 7.0, 6.0]);
        }
    }

    #[test]
    fn capture_declines_host_synchronization_and_recovers_after_errors() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0; 4]);
        let bias = values(&runtime, &[2.0; 4]);
        let (_, program) = capture_dispatch_program(
            &runtime,
            &[ProgramInput {
                buffer: input.clone(),
                required_bytes: 16,
            }],
            || -> Result<_> {
                let output = runtime.buffer(16, "program_recovered_output")?;
                launch(&runtime, &pipeline, &input, &bias, &output, 0);
                runtime.synchronize()?;
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: output.clone(),
                        offset: 0,
                    }],
                ))
            },
        )
        .unwrap();
        assert!(program.is_none());
        assert!(
            capture_dispatch_program::<(), _>(&runtime, &[], || Err(crate::error::Error::msg(
                "expected"
            )))
            .is_err()
        );
        let (_, program) = capture_dispatch_program(
            &runtime,
            &[ProgramInput {
                buffer: input.clone(),
                required_bytes: 16,
            }],
            || -> Result<_> {
                let output = runtime.buffer(16, "program_recovered_output")?;
                launch(&runtime, &pipeline, &input, &bias, &output, 0);
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: output.clone(),
                        offset: 0,
                    }],
                ))
            },
        )
        .unwrap();
        assert!(program.is_some());
        runtime.synchronize().unwrap();
    }

    #[test]
    fn replay_preserves_full_write_and_compact_allocation_policies() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0, 2.0, 3.0, 4.0]);
        let bias = values(&runtime, &[0.25, -0.5, 1.0, -2.0]);
        // Poison a large reusable bucket. Its capacity must not become the
        // replay request, and compact persistent outputs must stay compact.
        drop(values(&runtime, &[12345.0; 8192]));
        let (_, program) = capture_dispatch_program(
            &runtime,
            &[ProgramInput {
                buffer: input.clone(),
                required_bytes: 16,
            }],
            || -> Result<_> {
                let middle = runtime.buffer_for_audited_full_gpu_write(16, "program_full")?;
                let output = runtime.buffer_for_audited_compact_gpu_write(16, "program_compact")?;
                launch(&runtime, &pipeline, &input, &bias, &middle, 0);
                launch(&runtime, &pipeline, &middle, &bias, &output, 0);
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: output,
                        offset: 4,
                    }],
                ))
            },
        )
        .unwrap();
        let program = program.unwrap();
        assert_eq!(program.fresh_bytes(), 2048);
        let first = program
            .replay(
                &runtime,
                &[ProgramInput {
                    buffer: input.clone(),
                    required_bytes: 16,
                }],
            )
            .unwrap()
            .unwrap();
        assert_eq!(first[0].buffer.length(), 1024);
        let second = program
            .replay(
                &runtime,
                &[ProgramInput {
                    buffer: input,
                    required_bytes: 16,
                }],
            )
            .unwrap()
            .unwrap();
        assert_ne!(
            resource_id(&first[0].buffer),
            resource_id(&second[0].buffer)
        );
        runtime.synchronize().unwrap();
        assert_eq!(
            read(&first[0].buffer, first[0].offset, 3),
            vec![6.5, 15.0, 10.0]
        );
        assert_eq!(
            read(&second[0].buffer, second[0].offset, 3),
            vec![6.5, 15.0, 10.0]
        );
    }

    #[test]
    fn capture_declines_unregistered_external_writes() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0; 4]);
        let bias = values(&runtime, &[2.0; 4]);
        let output = runtime.buffer(16, "program_external_output").unwrap();
        let (_, program) = capture_dispatch_program(
            &runtime,
            &[ProgramInput {
                buffer: input.clone(),
                required_bytes: 16,
            }],
            || -> Result<_> {
                launch(&runtime, &pipeline, &input, &bias, &output, 0);
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: output.clone(),
                        offset: 0,
                    }],
                ))
            },
        )
        .unwrap();
        assert!(program.is_none());
        runtime.synchronize().unwrap();
    }

    #[test]
    fn replay_requires_view_span_not_captured_pool_capacity() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0; 8192]);
        let bias = values(&runtime, &[2.0; 4]);
        let (_, program) = capture_dispatch_program(
            &runtime,
            &[ProgramInput {
                buffer: input.clone(),
                required_bytes: 16,
            }],
            || -> Result<_> {
                let output = runtime.buffer(16, "program_span_output")?;
                launch(&runtime, &pipeline, &input, &bias, &output, 0);
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: output,
                        offset: 0,
                    }],
                ))
            },
        )
        .unwrap();
        let program = program.unwrap();
        let smaller = values(&runtime, &[3.0; 4]);
        assert!(smaller.length() < input.length());
        let output = program
            .replay(
                &runtime,
                &[ProgramInput {
                    buffer: smaller.clone(),
                    required_bytes: 16,
                }],
            )
            .unwrap()
            .expect("the same view fits smaller storage");
        assert!(
            program
                .replay(
                    &runtime,
                    &[ProgramInput {
                        buffer: smaller,
                        required_bytes: 12,
                    }]
                )
                .unwrap()
                .is_none()
        );
        runtime.synchronize().unwrap();
        assert_eq!(read(&output[0].buffer, 0, 4), vec![8.0; 4]);
    }

    #[test]
    fn allocation_sections_preserve_order_and_encoder_dependencies() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0, 2.0, 3.0, 4.0]);
        let bias = values(&runtime, &[0.25, -0.5, 1.0, -2.0]);
        let inputs = [ProgramInput {
            buffer: input.clone(),
            required_bytes: 16,
        }];
        let (_, program) = capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
            let middle = runtime.buffer_for_audited_full_gpu_write(16, "program_order_middle")?;
            launch(&runtime, &pipeline, &input, &bias, &middle, 0);
            runtime.commands.flush_encoder();
            let output = runtime.buffer(32, "program_order_output")?;
            launch(&runtime, &pipeline, &middle, &bias, &output, 0);
            // A returned zero allocation may have no GPU binding at all. It
            // still needs its original allocation and initialization at replay.
            let zero = runtime.buffer(28, "program_order_zero")?;
            Ok((
                (),
                vec![
                    ProgramOutput {
                        buffer: output,
                        offset: 4,
                    },
                    ProgramOutput {
                        buffer: zero,
                        offset: 0,
                    },
                ],
            ))
        })
        .unwrap();
        let program = program.unwrap();
        let order = program
            .sections
            .iter()
            .map(|section| match section {
                Section::Allocate(slot) => match program.storage[*slot] {
                    Storage::Fresh { bytes, .. } => format!("allocate {bytes}"),
                    _ => panic!("allocation must refer to fresh storage"),
                },
                Section::Encoder(_) => "encoder".into(),
                Section::Boundary => "boundary".into(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            [
                "allocate 16",
                "encoder",
                "boundary",
                "allocate 32",
                "encoder",
                "allocate 28"
            ]
        );
        assert_eq!(
            program
                .storage
                .iter()
                .filter_map(|storage| match storage {
                    Storage::Fresh { bytes, .. } => Some(*bytes),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [16, 32, 28]
        );
        let output = program.replay(&runtime, &inputs).unwrap().unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(
            read(&output[0].buffer, output[0].offset, 3),
            [6.5, 15.0, 10.0]
        );
        assert_eq!(read(&output[1].buffer, 0, 7), [0.0; 7]);
    }

    #[test]
    fn capture_declines_allocation_while_encoder_guard_is_held() {
        let runtime = Arc::new(MetalRuntime::new(4).unwrap());
        let pipeline = runtime.compile(SOURCE, "program_affine").unwrap();
        let input = values(&runtime, &[1.0; 4]);
        let bias = values(&runtime, &[2.0; 4]);
        let (_, program) = capture_dispatch_program(
            &runtime,
            &[ProgramInput {
                buffer: input.clone(),
                required_bytes: 16,
            }],
            || -> Result<_> {
                let output = runtime.buffer(16, "program_guard_output")?;
                launch(&runtime, &pipeline, &input, &bias, &output, 0);
                {
                    let _guard = runtime.commands.encoder()?;
                    // This tiny unrestricted test allocation cannot trigger a
                    // budget wait. Production capture must still decline the
                    // pattern, since a later replay could have a tighter budget.
                    let _scratch = runtime.buffer(16, "program_guard_scratch")?;
                }
                Ok((
                    (),
                    vec![ProgramOutput {
                        buffer: output,
                        offset: 0,
                    }],
                ))
            },
        )
        .unwrap();
        assert!(program.is_none());
        runtime.synchronize().unwrap();
    }
}
