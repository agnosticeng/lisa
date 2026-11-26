use super::*;

// These are real allocator/ownership checks with tiny Metal buffers. Explicit
// used-command-buffer IDs exercise pending eligibility deterministically; no
// kernel submission is needed to establish the retention/admission contract.
const KIB: usize = 1024;

fn default_policy() -> MetalRuntime {
    let runtime = MetalRuntime::new(4).unwrap();
    // Set the construction default directly so a test-runner environment does
    // not turn these cases into explicit-override tests.
    runtime.pool.set_default_limit(KIB);
    runtime
}

fn poison(buffer: &Buffer, byte: u8) {
    // SAFETY: every caller owns a freshly allocated or known-completed buffer.
    unsafe { std::ptr::write_bytes(buffer.contents(), byte, buffer.length()) };
}

fn assert_bytes(buffer: &Buffer, byte: u8) {
    assert!(
        // SAFETY: callers pass idle buffers; the range is exactly the allocation.
        unsafe { std::slice::from_raw_parts(buffer.contents(), buffer.length()) }
            .iter()
            .all(|&value| value == byte)
    );
}

#[test]
fn resident_shape_misses_retain_general_buffers_and_exact_private_arenas() {
    let runtime = default_policy();
    let _reservation = runtime.reserve_model_memory(KIB, 64 * KIB).unwrap();
    let held = runtime.buffer(4 * KIB, "retention_held").unwrap();
    let general = runtime
        .buffer_for_compact_full_write(2 * KIB, "retention_general")
        .unwrap();
    let arena = runtime.buffer_for_recorded_scratch(3 * KIB).unwrap();
    poison(&general, 0xa5);
    poison(&arena, 0x6b);
    let general_id = general.id();
    let arena_id = arena.id();
    drop((general, arena));

    // Neither 5 KiB request can use either cached capacity. The misses must
    // preserve both completed caches while the explicit hard budget has room.
    let general_miss = runtime
        .buffer_for_compact_full_write(5 * KIB, "retention_new_shape")
        .unwrap();
    let arena_miss = runtime.buffer_for_recorded_scratch(5 * KIB).unwrap();
    assert_eq!(runtime.pool.bytes(), 19 * KIB);
    assert_eq!(runtime.pool.total_bytes(), 20 * KIB);
    let general = runtime
        .buffer_for_compact_full_write(2 * KIB, "retention_general_again")
        .unwrap();
    let arena = runtime.buffer_for_recorded_scratch(3 * KIB).unwrap();
    assert_eq!(general.id(), general_id);
    assert_eq!(arena.id(), arena_id);
    assert_bytes(&general, 0xa5);
    assert_bytes(&arena, 0x6b);
    assert_eq!(runtime.pool.bytes(), 19 * KIB, "hits add no storage");
    assert_eq!(held.length(), 4 * KIB);
    assert_ne!(general_miss.id(), arena_miss.id());

    // Explicit clearing still returns every completed unused allocation even
    // with a live reservation; live outputs remain owned and charged.
    drop((general, arena));
    assert_eq!(runtime.pool.sweep_completed(0), 5 * KIB);
    assert_eq!(runtime.pool.total_bytes(), 15 * KIB);
    assert_eq!(general_miss.length(), 5 * KIB);
    assert_eq!(arena_miss.length(), 5 * KIB);
}

#[test]
fn resident_hard_cap_reclaims_completed_caches_then_refuses_live_overflow() {
    let runtime = default_policy();
    let _reservation = runtime.reserve_model_memory(KIB, 12 * KIB).unwrap();
    let held = runtime.buffer(4 * KIB, "retention_live").unwrap();
    poison(&held, 0x8c);
    let general = runtime
        .buffer_for_compact_full_write(2 * KIB, "retention_evict_general")
        .unwrap();
    let arena = runtime.buffer_for_recorded_scratch(3 * KIB).unwrap();
    let general_weak = Arc::downgrade(&general);
    let arena_weak = Arc::downgrade(&arena);
    drop((general, arena));
    assert_eq!(runtime.pool.total_bytes(), 10 * KIB);

    // 10 + 5 exceeds 12 KiB. Hard admission must reclaim the 5 KiB of
    // completed idle storage before creating the new allocation.
    let fresh = runtime
        .buffer_for_compact_full_write(5 * KIB, "retention_budget_reclaim")
        .unwrap();
    assert!(general_weak.upgrade().is_none());
    assert!(arena_weak.upgrade().is_none());
    assert_eq!(runtime.pool.total_bytes(), 10 * KIB);
    let denied = runtime
        .buffer_for_compact_full_write(3 * KIB, "retention_live_overflow")
        .err()
        .unwrap();
    assert!(matches!(
        denied,
        crate::error::Error::AllocationUnavailable(_)
    ));
    assert_eq!(runtime.pool.total_bytes(), 10 * KIB);
    assert_bytes(&held, 0x8c);
    assert_eq!(fresh.length(), 5 * KIB);
}

#[test]
fn zero_byte_reservation_preserves_retention_admission_and_release_lifetime() {
    let runtime = default_policy();
    let mut reservation = runtime.reserve_model_memory(KIB, 8 * KIB).unwrap();
    reservation.shrink_to(&runtime.pool, 0).unwrap();
    assert_eq!(runtime.pool.reserved_model_bytes(), 0);
    assert!(runtime.pool.has_resident_admission());
    let held = runtime.buffer(4 * KIB, "retention_zero_held").unwrap();
    let hot = runtime
        .buffer_for_compact_full_write(2 * KIB, "retention_zero_hot")
        .unwrap();
    let hot_id = hot.id();
    drop(hot);
    let small = runtime
        .buffer_for_compact_full_write(KIB, "retention_zero_miss")
        .unwrap();
    let hot = runtime
        .buffer_for_compact_full_write(2 * KIB, "retention_zero_reuse")
        .unwrap();
    assert_eq!(hot.id(), hot_id);
    assert_eq!(runtime.pool.total_bytes(), 7 * KIB);
    assert!(
        runtime
            .buffer_for_compact_full_write(2 * KIB, "retention_zero_denied")
            .is_err()
    );
    assert_eq!(runtime.pool.total_bytes(), 7 * KIB);

    let retired = [
        Arc::downgrade(&held),
        Arc::downgrade(&hot),
        Arc::downgrade(&small),
    ];
    drop((held, hot, small));
    drop(reservation);
    assert!(!runtime.pool.has_resident_admission());
    let after = runtime
        .buffer_for_compact_full_write(8 * KIB, "retention_after_reservation")
        .unwrap();
    // The denied allocation above attempted a hard-budget sweep with all
    // 7 KiB live, establishing the existing growth baseline at 7 KiB. The
    // first unadmitted miss adds storage before another soft sweep is due.
    assert_eq!(runtime.pool.bytes(), 15 * KIB);
    assert!(retired.iter().all(|buffer| buffer.upgrade().is_some()));
    poison(&after, 0xac);
    let next = runtime
        .buffer_for_compact_full_write(16 * KIB, "retention_after_growth")
        .unwrap();
    assert_eq!(
        runtime.pool.bytes(),
        24 * KIB,
        "the next miss reclaims the old 7 KiB while the 8 KiB lease stays live"
    );
    assert!(retired.iter().all(|buffer| buffer.upgrade().is_none()));
    assert_bytes(&after, 0xac);
    assert_eq!(after.length(), 8 * KIB);
    assert_eq!(next.length(), 16 * KIB);
}

#[test]
fn lower_budget_does_not_reclaim_pending_cpu_or_foreign_queue_storage() {
    let runtime = default_policy();
    let _reservation = runtime.reserve_model_memory(KIB, 8 * KIB).unwrap();
    let held = runtime.buffer(2 * KIB, "retention_pending_held").unwrap();
    let general = runtime
        .buffer_for_compact_full_write(2 * KIB, "retention_pending_general")
        .unwrap();
    let arena = runtime.buffer_for_recorded_scratch(2 * KIB).unwrap();
    poison(&general, 0x54);
    poison(&arena, 0x39);
    let general_id = general.id();
    let arena_id = arena.id();
    general.mark_used_cb(100);
    arena.mark_used_cb(100);
    drop((general, arena));
    assert_eq!(runtime.pool.total_bytes(), 7 * KIB);

    let denied = runtime
        .buffer(3 * KIB, "retention_pending_clear_denied")
        .err()
        .unwrap();
    assert!(matches!(
        denied,
        crate::error::Error::AllocationUnavailable(_)
    ));
    assert_eq!(runtime.pool.total_bytes(), 7 * KIB);
    assert!(runtime.pool.take_exact(2 * KIB, 99, false).is_none());
    let queue = runtime.commands.program_identity();
    assert!(
        runtime
            .pool
            .take_program_scratch(2 * KIB, queue.wrapping_add(1))
            .is_none()
    );
    let general = runtime.pool.take_exact(2 * KIB, 100, false).unwrap();
    let arena = runtime.pool.take_program_scratch(2 * KIB, queue).unwrap();
    assert_eq!(general.id(), general_id);
    assert_eq!(arena.id(), arena_id);
    assert_bytes(&general, 0x54);
    assert_bytes(&arena, 0x39);
    drop((general, arena));
    assert_eq!(runtime.pool.sweep_completed(99), 0);
    assert_eq!(runtime.pool.sweep_completed(100), 4 * KIB);
    assert_eq!(runtime.pool.total_bytes(), 3 * KIB);
    assert_eq!(held.length(), 2 * KIB);
}

#[test]
fn explicit_pool_limit_keeps_its_policy_under_resident_admission() {
    let runtime = default_policy();
    let _reservation = runtime.reserve_model_memory(KIB, 64 * KIB).unwrap();
    let held = runtime.buffer(4 * KIB, "retention_override_held").unwrap();
    let old = runtime.buffer_for_recorded_scratch(2 * KIB).unwrap();
    let old_weak = Arc::downgrade(&old);
    drop(old);
    runtime.pool.set_limit(KIB);
    let fresh = runtime
        .buffer_for_compact_full_write(3 * KIB, "retention_override_miss")
        .unwrap();
    assert!(old_weak.upgrade().is_none());
    assert_eq!(runtime.pool.total_bytes(), 8 * KIB);

    // Explicit zero continues to mean no automatic soft sweep. Hard admission
    // and explicit clearing remain available independently of this override.
    runtime.pool.set_limit(0);
    let cached = runtime.buffer_for_recorded_scratch(2 * KIB).unwrap();
    let id = cached.id();
    drop(cached);
    let second = runtime
        .buffer_for_compact_full_write(5 * KIB, "retention_zero_override_miss")
        .unwrap();
    let cached = runtime.buffer_for_recorded_scratch(2 * KIB).unwrap();
    assert_eq!(cached.id(), id);
    assert_eq!(runtime.pool.total_bytes(), 15 * KIB);
    assert_eq!(held.length() + fresh.length() + second.length(), 12 * KIB);
}

#[test]
fn native_allocation_limit_still_reclaims_and_refuses_under_resident_budget() {
    let runtime = default_policy();
    let _reservation = runtime.reserve_model_memory(KIB, 128 * KIB).unwrap();
    let held = runtime.buffer(4 * KIB, "retention_native_held").unwrap();
    let cached = runtime.buffer_for_recorded_scratch(2 * KIB).unwrap();
    let cached_weak = Arc::downgrade(&cached);
    drop(cached);
    runtime.swap_allocation_limit(runtime.allocated_bytes());
    let denied = runtime
        .buffer_for_compact_full_write(64 * KIB, "retention_native_limit")
        .err()
        .unwrap();
    assert!(matches!(
        denied,
        crate::error::Error::AllocationUnavailable(_)
    ));
    assert!(cached_weak.upgrade().is_none());
    assert_eq!(runtime.pool.total_bytes(), 5 * KIB);
    assert_eq!(held.length(), 4 * KIB);
}

#[test]
fn concurrent_misses_share_one_atomic_resident_budget() {
    let runtime = Arc::new(default_policy());
    let _reservation = runtime.reserve_model_memory(KIB, 7 * KIB).unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let workers = (0..8)
        .map(|_| {
            let runtime = runtime.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                runtime.buffer_for_compact_full_write(2 * KIB, "retention_concurrent")
            })
        })
        .collect::<Vec<_>>();
    let results = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 3);
    assert!(
        results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .all(|error| matches!(error, crate::error::Error::AllocationUnavailable(_)))
    );
    assert_eq!(runtime.pool.total_bytes(), 7 * KIB);
    let mut ids = results
        .iter()
        .filter_map(|result| result.as_ref().ok())
        .map(|buffer| buffer.id())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        3,
        "every successful caller owns distinct storage"
    );
}
