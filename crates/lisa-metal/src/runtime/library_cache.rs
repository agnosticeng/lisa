//! Bounded, runtime-local retention of compiled source libraries.
//!
//! Callers hold a separate library mutex across lookup and creation. This makes
//! concurrent requests for a retained identity share one creation, without
//! holding pipeline/allocator locks or waiting for GPU completion. Failed
//! creations never enter the cache and do not evict successful libraries.

use std::collections::HashMap;

use super::device::Math;

const MAX_LIBRARIES: usize = 128;
const MAX_SOURCE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Options {
    math: Math,
    nax: bool,
}

struct Entry<V> {
    value: V,
    last_used: u64,
}

/// Limits bound cache-retained handles and source-key bytes, not driver-owned
/// IR or pipeline memory. Eviction drops only this cache's handle; an active
/// compiler or pipeline keeps its own Metal references. Oversized source text
/// is compiled without retention. All ownership ends with its runtime.
pub(super) struct LibraryCache<V> {
    // Splitting the key lets a hit borrow `source` without allocating a String.
    // HashMap still compares the complete source, so hash collisions cannot
    // identify different libraries. Options include the NAX language target.
    groups: HashMap<Options, HashMap<String, Entry<V>>>,
    max_entries: usize,
    max_source_bytes: usize,
    entries: usize,
    source_bytes: usize,
    clock: u64,
    #[cfg(test)]
    lookups: usize,
    #[cfg(test)]
    creations: usize,
}

impl<V> Default for LibraryCache<V> {
    fn default() -> Self {
        Self::new(MAX_LIBRARIES, MAX_SOURCE_BYTES)
    }
}

impl<V> LibraryCache<V> {
    fn new(max_entries: usize, max_source_bytes: usize) -> Self {
        Self {
            groups: HashMap::new(),
            max_entries,
            max_source_bytes,
            entries: 0,
            source_bytes: 0,
            clock: 0,
            #[cfg(test)]
            lookups: 0,
            #[cfg(test)]
            creations: 0,
        }
    }

    fn evict_oldest(&mut self) {
        let oldest = self
            .groups
            .iter()
            .flat_map(|(options, group)| {
                group
                    .iter()
                    .map(move |(source, entry)| (*options, source, entry.last_used))
            })
            .min_by_key(|(_, _, last_used)| *last_used)
            .map(|(options, source, _)| (options, source.clone()));
        let Some((options, source)) = oldest else {
            return;
        };
        let group = self.groups.get_mut(&options).unwrap();
        group.remove(&source);
        self.source_bytes -= source.len();
        self.entries -= 1;
        if group.is_empty() {
            self.groups.remove(&options);
        }
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> (usize, usize, usize, usize) {
        (
            self.lookups,
            self.creations,
            self.entries,
            self.source_bytes,
        )
    }
}

impl<V: Clone> LibraryCache<V> {
    pub(super) fn get_or_try_insert_with<E>(
        &mut self,
        source: &str,
        math: Math,
        nax: bool,
        create: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        #[cfg(test)]
        {
            self.lookups += 1;
        }
        self.clock = self.clock.saturating_add(1);
        let options = Options { math, nax };
        if let Some(entry) = self
            .groups
            .get_mut(&options)
            .and_then(|group| group.get_mut(source))
        {
            entry.last_used = self.clock;
            return Ok(entry.value.clone());
        }
        #[cfg(test)]
        {
            self.creations += 1;
        }
        let value = create()?;
        if self.max_entries == 0 || source.len() > self.max_source_bytes {
            return Ok(value);
        }
        // Do not reclaim useful entries until creation succeeds. Subtraction
        // avoids overflow for adversarial source lengths and cache budgets.
        while self.entries >= self.max_entries
            || self.source_bytes > self.max_source_bytes - source.len()
        {
            self.evict_oldest();
        }
        self.groups.entry(options).or_default().insert(
            source.to_owned(),
            Entry {
                value: value.clone(),
                last_used: self.clock,
            },
        );
        self.entries += 1;
        self.source_bytes += source.len();
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn insert(cache: &mut LibraryCache<Arc<u32>>, source: &str, value: u32) -> Arc<u32> {
        cache
            .get_or_try_insert_with(source, Math::Safe, false, || Ok::<_, ()>(Arc::new(value)))
            .unwrap()
    }

    #[test]
    fn library_cache_lru_preserves_live_handles_and_exact_entry_bound() {
        let mut cache = LibraryCache::new(2, 64);
        let a = insert(&mut cache, "a", 1);
        let b = insert(&mut cache, "b", 2);
        let b_weak = Arc::downgrade(&b);
        drop(b);
        assert!(Arc::ptr_eq(&insert(&mut cache, "a", 99), &a));
        let c = insert(&mut cache, "c", 3);
        assert!(b_weak.upgrade().is_none(), "least-recent cache owner drops");
        assert_eq!(cache.stats(), (4, 3, 2, 2));
        let a_weak = Arc::downgrade(&a);
        insert(&mut cache, "d", 4);
        assert_eq!(*a, 1, "an evicted compiler handle stays valid");
        drop(a);
        assert!(a_weak.upgrade().is_none());
        assert_eq!(*c, 3);
        assert_eq!(cache.stats(), (5, 4, 2, 2));
    }

    #[test]
    fn library_cache_source_budget_and_oversized_bypass_are_bounded() {
        let mut cache = LibraryCache::new(8, 5);
        insert(&mut cache, "aa", 1);
        insert(&mut cache, "bbb", 2);
        insert(&mut cache, "c", 3);
        assert_eq!(cache.stats(), (3, 3, 2, 4));
        insert(&mut cache, "123456", 4);
        insert(&mut cache, "123456", 5);
        assert_eq!(cache.stats(), (5, 5, 2, 4));
        assert_eq!(*insert(&mut cache, "bbb", 99), 2);
        assert_eq!(cache.stats(), (6, 5, 2, 4));
        let mut disabled = LibraryCache::new(0, usize::MAX);
        insert(&mut disabled, "not retained", 1);
        assert_eq!(disabled.stats(), (1, 1, 0, 0));
    }

    #[test]
    fn library_cache_failure_keeps_entries_and_retry_can_succeed() {
        let mut cache = LibraryCache::new(1, 16);
        let a = insert(&mut cache, "a", 1);
        for _ in 0..2 {
            assert_eq!(
                cache.get_or_try_insert_with("b", Math::Safe, false, || Err("compile")),
                Err("compile")
            );
        }
        assert_eq!(cache.stats(), (3, 3, 1, 1));
        assert!(Arc::ptr_eq(&insert(&mut cache, "a", 99), &a));
        assert_eq!(*insert(&mut cache, "b", 2), 2);
        assert_eq!(cache.stats(), (5, 4, 1, 1));
    }

    #[test]
    fn library_cache_source_math_and_nax_are_independent_identities() {
        let mut cache = LibraryCache::new(16, 1024);
        for (i, (source, math, nax)) in [
            ("source", Math::CompilerDefault, false),
            ("source", Math::Jit, false),
            ("source", Math::Safe, false),
            ("source", Math::SafeNoLang, false),
            ("source", Math::Fast, false),
            ("source", Math::Safe, true),
            ("source ", Math::Safe, false),
        ]
        .into_iter()
        .enumerate()
        {
            for _ in 0..2 {
                assert_eq!(
                    cache
                        .get_or_try_insert_with(source, math, nax, || Ok::<_, ()>(i))
                        .unwrap(),
                    i
                );
            }
        }
        assert_eq!(cache.stats(), (14, 7, 7, 43));
    }
}
