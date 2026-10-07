//! Process-level prompt prefix cache (specs/06 item 2): a small LRU of
//! prompt-boundary states. An entry stores the exact tokens fed and a
//! per-layer clone of the cache state at that boundary (full-attention KV
//! rows + indexer tape, GDN conv/SSM/PLE state). A later request whose
//! prompt token-extends an entry resumes from the split point and prefills
//! only the suffix.
//!
//! Only PROMPT state is cached (never a mid-generation state): the prompt is
//! independent of the sampled continuation, so a temp>0 regeneration or a
//! tool-call turn needs no invalidation — a stale entry can only be reused
//! when the new prompt is a strict token-prefix extension of it.
//!
//! SSD spill tier (the reference hot-cache spill semantics, `prefix_cache.zig`
//! + `kv_disk_cache.zig` in the reference tree): entries evicted from the RAM
//! LRU persist under `~/.lisa/prefix_spill/` — one self-describing file per
//! entry, keyed by a content hash of its token ids, LRU-capped by mtime — and
//! a disk entry RESTORES on lookup whenever it beats the best RAM match.
//! Disk is NOT RAM: spilled entries hold no resident budget, so the footprint
//! ledger is unchanged. A restore re-warms the entry into the RAM LRU; the
//! file stays (it survives server restarts). RAM invalidation propagates:
//! spill files recorded under a foreign numerical law are unlinked on the
//! next insert/scan.

use crate::core::cache::{LayerCache, LayerState};
use crate::core::session::Session;
use crate::models::LanguageModel;
use std::path::PathBuf;

/// Do not spend ~113 MB of GDN state on prefixes shorter than this.
pub const MIN_PREFIX_TOKENS: usize = 128;

/// Spill files are LRU-capped by mtime at this count (the RAM tier is count-
/// capped too; the disk tier mirrors it).
const SPILL_MAX_FILES: usize = 8;
const SPILL_MAGIC: [u8; 4] = *b"LPS1";
/// Spill-file header: magic(4) law(8) ntok(4).
const SPILL_HEADER: usize = 16;

pub struct PrefixCache {
    max_entries: usize,
    /// Front = most recently used.
    entries: Vec<PrefixEntry>,
}

struct PrefixEntry {
    tokens: Vec<u32>,
    /// O2: the numerical law (`lisa_mlx::runtime::kernel_law()`) under
    /// which `layers` was computed. A different law means those KV/GDN rows came
    /// from different kernel arithmetic (math mode, NAX path) and are not a
    /// valid resume point — such an entry must never be served.
    law: u64,
    layers: Vec<LayerState>,
}

/// FNV-1a 64 over the law + token ids (LE): the spill file's content hash.
fn entry_hash(law: u64, tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in law.to_le_bytes().into_iter().chain(tokens.iter().flat_map(|t| t.to_le_bytes())) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn spill_dir() -> Option<PathBuf> {
    let d = std::env::var_os("HOME")
        .map(PathBuf::from)?
        .join(".lisa")
        .join("prefix_spill");
    let _ = std::fs::create_dir_all(&d);
    Some(d)
}

impl PrefixCache {
    /// `max_entries == 0` disables the cache.
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            entries: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.max_entries > 0
    }

    /// Snapshot the caches at the `tokens` boundary (a clean commit point:
    /// call right after the prompt prefill, before any decode).
    pub fn insert(&mut self, tokens: &[u32], caches: &[LayerCache]) {
        if !self.enabled() || tokens.len() < MIN_PREFIX_TOKENS {
            return;
        }
        // O2: states are only comparable within one numerical law. Drop any
        // entry recorded under a different kernel fingerprint (a recompiled or
        // re-targeted kernel invalidates its rows), alongside the same-prefix
        // dedupe. The invalidation propagates to the spill tier.
        let law = lisa_mlx::runtime::kernel_law();
        self.entries.retain(|e| e.tokens != tokens && e.law == law);
        prune_spill_foreign(law);
        let layers = caches.iter().map(|c| c.snapshot_prefix()).collect();
        self.admit(PrefixEntry {
            tokens: tokens.to_vec(),
            law,
            layers,
        });
    }

    /// Admit an entry at the LRU front, spilling (not dropping) whatever
    /// falls off the tail.
    fn admit(&mut self, e: PrefixEntry) {
        self.entries.insert(0, e);
        while self.entries.len() > self.max_entries {
            let dropped = self.entries.pop().expect("len > max");
            spill_entry(&dropped);
        }
    }

    /// The longest cached token-prefix of `prompt` that `prompt` STRICTLY
    /// extends, so the caller always has at least one token left to prefill (a
    /// resumed state must be fed a fresh token to yield next-token logits; a
    /// full match would need a resume-at-end path this engine does not have —
    /// replaying the last token instead DOUBLE-COUNTS it and changes the
    /// distribution, verified: "a long string" vs "a large block").
    /// RAM entries first; the spill tier counts when it beats the RAM match.
    pub fn lookup(&self, prompt: &[u32]) -> Option<usize> {
        let law = lisa_mlx::runtime::kernel_law();
        let ram = self.lookup_ram(prompt, law);
        match best_disk_match(prompt, law, ram) {
            Some((n, _)) => Some(n),
            None => ram,
        }
    }

    /// Resume a session from the longest cached token-prefix of `prompt`.
    /// Returns `(matched_len, session)`; the caller prefills `prompt[matched..]`.
    /// A disk entry that beats the RAM match is loaded from its spill file and
    /// re-warmed into the RAM LRU (the file stays — it survives restarts).
    pub fn restore_session(
        &mut self,
        tower: &mut dyn LanguageModel,
        prompt: &[u32],
    ) -> Option<(usize, Session)> {
        let law = lisa_mlx::runtime::kernel_law();
        let ram_len = self.lookup_ram(prompt, law);
        if let Some((n, path)) = best_disk_match(prompt, law, ram_len) {
            if n > ram_len.unwrap_or(0) {
                if let Some(e) = load_spill_entry(&path, prompt, n, law) {
                    self.rewarm(e);
                    return Some((
                        n,
                        Session::from_prefix(tower, &self.entries[0].layers, prompt, n),
                    ));
                }
            }
        }
        let l = ram_len?;
        let e = self
            .entries
            .iter()
            .find(|e| e.tokens.len() == l && e.law == law)?;
        Some((l, Session::from_prefix(tower, &e.layers, prompt, l)))
    }

    /// Re-warm a restored disk entry: admit only if no identical-token entry
    /// is resident (the restored state is the same content hash).
    fn rewarm(&mut self, e: PrefixEntry) {
        if self.entries.iter().any(|r| r.tokens == e.tokens) {
            return;
        }
        self.admit(e);
    }

    fn lookup_ram(&self, prompt: &[u32], law: u64) -> Option<usize> {
        let mut best: Option<usize> = None;
        for e in &self.entries {
            // O2: never resume rows computed under a different numerical law.
            if e.law != law {
                continue;
            }
            let n = e.tokens.len();
            if n < prompt.len() && prompt[..n] == e.tokens[..] && best.is_none_or(|b| n > b) {
                best = Some(n);
            }
        }
        best
    }
}

/// Spill-file header scan: `(law, token_len)`, or `None` for junk/crash
/// leftovers (partial write before the tmp+rename completes is invisible —
/// only the renamed file exists; a truncated one fails this scan).
fn spill_header(path: &std::path::Path) -> Option<(u64, usize)> {
    let b = std::fs::read(path).ok()?;
    if b.len() < SPILL_HEADER || b[..4] != SPILL_MAGIC {
        return None;
    }
    let law = u64::from_le_bytes(b[4..12].try_into().ok()?);
    let ntok = u32::from_le_bytes(b[12..SPILL_HEADER].try_into().ok()?) as usize;
    if b.len() < SPILL_HEADER + ntok * 4 + 4 {
        return None;
    }
    Some((law, ntok))
}

/// The best disk prefix match that strictly beats `at_least`. Unlinks
/// foreign-law files it encounters (RAM invalidation propagates) and
/// truncated junk. Returns `(match_len, path)`.
fn best_disk_match(
    prompt: &[u32],
    law: u64,
    at_least: Option<usize>,
) -> Option<(usize, PathBuf)> {
    let dir = spill_dir()?;
    let rd = std::fs::read_dir(&dir).ok()?;
    let floor = at_least.unwrap_or(0);
    let mut best: Option<(usize, PathBuf)> = None;
    for p in rd.flatten() {
        let path = p.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let Some((file_law, ntok)) = spill_header(&path) else {
            let _ = std::fs::remove_file(&path);
            continue;
        };
        if file_law != law {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        // Cannot beat the floor / cannot serve a strict extension.
        if ntok <= floor || ntok >= prompt.len() {
            continue;
        }
        if best.as_ref().is_some_and(|(b, _)| ntok <= *b) {
            continue;
        }
        // Full token read only when this file could win.
        if let Some(tokens) = spill_tokens(&path, ntok) {
            if prompt[..ntok] == tokens[..] {
                best = Some((ntok, path));
            }
        }
    }
    best
}

fn spill_tokens(path: &std::path::Path, ntok: usize) -> Option<Vec<u32>> {
    let b = std::fs::read(path).ok()?;
    let s = &b[SPILL_HEADER..SPILL_HEADER + ntok * 4];
    Some(
        s.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("tok")))
            .collect(),
    )
}

/// Serialize one evicted entry to the spill dir (tmp + rename), then LRU-cap
/// the dir by mtime (the reference's disk-tier LRU). Best-effort: a spill
/// failure just drops the entry (the pre-spill behavior); it never costs the
/// request that triggered the evict.
fn spill_entry(e: &PrefixEntry) {
    let Some(dir) = spill_dir() else { return };
    let name = format!("p{:016x}.bin", entry_hash(e.law, &e.tokens));
    let path = dir.join(&name);
    let mut out = Vec::new();
    out.extend_from_slice(&SPILL_MAGIC);
    out.extend_from_slice(&e.law.to_le_bytes());
    out.extend_from_slice(&(e.tokens.len() as u32).to_le_bytes());
    for t in &e.tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out.extend_from_slice(&(e.layers.len() as u32).to_le_bytes());
    for l in &e.layers {
        match l.encode() {
            Ok(mut b) => out.append(&mut b),
            Err(_) => return, // never poison: a half-codec drops the entry
        }
    }
    let tmp = dir.join(format!(".{name}.tmp"));
    if std::fs::write(&tmp, &out).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    lru_cap_dir(&dir);
}

/// Keep at most `SPILL_MAX_FILES` spill files, evicting the oldest mtime.
fn lru_cap_dir(dir: &std::path::Path) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .map(|p| p.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("bin"))
        .filter_map(|p| {
            let m = std::fs::metadata(&p).ok()?.modified().ok()?;
            Some((m, p))
        })
        .collect();
    if files.len() <= SPILL_MAX_FILES {
        return;
    }
    files.sort();
    let excess = files.len() - SPILL_MAX_FILES;
    for (_, p) in files.into_iter().take(excess) {
        let _ = std::fs::remove_file(p);
    }
}

/// Unlink every spill file recorded under a law other than `law`.
fn prune_spill_foreign(law: u64) {
    let Some(dir) = spill_dir() else { return };
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return;
    };
    for p in rd.flatten() {
        let path = p.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        if let Some((file_law, _)) = spill_header(&path) {
            if file_law != law {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// Load a spill entry and verify it still matches `prompt[..n]` exactly
/// (the mtime the scan saw could predate an unlink in flight).
fn load_spill_entry(
    path: &std::path::Path,
    prompt: &[u32],
    n: usize,
    law: u64,
) -> Option<PrefixEntry> {
    let b = std::fs::read(path).ok()?;
    if b.len() < SPILL_HEADER || b[..4] != SPILL_MAGIC {
        return None;
    }
    let file_law = u64::from_le_bytes(b[4..12].try_into().ok()?);
    let ntok = u32::from_le_bytes(b[12..SPILL_HEADER].try_into().ok()?) as usize;
    if file_law != law || ntok != n {
        return None;
    }
    let mut off = SPILL_HEADER;
    let mut tokens = Vec::with_capacity(ntok);
    for _ in 0..ntok {
        tokens.push(u32::from_le_bytes(b[off..off + 4].try_into().ok()?));
        off += 4;
    }
    if prompt[..n] != tokens[..] {
        return None;
    }
    let nlayers = u32::from_le_bytes(b[off..off + 4].try_into().ok()?) as usize;
    off += 4;
    let mut cur: &[u8] = &b[off..];
    let mut layers = Vec::with_capacity(nlayers);
    for _ in 0..nlayers {
        let l = LayerState::decode(&mut cur).ok()?;
        layers.push(l);
    }
    Some(PrefixEntry {
        tokens,
        law,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tokens: Vec<u32>, law: u64) -> PrefixEntry {
        PrefixEntry {
            tokens,
            law,
            layers: Vec::new(),
        }
    }

    /// O2: states computed under a different numerical law must never be served,
    /// even when the token prefix matches exactly.
    #[test]
    fn law_binding_rejects_foreign_states() {
        let mut c = PrefixCache::new(4);
        let law = lisa_mlx::runtime::kernel_law();
        let toks: Vec<u32> = (0..MIN_PREFIX_TOKENS as u32).collect();
        let mut prompt = toks.clone();
        prompt.push(999);

        // Foreign law, identical tokens => rejected.
        c.entries.push(entry(toks.clone(), law ^ 0xdead_beef));
        assert_eq!(c.lookup(&prompt), None);

        // Current law => served, and the foreign entry is pruned on insert.
        c.entries.push(entry(toks.clone(), law));
        assert_eq!(c.lookup(&prompt), Some(MIN_PREFIX_TOKENS));
    }

    /// The content hash is stable and token-sensitive (the spill key).
    #[test]
    fn entry_hash_is_content_keyed() {
        let law = 7;
        assert_eq!(entry_hash(law, &[1, 2, 3]), entry_hash(law, &[1, 2, 3]));
        assert_ne!(entry_hash(law, &[1, 2, 3]), entry_hash(law, &[1, 2, 4]));
        assert_ne!(entry_hash(law, &[1, 2, 3]), entry_hash(law + 1, &[1, 2, 3]));
    }

    /// Spill on eviction, restore on lookup, RAM LRU re-warm — the tier
    /// mechanics with empty-layer entries (the array codec is exercised by
    /// the encode/decode test in cache.rs). Files it creates are removed.
    #[test]
    fn spill_round_trip_restores_evicted_entry() {
        let Some(dir) = spill_dir() else { return };
        let law = lisa_mlx::runtime::kernel_law();
        // Distinctive ids: no collision with a real serving process.
        let a: Vec<u32> = (0xc0de_0000..0xc0de_0000 + MIN_PREFIX_TOKENS as u32).collect();
        let b: Vec<u32> = (0xc0de_1000..0xc0de_1000 + MIN_PREFIX_TOKENS as u32).collect();
        let hash = entry_hash(law, &a);
        let mut c = PrefixCache::new(1);
        c.rewarm(PrefixEntry {
            tokens: a.clone(),
            law,
            layers: Vec::new(),
        });
        // The second admission evicts `a` -> spilled.
        c.rewarm(PrefixEntry {
            tokens: b.clone(),
            law,
            layers: Vec::new(),
        });
        let mut pa = a.clone();
        pa.push(999);
        assert_eq!(c.lookup_ram(&pa, law), None, "a was evicted from RAM");
        assert_eq!(c.lookup(&pa), Some(a.len()), "disk match fills the gap");

        let (n, path) = best_disk_match(&pa, law, None).expect("spill file exists");
        assert_eq!(n, a.len());
        let e = load_spill_entry(&path, &pa, n, law).expect("load");
        assert_eq!(e.tokens, a);
        c.rewarm(e);
        assert_eq!(c.lookup_ram(&pa, law), Some(a.len()), "re-warmed into RAM");

        let _ = std::fs::remove_file(dir.join(format!("p{hash:016x}.bin")));
        let _ = std::fs::remove_file(dir.join(format!("p{:016x}.bin", entry_hash(law, &b))));
        // Foreign-law pruning propagates the RAM invalidation.
        prune_spill_foreign(law ^ 1);
        assert!(best_disk_match(&pa, law, None).is_none());
        prune_spill_foreign(law);
    }
}
