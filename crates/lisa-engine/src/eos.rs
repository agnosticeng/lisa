//! Shared generation stop policy.
use crate::language::LanguageModel;
use std::sync::RwLock;

static EOS_IDS: RwLock<Vec<u32>> = RwLock::new(Vec::new());

/// Overwrites the process-wide EOS set. Call order matters (e.g. `run` arms
/// `im_end` first, then `--ignore-eos` clears it) — the last call wins, unlike
/// the previous OnceLock whose first write silently stuck (the 199/204-token
/// ignore_eos bug).
pub fn set_eos_ids(ids: Vec<u32>) {
    *EOS_IDS.write().unwrap() = ids;
}

/// Temporarily suppress EOS for one serialized generation job. Restores the
/// previous set on every exit, including failures before or during decode.
/// The caller must not overlap this scope with another generation job.
pub fn ignore_eos_scope() -> IgnoreEosScope {
    let previous = std::mem::take(&mut *EOS_IDS.write().unwrap());
    IgnoreEosScope { previous }
}

#[must_use = "dropping the scope restores EOS handling"]
pub struct IgnoreEosScope {
    previous: Vec<u32>,
}

impl Drop for IgnoreEosScope {
    fn drop(&mut self) {
        set_eos_ids(std::mem::take(&mut self.previous));
    }
}

/// Configure tokenizer stop ids together with any additional model-declared
/// stop ids. The caller owns the process-wide EOS scope, so ignore-EOS paths
/// can still explicitly clear it with `set_eos_ids(Vec::new())`.
pub fn set_eos_ids_for_model<R: crate::language::LanguageRuntime>(
    ids: Vec<u32>,
    tower: &dyn LanguageModel<R>,
) {
    let mut all = ids;
    for id in tower
        .eos_token_ids()
        .into_iter()
        .filter_map(|id| u32::try_from(id).ok())
    {
        if !all.contains(&id) {
            all.push(id);
        }
    }
    set_eos_ids(all);
}

/// The current stop tokens.
pub fn eos_ids() -> Vec<u32> {
    EOS_IDS.read().unwrap().clone()
}

pub fn is_eos(id: u32) -> bool {
    EOS_IDS.read().unwrap().contains(&id)
}
