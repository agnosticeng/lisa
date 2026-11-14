//! SGLang SPS policy adapted to Metal's single-request DSpine execution.
//! SPS caps target verification independently of drafting. Untuned Metal keeps
//! the established seven-row snapshot schedule; profiles can select block 16
//! and compact replay. Profiles never cross device/checkpoint pairs.
use crate::{bindings::BindingDevice, execution::DeviceContext};
use anyhow::Context;
use lisa_backend::ManagedDevice;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub use lisa_models::qwen3_5::dspine::policy::{Identity, Policy, SsmMode};

fn checkpoint_identity(path: &Path) -> anyhow::Result<String> {
    let mut hash = Sha256::new();
    hash.update(std::fs::read(path.join("config.json"))?);
    // Bind immutable Hub revisions and local checkpoint file identities without
    // hashing tens of GB on every load. Local file replacement invalidates SPS.
    hash.update(path.canonicalize()?.to_string_lossy().as_bytes());
    let mut files: Vec<_> = std::fs::read_dir(path)?.collect::<Result<_, _>>()?;
    files.sort_by_key(|entry| entry.file_name());
    for entry in files {
        if entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "safetensors")
        {
            let meta = entry.metadata()?;
            hash.update(entry.file_name().to_string_lossy().as_bytes());
            hash.update(meta.len().to_le_bytes());
            hash.update(
                meta.modified()?
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
                    .to_le_bytes(),
            );
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn detect<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    target: &Path,
    draft: &Path,
) -> anyhow::Result<Identity> {
    let device = context.device().identity();
    Ok(Identity {
        schedule: match (
            std::env::var("LISA_QWEN_GDN_VERIFY_FUSION").as_deref() == Ok("0"),
            std::env::var("LISA_QWEN_SHARED_KV_VERIFY").as_deref() != Ok("1"),
        ) {
            (false, false) => "lisa-dspine-metal-verify-v9",
            (true, false) => "lisa-dspine-metal-verify-v9-unfused",
            (false, true) => "lisa-dspine-metal-verify-v9-unshared-kv",
            (true, true) => "lisa-dspine-metal-verify-v9-unfused-unshared-kv",
        }
        .replace("-metal-", &format!("-{}-", device.provider))
            + if super::dspine::full_tables_enabled()? {
                "-full-tables"
            } else {
                ""
            }
            + if super::dspine_retrieval::enabled()? {
                "-coarse2"
            } else {
                ""
            },
        gpu: device.device,
        architecture: device.architecture,
        memory_bytes: crate::memory::physical_ram(),
        target: checkpoint_identity(target)?,
        draft: checkpoint_identity(draft)?,
    })
}
pub fn cache_path(identity: &Identity) -> anyhow::Result<PathBuf> {
    let key = format!("{:x}", Sha256::digest(serde_json::to_vec(identity)?));
    let home = std::env::var_os("HOME").context("HOME unavailable for DSpine SPS profile")?;
    Ok(PathBuf::from(home)
        .join(".cache/lisa/dspine-sps")
        .join(format!("{key}.json")))
}

pub fn load<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    target: &Path,
    draft: &Path,
) -> anyhow::Result<(usize, usize, SsmMode)> {
    let identity = detect(context, target, draft)?;
    let explicit = std::env::var_os("LISA_DSPINE_SPS_PROFILE");
    let path = explicit
        .clone()
        .map(PathBuf::from)
        .map_or_else(|| cache_path(&identity), Ok)?;
    if explicit.is_none() && !path.exists() {
        return Ok((7, 7, SsmMode::Capture));
    }
    let policy: Policy = serde_json::from_slice(
        &std::fs::read(&path)
            .with_context(|| format!("reading DSpine SPS profile {}", path.display()))?,
    )?;
    policy.validate(&identity)?;
    eprintln!(
        "[dspine] measured SPS verify width {} (draft block {}, SSM {:?})",
        policy.verify_width, policy.draft_width, policy.ssm
    );
    Ok((policy.draft_width, policy.verify_width, policy.ssm))
}
