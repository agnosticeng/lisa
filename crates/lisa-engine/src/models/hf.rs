//! Minimal Hugging Face Hub downloader.
//!
//! Lists a repo's files through the Hub API and fetches each with `curl`
//! (present on macOS), writing the standard hub cache layout
//! (`blobs/<hash>` + `snapshots/<rev>/…` symlinks) so downloaded and
//! `huggingface_hub`-managed caches stay interchangeable. No
//! `huggingface_hub` dependency.
//!
//! Downloads are resumable: partial files live in `blobs/<hash>.incomplete`
//! (counted by the UI's byte-progress meter) and are renamed once complete. A
//! snapshot is only published under its final `snapshots/<rev>` name after
//! every file lands, so an interrupted download is never mistaken for a cached
//! model.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// The Hugging Face hub cache root: `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`.
pub fn hub_root() -> PathBuf {
    match std::env::var_os("HF_HOME") {
        Some(h) => PathBuf::from(h).join("hub"),
        None => home_dir().join(".cache").join("huggingface").join("hub"),
    }
}

/// Whether a snapshot has an entry point the resolver can recognize.
/// Tensor completeness and compatibility are checked when the model is opened.
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file()
        || dir.join("encoder/config.json").is_file()
        || super::flux2::is_checkpoint(dir)
        || dir.join(super::deepseek41::GGUF_ENTRY).is_file()
        || dir.join(super::glm53::gguf::ENTRY).is_file()
}

/// Resolve the explicit GGUF entry rather than guessing among quantizations,
/// Engram tables and vision packages. Ordinary checkpoint directories pass through.
pub fn checkpoint_path(dir: PathBuf) -> PathBuf {
    if dir.is_file() {
        return dir;
    }
    if !dir.join("config.json").exists() {
        for name in [super::deepseek41::GGUF_ENTRY, super::glm53::gguf::ENTRY] {
            let file = dir.join(name);
            if file.is_file() {
                return file;
            }
        }
    }
    dir
}

/// The cache directory for one repo (`models--org--name`).
pub fn repo_dir(repo: &str) -> PathBuf {
    hub_root().join(format!("models--{}", repo.replace('/', "--")))
}

/// Accept an unambiguous Hub ID, not a missing relative path or URL.
pub fn is_repo_id(repo: &str) -> bool {
    let Some((owner, name)) = repo.split_once('/') else {
        return false;
    };
    [owner, name].iter().all(|part| {
        !part.is_empty()
            && !part.starts_with(['.', '-'])
            && !part.ends_with(['.', '-'])
            && !part.contains("..")
            && !part.contains("--")
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    })
}

/// Usable local snapshots, active `refs/main` first, then newest directory.
/// CLI, desktop discovery and offline diagnostics use the same ordering.
pub fn cached_snapshots(repo: &str) -> Vec<PathBuf> {
    let root = repo_dir(repo);
    let active = std::fs::read_to_string(root.join("refs/main"))
        .ok()
        .map(|revision| revision.trim().to_owned())
        .filter(|revision| !revision.is_empty());
    let Ok(entries) = std::fs::read_dir(root.join("snapshots")) else {
        return Vec::new();
    };
    let mut snapshots: Vec<_> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            if name.ends_with(".lisa-partial") || !is_model_dir(&entry.path()) {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok();
            Some((active.as_deref() == Some(name), modified, entry.path()))
        })
        .collect();
    snapshots.sort_by(|a, b| b.cmp(a));
    snapshots.into_iter().map(|(_, _, path)| path).collect()
}

pub fn cached_snapshot(repo: &str) -> Option<PathBuf> {
    cached_snapshots(repo).into_iter().next()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// A Hub access token from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` or the
/// standard `token` file beside the cache.
fn token() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(t) = std::env::var(key) {
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    let root = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".cache").join("huggingface"));
    std::fs::read_to_string(root.join("token"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Run `curl` with the shared flags and optional auth; returns stdout bytes.
fn curl(args: &[String]) -> Result<Vec<u8>> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sSL", "--fail", "--retry", "3"]);
    if let Some(t) = token() {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    cmd.args(args);
    let out = cmd
        .output()
        .context("running `curl` (is curl installed?)")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!("curl failed: {}", err.trim());
    }
    Ok(out.stdout)
}

fn get_json(url: &str) -> Result<Value> {
    let bytes = curl(&[
        "--connect-timeout".into(),
        "10".into(),
        "--max-time".into(),
        "30".into(),
        url.to_string(),
    ])?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing Hub API JSON from {url}"))
}

/// One file to fetch: repo-relative path, expected size, and its content hash
/// (the blob name).
struct RemoteFile {
    path: String,
    size: u64,
    hash: String,
}

fn list_files(repo: &str, rev: &str) -> Result<Vec<RemoteFile>> {
    // Unlike the paginated tree endpoint, revision metadata lists every file.
    let url = format!("https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true");
    let metadata = get_json(&url)?;
    let arr = metadata
        .get("siblings")
        .and_then(Value::as_array)
        .with_context(|| format!("unexpected Hub file manifest for {repo}"))?;
    let mut files = Vec::new();
    for e in arr {
        let path = e
            .get("rfilename")
            .and_then(Value::as_str)
            .context("Hub file manifest is missing a filename")?;
        anyhow::ensure!(
            !path.is_empty()
                && Path::new(path)
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))),
            "invalid Hub file path {path:?}"
        );
        let size = e
            .get("size")
            .and_then(Value::as_u64)
            .with_context(|| format!("Hub file size is missing for {path}"))?;
        let hash = e
            .get("lfs")
            .and_then(|lfs| lfs.get("sha256"))
            .or_else(|| e.get("blobId"))
            .and_then(Value::as_str)
            .with_context(|| format!("Hub content hash is missing for {path}"))?;
        anyhow::ensure!(
            !hash.is_empty() && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid Hub content hash for {path}"
        );
        files.push(RemoteFile {
            path: path.into(),
            size,
            hash: hash.into(),
        });
    }
    Ok(files)
}

/// Complete selected package size, including nested tokenizer and auxiliary weights.
pub fn download_size(repo: &str) -> Result<u64> {
    anyhow::ensure!(is_repo_id(repo), "invalid Hub repository ID {repo:?}");
    let files = download_files(repo, "main")?;
    anyhow::ensure!(!files.is_empty(), "{repo} has no downloadable files");
    files.iter().try_fold(0u64, |total, file| {
        total
            .checked_add(file.size)
            .context("Hub package size exceeds u64")
    })
}

/// FLUX 3 publishes multiple complete tensor layouts and NVIDIA-only formats.
/// LISA consumes the native BF16 package, retaining its notices and encoder.
fn download_files(repo: &str, rev: &str) -> Result<Vec<RemoteFile>> {
    let mut files = list_files(repo, rev)?;
    files.retain(|file| include_download_file(repo, &file.path));
    Ok(files)
}

fn include_download_file(repo: &str, path: &str) -> bool {
    if repo == super::glm53::gguf::REPO {
        return path == super::glm53::gguf::ENTRY
            || matches!(
                path,
                "README.md" | "LICENSE" | "LICENSE.md" | "NOTICE" | "NOTICE.md"
            );
    }
    if repo == super::deepseek41::REPO {
        return (path.starts_with("IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-")
            && path.ends_with(".gguf"))
            || matches!(
                path,
                "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
                    | "README.md"
                    | "LICENSE"
                    | "LICENSE.md"
                    | "NOTICE"
                    | "NOTICE.md"
                    | "chat_template.jinja"
            )
            || path.starts_with("expert-selection/");
    }
    repo != super::flux3::REPO
        || path.starts_with("text_encoder/")
        || matches!(
            path,
            "config.json"
                | "manifest.json"
                | "model.safetensors"
                | "ae.safetensors"
                | "README.md"
                | "LICENSE.md"
                | "NOTICE.md"
        )
}

fn download_file(repo: &str, rev: &str, file: &RemoteFile, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let url = format!("https://huggingface.co/{repo}/resolve/{rev}/{}", file.path);
    curl(&[
        "-C".to_string(),
        "-".to_string(),
        "-o".to_string(),
        dest.display().to_string(),
        url,
    ])
    .with_context(|| format!("downloading {}", file.path))?;
    Ok(())
}

/// Download `repo` into the hub cache and return its snapshot directory.
///
/// Idempotent and resumable: complete blobs are left alone, partial ones resume
/// with a range request, and the snapshot is published atomically at the end.
pub fn download_repo(repo: &str) -> Result<PathBuf> {
    if !is_repo_id(repo) {
        bail!("{repo:?} is not a Hugging Face repo id (expected `org/name`)");
    }

    let rev = match get_json(&format!("https://huggingface.co/api/models/{repo}")) {
        Ok(v) => v
            .get("sha")
            .and_then(|s| s.as_str())
            .unwrap_or("main")
            .to_string(),
        // The metadata call can fail on a gated repo without a token; fall back
        // to the branch and let the per-file requests surface auth errors.
        Err(_) => "main".to_string(),
    };
    anyhow::ensure!(
        rev == "main" || (!rev.is_empty() && rev.bytes().all(|b| b.is_ascii_hexdigit())),
        "invalid Hub revision {rev:?}"
    );

    let root = repo_dir(repo);
    let snap = root.join("snapshots").join(&rev);
    let files = download_files(repo, &rev)?;
    if files.is_empty() {
        bail!("{repo}@{rev} has no downloadable files");
    }
    if is_model_dir(&snap)
        && files.iter().all(|file| {
            std::fs::metadata(snap.join(&file.path)).is_ok_and(|meta| meta.len() == file.size)
        })
    {
        return Ok(snap);
    }

    let blobs = root.join("blobs");
    let partial = root.join("snapshots").join(format!("{rev}.lisa-partial"));
    std::fs::create_dir_all(&blobs)?;
    std::fs::create_dir_all(&partial)?;

    for file in &files {
        let blob = blobs.join(&file.hash);
        let blob_ok =
            std::fs::metadata(&blob).is_ok_and(|meta| meta.is_file() && meta.len() == file.size);
        if !blob_ok {
            let incomplete = blobs.join(format!("{}.incomplete", file.hash));
            download_file(repo, &rev, file, &incomplete)?;
            let got = std::fs::metadata(&incomplete)?.len();
            anyhow::ensure!(
                got == file.size,
                "{}: expected {} bytes, got {got} — retry the download",
                file.path,
                file.size
            );
            std::fs::rename(&incomplete, &blob)?;
        }

        // snapshots/<rev>/<path> -> ../../blobs/<hash> (relative, so the tree
        // can be relocated).
        let link = partial.join(&file.path);
        if let Some(parent) = link.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _ = std::fs::remove_file(&link);
        let depth = file.path.matches('/').count() + 2;
        let target = format!("{}blobs/{}", "../".repeat(depth), file.hash);
        std::os::unix::fs::symlink(&target, &link)
            .with_context(|| format!("linking {}", file.path))?;
    }

    if snap.is_dir() {
        // Repair previously incomplete snapshots, preserving prepared storage.
        for file in &files {
            let destination = snap.join(&file.path);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::rename(partial.join(&file.path), destination)?;
        }
        std::fs::remove_dir_all(&partial)?;
    } else {
        std::fs::rename(&partial, &snap)?;
    }
    std::fs::create_dir_all(root.join("refs"))?;
    std::fs::write(root.join("refs").join("main"), rev.as_bytes())?;
    Ok(snap)
}

#[cfg(test)]
mod tests {
    #[test]
    fn glm53_download_selects_only_ds4_q2_package() {
        let repo = super::super::glm53::gguf::REPO;
        assert!(include_download_file(
            repo,
            super::super::glm53::gguf::ENTRY
        ));
        for file in [
            "GLM-5.3-Flash-Q4.gguf",
            "GLM-5.3-Flash-FP8.gguf",
            "GLM-5.3-Flash-Vision-Encoder.gguf",
            "image.png",
        ] {
            assert!(!include_download_file(repo, file), "{file}");
        }
    }

    use super::*;
    #[test]
    fn deepseek41_download_selects_text_shards_and_disk_engram() {
        for path in [
            super::super::deepseek41::GGUF_ENTRY,
            "IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00005-of-00006.gguf",
            "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",
            "README.md",
            "chat_template.jinja",
            "expert-selection/kept-experts.json",
        ] {
            assert!(
                include_download_file(super::super::deepseek41::REPO, path),
                "{path}"
            );
        }
        for path in [
            "mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf",
            "assets/banner.png",
            "Q4_K/model.gguf",
            "model.safetensors",
        ] {
            assert!(
                !include_download_file(super::super::deepseek41::REPO, path),
                "{path}"
            );
        }
    }
    #[test]
    fn flux3_selects_one_native_package_and_keeps_notices() {
        for path in [
            "model.safetensors",
            "ae.safetensors",
            "config.json",
            "manifest.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/tokenizer.json",
            "LICENSE.md",
            "NOTICE.md",
        ] {
            assert!(
                include_download_file(super::super::flux3::REPO, path),
                "{path}"
            );
        }
        for path in [
            "variants/nvfp4/model.safetensors",
            "variants/fp8r/model.safetensors",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "assets/example.png",
        ] {
            assert!(
                !include_download_file(super::super::flux3::REPO, path),
                "{path}"
            );
            assert!(
                include_download_file("some/other-repo", path),
                "other repositories must retain all files"
            );
        }
    }
}
