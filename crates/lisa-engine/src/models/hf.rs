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

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// The Hugging Face hub cache root: `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`.
pub fn hub_root() -> PathBuf {
    match std::env::var_os("HF_HOME") {
        Some(h) => PathBuf::from(h).join("hub"),
        None => home_dir().join(".cache").join("huggingface").join("hub"),
    }
}

/// Whether `dir` looks like a loaded model directory: a root `config.json` or a
/// configless checkpoint's `encoder/config.json` (e.g. Laya).
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file() || dir.join("encoder/config.json").is_file()
}

/// The cache directory for one repo (`models--org--name`).
pub fn repo_dir(repo: &str) -> PathBuf {
    hub_root()
        .join(format!("models--{}", repo.replace('/', "--")))
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
    let bytes = curl(&[url.to_string()])?;
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
    let url = format!(
        "https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&expand=true"
    );
    let tree = get_json(&url)?;
    let arr = tree
        .as_array()
        .with_context(|| format!("unexpected Hub tree response for {repo}"))?;
    let mut files = Vec::new();
    for e in arr {
        if e.get("type").and_then(|t| t.as_str()) != Some("file") {
            continue;
        }
        let Some(path) = e.get("path").and_then(|p| p.as_str()).map(str::to_string) else {
            continue;
        };
        let (size, hash) = match e.get("lfs") {
            Some(l) => (
                l.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                l.get("oid").and_then(|o| o.as_str()).map(str::to_string),
            ),
            None => (
                e.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                None,
            ),
        };
        let hash = hash.or_else(|| e.get("oid").and_then(|o| o.as_str()).map(str::to_string));
        let Some(hash) = hash else {
            // A file with no content hash cannot be addressed in the blob
            // store; skip it (the API always provides one).
            continue;
        };
        files.push(RemoteFile { path, size, hash });
    }
    Ok(files)
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
    if !repo.contains('/') {
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

    let root = repo_dir(repo);
    let snap = root.join("snapshots").join(&rev);
    if is_model_dir(&snap) {
        return Ok(snap);
    }

    let files = list_files(repo, &rev)?;
    if files.is_empty() {
        bail!("{repo}@{rev} has no downloadable files");
    }

    let blobs = root.join("blobs");
    let partial = root.join("snapshots").join(format!("{rev}.lisa-partial"));
    std::fs::create_dir_all(&blobs)?;
    std::fs::create_dir_all(&partial)?;

    for file in &files {
        let blob = blobs.join(&file.hash);
        let blob_ok = file.size > 0
            && std::fs::metadata(&blob)
                .map(|m| m.len() == file.size)
                .unwrap_or(false);
        if !blob_ok {
            let incomplete = blobs.join(format!("{}.incomplete", file.hash));
            download_file(repo, &rev, file, &incomplete)?;
            if file.size > 0 {
                let got = std::fs::metadata(&incomplete)
                    .map(|m| m.len())
                    .unwrap_or(0);
                if got != file.size {
                    bail!(
                        "{}: expected {} bytes, got {got} — retry the download",
                        file.path,
                        file.size
                    );
                }
            }
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

    std::fs::create_dir_all(root.join("refs"))?;
    std::fs::write(root.join("refs").join("main"), rev.as_bytes())?;
    if snap.is_dir() {
        let _ = std::fs::remove_dir_all(&partial);
    } else {
        std::fs::rename(&partial, &snap)?;
    }
    Ok(snap)
}
