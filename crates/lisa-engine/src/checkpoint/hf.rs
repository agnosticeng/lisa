//! Hugging Face downloads through the official Rust Hub client.
//!
//! The client owns HTTP transfers, retries, resumption and the standard
//! Hub cache. A snapshot remains hidden from Lisa discovery until every selected
//! file is present with its advertised size, including after an interrupted run.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, bail};
use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};

const INCOMPLETE_MARKER: &str = ".lisa-incomplete";

static APPLICATION_CACHE_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Set a sandbox-owned Hub cache before starting application workers. This
/// avoids changing process environment variables after Swift has spawned threads.
/// Explicit application configuration takes precedence over desktop environment
/// defaults and remains fixed for the process lifetime.
pub fn configure_cache_root(root: PathBuf) -> Result<()> {
    anyhow::ensure!(root.is_absolute(), "Hub cache path must be absolute");
    std::fs::create_dir_all(&root)
        .with_context(|| format!("creating application model cache {}", root.display()))?;
    APPLICATION_CACHE_ROOT
        .set(root)
        .map_err(|_| anyhow::anyhow!("Application model cache is already configured"))
}

/// Use the same cache root for Hub downloads and local discovery.
pub fn hub_root() -> PathBuf {
    if let Some(root) = APPLICATION_CACHE_ROOT.get() {
        return root.clone();
    }
    std::env::var_os("HF_HUB_CACHE")
        .or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("hub"))
}

/// Whether a snapshot has an entry point the resolver can recognize.
/// Tensor completeness and compatibility are checked when the model is opened.
pub fn is_model_dir(dir: &Path) -> bool {
    if dir.join(INCOMPLETE_MARKER).exists() {
        return false;
    }
    dir.join("config.json").is_file()
        || dir.join("encoder/config.json").is_file()
        || crate::checkpoint::catalog::is_flux2(dir)
        || dir
            .join(crate::checkpoint::catalog::deepseek41::GGUF_ENTRY)
            .is_file()
        || dir.join(crate::checkpoint::catalog::glm53::ENTRY).is_file()
        || dir
            .join(crate::checkpoint::catalog::qwen35::ENTRY)
            .is_file()
}

/// Resolve the explicit GGUF entry rather than guessing among quantizations,
/// Engram tables and vision packages. Ordinary checkpoint directories pass through.
pub fn checkpoint_path(dir: PathBuf) -> PathBuf {
    if dir.is_file() {
        return dir;
    }
    if !dir.join("config.json").exists() {
        for name in [
            crate::checkpoint::catalog::deepseek41::GGUF_ENTRY,
            crate::checkpoint::catalog::glm53::ENTRY,
            crate::checkpoint::catalog::qwen35::ENTRY,
        ] {
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

fn hf_home() -> PathBuf {
    std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
                        .join(".cache")
                })
                .join("huggingface")
        })
}

fn token() -> Option<String> {
    if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    }) {
        return None;
    }
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(key)
            && !token.trim().is_empty()
        {
            return Some(token.trim().into());
        }
    }
    let path = std::env::var_os("HF_TOKEN_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("token"));
    std::fs::read_to_string(path)
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn client(anonymous: bool) -> Result<Api> {
    ApiBuilder::from_env()
        .with_cache_dir(hub_root())
        .with_progress(false)
        .with_retries(3)
        .with_token(if anonymous { None } else { token() })
        .build()
        .context("creating Hugging Face client")
}

fn connect(repo: &str, make_client: impl Fn(bool) -> Result<Api>) -> Result<(Api, String)> {
    anyhow::ensure!(is_repo_id(repo), "invalid Hub repository ID {repo:?}");
    let authenticated = make_client(false)?;
    let revision = match authenticated.model(repo.into()).info_request().call() {
        Ok(response) => response.into_json::<hf_hub::api::RepoInfo>()?.sha,
        Err(error) => {
            let detail = error.to_string();
            if error
                .into_response()
                .is_some_and(|response| response.status() == 401)
            {
                // Expired credentials must not block a public checkpoint.
                let anonymous = make_client(true)?;
                if let Ok(info) = anonymous.model(repo.into()).info() {
                    return Ok((anonymous, info.sha));
                }
                bail!("Hugging Face login rejected for {repo}; refresh your token: {detail}");
            }
            bail!("reading {repo} metadata: {detail}");
        }
    };
    Ok((authenticated, revision))
}

#[derive(serde::Deserialize)]
struct RemoteFile {
    #[serde(rename = "rfilename")]
    path: String,
    size: u64,
}

fn download_files(repo: &str, remote: &ApiRepo, rev: &str) -> Result<Vec<RemoteFile>> {
    #[derive(serde::Deserialize)]
    struct Manifest {
        siblings: Vec<RemoteFile>,
    }
    // The flat revision manifest includes nested paths without tree pagination.
    let response = remote
        .info_request()
        .query("blobs", "true")
        .call()
        .with_context(|| format!("listing {repo}@{rev}"))?;
    let manifest: Manifest = response.into_json().context("reading Hub file metadata")?;
    let mut files = Vec::new();
    for file in manifest.siblings {
        anyhow::ensure!(
            !file.path.is_empty()
                && file.path != INCOMPLETE_MARKER
                && Path::new(&file.path)
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))),
            "invalid Hub file path {:?}",
            file.path
        );
        if include_download_file(repo, &file.path) {
            files.push(file);
        }
    }
    anyhow::ensure!(!files.is_empty(), "{repo}@{rev} has no downloadable files");
    Ok(files)
}

fn pinned_repo(client: &Api, repo: &str, revision: &str) -> ApiRepo {
    client.repo(Repo::with_revision(
        repo.into(),
        RepoType::Model,
        revision.into(),
    ))
}

fn package_size(files: &[RemoteFile]) -> Result<u64> {
    files.iter().try_fold(0u64, |total, file| {
        total
            .checked_add(file.size)
            .context("Hub package size exceeds u64")
    })
}

/// Optional MTP sidecars can be omitted when an external drafter replaces them.
/// Embedded tensors remain part of the original target checkpoint files.
#[derive(Clone, Copy, Default)]
pub enum DownloadProfile {
    #[default]
    Complete,
    WithoutMtp,
}

impl DownloadProfile {
    fn select(self, files: &mut Vec<RemoteFile>) {
        if matches!(self, Self::WithoutMtp) {
            files.retain(|file| !file.path.starts_with("mtp/"));
        }
    }
}

/// Complete selected package size, including nested tokenizer and auxiliary weights.
pub fn download_size(repo: &str) -> Result<u64> {
    download_size_with_profile(repo, DownloadProfile::Complete)
}

pub fn download_size_with_profile(repo: &str, profile: DownloadProfile) -> Result<u64> {
    let (client, revision) = connect(repo, client)?;
    let mut files = download_files(repo, &pinned_repo(&client, repo, &revision), &revision)?;
    profile.select(&mut files);
    package_size(&files)
}

fn include_download_file(repo: &str, path: &str) -> bool {
    if repo == crate::checkpoint::catalog::qwen35::REPO {
        return path == crate::checkpoint::catalog::qwen35::ENTRY
            || matches!(
                path,
                "README.md" | "LICENSE" | "LICENSE.md" | "NOTICE" | "NOTICE.md"
            );
    }
    if repo == crate::checkpoint::catalog::glm53::REPO {
        return path == crate::checkpoint::catalog::glm53::ENTRY
            || matches!(
                path,
                "README.md" | "LICENSE" | "LICENSE.md" | "NOTICE" | "NOTICE.md"
            );
    }
    if repo == crate::checkpoint::catalog::deepseek41::REPO {
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
    repo != crate::checkpoint::catalog::flux3::REPO
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

struct FileProgress {
    callback: Arc<dyn Fn(u64, u64) + Send + Sync>,
    completed: u64,
    size: u64,
    total: u64,
    transferred: u64,
    reported: u64,
}

impl hf_hub::api::Progress for FileProgress {
    fn init(&mut self, _size: usize, _filename: &str) {
        self.transferred = 0;
    }
    fn update(&mut self, bytes: usize) {
        self.transferred = self.transferred.saturating_add(bytes as u64).min(self.size);
        if self.transferred > self.reported {
            self.reported = self.transferred;
            (self.callback)(self.completed + self.reported, self.total);
        }
    }
    fn finish(&mut self) {}
}

/// Download a selected package into the standard Hub cache.
pub fn download_repo(repo: &str) -> Result<PathBuf> {
    download_repo_with_progress(repo, |_, _| {})
}

/// The callback reports completed bytes across the selected package, including
/// cache hits. Discovery stays disabled until the complete package is verified.
pub fn download_repo_with_progress(
    repo: &str,
    progress: impl Fn(u64, u64) + Send + Sync + 'static,
) -> Result<PathBuf> {
    download_repo_with_profile(repo, DownloadProfile::Complete, progress)
}

pub fn download_repo_with_profile(
    repo: &str,
    profile: DownloadProfile,
    progress: impl Fn(u64, u64) + Send + Sync + 'static,
) -> Result<PathBuf> {
    let (client, revision) = connect(repo, client)?;
    download_with_client_profile(
        repo,
        &client,
        &hub_root(),
        &revision,
        profile,
        Arc::new(progress),
    )
}

fn file_complete(path: &Path, expected: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() == expected)
}

#[cfg(test)]
fn download_with_client(
    repo: &str,
    client: &Api,
    cache_root: &Path,
    rev: &str,
    progress: Arc<dyn Fn(u64, u64) + Send + Sync>,
) -> Result<PathBuf> {
    download_with_client_profile(
        repo,
        client,
        cache_root,
        rev,
        DownloadProfile::Complete,
        progress,
    )
}

fn download_with_client_profile(
    repo: &str,
    client: &Api,
    cache_root: &Path,
    rev: &str,
    profile: DownloadProfile,
    progress: Arc<dyn Fn(u64, u64) + Send + Sync>,
) -> Result<PathBuf> {
    if !is_repo_id(repo) {
        bail!("{repo:?} is not a Hugging Face repo id (expected `org/name`)");
    }
    let remote = pinned_repo(client, repo, rev);
    anyhow::ensure!(
        rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid Hub revision {rev:?}"
    );
    let mut files = download_files(repo, &remote, rev)?;
    profile.select(&mut files);
    let total = package_size(&files)?;
    // Use the configured client's cache path so tests and explicit endpoints
    // exercise exactly the same download/discovery boundary as production.
    let root = cache_root.join(format!("models--{}", repo.replace('/', "--")));
    let snap = root.join("snapshots").join(rev);
    std::fs::create_dir_all(&snap)
        .with_context(|| format!("creating model snapshot {}", snap.display()))?;
    let marker = snap.join(INCOMPLETE_MARKER);
    std::fs::write(
        &marker,
        b"Download in progress; retry to complete this snapshot.\n",
    )
    .with_context(|| format!("marking incomplete snapshot {}", snap.display()))?;
    let mut completed = 0;
    progress(0, total);
    for file in &files {
        let destination = snap.join(&file.path);
        if !file_complete(&destination, file.size) {
            // Download bypasses the client's existing snapshot fast path, so a
            // truncated final file is replaced; `.part` files still resume.
            let downloaded = remote
                .download_with_progress(
                    &file.path,
                    FileProgress {
                        callback: progress.clone(),
                        completed,
                        size: file.size,
                        total,
                        transferred: 0,
                        reported: 0,
                    },
                )
                .with_context(|| format!("downloading {} from {repo}", file.path))?;
            anyhow::ensure!(
                downloaded == destination,
                "Hub returned an unexpected snapshot path"
            );
        }
        anyhow::ensure!(
            file_complete(&destination, file.size),
            "{}: expected {} bytes after download; retry to repair the cache",
            file.path,
            file.size
        );
        completed += file.size;
        progress(completed, total);
    }
    std::fs::create_dir_all(root.join("refs"))?;
    std::fs::write(root.join("refs/main"), rev.as_bytes())?;
    std::fs::remove_file(marker)?;
    Ok(snap)
}

#[cfg(test)]
mod tests {
    #[test]
    fn qwen_download_selects_exact_quant_without_extra_draft_or_vision() {
        let repo = crate::checkpoint::catalog::qwen35::REPO;
        assert!(include_download_file(
            repo,
            crate::checkpoint::catalog::qwen35::ENTRY
        ));
        for path in [
            "Qwen3.8-27B-UD-Q3_K_XL.gguf",
            "mmproj-F16.gguf",
            "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
            "config.json",
            "tokenizer.json",
        ] {
            assert!(!include_download_file(repo, path), "{path}");
        }
    }

    #[test]
    fn glm53_download_selects_only_ds4_q2_package() {
        let repo = crate::checkpoint::catalog::glm53::REPO;
        assert!(include_download_file(
            repo,
            crate::checkpoint::catalog::glm53::ENTRY
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
            crate::checkpoint::catalog::deepseek41::GGUF_ENTRY,
            "IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00005-of-00006.gguf",
            "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",
            "README.md",
            "chat_template.jinja",
            "expert-selection/kept-experts.json",
        ] {
            assert!(
                include_download_file(crate::checkpoint::catalog::deepseek41::REPO, path),
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
                !include_download_file(crate::checkpoint::catalog::deepseek41::REPO, path),
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
                include_download_file(crate::checkpoint::catalog::flux3::REPO, path),
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
                !include_download_file(crate::checkpoint::catalog::flux3::REPO, path),
                "{path}"
            );
            assert!(
                include_download_file("some/other-repo", path),
                "other repositories must retain all files"
            );
        }
    }
}

#[cfg(test)]
#[path = "hf_download_tests.rs"]
mod download_tests;
