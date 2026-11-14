//! Local conversation records. GPU sessions are rebuilt only when a chat resumes.
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Turn {
    pub user: String,
    pub assistant: String,
    #[serde(default)]
    pub images: Vec<PathBuf>,
    #[serde(default)]
    pub generated_image: Option<PathBuf>,
    /// Only successful turns may be replayed into the model's context.
    #[serde(default)]
    pub complete: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Chat {
    pub id: String,
    pub model: Option<String>,
    pub system: String,
    pub updated: u64,
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub draft: String,
}

impl Chat {
    pub fn new(model: Option<String>, system: String) -> Self {
        Self {
            id: format!("{}-{}", timestamp(), std::process::id()),
            model,
            system,
            updated: timestamp(),
            turns: Vec::new(),
            draft: String::new(),
        }
    }

    pub fn title(&self) -> String {
        let text = self
            .turns
            .first()
            .map(|t| t.user.as_str())
            .unwrap_or(&self.draft);
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            return "Image conversation".into();
        }
        let mut title: String = text.chars().take(60).collect();
        if text.chars().count() > 60 {
            title.push('…');
        }
        title
    }
}

pub fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[derive(Debug)]
pub struct Store {
    root: Option<PathBuf>,
}

impl Store {
    pub fn new() -> Self {
        Self {
            root: std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join("Library/Application Support/Lisa/chats")),
        }
    }

    pub fn at(root: PathBuf) -> Self {
        Self { root: Some(root) }
    }

    fn root(&self) -> Result<&Path> {
        self.root
            .as_deref()
            .context("HOME is not set; chats cannot be saved")
    }

    /// iOS can change the application container UUID during an app update.
    /// Repair only missing paths belonging to this app's own attachment stores.
    #[cfg(target_os = "ios")]
    fn relocate_owned_images(&self, mut chat: Chat) -> Chat {
        let Some(root) = &self.root else { return chat };
        for turn in &mut chat.turns {
            for path in &mut turn.images {
                if let Some(suffix) =
                    owned_suffix(path, "/Library/Application Support/Lisa/chats/attachments/")
                {
                    if suffix
                        .components()
                        .next()
                        .is_some_and(|part| part.as_os_str() == std::ffi::OsStr::new(&chat.id))
                    {
                        let candidate = root.join("attachments").join(suffix);
                        if candidate.is_file() {
                            *path = candidate;
                        }
                    }
                }
            }
            if let Some(path) = &mut turn.generated_image {
                if let Some(suffix) = owned_suffix(path, "/Pictures/Lisa/") {
                    if let Some(home) = std::env::var_os("HOME") {
                        let candidate = PathBuf::from(home).join("Pictures/Lisa").join(suffix);
                        if candidate.is_file() {
                            *path = candidate;
                        }
                    }
                }
            }
        }
        chat
    }

    /// Leave unreadable records untouched and report them instead of overwriting them.
    pub fn load(&self) -> (Vec<Chat>, Vec<String>) {
        let mut chats = Vec::new();
        let mut errors = Vec::new();
        let result = (|| -> Result<()> {
            let root = self.root()?;
            if !root.exists() {
                return Ok(());
            }
            for entry in fs::read_dir(root)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_none_or(|ext| ext != "json") {
                    continue;
                }
                let record = (|| -> Result<Chat> {
                    let chat: Chat = serde_json::from_slice(&fs::read(&path)?)?;
                    ensure!(valid_id(&chat.id), "invalid chat id");
                    ensure!(
                        path.file_stem().and_then(|s| s.to_str()) == Some(&chat.id),
                        "chat id does not match its file"
                    );
                    #[cfg(target_os = "ios")]
                    let chat = self.relocate_owned_images(chat);
                    Ok(chat)
                })();
                match record {
                    Ok(chat) => chats.push(chat),
                    Err(error) => {
                        errors.push(format!("Could not read {}: {error}", path.display()))
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            errors.push(format!("Could not read saved chats: {error}"));
        }
        chats.sort_by_key(|chat| std::cmp::Reverse(chat.updated));
        (chats, errors)
    }

    pub fn save(&self, chat: &Chat) -> Result<()> {
        ensure!(valid_id(&chat.id), "invalid chat id");
        let root = self.root()?;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(root)?;
        let target = root.join(format!("{}.json", chat.id));
        let temporary = root.join(format!(".{}-{}.tmp", chat.id, timestamp()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary)?;
            file.write_all(&serde_json::to_vec(chat)?)?;
            file.sync_all()?;
            fs::rename(&temporary, &target)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    /// Remove the record first so it cannot reappear after a restart. Attachment
    /// cleanup failures are warnings: the conversation has already been deleted.
    pub fn delete(&self, id: &str) -> Result<Option<String>> {
        ensure!(valid_id(id), "invalid chat id");
        let root = self.root()?;
        match fs::remove_file(root.join(format!("{id}.json"))) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("Could not delete the saved chat"),
        }
        // Only owned attachment copies are removed. Generated images and the
        // user's original input files live outside this directory.
        match fs::remove_dir_all(root.join("attachments").join(id)) {
            Ok(()) => Ok(None),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Ok(Some(format!(
                "Could not remove attachments for {id}: {error}"
            ))),
        }
    }

    /// Keep input images with the conversation even when the original is moved.
    pub fn keep_images(&self, chat_id: &str, images: &[PathBuf]) -> Result<Vec<PathBuf>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        ensure!(valid_id(chat_id), "invalid chat id");
        let directory = self.root()?.join("attachments").join(chat_id);
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&directory)?;
        let stamp = timestamp();
        let mut saved = Vec::new();
        for (index, path) in images.iter().enumerate() {
            let result = (|| -> Result<PathBuf> {
                ensure!(
                    fs::metadata(path)?.len() <= 20 * 1024 * 1024,
                    "An attachment exceeds 20 MiB"
                );
                let bytes =
                    fs::read(path).with_context(|| format!("Could not read {}", path.display()))?;
                ensure!(
                    bytes.len() <= 20 * 1024 * 1024,
                    "An attachment exceeds 20 MiB"
                );
                let extension = path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("png");
                let destination = directory.join(format!("{stamp}-{index}.{extension}"));
                let mut output = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&destination)?;
                if let Err(error) = output.write_all(&bytes).and_then(|_| output.sync_all()) {
                    let _ = fs::remove_file(&destination);
                    return Err(error.into());
                }
                Ok(destination)
            })();
            match result {
                Ok(path) => saved.push(path),
                Err(error) => {
                    for path in saved {
                        let _ = fs::remove_file(path);
                    }
                    return Err(error);
                }
            }
        }
        Ok(saved)
    }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit() || b == b'-')
}

#[cfg(any(test, target_os = "ios"))]
fn owned_suffix<'a>(path: &'a Path, marker: &str) -> Option<&'a Path> {
    if path.is_file() {
        return None;
    }
    let (_, suffix) = path.to_str()?.split_once(marker)?;
    let suffix = Path::new(suffix);
    (!suffix.as_os_str().is_empty()
        && suffix
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_))))
    .then_some(suffix)
}

#[cfg(test)]
mod container_tests {
    use super::*;
    #[test]
    fn relocation_only_accepts_owned_relative_components() {
        let marker = "/Library/Application Support/Lisa/chats/attachments/";
        assert_eq!(
            owned_suffix(
                Path::new(
                    "/old/Library/Application Support/Lisa/chats/attachments/123-4/photo.png"
                ),
                marker
            ),
            Some(Path::new("123-4/photo.png"))
        );
        assert!(
            owned_suffix(
                Path::new(
                    "/old/Library/Application Support/Lisa/chats/attachments/123-4/../../outside"
                ),
                marker
            )
            .is_none()
        );
        assert!(owned_suffix(Path::new("/private/photo.png"), marker).is_none());
    }
}
