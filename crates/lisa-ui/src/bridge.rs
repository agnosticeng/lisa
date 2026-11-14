//! Serialized C ABI used by Swift. Models remain on the existing engine thread.
//! Every returned string belongs to Rust and must be released with lisa_app_free.
use crate::{
    backend::{self, Command, UiEvent},
    browser::{self, BrowserEvent, BrowserRequest},
    catalog, chats, models,
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};

struct App {
    commands: mpsc::Sender<Command>,
    events: Arc<Mutex<Vec<UiEvent>>>,
    cancel: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    store: chats::Store,
    browser: HashMap<u64, mpsc::Sender<BrowserEvent>>,
    next_browser: u64,
    server_stop: Arc<AtomicBool>,
    busy: bool,
    loaded: bool,
    downloading: bool,
    unloading: bool,
}
static APP: Mutex<Option<App>> = Mutex::new(None);

fn text<'a>(request: &'a Value, key: &str) -> Result<&'a str> {
    request
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("Missing {key}"))
}

fn model_list() -> Value {
    let local = models::discover_models();
    let budget = models::model_ram_budget();
    let mut result = Vec::new();
    for entry in catalog::MODELS {
        let found = local.iter().find(|model| model.repo_id == entry.repo_id);
        let size = found.map_or_else(|| catalog::bundle_download_bytes(entry), |model| model.size);
        let resident = found
            .and_then(|model| model.resident_bytes)
            .or(entry.resident_bytes)
            .unwrap_or(size);
        result.push(
            json!({"id":entry.repo_id,"name":entry.name,"detail":entry.description,
            "size":size,"local":found.is_some(),"path":found.map(|model| &model.path),
            "fits":budget > 0 && resident <= budget}),
        );
    }
    for model in &local {
        if catalog::find(&model.repo_id).is_none() {
            result.push(
                json!({"id":model.repo_id,"name":models::display_name(&model.repo_id),
                "detail":model.model_type,"size":model.size,"local":true,"path":model.path,
                "fits":budget > 0 && model.resident_bytes.unwrap_or(model.size) <= budget}),
            );
        }
    }
    json!({"models":result,"budget":budget,"physical_ram":lisa_engine::memory::physical_ram(),"available_ram":lisa_engine::memory::available_ram()})
}

impl App {
    fn new(storage: PathBuf) -> Result<Self> {
        ensure!(
            storage.is_absolute(),
            "Application storage must be an absolute path"
        );
        lisa_metal::runtime::check_platform_support()?;
        lisa_metal::shader_pack::initialize()?;
        let (commands, rx) = mpsc::channel();
        let events = Arc::new(Mutex::new(Vec::new()));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = backend::spawn_controlled(rx, events.clone(), cancel.clone())?;
        Ok(Self {
            commands,
            events,
            cancel,
            worker: Some(worker),
            store: chats::Store::at(storage.join("chats")),
            browser: HashMap::new(),
            next_browser: 0,
            server_stop: Arc::new(AtomicBool::new(false)),
            busy: false,
            loaded: false,
            downloading: false,
            unloading: false,
        })
    }

    fn send(&self, command: Command) -> Result<()> {
        self.commands
            .send(command)
            .context("The inference worker stopped")
    }

    fn poll(&mut self) -> Result<Value> {
        if self
            .worker
            .as_ref()
            .is_none_or(|worker| worker.is_finished())
        {
            let details = self
                .worker
                .take()
                .and_then(|worker| worker.join().err())
                .map(|panic| {
                    panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()))
                        .unwrap_or_else(|| "unknown worker panic".into())
                })
                .unwrap_or_else(|| "worker exited".into());
            bail!("Inference worker stopped: {details}. Restart Lisa.");
        }
        let events = std::mem::take(
            &mut *self
                .events
                .lock()
                .map_err(|_| anyhow::anyhow!("Event queue failed"))?,
        );
        let mut output = Vec::new();
        for event in events {
            let (kind, value) = match event {
                UiEvent::Log(s) => ("log", json!(s)),
                UiEvent::UserMessage(s) => ("user", json!(s)),
                UiEvent::UserImages(paths) => ("images", json!(paths)),
                UiEvent::AssistantImage(path) => ("image", json!(path)),
                UiEvent::ImageMode(steps) => ("image_mode", json!(steps)),
                UiEvent::AssistantDelta(s) => ("delta", json!(s)),
                UiEvent::AssistantReset(s) => ("replace", json!(s)),
                UiEvent::AssistantDone => {
                    self.busy = self.unloading;
                    ("done", Value::Null)
                }
                UiEvent::TurnCommitted(s) => ("committed", json!(s)),
                UiEvent::Loaded(loaded) => {
                    if self.unloading {
                        continue;
                    }
                    self.busy = false;
                    self.loaded = loaded;
                    ("loaded", json!(loaded))
                }
                UiEvent::Unloaded => {
                    self.busy = false;
                    self.loaded = false;
                    self.unloading = false;
                    ("unloaded", Value::Null)
                }
                UiEvent::SupportsImages(supported) => ("supports_images", json!(supported)),
                UiEvent::ServeStarted(url) => ("serving", json!(url)),
                UiEvent::ServeStopped => ("serving", Value::Null),
                UiEvent::ModelsChanged => {
                    self.downloading = false;
                    ("models_changed", Value::Null)
                }
                UiEvent::DownloadProgress {
                    repo,
                    completed,
                    total,
                } => (
                    "download",
                    json!({"id":repo,"completed":completed,"total":total}),
                ),
                UiEvent::DrafterDownloadProgress(progress) => ("draft_download", json!(progress)),
                UiEvent::WebSearchRequested { request, reply } => {
                    self.next_browser += 1;
                    self.browser.insert(self.next_browser, reply);
                    let (url, script, search) = match request {
                        BrowserRequest::Search(query) => (
                            browser::google_search_url(&query),
                            browser::SEARCH_SCRIPT,
                            true,
                        ),
                        BrowserRequest::Open(url) => (url, browser::PAGE_SCRIPT, false),
                    };
                    (
                        "browse",
                        json!({"id":self.next_browser,"url":url,"search":search,
                        "script":format!("(() => {{ {} return {}; }})()",browser::GOOGLE_PROMPT_SCRIPT,script)}),
                    )
                }
                // Speech is provided directly by the Swift platform adapter.
                UiEvent::RemoteModels(_)
                | UiEvent::VoiceMicrophoneAccess(_)
                | UiEvent::VoiceSpeechAccess(_)
                | UiEvent::VoiceTranscript { .. }
                | UiEvent::VoiceError(_) => continue,
            };
            output.push(json!({"kind":kind,"value":value}));
        }
        Ok(json!(output))
    }

    fn request(&mut self, request: &Value) -> Result<Value> {
        match text(request, "op")? {
            "poll" => return self.poll(),
            "models" => return Ok(model_list()),
            "chats" => {
                let (chats, errors) = self.store.load();
                return Ok(json!({"chats":chats,"errors":errors}));
            }
            "new_chat" => {
                return Ok(serde_json::to_value(chats::Chat::new(
                    request["model"].as_str().map(str::to_owned),
                    text(request, "system")?.into(),
                ))?);
            }
            "save_chat" => {
                let chat: chats::Chat = serde_json::from_value(request["chat"].clone())?;
                self.store.save(&chat)?;
            }
            "delete_chat" => return Ok(json!(self.store.delete(text(request, "id")?)?)),
            "keep_images" => {
                let paths: Vec<PathBuf> = serde_json::from_value(request["paths"].clone())?;
                return Ok(json!(self.store.keep_images(text(request, "id")?, &paths)?));
            }
            "load" => {
                ensure!(!self.busy, "Wait for the current operation to finish");
                let id = text(request, "id")?;
                let model = models::discover_models()
                    .into_iter()
                    .find(|model| model.repo_id == id)
                    .context("Download the model first")?;
                let bytes = model.resident_bytes.unwrap_or(model.size);
                ensure!(
                    models::model_ram_budget() > 0 && bytes <= models::model_ram_budget(),
                    "This model exceeds the device memory budget"
                );
                #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
                ensure!(
                    lisa_engine::memory::available_ram()
                        .is_some_and(|free| bytes as usize <= free.saturating_sub(512 << 20)),
                    "Not enough app memory to load this model"
                );
                self.cancel.store(false, Ordering::Release);
                self.send(Command::Load(model.path.into()))?;
                self.busy = true;
                self.loaded = false;
            }
            "download" => {
                ensure!(!self.downloading, "A download is already in progress");
                let id = text(request, "id")?;
                ensure!(catalog::find(id).is_some(), "Unknown model");
                self.send(Command::Download(id.into()))?;
                self.downloading = true;
            }
            "restore" => {
                ensure!(!self.busy, "Wait for the current operation to finish");
                self.send(Command::Restore {
                    turns: serde_json::from_value(request["turns"].clone())?,
                    system: text(request, "system")?.into(),
                })?;
            }
            "clear" => {
                ensure!(!self.busy, "Wait for the current operation to finish");
                self.send(Command::Clear)?;
            }
            "chat" => {
                ensure!(
                    self.loaded && !self.busy,
                    "Choose a model and wait for it to finish loading"
                );
                let message = text(request, "text")?.to_owned();
                ensure!(!message.trim().is_empty(), "The message is empty");
                let system = text(request, "system")?.to_owned();
                let images: Vec<PathBuf> =
                    serde_json::from_value(request.get("images").cloned().unwrap_or(json!([])))?;
                self.cancel.store(false, Ordering::Release);
                let command = if request["browse"].as_bool().unwrap_or(true) {
                    Command::WebAgent {
                        text: message,
                        system,
                        images,
                    }
                } else {
                    Command::Chat {
                        text: message,
                        system,
                        images,
                    }
                };
                self.send(command)?;
                self.busy = true;
            }
            "cancel" => {
                self.cancel.store(true, Ordering::Release);
                self.browser.clear();
            }
            "unload" => {
                self.cancel.store(true, Ordering::Release);
                self.browser.clear();
                self.server_stop.store(true, Ordering::Release);
                if !self.unloading {
                    self.send(Command::Unload)?;
                }
                self.unloading = true;
                self.busy = true;
                self.loaded = false;
            }
            "serve" => {
                ensure!(self.loaded && !self.busy, "Load a model first");
                self.server_stop = Arc::new(AtomicBool::new(false));
                self.send(Command::StartServe {
                    addr: text(request, "address")?.into(),
                    stop: self.server_stop.clone(),
                })?;
            }
            "stop_server" => self.server_stop.store(true, Ordering::Release),
            "browser_result" => {
                let id = request["id"]
                    .as_u64()
                    .context("Missing browser request id")?;
                if let Some(reply) = self.browser.get(&id) {
                    if let Some(message) = request["needs_input"].as_str() {
                        let _ = reply.send(BrowserEvent::NeedsInput(message.into()));
                    } else {
                        let result = if let Some(error) = request["error"].as_str() {
                            Err(error.into())
                        } else if request["search"].as_bool().unwrap_or(false) {
                            browser::parse_search_page(text(request, "json")?)
                        } else {
                            browser::parse_page(text(request, "json")?)
                        };
                        let _ = reply.send(BrowserEvent::Finished(result));
                        self.browser.remove(&id);
                    }
                }
            }
            op => bail!("Unknown operation: {op}"),
        }
        Ok(Value::Null)
    }
}

/// # Safety
/// `input` must point to a live, NUL-terminated UTF-8 string for this call.
/// Calls are serialized; no Rust references or callbacks cross the ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lisa_app_request(input: *const c_char) -> *mut c_char {
    let result = std::panic::catch_unwind(|| -> Result<Value> {
        ensure!(!input.is_null(), "Null request");
        let request: Value = serde_json::from_str(unsafe { CStr::from_ptr(input) }.to_str()?)?;
        let mut state = APP
            .lock()
            .map_err(|_| anyhow::anyhow!("Application state failed; restart Lisa"))?;
        if text(&request, "op")? == "initialize" {
            ensure!(state.is_none(), "Lisa is already initialized");
            *state = Some(App::new(PathBuf::from(text(&request, "storage")?))?);
            return Ok(json!({"version":1}));
        }
        state
            .as_mut()
            .context("Lisa is not initialized")?
            .request(&request)
    });
    let response = match result {
        Ok(Ok(value)) => json!({"ok":true,"value":value}),
        Ok(Err(error)) => json!({"ok":false,"error":format!("{error:#}")}),
        Err(_) => json!({"ok":false,"error":"Application request failed; restart Lisa"}),
    };
    // JSON escapes embedded NULs, so this conversion cannot fail.
    CString::new(response.to_string()).unwrap().into_raw()
}

/// # Safety
/// `string` must be null or an unreleased result of lisa_app_request.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lisa_app_free(string: *mut c_char) {
    if !string.is_null() {
        drop(unsafe { CString::from_raw(string) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abi_reports_invalid_input_and_preserves_unicode() {
        for input in ["{", r#"{"op":"not-initialized","text":"Café 🙂"}"#] {
            let input = CString::new(input).unwrap();
            let output = unsafe { lisa_app_request(input.as_ptr()) };
            let value: Value =
                serde_json::from_str(unsafe { CStr::from_ptr(output) }.to_str().unwrap()).unwrap();
            assert_eq!(value["ok"], false);
            assert!(value["error"].as_str().is_some());
            unsafe { lisa_app_free(output) };
        }
        let output = unsafe { lisa_app_request(std::ptr::null()) };
        assert!(
            unsafe { CStr::from_ptr(output) }
                .to_str()
                .unwrap()
                .contains("Null request")
        );
        unsafe {
            lisa_app_free(output);
            lisa_app_free(std::ptr::null_mut())
        };
    }
}
