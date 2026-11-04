//! Per-server response identity and bounded Responses API history.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::Value;

const MAX_RESPONSES: usize = 256;
const MAX_RESPONSE_BYTES: usize = 64 << 20;

#[derive(Clone)]
pub(super) struct ApiState {
    pub model_id: String,
    responses: Arc<Mutex<ResponseStore>>,
}

impl ApiState {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            responses: Arc::new(Mutex::new(ResponseStore::default())),
        }
    }

    pub fn store(&self, id: &str, payload: Value, conversation: Option<Vec<Value>>) {
        // Count serialized bytes without allocating another copy of the history.
        // The entry cap also bounds per-value/container bookkeeping.
        let mut bytes = ByteCount::default();
        if serde_json::to_writer(&mut bytes, &payload).is_err()
            || serde_json::to_writer(&mut bytes, &conversation).is_err()
        {
            return;
        }
        let mut store = self.responses.lock().unwrap();
        if let Some(index) = store.entries.iter().position(|entry| entry.id == id) {
            store.remove(index);
        }
        if bytes.0 > MAX_RESPONSE_BYTES {
            return;
        }
        while store.entries.len() >= MAX_RESPONSES
            || store.bytes.saturating_add(bytes.0) > MAX_RESPONSE_BYTES
        {
            store.remove(0);
        }
        store.bytes += bytes.0;
        store.entries.push_back(StoredResponse {
            id: id.to_owned(),
            payload,
            conversation,
            bytes: bytes.0,
        });
    }

    pub fn response(&self, id: &str) -> Option<Value> {
        self.responses
            .lock()
            .unwrap()
            .entries
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| entry.payload.clone())
    }

    pub fn extend_conversation(&self, body: &mut Value) -> anyhow::Result<()> {
        let Some(previous) = body.get("previous_response_id") else {
            return Ok(());
        };
        if previous.is_null() {
            return Ok(());
        }
        let id = previous
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("previous_response_id must be a string"))?;
        let mut prior = self
            .responses
            .lock()
            .unwrap()
            .entries
            .iter()
            .find(|entry| entry.id == id)
            .and_then(|entry| entry.conversation.clone())
            .ok_or_else(|| anyhow::anyhow!("previous response {id:?} is unavailable or expired"))?;
        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            prior.extend(messages.iter().cloned());
        }
        body["messages"] = Value::Array(prior);
        Ok(())
    }
}

#[derive(Default)]
struct ResponseStore {
    entries: VecDeque<StoredResponse>,
    bytes: usize,
}

impl ResponseStore {
    fn remove(&mut self, index: usize) {
        if let Some(entry) = self.entries.remove(index) {
            self.bytes -= entry.bytes;
        }
    }
}

struct StoredResponse {
    id: String,
    payload: Value,
    conversation: Option<Vec<Value>>,
    bytes: usize,
}

#[derive(Default)]
struct ByteCount(usize);

impl std::io::Write for ByteCount {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
