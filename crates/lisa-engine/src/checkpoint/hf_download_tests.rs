//! Exercise real Hub client requests against a local fixture server.
use super::*;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

const REV: &str = "0123456789012345678901234567890123456789";
const CONFIG: &str = "{\"model_type\":\"qwen3_5\"}";
const WEIGHTS: &str = "fixture weights";

struct HubFixture {
    endpoint: String,
    fail_weights: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl HubFixture {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let fail_weights = Arc::new(AtomicBool::new(true));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (stop2, fail2, requests2) = (stop.clone(), fail_weights.clone(), requests.clone());
        let thread = thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                let (mut socket, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                let mut expired_token = false;
                let mut range = String::new();
                loop {
                    let mut line = String::new();
                    let count = reader.read_line(&mut line).unwrap();
                    expired_token |= line
                        .to_ascii_lowercase()
                        .starts_with("authorization: bearer expired-fixture");
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("range: ") {
                        range = value.trim().to_string();
                    }
                    if count == 0 || line == "\r\n" {
                        break;
                    }
                }
                let mut parts = request.split_whitespace();
                let method = parts.next().unwrap_or("");
                let path = parts.next().unwrap_or("");
                requests2
                    .lock()
                    .unwrap()
                    .push(format!("{method} {path} {range}"));
                let mut headers = format!("X-Repo-Commit: {REV}\r\n");
                let (mut status, mut body) = if expired_token {
                    (401, "expired fixture token".into())
                } else if path.starts_with("/api/models/test/model/revision/") {
                    (
                        200,
                        serde_json::json!({"sha": REV, "siblings": [
                            {"rfilename": "config.json", "size": CONFIG.len()},
                            {"rfilename": "nested/model.safetensors", "size": WEIGHTS.len()}
                        ]})
                        .to_string(),
                    )
                } else if path == format!("/test/model/resolve/{REV}/config.json") {
                    headers.push_str("ETag: \"aaaa\"\r\n");
                    (200, CONFIG.into())
                } else if path == format!("/test/model/resolve/{REV}/nested/model.safetensors")
                    && !fail2.load(Ordering::Relaxed)
                {
                    headers.push_str("ETag: \"bbbb\"\r\n");
                    (200, WEIGHTS.into())
                } else {
                    (404, "missing fixture file".into())
                };
                if status == 200 && path.contains("/resolve/") && !range.is_empty() {
                    let (start, end) = range
                        .strip_prefix("bytes=")
                        .unwrap()
                        .split_once('-')
                        .unwrap();
                    let start: usize = start.parse().unwrap();
                    let end = end.parse::<usize>().unwrap_or(body.len() - 1);
                    headers.push_str(&format!(
                        "Content-Range: bytes {start}-{end}/{}\r\n",
                        body.len()
                    ));
                    body = body[start..=end].to_string();
                    status = 206;
                }
                respond(&mut socket, method, status, &headers, &body);
            }
        });
        Self {
            endpoint,
            fail_weights,
            requests,
            stop,
            thread: Some(thread),
        }
    }
}

fn respond(socket: &mut TcpStream, method: &str, status: u16, headers: &str, body: &str) {
    write!(socket, "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n{headers}\r\n", body.len()).unwrap();
    if method != "HEAD" {
        socket.write_all(body.as_bytes()).unwrap();
    }
}

impl Drop for HubFixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
    }
}

#[test]
fn hub_download_retries_incomplete_packages_reuses_cache_and_repairs_bad_files() {
    let fixture = HubFixture::new();
    let cache = tempfile::tempdir().unwrap();
    let client = ApiBuilder::new()
        .with_endpoint(fixture.endpoint.clone())
        .with_token(None)
        .with_cache_dir(cache.path().to_path_buf())
        .with_progress(false)
        .build()
        .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let progress: Arc<dyn Fn(u64, u64) + Send + Sync> = Arc::new(move |done, total| {
        captured.lock().unwrap().push((done, total));
    });
    let snapshot = cache.path().join("models--test--model/snapshots").join(REV);
    assert!(
        download_with_client("test/model", &client, cache.path(), REV, progress.clone()).is_err()
    );
    assert!(snapshot.join("config.json").is_file());
    assert!(snapshot.join(INCOMPLETE_MARKER).is_file());
    assert!(!is_model_dir(&snapshot), "partial package must stay hidden");

    // The dependency resumes a partial transfer, rather than starting it again.
    std::fs::write(
        cache.path().join("models--test--model/blobs/bbbb.part"),
        &WEIGHTS.as_bytes()[..3],
    )
    .unwrap();
    fixture.fail_weights.store(false, Ordering::Relaxed);
    events.lock().unwrap().clear();
    assert_eq!(
        download_with_client("test/model", &client, cache.path(), REV, progress.clone()).unwrap(),
        snapshot
    );
    assert!(is_model_dir(&snapshot));
    assert!(!snapshot.join(INCOMPLETE_MARKER).exists());
    assert_eq!(
        std::fs::read_to_string(snapshot.join("nested/model.safetensors")).unwrap(),
        WEIGHTS
    );
    assert_eq!(
        std::fs::read_to_string(cache.path().join("models--test--model/refs/main")).unwrap(),
        REV
    );
    let observed = events.lock().unwrap().clone();
    let total = (CONFIG.len() + WEIGHTS.len()) as u64;
    assert_eq!(observed.last(), Some(&(total, total)));
    assert!(observed.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    assert!(
        observed
            .iter()
            .all(|(done, size)| *size == total && done <= size)
    );
    assert_eq!(
        fixture
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.starts_with("GET ") && request.contains("/config.json"))
            .count(),
        2
    );
    assert!(
        fixture
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|request| request.ends_with("bytes=3-"))
    );

    fixture.requests.lock().unwrap().clear();
    download_with_client("test/model", &client, cache.path(), REV, progress.clone()).unwrap();
    assert!(
        !fixture
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|request| request.contains("/resolve/"))
    );
    std::fs::write(snapshot.join("nested/model.safetensors"), b"bad").unwrap();
    download_with_client("test/model", &client, cache.path(), REV, progress).unwrap();
    assert_eq!(
        std::fs::read_to_string(snapshot.join("nested/model.safetensors")).unwrap(),
        WEIGHTS
    );
}

#[test]
fn expired_credentials_fall_back_for_public_models_only() {
    let fixture = HubFixture::new();
    let cache = tempfile::tempdir().unwrap();
    let make_client = |anonymous: bool| {
        Ok(ApiBuilder::new()
            .with_endpoint(fixture.endpoint.clone())
            .with_cache_dir(cache.path().to_path_buf())
            .with_progress(false)
            .with_token(if anonymous {
                None
            } else {
                Some("expired-fixture".into())
            })
            .build()?)
    };
    let (_, revision) = connect("test/model", make_client).unwrap();
    assert_eq!(revision, REV);
    assert_eq!(fixture.requests.lock().unwrap().len(), 2);
    assert!(connect("test/private", make_client).is_err());
}
