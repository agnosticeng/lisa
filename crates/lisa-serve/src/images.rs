//! Image generation requests run on the same owning engine thread as UI work.
use anyhow::{Context, ensure};
use base64::Engine;
use lisa_engine::models::image_generation::{Options, Pipeline};
use serde_json::{Value, json};
use std::{
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

struct Job {
    prompt: String,
    options: Options,
    reply: mpsc::Sender<anyhow::Result<Vec<u8>>>,
}
pub struct ImageServer {
    rx: mpsc::Receiver<Job>,
    stop: Arc<AtomicBool>,
    listener: Option<thread::JoinHandle<()>>,
}
impl ImageServer {
    pub fn bind(addr: &str, model_id: &str, stop: Arc<AtomicBool>) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let (tx, rx) = mpsc::sync_channel::<Job>(4);
        let quit = stop.clone();
        let model_id = model_id.to_string();
        let handle = thread::Builder::new()
            .name("lisa-image-http".into())
            .spawn(move || {
                while !quit.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let tx = tx.clone();
                            let id = model_id.clone();
                            thread::spawn(move || {
                                let _ = handle(stream, tx, &id);
                            });
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(20))
                        }
                        Err(_) => break,
                    }
                }
            })?;
        Ok(Self {
            rx,
            stop,
            listener: Some(handle),
        })
    }
    pub fn step(&self, model: &Pipeline) -> bool {
        let Ok(job) = self.rx.try_recv() else {
            return false;
        };
        let result = model.generate(&job.prompt, &job.options, &mut |stage, i, n| {
            ensure!(
                !self.stop.load(Ordering::Relaxed),
                "image generation cancelled"
            );
            eprintln!("[images] {stage} {i}/{n}");
            Ok(())
        });
        let _ = job.reply.send(result);
        true
    }
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.listener.take() {
            let _ = handle.join();
        }
    }
}
impl Drop for ImageServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn parse(body: &Value, model_id: &str) -> anyhow::Result<(String, Options)> {
    if let Some(model) = body.get("model") {
        ensure!(
            model.as_str() == Some(model_id),
            "requested model is not loaded"
        );
    }
    let prompt = body
        .get("prompt")
        .and_then(Value::as_str)
        .context("prompt must be a string")?;
    ensure!(
        !prompt.trim().is_empty() && prompt.len() <= 16384,
        "prompt must contain 1..=16384 bytes"
    );
    ensure!(
        body.get("n").is_none_or(|v| v.as_u64() == Some(1)),
        "n must be 1"
    );
    ensure!(
        body.get("response_format")
            .is_none_or(|v| v.as_str() == Some("b64_json")),
        "response_format must be b64_json"
    );
    let size = body
        .get("size")
        .map(|v| v.as_str().context("size must be WIDTHxHEIGHT"))
        .transpose()?
        .unwrap_or("512x512");
    let (w, h) = size.split_once('x').context("size must be WIDTHxHEIGHT")?;
    let mut options = Options {
        width: w.parse()?,
        height: h.parse()?,
        ..Options::default()
    };
    if let Some(seed) = body.get("seed") {
        options.seed = seed.as_u64().context("seed must be an unsigned integer")?;
    }
    if let Some(steps) = body.get("steps") {
        options.steps = Some(u32::try_from(
            steps.as_u64().context("steps must be an integer")?,
        )?);
    }
    if let Some(caption) = body.get("prompt_is_caption") {
        options.prompt_is_caption = caption
            .as_bool()
            .context("prompt_is_caption must be a boolean")?;
    }
    options.validate()?;
    Ok((prompt.to_string(), options))
}
fn handle(mut stream: TcpStream, tx: mpsc::SyncSender<Job>, model_id: &str) -> anyhow::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let Some((method, path, body)) = super::read_request(&mut stream)? else {
        return Ok(());
    };
    let send = |stream: &mut TcpStream, status: &str, value: Value| {
        super::write_response(
            stream,
            status,
            "application/json",
            value.to_string().as_bytes(),
        )
    };
    if method == "GET" && path == "/health" {
        return send(
            &mut stream,
            "200 OK",
            json!({"status":"ok","model":model_id,"image_generation":true}),
        );
    }
    if method == "GET" && path == "/v1/models" {
        return send(
            &mut stream,
            "200 OK",
            json!({"object":"list","data":[{"id":model_id,"object":"model","owned_by":"local","capabilities":["image_generation"]}]}),
        );
    }
    if method != "POST" || path != "/v1/images/generations" {
        return send(
            &mut stream,
            "404 Not Found",
            json!({"error":{"message":"use /v1/images/generations with the loaded image model"}}),
        );
    }
    let request = serde_json::from_slice::<Value>(&body)
        .map_err(anyhow::Error::from)
        .and_then(|b| parse(&b, model_id));
    let (prompt, options) = match request {
        Ok(v) => v,
        Err(e) => {
            return send(
                &mut stream,
                "400 Bad Request",
                json!({"error":{"message":e.to_string(),"type":"invalid_request_error"}}),
            );
        }
    };
    let (reply, rx) = mpsc::channel();
    if tx
        .try_send(Job {
            prompt,
            options,
            reply,
        })
        .is_err()
    {
        return send(
            &mut stream,
            "503 Service Unavailable",
            json!({"error":{"message":"image queue is full or stopping"}}),
        );
    }
    match rx.recv() {
        Ok(Ok(png)) => send(
            &mut stream,
            "200 OK",
            json!({"created":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),"data":[{"b64_json":base64::engine::general_purpose::STANDARD.encode(png)}]}),
        ),
        Ok(Err(e)) => send(
            &mut stream,
            "500 Internal Server Error",
            json!({"error":{"message":e.to_string()}}),
        ),
        Err(_) => send(
            &mut stream,
            "503 Service Unavailable",
            json!({"error":{"message":"image worker stopped"}}),
        ),
    }
}

pub fn run(model: Box<Pipeline>, cfg: super::ServerConfig) -> anyhow::Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let server = ImageServer::bind(&cfg.addr, &cfg.model_id, stop)?;
    eprintln!("[serve] image generation listening on http://{}", cfg.addr);
    loop {
        if !server.step(&model) {
            thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_image_request_parameters() {
        let (prompt, options) = parse(
            &json!({"model":"flux","prompt":"an apple","size":"256x512","seed":0}),
            "flux",
        )
        .unwrap();
        assert_eq!(prompt, "an apple");
        assert_eq!(options.width, 256);
        assert_eq!(options.height, 512);
        assert_eq!(options.seed, 0);
        assert_eq!(options.steps, None);
        let (_, caption) = parse(
            &json!({"prompt":"a complete scene", "steps":16, "prompt_is_caption":true}),
            "flux3",
        )
        .unwrap();
        assert_eq!(caption.steps, Some(16));
        assert!(caption.prompt_is_caption);
        for bad in [
            json!({"prompt":""}),
            json!({"prompt":"a","size":"255x256"}),
            json!({"prompt":"a","size":"2048x2048"}),
            json!({"prompt":"a","seed":-1}),
            json!({"prompt":"a","steps":0}),
            json!({"prompt":"a","n":2}),
            json!({"prompt":"a","response_format":"url"}),
            json!({"prompt":"a","model":"other"}),
            json!({"prompt":"a","prompt_is_caption":"true"}),
        ] {
            assert!(parse(&bad, "flux").is_err(), "{bad}");
        }
    }
}
