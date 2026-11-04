//! Bounded HTTP/1.x request parsing and response framing.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use anyhow::{Context, ensure};
use serde_json::Value;

const MAX_HEADERS: usize = 64 << 10;
const MAX_BODY: usize = 32 << 20;

type Request = (String, String, Vec<u8>);

/// This connection handles one request, then closes. Chunked request bodies
/// are unsupported and rejected explicitly rather than treated as empty JSON.
pub(super) fn read_request(stream: &mut TcpStream) -> anyhow::Result<Option<Request>> {
    let mut reader = BufReader::new(stream);
    let mut budget = MAX_HEADERS;
    let Some(line) = header_line(&mut reader, &mut budget)? else {
        return Ok(None);
    };
    let mut parts = line.split_whitespace();
    let method = parts.next().context("missing HTTP method")?.to_owned();
    let path = parts
        .next()
        .context("missing HTTP request target")?
        .to_owned();
    let version = parts.next().context("missing HTTP version")?;
    ensure!(
        matches!(version, "HTTP/1.0" | "HTTP/1.1") && parts.next().is_none(),
        "invalid HTTP request line"
    );
    ensure!(path.starts_with('/'), "HTTP request target must be a path");

    let mut content_length = None;
    loop {
        let line = header_line(&mut reader, &mut budget)?.context("truncated HTTP headers")?;
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':').context("invalid HTTP header")?;
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)),
            "invalid HTTP header name"
        );
        if name.eq_ignore_ascii_case("transfer-encoding") {
            anyhow::bail!("Transfer-Encoding is unsupported; send Content-Length");
        }
        if name.eq_ignore_ascii_case("content-length") {
            let value = value.trim();
            ensure!(
                !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
                "invalid Content-Length"
            );
            let length: usize = value.parse().context("invalid Content-Length")?;
            ensure!(
                content_length.is_none_or(|prior| prior == length),
                "conflicting Content-Length headers"
            );
            ensure!(
                length <= MAX_BODY,
                "request body too large ({length} bytes)"
            );
            content_length = Some(length);
        }
    }
    let mut body = vec![0; content_length.unwrap_or(0)];
    reader
        .read_exact(&mut body)
        .context("truncated HTTP body")?;
    Ok(Some((method, path, body)))
}

fn header_line(reader: &mut impl BufRead, budget: &mut usize) -> anyhow::Result<Option<String>> {
    let mut bytes = Vec::new();
    let n = reader
        .take((*budget + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if n == 0 {
        return Ok(None);
    }
    ensure!(n <= *budget, "HTTP headers exceed {MAX_HEADERS} bytes");
    *budget -= n;
    ensure!(bytes.last() == Some(&b'\n'), "truncated HTTP header line");
    bytes.pop();
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    ensure!(!bytes.contains(&b'\r'), "invalid HTTP header line");
    Ok(Some(
        String::from_utf8(bytes).context("invalid HTTP header encoding")?,
    ))
}

pub(super) fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

pub(super) fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    if data.is_empty() {
        stream.write_all(b"0\r\n\r\n")
    } else {
        stream.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
        stream.write_all(data)?;
        stream.write_all(b"\r\n")
    }
}

pub(super) fn write_sse(stream: &mut TcpStream, value: &Value) -> std::io::Result<()> {
    write_chunk(stream, format!("data: {value}\n\n").as_bytes())
}
