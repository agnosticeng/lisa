//! Image decoding and native vision encoders shared by the language models.

mod context;
pub mod glm;
mod processor;
pub mod qwen;
pub use context::Vision;

use std::io::Cursor;

use anyhow::{Context, ensure};
use base64::Engine;
use image::{ImageDecoder, ImageReader, RgbImage};

/// Bound encoded uploads and decoded pixels before allocating the pixel buffer.
pub const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_IMAGE_PIXELS: u64 = 32 * 1024 * 1024;
pub const MAX_IMAGES: usize = 8;

/// Decode PNG, JPEG or WebP, honoring EXIF orientation. The model processor
/// applies its own resizing and normalization after this format-only step.
pub fn decode_image(bytes: &[u8]) -> anyhow::Result<RgbImage> {
    ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_IMAGE_BYTES,
        "image must contain between 1 and {MAX_IMAGE_BYTES} encoded bytes"
    );
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32_768);
    limits.max_image_height = Some(32_768);
    limits.max_alloc = Some(MAX_IMAGE_PIXELS * 8);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .context("unsupported or invalid image")?;
    let (width, height) = decoder.dimensions();
    ensure!(
        width > 0 && height > 0 && u64::from(width) * u64::from(height) <= MAX_IMAGE_PIXELS,
        "image dimensions exceed the {MAX_IMAGE_PIXELS}-pixel limit"
    );
    let orientation = decoder.orientation()?;
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    // Transparent uploads are composited onto white, rather than interpreting
    // arbitrary RGB bytes beneath fully transparent pixels as visible content.
    let rgba = image.to_rgba8();
    let mut rgb = RgbImage::new(rgba.width(), rgba.height());
    for (dst, src) in rgb.pixels_mut().zip(rgba.pixels()) {
        let alpha = u32::from(src[3]);
        for channel in 0..3 {
            dst[channel] =
                ((u32::from(src[channel]) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
        }
    }
    Ok(rgb)
}

/// Decode an inline image without allowing a request to read a server file.
pub fn decode_data_url(url: &str) -> anyhow::Result<RgbImage> {
    let (header, data) = url
        .split_once(',')
        .context("image data URL has no payload")?;
    ensure!(
        matches!(
            header,
            "data:image/png;base64" | "data:image/jpeg;base64" | "data:image/webp;base64"
        ),
        "expected a base64 PNG, JPEG or WebP image data URL"
    );
    ensure!(
        data.len() <= MAX_IMAGE_BYTES.div_ceil(3) * 4,
        "encoded image exceeds the {MAX_IMAGE_BYTES}-byte limit"
    );
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .context("invalid image base64")?;
    decode_image(&bytes)
}

/// The UI sends the original encoded bytes; inference uses the same decoder
/// as HTTP uploads, including the same limits and orientation handling.
pub fn file_data_url(path: &std::path::Path) -> anyhow::Result<String> {
    let metadata = std::fs::metadata(path)?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_IMAGE_BYTES as u64,
        "image file is missing or exceeds the {MAX_IMAGE_BYTES}-byte limit"
    );
    let bytes = std::fs::read(path)?;
    decode_image(&bytes)?;
    let mime = match image::guess_format(&bytes)? {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::WebP => "image/webp",
        _ => anyhow::bail!("only PNG, JPEG and WebP images are supported"),
    };
    Ok(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

/// Whether normalized chat messages contain image content blocks.
pub fn messages_have_images(body: &serde_json::Value) -> bool {
    body.get("messages")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("content").and_then(serde_json::Value::as_array))
        .flatten()
        .any(is_image_part)
}

fn is_image_part(part: &serde_json::Value) -> bool {
    matches!(
        part.get("type").and_then(serde_json::Value::as_str),
        Some("image" | "image_url" | "input_image")
    ) || part.get("image_url").is_some()
}

/// Inline OpenAI/Responses images and Anthropic base64 sources share one
/// decoding path. Remote fetches and file IDs require a separate asset store.
pub fn decode_message_images(body: &serde_json::Value) -> anyhow::Result<Vec<RgbImage>> {
    use serde_json::Value;
    let mut images = Vec::new();
    let mut total_pixels = 0u64;
    for message in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for part in message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|p| is_image_part(p))
        {
            ensure!(
                message.get("role").and_then(Value::as_str) == Some("user"),
                "images are supported in user messages only"
            );
            ensure!(
                images.len() < MAX_IMAGES,
                "at most {MAX_IMAGES} images are allowed per request"
            );
            let image = if let Some(source) = part.get("source") {
                ensure!(
                    source.get("type").and_then(Value::as_str) == Some("base64"),
                    "image sources must use inline base64"
                );
                let media = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .context("image source has no media_type")?;
                let data = source
                    .get("data")
                    .and_then(Value::as_str)
                    .context("image source has no data")?;
                decode_data_url(&format!("data:{media};base64,{data}"))?
            } else {
                let url = part
                    .get("image_url")
                    .and_then(|v| v.as_str().or_else(|| v.get("url").and_then(Value::as_str)))
                    .context("image block needs an image_url data URL")?;
                decode_data_url(url)?
            };
            total_pixels += u64::from(image.width()) * u64::from(image.height());
            ensure!(
                total_pixels <= MAX_IMAGE_PIXELS,
                "combined decoded images exceed the {MAX_IMAGE_PIXELS}-pixel request limit"
            );
            images.push(image);
        }
    }
    Ok(images)
}

/// Admission for a media operation before allocating its GPU intermediates.
pub(super) fn admit_scratch(bytes: usize) -> anyhow::Result<()> {
    ensure!(
        lisa_mlx::memory::active_memory()?.saturating_add(bytes) < lisa_mlx::memory::ram_budget(),
        "image encoder exceeds the configured RAM budget"
    );
    if let Some(available) = lisa_mlx::memory::available_ram() {
        ensure!(
            available > bytes,
            "insufficient available memory for image encoding"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inline_images_validate_payload_and_composite_alpha() -> anyhow::Result<()> {
        let img = image::RgbaImage::from_pixel(2, 3, image::Rgba([255, 0, 0, 0]));
        let mut png = Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png)?;
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png.into_inner())
        );
        let decoded = decode_data_url(&url)?;
        assert_eq!(decoded.dimensions(), (2, 3));
        assert!(decoded.pixels().all(|p| p.0 == [255; 3]));
        assert!(decode_data_url("file:///etc/passwd").is_err());
        assert!(decode_data_url("https://example.com/x.png").is_err());
        assert!(decode_data_url("data:image/png;base64,!").is_err());
        assert!(decode_image(&[]).is_err());
        Ok(())
    }
    #[test]
    fn unsupported_image_blocks_fail_explicitly() {
        let body = serde_json::json!({"messages":[{"role":"user","content":[{"type":"input_image","file_id":"file-1"}]}]});
        assert!(messages_have_images(&body));
        assert!(decode_message_images(&body).is_err());
    }
}
