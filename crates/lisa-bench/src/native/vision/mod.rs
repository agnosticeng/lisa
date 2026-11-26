//! Application aliases for portable image decoding and vision bindings.
pub use lisa_engine::vision::{
    MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGES, decode_data_url, decode_image,
    decode_message_images, file_data_url, messages_have_images,
};
mod context;
pub use context::Vision;
#[cfg(test)]
pub mod glm;
#[cfg(test)]
pub mod qwen;
