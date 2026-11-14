//! Application provider composition and HTTP serving.
pub mod application;
#[cfg(feature = "metal")]
mod server;
#[cfg(feature = "metal")]
pub use server::*;
