//! Application composition: provider discovery, selection and explicit ownership.
pub mod device;
pub mod loading;
pub use loading::{Loaded, load, load_dir, load_dir_with};
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;
