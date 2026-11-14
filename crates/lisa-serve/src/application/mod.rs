//! Application composition: provider discovery, selection and explicit ownership.
pub mod device;
pub mod loading;
pub use loading::{
    LoadOptions, Loaded, load, load_dir, load_dir_with, load_dir_with_dspine, load_dir_with_options,
};
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;
