//! Original DS4 GLM-5.3 Flash GGUF package. Packed weights stay unchanged.

pub mod weights;

pub const REPO: &str = "antirez/glm-5.3-flash-gguf";
pub const ENTRY: &str = "GLM-5.3-Flash-Q2.gguf";
pub const REVISION: &str = "b2fa29d7a6b410db11221c904973967b80b760f5";
pub const DS4_REVISION: &str = "b3f50180ec2c7324892be1f50bc83555ccb8d5e7";

pub mod moe;
pub mod projection;
pub mod tower;

#[cfg(test)]
mod tests;
