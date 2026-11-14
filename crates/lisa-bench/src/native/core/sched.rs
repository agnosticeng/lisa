//! Metal scheduler aliases; scheduling policy is backend-independent.
pub use crate::scheduling::*;
pub type Request = crate::scheduling::Request<crate::core::sampler::Sampler>;
