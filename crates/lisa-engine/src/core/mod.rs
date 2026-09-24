//! Model-agnostic runtime: weight loading, quantization, norm/rope primitives,
//! the layer caches, sampling, tokenization, and the generation / batching /
//! scheduling loops. Nothing here names a concrete model.

pub mod batch;
pub mod cache;
pub mod generate;
pub mod loader;
pub mod mem;
pub mod norm;
pub mod quant;
pub mod sampler;
pub mod sched;
pub mod session;
pub mod tokenizer;
