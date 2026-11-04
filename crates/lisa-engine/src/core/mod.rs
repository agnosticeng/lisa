//! Model-agnostic runtime: weight loading, quantization, norm/rope primitives,
//! the layer caches, sampling, tokenization, and the generation / batching /
//! scheduling loops. Nothing here names a concrete model.

pub mod admission;
pub mod batch;
pub mod cache;
pub mod capture;
pub mod copy_draft;
pub mod generate;
pub mod gguf_resident;
pub mod loader;
pub mod mem;
pub mod norm;
pub mod oracle;
pub mod prefix_cache;
pub mod prepare;
pub mod quant;
pub mod round_cost;
pub mod sampler;
pub mod sched;
pub mod session;
pub mod tokenizer;
mod weight_arena;

pub mod gguf_tokenizer;
