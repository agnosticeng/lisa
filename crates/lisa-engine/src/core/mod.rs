//! Device-independent plan execution, checkpoint preparation and shared policy.
pub mod block_policy;
pub mod config;
pub mod copy_draft;
pub mod decoder;
pub mod gguf_tokenizer;
pub mod graph;
pub mod oracle;
pub mod plan;
pub mod prepare;
pub mod program;
pub mod round_cost;
pub mod sequence;
pub mod tokenizer;
pub mod traversal;
pub mod weight_arena;

pub mod admission;

pub mod capture;
