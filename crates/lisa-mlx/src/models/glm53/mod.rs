//! GLM-5.3 (`glm5_next`) Metal kernels — port of ds4 `glm53_kda.metal`.

pub mod gguf;
pub mod kernels;

#[cfg(all(test, feature = "glm53-fixtures"))]
mod test_fixtures;
