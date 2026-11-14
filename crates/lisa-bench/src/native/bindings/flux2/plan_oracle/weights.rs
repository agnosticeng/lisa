#![cfg(test)]
//! Image weights: original MLX affine 4-bit/group-64 packs and native BF16/F32 tensors.
use anyhow::{Context, ensure};
use lisa_metal::{Array, Dtype, ops};
use std::{cell::RefCell, collections::HashMap};

pub struct Weights(pub HashMap<String, Array>, RefCell<HashMap<String, Array>>);
impl Weights {
    pub fn from_map(tensors: HashMap<String, Array>) -> Self {
        Self(tensors, RefCell::new(HashMap::new()))
    }
    pub fn get(&self, key: &str) -> anyhow::Result<Array> {
        self.0
            .get(key)
            .cloned()
            .with_context(|| format!("missing FLUX tensor {key}"))
    }
    pub fn lin(&self, key: &str, x: &Array) -> anyhow::Result<Array> {
        let w = self.get(&format!("{key}.weight"))?;
        let y = if w.dtype() == Dtype::Uint32 {
            let s = self.get(&format!("{key}.scales"))?;
            let b = self.get(&format!("{key}.biases"))?;
            ensure!(
                w.dim(-1) * 8 == s.dim(-1) * 64 && x.dim(-1) == s.dim(-1) * 64,
                "FLUX requires affine 4-bit/group-64 weights at {key}"
            );
            ops::quantized_matmul(x, &w, &s, Some(&b), true, 64, 4)?
        } else {
            ensure!(
                w.ndim() == 2 && x.ndim() > 0 && x.dim(-1) == w.dim(1),
                "invalid dense FLUX projection {key}"
            );
            let mut shape = x.shape().to_vec();
            *shape.last_mut().unwrap() = w.dim(0);
            // The dense runtime requires a contiguous RHS. Cache this exact
            // layout once, instead of transposing every matrix on every token
            // and denoising step. The original payload and arithmetic stay intact.
            let cached = self.1.borrow().get(key).cloned();
            let transposed = match cached {
                Some(weight) => weight,
                None => {
                    let weight = w.transpose_axes(&[1, 0])?.contiguous()?;
                    self.1.borrow_mut().insert(key.to_owned(), weight.clone());
                    weight
                }
            };
            x.reshape(&[-1, w.dim(1)])?
                .matmul(&transposed)?
                .reshape(&shape)?
        };
        Ok(if let Some(b) = self.0.get(&format!("{key}.bias")) {
            y.add(b)?
        } else {
            y
        })
    }
}
