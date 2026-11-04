//! Image weights: original MLX affine 4-bit/group-64 packs and native BF16/F32 tensors.
use crate::core::loader::{Checkpoint, Shard};
use anyhow::{Context, ensure};
use lisa_mlx::{Array, Dtype, ops};
use std::{cell::RefCell, collections::HashMap, path::Path};

pub struct Weights(pub HashMap<String, Array>, RefCell<HashMap<String, Array>>);
impl Weights {
    pub fn from_map(tensors: HashMap<String, Array>) -> Self {
        Self(tensors, RefCell::new(HashMap::new()))
    }
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        Self::load_filtered(dir, |_| true)
    }
    pub fn load_file(path: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<Self> {
        let shard = Shard::open(path)?;
        let mut out = HashMap::new();
        for key in shard.tensor_names().filter(|key| keep(key)) {
            out.insert(key.clone(), crate::core::quant::get_tensor(&shard, key)?);
        }
        Ok(Self::from_map(out))
    }
    pub fn load_filtered(dir: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<Self> {
        if !dir.join("model.safetensors.index.json").is_file() {
            return Self::load_file(&dir.join("model.safetensors"), keep);
        }
        let cp = Checkpoint::open(dir)?;
        let mut shards = HashMap::new();
        let mut out = HashMap::new();
        for (key, file) in cp.weight_map {
            if !keep(&key) {
                continue;
            }
            if !shards.contains_key(&file) {
                shards.insert(file.clone(), Shard::open(&dir.join(&file))?);
            }
            out.insert(
                key.clone(),
                crate::core::quant::get_tensor(&shards[&file], &key)?,
            );
        }
        Ok(Self::from_map(out))
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
    pub fn count(&self, prefix: &str, suffix: &str) -> usize {
        (0..128)
            .take_while(|i| self.0.contains_key(&format!("{prefix}.{i}.{suffix}")))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_layout_cache_preserves_values_and_leading_dimensions() -> anyhow::Result<()> {
        for dtype in [Dtype::Bfloat16, Dtype::Float32] {
            let matrix = Array::from_slice(
                &(0..20).map(|v| v as f32 / 16.0 - 0.5).collect::<Vec<_>>(),
                &[5, 4],
            )
            .as_dtype(dtype)?;
            let weights = Weights::from_map(HashMap::from([(
                "projection.weight".into(),
                matrix.clone(),
            )]));
            let x = Array::from_slice(
                &(0..24).map(|v| v as f32 / 32.0).collect::<Vec<_>>(),
                &[1, 2, 3, 4],
            )
            .as_dtype(dtype)?;
            let reference = x
                .reshape(&[6, 4])?
                .matmul(&matrix.transpose_axes(&[1, 0])?)?
                .reshape(&[1, 2, 3, 5])?
                .as_dtype(Dtype::Float32)?;
            reference.eval()?;
            for _ in 0..2 {
                let actual = weights.lin("projection", &x)?.as_dtype(Dtype::Float32)?;
                actual.eval()?;
                assert_eq!(actual.shape(), reference.shape());
                assert_eq!(actual.as_slice::<f32>(), reference.as_slice::<f32>());
                assert_eq!(weights.1.borrow().len(), 1);
            }
        }
        Ok(())
    }
}
