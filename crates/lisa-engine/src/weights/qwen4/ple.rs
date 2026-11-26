//! Mapped checkpoint row source for Qwen per-layer embeddings.
//! The runtime only receives the NgramRows provider and original packed row tensors.
use crate::{
    bindings::{BindingDevice, qwen4::ple::NgramRows},
    execution::DeviceContext,
};
use half::bf16;
use lisa_backend::{HostTensor, ScalarType};
use memmap2::Mmap;
use std::{collections::HashMap, path::Path, sync::Arc};
pub struct ShardMap {
    pub map: Arc<Mmap>,
    pub weight_off: usize,
    pub scales_off: usize,
    pub biases_off: usize,
    pub rows: usize,
}

fn table_layout(
    source: &lisa_mlx::Shard,
    w: &str,
    s: &str,
    b: &str,
) -> anyhow::Result<(usize, usize, usize, usize)> {
    let get = |name| {
        source
            .tensor_info(name)
            .ok_or_else(|| anyhow::anyhow!("missing n-gram tensor {name}"))
    };
    let (w, s, b) = (get(w)?, get(s)?, get(b)?);
    anyhow::ensure!(
        w.dtype == lisa_mlx::Dtype::Uint32
            && s.dtype == lisa_mlx::Dtype::Bfloat16
            && b.dtype == lisa_mlx::Dtype::Bfloat16,
        "invalid n-gram storage dtypes"
    );
    let rows = lisa_models::qwen4::ngram::table_rows(&w.shape, &s.shape, &b.shape)?;
    Ok((w.start, s.start, b.start, rows))
}

/// Serves dequantized rows for global row ids from the checkpoint's n-gram
/// shard tensors (`...ngram_embedding.shard_N.{weight,scales,biases}`).
pub struct NgramTable {
    pub maps: Vec<Option<ShardMap>>,
    pub rows_per_shard: usize,
    pub row_dims: usize, // 160
}

impl NgramTable {
    /// Open the shard tensors out of the model directory's safetensors files.
    /// `shard_tensor_files` maps shard index -> (file, weight name, scales
    /// name, biases name).
    pub fn open(
        dir: &Path,
        weight_map: &HashMap<String, String>,
        tensor_prefix: &str,
        shard_count: usize,
    ) -> anyhow::Result<Self> {
        // Locate the three tensors per shard. Names in the raw checkpoint
        // carry a `language_model.` prefix; match either form.
        let mut shard_files: Vec<Option<(String, String, String, String)>> =
            vec![None; shard_count];
        let prefixes = [
            format!("{tensor_prefix}.shard_"),
            format!("language_model.{tensor_prefix}.shard_"),
        ];
        for (name, file) in weight_map {
            let Some(rest) = prefixes.iter().find_map(|p| name.strip_prefix(p.as_str())) else {
                continue;
            };
            let Some((idx, tensor)) = rest.split_once('.') else {
                continue;
            };
            let idx: usize = idx.parse()?;
            anyhow::ensure!(
                idx < shard_count,
                "n-gram shard index exceeds model geometry"
            );
            match tensor {
                "weight" | "scales" | "biases" => {
                    let entry = &mut shard_files[idx];
                    let e = entry.get_or_insert_with(|| {
                        (file.clone(), String::new(), String::new(), String::new())
                    });
                    anyhow::ensure!(e.0 == *file, "n-gram affine triple spans different files");
                    match tensor {
                        "weight" => e.1 = name.clone(),
                        "scales" => e.2 = name.clone(),
                        "biases" => e.3 = name.clone(),
                        _ => unreachable!(),
                    }
                }
                _ => {}
            }
        }

        let mut maps = Vec::with_capacity(shard_count);
        for (idx, entry) in shard_files.into_iter().enumerate() {
            let (file, w, s, b) =
                entry.ok_or_else(|| anyhow::anyhow!("ngram shard {idx} incomplete"))?;
            let path = dir.join(&file);
            let source = lisa_mlx::Shard::open(&path)?;
            let (weight_off, scales_off, biases_off, rows) = table_layout(&source, &w, &s, &b)?;
            maps.push(Some(ShardMap {
                map: source.mapping().clone(),
                weight_off,
                scales_off,
                biases_off,
                rows,
            }));
        }
        Ok(Self {
            maps,
            rows_per_shard: 0, // set by caller from constants
            row_dims: 160,
        })
    }

    /// Open a single merged `ngram.safetensors` (the merged pack:
    /// `ngram.weight/scales/biases`). The file is the concatenation of
    /// `shard_count` equal row-runs, so it is exposed as `shard_count` windows
    /// into one mmap — `gather` then addresses it exactly like the sharded form.
    pub fn open_merged(path: &Path, shard_count: usize) -> anyhow::Result<Self> {
        let source = lisa_mlx::Shard::open(path)?;
        let (w0, s0, b0, rows) =
            table_layout(&source, "ngram.weight", "ngram.scales", "ngram.biases")?;
        let mmap = source.mapping().clone();
        if shard_count == 0 || rows % shard_count != 0 {
            anyhow::bail!("merged ngram: {rows} rows not divisible by {shard_count} shards");
        }
        let rps = rows / shard_count;
        const W_BYTES: usize = 20 * 4; // 160 values / 8 per u32
        const G_BYTES: usize = 5 * 2; // 160 / 32 groups, bf16
        let mut maps = Vec::with_capacity(shard_count);
        for i in 0..shard_count {
            maps.push(Some(ShardMap {
                map: mmap.clone(),
                weight_off: w0 + i * rps * W_BYTES,
                scales_off: s0 + i * rps * G_BYTES,
                biases_off: b0 + i * rps * G_BYTES,
                rows: rps,
            }));
        }
        Ok(Self {
            maps,
            rows_per_shard: rps,
            row_dims: 160,
        })
    }

    pub fn set_rows_per_shard(&mut self, r: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            r > 0
                && self
                    .maps
                    .iter()
                    .all(|m| m.as_ref().is_some_and(|m| m.rows == r)),
            "n-gram storage rows differ from model hash geometry"
        );
        self.rows_per_shard = r;
        Ok(())
    }

    /// Gather rows for `global_ids` (row-major), returning three arrays:
    /// packed u32 [rows, 20], scales bf16 [rows, 5], biases bf16 [rows, 5].
    pub fn gather<D: BindingDevice>(
        &self,
        context: &DeviceContext<D>,
        global_ids: &[i64],
    ) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)> {
        let n = global_ids.len();
        const WEIGHT_WORDS: usize = 20; // 160 values / 8 per u32
        const GROUPS: usize = 5; // 160 / 32
        let mut packed = Vec::with_capacity(n * WEIGHT_WORDS);
        let mut scales = Vec::with_capacity(n * GROUPS);
        let mut biases = Vec::with_capacity(n * GROUPS);
        let rps = self.rows_per_shard as i64;
        anyhow::ensure!(rps > 0, "n-gram shard geometry is unbound");
        for &id in global_ids {
            anyhow::ensure!(id >= 0, "negative n-gram row id");
            let shard_idx = (id / rps) as usize;
            let row = (id % rps) as usize;
            let shard = self
                .maps
                .get(shard_idx)
                .and_then(Option::as_ref)
                .ok_or_else(|| anyhow::anyhow!("ngram shard {shard_idx} not mapped"))?;
            anyhow::ensure!(row < shard.rows, "n-gram row exceeds mapped shard");
            let w_start = shard.weight_off + row * WEIGHT_WORDS * 4;
            let s_start = shard.scales_off + row * GROUPS * 2;
            let b_start = shard.biases_off + row * GROUPS * 2;
            let wb = &shard.map[w_start..w_start + WEIGHT_WORDS * 4];
            let sb = &shard.map[s_start..s_start + GROUPS * 2];
            let bb = &shard.map[b_start..b_start + GROUPS * 2];
            for chunk in wb.as_chunks::<4>().0.iter() {
                packed.push(u32::from_le_bytes(*chunk));
            }
            for chunk in sb.as_chunks::<2>().0.iter() {
                scales.push(bf16::from_le_bytes(*chunk));
            }
            for chunk in bb.as_chunks::<2>().0.iter() {
                biases.push(bf16::from_le_bytes(*chunk));
            }
        }
        let packed = context.device().pull(
            lisa_backend::TensorSource::Borrowed(HostTensor {
                bytes: &packed
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
                shape: &[n as i32, WEIGHT_WORDS as i32],
                dtype: ScalarType::U32,
            }),
            Default::default(),
        )?;
        let scales = context.device().pull(
            lisa_backend::TensorSource::Borrowed(HostTensor {
                bytes: &scales
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
                shape: &[n as i32, GROUPS as i32],
                dtype: ScalarType::BF16,
            }),
            Default::default(),
        )?;
        let biases = context.device().pull(
            lisa_backend::TensorSource::Borrowed(HostTensor {
                bytes: &biases
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
                shape: &[n as i32, GROUPS as i32],
                dtype: ScalarType::BF16,
            }),
            Default::default(),
        )?;
        Ok((packed, scales, biases))
    }
}

impl<D: BindingDevice> NgramRows<D> for NgramTable {
    fn set_rows_per_shard(&mut self, rows: usize) -> anyhow::Result<()> {
        NgramTable::set_rows_per_shard(self, rows)
    }
    fn gather(
        &self,
        context: &DeviceContext<D>,
        ids: &[i64],
    ) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)> {
        NgramTable::gather(self, context, ids)
    }
}
