//! The PLE (per-layer embedding) block and the sharded n-gram table.

use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{HostTensor, ScalarType, TensorAccess};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use half::bf16;

use memmap2::Mmap;

use super::normalization::{RmsNorm, load_bf16};
use crate::weights::affine::{BITS, GROUP_SIZE, QuantizedLinear};
use crate::weights::source::TensorSource;

// ---------------------------------------------------------------------------
// Host-side hash constants (rebuilt from configuration — the checkpoint
// copies are never used).
// ---------------------------------------------------------------------------

pub use lisa_models::qwen4::ngram::NgramConstants;

// ---------------------------------------------------------------------------
// Row source: mmap'd shards, 100 bytes per row
// (80 packed u32 + 5 bf16 scales + 5 bf16 biases).
// ---------------------------------------------------------------------------

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
            for chunk in wb.chunks_exact(4) {
                packed.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
            for chunk in sb.chunks_exact(2) {
                scales.push(bf16::from_le_bytes(chunk.try_into().unwrap()));
            }
            for chunk in bb.chunks_exact(2) {
                biases.push(bf16::from_le_bytes(chunk.try_into().unwrap()));
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

/// The n-gram embedding: host hash -> row gather -> dequantize.
pub struct NgramEmbedding<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub constants: NgramConstants,
    pub table: std::sync::Arc<std::sync::RwLock<NgramTable>>,
    pub ngram_heads: usize,
    pub eos_token_id: i64,
    pub head_dimensions: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
}

impl<D: BindingDevice> NgramEmbedding<D> {
    /// Row ids on the host for one request row.
    ///
    /// `history` is `previous_context ++ ids` (int64); returns ids for the
    /// last `new_count` positions, `[new_count][ngram_heads]` flattened
    /// row-major. Bit-for-bit the device result: wrapping i64 multiply, XOR,
    /// Python-style remainder.
    pub fn host_row_ids(&self, history: &[i64], new_count: usize) -> Vec<i64> {
        self.constants.row_ids(
            history,
            new_count,
            self.eos_token_id,
            self.ngram_size,
            self.heads_per_ngram,
        )
    }

    fn embedding_ids(&self, ids: &[Vec<i64>], history: &[Vec<i64>]) -> anyhow::Result<Vec<i64>> {
        self.constants.batch_ids(
            ids,
            history,
            self.eos_token_id,
            self.ngram_size,
            self.heads_per_ngram,
        )
    }

    /// Forward: ids [B, S], previous_context [B, ctx] -> [B, S, 2560].
    pub fn forward(
        &self,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
    ) -> anyhow::Result<D::Tensor> {
        let b = ids.len();
        let s = ids.first().map_or(0, Vec::len);
        let all_ids = self.embedding_ids(ids, previous_context)?;
        let (packed, scales, biases) = {
            let table = self.table.read().unwrap();
            table.gather(&self.context, &all_ids)?
        };
        self.context.with_slot(
            "qwen4.ple.plans",
            || {
                std::cell::RefCell::new(crate::core::graph::PlanCache::new(
                    self.context.device().clone(),
                    64,
                ))
            },
            |cache| {
                cache
                    .borrow_mut()
                    .get_or_prepare(Key::Embedding(b as i32, s as i32), || {
                        lisa_models::qwen4::ple::embedding(b as i32, s as i32, GROUP_SIZE, BITS)
                    })?
                    .run(&[&packed, &scales, &biases], &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)
            },
        )
    }
}

/// PLE layer: n-gram embedding, gated against the hidden stream, plus a
/// dilated short convolution.
pub struct PleLayer<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub embedding: NgramEmbedding<D>,
    pub key_proj: QuantizedLinear<D::Tensor>,
    pub value_proj: QuantizedLinear<D::Tensor>,
    pub norm_key: RmsNorm<D::Tensor>,
    pub norm_query: RmsNorm<D::Tensor>,
    pub norm_conv: RmsNorm<D::Tensor>,
    pub conv1d_weight: D::Tensor, // [wide, 4, 1]
    pub hidden_size: usize,
    pub hc_count: usize,
    pub dilation: usize,
    pub short_conv_state_length: usize,
    /// Debug: last computed embedding rows (pre-projection).
    pub last_embed: Option<D::Tensor>,
    /// Debug: last computed gated stream.
    pub last_gated: Option<D::Tensor>,
}

impl<D: BindingDevice> PleLayer<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        eps: f32,
        hidden_size: usize,
        hc_count: usize,
        ngram_size: usize,
        heads_per_ngram: usize,
        ple_embed_dim: usize,
        ple_conv_kernel_size: usize,
        ple_layer_index: usize,
        seed: i64,
        vocab_size: usize,
        ngram_vocab_size_base: usize,
        divisible_by: usize,
        split_parts: usize,
        table: std::sync::Arc<std::sync::RwLock<NgramTable>>,
    ) -> anyhow::Result<Self> {
        let names = lisa_models::qwen4::ple::names(prefix);
        let constants = NgramConstants::new(
            ngram_size,
            heads_per_ngram,
            vocab_size,
            ngram_vocab_size_base,
            divisible_by,
            split_parts,
            seed,
            ple_layer_index,
        );
        table
            .write()
            .unwrap()
            .set_rows_per_shard(constants.rows_per_shard)?;

        let conv1d_weight = load_bf16(&context, src, &names.convolution)?;
        let prepare = crate::core::graph::PreparedGraph::new(
            context.device().clone(),
            lisa_models::qwen4::ple::prepare_convolution(conv1d_weight.shape()),
            vec![],
        )?;
        let conv1d_weight = prepare
            .run(&[&conv1d_weight], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            context: context.clone(),
            embedding: NgramEmbedding {
                context: context.clone(),
                constants,
                table,
                ngram_heads: (ngram_size - 1) * heads_per_ngram,
                eos_token_id: 0, // set by the model from config
                head_dimensions: ple_embed_dim / ((ngram_size - 1) * heads_per_ngram),
                ngram_size,
                heads_per_ngram,
            },
            key_proj: QuantizedLinear::load(src, prefix, names.key)?,
            value_proj: QuantizedLinear::load(src, prefix, names.value)?,
            norm_key: RmsNorm::load_on(&context, src, &names.norm_key, eps, Some(hidden_size))?,
            norm_query: RmsNorm::load_on(&context, src, &names.norm_query, eps, Some(hidden_size))?,
            norm_conv: RmsNorm::load_on(
                &context,
                src,
                &names.norm_convolution,
                eps,
                Some(hidden_size),
            )?,
            conv1d_weight,
            hidden_size,
            hc_count,
            dilation: ngram_size,
            short_conv_state_length: (ple_conv_kernel_size - 1) * ngram_size,
            last_embed: None,
            last_gated: None,
        })
    }

    pub fn set_eos(&mut self, eos: i64) {
        self.embedding.eos_token_id = eos;
    }

    pub fn forward(
        &mut self,
        hidden: &D::Tensor,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
        conv_state: &mut Option<D::Tensor>,
        capture: bool,
        capture_full: &mut Option<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        let embedded = self.embedding.forward(ids, previous_context)?;
        let spec = lisa_models::qwen4::ple::Spec {
            hidden: self.hidden_size as i32,
            streams: self.hc_count as i32,
            dilation: self.dilation as i32,
            state_rows: self.short_conv_state_length as i32,
            epsilon: self.norm_key.eps.to_bits(),
            key: (self.key_proj.group_size, self.key_proj.bits),
            value: (self.value_proj.group_size, self.value_proj.bits),
        };
        let mut inputs = vec![
            hidden,
            &embedded,
            &self.key_proj.weight,
            &self.key_proj.scales,
            &self.key_proj.biases,
            &self.value_proj.weight,
            &self.value_proj.scales,
            &self.value_proj.biases,
            &self.norm_key.weight,
            &self.norm_query.weight,
            &self.norm_conv.weight,
            &self.conv1d_weight,
        ];
        let has_state = conv_state.is_some();
        if let Some(state) = conv_state.as_ref() {
            inputs.push(state);
        }
        let mut output = self
            .context
            .with_slot(
                "qwen4.ple.plans",
                || {
                    std::cell::RefCell::new(crate::core::graph::PlanCache::new(
                        self.context.device().clone(),
                        64,
                    ))
                },
                |cache| {
                    cache
                        .borrow_mut()
                        .try_get_or_prepare(
                            Key::Forward(spec, hidden.shape().to_vec(), has_state),
                            || spec.graph(hidden.shape(), has_state),
                        )?
                        .run(&inputs, &mut [])
                },
            )?
            .into_iter();
        let result = output.next().unwrap();
        self.last_embed = output.next();
        self.last_gated = output.next();
        *conv_state = output.next();
        if capture {
            *capture_full = output.next();
        }
        Ok(result)
    }
}
#[derive(PartialEq)]
enum Key {
    Forward(lisa_models::qwen4::ple::Spec, Vec<i32>, bool),
    Embedding(i32, i32),
}
