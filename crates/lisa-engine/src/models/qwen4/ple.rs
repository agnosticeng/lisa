//! The PLE (per-layer embedding) block and the sharded n-gram table.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use half::bf16;
use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{ops, Array, Dtype};
use memmap2::Mmap;

use crate::core::loader::TensorSource;
use crate::core::norm::RmsNorm;
use crate::core::quant::{QuantizedLinear, GROUP_SIZE, BITS};

// ---------------------------------------------------------------------------
// Host-side hash constants (rebuilt from configuration, exactly as the
// reference does — the checkpoint copies are never used).
// ---------------------------------------------------------------------------

pub struct NgramConstants {
    pub sizes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub multipliers: Vec<i64>,
    pub rows_per_shard: usize,
    pub shard_count: usize,
}

fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut d: usize = 3;
    while d.checked_mul(d).map_or(false, |dd| dd <= value) {
        if value % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn nth_prime_after(start: usize, count: usize) -> usize {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

impl NgramConstants {
    pub fn new(
        ngram_size: usize,
        heads_per_ngram: usize,
        vocab_size: usize,
        ngram_vocab_size_base: usize,
        divisible_by: usize,
        split_parts: usize,
        seed: i64,
        ple_layer_index: usize,
    ) -> Self {
        let ngram_heads = (ngram_size - 1) * heads_per_ngram;
        let mut sizes = Vec::with_capacity(ngram_heads);
        let mut offsets = Vec::with_capacity(ngram_heads);
        let mut total: usize = 0;
        for head in 0..ngram_heads {
            let global = ple_layer_index * ngram_heads + head;
            let size = nth_prime_after(ngram_vocab_size_base - 1, global + 1);
            sizes.push(size as i64);
            offsets.push(total as i64);
            total += size;
        }
        let padded = total.div_ceil(divisible_by) * divisible_by;
        let shard_count = split_parts;
        let rows_per_shard = padded.div_ceil(shard_count);

        let gamma: u64 = 0x9E37_79B9_7F4A_7C15;
        let max_long = i64::MAX as u64;
        let half = 1u64.max((max_long / vocab_size.max(1) as u64) / 2);
        let base_seed = (seed as u64).wrapping_add(10007u64.wrapping_mul(ple_layer_index as u64));
        let mut multipliers = Vec::with_capacity(ngram_size);
        for i in 0..ngram_size {
            let mixed = splitmix64(base_seed.wrapping_add(gamma.wrapping_mul((i + 1) as u64)));
            multipliers.push((2u64.wrapping_mul(mixed % half).wrapping_add(1)) as i64);
        }
        Self {
            sizes,
            offsets,
            multipliers,
            rows_per_shard,
            shard_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Row source: mmap'd shards, 100 bytes per row
// (80 packed u32 + 5 bf16 scales + 5 bf16 biases).
// ---------------------------------------------------------------------------

struct ShardMap {
    map: Arc<Mmap>,
    weight_off: usize,
    scales_off: usize,
    biases_off: usize,
}

/// Serves dequantized rows for global row ids from the checkpoint's n-gram
/// shard tensors (`...ngram_embedding.shard_N.{weight,scales,biases}`).
pub struct NgramTable {
    maps: Vec<Option<ShardMap>>,
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
        let mut shard_files: Vec<Option<(String, String, String, String)>> = vec![None; shard_count];
        let prefixes = [
            format!("{tensor_prefix}.shard_"),
            format!("language_model.{tensor_prefix}.shard_"),
        ];
        for (name, file) in weight_map {
            let Some(rest) = prefixes
                .iter()
                .find_map(|p| name.strip_prefix(p.as_str()))
            else {
                continue;
            };
            let Some((idx, tensor)) = rest.split_once('.') else {
                continue;
            };
            let idx: usize = idx.parse()?;
            match tensor {
                "weight" | "scales" | "biases" => {
                    let entry = &mut shard_files[idx];
                    let e = entry.get_or_insert_with(|| (file.clone(), String::new(), String::new(), String::new()));
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
            let (file, w, s, b) = entry.ok_or_else(|| anyhow::anyhow!("ngram shard {idx} incomplete"))?;
            let path = dir.join(&file);
            let f = std::fs::File::open(&path)?;
            let mmap = unsafe { Mmap::map(&f)? };
            // Header parse for the three tensors' offsets.
            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len])?;
            let abs = |off: u64| 8 + header_len + off as usize;
            let w_info = &header[w];
            let s_info = &header[s];
            let b_info = &header[b];
            let weight_off = abs(w_info["data_offsets"].as_array().unwrap()[0].as_u64().unwrap());
            let scales_off = abs(s_info["data_offsets"].as_array().unwrap()[0].as_u64().unwrap());
            let biases_off = abs(b_info["data_offsets"].as_array().unwrap()[0].as_u64().unwrap());
            maps.push(Some(ShardMap {
                map: Arc::new(mmap),
                weight_off,
                scales_off,
                biases_off,
            }));
        }
        Ok(Self {
            maps,
            rows_per_shard: 0, // set by caller from constants
            row_dims: 160,
        })
    }

    /// Open a single merged `ngram.safetensors` (the mlx-serve pack:
    /// `ngram.weight/scales/biases`). The file is the concatenation of
    /// `shard_count` equal row-runs, so it is exposed as `shard_count` windows
    /// into one mmap — `gather` then addresses it exactly like the sharded form.
    pub fn open_merged(path: &Path, shard_count: usize) -> anyhow::Result<Self> {
        let f = std::fs::File::open(path)?;
        let mmap = Arc::new(unsafe { Mmap::map(&f)? });
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len])?;
        let abs = |off: u64| 8 + header_len + off as usize;
        let w0 = abs(header["ngram.weight"]["data_offsets"][0].as_u64().unwrap());
        let s0 = abs(header["ngram.scales"]["data_offsets"][0].as_u64().unwrap());
        let b0 = abs(header["ngram.biases"]["data_offsets"][0].as_u64().unwrap());
        let rows = header["ngram.weight"]["shape"][0].as_u64().unwrap() as usize;
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
            }));
        }
        Ok(Self {
            maps,
            rows_per_shard: rps,
            row_dims: 160,
        })
    }

    pub fn set_rows_per_shard(&mut self, r: usize) {
        self.rows_per_shard = r;
    }

    /// Gather rows for `global_ids` (row-major), returning three arrays:
    /// packed u32 [rows, 20], scales bf16 [rows, 5], biases bf16 [rows, 5].
    pub fn gather(&self, global_ids: &[i64]) -> anyhow::Result<(Array, Array, Array)> {
        let n = global_ids.len();
        const WEIGHT_WORDS: usize = 20; // 160 values / 8 per u32
        const GROUPS: usize = 5; // 160 / 32
        let mut packed = Vec::with_capacity(n * WEIGHT_WORDS);
        let mut scales = Vec::with_capacity(n * GROUPS);
        let mut biases = Vec::with_capacity(n * GROUPS);
        let rps = self.rows_per_shard as i64;
        for &id in global_ids {
            let shard_idx = (id / rps) as usize;
            let row = (id % rps) as usize;
            let shard = self.maps[shard_idx]
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("ngram shard {shard_idx} not mapped"))?;
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
        let packed = Array::from_slice(&packed, &[n as i32, WEIGHT_WORDS as i32]);
        let scales = Array::from_slice(&scales, &[n as i32, GROUPS as i32]);
        let biases = Array::from_slice(&biases, &[n as i32, GROUPS as i32]);
        Ok((packed, scales, biases))
    }
}

/// The n-gram embedding: host hash -> row gather -> dequantize.
pub struct NgramEmbedding {
    pub constants: NgramConstants,
    pub table: std::sync::Arc<std::sync::RwLock<NgramTable>>,
    pub ngram_heads: usize,
    pub eos_token_id: i64,
    pub head_dimensions: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
}

impl NgramEmbedding {
    /// Row ids on the host for one request row.
    ///
    /// `history` is `previous_context ++ ids` (int64); returns ids for the
    /// last `new_count` positions, `[new_count][ngram_heads]` flattened
    /// row-major. Bit-for-bit the device result: wrapping i64 multiply, XOR,
    /// Python-style remainder.
    pub fn host_row_ids(&self, history: &[i64], new_count: usize) -> Vec<i64> {
        let eos = self.eos_token_id;
        let t = history.len();
        // previous[t] = position of the last EOS strictly before t, or -1.
        let mut previous = vec![-1i64; t];
        let mut last = -1i64;
        for (i, &tok) in history.iter().enumerate() {
            previous[i] = last;
            if tok == eos {
                last = i as i64;
            }
        }
        let shifted = |s: usize, ti: usize| -> i64 {
            if s == 0 {
                return history[ti];
            }
            let in_segment = ti as i64 - (previous[ti] + 1);
            let source = ti as i64 - s as i64;
            if in_segment >= s as i64 && source >= 0 {
                history[source as usize]
            } else {
                eos
            }
        };

        let heads_per_ngram = self.heads_per_ngram;
        let mut out = Vec::with_capacity(new_count * self.ngram_heads);
        for ti in t.saturating_sub(new_count)..t {
            for ngram in 2..=self.ngram_size {
                let mut mixed = shifted(0, ti).wrapping_mul(self.constants.multipliers[0]);
                for p in 1..ngram {
                    mixed ^= shifted(p, ti).wrapping_mul(self.constants.multipliers[p]);
                }
                let low = (ngram - 2) * heads_per_ngram;
                for head in low..low + heads_per_ngram {
                    let size = self.constants.sizes[head];
                    let mut r = mixed % size;
                    if r != 0 && (r < 0) != (size < 0) {
                        r += size;
                    }
                    out.push(r + self.constants.offsets[head]);
                }
            }
        }
        out
    }

    /// Forward: ids [B, S], previous_context [B, ctx] -> [B, S, 2560].
    pub fn forward(&self, ids: &[Vec<i64>], previous_context: &[Vec<i64>]) -> anyhow::Result<Array> {
        let b = ids.len();
        let s = ids[0].len();
        let ctx = previous_context[0].len();
        let mut all_ids = Vec::with_capacity(b * s * self.ngram_heads);
        for (bi, row_ids) in ids.iter().enumerate() {
            let mut history = previous_context[bi].clone();
            history.extend_from_slice(row_ids);
            let mut row = self.host_row_ids(&history, s);
            all_ids.append(&mut row);
        }
        let _ = ctx;
        if std::env::var("LISA_DEBUG_ROWS").is_ok() {
            println!("RS mults: {:?} sizes[:4]: {:?} half: {}", self.constants.multipliers, &self.constants.sizes[..4], ((i64::MAX as u64) / 248320) / 2);
            println!("RS row ids[:32]: {:?}", &all_ids[..all_ids.len().min(32)]);
        }
        let (packed, scales, biases) = {
            let table = self.table.read().unwrap();
            table.gather(&all_ids)?
        };
        let rows = ops::dequantize(&packed, &scales, &biases, GROUP_SIZE, BITS)?;
        rows.reshape(&[b as i32, s as i32, -1]).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// PLE layer: n-gram embedding, gated against the hidden stream, plus a
/// dilated short convolution.
pub struct PleLayer {
    pub embedding: NgramEmbedding,
    pub key_proj: QuantizedLinear,
    pub value_proj: QuantizedLinear,
    pub norm_key: RmsNorm,
    pub norm_query: RmsNorm,
    pub norm_conv: RmsNorm,
    pub conv1d_weight: Array, // [wide, 4, 1]
    pub hidden_size: usize,
    pub hc_count: usize,
    pub dilation: usize,
    pub short_conv_state_length: usize,
    /// Debug: last computed embedding rows (pre-projection).
    pub last_embed: Option<Array>,
    /// Debug: last computed gated stream.
    pub last_gated: Option<Array>,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load<S: TensorSource>(
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
        let _wide = hidden_size * hc_count;
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
        table.write().unwrap().set_rows_per_shard(constants.rows_per_shard);

        let mut conv1d_weight = src.get_bf16(&format!("{prefix}.conv1d.weight"))?;
        let cs = conv1d_weight.shape();
        if cs.len() == 3 && cs[1] == 1 && cs[2] > 1 {
            conv1d_weight = conv1d_weight.transpose_axes(&[0, 2, 1])?;
        }
        Ok(Self {
            embedding: NgramEmbedding {
                constants,
                table,
                ngram_heads: (ngram_size - 1) * heads_per_ngram,
                eos_token_id: 0, // set by the model from config
                head_dimensions: ple_embed_dim / ((ngram_size - 1) * heads_per_ngram),
                ngram_size,
                heads_per_ngram,
            },
            key_proj: QuantizedLinear::load(src, prefix, "key_proj")?,
            value_proj: QuantizedLinear::load(src, prefix, "value_proj")?,
            norm_key: RmsNorm::load(src, &format!("{prefix}.norm_key"), eps, Some(hidden_size))?,
            norm_query: RmsNorm::load(src, &format!("{prefix}.norm_query"), eps, Some(hidden_size))?,
            norm_conv: RmsNorm::load(src, &format!("{prefix}.norm_conv"), eps, Some(hidden_size))?,
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

    fn short_conv(&mut self, x: &Array, conv_state: &mut Option<Array>) -> lisa_mlx::error::Result<Array> {
        let s = x.dim(1) as usize;
        let n = self.short_conv_state_length;
        let state = match conv_state {
            Some(st) => st.clone(),
            None => ops::zeros::<half::bf16>(&[
                x.dim(0),
                n as i32,
                x.dim(-1),
            ])?,
        };
        let full = ops::concatenate(&[&state, x], 1)?;
        // keep the last n rows as the new state
        let tail = full
            .index((Ellipsis, (full.dim(1) - n as i32).., ..))
            .contiguous()?;
        *conv_state = Some(tail);
        // conv over the last n+S rows (= all of full), kernel 4, dilation 3
        let start = full.dim(1) - (n + s) as i32;
        let window = full.index((Ellipsis, start.., ..)).contiguous()?;
        let conv = ops::conv1d(
            &window,
            &self.conv1d_weight,
            None,
            None,
            Some(self.dilation as i32),
            Some(window.dim(-1)),
        )?;
        let conv = lisa_mlx::nn::silu(&conv)?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            let f = conv.as_dtype(Dtype::Float32).unwrap();
            let a = f.as_slice::<f32>();
            let _ = std::fs::write(
                format!("{dir}/rs_ple_conv.bin"),
                unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
            );
        }
        Ok(conv)
    }

    /// hidden: the full hyper stream [B, S, wide]; ids/ctx for the hash.
    pub fn forward(
        &mut self,
        hidden: &Array,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
        conv_state: &mut Option<Array>,
        capture: bool,
        capture_full: &mut Option<Array>,
    ) -> anyhow::Result<Array> {
        let b = hidden.dim(0);
        let s = hidden.dim(1);
        let h = self.hidden_size as i32;

        let embedded = self.embedding.forward(ids, previous_context)?;
        let embedded = embedded.as_dtype(hidden.dtype()).map_err(|e| anyhow::anyhow!("{e}"))?;
        self.last_embed = Some(embedded.clone());
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            let f = embedded.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
            let a = f.as_slice::<f32>();
            let _ = std::fs::write(format!("{dir}/rs_ple_embed.bin"), unsafe {
                std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
            });
        }

        // Three fused kernels replace the ~22-launch op chain (see the engine's
        // `TrackFastPLE.swift`); the reduction between prod and gated stays MLX.
        let fused = true;
        if fused {
            let hc = self.hc_count as i32;
            let hidden_i = self.hidden_size as i32;
            let w = hc * hidden_i;
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let eps1 = Array::from_f32(self.norm_key.eps)
                .as_dtype(Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .reshape(&[1])?;
            let key_raw = self.key_proj.forward(&embedded)?;
            let value = self.value_proj.forward(&embedded)?;
            if let Some(prod) = lisa_mlx::kernels::ple_prod(
                &key_raw,
                hidden,
                &self.norm_key.weight,
                &self.norm_query.weight,
                &eps1,
                hc,
                hidden_i,
                &stream,
            ) {
                // g0 = the sum over the last axis (MLX's own reduction).
                let g0 = prod.reshape(&[b, s, hc, hidden_i])?.sum_axis(-1, None)?;
                let divisor = Array::from_f32((self.hidden_size as f32).sqrt())
                    .as_dtype(Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .reshape(&[1])?;
                let floor = Array::from_f32(1e-6)
                    .as_dtype(Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .reshape(&[1])?;
                if let Some((gated, normed)) = lisa_mlx::kernels::ple_gated(
                    &g0,
                    &value,
                    &self.norm_conv.weight,
                    &divisor,
                    &floor,
                    &eps1,
                    hc,
                    hidden_i,
                    &stream,
                ) {
                    self.last_gated = Some(gated.clone());
                    let n = self.short_conv_state_length;
                    let state = match conv_state.as_ref() {
                        Some(st) => st.clone(),
                        None => ops::zeros::<half::bf16>(&[b, n as i32, w])?,
                    };
                    let full = ops::concatenate(&[&state, &normed], 1)?;
                    if capture {
                        *capture_full = Some(full.clone());
                    }
                    let tail = full
                        .index((Ellipsis, (full.dim(1) - n as i32).., ..))
                        .contiguous()?;
                    *conv_state = Some(tail);
                    if let Some(out) = lisa_mlx::kernels::ple_conv(
                        &full,
                        &self.conv1d_weight,
                        &gated,
                        self.dilation as i32,
                        &stream,
                    ) {
                        return Ok(out);
                    }
                }
            }
        }

        // S == 1: the engine's PLE fusion (`track_ple_prepare_fuse2` +
        // `track_ple_convolution_fuse2`).
        // The fuse2 kernels declare B=1 outputs, so batched windows take the
        // generic path.
        if b == 1 && s == 1 && self.hidden_size == 2560 && self.hc_count == 4 {
            let key = self.key_proj.forward(&embedded)?;
            let value = self.value_proj.forward(&embedded)?;
            if key.dim(-1) == 10240
                && value.dim(-1) == 2560
                && hidden.dim(-1) == 10240
                && self.dilation == 3
                && self.conv1d_weight.shape() == [10240, 4, 1]
            {
                let state = match conv_state.as_ref() {
                    Some(st) => st.clone(),
                    None => ops::zeros::<half::bf16>(&[b, 9, 10240])?,
                };
                let stream = lisa_mlx::Stream::thread_local_or_default();
                if let Some((full, out)) = lisa_mlx::kernels::ple_fuse2(
                    &key, hidden, &value,
                    &self.norm_key.weight, &self.norm_query.weight, &self.norm_conv.weight,
                    &state, &self.conv1d_weight, self.norm_key.eps, &stream,
                ) {
                    *conv_state = Some(full.index((.., 1..10, ..)).contiguous()?);
                    self.last_gated = None;
                    return Ok(out.reshape(&[b, s, self.hidden_size as i32])?);
                }
            }
        }

        let key_flat = self.norm_key.forward(&self.key_proj.forward(&embedded)?)?;
        let key = key_flat.reshape(&[b, s, self.hc_count as i32, h])?;
        let value = self.value_proj.forward(&embedded)?;
        let query_flat = self.norm_query.forward(hidden)?;
        let query = query_flat.reshape(&[b, s, self.hc_count as i32, h])?;

        let mut gate = key.multiply(&query)?.sum_axis(-1, Some(true))?;
        gate = gate / crate::core::norm::bf16_scalar((self.hidden_size as f32).sqrt());
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            let f = gate.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
            let a = f.as_slice::<f32>();
            let _ = std::fs::write(
                format!("{dir}/rs_ple_gate_raw.bin"),
                unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
            );
        }
        // Floor scalar built IN THE GATE'S DTYPE (bf16) — an f32 scalar here
        // would silently promote the whole stream to f32.
        let floor = Array::from_f32(1e-6).as_dtype(gate.dtype()).map_err(|e| anyhow::anyhow!("{e}"))?;
        gate = ops::sqrt(&ops::maximum(ops::abs(&gate)?, &floor)?)?.multiply(&ops::sign(&gate)?)?;

        let sg = ops::sigmoid(&gate)?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            for (nm, arr) in [("rs_ple_sigmoid", &sg), ("rs_ple_value", &value)] {
                let f = arr.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                let a = f.as_slice::<f32>();
                let _ = std::fs::write(
                    format!("{dir}/{nm}.bin"),
                    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                );
            }
        }
        let gated = sg.multiply(&value.expand_dims(-2)?)?;
        let gated = gated.reshape(&[b, s, -1])?;
        self.last_gated = Some(gated.clone());
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            let f = gated.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
            let a = f.as_slice::<f32>();
            let _ = std::fs::write(format!("{dir}/rs_ple_gated.bin"), unsafe {
                std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
            });
        }
        let conv = self.short_conv(&self.norm_conv.forward(&gated)?, conv_state)?;
        gated.add(&conv).map_err(|e| anyhow::anyhow!("{e}"))
    }
}
