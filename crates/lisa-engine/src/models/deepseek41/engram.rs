//! DS4's n-gram history and row lookup, with original IQ4_NL storage support.
//! Arithmetic is ported from ds4_engram.c at `super::DS4_REVISION` (MIT).

use super::config::Metadata;
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Header, Package, Tensor, TensorType, quant};
use rayon::prelude::*;

pub const LAYERS: [usize; 2] = [1, 14];
pub const HEADS: usize = 24;
pub const WIDTH: usize = 256;
pub type RowIds = [[u32; HEADS]; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct History {
    /// Compressed token IDs, newest first; -1 breaks an n-gram.
    tail: [i32; 3],
}

impl Default for History {
    fn default() -> Self {
        Self { tail: [-1; 3] }
    }
}

impl History {
    pub fn tail(&self) -> [i32; 3] {
        self.tail
    }
    pub fn restore(tail: [i32; 3], layout: &Layout) -> Result<Self> {
        ensure!(
            tail.iter()
                .all(|&x| x >= -1 && (x < 0 || (x as u32) < layout.compressed_vocab)),
            "invalid Engram history"
        );
        Ok(Self { tail })
    }
}

#[derive(Clone, Debug)]
pub struct Layout {
    token_map: Vec<u32>,
    compressed_vocab: u32,
    pad: u32,
    multipliers: [[u64; 4]; 2],
    primes: [[u32; HEADS]; 2],
    offsets: [[u32; HEADS]; 2],
    pub rows: [u32; 2],
    pub combined: bool,
}

impl Layout {
    pub fn from_header(header: &Header) -> Result<Self> {
        let m = Metadata(header);
        ensure!(
            m.integers(&["ple.layers", "engram.layer_ids"])? == [1, 14],
            "unsupported Engram layers"
        );
        let combined = header.metadata.contains_key("deepseek41.ple.layers");
        if combined {
            ensure!(
                m.integer(&["ple.ngram_size"])? == 4
                    && m.integer(&["ple.heads_per_ngram"])? == 8
                    && m.integer(&["embedding_length_per_layer_input"])? == WIDTH as u64,
                "unsupported Engram geometry"
            );
        }
        let token_map: Vec<u32> = m
            .integers(&["engram.token_map"])?
            .into_iter()
            .map(u32::try_from)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(!token_map.is_empty(), "empty Engram token map");
        let compressed_vocab = token_map
            .iter()
            .copied()
            .max()
            .unwrap()
            .checked_add(1)
            .context("Engram vocabulary overflow")?;
        ensure!(
            compressed_vocab <= i32::MAX as u32,
            "Engram vocabulary exceeds int32"
        );
        if header
            .metadata
            .contains_key("deepseek41.engram.compressed_vocab_size")
        {
            ensure!(
                m.integer(&["engram.compressed_vocab_size"])? == compressed_vocab as u64,
                "Engram compressed vocabulary mismatch"
            );
        }
        let pad = u32::try_from(m.integer(&["engram.pad_token_id", "engram.pad_id"])?)?;
        ensure!(pad < compressed_vocab, "invalid Engram padding token");
        let mult = m.integers(&["ple.layer_multipliers", "engram.multipliers"])?;
        let sizes = m.integers(&["ple.head_vocab_sizes", "engram.primes"])?;
        ensure!(
            mult.len() == 8 && sizes.len() == 48,
            "invalid Engram hash geometry"
        );
        let mut multipliers = [[0; 4]; 2];
        let mut primes = [[0; HEADS]; 2];
        let mut offsets = [[0; HEADS]; 2];
        let mut rows = [0u32; 2];
        for layer in 0..2 {
            for j in 0..4 {
                let value = mult[4 * layer + j];
                ensure!(
                    value & 1 == 1 && value <= i64::MAX as u64 / compressed_vocab as u64,
                    "invalid Engram multiplier"
                );
                multipliers[layer][j] = value;
            }
            for head in 0..HEADS {
                let prime = u32::try_from(sizes[layer * HEADS + head])?;
                ensure!(prime >= 2, "invalid Engram bucket count");
                primes[layer][head] = prime;
                offsets[layer][head] = rows[layer];
                rows[layer] = rows[layer]
                    .checked_add(prime)
                    .context("Engram row count overflow")?;
            }
        }
        if combined {
            let declared = m.integers(&["ple.head_offsets"])?;
            ensure!(declared.len() == 48, "invalid Engram head offsets");
            for layer in 0..2 {
                let base = if layer == 0 { 0 } else { rows[0] };
                for head in 0..HEADS {
                    offsets[layer][head] = offsets[layer][head]
                        .checked_add(base)
                        .context("Engram offset overflow")?;
                    ensure!(
                        declared[layer * HEADS + head] == offsets[layer][head] as u64,
                        "Engram head offsets are not contiguous in layer order"
                    );
                }
            }
        } else {
            ensure!(
                m.integers(&["engram.rows"])? == rows.map(u64::from),
                "Engram row metadata mismatch"
            );
        }
        ensure!(
            rows[0] as u64 + rows[1] as u64 <= i32::MAX as u64,
            "Engram rows exceed int32"
        );
        Ok(Self {
            token_map,
            compressed_vocab,
            pad,
            multipliers,
            primes,
            offsets,
            rows,
            combined,
        })
    }

    /// Validate the entire request before changing history. Masked image tokens
    /// break every n-gram crossing that position, as in DS4.
    pub fn hash(
        &self,
        history: &mut History,
        tokens: &[u32],
        mask: Option<&[u8]>,
    ) -> Result<Vec<RowIds>> {
        ensure!(
            tokens.len() <= 8192,
            "Engram hash request exceeds the DS4 prefill chunk"
        );
        ensure!(
            mask.is_none_or(|m| m.len() == tokens.len()),
            "Engram mask length mismatch"
        );
        ensure!(
            tokens.iter().all(|&t| (t as usize) < self.token_map.len()),
            "Engram token out of range"
        );
        History::restore(history.tail, self)?;
        let mut result = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            let current = if mask.is_some_and(|m| m[i] == 0) {
                -1
            } else {
                self.token_map[token as usize] as i32
            };
            let mut ids = [0u32; 4];
            let mut blocked = false;
            for j in 0..4 {
                let id = if j == 0 { current } else { history.tail[j - 1] };
                blocked |= id == -1;
                ids[j] = if blocked { self.pad } else { id as u32 };
            }
            let mut rows = [[0; HEADS]; 2];
            for (layer, out) in rows.iter_mut().enumerate() {
                let mut hash = ids[0] as u64 * self.multipliers[layer][0];
                for j in 1..4 {
                    hash ^= ids[j] as u64 * self.multipliers[layer][j];
                    for head in 0..8 {
                        let col = (j - 1) * 8 + head;
                        out[col] = (hash % self.primes[layer][col] as u64) as u32
                            + self.offsets[layer][col];
                    }
                }
            }
            history.tail = [current, history.tail[0], history.tail[1]];
            result.push(rows);
        }
        Ok(result)
    }
}

/// Disk-only table handles: no mmap, residency reservation or GPU buffer.
#[derive(Clone)]
pub struct Tables {
    tables: [Tensor; 2],
    readers: [std::sync::Arc<rayon::ThreadPool>; 2],
}

impl Tables {
    pub fn open(package: &Package, layout: &Layout) -> Result<Self> {
        let tables = if layout.combined {
            let tensor = package.tensor("per_layer_token_embd.weight")?;
            ensure!(
                tensor.info.dimensions
                    == [WIDTH as u64, layout.rows[0] as u64 + layout.rows[1] as u64]
                    && tensor.info.kind == TensorType::IQ4_NL,
                "unsupported combined Engram table layout"
            );
            let reader = tensor.disk_reader()?;
            [reader.clone(), reader]
        } else {
            let mut readers = Vec::new();
            for (index, layer) in LAYERS.into_iter().enumerate() {
                let tensor = package.tensor(&format!("blk.{layer}.engram_embd.weight"))?;
                ensure!(
                    tensor.info.dimensions == [264, layout.rows[index] as u64]
                        && tensor.info.kind.0 == 24,
                    "unsupported DS4 Engram table layout"
                );
                readers.push(tensor.disk_reader()?);
            }
            readers
                .try_into()
                .map_err(|_| anyhow::anyhow!("missing Engram reader"))?
        };
        Self::from_readers(tables)
    }

    fn from_readers(tables: [Tensor; 2]) -> Result<Self> {
        let pool = |layer| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(16)
                .thread_name(move |i| format!("engram-{layer}-{i}"))
                .build()
                .map(std::sync::Arc::new)
        };
        Ok(Self {
            tables,
            readers: [pool(0)?, pool(1)?],
        })
    }

    /// Metal DS4 starts both disk reads before layer 0 and joins each at its
    /// Engram layer (1 and 14). Each table has at most sixteen read workers.
    /// The returned job owns all host buffers and joins on drop, including
    /// failures or cancellation before the graph reaches an Engram layer.
    pub fn prefetch(&self, ids: &[RowIds]) -> Result<ReadAhead> {
        ensure!(
            !ids.is_empty() && ids.len() <= 8192,
            "invalid Engram prefetch size"
        );
        for layer in 0..2 {
            ensure!(
                ids.iter().all(|id| id[layer]
                    .iter()
                    .all(|&r| (r as u64) < self.tables[layer].info.rows())),
                "Engram prefetch row out of range"
            );
        }
        let ids: std::sync::Arc<[RowIds]> = ids.into();
        let mut job = ReadAhead {
            tasks: [None, None],
        };
        for layer in 0..2 {
            let tables = self.clone();
            let ids = ids.clone();
            job.tasks[layer] = Some(
                std::thread::Builder::new()
                    .name(format!("engram-prefetch-{layer}"))
                    .spawn(move || {
                        let mut output = vec![0f32; ids.len() * HEADS * WIDTH];
                        tables.read(&ids, layer, &mut output)?;
                        Ok(output)
                    })?,
            );
        }
        Ok(job)
    }

    /// DS4's sorted, deduplicated row reads in blocks of 2048 tokens. The
    /// sorting scratch is bounded independently of context and table size.
    pub fn read(&self, ids: &[RowIds], layer: usize, output: &mut [f32]) -> Result<()> {
        ensure!(
            layer < 2 && ids.len() <= 8192,
            "invalid Engram read request"
        );
        ensure!(
            output.len() == ids.len() * HEADS * WIDTH,
            "invalid Engram output size"
        );
        let table = &self.tables[layer];
        ensure!(
            ids.iter()
                .all(|id| id[layer].iter().all(|&r| (r as u64) < table.info.rows())),
            "Engram row out of range"
        );
        for (block, out) in ids
            .chunks(2048)
            .zip(output.chunks_mut(2048 * HEADS * WIDTH))
        {
            let mut requests = Vec::with_capacity(block.len() * HEADS);
            for (token, ids) in block.iter().enumerate() {
                for (head, &row) in ids[layer].iter().enumerate() {
                    requests.push((row, (token * HEADS + head) as u32));
                }
            }
            requests.sort_unstable();
            // DS4 uses sixteen readers per table from eight rows upward on
            // macOS. Private pools preserve that bound when the tables overlap.
            let pool = &self.readers[layer];
            let readers = if requests.len() >= 8 {
                requests.len().min(16)
            } else {
                1
            };
            let scatter = Scatter::new(out);
            let read_part = |part: usize| -> Result<()> {
                let begin = requests.len() * part / readers;
                let end = requests.len() * (part + 1) / readers;
                let mut packed = [0u8; 264];
                let bytes = table.info.row_bytes()? as usize;
                ensure!(bytes <= packed.len(), "unsupported Engram row size");
                let mut decoded = [0f32; WIDTH];
                let mut previous = None;
                for &(row, destination) in &requests[begin..end] {
                    if previous != Some(row) {
                        table.read_rows(&[row as u64], &mut packed[..bytes])?;
                        if table.info.kind == TensorType::IQ4_NL {
                            quant::decode(table.info.kind, &packed[..bytes], &mut decoded)?;
                        } else {
                            decode_mxfp8(&packed[..bytes], &mut decoded)?;
                        }
                        previous = Some(row);
                    }
                    // SAFETY: request destinations are the unique enumeration
                    // 0..block.len()*HEADS. Sorting only permutes them, and the
                    // workers receive disjoint ranges. The pool joins before
                    // the output slice is accessed again or released.
                    unsafe {
                        scatter.write(destination as usize, &decoded);
                    }
                }
                Ok(())
            };
            if readers == 1 {
                read_part(0)?;
            } else {
                let results: Vec<Result<()>> =
                    pool.install(|| (0..readers).into_par_iter().map(read_part).collect());
                for result in results {
                    result?;
                }
            }
        }
        Ok(())
    }
}

pub struct ReadAhead {
    tasks: [Option<std::thread::JoinHandle<Result<Vec<f32>>>>; 2],
}

impl ReadAhead {
    pub fn wait(&mut self, layer: usize) -> Result<Vec<f32>> {
        ensure!(layer < 2, "invalid Engram prefetch layer");
        let task = self.tasks[layer]
            .take()
            .ok_or_else(|| anyhow::anyhow!("Engram prefetch already consumed"))?;
        task.join()
            .map_err(|_| anyhow::anyhow!("Engram prefetch worker panicked"))?
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        for task in &mut self.tasks {
            if let Some(task) = task.take() {
                let _ = task.join();
            }
        }
    }
}

/// Private, scoped scatter target. Its only caller proves unique row writes.
struct Scatter<'a> {
    pointer: *mut f32,
    _borrow: std::marker::PhantomData<&'a mut [f32]>,
}
unsafe impl Send for Scatter<'_> {}
unsafe impl Sync for Scatter<'_> {}
impl<'a> Scatter<'a> {
    fn new(output: &'a mut [f32]) -> Self {
        Self {
            pointer: output.as_mut_ptr(),
            _borrow: std::marker::PhantomData,
        }
    }
    unsafe fn write(&self, row: usize, values: &[f32; WIDTH]) {
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), self.pointer.add(row * WIDTH), WIDTH);
        }
    }
}

fn decode_mxfp8(packed: &[u8], output: &mut [f32; WIDTH]) -> Result<()> {
    ensure!(packed.len() == 264, "invalid MXFP8 Engram row");
    for i in 0..WIDTH {
        let byte = packed[i];
        let exponent = (byte >> 3) & 15;
        let mantissa = byte & 7;
        let scale = packed[256 + i / 32];
        ensure!(
            byte & 127 != 127 && scale != 255,
            "non-finite MXFP8 Engram value"
        );
        let base = if exponent == 0 {
            mantissa as f32 * 2f32.powi(-9)
        } else {
            (8 + mantissa) as f32 * 2f32.powi(exponent as i32 - 10)
        };
        // Scale in f64 then round once to f32, matching ldexpf even when the
        // scale itself would overflow or underflow as a standalone f32.
        let value = ((if byte & 128 != 0 {
            -(base as f64)
        } else {
            base as f64
        }) * 2f64.powi(scale as i32 - 127)) as f32;
        let rounded = half::bf16::from_f32(value).to_f32();
        ensure!(rounded.is_finite(), "overflow in MXFP8 Engram row");
        output[i] = rounded;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> Layout {
        let mut l = Layout {
            token_map: (0..256).map(|i| i * 37 % 101).collect(),
            compressed_vocab: 101,
            pad: 2,
            multipliers: [[0; 4]; 2],
            primes: [[0; HEADS]; 2],
            offsets: [[0; HEADS]; 2],
            rows: [0; 2],
            combined: false,
        };
        for e in 0..2 {
            for j in 0..4 {
                l.multipliers[e][j] = 1000000007 + 2 * (e * 4 + j) as u64;
            }
            for j in 0..HEADS {
                l.primes[e][j] = 19 + 2 * (e * HEADS + j) as u32;
                l.offsets[e][j] = l.rows[e];
                l.rows[e] += l.primes[e][j];
            }
        }
        l
    }

    #[test]
    fn engram_hash_matches_ds4_c_and_chunking() {
        let l = layout();
        let tokens: Vec<u32> = (0..37).map(|i| i * 13 % 256).collect();
        let mask: Vec<u8> = (0..37).map(|i| u8::from(i % 11 != 5)).collect();
        let mut history = History::default();
        let ids = l.hash(&mut history, &tokens, Some(&mask)).unwrap();
        let mut hash = 0xcbf29ce484222325u64;
        for row in ids.iter().flatten().flatten() {
            for byte in row.to_le_bytes() {
                hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
            }
        }
        // Original ds4_engram.c at super::super::DS4_REVISION, same layout,
        // tokens and mask; output includes every row of both Engram layers.
        assert_eq!(hash, 0x5aa4ae04a4d87ea9);
        assert_eq!(history.tail, [67, 91, 14]);
        for chunk in 1..=37 {
            let mut h = History::default();
            let mut result = Vec::new();
            for (tokens, mask) in tokens.chunks(chunk).zip(mask.chunks(chunk)) {
                result.extend(l.hash(&mut h, tokens, Some(mask)).unwrap());
            }
            assert_eq!(result, ids);
            assert_eq!(h, history);
        }
    }

    #[test]
    fn engram_invalid_input_does_not_advance_history() {
        let l = layout();
        let mut history = History::restore([1, 2, 3], &l).unwrap();
        let before = history;
        assert!(l.hash(&mut history, &[1, 256], None).is_err());
        assert_eq!(history, before);
        assert!(l.hash(&mut history, &[1, 2], Some(&[1])).is_err());
        assert_eq!(history, before);
        assert!(History::restore([-2, 0, 0], &l).is_err());
        assert!(History::restore([101, 0, 0], &l).is_err());
    }

    #[test]
    fn engram_mxfp8_special_values() {
        let mut row = [0u8; 264];
        row[256..].fill(127);
        row[0] = 0x38; // 1.0
        row[1] = 0xb8; // -1.0
        row[2] = 0x01; // minimum E4M3 subnormal
        let mut out = [0f32; WIDTH];
        decode_mxfp8(&row, &mut out).unwrap();
        assert_eq!(&out[..3], &[1.0, -1.0, 1.0 / 512.0]);
        row[0] = 0x7f;
        assert!(decode_mxfp8(&row, &mut out).is_err());
    }

    #[test]
    fn engram_parallel_disk_reads_restore_order_across_chunks() {
        use std::io::Write;

        // Own this small fixture here so engine checks use only the public
        // GGUF API and never depend on another crate's test internals.
        struct EngramFile(std::path::PathBuf);
        impl Drop for EngramFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let mut payload = Vec::new();
        let mut reference = Vec::new();
        for row in 0..17 {
            let mut packed = [0u8; 144];
            for block in 0..8 {
                packed[block * 18..block * 18 + 2].copy_from_slice(&0x3000u16.to_le_bytes());
                for j in 2..18 {
                    packed[block * 18 + j] = (row * 13 + block * 7 + j) as u8;
                }
            }
            let mut decoded = [0f32; WIDTH];
            quant::decode(TensorType::IQ4_NL, &packed, &mut decoded).unwrap();
            payload.extend_from_slice(&packed);
            reference.push(decoded);
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata count
        let string = |bytes: &mut Vec<u8>, text: &str| {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        };
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string metadata
        string(&mut bytes, "deepseek41");
        string(&mut bytes, "engram");
        bytes.extend_from_slice(&2u32.to_le_bytes()); // dimension count
        bytes.extend_from_slice(&256u64.to_le_bytes());
        bytes.extend_from_slice(&17u64.to_le_bytes());
        bytes.extend_from_slice(&TensorType::IQ4_NL.0.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // payload offset
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        bytes.extend_from_slice(&payload);
        let path = std::env::temp_dir().join(format!("lisa-engram-{}.gguf", std::process::id()));
        let mut output = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        let file = EngramFile(path);
        output.write_all(&bytes).unwrap();
        drop(output);
        let package = Package::open(&file.0).unwrap();
        let reader = package.tensor("engram").unwrap().disk_reader().unwrap();
        let tables = Tables::from_readers([reader.clone(), reader]).unwrap();
        let ids: Vec<RowIds> = (0..2051)
            .map(|token| {
                let mut ids = [[0; HEADS]; 2];
                for (layer, rows) in ids.iter_mut().enumerate() {
                    for (head, row) in rows.iter_mut().enumerate() {
                        *row = ((token * 7 + head * 3 + layer) % 17) as u32;
                    }
                }
                ids
            })
            .collect();
        let mut output = vec![f32::NAN; ids.len() * HEADS * WIDTH];
        for layer in 0..2 {
            tables.read(&ids, layer, &mut output).unwrap();
            for (index, values) in output.chunks_exact(WIDTH).enumerate() {
                assert_eq!(
                    values,
                    reference[ids[index / HEADS][layer][index % HEADS] as usize]
                );
            }
        }
        let mut prefetch = tables.prefetch(&ids[..3]).unwrap();
        // Either layer may be joined first; its result remains ordered.
        for layer in [1, 0] {
            let result = prefetch.wait(layer).unwrap();
            for (index, values) in result.chunks_exact(WIDTH).enumerate() {
                assert_eq!(
                    values,
                    reference[ids[index / HEADS][layer][index % HEADS] as usize]
                );
            }
            assert!(prefetch.wait(layer).is_err());
        }
        drop(tables.prefetch(&ids[..1]).unwrap());
        let mut invalid = ids[0];
        invalid[1][0] = 17;
        assert!(tables.prefetch(&[invalid]).is_err());
    }
}
