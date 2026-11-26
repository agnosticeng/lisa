//! Disk reads and format decoding for model-supplied Engram row requests.
use anyhow::{Result, ensure};
use lisa_gguf::{Package, Tensor, TensorType, quant};
pub use lisa_models::deepseek41::engram::{HEADS, History, LAYERS, Layout, RowIds, WIDTH};
use rayon::prelude::*;

/// Disk-only table handles: no mmap, residency reservation or GPU buffer.
#[derive(Clone)]
pub struct DiskTables {
    tables: [Tensor; 2],
    readers: [std::sync::Arc<rayon::ThreadPool>; 2],
}

impl DiskTables {
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

    /// Metal reference starts both disk reads before layer 0 and joins each at its
    /// Engram layer (1 and 14). Each table has at most sixteen read workers.
    /// The returned job owns all host buffers and joins on drop, including
    /// failures or cancellation before the graph reaches an Engram layer.
    pub fn prefetch(&self, ids: &[RowIds]) -> Result<DiskReadAhead> {
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
        let mut job = DiskReadAhead {
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

    /// The reference's sorted, deduplicated row reads in blocks of 2048 tokens. The
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
            // The reference uses sixteen readers per table from eight rows upward on
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
                            quant::decode_ds4_mxfp8(&packed[..bytes], &mut decoded)?;
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

pub struct DiskReadAhead {
    tasks: [Option<std::thread::JoinHandle<Result<Vec<f32>>>>; 2],
}

impl DiskReadAhead {
    pub fn wait(&mut self, layer: usize) -> Result<Vec<f32>> {
        ensure!(layer < 2, "invalid Engram prefetch layer");
        let task = self.tasks[layer]
            .take()
            .ok_or_else(|| anyhow::anyhow!("Engram prefetch already consumed"))?;
        task.join()
            .map_err(|_| anyhow::anyhow!("Engram prefetch worker panicked"))?
    }
}

impl Drop for DiskReadAhead {
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
// SAFETY: the pointer comes from an exclusively borrowed &mut [f32] that the
// lifetime keeps alive; `write` callers guarantee disjoint rows per thread.
unsafe impl Send for Scatter<'_> {}
// SAFETY: shared `&Scatter` only permits `write`, whose callers prove that
// concurrent writes target disjoint rows; nothing reads through it.
unsafe impl Sync for Scatter<'_> {}
impl<'a> Scatter<'a> {
    fn new(output: &'a mut [f32]) -> Self {
        Self {
            pointer: output.as_mut_ptr(),
            _borrow: std::marker::PhantomData,
        }
    }
    unsafe fn write(&self, row: usize, values: &[f32; WIDTH]) {
        // SAFETY: the caller guarantees (row + 1) * WIDTH <= output length and that
        // no other thread touches this row, so the destination is in bounds and unaliased.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), self.pointer.add(row * WIDTH), WIDTH);
        }
    }
}

impl crate::bindings::deepseek41::engram::Tables for DiskTables {
    fn prefetch(
        &self,
        ids: &[RowIds],
    ) -> Result<Box<dyn crate::bindings::deepseek41::engram::ReadAhead>> {
        Ok(Box::new(DiskTables::prefetch(self, ids)?))
    }
}
impl crate::bindings::deepseek41::engram::ReadAhead for DiskReadAhead {
    fn wait(&mut self, layer: usize) -> Result<Vec<f32>> {
        DiskReadAhead::wait(self, layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let tables = DiskTables::from_readers([reader.clone(), reader]).unwrap();
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
            for (index, values) in output.as_chunks::<WIDTH>().0.iter().enumerate() {
                assert_eq!(
                    *values,
                    reference[ids[index / HEADS][layer][index % HEADS] as usize]
                );
            }
        }
        let provider: &dyn crate::bindings::deepseek41::engram::Tables = &tables;
        let mut prefetch = provider.prefetch(&ids[..3]).unwrap();
        // Either layer may be joined first; its result remains ordered.
        for layer in [1, 0] {
            let result = prefetch.wait(layer).unwrap();
            for (index, values) in result.as_chunks::<WIDTH>().0.iter().enumerate() {
                assert_eq!(
                    *values,
                    reference[ids[index / HEADS][layer][index % HEADS] as usize]
                );
            }
            assert!(prefetch.wait(layer).is_err());
        }
        drop(provider.prefetch(&ids[..1]).unwrap());
        let mut invalid = ids[0];
        invalid[1][0] = 17;
        assert!(provider.prefetch(&[invalid]).is_err());
    }
}
