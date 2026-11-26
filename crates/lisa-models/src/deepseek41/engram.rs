//! The reference n-gram history, metadata contract and row hashing, independent of storage.
//! Ported from ds4_engram.c at b3f50180ec2c7324892be1f50bc83555ccb8d5e7 (MIT).
use crate::metadata::Metadata;
use anyhow::{Context, Result, ensure};

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
    pub fn from_metadata(m: &impl Metadata) -> Result<Self> {
        ensure!(
            m.integers(&["ple.layers", "engram.layer_ids"])? == [1, 14],
            "unsupported Engram layers"
        );
        let combined = m.contains("ple.layers");
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
        if m.contains("engram.compressed_vocab_size") {
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
    /// break every n-gram crossing that position, as in the reference.
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

/// Ordered normalized four-stream gate. The device exposes this operation;
/// checkpoint policy chooses its inputs, epsilon and placement in the tower.
pub fn gate() -> crate::graph::Graph {
    use crate::graph::{Builder, Value as V};
    use lisa_backend::{Kernel, PackedCompute};
    let mut b = Builder::new(4, 0, 0);
    b.writable_input(0);
    let x = b.one(
        Kernel::PackedCompute(PackedCompute::GatedResidual4 {
            epsilon: 1e-20f32.to_bits(),
            masked: false,
        }),
        vec![V::Input(0), V::Input(1), V::Input(2), V::Input(3)],
    );
    b.finish(vec![x])
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
        // Original ds4_engram.c at the pinned revision above, same layout,
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
}
