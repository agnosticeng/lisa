//! Session geometry and compression ownership from the V4.1 checkpoint law.
pub const MAX_CONTEXT: usize = 1_048_576;
pub const LAYERS: usize = 40;
pub const RAW_ROWS: usize = 128;
pub const LATENT: usize = 512;
pub const INDEX_WIDTH: usize = 128;
pub const OWNERS: usize = 4;
pub const MIRROR_THRESHOLD: usize = 8192;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owner {
    pub rows: usize,
    pub mirror: bool,
}
pub fn owner(layer: usize) -> usize {
    match layer {
        0..8 => 0,
        8..14 => 1,
        14..20 => 2,
        _ => 3,
    }
}
pub fn layout(capacity: usize) -> [Owner; OWNERS] {
    std::array::from_fn(|owner| {
        let rows = if owner < 3 { capacity / 2 } else { capacity }.max(1);
        Owner {
            rows,
            mirror: rows >= MIRROR_THRESHOLD,
        }
    })
}
pub fn storage_bytes(capacity: usize) -> usize {
    let fixed = LAYERS * RAW_ROWS * LATENT * 4 + OWNERS * 2 * LATENT * 4;
    layout(capacity).iter().fold(fixed, |total, owner| {
        total
            + owner.rows
                * ((LATENT + INDEX_WIDTH) * 4 + if owner.mirror { INDEX_WIDTH * 2 } else { 0 })
    })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Empty,
    Full { mirror: bool, publish: bool },
    Candidates,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
    pub owner: usize,
    pub compressed: bool,
    pub compressed_rows: usize,
    pub pool_pair: bool,
    pub raw_row: usize,
    pub raw_count: u32,
    pub raw_start: u32,
    pub selected: usize,
    pub selection: Selection,
}
impl Position {
    pub fn new(layer: usize, ratio: usize, position: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            layer < LAYERS && position < MAX_CONTEXT && ratio <= 2,
            "invalid attention cache position"
        );
        let compressed_rows = if ratio == 0 {
            0
        } else {
            (position + 1) / ratio
        };
        let raw_count = (position + 1).min(RAW_ROWS) as u32;
        Ok(Self {
            owner: owner(layer),
            compressed: ratio != 0,
            compressed_rows,
            pool_pair: position % 2 == 1,
            raw_row: position % RAW_ROWS,
            raw_count,
            raw_start: ((position + 1 - raw_count as usize) % RAW_ROWS) as u32,
            selected: compressed_rows.min(512),
            selection: if compressed_rows == 0 {
                Selection::Empty
            } else if layer > 20 && compressed_rows > 16384 {
                Selection::Candidates
            } else {
                Selection::Full {
                    mirror: compressed_rows >= MIRROR_THRESHOLD,
                    publish: layer == 20 && compressed_rows > 16384,
                }
            },
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ownership_and_selection_boundaries_preserve_scalar_schedule() -> anyhow::Result<()> {
        for layer in 0..40 {
            for ratio in [0, 1, 2] {
                for p in [
                    0,
                    1,
                    127,
                    128,
                    8190,
                    8191,
                    16382,
                    16383,
                    16384,
                    32767,
                    32768,
                    MAX_CONTEXT - 1,
                ] {
                    let s = Position::new(layer, ratio, p)?;
                    let n = if ratio == 0 { 0 } else { (p + 1) / ratio };
                    assert_eq!(
                        s.owner,
                        if layer < 8 {
                            0
                        } else if layer < 14 {
                            1
                        } else if layer < 20 {
                            2
                        } else {
                            3
                        }
                    );
                    assert_eq!(s.compressed_rows, n);
                    assert_eq!(s.raw_start, ((p + 1 - (p + 1).min(128)) % 128) as u32);
                    assert_eq!(
                        s.selection,
                        if n == 0 {
                            Selection::Empty
                        } else if layer > 20 && n > 16384 {
                            Selection::Candidates
                        } else {
                            Selection::Full {
                                mirror: n >= 8192,
                                publish: layer == 20 && n > 16384,
                            }
                        }
                    );
                }
            }
        }
        assert!(Position::new(40, 1, 0).is_err());
        assert!(Position::new(0, 1, MAX_CONTEXT).is_err());
        Ok(())
    }
}
pub fn allocate(capacity: usize) -> anyhow::Result<crate::graph::Graph> {
    use crate::graph::Builder;
    use lisa_backend::{Kernel, ScalarType as D};
    anyhow::ensure!(
        (1..=MAX_CONTEXT).contains(&capacity),
        "context capacity is outside the checkpoint limit"
    );
    let mut b = Builder::new(0, 0, 0);
    let mut out = Vec::new();
    for _ in 0..LAYERS {
        out.push(b.one(
            Kernel::Zeros {
                shape: vec![RAW_ROWS as i32, LATENT as i32],
                dtype: D::F32,
            },
            vec![],
        ));
    }
    for owner in layout(capacity) {
        for width in [LATENT, INDEX_WIDTH] {
            out.push(b.one(
                Kernel::Zeros {
                    shape: vec![owner.rows as i32, width as i32],
                    dtype: D::F32,
                },
                vec![],
            ));
        }
        if owner.mirror {
            out.push(b.one(
                Kernel::Zeros {
                    shape: vec![owner.rows as i32, INDEX_WIDTH as i32],
                    dtype: D::BF16,
                },
                vec![],
            ));
        }
        for _ in 0..2 {
            out.push(b.one(
                Kernel::Zeros {
                    shape: vec![1, LATENT as i32],
                    dtype: D::F32,
                },
                vec![],
            ));
        }
    }
    Ok(b.finish(out))
}
