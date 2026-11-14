//! Affine absorbed-attention dataflow, numerical schedules and value partitions.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{Kernel as K, ScalarType as D, SparseF32 as S};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub shapes: Vec<Vec<i32>>,
    pub dtypes: Vec<D>,
    pub group: usize,
    pub selected: bool,
    pub prefix: usize,
    pub scale: u32,
    pub capture: bool,
    pub fused: bool,
    pub capture_fused: bool,
    pub boundaries: bool,
    pub native: bool,
    pub dense_prefix: Option<usize>,
    pub flash: Option<usize>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Prepare,
    Query,
    Boundary,
    Attend,
    Value,
    Output,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Stage(Spec, Stage),
    Value {
        rows: usize,
        heads: usize,
        group: usize,
        native: bool,
        prefix: Option<usize>,
    },
}
pub fn value_slices(tokens: usize, dense_prefix: usize) -> Vec<(usize, usize)> {
    let mut slices = vec![];
    let mut start = 0;
    while start < tokens {
        let mut length = (tokens - start).min(2048);
        if start < dense_prefix {
            length = length.min(dense_prefix - start);
        }
        slices.push((start, length));
        start += length;
    }
    slices
}
impl Spec {
    pub fn geometry(&self) -> (usize, usize, usize, usize, usize) {
        let q = &self.shapes[0];
        let (b, h, t) = (q[0] as usize, q[1] as usize, q[2] as usize);
        let tc = self.shapes[4][1] as usize;
        let selected = if self.selected {
            self.shapes[5][2] as usize
        } else {
            tc
        };
        (b, h, t, tc, selected)
    }
    pub fn validate(&self) -> bool {
        if self.shapes.len() != 6
            || self.dtypes.len() != 6
            || self.shapes[0].len() != 4
            || self.shapes[4].len() != 3
            || self.shapes[5].len() != 3
            || self.shapes.iter().flatten().any(|&d| d <= 0)
            || self.group == 0
            || 512 % self.group != 0
        {
            return false;
        }
        let (b, h, t, tc, n) = self.geometry();
        let valid = self.shapes[0][3] == 256
            && self.shapes[1] == [(h * 512) as i32, 128]
            && self.shapes[2] == [(h * 512) as i32, (512 / self.group) as i32]
            && self.shapes[3] == self.shapes[2]
            && self.shapes[4] == [b as i32, tc as i32, 512]
            && (!self.selected || self.shapes[5][..2] == [b as i32, t as i32])
            && self.dtypes == [D::F32, D::U32, D::F32, D::F32, D::F16, D::I32]
            && self.prefix <= n;
        if !valid {
            return false;
        }
        if self.flash.is_some() {
            return b == 1 && h == 64 && t > 0;
        }
        n <= 2051
            || (b == 1
                && t == 1
                && h == 64
                && self.selected
                && n <= tc
                && self.prefix == 0
                && !self.boundaries)
    }
    pub fn native_projection(&self) -> bool {
        self.native && self.geometry().1 == 64 && self.group == 64
    }
    pub fn fused_decode(&self) -> bool {
        let (b, h, t, _, n) = self.geometry();
        self.flash.is_none()
            && b == 1
            && t == 1
            && h == 64
            && n <= 2051
            && ((self.native && !self.boundaries && self.group == 64)
                || (!self.capture && self.fused))
    }
    pub fn fused_capture(&self) -> bool {
        self.capture_fused || (self.native && !self.boundaries && self.group == 64 && self.capture)
    }
    pub fn stages(&self) -> anyhow::Result<Vec<Stage>> {
        anyhow::ensure!(self.validate(), "unsupported absorbed-attention geometry");
        let mut v = vec![Stage::Prepare, Stage::Query];
        if self.boundaries {
            v.push(Stage::Boundary);
        }
        v.push(Stage::Attend);
        if !self.fused_decode() {
            v.push(Stage::Value);
            if self.boundaries {
                v.push(Stage::Boundary);
            }
        }
        v.push(Stage::Output);
        Ok(v)
    }
}
fn value(
    p: &mut Build<'_>,
    x: V,
    weights: [V; 3],
    rows: usize,
    heads: usize,
    group: usize,
    native: bool,
    prefix: Option<usize>,
) -> V {
    let native = native && heads == 64 && group == 64;
    let slices = if native {
        prefix
            .map(|n| value_slices(rows / heads, n))
            .unwrap_or_else(|| vec![(0, rows / heads)])
    } else {
        vec![(0, rows / heads)]
    };
    let split = slices.len() > 1;
    let mut out = vec![];
    for (start, len) in slices {
        let x = if split {
            p.slice(x, 0, (start * heads) as i32, ((start + len) * heads) as i32)
        } else {
            x
        };
        out.push(p.graph.one(
            K::SparseF32(S::AffineHeadValue {
                group,
                heads,
                native,
                matrix: native && len >= 32,
            }),
            vec![x, weights[0], weights[1], weights[2]],
        ));
    }
    if split { p.concat(out, 0) } else { out[0] }
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        if let Self::Value {
            rows,
            heads,
            group,
            native,
            prefix,
        } = self
        {
            anyhow::ensure!(
                *heads > 0 && *rows > 0 && rows % heads == 0,
                "invalid value rows"
            );
            let mut p = Build::new(4, &schema);
            let y = value(
                &mut p,
                V::Input(0),
                [V::Input(1), V::Input(2), V::Input(3)],
                *rows,
                *heads,
                *group,
                *native,
                *prefix,
            );
            return Ok(p.finish(vec![y])?.graph);
        }
        let Self::Stage(s, stage) = self else {
            unreachable!()
        };
        anyhow::ensure!(s.validate(), "invalid attention plan");
        let (b, h, t, tc, n) = s.geometry();
        let mut p = Build::new(
            match stage {
                Stage::Prepare => 6,
                Stage::Query | Stage::Value => 4,
                Stage::Attend => 7,
                Stage::Output => 1,
                Stage::Boundary => unreachable!(),
            },
            &schema,
        );
        let out = match stage {
            Stage::Prepare => {
                let mut v: Vec<_> = (0..6).map(|i| p.contiguous(V::Input(i))).collect();
                if s.flash.is_none() && s.selected && b * t * n < 8 {
                    let pad = (8 - b * t * n).div_ceil(b * t);
                    let k = p
                        .graph
                        .constant_i32(&vec![-1; b * t * pad], &[b as i32, t as i32, pad as i32]);
                    let y = p.concat(vec![v[5], k], 2);
                    v[5] = p.contiguous(y);
                }
                v
            }
            Stage::Query => vec![p.graph.one(
                K::SparseF32(S::AffineHeadQuery {
                    group: s.group,
                    native: s.native_projection(),
                }),
                (0..4).map(V::Input).collect::<Vec<_>>(),
            )],
            Stage::Attend => {
                // Input order: query, low query, compact cache, selected IDs, W/S/B.
                if let Some(position) = s.flash {
                    vec![p.graph.one(
                        K::SparseF32(S::FlashLatent {
                            position,
                            scale: s.scale,
                        }),
                        vec![V::Input(1), V::Input(2)],
                    )]
                } else if s.fused_decode() {
                    p.graph.call(
                        K::SparseF32(S::AffineIndexedDecode {
                            group: s.group,
                            selected: n,
                            dense: !s.selected,
                            scale: s.scale,
                            native: s.native_projection(),
                            capture: s.fused_capture(),
                        }),
                        vec![
                            V::Input(1),
                            V::Input(2),
                            V::Input(3),
                            V::Input(4),
                            V::Input(5),
                            V::Input(6),
                        ],
                    )
                } else if n > 2051 {
                    vec![p.graph.one(
                        K::SparseF32(S::IndexedLatent { scale: s.scale }),
                        (0..4).map(V::Input).collect::<Vec<_>>(),
                    )]
                } else {
                    let padded = if s.selected && b * t * n < 8 {
                        n + (8 - b * t * n).div_ceil(b * t)
                    } else {
                        n
                    };
                    vec![p.graph.one(
                        K::SparseF32(S::IndexedLatentShort {
                            selected: if s.selected { padded } else { tc },
                            prefix: s.prefix,
                            dense: !s.selected,
                            scale: s.scale,
                        }),
                        vec![V::Input(1), V::Input(2), V::Input(3)],
                    )]
                }
            }
            Stage::Value => {
                let x = p.reshape(V::Input(0), &[(b * t * h) as i32, 512]);
                let y = value(
                    &mut p,
                    x,
                    [V::Input(1), V::Input(2), V::Input(3)],
                    b * t * h,
                    h,
                    s.group,
                    s.native,
                    s.dense_prefix,
                );
                vec![p.reshape(y, &[b as i32, t as i32, h as i32, 256])]
            }
            Stage::Output => {
                let y = p.transpose(V::Input(0), &[0, 2, 1, 3]);
                vec![p.contiguous(y)]
            }
            Stage::Boundary => unreachable!(),
        };
        Ok(p.finish(out)?.graph)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partitions_preserve_dense_boundary() {
        assert_eq!(value_slices(34, 3), [(0, 3), (3, 31)]);
        assert_eq!(
            value_slices(4096, 2051),
            [(0, 2048), (2048, 3), (2051, 2045)]
        );
        assert_eq!(value_slices(2052, 0), [(0, 2048), (2048, 4)]);
        assert!(value_slices(0, 2051).is_empty());
    }
    #[test]
    fn wide_and_prefill_schedules_keep_distinct_contracts() {
        let mut s = Spec {
            shapes: vec![
                vec![1, 64, 1, 256],
                vec![32768, 128],
                vec![32768, 8],
                vec![32768, 8],
                vec![1, 2052, 512],
                vec![1, 1, 2051],
            ],
            dtypes: vec![D::F32, D::U32, D::F32, D::F32, D::F16, D::I32],
            group: 64,
            selected: true,
            prefix: 0,
            scale: 0.0625f32.to_bits(),
            capture: false,
            fused: false,
            capture_fused: false,
            boundaries: false,
            native: true,
            dense_prefix: None,
            flash: None,
        };
        assert!(s.fused_decode());
        assert_eq!(
            s.stages().unwrap(),
            [Stage::Prepare, Stage::Query, Stage::Attend, Stage::Output]
        );
        s.shapes[5][2] = 2052;
        assert!(!s.fused_decode());
        assert!(s.validate());
        s.boundaries = true;
        assert!(!s.validate());
        s.boundaries = false;
        s.prefix = 1;
        assert!(!s.validate());
        s.prefix = 0;
        s.shapes[0][2] = 24;
        s.shapes[5][1] = 24;
        s.flash = Some(0);
        s.boundaries = true;
        assert!(s.validate());
        assert_eq!(
            s.stages().unwrap(),
            [
                Stage::Prepare,
                Stage::Query,
                Stage::Boundary,
                Stage::Attend,
                Stage::Value,
                Stage::Boundary,
                Stage::Output
            ]
        );
    }
}
