//! Indirect storage and individual affine kernels. Resource handles are opaque;
//! model plans never contain addresses, residency sets or backend objects.
use crate::{Device, Dispatch, Outputs, ScalarType};

pub trait IndirectDevice: Device {
    type Resources;
    /// A provider-owned table in the supplied slot order. Retains every source
    /// tensor, including its mapped owner, for all queued indirect consumers.
    fn resources(&self, slots: &[&Self::Tensor]) -> Result<Self::Resources, Self::Error>;
    fn dispatch_indirect(
        &self,
        handle: &Self::KernelHandle,
        inputs: &[&Self::Tensor],
        resources: &Self::Resources,
        stream: &Self::Stream,
    ) -> Result<Option<Dispatch<Outputs<Self::Tensor>, Self::Completion>>, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndirectAffine {
    RouteMap {
        rows: usize,
        top_k: usize,
        experts: usize,
        tile: usize,
    },
    Pair {
        intermediate: usize,
        top_k: usize,
        limit: u32,
        metadata: ScalarType,
    },
    Gather {
        hidden: usize,
        ds4: bool,
        metadata: ScalarType,
    },
    DownSum8 {
        hidden: usize,
        shared: bool,
        metadata: ScalarType,
    },
    DownSum8SharedProjection {
        hidden: usize,
        metadata: ScalarType,
    },
    Sum8 {
        rows: usize,
        hidden: usize,
    },
    GroupedProjection {
        rows: usize,
        top_k: usize,
        experts: usize,
        intermediate: usize,
        up: bool,
        vector: bool,
        metadata: ScalarType,
        workspace: bool,
    },
    GroupedPair {
        rows: usize,
        top_k: usize,
        experts: usize,
        intermediate: usize,
        limit: u32,
        metadata: ScalarType,
        workspace: bool,
    },
    GroupedSwiGlu {
        rows: usize,
        intermediate: usize,
        limit: u32,
        workspace: bool,
    },
    GroupedDown {
        rows: usize,
        top_k: usize,
        experts: usize,
        hidden: usize,
        vector: bool,
        tile: usize,
        metadata: ScalarType,
        workspace: bool,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndirectAffineKind {
    RouteMap,
    Pair,
    Gather(bool),
    DownSum8(bool),
    DownSum8SharedProjection,
    Sum8,
    GroupedProjection(bool),
    GroupedPair(bool),
    GroupedSwiGlu(bool),
    GroupedDown(bool),
}
impl IndirectAffine {
    pub fn kind(&self) -> IndirectAffineKind {
        use IndirectAffineKind as K;
        match *self {
            Self::RouteMap { .. } => K::RouteMap,
            Self::Pair { .. } => K::Pair,
            Self::Gather { ds4, .. } => K::Gather(ds4),
            Self::DownSum8 { shared, .. } => K::DownSum8(shared),
            Self::DownSum8SharedProjection { .. } => K::DownSum8SharedProjection,
            Self::Sum8 { .. } => K::Sum8,
            Self::GroupedProjection { workspace, .. } => K::GroupedProjection(workspace),
            Self::GroupedPair { workspace, .. } => K::GroupedPair(workspace),
            Self::GroupedSwiGlu { workspace, .. } => K::GroupedSwiGlu(workspace),
            Self::GroupedDown { workspace, .. } => K::GroupedDown(workspace),
        }
    }
    pub fn requires_resources(&self) -> bool {
        !matches!(
            self,
            Self::RouteMap { .. } | Self::Sum8 { .. } | Self::GroupedSwiGlu { .. }
        )
    }
    pub fn outputs(&self) -> usize {
        if matches!(self, Self::RouteMap { .. }) {
            4
        } else {
            1
        }
    }
    pub fn arity(&self) -> usize {
        match *self {
            Self::RouteMap { .. } | Self::Sum8 { .. } => 1,
            Self::Pair { .. } => 3,
            Self::Gather { .. } => 2,
            Self::DownSum8SharedProjection { .. } => 6,
            Self::DownSum8 { shared, .. } => 2 + usize::from(shared),
            Self::GroupedProjection { workspace, .. } | Self::GroupedDown { workspace, .. } => {
                5 + usize::from(workspace)
            }
            Self::GroupedPair { workspace, .. } => 6 + usize::from(workspace),
            Self::GroupedSwiGlu { workspace, .. } => 3 + usize::from(workspace),
        }
    }
    pub fn overwritten_inputs(&self) -> &'static [usize] {
        match self {
            Self::GroupedProjection {
                workspace: true, ..
            }
            | Self::GroupedDown {
                workspace: true, ..
            } => &[5],
            Self::GroupedPair {
                workspace: true, ..
            } => &[6],
            Self::GroupedSwiGlu {
                workspace: true, ..
            } => &[3],
            _ => &[],
        }
    }
    pub fn validate(&self) -> bool {
        let dims = |r: usize, k: usize, h: usize| {
            r > 0
                && k > 0
                && h > 0
                && r.checked_mul(k)
                    .and_then(|n| n.checked_mul(h))
                    .is_some_and(|n| n <= i32::MAX as usize)
        };
        let meta = |d| matches!(d, ScalarType::BF16 | ScalarType::F32);
        match *self {
            Self::RouteMap {
                rows,
                top_k,
                experts,
                tile,
            } => {
                dims(rows, top_k, 1)
                    && experts > 0
                    && experts <= i32::MAX as usize
                    && top_k <= experts
                    && tile == 32
            }
            Self::Pair {
                intermediate,
                top_k,
                limit,
                metadata,
            } => {
                dims(1, top_k, intermediate)
                    && intermediate % 32 == 0
                    && meta(metadata)
                    && f32::from_bits(limit).is_finite()
            }
            Self::Gather {
                hidden, metadata, ..
            }
            | Self::DownSum8SharedProjection { hidden, metadata }
            | Self::DownSum8 {
                hidden, metadata, ..
            } => dims(1, 1, hidden) && hidden % 32 == 0 && meta(metadata),
            Self::Sum8 { rows, hidden } => dims(rows, 8, hidden),
            Self::GroupedProjection {
                rows,
                top_k,
                experts,
                intermediate,
                metadata,
                ..
            }
            | Self::GroupedPair {
                rows,
                top_k,
                experts,
                intermediate,
                metadata,
                ..
            } => {
                dims(rows, top_k, intermediate)
                    && experts > 0
                    && experts <= i32::MAX as usize
                    && intermediate % 64 == 0
                    && meta(metadata)
            }
            Self::GroupedSwiGlu {
                rows,
                intermediate,
                limit,
                ..
            } => dims(rows, 1, intermediate) && f32::from_bits(limit).is_finite(),
            Self::GroupedDown {
                rows,
                top_k,
                experts,
                hidden,
                tile,
                metadata,
                ..
            } => {
                dims(rows, top_k, hidden)
                    && experts > 0
                    && experts <= i32::MAX as usize
                    && hidden % 128 == 0
                    && matches!(tile, 64 | 128)
                    && meta(metadata)
            }
        }
    }
}
