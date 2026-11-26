//! Container-independent tensor views handed from loading to execution.
use super::ResidencyBytes;
use anyhow::{Context, Result};
use lisa_backend::ScalarType;
use lisa_models::packed::Projection;
use std::collections::BTreeMap;

/// The arithmetic encoding and tensor geometry needed by a computation plan.
/// Shapes use the backend's outer-to-inner axis order, never file axis order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedLayout {
    pub projection: Projection,
    pub dtype: ScalarType,
    pub logical_shape: Vec<i32>,
    pub storage_shape: Vec<i32>,
    pub row_bytes: i32,
}

/// Original payload and, for unquantized weights, a typed view of the same bytes.
/// Creating views is a loading operation; execution only clones tensor handles.
#[derive(Clone)]
pub struct PreparedTensor<T> {
    pub layout: PreparedLayout,
    bytes: T,
    plain: Option<T>,
}

impl<T> PreparedTensor<T> {
    pub fn new(layout: PreparedLayout, bytes: T, plain: Option<T>) -> Result<Self> {
        anyhow::ensure!(
            (layout.projection == Projection::Plain) == plain.is_some(),
            "plain tensor view must match its prepared encoding"
        );
        Ok(Self {
            layout,
            bytes,
            plain,
        })
    }

    pub fn packed_bytes(&self) -> &T {
        &self.bytes
    }

    pub fn plain_view(&self) -> Result<T>
    where
        T: Clone,
    {
        self.plain
            .clone()
            .context("weight is not an unquantized tensor")
    }
}

/// The reservation is released before tensor handles, draining outstanding users
/// on the owning thread rather than waiting inside a device deallocator.
pub struct PreparedWeights<T, L> {
    pub(crate) reservation: L,
    pub tensors: BTreeMap<String, PreparedTensor<T>>,
    pub mapped_bytes: usize,
    pub residency: ResidencyBytes,
}

impl<T, L> PreparedWeights<T, L> {
    pub fn into_parts(self) -> (L, BTreeMap<String, PreparedTensor<T>>) {
        (self.reservation, self.tensors)
    }
}
