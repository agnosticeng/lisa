//! The persistent decode state of one Inkling sequence, as the pinned
//! reference lays it out: paged INT8 keys and values of all 48 applications
//! in one extent, its page table, the double-buffered bf16 convolution state,
//! the committed token count and, with a draft, the draft's context rings.
//! Everything else a step needs is per-step scratch. Snapshots and growth
//! copy the committed pages and the other buffers on the device.
use super::kernels::draft_ring_bytes;
use anyhow::{Context, Result, ensure};
use lisa_backend::{
    Device, Kernel, PackedCompute as P, ScalarType as Dtype, TensorAccess, VendoredDispatch,
    VendoredSource,
};

/// Tokens of one KV page.
pub const PAGE_TOKENS: usize = 32;
/// Bytes of one page of one tensor's data and scales.
const PAGE_DATA: usize = 8 * PAGE_TOKENS * 128;
const PAGE_SCALES: usize = 8 * PAGE_TOKENS * 4;
/// Applications (layer visits) of a step.
pub const APPLICATIONS: usize = 48;
/// Convolution-state columns of one application: key, value, mixer, FFN.
pub const APPLICATION_COLUMNS: usize = 6656;
pub const STATE_COLUMNS: usize = APPLICATIONS * APPLICATION_COLUMNS;
/// Column of each convolution within an application.
pub const KEY_COLUMN: usize = 0;
pub const VALUE_COLUMN: usize = 1024;
pub const MIXER_COLUMN: usize = 2048;
pub const FFN_COLUMN: usize = 4352;
/// Rows a step may write past the committed tokens.
pub const SCRATCH_TOKENS: usize = 16;
/// Bytes of one row of an extent's tensor.
const EXTENT_ROW: usize = 1 << 16;
/// Pages an extent grows by: regions stay 64 KiB-aligned.
const PAGE_GRANULE: usize = 64;
/// Most pages of an extent: the last region's byte offset must fit the
/// kernels' 32-bit layer offsets.
const MOST_PAGES: usize =
    u32::MAX as usize / ((APPLICATIONS - 1) * region_bytes(1)) / PAGE_GRANULE * PAGE_GRANULE;
/// Most tokens a sequence holds.
pub const MOST_TOKENS: usize = MOST_PAGES * PAGE_TOKENS - SCRATCH_TOKENS;

/// Whether application `a` attends over the whole history.
pub fn full(a: usize) -> bool {
    a % 6 == 5
}

/// The KV region of application `a`: full applications first, in order,
/// then the windowed ones.
pub fn region(a: usize) -> usize {
    if full(a) {
        a / 6
    } else {
        APPLICATIONS / 6 + a - (a + 1) / 6
    }
}

/// Bytes of one region of an extent of `extent_pages` pages.
pub const fn region_bytes(extent_pages: usize) -> usize {
    extent_pages * 2 * (PAGE_DATA + PAGE_SCALES)
}

pub(crate) struct StepState<T> {
    /// One extent: 48 regions of `extent_pages` pages.
    pub extent: T,
    pub extent_pages: usize,
    /// U64 page-table entries `[extent_pages]` (as U32 pairs).
    pub page_table: T,
    /// bf16 `[3, STATE_COLUMNS]`: the current cell and the one a commit
    /// writes.
    pub conv: [T; 2],
    pub parity: usize,
    /// Tokens whose keys and values the pages hold.
    pub committed: usize,
    /// The draft's keys and values rings of each layer (none without a
    /// draft); they hold the committed tokens' context.
    pub rings: Vec<[T; 2]>,
}

impl<T: Clone> StepState<T> {
    pub fn current(&self) -> &T {
        &self.conv[self.parity]
    }
    pub fn next(&self) -> &T {
        &self.conv[1 - self.parity]
    }
    pub fn swap(&mut self) {
        self.parity = 1 - self.parity;
    }
}

/// Zeroed state storage, cleared on the device: only device kernels touch
/// it, and a host clear of an extent would delay every sequence's start.
fn zeros<D: Device>(device: &D, shape: Vec<i32>, dtype: Dtype) -> Result<D::Tensor> {
    crate::tensor_ops::TensorOps::new(device, crate::tensor_ops::TensorLaw::Native)
        .required(Kernel::DeviceZeros { shape, dtype }, &[])
}

/// Pages of an extent holding `tokens` tokens and a step's scratch rows.
fn pages_for(tokens: usize) -> usize {
    (tokens + SCRATCH_TOKENS)
        .div_ceil(PAGE_TOKENS)
        .next_multiple_of(PAGE_GRANULE)
}

/// A zeroed extent of `extent_pages` pages and its page table.
fn extent<D: Device>(device: &D, extent_pages: usize) -> Result<(D::Tensor, D::Tensor)> {
    ensure!(
        extent_pages <= MOST_PAGES,
        "Inkling context exceeds {MOST_TOKENS} tokens"
    );
    let bytes = APPLICATIONS * region_bytes(extent_pages);
    // Rows of 64 KiB (regions stay 64 KiB-aligned): an extent past 2 GiB,
    // about 21k tokens, has no dimension an i32 cannot hold.
    ensure!(
        bytes.is_multiple_of(EXTENT_ROW),
        "Inkling extent is not row-aligned"
    );
    let extent = zeros(
        device,
        vec![i32::try_from(bytes / EXTENT_ROW)?, EXTENT_ROW as i32],
        Dtype::U8,
    )?;
    let page_table =
        crate::tensor_ops::TensorOps::new(device, crate::tensor_ops::TensorLaw::Native).required(
            Kernel::PackedCompute(P::KvPageTable {
                extent_pages: extent_pages as u32,
            }),
            &[&extent],
        )?;
    Ok((extent, page_table))
}

fn glue<D: Device>(device: &D, dispatch: VendoredDispatch, inputs: &[&D::Tensor]) -> Result<()> {
    let entry = dispatch.entry.clone();
    crate::tensor_ops::TensorOps::new(device, crate::tensor_ops::TensorLaw::Native)
        .required(
            Kernel::PackedCompute(P::Vendored(Box::new(dispatch))),
            inputs,
        )
        .with_context(|| format!("Inkling state {entry}"))?;
    Ok(())
}

/// Copy the first `pages` pages of every region of `source` (`source_pages`
/// pages a region) into `target` (`target_pages`).
fn copy_pages<D: Device>(
    device: &D,
    source: &D::Tensor,
    source_pages: usize,
    target: &D::Tensor,
    target_pages: usize,
    pages: usize,
) -> Result<()> {
    if pages == 0 {
        return Ok(());
    }
    let words = |x: usize| (x as u32).to_le_bytes().to_vec();
    let mut bytes = words(source_pages);
    bytes.extend(words(target_pages));
    bytes.extend(words(pages));
    glue(
        device,
        VendoredDispatch {
            source: VendoredSource::StateGlue,
            entry: "lisa_kv_extent_copy".into(),
            constants: vec![],
            slots: vec![0, 1],
            addressed: vec![],
            bytes,
            bytes_slot: 2,
            grid: [
                (pages * PAGE_DATA / 16).div_ceil(256) as u32,
                4,
                APPLICATIONS as u32,
            ],
            threads: [256, 1, 1],
            indirect: None,
        },
        &[source, target],
    )
}

/// A new tensor of `source`'s shape and type holding its values.
fn copy<D: Device<Tensor: TensorAccess>>(device: &D, source: &D::Tensor) -> Result<D::Tensor> {
    let dtype = device.dtype(source).context("untyped Inkling state")?;
    let target = zeros(device, source.shape().to_vec(), dtype)?;
    let words = source.size() * dtype.size_of() / 4;
    glue(
        device,
        VendoredDispatch {
            source: VendoredSource::StateGlue,
            entry: "lisa_words_copy".into(),
            constants: vec![],
            slots: vec![0, 1],
            addressed: vec![],
            bytes: (words as u32).to_le_bytes().to_vec(),
            bytes_slot: 2,
            grid: [words.div_ceil(256) as u32, 1, 1],
            threads: [256, 1, 1],
            indirect: None,
        },
        &[source, &target],
    )?;
    Ok(target)
}

impl<T: TensorAccess> StepState<T> {
    /// An empty sequence's state for `tokens` tokens, with `draft_layers`
    /// layers of draft rings.
    pub(crate) fn new<D: Device<Tensor = T>>(
        device: &D,
        tokens: usize,
        draft_layers: usize,
    ) -> Result<Self> {
        let extent_pages = pages_for(tokens);
        let (extent, page_table) = extent(device, extent_pages)?;
        let conv = || zeros(device, vec![3, STATE_COLUMNS as i32], Dtype::BF16);
        // The draft attention reads whole tiles of slots: unwritten slots
        // must hold finite values.
        let ring = || zeros(device, vec![(draft_ring_bytes() / 2) as i32], Dtype::BF16);
        Ok(Self {
            extent,
            extent_pages,
            page_table,
            conv: [conv()?, conv()?],
            parity: 0,
            committed: 0,
            rings: (0..draft_layers)
                .map(|_| Ok([ring()?, ring()?]))
                .collect::<Result<_>>()?,
        })
    }

    /// Grow the extent, if needed, to hold `tokens` committed tokens and a
    /// step's scratch rows.
    pub(crate) fn reserve<D: Device<Tensor = T>>(
        &mut self,
        device: &D,
        tokens: usize,
    ) -> Result<()> {
        if tokens + SCRATCH_TOKENS <= self.extent_pages * PAGE_TOKENS {
            return Ok(());
        }
        let pages = pages_for(tokens.max(2 * self.committed))
            .min(MOST_PAGES)
            .max(pages_for(tokens));
        let (extent, page_table) = extent(device, pages)?;
        copy_pages(
            device,
            &self.extent,
            self.extent_pages,
            &extent,
            pages,
            self.committed.div_ceil(PAGE_TOKENS),
        )?;
        self.extent = extent;
        self.page_table = page_table;
        self.extent_pages = pages;
        Ok(())
    }

    /// An independent copy of the committed state.
    pub(crate) fn snapshot<D: Device<Tensor = T>>(&self, device: &D) -> Result<Self> {
        let pages = pages_for(self.committed);
        let (extent, page_table) = extent(device, pages)?;
        copy_pages(
            device,
            &self.extent,
            self.extent_pages,
            &extent,
            pages,
            self.committed.div_ceil(PAGE_TOKENS),
        )?;
        let next = zeros(device, vec![3, STATE_COLUMNS as i32], Dtype::BF16)?;
        Ok(Self {
            extent,
            extent_pages: pages,
            page_table,
            conv: [copy(device, self.current())?, next],
            parity: 0,
            committed: self.committed,
            rings: self
                .rings
                .iter()
                .map(|[k, v]| Ok([copy(device, k)?, copy(device, v)?]))
                .collect::<Result<_>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_put_full_applications_first() {
        let regions = (0..APPLICATIONS).map(region).collect::<Vec<_>>();
        for (i, a) in (0..APPLICATIONS).filter(|&a| full(a)).enumerate() {
            assert_eq!(regions[a], i);
        }
        let mut sorted = regions.clone();
        sorted.sort();
        assert_eq!(sorted, (0..APPLICATIONS).collect::<Vec<_>>());
        assert_eq!(region(0), 8);
        assert_eq!(region(6), 13);
        assert_eq!(region(47), 7);
    }

    #[test]
    fn extents_grow_in_whole_granules_within_the_layer_offsets() {
        assert_eq!(pages_for(0), 64);
        assert_eq!(pages_for(2048 - SCRATCH_TOKENS), 64);
        assert_eq!(pages_for(2048), 128);
        const {
            assert!(MOST_PAGES.is_multiple_of(PAGE_GRANULE));
            assert!((APPLICATIONS - 1) * region_bytes(MOST_PAGES) <= u32::MAX as usize);
            assert!(MOST_TOKENS >= 32768);
        }
    }

    #[test]
    fn every_extent_shape_fits_i32_rows() {
        // Extents past 2 GiB (about 21k tokens) must still allocate.
        assert!(APPLICATIONS * region_bytes(pages_for(MOST_TOKENS)) > i32::MAX as usize);
        for pages in (PAGE_GRANULE..=MOST_PAGES).step_by(PAGE_GRANULE) {
            let bytes = APPLICATIONS * region_bytes(pages);
            assert!(bytes.is_multiple_of(EXTENT_ROW));
            assert!(i32::try_from(bytes / EXTENT_ROW).is_ok());
        }
    }
}
