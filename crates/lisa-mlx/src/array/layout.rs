// ─────────────────────────────── layout ───────────────────────────────

/// Shape, element strides and element offset. Strides are in elements (not
/// bytes) so a view can be described without knowing the dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Layout {
    /// Row-major strides for `shape`.
    pub fn contiguous(shape: &[usize]) -> Self {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        Self {
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// `Layout::stride()`.
    pub fn stride(&self) -> &[usize] {
        &self.strides
    }

    /// `Layout::shape()`.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// `Layout::start_offset()`.
    pub fn start_offset(&self) -> usize {
        self.offset
    }

    /// Row-major contiguous, treating size-1 dimensions as stride-agnostic
    /// (as MLX does).
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for (&d, &s) in self.shape.iter().zip(self.strides.iter()).rev() {
            if d != 1 && s != expected {
                return false;
            }
            expected *= d;
        }
        true
    }
}
