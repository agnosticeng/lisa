//! Packed projection semantics, independent of the reader supplying the bytes.
use lisa_backend::{Kernel, PackedCompute, PackedEncoding};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    Plain,
    Q8,
    Block(PackedEncoding),
}
impl Projection {
    pub fn linear(self, round_output: bool) -> Kernel {
        Kernel::PackedCompute(match self {
            Self::Plain => PackedCompute::PlainLinear { round_output },
            Self::Q8 => PackedCompute::Q8Linear {
                round_output,
                round_input: false,
            },
            Self::Block(encoding) => PackedCompute::BlockLinear {
                encoding,
                round_output,
            },
        })
    }
    pub fn experts(self) -> anyhow::Result<Kernel> {
        let Self::Block(encoding) = self else {
            anyhow::bail!("expert projection requires original packed block storage");
        };
        Ok(Kernel::PackedCompute(PackedCompute::ExpertLinear {
            encoding,
        }))
    }
}
