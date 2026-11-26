//! The shim's error type.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Msg(String),
    /// Device allocation or host admission could not provide the requested
    /// contiguous storage. GPU submission/completion failures remain distinct.
    AllocationUnavailable(String),
}

impl Error {
    pub fn msg(m: impl Into<String>) -> Self {
        Error::Msg(m.into())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Msg(m) | Error::AllocationUnavailable(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

/// `bail!`.
#[macro_export]
macro_rules! bail {
    ($($t:tt)*) => {
        return Err($crate::error::Error::Msg(format!($($t)*)))
    };
}

/// mlx's exception type. Only `Exception::custom(msg)` is used by the tree.
pub struct Exception;

impl Exception {
    pub fn custom(msg: String) -> Error {
        Error::Msg(msg)
    }
}

impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self::msg(error.to_string())
    }
}
