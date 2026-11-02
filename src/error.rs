use std::fmt;

/// Unified error type for the engine.
#[derive(Debug)]
pub enum DeferError {
    Io(std::io::Error),
    Parse(String),
    Model(String),
    Checkpoint(String),
    Wal(String),
    Tokenizer(String),
    /// Replay produced a different result than what was recorded.
    Divergence(String),
}

pub type Result<T> = std::result::Result<T, DeferError>;

impl fmt::Display for DeferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeferError::Io(e) => write!(f, "I/O error: {e}"),
            DeferError::Parse(m) => write!(f, "parse error: {m}"),
            DeferError::Model(m) => write!(f, "model error: {m}"),
            DeferError::Checkpoint(m) => write!(f, "checkpoint error: {m}"),
            DeferError::Wal(m) => write!(f, "WAL error: {m}"),
            DeferError::Tokenizer(m) => write!(f, "tokenizer error: {m}"),
            DeferError::Divergence(m) => write!(f, "determinism violation: {m}"),
        }
    }
}

impl std::error::Error for DeferError {}

impl From<std::io::Error> for DeferError {
    fn from(e: std::io::Error) -> Self {
        DeferError::Io(e)
    }
}
