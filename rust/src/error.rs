//! Error type shared by the whole crate.
//!
//! Per the project error policy, invalid input is rejected eagerly with a
//! `Result` — the library never panics on bad input.  Two conditions are
//! *reported, not raised*: calendar arbitrage in an input surface (violation
//! count on the surface) and PSOR non-convergence (logged, computation
//! proceeds).

use std::fmt;

/// Errors produced by `localvol` (manual thiserror-style enum).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalVolError {
    /// A caller-supplied value violates the documented domain
    /// (non-finite, out of range, inconsistent shapes, ...).
    InvalidInput(String),
    /// A numerical kernel could not proceed (e.g. a zero pivot in the
    /// Thomas solver) or a price violates no-arbitrage bounds.
    Numerical(String),
    /// A data file could not be read or parsed.
    Io(String),
}

impl fmt::Display for LocalVolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocalVolError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            LocalVolError::Numerical(msg) => write!(f, "numerical error: {msg}"),
            LocalVolError::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for LocalVolError {}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, LocalVolError>;

/// Shorthand for building an [`LocalVolError::InvalidInput`].
pub(crate) fn invalid(msg: impl Into<String>) -> LocalVolError {
    LocalVolError::InvalidInput(msg.into())
}
