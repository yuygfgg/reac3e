//! Error type for the encoder.

use thiserror::Error;

/// Errors returned while configuring or running the encoder.
#[derive(Debug, Error)]
pub enum EncodeError {
    /// A configuration value is not supported by the encoder or by E-AC-3.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// The PCM frame does not match the configured channel layout.
    #[error("invalid PCM frame: {0}")]
    Pcm(String),

    /// Bit allocation could not fit the frame at the configured bitrate.
    #[error("bit allocation failed; try increasing the bitrate")]
    BitAllocation,

    /// An internal invariant was violated.  This indicates a bug in the crate.
    #[error("internal error: {0}")]
    Internal(String),
}
