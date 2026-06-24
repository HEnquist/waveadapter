//! Error type for the waveadapter crate.

use std::fmt;
use std::io;

/// Result type used throughout this crate.
pub type Result<T> = std::result::Result<T, WavError>;

/// Errors that can occur while reading or writing wav files.
#[derive(Debug)]
pub enum WavError {
    /// An underlying I/O error.
    Io(io::Error),
    /// The data does not contain a valid wav header,
    /// for example because the RIFF or WAVE markers are missing,
    /// or a required chunk could not be found.
    InvalidHeader(String),
    /// The file uses a sample format that this crate does not support.
    UnsupportedFormat(String),
    /// The requested output parameters cannot be represented in a wav header,
    /// for example a channel count or sample rate that does not fit in the
    /// header fields, or a value that would overflow a derived field.
    InvalidSpec(String),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WavError::Io(err) => write!(f, "I/O error: {err}"),
            WavError::InvalidHeader(msg) => write!(f, "Invalid wav header: {msg}"),
            WavError::UnsupportedFormat(msg) => write!(f, "Unsupported wav format: {msg}"),
            WavError::InvalidSpec(msg) => write!(f, "Invalid wav output spec: {msg}"),
        }
    }
}

impl std::error::Error for WavError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WavError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for WavError {
    fn from(err: io::Error) -> Self {
        WavError::Io(err)
    }
}
