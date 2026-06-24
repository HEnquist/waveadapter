#![doc = include_str!("../README.md")]

mod dispatch;
mod error;
mod format;
pub mod header;
mod reader;
mod writer;

pub use error::{Result, WavError};
pub use format::{SampleFormat, WavSpec};
pub use header::{Chunk, WavParams};
pub use reader::WavReader;
pub use writer::WavWriter;
