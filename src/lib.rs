#![doc = include_str!("../README.md")]

mod dispatch;
mod error;
mod format;
pub mod header;
pub mod metadata;
mod reader;
mod writer;

pub use error::{Result, WavError};
pub use format::{RawSpec, SampleFormat, WavSpec};
pub use header::{Chunk, WavParams};
pub use metadata::{Bext, InfoList};
pub use reader::WavReader;
pub use writer::WavWriter;
