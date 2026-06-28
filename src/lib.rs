#![doc = include_str!("../README.md")]

mod dispatch;
mod error;
mod format;
pub mod header;
mod highlevel;
pub mod metadata;
mod reader;
mod writer;

pub use error::{Result, WavError};
pub use format::{RawSpec, SampleFormat, WavSpec};
pub use header::{Chunk, WavParams};
pub use highlevel::{WavData, read_wav_file, write_wav_file};
pub use metadata::{AdtlEntry, AdtlList, Bext, Cue, CuePoint, InfoList};
pub use reader::WavReader;
pub use writer::WavWriter;
