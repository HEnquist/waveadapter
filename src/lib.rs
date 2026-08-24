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
pub use format::{SampleFormat, WavSpec};
pub use header::{Chunk, FmtChunk, WavParams};
pub use highlevel::{WavData, read_wav_file, write_wav_file, write_wav_file_raw};
pub use metadata::{AdtlEntry, AdtlList, Bext, Cue, CuePoint, InfoList, SampleLoop, Smpl};
pub use reader::WavReader;
pub use writer::{Fact, IntoFmtChunk, Rf64, Riff, Truncate, WavWriter, WavWriterBuilder};
