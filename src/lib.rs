//! Reading and writing wav files into and out of [audioadapter] buffers.
//!
//! This crate handles the wav container (header parsing and writing, borrowed
//! and generalized from CamillaDSP) and bridges it to the byte and sample
//! handling in the audioadapter family of crates. Audio data can be read into
//! and written from any [`Adapter`](audioadapter::Adapter) /
//! [`AdapterMut`](audioadapter::AdapterMut) buffer as scaled floating point
//! samples, or moved as raw interleaved bytes for the caller to wrap with the
//! audioadapter adapters directly.
//!
//! # Examples
//!
//! Read a wav file into an owned float buffer:
//!
//! ```no_run
//! use std::fs::File;
//! use waveadapter::WavReader;
//!
//! let mut reader = WavReader::new(File::open("input.wav")?)?;
//! let buffer = reader.read_all_to_float::<f32>()?;
//! # Ok::<(), waveadapter::WavError>(())
//! ```
//!
//! Write a float buffer to a new wav file:
//!
//! ```no_run
//! use std::fs::File;
//! use audioadapter_buffers::owned::InterleavedOwned;
//! use waveadapter::{SampleFormat, WavSpec, WavWriter};
//!
//! let data = InterleavedOwned::<f32>::new(0.0, 2, 128);
//! let spec = WavSpec {
//!     channels: 2,
//!     sample_rate: 44100,
//!     sample_format: SampleFormat::I32,
//! };
//! let mut writer = WavWriter::new(File::create("output.wav")?, spec)?;
//! writer.write_float_buffer(&data)?;
//! writer.finalize()?;
//! # Ok::<(), waveadapter::WavError>(())
//! ```

mod dispatch;
mod error;
mod format;
pub mod header;
mod reader;
mod writer;

pub use error::{Result, WavError};
pub use format::{SampleFormat, WavSpec};
pub use header::WavParams;
pub use reader::WavReader;
pub use writer::WavWriter;
