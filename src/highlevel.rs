//! One-call helpers for the common "just read/write a file" cases.
//!
//! These wrap [`WavReader`]/[`WavWriter`] for callers who do not need streaming,
//! random access or chunk handling: open a path, get the audio as floats; or
//! hand over a float buffer and a format, get a finalized file. Anything beyond
//! that (metadata chunks, RF64, raw formats, seeking) needs the full
//! reader/writer types.

use std::fs::File;
use std::path::Path;

use audioadapter::Adapter;
use audioadapter_buffers::owned::InterleavedOwned;
use num_traits::float::FloatCore;
use num_traits::{ToPrimitive, Zero};

use crate::error::Result;
use crate::format::{SampleFormat, WavSpec};
use crate::reader::WavReader;
use crate::writer::WavWriter;

/// The decoded contents of a wav file: the audio as an owned interleaved float
/// buffer, plus the sample rate (which the buffer itself does not carry).
///
/// Returned by [`read_wav_file`]. The channel count and frame count live on the
/// [`samples`](WavData::samples) buffer and are surfaced here as
/// [`channels`](WavData::channels) / [`frames`](WavData::frames) for convenience.
pub struct WavData<T> {
    /// The audio samples, scaled to -1.0..1.0, as an interleaved buffer.
    pub samples: InterleavedOwned<T>,
    /// The sample rate in Hz.
    pub sample_rate: usize,
}

impl<T: Clone> WavData<T> {
    /// The number of channels.
    pub fn channels(&self) -> usize {
        self.samples.channels()
    }

    /// The number of frames.
    pub fn frames(&self) -> usize {
        self.samples.frames()
    }
}

/// Read an entire wav file at `path` into floats.
///
/// Opens the file, parses the header and reads all frames into an owned
/// interleaved buffer, converting whatever the on-disk format is into floats
/// scaled to -1.0..1.0. The element type is chosen by the caller; pick `f32` or
/// `f64`:
///
/// ```no_run
/// let audio = waveadapter::read_wav_file::<f32, _>("input.wav")?;
/// println!("{} ch, {} Hz, {} frames", audio.channels(), audio.sample_rate, audio.frames());
/// # Ok::<(), waveadapter::WavError>(())
/// ```
///
/// This is the "I don't care, just give me the data" path. It does not expose
/// metadata chunks and returns [`WavError::UnsupportedFormat`](crate::WavError::UnsupportedFormat)
/// for formats the float path cannot decode (such as 8-bit PCM or A-law); use
/// [`WavReader`] directly for those.
pub fn read_wav_file<T, P>(path: P) -> Result<WavData<T>>
where
    T: FloatCore + ToPrimitive + Zero,
    P: AsRef<Path>,
{
    let mut reader = WavReader::new(File::open(path)?)?;
    let sample_rate = reader.sample_rate();
    let samples = reader.read_all_to_float::<T>()?;
    Ok(WavData {
        samples,
        sample_rate,
    })
}

/// Write a float buffer to a wav file at `path` in the given format.
///
/// Creates (or truncates) the file, writes a standard RIFF/WAVE header for the
/// buffer's channel count and the given `sample_rate`, converts the samples from
/// -1.0..1.0 into `sample_format`, and finalizes the size fields. For a 16-bit
/// file, pass [`SampleFormat::I16`]:
///
/// ```no_run
/// # use audioadapter_buffers::owned::InterleavedOwned;
/// # let buffer = InterleavedOwned::<f32>::new(0.0, 2, 0);
/// use waveadapter::SampleFormat;
/// let clipped = waveadapter::write_wav_file("output.wav", &buffer, 44100, SampleFormat::I16)?;
/// # Ok::<(), waveadapter::WavError>(())
/// ```
///
/// Returns the number of samples that were clipped during conversion. This is
/// the "I have this audio, just write it" path: a plain RIFF file with no extra
/// chunks. Use [`WavWriter`] for metadata, RF64, raw formats or streaming output.
pub fn write_wav_file<T, P>(
    path: P,
    samples: &dyn Adapter<T>,
    sample_rate: usize,
    sample_format: SampleFormat,
) -> Result<usize>
where
    T: FloatCore + ToPrimitive,
    P: AsRef<Path>,
{
    let spec = WavSpec::new(samples.channels(), sample_rate, sample_format);
    let mut writer = WavWriter::new(File::create(path)?, spec)?;
    let clipped = writer.write_float_buffer(samples)?;
    writer.finalize()?;
    Ok(clipped)
}
