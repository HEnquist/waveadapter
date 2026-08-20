//! One-call helpers for the common "just read/write a file" cases.
//!
//! These wrap [`WavReader`]/[`WavWriter`] for callers who do not need streaming,
//! random access or chunk handling: open a path, get the audio as floats; or
//! hand over a buffer and a format, get a finalized file. Anything beyond
//! that (metadata chunks, RF64, seeking) needs the full reader/writer types.
//!
//! The write side comes in both flavours, [`write_wav_file`] converting from
//! floats and [`write_wav_file_raw`] taking bytes that are already in the target
//! format. There is no read-side counterpart to the latter: decoding a format
//! this crate does not model needs the `fmt ` chunk as well as the bytes, so it
//! goes through [`WavReader`] and
//! [`read_raw_all`](WavReader::read_raw_all).

use std::fs::File;
use std::path::Path;

use audioadapter::Adapter;
use audioadapter_buffers::owned::InterleavedOwned;
use num_traits::float::FloatCore;
use num_traits::{ToPrimitive, Zero};

use crate::error::{Result, WavError};
use crate::format::{SampleFormat, WavSpec};
use crate::reader::WavReader;
use crate::writer::{IntoFmtChunk, WavWriter};

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
/// for formats the float path cannot decode (such as ADPCM); use
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
/// Returns the number of samples that were clipped during conversion, which is
/// always zero for the float formats (see
/// [`WavWriter::write_float_buffer`]). This is the "I have this audio, just
/// write it" path: a plain RIFF file with no extra chunks. Use [`WavWriter`]
/// for metadata, RF64, raw formats or streaming output.
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

/// Write already-encoded interleaved bytes to a wav file at `path`.
///
/// The raw counterpart to [`write_wav_file`]: the audio data is written exactly
/// as given, so `format` only has to describe it truthfully. It is anything
/// implementing [`IntoFmtChunk`], which means a [`WavSpec`] for a format this
/// crate models, or a [`FmtChunk`](crate::FmtChunk) for one it does not (ADPCM,
/// GSM, an exotic extensible subtype), including one taken straight off a file
/// that was just read.
///
/// ```no_run
/// use waveadapter::{SampleFormat, WavSpec, write_wav_file_raw};
///
/// let bytes: Vec<u8> = vec![0; 4 * 1024]; // interleaved 16-bit stereo
/// write_wav_file_raw("output.wav", &bytes, WavSpec::new(2, 44100, SampleFormat::I16))?;
/// # Ok::<(), waveadapter::WavError>(())
/// ```
///
/// The one part of the promise that is checked is the framing: `data` must be a
/// whole number of frames for the format, otherwise the file would declare a
/// `data` size that is not a multiple of `nBlockAlign`. A partial frame is
/// [`WavError::InvalidSpec`](crate::WavError::InvalidSpec) rather than a
/// silently malformed file. For a format this crate does not model the unit is
/// whatever `nBlockAlign` describes, so for a block-compressed format that means
/// whole compressed blocks.
///
/// This writes a plain seekable RIFF file with no extra chunks, and leaves the
/// `fact` chunk to [`Fact::Auto`](crate::Fact::Auto): counted for the non-PCM
/// formats this crate models, and *omitted* for one it does not, since
/// `data_bytes / block_align` is a block count there rather than a frame count.
/// So a block-compressed file written this way loses its frame count. Supplying
/// that count, like anything else beyond a plain file (metadata, RF64,
/// streaming), means going through [`WavWriter`] directly:
/// `WavWriter::builder(fmt)?.fact(Fact::Samples(n))`.
pub fn write_wav_file_raw<S, P>(path: P, data: &[u8], format: S) -> Result<()>
where
    S: IntoFmtChunk,
    P: AsRef<Path>,
{
    let fmt = format.into_fmt_chunk()?;
    let frame_bytes = fmt.frame_bytes();
    if frame_bytes == 0 {
        return Err(WavError::InvalidSpec(
            "cannot write raw audio data: block alignment is zero".to_string(),
        ));
    }
    let leftover = data.len() % frame_bytes;
    if leftover != 0 {
        return Err(WavError::InvalidSpec(format!(
            "raw audio data is {} bytes, {leftover} more than a whole number of \
             {frame_bytes}-byte frames",
            data.len()
        )));
    }
    let mut writer = WavWriter::builder(fmt)?.open(File::create(path)?)?;
    writer.write_raw_interleaved(data)?;
    writer.finalize()?;
    Ok(())
}
