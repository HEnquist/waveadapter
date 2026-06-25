//! Sample formats supported in wav files.

use crate::dispatch::with_sample_type;

/// The binary sample format of the audio data in a wav file.
///
/// Wav data is always little-endian, and 24-bit-in-4-byte data is always left
/// justified, so those qualifiers are left out of the names. Each variant
/// corresponds to one of the byte-wrapper sample types from
/// [`audioadapter_sample::sample`], which is what the reader and writer use to
/// convert between raw bytes and numbers.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Signed integer, 16 bits in 2 bytes.
    I16,
    /// Signed integer, 24 bits in 3 bytes (packed).
    I24_3,
    /// Signed integer, 24 bits in 4 bytes (padded with a zero low byte).
    I24_4,
    /// Signed integer, 32 bits in 4 bytes.
    I32,
    /// Single precision floating point, 32 bits in 4 bytes.
    F32,
    /// Double precision floating point, 64 bits in 8 bytes.
    F64,
}

impl SampleFormat {
    /// The number of significant bits per sample, as stored in the wav `fmt ` chunk.
    pub fn bits_per_sample(&self) -> usize {
        match self {
            SampleFormat::I16 => 16,
            SampleFormat::I24_3 => 24,
            SampleFormat::I24_4 => 24,
            SampleFormat::I32 => 32,
            SampleFormat::F32 => 32,
            SampleFormat::F64 => 64,
        }
    }

    /// The number of bytes occupied by one sample on disk.
    ///
    /// Sourced from the `BYTES_PER_SAMPLE` constant of the corresponding
    /// audioadapter byte-wrapper sample type, so it stays in sync with it.
    pub fn bytes_per_sample(&self) -> usize {
        with_sample_type!(*self, S, { S::BYTES_PER_SAMPLE })
    }

    /// The wav format code: `1` for integer PCM, `3` for IEEE float.
    pub fn format_code(&self) -> u16 {
        match self {
            SampleFormat::F32 | SampleFormat::F64 => 3,
            _ => 1,
        }
    }
}

/// The properties needed to start writing a wav file: channel count, sample
/// rate and sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavSpec {
    /// The number of channels.
    pub channels: usize,
    /// The sample rate in Hz.
    pub sample_rate: usize,
    /// The binary sample format to store the audio data as.
    pub sample_format: SampleFormat,
    /// The speaker-position channel mask (`dwChannelMask`) to write into a
    /// `WAVEFORMATEXTENSIBLE` header, or `None` to leave it unspecified (`0`).
    ///
    /// This crate does not interpret the mask, it only stores it. A non-zero mask
    /// must have exactly one bit set per channel (`channel_mask.count_ones() ==
    /// channels`); otherwise the writer returns
    /// [`WavError::InvalidSpec`](crate::WavError::InvalidSpec). Supplying a
    /// non-zero mask forces the extensible header form even for mono/stereo, since
    /// that is the only place the mask can be stored.
    pub channel_mask: Option<u32>,
}

impl WavSpec {
    /// Build a spec with no channel mask (`channel_mask: None`).
    pub fn new(channels: usize, sample_rate: usize, sample_format: SampleFormat) -> Self {
        WavSpec {
            channels,
            sample_rate,
            sample_format,
            channel_mask: None,
        }
    }

    /// The number of bytes occupied by one frame (one sample for each channel).
    pub fn frame_bytes(&self) -> usize {
        self.channels * self.sample_format.bytes_per_sample()
    }
}

/// The properties needed to write a wav file in *raw* (uninterpreted) mode: the
/// `fmt ` chunk fields written verbatim, with no attempt to map them to a
/// [`SampleFormat`].
///
/// This is the write-side counterpart to a [`WavParams`](crate::WavParams) whose
/// `sample_format` is `None`: it lets a caller emit a container for a format this
/// crate does not model (8-bit PCM, A-law/µ-law, ADPCM, an exotic
/// `WAVEFORMATEXTENSIBLE` subtype, ...) and then push the audio through
/// [`WavWriter::write_raw_interleaved`](crate::WavWriter::write_raw_interleaved).
/// The float write path is not available for a raw writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawSpec {
    /// The `fmt ` format code (`wFormatTag`), for example `1` for integer PCM or
    /// `6`/`7` for A-law/µ-law.
    pub format_code: u16,
    /// The number of channels.
    pub channels: usize,
    /// The sample rate in Hz.
    pub sample_rate: usize,
    /// Bits per single-channel sample (`wBitsPerSample`).
    pub bits_per_sample: u16,
    /// Bytes per frame (`nBlockAlign`). This is what the reader and writer use to
    /// frame the raw byte stream, so the caller must set it to match the audio.
    pub block_align: u16,
}

impl RawSpec {
    /// The number of bytes occupied by one frame, taken directly from
    /// [`block_align`](RawSpec::block_align).
    pub fn frame_bytes(&self) -> usize {
        self.block_align as usize
    }
}
