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
    /// Unsigned integer, 8 bits in 1 byte, centered at 128.
    ///
    /// Wav 8-bit PCM is unsigned, unlike every deeper integer depth. A stored
    /// `0` is -1.0 and `255` is +1.0.
    U8,
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
    /// A-law companded, 1 byte per sample, as defined by ITU-T G.711.
    ///
    /// The telephony format of Europe and most of the world, stored in wav files
    /// as `WAVE_FORMAT_ALAW`. It packs roughly 13 bits of dynamic range into a
    /// byte by spacing the quantization steps logarithmically, so it is far
    /// better than [`U8`](SampleFormat::U8) at the same size, but it quantizes:
    /// a value written and read back does not come out unchanged.
    ALAW,
    /// Mu-law companded, 1 byte per sample, as defined by ITU-T G.711.
    ///
    /// The telephony format of North America and Japan, stored in wav files as
    /// `WAVE_FORMAT_MULAW`. Like [`ALAW`](SampleFormat::ALAW) but with roughly
    /// 14 bits of dynamic range, and it does have a code for exact silence.
    ///
    /// Also written μ-law, u-law or ulaw elsewhere; the spelling here follows the
    /// wav format tag.
    MULAW,
}

impl SampleFormat {
    /// The number of significant bits per sample, as stored in the wav `fmt ` chunk.
    pub fn bits_per_sample(&self) -> usize {
        match self {
            SampleFormat::U8 => 8,
            SampleFormat::I16 => 16,
            SampleFormat::I24_3 => 24,
            SampleFormat::I24_4 => 24,
            SampleFormat::I32 => 32,
            SampleFormat::F32 => 32,
            SampleFormat::F64 => 64,
            SampleFormat::ALAW => 8,
            SampleFormat::MULAW => 8,
        }
    }

    /// The number of bytes occupied by one sample on disk.
    ///
    /// Sourced from the `BYTES_PER_SAMPLE` constant of the corresponding
    /// audioadapter byte-wrapper sample type, so it stays in sync with it.
    pub fn bytes_per_sample(&self) -> usize {
        with_sample_type!(*self, S, { S::BYTES_PER_SAMPLE })
    }

    /// The wav format code: `1` for integer PCM, `3` for IEEE float, `6` for
    /// A-law and `7` for mu-law.
    pub fn format_code(&self) -> u16 {
        match self {
            SampleFormat::F32 | SampleFormat::F64 => 3,
            SampleFormat::ALAW => 6,
            SampleFormat::MULAW => 7,
            _ => 1,
        }
    }

    /// Whether this is plain integer PCM (`WAVE_FORMAT_PCM`).
    ///
    /// Everything else counts as non-PCM in the wav spec, which is what decides
    /// whether the header needs the `cbSize` field and a `fact` chunk.
    pub(crate) fn is_pcm(&self) -> bool {
        self.format_code() == 1
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

// The write-side counterpart to a `WavParams` whose format is not one this crate
// models used to be a `RawSpec` of loose fields. It is now `FmtChunk` in
// `header.rs`: the same struct the reader hands back, so the bytes survive a
// round trip instead of being rebuilt from a PCM-shaped guess.
