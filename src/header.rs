//! Reading and writing wav headers.
//!
//! The chunk parsing and header layout are adapted from the wav handling in
//! CamillaDSP (<https://github.com/HEnquist/camilladsp>), generalized to this
//! crate's [`SampleFormat`] and error types.

use std::convert::TryInto;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem;

use crate::error::{Result, WavError};
use crate::format::SampleFormat;

const RIFF: &[u8] = b"RIFF";
const WAVE: &[u8] = b"WAVE";
const DATA: &[u8] = b"data";
const FMT: &[u8] = b"fmt ";

/// Byte offset of the 32-bit RIFF chunk size field, measured from the start of the file.
pub(crate) const RIFF_SIZE_OFFSET: u64 = 4;

/// Whether a header is written as `WAVE_FORMAT_EXTENSIBLE`.
///
/// Two cases force the extensible form:
///
/// * 24-bit-in-4-byte data is ambiguous as plain PCM (the block alignment
///   implies a 4-byte/32-bit sample, but only 24 bits are meaningful), so we
///   write it the strict-spec way: the 32-bit container size in `wBitsPerSample`
///   and the real 24 bits in `wValidBitsPerSample`.
/// * More than two channels: the spec recommends extensible (with a channel
///   mask) once the layout is no longer plain mono/stereo.
///
/// Anything else is unambiguous as plain PCM or float and uses the minimal
/// 16-byte `fmt ` chunk.
pub(crate) fn writes_as_extensible(channels: usize, format: SampleFormat) -> bool {
    matches!(format, SampleFormat::I24_4) || channels > 2
}

/// The size in bytes of the `fmt ` chunk body written for a header.
pub(crate) fn fmt_body_size(channels: usize, format: SampleFormat) -> u32 {
    if writes_as_extensible(channels, format) {
        FmtChunk::EXTENSIBLE_SIZE
    } else {
        FmtChunk::CORE_SIZE
    }
}

/// Byte offset of the 32-bit data chunk size field for a header.
///
/// The header is RIFF id + size + WAVE (12 bytes), the `fmt ` chunk id + size
/// (8 bytes) and body, then the data chunk id (4 bytes); the size field follows.
pub(crate) fn data_size_offset(channels: usize, format: SampleFormat) -> u64 {
    12 + 8 + fmt_body_size(channels, format) as u64 + 4
}

/// Value to write into the RIFF chunk size field for a finished file.
///
/// It covers everything after the size field: the WAVE id, the whole `fmt `
/// chunk and the whole data chunk.
pub(crate) fn riff_size(channels: usize, format: SampleFormat, data_bytes: u64) -> u64 {
    4 + (8 + fmt_body_size(channels, format) as u64) + (8 + data_bytes)
}

/// Windows GUID, used to give the sample format in the extended
/// `WAVEFORMATEXTENSIBLE` wav header.
#[derive(Debug, PartialEq, Eq)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

impl Guid {
    fn from_slice(data: &[u8; 16]) -> Guid {
        Guid {
            data1: read_u32(data, 0),
            data2: read_u16(data, 4),
            data3: read_u16(data, 6),
            data4: data[8..16].try_into().unwrap_or([0; 8]),
        }
    }

    fn to_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.data1.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.data2.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.data3.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.data4);
        bytes
    }
}

/// `KSDATAFORMAT_SUBTYPE_IEEE_FLOAT`
const SUBTYPE_FLOAT: Guid = Guid {
    data1: 3,
    data2: 0,
    data3: 16,
    data4: [128, 0, 0, 170, 0, 56, 155, 113],
};

/// `KSDATAFORMAT_SUBTYPE_PCM`
const SUBTYPE_PCM: Guid = Guid {
    data1: 1,
    data2: 0,
    data3: 16,
    data4: [128, 0, 0, 170, 0, 56, 155, 113],
};

/// The parameters extracted from a wav header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavParams {
    /// The binary sample format of the audio data.
    pub sample_format: SampleFormat,
    /// The sample rate in Hz.
    pub sample_rate: usize,
    /// The number of channels.
    pub channels: usize,
    /// Byte offset from the start of the file to the first audio sample.
    pub data_offset: usize,
    /// The length of the audio data in bytes, as declared in the header.
    ///
    /// Files written in streaming mode declare this as [`u32::MAX`] because the
    /// final length is not known up front, so do not rely on it to be accurate.
    pub data_length: usize,
}

fn read_u32(buffer: &[u8], start_index: usize) -> u32 {
    u32::from_le_bytes(
        buffer[start_index..start_index + mem::size_of::<u32>()]
            .try_into()
            .unwrap_or_default(),
    )
}

fn read_u16(buffer: &[u8], start_index: usize) -> u16 {
    u16::from_le_bytes(
        buffer[start_index..start_index + mem::size_of::<u16>()]
            .try_into()
            .unwrap_or_default(),
    )
}

fn compare_4cc(buffer: &[u8], bytes: &[u8]) -> bool {
    buffer.iter().take(4).zip(bytes).all(|(a, b)| *a == *b)
}

/// Write a chunk header: a four-character code followed by the little-endian
/// 32-bit chunk size.
fn write_chunk_header(dest: &mut impl Write, fourcc: &[u8], size: u32) -> std::io::Result<()> {
    dest.write_all(fourcc)?;
    dest.write_all(&size.to_le_bytes())
}

/// The `WAVEFORMATEXTENSIBLE` extension: the 24 bytes that follow the 16-byte
/// core (cbSize is implied to be 22).
struct FmtExtension {
    valid_bits_per_sample: u16,
    channel_mask: u32,
    sub_format: [u8; 16],
}

/// The fields of a `fmt ` chunk body, with an optional extensible extension.
///
/// This gives the byte layout a single named definition, shared by the header
/// parser and writer instead of repeating raw offsets and byte sequences. The
/// parser populates only the 16-byte core (the extension fields are read
/// directly in [`look_up_extended_format`]); the writer fills in the extension
/// for formats that need an extensible header.
struct FmtChunk {
    format_code: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    extension: Option<FmtExtension>,
}

impl FmtChunk {
    /// Size of the 16-byte core `fmt ` chunk body.
    const CORE_SIZE: u32 = 16;
    /// Size of the 40-byte `WAVEFORMATEXTENSIBLE` `fmt ` chunk body.
    const EXTENSIBLE_SIZE: u32 = 40;

    /// Build the chunk describing a given format.
    ///
    /// Returns an error if the parameters cannot be represented in the header
    /// fields, rather than silently truncating an out-of-range value.
    fn for_format(
        channels: usize,
        sample_format: SampleFormat,
        sample_rate: usize,
    ) -> Result<Self> {
        if channels == 0 {
            return Err(WavError::InvalidSpec(
                "channel count must be at least 1".to_string(),
            ));
        }
        let bytes_per_sample = sample_format.bytes_per_sample();
        let channels_u16 = u16::try_from(channels).map_err(|_| {
            WavError::InvalidSpec(format!("channel count {channels} does not fit in 16 bits"))
        })?;
        let sample_rate_u32 = u32::try_from(sample_rate).map_err(|_| {
            WavError::InvalidSpec(format!("sample rate {sample_rate} does not fit in 32 bits"))
        })?;
        let block_align = channels
            .checked_mul(bytes_per_sample)
            .and_then(|v| u16::try_from(v).ok())
            .ok_or_else(|| {
                WavError::InvalidSpec(format!(
                    "block alignment for {channels} channels of {sample_format:?} does not fit in 16 bits"
                ))
            })?;
        let byte_rate = channels
            .checked_mul(sample_rate)
            .and_then(|v| v.checked_mul(bytes_per_sample))
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                WavError::InvalidSpec("bytes per second does not fit in 32 bits".to_string())
            })?;
        // The bit depth and container width are both at most 64, so these casts
        // never truncate.
        if writes_as_extensible(channels, sample_format) {
            // Strict-spec extensible: wBitsPerSample carries the container size
            // (bytes per sample * 8), and the real depth goes in validBits. The
            // subformat GUID mirrors the plain format code (PCM vs IEEE float).
            // The channel mask is left at 0 ("no assignment") since the spec
            // carries no speaker layout to map.
            let sub_format = match sample_format.format_code() {
                3 => SUBTYPE_FLOAT,
                _ => SUBTYPE_PCM,
            };
            Ok(FmtChunk {
                format_code: 0xFFFE,
                channels: channels_u16,
                sample_rate: sample_rate_u32,
                byte_rate,
                block_align,
                bits_per_sample: (bytes_per_sample * 8) as u16,
                extension: Some(FmtExtension {
                    valid_bits_per_sample: sample_format.bits_per_sample() as u16,
                    channel_mask: 0,
                    sub_format: sub_format.to_bytes(),
                }),
            })
        } else {
            Ok(FmtChunk {
                format_code: sample_format.format_code(),
                channels: channels_u16,
                sample_rate: sample_rate_u32,
                byte_rate,
                block_align,
                bits_per_sample: sample_format.bits_per_sample() as u16,
                extension: None,
            })
        }
    }

    /// Parse the first 16 bytes of a `fmt ` chunk body (the core; any extensible
    /// fields are read separately in [`look_up_extended_format`]).
    fn parse(data: &[u8]) -> Self {
        FmtChunk {
            format_code: read_u16(data, 0),
            channels: read_u16(data, 2),
            sample_rate: read_u32(data, 4),
            byte_rate: read_u32(data, 8),
            block_align: read_u16(data, 12),
            bits_per_sample: read_u16(data, 14),
            extension: None,
        }
    }

    /// The size in bytes of this chunk's body when written.
    fn body_size(&self) -> u32 {
        if self.extension.is_some() {
            Self::EXTENSIBLE_SIZE
        } else {
            Self::CORE_SIZE
        }
    }

    /// Write the `fmt ` chunk body (16 bytes, or 40 with the extension).
    fn write_body(&self, dest: &mut impl Write) -> std::io::Result<()> {
        dest.write_all(&self.format_code.to_le_bytes())?;
        dest.write_all(&self.channels.to_le_bytes())?;
        dest.write_all(&self.sample_rate.to_le_bytes())?;
        dest.write_all(&self.byte_rate.to_le_bytes())?;
        dest.write_all(&self.block_align.to_le_bytes())?;
        dest.write_all(&self.bits_per_sample.to_le_bytes())?;
        if let Some(ext) = &self.extension {
            dest.write_all(&22u16.to_le_bytes())?; // cbSize
            dest.write_all(&ext.valid_bits_per_sample.to_le_bytes())?;
            dest.write_all(&ext.channel_mask.to_le_bytes())?;
            dest.write_all(&ext.sub_format)?;
        }
        Ok(())
    }

    /// The number of bytes per single-channel sample, derived from the block
    /// alignment and channel count. Returns `None` if the channel count is zero.
    fn bytes_per_sample(&self) -> Option<u16> {
        self.block_align.checked_div(self.channels)
    }
}

fn look_up_format(
    data: &[u8],
    formatcode: u16,
    bits: u16,
    bytes_per_sample: u16,
    chunk_length: u32,
) -> Result<SampleFormat> {
    match (formatcode, bits, bytes_per_sample) {
        (1, 16, 2) => Ok(SampleFormat::I16),
        (1, 24, 3) => Ok(SampleFormat::I24_3),
        (1, 24, 4) => Ok(SampleFormat::I24_4),
        (1, 32, 4) => Ok(SampleFormat::I32),
        (3, 32, 4) => Ok(SampleFormat::F32),
        (3, 64, 8) => Ok(SampleFormat::F64),
        (0xFFFE, _, _) => look_up_extended_format(data, bits, bytes_per_sample, chunk_length),
        (code, bits, bytes) => Err(WavError::UnsupportedFormat(format!(
            "format code {code}, {bits} bits, {bytes} bytes per sample"
        ))),
    }
}

fn look_up_extended_format(
    data: &[u8],
    bits: u16,
    bytes_per_sample: u16,
    chunk_length: u32,
) -> Result<SampleFormat> {
    if chunk_length != 40 {
        return Err(WavError::InvalidHeader(
            "extended fmt chunk must be 40 bytes".to_string(),
        ));
    }
    let valid_bits_per_sample = read_u16(data, 18);
    let subformat = &data[24..40];
    let subformat_guid = Guid::from_slice(subformat.try_into().unwrap());
    match (
        subformat_guid,
        bits,
        bytes_per_sample,
        valid_bits_per_sample,
    ) {
        (SUBTYPE_PCM, 16, 2, 16) => Ok(SampleFormat::I16),
        (SUBTYPE_PCM, 24, 3, 24) => Ok(SampleFormat::I24_3),
        // 24-in-4-byte: the lenient form (wBitsPerSample = 24) and the
        // strict-spec form (wBitsPerSample = container size 32, validBits = 24).
        (SUBTYPE_PCM, 24, 4, 24) => Ok(SampleFormat::I24_4),
        (SUBTYPE_PCM, 32, 4, 24) => Ok(SampleFormat::I24_4),
        (SUBTYPE_PCM, 32, 4, 32) => Ok(SampleFormat::I32),
        (SUBTYPE_FLOAT, 32, 4, 32) => Ok(SampleFormat::F32),
        (SUBTYPE_FLOAT, 64, 8, 64) => Ok(SampleFormat::F64),
        (guid, bits, bytes, valid) => Err(WavError::UnsupportedFormat(format!(
            "extended subformat {guid:?}, {bits} bits, {bytes} bytes per sample, {valid} valid bits"
        ))),
    }
}

/// Parse the header of a wav stream, returning the [`WavParams`].
///
/// The stream is left positioned at an unspecified location; callers that want
/// to read audio data should seek to [`WavParams::data_offset`] afterwards.
pub fn read_wav_header(mut stream: impl Read + Seek) -> Result<WavParams> {
    let filesize = stream.seek(SeekFrom::End(0))?;
    stream.seek(SeekFrom::Start(0))?;
    let mut file = BufReader::new(stream);
    let mut header = [0; 12];
    file.read_exact(&mut header)?;

    // The file must start with RIFF, and bytes 8..12 must be WAVE.
    if !compare_4cc(&header, RIFF) || !compare_4cc(&header[8..], WAVE) {
        return Err(WavError::InvalidHeader(
            "missing RIFF or WAVE marker".to_string(),
        ));
    }

    let mut next_chunk_location = 12;
    let mut found_fmt = false;
    let mut found_data = false;
    let mut buffer = [0; 8];

    // Dummy values until the real ones are found.
    let mut sample_format = SampleFormat::I16;
    let mut sample_rate = 0;
    let mut channels = 0;
    let mut data_offset = 0;
    let mut data_length = 0;

    // Analyze each chunk to find the format and the data.
    while (!found_fmt || !found_data) && next_chunk_location < filesize {
        file.seek(SeekFrom::Start(next_chunk_location))?;
        file.read_exact(&mut buffer)?;
        let chunk_length = read_u32(&buffer, 4);
        let is_data = compare_4cc(&buffer, DATA);
        let is_fmt = compare_4cc(&buffer, FMT);
        if is_fmt && (chunk_length == 16 || chunk_length == 18 || chunk_length == 40) {
            found_fmt = true;
            let mut data = vec![0; chunk_length as usize];
            file.read_exact(&mut data)?;
            let fmt = FmtChunk::parse(&data);
            channels = fmt.channels;
            sample_rate = fmt.sample_rate;
            let bytes_per_sample = fmt
                .bytes_per_sample()
                .ok_or_else(|| WavError::InvalidHeader("zero channels".to_string()))?;
            sample_format = look_up_format(
                &data,
                fmt.format_code,
                fmt.bits_per_sample,
                bytes_per_sample,
                chunk_length,
            )?;
        } else if is_data {
            found_data = true;
            data_offset = next_chunk_location + 8;
            data_length = chunk_length;
        }
        next_chunk_location += 8 + chunk_length as u64;
    }
    if found_data && found_fmt {
        return Ok(WavParams {
            sample_format,
            sample_rate: sample_rate as usize,
            channels: channels as usize,
            data_length: data_length as usize,
            data_offset: data_offset as usize,
        });
    }
    Err(WavError::InvalidHeader(
        "could not find both fmt and data chunks".to_string(),
    ))
}

/// Write a wav header: RIFF + WAVE, a `fmt ` chunk (16-byte core, or a 40-byte
/// `WAVEFORMATEXTENSIBLE` chunk for formats that need it), then the data chunk
/// header.
///
/// The RIFF size and data size fields are written from the supplied values. Pass
/// [`u32::MAX`] for streaming output where the final length is not yet known, or
/// the real byte counts when they are known (for example when patching the
/// header on finalize).
///
/// Returns [`WavError::InvalidSpec`] if the channel count or sample rate cannot
/// be represented in the header fields.
pub fn write_wav_header(
    dest: &mut impl Write,
    channels: usize,
    sample_format: SampleFormat,
    sample_rate: usize,
    riff_size: u32,
    data_size: u32,
) -> Result<()> {
    let fmt = FmtChunk::for_format(channels, sample_format, sample_rate)?;

    // RIFF chunk header, then the WAVE form type.
    write_chunk_header(dest, RIFF, riff_size)?;
    dest.write_all(WAVE)?;

    // fmt chunk.
    write_chunk_header(dest, FMT, fmt.body_size())?;
    fmt.write_body(dest)?;

    // data chunk header. The audio data starts immediately after.
    write_chunk_header(dest, DATA, data_size)?;

    Ok(())
}
