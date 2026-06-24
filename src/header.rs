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

/// A raw, uninterpreted wav chunk.
///
/// This crate parses only the `fmt ` and `data` chunks; every other chunk
/// (`LIST`/`INFO`, `bext`, `cue `, `fact`, `iXML`, `id3 `, ...) is exposed
/// verbatim so a higher-level library can give it meaning. The `data` chunk is
/// not included here (it is the audio, described by
/// [`WavParams::data_offset`]/[`data_length`](WavParams::data_length)); the
/// `fmt ` chunk is not included either, since it is already parsed into the
/// typed fields of [`WavParams`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The four-character chunk id, for example `*b"LIST"`.
    pub id: [u8; 4],
    /// The raw chunk body, excluding the 8-byte id/size header and any trailing
    /// pad byte.
    pub data: Vec<u8>,
}

/// The parameters extracted from a wav header.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Every non-audio chunk found in the file, in the order encountered, with
    /// its body bytes read out. See [`Chunk`].
    pub chunks: Vec<Chunk>,
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
    let mut chunks: Vec<Chunk> = Vec::new();

    // Walk every chunk to the end of the file, so that metadata chunks placed
    // after the data chunk are captured too. A chunk is padded to an even length
    // with a trailing byte that is not counted in its declared size.
    while next_chunk_location + 8 <= filesize {
        file.seek(SeekFrom::Start(next_chunk_location))?;
        file.read_exact(&mut buffer)?;
        let chunk_length = read_u32(&buffer, 4);
        let is_data = compare_4cc(&buffer, DATA);
        let is_fmt = compare_4cc(&buffer, FMT);
        if is_fmt {
            // Honor the first valid fmt chunk; ignore any later or malformed one.
            if !found_fmt && (chunk_length == 16 || chunk_length == 18 || chunk_length == 40) {
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
            }
        } else if is_data {
            // Honor the first data chunk; ignore any later one.
            if !found_data {
                found_data = true;
                data_offset = next_chunk_location + 8;
                data_length = chunk_length;
                // A streaming placeholder length means the data runs to the end
                // of the file, so there is nothing to scan past it.
                if chunk_length == u32::MAX {
                    break;
                }
            }
        } else {
            // Any other chunk is captured verbatim, tolerating a bogus length
            // that would overrun the file by stopping the scan instead of erroring.
            let body_end = next_chunk_location + 8 + chunk_length as u64;
            if body_end > filesize {
                break;
            }
            let mut body = vec![0; chunk_length as usize];
            file.read_exact(&mut body)?;
            let mut id = [0u8; 4];
            id.copy_from_slice(&buffer[0..4]);
            chunks.push(Chunk { id, data: body });
        }
        next_chunk_location += 8 + chunk_length as u64 + (chunk_length as u64 & 1);
    }
    if found_data && found_fmt {
        return Ok(WavParams {
            sample_format,
            sample_rate: sample_rate as usize,
            channels: channels as usize,
            data_length: data_length as usize,
            data_offset: data_offset as usize,
            chunks,
        });
    }
    Err(WavError::InvalidHeader(
        "could not find both fmt and data chunks".to_string(),
    ))
}

/// Write the RIFF chunk header and the WAVE form type (12 bytes).
///
/// Pass [`u32::MAX`] for `riff_size` in streaming output where the final length
/// is not yet known.
pub(crate) fn write_riff_wave(dest: &mut impl Write, riff_size: u32) -> std::io::Result<()> {
    write_chunk_header(dest, RIFF, riff_size)?;
    dest.write_all(WAVE)
}

/// Write the `fmt ` chunk (id, size and body) and return its body size in bytes.
///
/// Returns [`WavError::InvalidSpec`] if the channel count or sample rate cannot
/// be represented in the header fields.
pub(crate) fn write_fmt_chunk(
    dest: &mut impl Write,
    channels: usize,
    sample_format: SampleFormat,
    sample_rate: usize,
) -> Result<u32> {
    let fmt = FmtChunk::for_format(channels, sample_format, sample_rate)?;
    let body_size = fmt.body_size();
    write_chunk_header(dest, FMT, body_size)?;
    fmt.write_body(dest)?;
    Ok(body_size)
}

/// Write the `data` chunk header (id and size, 8 bytes). The audio data is
/// written immediately after.
///
/// Pass [`u32::MAX`] for `data_size` when the final length is not yet known.
pub(crate) fn write_data_header(dest: &mut impl Write, data_size: u32) -> std::io::Result<()> {
    write_chunk_header(dest, DATA, data_size)
}

/// Write an arbitrary named chunk: the 4-byte id, the 32-bit little-endian size,
/// the body, and a pad byte if the body length is odd. Returns the total number
/// of bytes written, including the header and any pad byte.
///
/// The caller must ensure `body.len()` fits in a `u32`.
pub(crate) fn write_named_chunk(
    dest: &mut impl Write,
    id: &[u8; 4],
    body: &[u8],
) -> std::io::Result<u64> {
    write_chunk_header(dest, id, body.len() as u32)?;
    dest.write_all(body)?;
    let mut written = 8 + body.len() as u64;
    if body.len() % 2 == 1 {
        dest.write_all(&[0])?;
        written += 1;
    }
    Ok(written)
}

/// Write a plain wav header: RIFF + WAVE, a `fmt ` chunk (16-byte core, or a
/// 40-byte `WAVEFORMATEXTENSIBLE` chunk for formats that need it), then the data
/// chunk header. No `fact` or metadata chunks are written.
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
    write_riff_wave(dest, riff_size)?;
    write_fmt_chunk(dest, channels, sample_format, sample_rate)?;
    write_data_header(dest, data_size)?;
    Ok(())
}
