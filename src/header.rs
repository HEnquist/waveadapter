//! Reading and writing wav headers.
//!
//! The chunk parsing and header layout are adapted from the wav handling in
//! CamillaDSP (<https://github.com/HEnquist/camilladsp>), generalized to this
//! crate's [`SampleFormat`] and error types.

use std::convert::TryInto;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem;

use crate::error::{Result, WavError};
use crate::format::{RawSpec, SampleFormat};

const RIFF: &[u8] = b"RIFF";
/// The RF64 form id, used in place of `RIFF` for files that may exceed 4 GB.
const RF64: &[u8] = b"RF64";
/// The BW64 form id (ITU-R BS.2088), structurally identical to RF64.
const BW64: &[u8] = b"BW64";
const WAVE: &[u8] = b"WAVE";
const DATA: &[u8] = b"data";
const FMT: &[u8] = b"fmt ";
/// The `ds64` chunk that carries the 64-bit sizes of an RF64/BW64 file.
const DS64: &[u8] = b"ds64";

/// Byte offset of the 32-bit RIFF chunk size field, measured from the start of the file.
pub(crate) const RIFF_SIZE_OFFSET: u64 = 4;

/// The marker written into a 32-bit size field when the real size lives in the
/// `ds64` chunk (RF64), and also the streaming "length unknown" placeholder for
/// plain RIFF. The two uses are told apart by the file's form id.
const SIZE_IN_DS64: u32 = u32::MAX;

/// Body size of a `ds64` chunk with no oversized-chunk table: three 64-bit sizes
/// plus the 32-bit table length.
pub(crate) const DS64_BODY_SIZE: u32 = 28;

/// Whether a header is written as `WAVE_FORMAT_EXTENSIBLE`.
///
/// Three cases force the extensible form:
///
/// * 24-bit-in-4-byte data is ambiguous as plain PCM (the block alignment
///   implies a 4-byte/32-bit sample, but only 24 bits are meaningful), so we
///   write it the strict-spec way: the 32-bit container size in `wBitsPerSample`
///   and the real 24 bits in `wValidBitsPerSample`.
/// * More than two channels: the spec recommends extensible (with a channel
///   mask) once the layout is no longer plain mono/stereo.
/// * A non-zero channel mask: it can only be stored in the extensible form.
///
/// Anything else is unambiguous as plain PCM or float and uses the minimal
/// 16-byte `fmt ` chunk.
pub(crate) fn writes_as_extensible(
    channels: usize,
    format: SampleFormat,
    channel_mask: Option<u32>,
) -> bool {
    matches!(format, SampleFormat::I24_4)
        || channels > 2
        || matches!(channel_mask, Some(mask) if mask != 0)
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
    /// The binary sample format of the audio data, if it is one this crate can
    /// interpret. `None` means the `fmt ` chunk described a valid but
    /// unsupported format (for example 8-bit PCM or A-law); the audio can still
    /// be read as raw bytes via
    /// [`WavReader::read_raw_interleaved`](crate::WavReader::read_raw_interleaved),
    /// using the raw `fmt ` fields below to make sense of it. The float read path
    /// is unavailable in that case.
    pub sample_format: Option<SampleFormat>,
    /// The `fmt ` format code (`wFormatTag`) as stored in the file. For
    /// `WAVEFORMATEXTENSIBLE` files this is `0xFFFE`.
    pub format_code: u16,
    /// Bits per single-channel sample (`wBitsPerSample`) as stored in the file.
    pub bits_per_sample: u16,
    /// Bytes per frame (`nBlockAlign`) as stored in the file. This is the source
    /// of truth for framing raw byte reads when `sample_format` is `None`.
    pub block_align: u16,
    /// The sample rate in Hz.
    pub sample_rate: usize,
    /// The number of channels.
    pub channels: usize,
    /// The speaker-position channel mask (`dwChannelMask`) read from a
    /// `WAVEFORMATEXTENSIBLE` header, or `None` if the file uses a plain
    /// `WAVEFORMAT`/`WAVEFORMATEX` header that carries no mask. A value of
    /// `Some(0)` means the extensible header was present but left the layout
    /// unspecified. This crate stores the mask but does not interpret it.
    pub channel_mask: Option<u32>,
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

impl WavParams {
    /// The number of bytes per frame (one sample for each channel).
    ///
    /// For an interpreted format this is the channel count times the format's
    /// byte width; for a raw/unsupported format (`sample_format` is `None`) it is
    /// the `nBlockAlign` field read from the file.
    pub fn frame_bytes(&self) -> usize {
        match self.sample_format {
            Some(format) => self.channels * format.bytes_per_sample(),
            None => self.block_align as usize,
        }
    }
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

fn read_u64(buffer: &[u8], start_index: usize) -> u64 {
    u64::from_le_bytes(
        buffer[start_index..start_index + mem::size_of::<u64>()]
            .try_into()
            .unwrap_or_default(),
    )
}

/// The 64-bit sizes carried by an RF64/BW64 `ds64` chunk.
///
/// The dedicated `riff_size`/`data_size` fields override the `0xFFFFFFFF`
/// markers in the RIFF and `data` 32-bit size fields; any other chunk whose
/// 32-bit size is `0xFFFFFFFF` is looked up by id in `table`.
struct Ds64 {
    data_size: u64,
    table: Vec<([u8; 4], u64)>,
}

impl Ds64 {
    /// Parse a `ds64` chunk body. Missing or short bodies yield zeroed sizes
    /// rather than erroring, matching the lenient handling elsewhere in the parser.
    fn parse(body: &[u8]) -> Self {
        // riffSize (0..8) is recomputed from the file, so it is not retained.
        let data_size = if body.len() >= 16 {
            read_u64(body, 8)
        } else {
            0
        };
        // sampleCount (16..24) is surfaced via the fact chunk path, not here.
        let table_length = if body.len() >= 28 {
            read_u32(body, 24) as usize
        } else {
            0
        };
        let mut table = Vec::new();
        let mut offset = 28;
        for _ in 0..table_length {
            if offset + 12 > body.len() {
                break;
            }
            let mut id = [0u8; 4];
            id.copy_from_slice(&body[offset..offset + 4]);
            table.push((id, read_u64(body, offset + 4)));
            offset += 12;
        }
        Ds64 { data_size, table }
    }

    /// The real body length of a chunk whose 32-bit size field is the
    /// `0xFFFFFFFF` marker: the `data` chunk uses the dedicated field, anything
    /// else is looked up by id in the table (falling back to the marker value).
    fn size_for(&self, id: &[u8], declared: u32) -> u64 {
        if declared != SIZE_IN_DS64 {
            return declared as u64;
        }
        if compare_4cc(id, DATA) {
            return self.data_size;
        }
        self.table
            .iter()
            .find(|(tid, _)| compare_4cc(id, tid))
            .map(|(_, size)| *size)
            .unwrap_or(declared as u64)
    }
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
        channel_mask: Option<u32>,
    ) -> Result<Self> {
        if channels == 0 {
            return Err(WavError::InvalidSpec(
                "channel count must be at least 1".to_string(),
            ));
        }
        // The mask is stored, not interpreted: the only rule we enforce is that a
        // non-zero mask assigns exactly one speaker position per channel.
        if let Some(mask) = channel_mask
            && mask != 0
            && mask.count_ones() as usize != channels
        {
            return Err(WavError::InvalidSpec(format!(
                "channel mask {mask:#x} has {} bits set but there are {channels} channels",
                mask.count_ones()
            )));
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
        if writes_as_extensible(channels, sample_format, channel_mask) {
            // Strict-spec extensible: wBitsPerSample carries the container size
            // (bytes per sample * 8), and the real depth goes in validBits. The
            // subformat GUID mirrors the plain format code (PCM vs IEEE float).
            // The channel mask is the caller's value, or 0 ("no assignment") when
            // none was given.
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
                    channel_mask: channel_mask.unwrap_or(0),
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

    /// Build a plain 16-byte core chunk from raw `fmt ` fields, without
    /// interpreting them as a [`SampleFormat`]. Used by the raw writer.
    ///
    /// Returns an error if the channel count or sample rate cannot be
    /// represented in the header fields.
    fn for_raw(spec: &RawSpec) -> Result<Self> {
        if spec.channels == 0 {
            return Err(WavError::InvalidSpec(
                "channel count must be at least 1".to_string(),
            ));
        }
        let channels = u16::try_from(spec.channels).map_err(|_| {
            WavError::InvalidSpec(format!(
                "channel count {} does not fit in 16 bits",
                spec.channels
            ))
        })?;
        let sample_rate = u32::try_from(spec.sample_rate).map_err(|_| {
            WavError::InvalidSpec(format!(
                "sample rate {} does not fit in 32 bits",
                spec.sample_rate
            ))
        })?;
        let byte_rate = (spec.block_align as u32)
            .checked_mul(sample_rate)
            .ok_or_else(|| {
                WavError::InvalidSpec("bytes per second does not fit in 32 bits".to_string())
            })?;
        Ok(FmtChunk {
            format_code: spec.format_code,
            channels,
            sample_rate,
            byte_rate,
            block_align: spec.block_align,
            bits_per_sample: spec.bits_per_sample,
            extension: None,
        })
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

    // The file must start with RIFF (plain wav) or RF64/BW64 (64-bit form), and
    // bytes 8..12 must be WAVE. RF64 and BW64 share the RIFF layout but move the
    // real sizes into a leading `ds64` chunk.
    let is_rf64 = compare_4cc(&header, RF64) || compare_4cc(&header, BW64);
    if (!compare_4cc(&header, RIFF) && !is_rf64) || !compare_4cc(&header[8..], WAVE) {
        return Err(WavError::InvalidHeader(
            "missing RIFF/RF64/BW64 or WAVE marker".to_string(),
        ));
    }

    let mut next_chunk_location = 12;
    let mut found_fmt = false;
    let mut found_data = false;
    let mut buffer = [0; 8];

    // The 64-bit sizes for an RF64/BW64 file, filled in when the `ds64` chunk is
    // reached (it is required to come first). Stays zeroed for plain RIFF.
    let mut ds64 = Ds64 {
        data_size: 0,
        table: Vec::new(),
    };

    // Dummy values until the real ones are found.
    let mut sample_format = None;
    let mut format_code = 0u16;
    let mut bits_per_sample = 0u16;
    let mut block_align = 0u16;
    let mut sample_rate = 0;
    let mut channels = 0;
    let mut channel_mask = None;
    let mut data_offset = 0;
    let mut data_length: u64 = 0;
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
        let is_ds64 = is_rf64 && compare_4cc(&buffer, DS64);
        // The real body length: for RF64 a `0xFFFFFFFF` size is resolved through
        // the ds64 chunk, otherwise the 32-bit field is taken at face value.
        let body_len = if is_rf64 {
            ds64.size_for(&buffer[0..4], chunk_length)
        } else {
            chunk_length as u64
        };
        if is_ds64 {
            // The ds64 chunk is container metadata, not exposed as a raw chunk.
            // Honor the first one and parse its 64-bit sizes for later chunks.
            let body_end = next_chunk_location + 8 + chunk_length as u64;
            if body_end <= filesize {
                let mut body = vec![0; chunk_length as usize];
                file.read_exact(&mut body)?;
                ds64 = Ds64::parse(&body);
            }
            next_chunk_location += 8 + chunk_length as u64 + (chunk_length as u64 & 1);
            continue;
        }
        if is_fmt {
            // Honor the first valid fmt chunk; ignore any later or malformed one.
            if !found_fmt && (chunk_length == 16 || chunk_length == 18 || chunk_length == 40) {
                found_fmt = true;
                let mut data = vec![0; chunk_length as usize];
                file.read_exact(&mut data)?;
                let fmt = FmtChunk::parse(&data);
                channels = fmt.channels;
                sample_rate = fmt.sample_rate;
                format_code = fmt.format_code;
                bits_per_sample = fmt.bits_per_sample;
                block_align = fmt.block_align;
                // The channel mask lives only in the 40-byte extensible form, at
                // offset 20 (after cbSize and wValidBitsPerSample).
                if chunk_length == 40 {
                    channel_mask = Some(read_u32(&data, 20));
                }
                let bytes_per_sample = fmt
                    .bytes_per_sample()
                    .ok_or_else(|| WavError::InvalidHeader("zero channels".to_string()))?;
                // A valid but unsupported format (no matching audioadapter sample
                // type, e.g. 8-bit PCM) is not an error here: it is recorded as
                // `None` so the file can still be read as raw bytes. A genuinely
                // malformed fmt chunk still errors.
                sample_format = match look_up_format(
                    &data,
                    fmt.format_code,
                    fmt.bits_per_sample,
                    bytes_per_sample,
                    chunk_length,
                ) {
                    Ok(format) => Some(format),
                    Err(WavError::UnsupportedFormat(_)) => None,
                    Err(other) => return Err(other),
                };
            }
        } else if is_data {
            // Honor the first data chunk; ignore any later one.
            if !found_data {
                found_data = true;
                data_offset = next_chunk_location + 8;
                data_length = body_len;
                // For plain RIFF a `0xFFFFFFFF` length is the streaming
                // placeholder, meaning the data runs to the end of the file, so
                // there is nothing to scan past it. For RF64 the same field was
                // already resolved through ds64 into a real length.
                if !is_rf64 && chunk_length == u32::MAX {
                    break;
                }
            }
        } else {
            // Any other chunk is captured verbatim, tolerating a bogus length
            // that would overrun the file by stopping the scan instead of erroring.
            let body_end = next_chunk_location + 8 + body_len;
            if body_end > filesize {
                break;
            }
            let read_len = usize::try_from(body_len).map_err(|_| {
                WavError::InvalidHeader("chunk length does not fit in memory".to_string())
            })?;
            let mut body = vec![0; read_len];
            file.read_exact(&mut body)?;
            let mut id = [0u8; 4];
            id.copy_from_slice(&buffer[0..4]);
            chunks.push(Chunk { id, data: body });
        }
        next_chunk_location += 8 + body_len + (body_len & 1);
    }
    if found_data && found_fmt {
        return Ok(WavParams {
            sample_format,
            format_code,
            bits_per_sample,
            block_align,
            sample_rate: sample_rate as usize,
            channels: channels as usize,
            channel_mask,
            data_length: usize::try_from(data_length).map_err(|_| {
                WavError::InvalidHeader("data length does not fit in memory".to_string())
            })?,
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

/// Write the RF64 chunk header and the WAVE form type (12 bytes).
///
/// The 32-bit RIFF size field is always the `0xFFFFFFFF` marker for RF64; the
/// real size lives in the following `ds64` chunk.
pub(crate) fn write_rf64_wave(dest: &mut impl Write) -> std::io::Result<()> {
    write_chunk_header(dest, RF64, SIZE_IN_DS64)?;
    dest.write_all(WAVE)
}

/// Write a `ds64` chunk with zeroed 64-bit size fields and no oversized-chunk
/// table (28-byte body). The `riffSize`, `dataSize` and `sampleCount` fields are
/// patched with the real values on finalize.
pub(crate) fn write_ds64_chunk(dest: &mut impl Write) -> std::io::Result<()> {
    write_chunk_header(dest, DS64, DS64_BODY_SIZE)?;
    dest.write_all(&0u64.to_le_bytes())?; // riffSize
    dest.write_all(&0u64.to_le_bytes())?; // dataSize
    dest.write_all(&0u64.to_le_bytes())?; // sampleCount
    dest.write_all(&0u32.to_le_bytes()) // tableLength
}

/// The `0xFFFFFFFF` marker written into the `data` chunk's 32-bit size field in
/// an RF64 file, where the real size lives in the `ds64` chunk.
pub(crate) const RF64_DATA_SIZE_MARKER: u32 = SIZE_IN_DS64;

/// Write the `fmt ` chunk (id, size and body) and return its body size in bytes.
///
/// Returns [`WavError::InvalidSpec`] if the channel count or sample rate cannot
/// be represented in the header fields.
pub(crate) fn write_fmt_chunk(
    dest: &mut impl Write,
    channels: usize,
    sample_format: SampleFormat,
    sample_rate: usize,
    channel_mask: Option<u32>,
) -> Result<u32> {
    let fmt = FmtChunk::for_format(channels, sample_format, sample_rate, channel_mask)?;
    let body_size = fmt.body_size();
    write_chunk_header(dest, FMT, body_size)?;
    fmt.write_body(dest)?;
    Ok(body_size)
}

/// Write a `fmt ` chunk from raw, uninterpreted fields (a 16-byte core chunk)
/// and return its body size in bytes.
///
/// Returns [`WavError::InvalidSpec`] if the channel count or sample rate cannot
/// be represented in the header fields.
pub(crate) fn write_fmt_chunk_raw(dest: &mut impl Write, spec: &RawSpec) -> Result<u32> {
    let fmt = FmtChunk::for_raw(spec)?;
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
    write_fmt_chunk(dest, channels, sample_format, sample_rate, None)?;
    write_data_header(dest, data_size)?;
    Ok(())
}
