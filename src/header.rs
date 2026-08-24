//! Reading and writing wav headers.
//!
//! The chunk parsing and header layout are adapted from the wav handling in
//! CamillaDSP (<https://github.com/HEnquist/camilladsp>), generalized to this
//! crate's [`SampleFormat`] and error types.

use std::convert::TryInto;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem;

use crate::error::{Result, WavError};
use crate::format::{SampleFormat, WavSpec};

const RIFF: &[u8] = b"RIFF";
/// The RF64 form id, used in place of `RIFF` for files that may exceed 4 GB.
const RF64: &[u8] = b"RF64";
/// The BW64 form id (ITU-R BS.2088), structurally identical to RF64.
const BW64: &[u8] = b"BW64";
const WAVE: &[u8] = b"WAVE";
const DATA: &[u8] = b"data";
const FMT: &[u8; 4] = b"fmt ";
/// The `ds64` chunk that carries the 64-bit sizes of an RF64/BW64 file.
const DS64: &[u8] = b"ds64";
/// The `fact` chunk, carrying the sample-frame count for non-PCM formats.
const FACT: &[u8; 4] = b"fact";

/// The `wFormatTag` value that marks a `WAVEFORMATEXTENSIBLE` header, where the
/// real format is named by the subformat GUID in the extension instead.
const EXTENSIBLE_FORMAT_CODE: u16 = 0xFFFE;

/// Byte offset of the 32-bit RIFF chunk size field, measured from the start of the file.
pub(crate) const RIFF_SIZE_OFFSET: u64 = 4;

/// The marker written into a 32-bit size field when the real size lives in the
/// `ds64` chunk (RF64), and also the "length unknown" placeholder for plain
/// RIFF. The two uses are told apart by the file's form id.
const SIZE_IN_DS64: u32 = u32::MAX;

/// The value a plain RIFF size field carries while the real length is not yet
/// known: the data runs to the end of the file. Written by both the streaming
/// and the seekable writer, the latter patching it in `finalize`.
pub(crate) const UNKNOWN_SIZE: u32 = SIZE_IN_DS64;

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

/// `KSDATAFORMAT_SUBTYPE_ALAW`
const SUBTYPE_ALAW: Guid = Guid {
    data1: 6,
    data2: 0,
    data3: 16,
    data4: [128, 0, 0, 170, 0, 56, 155, 113],
};

/// `KSDATAFORMAT_SUBTYPE_MULAW`
const SUBTYPE_MULAW: Guid = Guid {
    data1: 7,
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
    /// The `fmt ` chunk exactly as the file carries it, including any
    /// format-specific extension this crate does not interpret.
    ///
    /// This is the source of truth for everything about the format. The
    /// accessors below are the interpreted view of it; the individual fields are
    /// reachable directly as `params.fmt.block_align` and so on. Handing this
    /// back to the writer reproduces the chunk byte for byte, which is what lets
    /// a caller round-trip a format waveadapter does not model.
    pub fmt: FmtChunk,
    /// The body of the `fact` chunk, verbatim, or `None` if the file has none.
    ///
    /// The spec requires one for every non-PCM format, where it carries the
    /// sample-frame count that the data size cannot give (a block-compressed
    /// file's data size is a count of blocks). Its first four bytes are that
    /// count, reachable through [`fact_samples`](WavParams::fact_samples); the
    /// rest, if a format defines any, is kept as-is.
    ///
    /// Like [`fmt`](WavParams::fmt) this is *not* repeated in
    /// [`chunks`](WavParams::chunks): the writer produces `fact` itself, so a
    /// caller passing `chunks` straight back would otherwise be handing over a
    /// chunk the writer is about to write again.
    pub fact: Option<Vec<u8>>,
    /// The sample-frame count from the `ds64` chunk of an RF64/BW64 file, which
    /// is where that form keeps what `fact` carries in a plain RIFF file.
    /// `None` for plain RIFF. Prefer [`sample_count`](WavParams::sample_count),
    /// which reads whichever of the two the file actually has.
    pub ds64_sample_count: Option<u64>,
    /// Byte offset from the start of the file to the first audio sample.
    pub data_offset: u64,
    /// The length of the audio data in bytes, as declared in the header.
    ///
    /// Files written in streaming mode declare this as [`u32::MAX`] because the
    /// final length is not known up front, so do not rely on it to be accurate.
    pub data_length: u64,
    /// The non-audio chunks stored *before* the audio, in the order encountered.
    ///
    /// These go back through a leading-chunk constructor such as
    /// [`WavWriter::new_with_chunks`](crate::WavWriter::new_with_chunks). See
    /// [`Chunk`].
    pub chunks_before: Vec<Chunk>,
    /// The non-audio chunks stored *after* the audio, in the order encountered.
    ///
    /// These go back through [`WavWriter::write_chunk`](crate::WavWriter::write_chunk),
    /// once the audio is written. Keeping the two sides apart is what stops a
    /// rewrite from silently moving a trailing chunk to the front of the file,
    /// which matters for the ones whose position is conventional: `cue `/`adtl`
    /// pairs and `id3 ` are usually written after the audio.
    pub chunks_after: Vec<Chunk>,
}

impl WavParams {
    /// The binary sample format of the audio data, if it is one this crate can
    /// interpret.
    ///
    /// `None` means the `fmt ` chunk described a valid but unsupported format
    /// (for example ADPCM or GSM); the audio can still be read as raw bytes via
    /// [`WavReader::read_raw_interleaved`](crate::WavReader::read_raw_interleaved),
    /// using [`fmt`](WavParams::fmt) to make sense of it. The float read path is
    /// unavailable in that case.
    pub fn sample_format(&self) -> Option<SampleFormat> {
        self.fmt.sample_format()
    }

    /// The number of channels.
    pub fn channels(&self) -> usize {
        self.fmt.channels as usize
    }

    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> usize {
        self.fmt.sample_rate as usize
    }

    /// The speaker-position channel mask (`dwChannelMask`), or `None` if the
    /// header is not `WAVEFORMATEXTENSIBLE` and so carries no mask. `Some(0)`
    /// means the extensible header was present but left the layout unspecified.
    /// This crate stores the mask but does not interpret it.
    pub fn channel_mask(&self) -> Option<u32> {
        self.fmt.channel_mask()
    }

    /// Every non-audio chunk, before and after the audio, in file order.
    ///
    /// Use this when the position does not matter. When re-writing a file it
    /// does, so keep [`chunks_before`](WavParams::chunks_before) and
    /// [`chunks_after`](WavParams::chunks_after) apart.
    pub fn chunks(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks_before.iter().chain(&self.chunks_after)
    }

    /// The sample-frame count declared in the `fact` chunk, if the file has one.
    ///
    /// This is the field as the file stores it, 32 bits and all. To preserve a
    /// count on rewrite, reach for [`sample_count`](WavParams::sample_count)
    /// instead and pass that through
    /// [`Fact::Samples`](crate::Fact::Samples): it is the same number for a
    /// RIFF file and the right one for an RF64 file, which keeps no `fact`
    /// chunk. Either way it beats [`WavReader::frames`](crate::WavReader::frames)
    /// for a block-compressed format, where that counts compressed blocks.
    pub fn fact_samples(&self) -> Option<u32> {
        let body = self.fact.as_deref()?;
        (body.len() >= 4).then(|| read_u32(body, 0))
    }

    /// The declared sample-frame count, from wherever this container keeps it:
    /// the `ds64` chunk for RF64/BW64, the `fact` chunk for RIFF.
    ///
    /// The two forms carry the same number in different places, so this is the
    /// one to reach for when the container form is not the point. A file that
    /// carries both (an RF64 file with a legacy `fact` chunk) is read from
    /// `ds64`, the only one of the two that is 64-bit and so the only one an
    /// RF64-sized file can state its count in. The exception is a `ds64` count
    /// left at zero, which a writer that fills in only the sizes produces: a
    /// `fact` chunk is the better source than a field nobody filled in.
    pub fn sample_count(&self) -> Option<u64> {
        self.ds64_sample_count
            .filter(|&count| count != 0)
            .or_else(|| self.fact_samples().map(u64::from))
            .or(self.ds64_sample_count)
    }

    /// The spec that would reproduce this file's format through the typed write
    /// path, or `None` if the format is not one this crate models.
    ///
    /// This is the way to re-write a file with one property changed while
    /// keeping the rest: `WavSpec { sample_rate: 48000, ..params.spec()? }`.
    /// Editing [`fmt`](WavParams::fmt) directly is not the way, because changing
    /// a field there leaves `byte_rate` stale and no one can recompute it for a
    /// compressed format.
    pub fn spec(&self) -> Option<WavSpec> {
        Some(WavSpec {
            channels: self.channels(),
            sample_rate: self.sample_rate(),
            sample_format: self.sample_format()?,
            channel_mask: self.channel_mask(),
        })
    }

    /// The number of bytes per frame (one sample for each channel).
    ///
    /// For an interpreted format this is the channel count times the format's
    /// byte width; for a raw/unsupported format (`sample_format` is `None`) it is
    /// the `nBlockAlign` field read from the file.
    ///
    /// For an uninterpreted format this is a byte-framing convenience, not
    /// necessarily one audio frame. `nBlockAlign` means different things across the
    /// compressed format tags: A-law and mu-law declare a real frame, ADPCM and GSM
    /// declare a whole compressed block spanning many audio frames, and MPEG Layer 3
    /// (`0x0055`) declares 1, which makes the unit a single byte. Anything derived
    /// from this value, such as [`WavReader::frames`](crate::WavReader::frames) and
    /// [`WavReader::seek_to_frame`](crate::WavReader::seek_to_frame), inherits that
    /// meaning.
    pub fn frame_bytes(&self) -> usize {
        self.fmt.frame_bytes()
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
    sample_count: u64,
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
        // sampleCount is what `fact` carries in a plain RIFF file, so it is
        // surfaced the same way (see `WavParams::sample_count`).
        let sample_count = if body.len() >= 24 {
            read_u64(body, 16)
        } else {
            0
        };
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
        Ds64 {
            data_size,
            sample_count,
            table,
        }
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

/// The `fmt ` chunk: the six core fields every wav file has, plus the
/// format-specific extension that follows them.
///
/// This is the typed view of the chunk, in the same shape as the metadata types
/// in [`crate::metadata`]: [`from_bytes`](FmtChunk::from_bytes) /
/// [`to_bytes`](FmtChunk::to_bytes) to decode and encode, with the bytes staying
/// the source of truth. The reader hands one back on
/// [`WavParams::fmt`], and the writer takes one, so a `fmt ` chunk this crate
/// does not model survives a read/write cycle unchanged. That is what makes it
/// possible to build a codec for a format waveadapter has no
/// [`SampleFormat`] for.
///
/// # The extension
///
/// The wav spec grew the chunk in three steps, and [`extension`](FmtChunk::extension)
/// is how the three are told apart:
///
/// | `extension` | Chunk body | Form |
/// | --- | --- | --- |
/// | `None` | 16 bytes | `WAVEFORMAT`, valid only for integer PCM |
/// | `Some(&[])` | 18 bytes | `WAVEFORMATEX`, a `cbSize` of 0, required for non-PCM |
/// | `Some(bytes)` | 18 + n | `cbSize` is `bytes.len()`, contents defined by the format |
///
/// `cbSize` is *derived, never stored*: it is recomputed from the extension
/// length on write, and a stored value that disagrees with the real body length
/// is ignored on read. Encoders do write 40-byte chunks with a `cbSize` of 0,
/// and honoring that would silently drop the subformat GUID. This is the one
/// place a round trip is knowingly not byte-exact: such a chunk comes back with
/// its `cbSize` corrected.
///
/// # Trust
///
/// The fields are what the file says, not what they ought to be. `byte_rate` is
/// not assumed to be `block_align * sample_rate` (for GSM 6.10 it is not),
/// `block_align` is not assumed to be `channels * bits_per_sample / 8`, and
/// `bits_per_sample` of 0 is legal (GSM again). Nothing is normalized, because
/// the codec owns these numbers and the container does not get to second-guess
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FmtChunk {
    /// The format tag (`wFormatTag`): `1` for integer PCM, `3` for IEEE float,
    /// `6`/`7` for G.711 A-law/mu-law, `0xFFFE` for `WAVEFORMATEXTENSIBLE`, and
    /// a few hundred others this crate does not model.
    pub format_code: u16,
    /// The number of channels (`nChannels`).
    pub channels: u16,
    /// The sample rate in Hz (`nSamplesPerSec`).
    pub sample_rate: u32,
    /// Bytes per second (`nAvgBytesPerSec`), as declared. For a compressed
    /// format this is not derivable from the other fields, which is why it is
    /// carried rather than computed.
    pub byte_rate: u32,
    /// Bytes per frame (`nBlockAlign`). For a block-compressed format this is
    /// the size of a whole compressed block, not of one sample frame.
    pub block_align: u16,
    /// Bits per single-channel sample (`wBitsPerSample`). Zero is legal and
    /// means the format does not have a meaningful per-sample bit depth.
    pub bits_per_sample: u16,
    /// The bytes after `cbSize`, or `None` for the bare 16-byte form. See the
    /// table above.
    pub extension: Option<Vec<u8>>,
}

impl FmtChunk {
    /// Size of the 16-byte core `fmt ` chunk body.
    pub(crate) const CORE_SIZE: u32 = 16;
    /// Size of the 18-byte `WAVEFORMATEX` `fmt ` chunk body.
    pub(crate) const EX_SIZE: u32 = 18;
    /// Length of the `WAVEFORMATEXTENSIBLE` extension: valid bits, channel mask
    /// and the 16-byte subformat GUID.
    pub(crate) const EXTENSIBLE_EXTENSION_LEN: usize = 22;

    /// Build the chunk describing a [`WavSpec`], the way this crate writes it.
    ///
    /// Returns an error if the parameters cannot be represented in the header
    /// fields, rather than silently truncating an out-of-range value.
    pub fn for_spec(spec: &WavSpec) -> Result<Self> {
        Self::for_format(
            spec.channels,
            spec.sample_format,
            spec.sample_rate,
            spec.channel_mask,
        )
    }

    /// Check the fields that make a file unreadable rather than merely unusual.
    ///
    /// A `FmtChunk` handed to the writer is taken as authoritative, so this is
    /// deliberately just the one rule the parser also enforces: a header with no
    /// channels is rejected on read, and the writer must not produce a file this
    /// crate cannot read back. Everything else, including a `nBlockAlign` of
    /// zero or a format code nobody has heard of, is the caller's business.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.channels == 0 {
            return Err(WavError::InvalidSpec(
                "channel count must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

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
            // subformat GUID mirrors the plain format code, so every code this
            // crate can write needs an arm here. The channel mask is the caller's
            // value, or 0 ("no assignment") when none was given.
            let sub_format = match sample_format.format_code() {
                3 => SUBTYPE_FLOAT,
                6 => SUBTYPE_ALAW,
                7 => SUBTYPE_MULAW,
                _ => SUBTYPE_PCM,
            };
            let mut extension = Vec::with_capacity(Self::EXTENSIBLE_EXTENSION_LEN);
            extension.extend_from_slice(&(sample_format.bits_per_sample() as u16).to_le_bytes());
            extension.extend_from_slice(&channel_mask.unwrap_or(0).to_le_bytes());
            extension.extend_from_slice(&sub_format.to_bytes());
            Ok(FmtChunk {
                format_code: EXTENSIBLE_FORMAT_CODE,
                channels: channels_u16,
                sample_rate: sample_rate_u32,
                byte_rate,
                block_align,
                bits_per_sample: (bytes_per_sample * 8) as u16,
                extension: Some(extension),
            })
        } else {
            Ok(FmtChunk {
                format_code: sample_format.format_code(),
                channels: channels_u16,
                sample_rate: sample_rate_u32,
                byte_rate,
                block_align,
                bits_per_sample: sample_format.bits_per_sample() as u16,
                // Only plain integer PCM may use the bare 16-byte form. Every
                // other format (float, A-law, mu-law) needs the `cbSize` field,
                // even though it is zero.
                extension: if sample_format.is_pcm() {
                    None
                } else {
                    Some(Vec::new())
                },
            })
        }
    }

    /// Decode a `fmt ` chunk body.
    ///
    /// Returns `None` if the body is shorter than the 16-byte core. Everything
    /// from byte 18 on becomes the [`extension`](FmtChunk::extension); the
    /// stored `cbSize` at bytes 16..18 is ignored, since the real body length is
    /// the trustworthy one.
    ///
    /// A 17-byte body is the one length that cannot be represented, holding half
    /// a `cbSize` field and nothing else. It decodes as the bare core and
    /// [`to_bytes`](FmtChunk::to_bytes) re-emits 16 bytes, so that single byte is
    /// dropped. That is deliberate on both counts: rejecting the chunk would fail
    /// a whole file this crate can otherwise read, and keeping the byte would
    /// mean a public field for a half-written `cbSize`, a value that is derived
    /// here and never stored. Nothing is lost but the length, which the rewrite
    /// changes from 17 to 16 either way.
    pub fn from_bytes(body: &[u8]) -> Option<Self> {
        if body.len() < Self::CORE_SIZE as usize {
            return None;
        }
        Some(FmtChunk {
            format_code: read_u16(body, 0),
            channels: read_u16(body, 2),
            sample_rate: read_u32(body, 4),
            byte_rate: read_u32(body, 8),
            block_align: read_u16(body, 12),
            bits_per_sample: read_u16(body, 14),
            // A 17-byte body has half a cbSize field and nothing after it, which
            // is not a form the spec defines; treat it as the bare core.
            extension: if body.len() >= Self::EX_SIZE as usize {
                Some(body[Self::EX_SIZE as usize..].to_vec())
            } else {
                None
            },
        })
    }

    /// Decode a `fmt ` chunk, checking the id first.
    pub fn from_chunk(chunk: &Chunk) -> Option<Self> {
        if &chunk.id != FMT {
            return None;
        }
        Self::from_bytes(&chunk.data)
    }

    /// Encode the chunk body, computing `cbSize` from the extension length.
    ///
    /// Returns [`WavError::InvalidSpec`] if the extension is too long to
    /// describe in the 16-bit `cbSize` field.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let extension = self.extension.as_deref();
        let cb_size = match extension {
            Some(bytes) => Some(u16::try_from(bytes.len()).map_err(|_| {
                WavError::InvalidSpec(format!(
                    "fmt extension of {} bytes does not fit in the 16-bit cbSize field",
                    bytes.len()
                ))
            })?),
            None => None,
        };
        let mut out = Vec::with_capacity(self.body_size() as usize);
        out.extend_from_slice(&self.format_code.to_le_bytes());
        out.extend_from_slice(&self.channels.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.byte_rate.to_le_bytes());
        out.extend_from_slice(&self.block_align.to_le_bytes());
        out.extend_from_slice(&self.bits_per_sample.to_le_bytes());
        if let (Some(cb_size), Some(bytes)) = (cb_size, extension) {
            out.extend_from_slice(&cb_size.to_le_bytes());
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }

    /// Encode as a [`Chunk`] with the `fmt ` id.
    pub fn to_chunk(&self) -> Result<Chunk> {
        Ok(Chunk {
            id: *b"fmt ",
            data: self.to_bytes()?,
        })
    }

    /// The size in bytes of this chunk's body when written.
    pub(crate) fn body_size(&self) -> u32 {
        match &self.extension {
            None => Self::CORE_SIZE,
            Some(bytes) => Self::EX_SIZE + bytes.len() as u32,
        }
    }

    /// Whether this is a `WAVEFORMATEXTENSIBLE` header, meaning the real format
    /// is named by [`sub_format`](FmtChunk::sub_format) rather than by
    /// [`format_code`](FmtChunk::format_code).
    ///
    /// The extension has to be long enough to hold the fields as well as the
    /// tag saying they are there; a truncated one is not extensible.
    pub fn is_extensible(&self) -> bool {
        self.format_code == EXTENSIBLE_FORMAT_CODE
            && self
                .extension
                .as_ref()
                .is_some_and(|ext| ext.len() >= Self::EXTENSIBLE_EXTENSION_LEN)
    }

    /// The real bit depth (`wValidBitsPerSample`), for an extensible header
    /// whose container is wider than its samples. `None` if not extensible.
    pub fn valid_bits_per_sample(&self) -> Option<u16> {
        self.extensible_extension().map(|ext| read_u16(ext, 0))
    }

    /// The speaker-position channel mask (`dwChannelMask`). `None` if the header
    /// is not extensible, since that is the only form that carries one.
    pub fn channel_mask(&self) -> Option<u32> {
        self.extensible_extension().map(|ext| read_u32(ext, 2))
    }

    /// The subformat GUID naming the real format. `None` if not extensible.
    pub fn sub_format(&self) -> Option<[u8; 16]> {
        self.extensible_extension()
            .map(|ext| ext[6..22].try_into().unwrap())
    }

    /// The extension bytes, but only when they are the `WAVEFORMATEXTENSIBLE`
    /// ones. Everything typed above reads through here, so a long `fmt ` chunk
    /// that merely happens to reach offset 20 (MS ADPCM keeps `wNumCoef` there)
    /// is never mistaken for an extensible one.
    fn extensible_extension(&self) -> Option<&[u8]> {
        if !self.is_extensible() {
            return None;
        }
        self.extension.as_deref()
    }

    /// The number of bytes per single-channel sample, derived from the block
    /// alignment and channel count.
    ///
    /// `None` when there is no whole number of them: zero channels, or a
    /// `nBlockAlign` that is not a multiple of the channel count. The second
    /// case is a malformed header, and rounding it down would name a format
    /// whose frames are narrower than the ones the file declares, leaving every
    /// read after the first frame misaligned. Uninterpreted is the honest
    /// answer, and puts the file on the raw path where `nBlockAlign` is used as
    /// stated.
    fn bytes_per_sample(&self) -> Option<u16> {
        if self.channels == 0 || !self.block_align.is_multiple_of(self.channels) {
            return None;
        }
        Some(self.block_align / self.channels)
    }

    /// The number of bytes one frame occupies.
    ///
    /// For a format this crate models that is the channel count times the sample
    /// width; otherwise it is `nBlockAlign` as declared, which for a
    /// block-compressed format is a whole compressed block rather than one
    /// sample frame.
    pub fn frame_bytes(&self) -> usize {
        match self.sample_format() {
            Some(format) => self.channels as usize * format.bytes_per_sample(),
            None => self.block_align as usize,
        }
    }

    /// The sample format this chunk describes, if it is one this crate can
    /// convert. `None` covers every valid-but-unmodeled format (ADPCM, GSM, an
    /// exotic extensible subtype), which is readable only through the raw path.
    pub fn sample_format(&self) -> Option<SampleFormat> {
        let bytes_per_sample = self.bytes_per_sample()?;
        let bits = self.bits_per_sample;
        match (self.format_code, bits, bytes_per_sample) {
            (1, 8, 1) => Some(SampleFormat::U8),
            (1, 16, 2) => Some(SampleFormat::I16),
            (1, 24, 3) => Some(SampleFormat::I24_3),
            (1, 24, 4) => Some(SampleFormat::I24_4),
            (1, 32, 4) => Some(SampleFormat::I32),
            (3, 32, 4) => Some(SampleFormat::F32),
            (3, 64, 8) => Some(SampleFormat::F64),
            (6, 8, 1) => Some(SampleFormat::ALAW),
            (7, 8, 1) => Some(SampleFormat::MULAW),
            (EXTENSIBLE_FORMAT_CODE, _, _) => self.extended_sample_format(bytes_per_sample),
            _ => None,
        }
    }

    fn extended_sample_format(&self, bytes_per_sample: u16) -> Option<SampleFormat> {
        let guid = Guid::from_slice(&self.sub_format()?);
        let valid_bits_per_sample = self.valid_bits_per_sample()?;
        look_up_extended_format(
            guid,
            self.bits_per_sample,
            bytes_per_sample,
            valid_bits_per_sample,
        )
    }
}

impl TryFrom<WavSpec> for FmtChunk {
    type Error = WavError;

    fn try_from(spec: WavSpec) -> Result<Self> {
        FmtChunk::for_spec(&spec)
    }
}

impl TryFrom<&WavSpec> for FmtChunk {
    type Error = WavError;

    fn try_from(spec: &WavSpec) -> Result<Self> {
        FmtChunk::for_spec(spec)
    }
}

fn look_up_extended_format(
    subformat_guid: Guid,
    bits: u16,
    bytes_per_sample: u16,
    valid_bits_per_sample: u16,
) -> Option<SampleFormat> {
    match (
        subformat_guid,
        bits,
        bytes_per_sample,
        valid_bits_per_sample,
    ) {
        (SUBTYPE_PCM, 8, 1, 8) => Some(SampleFormat::U8),
        (SUBTYPE_PCM, 16, 2, 16) => Some(SampleFormat::I16),
        (SUBTYPE_PCM, 24, 3, 24) => Some(SampleFormat::I24_3),
        // 24-in-4-byte: the lenient form (wBitsPerSample = 24) and the
        // strict-spec form (wBitsPerSample = container size 32, validBits = 24).
        (SUBTYPE_PCM, 24, 4, 24) => Some(SampleFormat::I24_4),
        (SUBTYPE_PCM, 32, 4, 24) => Some(SampleFormat::I24_4),
        (SUBTYPE_PCM, 32, 4, 32) => Some(SampleFormat::I32),
        (SUBTYPE_FLOAT, 32, 4, 32) => Some(SampleFormat::F32),
        (SUBTYPE_FLOAT, 64, 8, 64) => Some(SampleFormat::F64),
        (SUBTYPE_ALAW, 8, 1, 8) => Some(SampleFormat::ALAW),
        (SUBTYPE_MULAW, 8, 1, 8) => Some(SampleFormat::MULAW),
        _ => None,
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

    let mut next_chunk_location: u64 = 12;
    let mut found_fmt = false;
    let mut found_data = false;
    let mut buffer = [0; 8];

    // The 64-bit sizes for an RF64/BW64 file, filled in when the `ds64` chunk is
    // reached (it is required to come first). Stays zeroed for plain RIFF.
    let mut ds64 = Ds64 {
        data_size: 0,
        sample_count: 0,
        table: Vec::new(),
    };

    let mut fmt: Option<FmtChunk> = None;
    let mut fact: Option<Vec<u8>> = None;
    let mut ds64_sample_count: Option<u64> = None;
    let mut data_offset = 0;
    let mut data_length: u64 = 0;
    let mut chunks_before: Vec<Chunk> = Vec::new();
    let mut chunks_after: Vec<Chunk> = Vec::new();

    // Walk every chunk to the end of the file, so that metadata chunks placed
    // after the data chunk are captured too. A chunk is padded to an even length
    // with a trailing byte that is not counted in its declared size.
    //
    // Every offset computed from a declared length uses saturating arithmetic.
    // For RF64 the length comes from the ds64 table as an unvalidated 64-bit
    // value, so a hostile or corrupt file can name a size near `u64::MAX`.
    // Saturating turns that into an offset past `filesize`, which the existing
    // overrun checks already treat as "stop scanning".
    while next_chunk_location.saturating_add(8) <= filesize {
        file.seek(SeekFrom::Start(next_chunk_location))?;
        file.read_exact(&mut buffer)?;
        let chunk_length = read_u32(&buffer, 4);
        let is_data = compare_4cc(&buffer, DATA);
        let is_fmt = compare_4cc(&buffer, FMT);
        let is_fact = compare_4cc(&buffer, FACT);
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
            // Honor the first one and parse its 64-bit sizes for later chunks. A
            // second one is dropped, like a second `fmt `, `data` or `fact`, and
            // for a sharper reason: its sizes frame every chunk after it, so
            // letting it through would let a duplicate re-point the audio.
            let body_end = next_chunk_location.saturating_add(8 + chunk_length as u64);
            if ds64_sample_count.is_none() && body_end <= filesize {
                let mut body = vec![0; chunk_length as usize];
                file.read_exact(&mut body)?;
                ds64 = Ds64::parse(&body);
                ds64_sample_count = Some(ds64.sample_count);
            }
            next_chunk_location = next_chunk_location
                .saturating_add(8 + chunk_length as u64 + (chunk_length as u64 & 1));
            continue;
        }
        let mut consumed = false;
        if is_fact && fact.is_none() {
            // The writer emits `fact` itself, so the reader owns it too rather
            // than letting it through as an opaque chunk a caller could hand
            // back and have written twice. A second one is dropped, like a
            // second `fmt ` or `data`.
            let body_end = next_chunk_location
                .saturating_add(8)
                .saturating_add(body_len);
            if body_end <= filesize {
                let read_len = usize::try_from(body_len).map_err(|_| {
                    WavError::InvalidHeader("fact chunk length does not fit in memory".to_string())
                })?;
                let mut body = vec![0; read_len];
                file.read_exact(&mut body)?;
                fact = Some(body);
                consumed = true;
            }
        }
        if !consumed && is_fmt && !found_fmt {
            // Honor the first valid fmt chunk. Anything from the 16-byte core
            // upwards counts: the three standard sizes are 16, 18 and 40, but a
            // format may carry any amount of format-specific data after `cbSize`
            // (IMA ADPCM two bytes, MS ADPCM a coefficient table, GSM two), and
            // those files have to reach the raw path with their bytes intact.
            //
            // The body is bounded the same way the catch-all branch bounds its
            // chunks: it has to fit inside the file. That is what stops a
            // declared 4 GB fmt chunk from being allocated, now that the read is
            // no longer capped at 40 bytes.
            let body_end = next_chunk_location
                .saturating_add(8)
                .saturating_add(body_len);
            if chunk_length >= FmtChunk::CORE_SIZE && body_end <= filesize {
                let read_len = usize::try_from(body_len).map_err(|_| {
                    WavError::InvalidHeader("fmt chunk length does not fit in memory".to_string())
                })?;
                let mut body = vec![0; read_len];
                file.read_exact(&mut body)?;
                let parsed = FmtChunk::from_bytes(&body).ok_or_else(|| {
                    WavError::InvalidHeader("fmt chunk is shorter than 16 bytes".to_string())
                })?;
                // A zero channel count would make the frame size zero and every
                // frame-based offset meaningless, so it is the one core field
                // worth rejecting outright.
                if parsed.channels == 0 {
                    return Err(WavError::InvalidHeader("zero channels".to_string()));
                }
                found_fmt = true;
                consumed = true;
                fmt = Some(parsed);
            }
        }
        if consumed || is_fmt || is_fact {
            // Either read above, or a duplicate of a chunk this crate owns.
            //
            // A second `fmt `, `data` or `fact` chunk is dropped rather than
            // captured. Capturing would be worse than useless: those ids are
            // reserved on the write side precisely because the writer produces
            // them, so a caller feeding the chunks back would be handed
            // something the writer then refuses. For `data` there is the
            // additional problem that its body is audio, and reading a second
            // one into memory to hand back is unbounded.
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
            let body_end = next_chunk_location
                .saturating_add(8)
                .saturating_add(body_len);
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
            // Which side of the audio a chunk sits on is part of the file's
            // shape, so it is recorded rather than flattened away.
            let target = if found_data {
                &mut chunks_after
            } else {
                &mut chunks_before
            };
            target.push(Chunk { id, data: body });
        }
        next_chunk_location = next_chunk_location
            .saturating_add(8)
            .saturating_add(body_len)
            .saturating_add(body_len & 1);
    }
    if let (true, Some(fmt)) = (found_data, fmt) {
        return Ok(WavParams {
            fmt,
            fact,
            ds64_sample_count,
            data_length,
            data_offset,
            chunks_before,
            chunks_after,
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

/// Write the `fmt ` chunk (id, size and body), returning the total number of
/// bytes written including the 8-byte header and any pad byte.
///
/// This goes through [`write_named_chunk`] rather than emitting the body
/// directly, because a caller-supplied extension can be an odd number of bytes
/// and RIFF requires the pad. Without it every offset after the header, the
/// `data` size field included, would be one byte out.
pub(crate) fn write_fmt_chunk(dest: &mut impl Write, fmt: &FmtChunk) -> Result<u64> {
    Ok(write_named_chunk(dest, FMT, &fmt.to_bytes()?)?)
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
