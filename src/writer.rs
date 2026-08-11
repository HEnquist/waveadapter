//! Writing audio data to a wav file.

use std::io::{self, BufWriter, Cursor, Seek, SeekFrom, Write};

use audioadapter::Adapter;
use audioadapter_sample::readwrite::WriteSamples;
use num_traits::ToPrimitive;
use num_traits::float::FloatCore;

use crate::dispatch::with_sample_type;
use crate::error::{Result, WavError};
use crate::format::{RawSpec, WavSpec};
use crate::header::{self, Chunk, RIFF_SIZE_OFFSET};

/// The format a writer was created with: an interpreted [`WavSpec`] (the float
/// and raw write paths are both available) or raw [`RawSpec`] fmt fields (only
/// the raw byte path is available).
enum WriterSpec {
    Typed(WavSpec),
    Raw(RawSpec),
}

impl WriterSpec {
    /// The number of bytes per frame, used to count frames on finalize.
    fn frame_bytes(&self) -> usize {
        match self {
            WriterSpec::Typed(spec) => spec.frame_bytes(),
            WriterSpec::Raw(spec) => spec.frame_bytes(),
        }
    }

    /// Write the `fmt ` chunk for this format and report whether a `fact` chunk
    /// is required. Returns `(fmt body size, needs_fact)`.
    fn write_fmt(&self, dest: &mut impl Write) -> Result<(u32, bool)> {
        match self {
            WriterSpec::Typed(spec) => {
                let body = header::write_fmt_chunk(
                    dest,
                    spec.channels,
                    spec.sample_format,
                    spec.sample_rate,
                    spec.channel_mask,
                )?;
                // The spec requires a `fact` chunk for every format that is not
                // plain WAVE_FORMAT_PCM: float, A-law, mu-law, and the
                // WAVEFORMATEXTENSIBLE form even when its subformat is PCM.
                let needs_fact = !spec.sample_format.is_pcm()
                    || header::writes_as_extensible(
                        spec.channels,
                        spec.sample_format,
                        spec.channel_mask,
                    );
                Ok((body, needs_fact))
            }
            // Raw writers never emit a `fact` chunk: the format is uninterpreted,
            // so the sample-frame count carries no defined meaning.
            WriterSpec::Raw(spec) => Ok((header::write_fmt_chunk_raw(dest, spec)?, false)),
        }
    }
}

/// An output stream that can be shortened.
///
/// The [`Write`] and [`Seek`] traits can grow a stream but never shrink one, so
/// trimming a file needs this extra capability.
/// [`WavWriter::truncate`] and [`WavWriter::truncate_to_frame`] are available
/// only when the inner writer implements it.
///
/// It is implemented for [`File`](std::fs::File) (which is where it matters),
/// for [`Cursor<Vec<u8>>`](std::io::Cursor), and for the
/// [`BufWriter`](std::io::BufWriter) and `&mut` wrappers that usually sit
/// between the two and a [`WavWriter`]. Implement it for a custom output to
/// enable trimming there too.
pub trait Truncate {
    /// Shorten the stream to `len` bytes.
    ///
    /// `len` is never larger than the current length when called by
    /// [`WavWriter`]. The stream position is not required to change; the writer
    /// seeks to where it needs to be afterwards.
    fn truncate_to(&mut self, len: u64) -> io::Result<()>;
}

impl Truncate for std::fs::File {
    fn truncate_to(&mut self, len: u64) -> io::Result<()> {
        self.set_len(len)
    }
}

// A `File` is also writable through a shared reference, so a writer built on
// `&File` can be trimmed as well.
impl Truncate for &std::fs::File {
    fn truncate_to(&mut self, len: u64) -> io::Result<()> {
        self.set_len(len)
    }
}

impl Truncate for Cursor<Vec<u8>> {
    fn truncate_to(&mut self, len: u64) -> io::Result<()> {
        let len = usize::try_from(len).map_err(|_| io::Error::other("length overflows usize"))?;
        self.get_mut().truncate(len);
        Ok(())
    }
}

impl<W: Truncate + Write> Truncate for BufWriter<W> {
    fn truncate_to(&mut self, len: u64) -> io::Result<()> {
        // Buffered bytes may belong past the cut, so they have to reach the
        // inner writer before it is shortened.
        self.flush()?;
        self.get_mut().truncate_to(len)
    }
}

impl<W: Truncate + ?Sized> Truncate for &mut W {
    fn truncate_to(&mut self, len: u64) -> io::Result<()> {
        (**self).truncate_to(len)
    }
}

/// Chunk ids this crate manages itself, which callers may not supply as extra
/// metadata chunks.
const RESERVED_IDS: [&[u8; 4]; 4] = [b"RIFF", b"fmt ", b"data", b"fact"];

/// Check that an extra chunk supplied by the caller can be written: a non-reserved
/// id and a body that fits in the 32-bit chunk size field.
fn check_extra_chunk(id: &[u8; 4], body_len: usize) -> Result<()> {
    if RESERVED_IDS.contains(&id) {
        return Err(WavError::InvalidSpec(format!(
            "chunk id {:?} is reserved and written automatically",
            String::from_utf8_lossy(id)
        )));
    }
    if u32::try_from(body_len).is_err() {
        return Err(WavError::InvalidSpec(format!(
            "chunk {:?} body of {body_len} bytes does not fit in 32 bits",
            String::from_utf8_lossy(id)
        )));
    }
    Ok(())
}

/// The container form to write: plain RIFF, or the 64-bit RF64 form.
enum Container {
    /// Plain RIFF/WAVE. The not-yet-known size fields get the
    /// [`UNKNOWN_SIZE`](header::UNKNOWN_SIZE) placeholder, which a seekable
    /// writer patches on finalize and a streaming writer leaves as-is.
    Riff,
    /// RF64: 32-bit size fields carry the `0xFFFFFFFF` marker and the real sizes
    /// live in a leading `ds64` chunk, patched on finalize (always seekable).
    Rf64,
}

/// The size fields recorded while writing the header, needed to patch them on
/// finalize. Which fields exist depends on the container form.
enum SizeFields {
    /// Plain RIFF: the RIFF size is at the fixed [`RIFF_SIZE_OFFSET`].
    Riff {
        /// File offset of the 32-bit data chunk size field.
        data_size_offset: u64,
        /// File offset of the 4-byte `fact` sample-count field, if a `fact`
        /// chunk was written (non-PCM formats only).
        fact_offset: Option<u64>,
    },
    /// RF64: the 64-bit sizes all live in the `ds64` chunk.
    Rf64 {
        riff_size_offset: u64,
        data_size_offset: u64,
        sample_count_offset: u64,
    },
}

/// The byte offsets recorded while writing the header, needed to patch the size
/// fields on finalize.
struct Layout {
    /// Bytes written by the header, up to and including the data chunk header.
    header_len: u64,
    sizes: SizeFields,
}

/// Write the full header and record the offsets needed to patch sizes later.
///
/// For RIFF this is RIFF + WAVE, `fmt `, an optional `fact` chunk for non-PCM
/// formats, the caller's leading chunks, then the data chunk header. For RF64 it
/// is RF64 + WAVE, a `ds64` chunk (carrying the 64-bit sizes and sample count,
/// so no `fact` chunk is written), `fmt `, the leading chunks, then the data
/// chunk header with a `0xFFFFFFFF` size marker.
fn write_header(
    inner: &mut impl Write,
    spec: &WriterSpec,
    leading: &[Chunk],
    container: Container,
) -> Result<Layout> {
    for chunk in leading {
        check_extra_chunk(&chunk.id, chunk.data.len())?;
    }

    let mut pos: u64 = 12;
    let sizes = match container {
        Container::Riff => {
            header::write_riff_wave(inner, header::UNKNOWN_SIZE)?;

            let (fmt_body, needs_fact) = spec.write_fmt(inner)?;
            pos += 8 + fmt_body as u64;

            // A `fact` chunk (sample-frame count) follows for every format that
            // is not plain WAVE_FORMAT_PCM (see `WriterSpec::write_fmt`). The
            // 4-byte body sits right after the 8-byte chunk header.
            let fact_offset = if needs_fact {
                let offset = pos + 8;
                pos +=
                    header::write_named_chunk(inner, b"fact", &header::UNKNOWN_SIZE.to_le_bytes())?;
                Some(offset)
            } else {
                None
            };

            for chunk in leading {
                pos += header::write_named_chunk(inner, &chunk.id, &chunk.data)?;
            }

            let data_size_offset = pos + 4;
            header::write_data_header(inner, header::UNKNOWN_SIZE)?;
            pos += 8;

            SizeFields::Riff {
                data_size_offset,
                fact_offset,
            }
        }
        Container::Rf64 => {
            header::write_rf64_wave(inner)?;
            // The ds64 chunk follows immediately: 8-byte chunk header, then the
            // riffSize/dataSize/sampleCount 64-bit fields.
            let riff_size_offset = pos + 8;
            let data_size_offset = pos + 8 + 8;
            let sample_count_offset = pos + 8 + 16;
            header::write_ds64_chunk(inner)?;
            pos += 8 + header::DS64_BODY_SIZE as u64;

            let (fmt_body, _needs_fact) = spec.write_fmt(inner)?;
            pos += 8 + fmt_body as u64;

            // No `fact` chunk: RF64 carries the sample count in the ds64 chunk.
            for chunk in leading {
                pos += header::write_named_chunk(inner, &chunk.id, &chunk.data)?;
            }

            header::write_data_header(inner, header::RF64_DATA_SIZE_MARKER)?;
            pos += 8;

            SizeFields::Rf64 {
                riff_size_offset,
                data_size_offset,
                sample_count_offset,
            }
        }
    };

    Ok(Layout {
        header_len: pos,
        sizes,
    })
}

/// A writer for wav files.
///
/// It writes the header on construction and then accepts audio data, either
/// from a floating point [`Adapter`](audioadapter::Adapter) buffer (converting
/// on the fly) or as raw interleaved bytes.
///
/// There are two modes:
///
/// * Seekable, created with [`WavWriter::new`]. The size fields are written as
///   placeholders and patched with the real values by [`WavWriter::finalize`],
///   producing a standard-compliant file.
/// * Streaming, created with [`WavWriter::new_streaming`]. The size fields are
///   never updated, which is useful for pipes and other non-seekable outputs.
///   Call [`WavWriter::into_inner`] when done.
///
/// Both write the placeholder as [`u32::MAX`], the "runs to the end of the file"
/// convention players expect from a stream of unknown length. So a seekable file
/// whose writer never got to [`finalize`](WavWriter::finalize), because the
/// process was interrupted or crashed, still reads back as the audio that made
/// it to disk rather than as an empty file.
///
/// For files that may exceed the 4 GB size limit of plain RIFF, use
/// [`WavWriter::new_rf64`], which writes the 64-bit RF64 form (with a `ds64`
/// chunk) and is patched by [`finalize`](WavWriter::finalize) like the seekable
/// RIFF writer. A seekable RIFF writer instead errors if a write would exceed
/// 4 GB.
///
/// A `fact` chunk (sample-frame count) is written automatically for every
/// format the spec considers non-PCM: float, and the `WAVEFORMATEXTENSIBLE`
/// form (`I24_4` or more than two channels). Only plain integer PCM omits it.
/// Arbitrary extra
/// chunks can be written before the audio (leading chunks, via
/// [`WavWriter::new_with_chunks`] / [`WavWriter::new_streaming_with_chunks`]) or
/// after it (trailing chunks, via [`WavWriter::write_chunk`]), so a higher-level
/// library can attach metadata such as `LIST`/`INFO`.
///
/// # Picking a constructor
///
/// The constructors vary along two independent axes: how the output is written
/// (seekable RIFF, seekable RF64, or streaming), and whether the sample format
/// is interpreted, a [`WavSpec`] giving both write paths, or passed through
/// verbatim, a [`RawSpec`] giving only the raw byte path.
///
/// | Output | [`WavSpec`]: float and raw | [`RawSpec`]: raw only |
/// |---|---|---|
/// | Seekable, plain RIFF, up to 4 GB | [`new`](WavWriter::new) | [`new_raw`](WavWriter::new_raw) |
/// | Seekable, RF64, no size limit | [`new_rf64`](WavWriter::new_rf64) | not available |
/// | Streaming, no seeking needed | [`new_streaming`](WavWriter::new_streaming) | [`new_streaming_raw`](WavWriter::new_streaming_raw) |
///
/// Each of the five has a `_with_chunks` twin taking a `&[Chunk]` of metadata
/// chunks to write ahead of the audio, for example
/// [`new_with_chunks`](WavWriter::new_with_chunks). Finish a seekable writer
/// with [`finalize`](WavWriter::finalize), which patches the size fields and
/// hands back the inner writer, and a streaming one with
/// [`into_inner`](WavWriter::into_inner).
///
/// # Examples
///
/// The common case, a seekable output written from a float buffer:
///
/// ```
/// use std::io::Cursor;
/// use audioadapter_buffers::owned::InterleavedOwned;
/// use waveadapter::{SampleFormat, WavSpec, WavWriter};
///
/// let audio = InterleavedOwned::<f32>::new(0.0, 2, 128);
/// let spec = WavSpec::new(2, 44100, SampleFormat::I16);
///
/// let mut writer = WavWriter::new(Cursor::new(Vec::new()), spec)?;
/// let clipped = writer.write_float_buffer(&audio)?;
/// let file = writer.finalize()?.into_inner();
/// assert_eq!(&file[0..4], b"RIFF");
/// # Ok::<(), waveadapter::WavError>(())
/// ```
///
/// The same audio to an output that cannot seek, a pipe or a plain
/// [`Vec<u8>`], where nothing can be patched afterwards, so the sizes keep
/// their [`u32::MAX`] placeholders:
///
/// ```
/// # use audioadapter_buffers::owned::InterleavedOwned;
/// # use waveadapter::{SampleFormat, WavSpec, WavWriter};
/// # let audio = InterleavedOwned::<f32>::new(0.0f32, 2, 128);
/// # let spec = WavSpec::new(2, 44100, SampleFormat::I16);
/// let mut writer = WavWriter::new_streaming(Vec::new(), spec)?;
/// writer.write_float_buffer(&audio)?;
/// let file = writer.into_inner()?;
/// assert_eq!(file[4..8], u32::MAX.to_le_bytes());
/// # Ok::<(), waveadapter::WavError>(())
/// ```
pub struct WavWriter<W: Write> {
    inner: W,
    spec: WriterSpec,
    /// The highest extent of the data chunk reached so far, in bytes. This is the
    /// declared `data` size and may be larger than [`data_pos`](Self::data_pos)
    /// after a backwards [`seek_to_frame`](WavWriter::seek_to_frame).
    data_bytes: u64,
    /// The current write cursor within the data chunk, in bytes from its start.
    /// Equal to [`data_bytes`](Self::data_bytes) for append-only writing; it moves
    /// independently once [`seek_to_frame`](WavWriter::seek_to_frame) is used.
    data_pos: u64,
    /// Bytes written by trailing chunks (and the data pad byte), after the audio
    /// data. Tracked so [`finalize`](WavWriter::finalize) can size the RIFF chunk.
    trailing_bytes: u64,
    seekable: bool,
    layout: Layout,
}

impl<W: Write> WavWriter<W> {
    /// Create a streaming writer.
    ///
    /// The RIFF and data size fields are set to [`u32::MAX`], matching what
    /// players expect from a stream of unknown length. The output does not need
    /// to be seekable.
    pub fn new_streaming(inner: W, spec: WavSpec) -> Result<Self> {
        Self::new_streaming_with_chunks(inner, spec, &[])
    }

    /// Create a streaming writer that emits `leading` metadata chunks between the
    /// `fmt ` chunk and the audio data.
    ///
    /// Like [`new_streaming`](WavWriter::new_streaming), the size fields are left
    /// at [`u32::MAX`]. See [`Chunk`] for the chunk representation. Reserved ids
    /// (`fmt `, `data`, `fact`, `RIFF`) are rejected with
    /// [`WavError::InvalidSpec`](crate::WavError::InvalidSpec).
    pub fn new_streaming_with_chunks(
        mut inner: W,
        spec: WavSpec,
        leading: &[Chunk],
    ) -> Result<Self> {
        let spec = WriterSpec::Typed(spec);
        let layout = write_header(&mut inner, &spec, leading, Container::Riff)?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            data_pos: 0,
            trailing_bytes: 0,
            seekable: false,
            layout,
        })
    }

    /// Create a streaming raw writer for a format this crate does not interpret.
    ///
    /// The `fmt ` chunk is written verbatim from the [`RawSpec`] and no `fact`
    /// chunk is emitted. Only [`write_raw_interleaved`](WavWriter::write_raw_interleaved)
    /// is available; [`write_float_buffer`](WavWriter::write_float_buffer) returns
    /// [`WavError::UnsupportedFormat`](crate::WavError::UnsupportedFormat). Size
    /// fields are left at [`u32::MAX`]; finish with [`into_inner`](WavWriter::into_inner).
    pub fn new_streaming_raw(inner: W, spec: RawSpec) -> Result<Self> {
        Self::new_streaming_raw_with_chunks(inner, spec, &[])
    }

    /// Like [`new_streaming_raw`](WavWriter::new_streaming_raw) but with leading
    /// metadata chunks between the `fmt ` chunk and the audio data.
    pub fn new_streaming_raw_with_chunks(
        mut inner: W,
        spec: RawSpec,
        leading: &[Chunk],
    ) -> Result<Self> {
        let spec = WriterSpec::Raw(spec);
        let layout = write_header(&mut inner, &spec, leading, Container::Riff)?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            data_pos: 0,
            trailing_bytes: 0,
            seekable: false,
            layout,
        })
    }

    /// The interpreted spec the writer was created with, or `None` for a raw
    /// writer (see [`raw_spec`](WavWriter::raw_spec)).
    pub fn spec(&self) -> Option<WavSpec> {
        match self.spec {
            WriterSpec::Typed(spec) => Some(spec),
            WriterSpec::Raw(_) => None,
        }
    }

    /// The raw spec a raw writer was created with, or `None` for a typed writer.
    pub fn raw_spec(&self) -> Option<RawSpec> {
        match self.spec {
            WriterSpec::Raw(spec) => Some(spec),
            WriterSpec::Typed(_) => None,
        }
    }

    /// The number of audio data bytes written so far.
    ///
    /// This is the furthest extent reached, the value that becomes the declared
    /// `data` chunk size. See [`frames_written`](WavWriter::frames_written) for
    /// the same number in frames, and [`position`](WavWriter::position) for where
    /// the write cursor currently is.
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    /// The absolute byte offset in the output where the audio data starts.
    ///
    /// This is the size of everything written before the audio: the RIFF/RF64
    /// header, the `fmt ` chunk, any `fact` or `ds64` chunk, the leading metadata
    /// chunks and the `data` chunk header. It mirrors
    /// [`WavParams::data_offset`](crate::WavParams::data_offset) on the read side.
    pub fn data_offset(&self) -> u64 {
        self.layout.header_len
    }

    /// The number of frames written, counted to the furthest extent reached.
    ///
    /// After a backwards [`seek_to_frame`](WavWriter::seek_to_frame) this stays at
    /// the high-water mark, since overwriting does not shrink the file, until
    /// [`truncate`](WavWriter::truncate) moves it down. A partial
    /// trailing frame, only possible through
    /// [`write_raw_interleaved`](WavWriter::write_raw_interleaved), is not counted.
    /// Zero if the frame size is unknown (a [`RawSpec`] with a zero block
    /// alignment).
    pub fn frames_written(&self) -> usize {
        self.frames_at(self.data_bytes)
    }

    /// The current write position, in frames from the start of the audio data.
    ///
    /// Equal to [`frames_written`](WavWriter::frames_written) for append-only
    /// writing; the two differ once [`seek_to_frame`](WavWriter::seek_to_frame) has
    /// moved the cursor back. Mirrors [`WavReader::position`](crate::WavReader::position).
    pub fn position(&self) -> usize {
        self.frames_at(self.data_pos)
    }

    /// The number of frames that can still be written before the 4 GB size limit
    /// of plain RIFF, or `None` when no limit applies.
    ///
    /// The limit only exists for a seekable RIFF writer, the one case where
    /// [`finalize`](WavWriter::finalize) has to fit the real lengths into the
    /// 32-bit size fields. A streaming writer never patches them and RF64 has
    /// 64-bit fields, so both return `None`, as does a [`RawSpec`] writer with a
    /// zero block alignment, where frames have no size.
    ///
    /// The room left is counted from the current write position, and assumes no
    /// further trailing chunks: every byte a later
    /// [`write_chunk`](WavWriter::write_chunk) adds comes out of this budget.
    /// Once a trailing chunk has been written the answer is `Some(0)`, since no
    /// more audio can follow it.
    pub fn remaining_frames(&self) -> Option<usize> {
        if !self.seekable || !matches!(self.layout.sizes, SizeFields::Riff { .. }) {
            return None;
        }
        let frame_bytes = self.spec.frame_bytes() as u64;
        if frame_bytes == 0 {
            return None;
        }
        if self.trailing_bytes != 0 {
            return Some(0);
        }
        // The largest data extent that still fits, the same budget `check_capacity`
        // enforces, counted from the cursor since that is where a write lands.
        let ceiling =
            (u32::MAX as u64 + 8).saturating_sub(self.layout.header_len + self.trailing_bytes);
        let bytes = ceiling.saturating_sub(self.data_pos);
        Some(usize::try_from(bytes / frame_bytes).unwrap_or(usize::MAX))
    }

    /// Whole frames covered by `bytes` of audio data, zero if the frame size is
    /// unknown.
    fn frames_at(&self, bytes: u64) -> usize {
        let frames = bytes
            .checked_div(self.spec.frame_bytes() as u64)
            .unwrap_or(0);
        usize::try_from(frames).unwrap_or(usize::MAX)
    }

    /// Write all frames of a floating point buffer, converting to the file's
    /// sample format.
    ///
    /// Each sample is scaled from the range -1.0..1.0. For the integer formats,
    /// values outside that range are clipped to the nearest limit, and the
    /// return value counts how many samples were clipped. The float formats
    /// ([`SampleFormat::F32`](crate::SampleFormat::F32) and
    /// [`SampleFormat::F64`](crate::SampleFormat::F64)) are not range limited:
    /// values outside -1.0..1.0 are valid headroom and pass through unchanged,
    /// so writing to them always returns zero.
    ///
    /// The G.711 formats ([`ALAW`](crate::SampleFormat::ALAW) and
    /// [`MULAW`](crate::SampleFormat::MULAW)) are lossy: every value is
    /// quantized to one of 256 code words, not just the out-of-range ones. The
    /// count still means "the sample exceeded full scale", not "the sample was
    /// altered". Their largest code is also slightly short of full scale
    /// (32256 and 32124 of the 32768 the conversion scales to), so a value just
    /// under 1.0 saturates to the largest code without being counted.
    ///
    /// Returns [`WavError::InvalidSpec`](crate::WavError::InvalidSpec) if a
    /// trailing chunk has already been written, since audio data must precede
    /// trailing chunks.
    pub fn write_float_buffer<T>(&mut self, src: &dyn Adapter<T>) -> Result<usize>
    where
        T: FloatCore + ToPrimitive,
    {
        self.ensure_data_open()?;
        let spec = match self.spec {
            WriterSpec::Typed(spec) => spec,
            WriterSpec::Raw(_) => {
                return Err(WavError::UnsupportedFormat(
                    "cannot write a float buffer to a raw writer; \
                     use write_raw_interleaved instead"
                        .to_string(),
                ));
            }
        };
        let frames = src.frames();
        let channels = src.channels();
        let byte_count = (frames * channels * spec.sample_format.bytes_per_sample()) as u64;
        self.check_capacity(byte_count)?;
        let mut clipped = 0;
        with_sample_type!(spec.sample_format, S, {
            for frame in 0..frames {
                for ch in 0..channels {
                    let value = src.read_sample(ch, frame).unwrap();
                    if self.inner.write_converted::<S, T>(value)? {
                        clipped += 1;
                    }
                }
            }
        });
        self.advance(byte_count);
        Ok(clipped)
    }

    /// Write raw interleaved bytes directly to the data chunk.
    ///
    /// The caller is responsible for the bytes being in the file's sample format
    /// and channel interleaving.
    ///
    /// Returns [`WavError::InvalidSpec`](crate::WavError::InvalidSpec) if a
    /// trailing chunk has already been written.
    pub fn write_raw_interleaved(&mut self, data: &[u8]) -> Result<()> {
        self.ensure_data_open()?;
        self.check_capacity(data.len() as u64)?;
        self.inner.write_all(data)?;
        self.advance(data.len() as u64);
        Ok(())
    }

    /// Advance the data cursor by `byte_count` after a write, growing the recorded
    /// data extent if the cursor moved past it.
    fn advance(&mut self, byte_count: u64) {
        self.data_pos += byte_count;
        self.data_bytes = self.data_bytes.max(self.data_pos);
    }

    /// Write `count` zero bytes at the current position, as audio data.
    ///
    /// Used to fill the gap left by a forward seek. Callers must have checked the
    /// capacity first.
    fn write_zeros(&mut self, count: u64) -> Result<()> {
        const ZEROS: [u8; 8192] = [0; 8192];
        let mut left = count;
        while left > 0 {
            let chunk = left.min(ZEROS.len() as u64) as usize;
            self.inner.write_all(&ZEROS[..chunk])?;
            left -= chunk as u64;
        }
        self.advance(count);
        Ok(())
    }

    /// Write a metadata chunk after the audio data.
    ///
    /// Call this once all audio data has been written; afterwards no more audio
    /// can be written. The data chunk is padded to an even length first, as the
    /// RIFF spec requires before a following chunk. Reserved ids (`fmt `, `data`,
    /// `fact`, `RIFF`) are rejected with
    /// [`WavError::InvalidSpec`](crate::WavError::InvalidSpec).
    pub fn write_chunk(&mut self, id: [u8; 4], data: &[u8]) -> Result<()> {
        check_extra_chunk(&id, data.len())?;
        // Pad the data chunk to an even length before the first trailing chunk.
        // The pad byte is not part of the data chunk's declared size but does
        // count toward the RIFF size.
        if self.trailing_bytes == 0 && self.data_bytes % 2 == 1 {
            self.inner.write_all(&[0])?;
            self.trailing_bytes += 1;
        }
        self.trailing_bytes += header::write_named_chunk(&mut self.inner, &id, data)?;
        Ok(())
    }

    /// Reject an audio write once trailing chunks have started.
    fn ensure_data_open(&self) -> Result<()> {
        if self.trailing_bytes != 0 {
            return Err(WavError::InvalidSpec(
                "cannot write audio data after a trailing chunk".to_string(),
            ));
        }
        Ok(())
    }

    /// Reject a write that would push a seekable RIFF file past the 4 GB size
    /// limit, where the RIFF/data size fields can no longer hold the real length.
    ///
    /// This only applies to seekable RIFF output: streaming RIFF deliberately
    /// leaves the size fields at [`u32::MAX`], and RF64 has no such limit.
    fn check_capacity(&self, additional: u64) -> Result<()> {
        // The write may land before the current end (after a backwards seek),
        // so size against whichever extent is larger.
        self.check_extent(self.data_bytes.max(self.data_pos + additional))
    }

    /// Reject a data extent that a seekable RIFF file could not describe.
    fn check_extent(&self, extent: u64) -> Result<()> {
        if self.seekable && matches!(self.layout.sizes, SizeFields::Riff { .. }) {
            let projected = self.layout.header_len + extent + self.trailing_bytes;
            if projected.saturating_sub(8) > u32::MAX as u64 {
                return Err(WavError::InvalidSpec(
                    "writing this data would exceed the 4 GB RIFF size limit; \
                     use an RF64 writer (WavWriter::new_rf64)"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Flush and return the inner writer without patching the size fields.
    ///
    /// This is the way to finish a streaming writer. For a seekable writer that
    /// should have correct size fields, use [`WavWriter::finalize`] instead.
    pub fn into_inner(mut self) -> Result<W> {
        self.inner.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write + Seek> WavWriter<W> {
    /// Create a seekable writer.
    ///
    /// The RIFF and data size fields are written as placeholders and filled in
    /// with the real values by [`WavWriter::finalize`].
    pub fn new(inner: W, spec: WavSpec) -> Result<Self> {
        Self::new_with_chunks(inner, spec, &[])
    }

    /// Create a seekable writer that emits `leading` metadata chunks between the
    /// `fmt ` chunk and the audio data.
    ///
    /// Like [`new`](WavWriter::new), the size fields are placeholders patched by
    /// [`finalize`](WavWriter::finalize). See [`Chunk`] for the chunk
    /// representation. Reserved ids (`fmt `, `data`, `fact`, `RIFF`) are rejected
    /// with [`WavError::InvalidSpec`](crate::WavError::InvalidSpec).
    pub fn new_with_chunks(mut inner: W, spec: WavSpec, leading: &[Chunk]) -> Result<Self> {
        let spec = WriterSpec::Typed(spec);
        let layout = write_header(&mut inner, &spec, leading, Container::Riff)?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            data_pos: 0,
            trailing_bytes: 0,
            seekable: true,
            layout,
        })
    }

    /// Create a seekable raw writer for a format this crate does not interpret.
    ///
    /// The `fmt ` chunk is written verbatim from the [`RawSpec`] and no `fact`
    /// chunk is emitted. Only [`write_raw_interleaved`](WavWriter::write_raw_interleaved)
    /// is available; [`write_float_buffer`](WavWriter::write_float_buffer) returns
    /// [`WavError::UnsupportedFormat`](crate::WavError::UnsupportedFormat). Size
    /// fields are patched by [`finalize`](WavWriter::finalize).
    pub fn new_raw(inner: W, spec: RawSpec) -> Result<Self> {
        Self::new_raw_with_chunks(inner, spec, &[])
    }

    /// Like [`new_raw`](WavWriter::new_raw) but with leading metadata chunks
    /// between the `fmt ` chunk and the audio data.
    pub fn new_raw_with_chunks(mut inner: W, spec: RawSpec, leading: &[Chunk]) -> Result<Self> {
        let spec = WriterSpec::Raw(spec);
        let layout = write_header(&mut inner, &spec, leading, Container::Riff)?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            data_pos: 0,
            trailing_bytes: 0,
            seekable: true,
            layout,
        })
    }

    /// Create a seekable RF64 writer, for files that may exceed the 4 GB limit of
    /// plain RIFF.
    ///
    /// The file is written with the `RF64` form id and a `ds64` chunk; the 64-bit
    /// size and sample-count fields are patched by [`finalize`](WavWriter::finalize).
    /// Unlike a plain RIFF writer there is no 4 GB ceiling, so audio writes never
    /// fail on size. RF64 requires a seekable output.
    pub fn new_rf64(inner: W, spec: WavSpec) -> Result<Self> {
        Self::new_rf64_with_chunks(inner, spec, &[])
    }

    /// Create a seekable RF64 writer that emits `leading` metadata chunks between
    /// the `fmt ` chunk and the audio data.
    ///
    /// Like [`new_rf64`](WavWriter::new_rf64) but with leading chunks. See
    /// [`Chunk`] for the chunk representation. Reserved ids (`fmt `, `data`,
    /// `fact`, `RIFF`) are rejected with
    /// [`WavError::InvalidSpec`](crate::WavError::InvalidSpec).
    pub fn new_rf64_with_chunks(mut inner: W, spec: WavSpec, leading: &[Chunk]) -> Result<Self> {
        let spec = WriterSpec::Typed(spec);
        let layout = write_header(&mut inner, &spec, leading, Container::Rf64)?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            data_pos: 0,
            trailing_bytes: 0,
            seekable: true,
            layout,
        })
    }

    /// Seek to a frame for random-access writing.
    ///
    /// Positions the write cursor at the start of frame `frame`, so the next
    /// [`write_float_buffer`](WavWriter::write_float_buffer) or
    /// [`write_raw_interleaved`](WavWriter::write_raw_interleaved) overwrites the
    /// audio from there. Seeking backwards and overwriting does not shrink the
    /// file: the declared `data` size still covers the furthest point reached, so
    /// data beyond the rewritten region is preserved. Use
    /// [`truncate`](WavWriter::truncate) to drop it instead.
    ///
    /// Seeking past the current end fills the gap with zeros, so skipping ahead
    /// leaves silence rather than undefined bytes. The zeros are written straight
    /// away and count as audio data, so the file grows to the seek target even if
    /// nothing is written afterwards.
    ///
    /// Returns [`WavError::InvalidSpec`](crate::WavError::InvalidSpec) if a
    /// trailing chunk has already been written, since audio data must precede
    /// trailing chunks, or if the target lies beyond the 4 GB limit of a seekable
    /// RIFF file.
    pub fn seek_to_frame(&mut self, frame: usize) -> Result<()> {
        self.ensure_data_open()?;
        let frame_bytes = self.spec.frame_bytes() as u64;
        if frame_bytes == 0 {
            return Err(WavError::InvalidSpec(
                "cannot seek: frame size is zero".to_string(),
            ));
        }
        let offset = frame_bytes
            .checked_mul(frame as u64)
            .ok_or_else(|| WavError::InvalidSpec("seek target overflows".to_string()))?;
        if offset > self.data_bytes {
            // Past the end: fill the gap with silence from the current extent, so
            // the skipped region has defined contents whatever the inner writer
            // does with a seek beyond the end of the stream.
            self.check_extent(offset)?;
            self.inner
                .seek(SeekFrom::Start(self.layout.header_len + self.data_bytes))?;
            self.data_pos = self.data_bytes;
            self.write_zeros(offset - self.data_bytes)?;
        } else {
            self.inner
                .seek(SeekFrom::Start(self.layout.header_len + offset))?;
            self.data_pos = offset;
        }
        Ok(())
    }

    /// Patch the size fields with the real lengths and return the inner writer.
    ///
    /// For a streaming writer (created with [`WavWriter::new_streaming`]) the
    /// size fields are left at [`u32::MAX`]; only the inner writer is flushed and
    /// returned.
    pub fn finalize(mut self) -> Result<W> {
        self.inner.flush()?;
        if !self.seekable {
            return Ok(self.inner);
        }
        self.patch_sizes()?;
        self.inner.seek(SeekFrom::End(0))?;
        Ok(self.inner)
    }

    /// Update the size fields in the header to cover the audio written so far,
    /// then return to the current write position.
    ///
    /// Call this periodically during a long recording so that the file on disk
    /// stays valid: if the process is interrupted before
    /// [`finalize`](WavWriter::finalize), the file describes everything up to the
    /// last update instead of being unusable. It flushes the inner writer before
    /// patching, so the header never claims more audio than has been handed to
    /// the underlying stream.
    ///
    /// This matters most for [RF64](WavWriter::new_rf64), where the `ds64` sizes
    /// start at zero and an interrupted file would otherwise declare no audio at
    /// all. A plain RIFF file degrades more gracefully on its own, since its
    /// placeholder is the [`u32::MAX`] "runs to the end of the file" marker, but
    /// updating still turns it into a properly sized file.
    ///
    /// Writing continues normally afterwards, and the sizes are written again by
    /// [`finalize`](WavWriter::finalize). For a streaming writer this does
    /// nothing but flush, since its sizes stay at [`u32::MAX`] by design.
    pub fn update_header(&mut self) -> Result<()> {
        self.inner.flush()?;
        if !self.seekable {
            return Ok(());
        }
        let resume = self.inner.stream_position()?;
        self.patch_sizes()?;
        self.inner.seek(SeekFrom::Start(resume))?;
        Ok(())
    }

    /// Write the current lengths into the header size fields. Leaves the stream
    /// positioned at the last field written, so callers must reposition after.
    fn patch_sizes(&mut self) -> Result<()> {
        // Everything after the 8-byte RIFF/RF64 id/size: the header body, the
        // audio data and any trailing chunks (with the data pad byte).
        let riff_size = self.layout.header_len + self.data_bytes + self.trailing_bytes - 8;
        let frame_bytes = self.spec.frame_bytes() as u64;
        let frames = self.data_bytes.checked_div(frame_bytes).unwrap_or(0);

        match self.layout.sizes {
            SizeFields::Riff {
                data_size_offset,
                fact_offset,
            } => {
                // The eager capacity check keeps writes under 4 GB, so these
                // conversions only fail if the caller bypassed the writer; treat
                // that as a spec error rather than silently truncating.
                let too_large = || {
                    WavError::InvalidSpec(
                        "file exceeds the 4 GB RIFF size limit; use an RF64 writer".to_string(),
                    )
                };
                let riff_size = u32::try_from(riff_size).map_err(|_| too_large())?;
                let data_size = u32::try_from(self.data_bytes).map_err(|_| too_large())?;

                self.inner.seek(SeekFrom::Start(RIFF_SIZE_OFFSET))?;
                self.inner.write_all(&riff_size.to_le_bytes())?;
                self.inner.seek(SeekFrom::Start(data_size_offset))?;
                self.inner.write_all(&data_size.to_le_bytes())?;

                if let Some(fact_offset) = fact_offset {
                    let frames = u32::try_from(frames).unwrap_or(u32::MAX);
                    self.inner.seek(SeekFrom::Start(fact_offset))?;
                    self.inner.write_all(&frames.to_le_bytes())?;
                }
            }
            SizeFields::Rf64 {
                riff_size_offset,
                data_size_offset,
                sample_count_offset,
            } => {
                // The 32-bit RIFF and data size fields keep their 0xFFFFFFFF
                // markers; the real 64-bit values go into the ds64 chunk.
                self.inner.seek(SeekFrom::Start(riff_size_offset))?;
                self.inner.write_all(&riff_size.to_le_bytes())?;
                self.inner.seek(SeekFrom::Start(data_size_offset))?;
                self.inner.write_all(&self.data_bytes.to_le_bytes())?;
                self.inner.seek(SeekFrom::Start(sample_count_offset))?;
                self.inner.write_all(&frames.to_le_bytes())?;
            }
        }

        Ok(())
    }
}

impl<W: Write + Seek + Truncate> WavWriter<W> {
    /// Discard the audio data after the current write position.
    ///
    /// The escape hatch from the never-shrink rule: after a backwards
    /// [`seek_to_frame`](WavWriter::seek_to_frame) the declared size still covers
    /// the furthest point reached, and this drops everything past the cursor
    /// instead. Equivalent to
    /// `truncate_to_frame(self.position())`.
    pub fn truncate(&mut self) -> Result<()> {
        let frame = self.position();
        self.truncate_to_frame(frame)
    }

    /// Discard the audio data after frame `frame`, keeping frames `0..frame`.
    ///
    /// The file is shortened on the spot and the declared sizes follow when
    /// [`finalize`](WavWriter::finalize) or
    /// [`update_header`](WavWriter::update_header) next writes them. The write
    /// cursor stays where it is, unless it was inside the discarded region, in
    /// which case it moves to the new end.
    ///
    /// A `frame` at or past the current end does nothing: this only ever shrinks.
    /// Use [`seek_to_frame`](WavWriter::seek_to_frame) to extend a file, which
    /// fills the gap with silence.
    ///
    /// Returns [`WavError::InvalidSpec`](crate::WavError::InvalidSpec) if a
    /// trailing chunk has already been written, since trimming the audio would
    /// leave it stranded, if the writer is streaming (its sizes are never
    /// patched), or if the frame size is unknown (a [`RawSpec`] with a zero block
    /// alignment).
    pub fn truncate_to_frame(&mut self, frame: usize) -> Result<()> {
        self.ensure_data_open()?;
        if !self.seekable {
            return Err(WavError::InvalidSpec(
                "cannot truncate a streaming writer".to_string(),
            ));
        }
        let frame_bytes = self.spec.frame_bytes() as u64;
        if frame_bytes == 0 {
            return Err(WavError::InvalidSpec(
                "cannot truncate: frame size is zero".to_string(),
            ));
        }
        let offset = frame_bytes.saturating_mul(frame as u64);
        if offset >= self.data_bytes {
            return Ok(());
        }

        // Buffered bytes past the cut have to reach the stream before it is
        // shortened, or they would land after it and undo the trim.
        self.inner.flush()?;
        self.inner.truncate_to(self.layout.header_len + offset)?;
        self.data_bytes = offset;
        // Shortening the stream does not move the cursor, so put it somewhere
        // valid: where it was, or the new end if that is now past it.
        self.data_pos = self.data_pos.min(offset);
        self.inner
            .seek(SeekFrom::Start(self.layout.header_len + self.data_pos))?;
        Ok(())
    }
}
