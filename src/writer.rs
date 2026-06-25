//! Writing audio data to a wav file.

use std::io::{Seek, SeekFrom, Write};

use audioadapter::Adapter;
use audioadapter_sample::readwrite::WriteSamples;
use num_traits::ToPrimitive;
use num_traits::float::FloatCore;

use crate::dispatch::with_sample_type;
use crate::error::{Result, WavError};
use crate::format::WavSpec;
use crate::header::{self, Chunk, RIFF_SIZE_OFFSET};

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
    /// Plain RIFF/WAVE. `placeholder` is written into the size fields that are
    /// not yet known: `0` for a seekable writer (patched on finalize) or
    /// [`u32::MAX`] for a streaming writer (left as-is).
    Riff { placeholder: u32 },
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
    spec: &WavSpec,
    leading: &[Chunk],
    container: Container,
) -> Result<Layout> {
    for chunk in leading {
        check_extra_chunk(&chunk.id, chunk.data.len())?;
    }

    let mut pos: u64 = 12;
    let sizes = match container {
        Container::Riff { placeholder } => {
            header::write_riff_wave(inner, placeholder)?;

            let fmt_body = header::write_fmt_chunk(
                inner,
                spec.channels,
                spec.sample_format,
                spec.sample_rate,
            )?;
            pos += 8 + fmt_body as u64;

            // The spec requires a `fact` chunk (sample-frame count) for every
            // format that is not plain WAVE_FORMAT_PCM: that means float, and
            // also the WAVEFORMATEXTENSIBLE form (format tag 0xFFFE), even when
            // its subformat is PCM. Plain integer PCM is the only case that omits
            // it. The 4-byte body sits right after the 8-byte chunk header.
            let needs_fact = spec.sample_format.format_code() == 3
                || header::writes_as_extensible(spec.channels, spec.sample_format);
            let fact_offset = if needs_fact {
                let offset = pos + 8;
                pos += header::write_named_chunk(inner, b"fact", &placeholder.to_le_bytes())?;
                Some(offset)
            } else {
                None
            };

            for chunk in leading {
                pos += header::write_named_chunk(inner, &chunk.id, &chunk.data)?;
            }

            let data_size_offset = pos + 4;
            header::write_data_header(inner, placeholder)?;
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

            let fmt_body = header::write_fmt_chunk(
                inner,
                spec.channels,
                spec.sample_format,
                spec.sample_rate,
            )?;
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
///   set to [`u32::MAX`] up front and never updated, which is useful for pipes
///   and other non-seekable outputs. Call [`WavWriter::into_inner`] when done.
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
pub struct WavWriter<W: Write> {
    inner: W,
    spec: WavSpec,
    data_bytes: u64,
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
        let layout = write_header(
            &mut inner,
            &spec,
            leading,
            Container::Riff {
                placeholder: u32::MAX,
            },
        )?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            trailing_bytes: 0,
            seekable: false,
            layout,
        })
    }

    /// The spec the writer was created with.
    pub fn spec(&self) -> WavSpec {
        self.spec
    }

    /// The number of audio data bytes written so far.
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    /// Write all frames of a floating point buffer, converting to the file's
    /// sample format.
    ///
    /// Each sample is scaled from the range -1.0..1.0 and clipped if it falls
    /// outside the range representable by the target format. Returns the number
    /// of samples that were clipped.
    ///
    /// Returns [`WavError::InvalidSpec`](crate::WavError::InvalidSpec) if a
    /// trailing chunk has already been written, since audio data must precede
    /// trailing chunks.
    pub fn write_float_buffer<T>(&mut self, src: &dyn Adapter<T>) -> Result<usize>
    where
        T: FloatCore + ToPrimitive,
    {
        self.ensure_data_open()?;
        let frames = src.frames();
        let channels = src.channels();
        let byte_count = (frames * channels * self.spec.sample_format.bytes_per_sample()) as u64;
        self.check_capacity(byte_count)?;
        let mut clipped = 0;
        with_sample_type!(self.spec.sample_format, S, {
            for frame in 0..frames {
                for ch in 0..channels {
                    let value = src.read_sample(ch, frame).unwrap();
                    if self.inner.write_converted::<S, T>(value)? {
                        clipped += 1;
                    }
                }
            }
        });
        self.data_bytes += byte_count;
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
        self.data_bytes += data.len() as u64;
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
        if self.seekable && matches!(self.layout.sizes, SizeFields::Riff { .. }) {
            let projected =
                self.layout.header_len + self.data_bytes + self.trailing_bytes + additional;
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
        let layout = write_header(
            &mut inner,
            &spec,
            leading,
            Container::Riff { placeholder: 0 },
        )?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
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
        let layout = write_header(&mut inner, &spec, leading, Container::Rf64)?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            trailing_bytes: 0,
            seekable: true,
            layout,
        })
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

        self.inner.seek(SeekFrom::End(0))?;
        Ok(self.inner)
    }
}
