//! Writing audio data to a wav file.

use std::io::{self, BufWriter, Cursor, Seek, SeekFrom, Write};
use std::marker::PhantomData;

use audioadapter::Adapter;
use audioadapter_sample::readwrite::WriteSamples;
use num_traits::ToPrimitive;
use num_traits::float::FloatCore;

use crate::dispatch::with_sample_type;
use crate::error::{Result, WavError};
use crate::format::{SampleFormat, WavSpec};
use crate::header::{self, Chunk, FmtChunk, RIFF_SIZE_OFFSET};

/// What to write into the `fact` chunk, which the spec requires for every format
/// that is not plain integer PCM.
///
/// The chunk holds a *sample-frame* count. For a modelled format the writer can
/// count those itself, but for a block-compressed one it cannot: the data is a
/// whole number of blocks and only the codec knows how many frames a block
/// decodes to. The GSM 6.10 fixture is three 65-byte blocks and 960 frames, and
/// nothing in the container relates those two numbers. So the count is the
/// caller's to supply whenever the writer cannot derive it.
///
/// RF64 has no `fact` chunk, but the same choice applies there: the count goes
/// into the `ds64` chunk's `sampleCount` field instead, and a variant that
/// writes nothing leaves it at zero.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Fact {
    /// Write one if the format needs it, counting the frames written. Only
    /// possible for a format this crate models; a `fact` chunk is omitted for
    /// anything else (and the RF64 `sampleCount` left at zero), since a guess
    /// would be worse than nothing.
    ///
    /// Counting means patching the field once the audio is written, so it also
    /// needs a seekable output. A streaming writer omits the chunk instead:
    /// leaving the placeholder in place would claim `u32::MAX` frames, and
    /// unlike the data size there is no convention that reads as "not known".
    /// A streaming caller who does know the count says so with
    /// [`Samples`](Fact::Samples).
    #[default]
    Auto,
    /// Never write one, whatever the format. Leaves the RF64 `sampleCount` at
    /// zero.
    None,
    /// Write this exact sample-frame count.
    ///
    /// The RF64 `sampleCount` field is 64-bit and takes any value here. The
    /// RIFF `fact` chunk's is 32-bit, so a count above [`u32::MAX`] is rejected
    /// when the writer is opened rather than truncated: a file with that many
    /// frames needs RF64 anyway.
    Samples(u64),
    /// Write this body verbatim, whatever it holds.
    ///
    /// The chunk's first four bytes are the count, but a format may define more
    /// after it, and the reader keeps whatever it finds in
    /// [`WavParams::fact`](crate::WavParams::fact). This is how those bytes get
    /// back out again, and so how a file with a longer `fact` chunk rewrites
    /// byte for byte. Nothing is patched on finalize: verbatim is verbatim.
    ///
    /// For RF64 there is no `fact` chunk to write, so the `ds64` `sampleCount`
    /// takes the body's first four bytes, that field being `dwSampleLength` by
    /// definition. A body too short to hold one leaves the count at zero.
    Body(Vec<u8>),
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

/// The chunks the writer produces itself, which a caller may not also supply.
///
/// `ds64` belongs here and was missing: a caller could inject a second one into
/// an RF64 file as a leading chunk. `fact` stays because the writer emits and
/// patches it; the way to control its contents is [`Fact`], not a hand-rolled
/// chunk. That does not clash with the pass-everything-through editing loop,
/// because the reader consumes `fact` into
/// [`WavParams::fact`](crate::WavParams::fact) rather than leaving it among the
/// chunks for a caller to hand back.
const RESERVED_IDS: [&[u8; 4]; 5] = [b"RIFF", b"fmt ", b"data", b"fact", b"ds64"];

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
        /// What to put in the `ds64` sample-count field, or `None` to leave it
        /// at zero because no honest count is available.
        sample_count: Option<Ds64Count>,
    },
}

/// What the `ds64` sample-count field should hold, once the format and the
/// caller's [`Fact`] choice are both taken into account. Recorded in
/// [`SizeFields`] and applied on finalize, so it stays `Copy`.
#[derive(Clone, Copy)]
enum Ds64Count {
    /// Count the frames as they are written and patch the field on finalize.
    Counted,
    /// A count the caller supplied up front. 64-bit, since that is what the
    /// `ds64` field holds; the 32-bit `fact` field range-checks it on write.
    Fixed(u64),
}

/// What to put in the `fact` chunk body, decided while the header is written
/// and never stored, which is what lets it borrow the caller's bytes.
enum FactWrite<'a> {
    /// A placeholder count now, patched with the frames written on finalize.
    Counted,
    /// A four-byte count the caller supplied up front.
    Fixed(u64),
    /// A whole body the caller supplied, written as it stands.
    Verbatim(&'a [u8]),
}

/// Decide whether to write a `fact` chunk, and with what.
///
/// The spec wants one for every format that is not plain integer PCM. `Auto` can
/// only honor that for a format this crate models, since counting frames means
/// knowing how many bytes one takes. For anything else `Auto` writes nothing:
/// `data_bytes / block_align` is a count of compressed blocks, and putting that
/// in a field defined as sample frames would be a plausible-looking lie. Callers
/// who know the real number say so with [`Fact::Samples`], and callers holding
/// a whole body from a file they read hand it back with [`Fact::Body`].
fn fact_body<'a>(fmt: &FmtChunk, fact: &'a Fact) -> Option<FactWrite<'a>> {
    match fact {
        Fact::None => None,
        Fact::Samples(samples) => Some(FactWrite::Fixed(*samples)),
        Fact::Body(body) => Some(FactWrite::Verbatim(body)),
        // The extensible form counts as non-PCM even when its subformat is PCM.
        Fact::Auto => match fmt.sample_format() {
            Some(format) if format.is_pcm() && !fmt.is_extensible() => None,
            Some(_) => Some(FactWrite::Counted),
            None => None,
        },
    }
}

/// Decide what goes in the `ds64` sample-count field.
///
/// RF64 has no `fact` chunk; the count lives in `ds64` instead, and that field
/// is there for every format, PCM included. The honesty rule from
/// [`fact_body`] still applies, though: `Auto` can only count frames for a
/// format the crate models, since `data_bytes / block_align` is a block count
/// for a compressed format. With nothing to say, the field keeps the zero it
/// was written with rather than a plausible-looking lie. [`Fact::Samples`] is
/// how a codec supplies the real number, at the full 64-bit width this field
/// has and the `fact` chunk's does not.
fn ds64_sample_count(fmt: &FmtChunk, fact: &Fact) -> Option<Ds64Count> {
    match fact {
        Fact::None => None,
        Fact::Samples(samples) => Some(Ds64Count::Fixed(*samples)),
        // A verbatim body has no `fact` chunk to live in here, but its first
        // four bytes are `dwSampleLength`, which is exactly what this field
        // holds. A body too short to have one says nothing, so the field keeps
        // its zero.
        Fact::Body(body) => body
            .get(..4)
            .map(|head| Ds64Count::Fixed(u64::from(u32::from_le_bytes(head.try_into().unwrap())))),
        Fact::Auto => fmt.sample_format().map(|_| Ds64Count::Counted),
    }
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
    fmt: &FmtChunk,
    fact: &Fact,
    leading: &[Chunk],
    container: Container,
    seekable: bool,
) -> Result<Layout> {
    for chunk in leading {
        check_extra_chunk(&chunk.id, chunk.data.len())?;
    }

    let mut pos: u64 = 12;
    let sizes = match container {
        Container::Riff => {
            header::write_riff_wave(inner, header::UNKNOWN_SIZE)?;

            pos += header::write_fmt_chunk(inner, fmt)?;

            // A `fact` chunk (sample-frame count) is required for every format
            // that is not plain WAVE_FORMAT_PCM. The count sits in the first
            // four bytes of the body, right after the 8-byte chunk header, and
            // `Auto` leaves it as a placeholder for `finalize` to patch with the
            // frames written.
            let mut fact_offset = None;
            // A counted body is a placeholder that `finalize` fills in, so it
            // needs a writer that comes back. A streaming one never does, and a
            // `fact` chunk stuck at the placeholder claims 4.29 billion frames
            // with nothing to mark it as unset, unlike the data size where that
            // value is the convention. So the chunk is omitted, the same answer
            // `Auto` gives for a format it cannot count.
            let body =
                fact_body(fmt, fact).filter(|body| seekable || !matches!(body, FactWrite::Counted));
            if let Some(body) = body {
                let offset = pos + 8;
                match body {
                    // Verbatim means verbatim: whatever the caller kept from the
                    // file it read, including any bytes a format defines after
                    // the count. Nothing here is patched on finalize.
                    FactWrite::Verbatim(bytes) => {
                        if u32::try_from(bytes.len()).is_err() {
                            return Err(WavError::InvalidSpec(format!(
                                "fact body of {} bytes does not fit in 32 bits",
                                bytes.len()
                            )));
                        }
                        pos += header::write_named_chunk(inner, b"fact", bytes)?;
                    }
                    FactWrite::Counted | FactWrite::Fixed(_) => {
                        let count = match body {
                            FactWrite::Counted => header::UNKNOWN_SIZE,
                            // The `fact` field is 32 bits wide. A larger count is
                            // a file that needs RF64, where `ds64` takes it.
                            FactWrite::Fixed(samples) => u32::try_from(samples).map_err(|_| {
                                WavError::InvalidSpec(format!(
                                    "sample count {samples} does not fit in the 32-bit fact \
                                         chunk field; write the file as RF64 instead"
                                ))
                            })?,
                            FactWrite::Verbatim(_) => unreachable!("handled above"),
                        };
                        pos += header::write_named_chunk(inner, b"fact", &count.to_le_bytes())?;
                        if matches!(body, FactWrite::Counted) {
                            fact_offset = Some(offset);
                        }
                    }
                }
            }

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

            pos += header::write_fmt_chunk(inner, fmt)?;

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
                sample_count: ds64_sample_count(fmt, fact),
            }
        }
    };

    Ok(Layout {
        header_len: pos,
        sizes,
    })
}

/// Anything that can describe the format for a [`WavWriter`].
///
/// A [`WavSpec`] means "I have audio in a format you model, build me a correct
/// header"; a [`FmtChunk`] means "I am the codec, write exactly these bytes".
/// Both are accepted wherever a writer is created.
pub trait IntoFmtChunk {
    /// Produce the `fmt ` chunk to write.
    fn into_fmt_chunk(self) -> Result<FmtChunk>;
}

impl IntoFmtChunk for FmtChunk {
    fn into_fmt_chunk(self) -> Result<FmtChunk> {
        Ok(self)
    }
}

impl IntoFmtChunk for &FmtChunk {
    fn into_fmt_chunk(self) -> Result<FmtChunk> {
        Ok(self.clone())
    }
}

impl IntoFmtChunk for WavSpec {
    fn into_fmt_chunk(self) -> Result<FmtChunk> {
        FmtChunk::for_spec(&self)
    }
}

impl IntoFmtChunk for &WavSpec {
    fn into_fmt_chunk(self) -> Result<FmtChunk> {
        FmtChunk::for_spec(self)
    }
}

/// Marks a builder that will write a plain RIFF file.
pub struct Riff;
/// Marks a builder that will write an RF64 file, which needs a seekable output.
pub struct Rf64;

/// Assembles a [`WavWriter`], for the cases the plain constructors do not cover:
/// a hand-built [`FmtChunk`], an explicit [`Fact`] count, or a combination of
/// RF64 and leading chunks.
///
/// The container form is a type parameter so that the one impossible
/// combination, streaming RF64, does not compile: `open_streaming` exists only
/// on the [`Riff`] form. RF64 records its sizes in a `ds64` chunk that has to be
/// patched afterwards, which a stream cannot do.
///
/// # Examples
///
/// Re-writing a file whose format this crate does not model, keeping the `fmt `
/// chunk and the sample count the original declared:
///
/// ```no_run
/// use std::fs::File;
/// use waveadapter::{Fact, WavReader, WavWriter};
///
/// let mut reader = WavReader::new(File::open("in.wav")?)?;
/// let mut audio = Vec::new();
/// reader.read_raw_interleaved(reader.frames(), &mut audio)?;
///
/// // The whole `fact` body, so any bytes a format keeps after the count
/// // come back too. A codec that computed the count instead says
/// // `Fact::Samples(n)`.
/// let fact = reader.params().fact.clone().map_or(Fact::None, Fact::Body);
/// let mut writer = WavWriter::builder(reader.params().fmt.clone())?
///     .fact(fact)
///     .open(File::create("out.wav")?)?;
/// writer.write_raw_interleaved(&audio)?;
/// writer.finalize()?;
/// # Ok::<(), waveadapter::WavError>(())
/// ```
pub struct WavWriterBuilder<'a, C = Riff> {
    fmt: FmtChunk,
    fact: Fact,
    leading: &'a [Chunk],
    container: PhantomData<C>,
}

impl<'a> WavWriterBuilder<'a, Riff> {
    /// Start building a writer from either a [`WavSpec`] or a [`FmtChunk`].
    ///
    /// A `WavSpec` says "I have audio in a format you model, build me a correct
    /// header" and the crate picks the header form, the subformat GUID and the
    /// `fact` chunk. A `FmtChunk` says "I am the codec, write these bytes", and
    /// is how a format waveadapter does not model gets written; pair it with
    /// [`WavParams::fmt`](crate::WavParams::fmt) to re-write a file unchanged.
    ///
    /// The container starts as plain RIFF; [`rf64`](Self::rf64) switches it. The
    /// output type is fixed only by the terminal [`open`](Self::open) /
    /// [`open_streaming`](Self::open_streaming) call.
    ///
    /// Returns [`WavError::InvalidSpec`] if the chunk declares zero channels,
    /// the one field a hand-built chunk can get wrong badly enough that this
    /// crate's own reader would refuse the file.
    pub fn new<T: IntoFmtChunk>(source: T) -> Result<Self> {
        let fmt = source.into_fmt_chunk()?;
        // A `WavSpec` was validated on the way in, a hand-built `FmtChunk` was
        // not, and only one field can make a file the parser refuses.
        fmt.validate()?;
        Ok(WavWriterBuilder {
            fmt,
            fact: Fact::Auto,
            leading: &[],
            container: PhantomData,
        })
    }
}

impl<'a, C> WavWriterBuilder<'a, C> {
    /// Set the metadata chunks to write between the `fmt ` chunk and the audio.
    pub fn chunks(mut self, leading: &'a [Chunk]) -> Self {
        self.leading = leading;
        self
    }

    /// Choose what goes in the `fact` chunk. See [`Fact`]; the default is
    /// [`Fact::Auto`].
    pub fn fact(mut self, fact: Fact) -> Self {
        self.fact = fact;
        self
    }

    fn build<W: Write>(
        self,
        mut inner: W,
        container: Container,
        seekable: bool,
    ) -> Result<WavWriter<W>> {
        let layout = write_header(
            &mut inner,
            &self.fmt,
            &self.fact,
            self.leading,
            container,
            seekable,
        )?;
        Ok(WavWriter {
            inner,
            fmt: self.fmt,
            data_bytes: 0,
            data_pos: 0,
            trailing_bytes: 0,
            seekable,
            layout,
        })
    }
}

impl<'a> WavWriterBuilder<'a, Riff> {
    /// Write the 64-bit RF64 form instead of plain RIFF, lifting the 4 GB limit.
    ///
    /// This drops `open_streaming`: RF64 sizes live in a `ds64` chunk that is
    /// only filled in on finalize, so the output has to be seekable. No `fact`
    /// chunk is written either; the [`Fact`] choice steers the `ds64`
    /// sample-frame count instead.
    pub fn rf64(self) -> WavWriterBuilder<'a, Rf64> {
        WavWriterBuilder {
            fmt: self.fmt,
            fact: self.fact,
            leading: self.leading,
            container: PhantomData,
        }
    }

    /// Create a seekable writer, its size fields patched by
    /// [`finalize`](WavWriter::finalize).
    pub fn open<W: Write + Seek>(self, inner: W) -> Result<WavWriter<W>> {
        self.build(inner, Container::Riff, true)
    }

    /// Create a streaming writer, leaving the size fields at [`u32::MAX`].
    /// Finish with [`into_inner`](WavWriter::into_inner).
    ///
    /// Nothing here is patched afterwards, so [`Fact::Auto`] writes no `fact`
    /// chunk at all rather than one stuck at its placeholder. Supply the count
    /// with [`Fact::Samples`] to get one.
    pub fn open_streaming<W: Write>(self, inner: W) -> Result<WavWriter<W>> {
        self.build(inner, Container::Riff, false)
    }
}

impl<'a> WavWriterBuilder<'a, Rf64> {
    /// Create the RF64 writer, its `ds64` sizes patched by
    /// [`finalize`](WavWriter::finalize).
    pub fn open<W: Write + Seek>(self, inner: W) -> Result<WavWriter<W>> {
        self.build(inner, Container::Rf64, true)
    }
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
/// format the spec considers non-PCM: float, A-law, mu-law, and the
/// `WAVEFORMATEXTENSIBLE` form. Only plain integer PCM omits it. See [`Fact`]
/// to control it, which a codec has to do for a compressed format because the
/// container cannot count frames it does not understand. Arbitrary extra chunks
/// can be written before the audio (leading chunks, via
/// [`WavWriter::new_with_chunks`] / [`WavWriter::new_streaming_with_chunks`]) or
/// after it (trailing chunks, via [`WavWriter::write_chunk`]), so a higher-level
/// library can attach metadata such as `LIST`/`INFO`.
///
/// # Picking a constructor
///
/// | Output | Constructor |
/// |---|---|
/// | Seekable, plain RIFF, up to 4 GB | [`new`](WavWriter::new) |
/// | Seekable, RF64, no size limit | [`new_rf64`](WavWriter::new_rf64) |
/// | Streaming, no seeking needed | [`new_streaming`](WavWriter::new_streaming) |
///
/// Each has a `_with_chunks` twin taking a `&[Chunk]` of metadata chunks to
/// write ahead of the audio. Finish a seekable writer with
/// [`finalize`](WavWriter::finalize), which patches the size fields and hands
/// back the inner writer, and a streaming one with
/// [`into_inner`](WavWriter::into_inner).
///
/// All six take a [`WavSpec`]. For anything else, including a format this crate
/// does not model, go through [`WavWriter::builder`], which accepts a
/// [`FmtChunk`] as readily as a `WavSpec` and adds the [`Fact`] choice. There is
/// no separate raw mode: [`write_float_buffer`](WavWriter::write_float_buffer)
/// works whenever the `fmt ` chunk describes a format the crate can convert,
/// however that chunk was built, and
/// [`write_raw_interleaved`](WavWriter::write_raw_interleaved) always works.
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
    /// The `fmt ` chunk this writer emitted. Everything about the format is read
    /// back off it, including whether the float write path is available, so the
    /// writer has no notion of being in a "typed" or "raw" mode.
    fmt: FmtChunk,
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

// The instantiation is arbitrary and unrelated to what the builder produces: an
// associated function whose return type does not mention `W` cannot be called on
// the generic `impl<W: Write> WavWriter<W>`, since nothing would fix `W`. Naming
// one concrete output type here is what lets `WavWriter::builder(spec)` be
// written without a turbofish.
impl WavWriter<Cursor<Vec<u8>>> {
    /// Start building a writer from either a [`WavSpec`] or a [`FmtChunk`].
    ///
    /// [`WavWriterBuilder::new`] is the same thing under the builder's own name.
    ///
    /// A `WavSpec` says "I have audio in a format you model, build me a correct
    /// header" and the crate picks the header form, the subformat GUID and the
    /// `fact` chunk. A `FmtChunk` says "I am the codec, write these bytes", and
    /// is how a format waveadapter does not model gets written; pair it with
    /// [`WavParams::fmt`](crate::WavParams::fmt) to re-write a file unchanged.
    ///
    /// The output type is only fixed by the terminal `open` / `open_streaming`
    /// call, so the writer this ends up building is not tied to the `W` this
    /// function happens to be reached through.
    pub fn builder<'a, T: IntoFmtChunk>(source: T) -> Result<WavWriterBuilder<'a, Riff>> {
        WavWriterBuilder::new(source)
    }
}

impl<W: Write> WavWriter<W> {
    /// Create a streaming writer.
    ///
    /// The RIFF and data size fields are set to [`u32::MAX`], matching what
    /// players expect from a stream of unknown length. The output does not need
    /// to be seekable. No `fact` chunk is written for a non-PCM format either,
    /// since counting the frames would need a patch on finalize; see
    /// [`Fact::Auto`].
    pub fn new_streaming(inner: W, spec: WavSpec) -> Result<Self> {
        WavWriter::builder(spec)?.open_streaming(inner)
    }

    /// Create a streaming writer that emits `leading` metadata chunks between the
    /// `fmt ` chunk and the audio data.
    ///
    /// Like [`new_streaming`](WavWriter::new_streaming), the size fields are left
    /// at [`u32::MAX`]. See [`Chunk`] for the chunk representation. Reserved ids
    /// (`RIFF`, `fmt `, `data`, `fact`, `ds64`) are rejected with
    /// [`WavError::InvalidSpec`](crate::WavError::InvalidSpec).
    pub fn new_streaming_with_chunks(inner: W, spec: WavSpec, leading: &[Chunk]) -> Result<Self> {
        WavWriter::builder(spec)?
            .chunks(leading)
            .open_streaming(inner)
    }

    /// The `fmt ` chunk this writer wrote.
    ///
    /// For a writer created from a [`WavSpec`] this is the chunk the crate built
    /// for it; for one created from a [`FmtChunk`] it is what the caller supplied.
    pub fn fmt(&self) -> &FmtChunk {
        &self.fmt
    }

    /// The sample format the float write path will use, or `None` if the `fmt `
    /// chunk describes a format this crate does not model. In that case only
    /// [`write_raw_interleaved`](WavWriter::write_raw_interleaved) is available.
    pub fn sample_format(&self) -> Option<SampleFormat> {
        self.fmt.sample_format()
    }

    /// The spec equivalent to this writer's `fmt ` chunk, or `None` if the format
    /// is not one this crate models.
    pub fn spec(&self) -> Option<WavSpec> {
        Some(WavSpec {
            channels: self.fmt.channels as usize,
            sample_rate: self.fmt.sample_rate as usize,
            sample_format: self.fmt.sample_format()?,
            channel_mask: self.fmt.channel_mask(),
        })
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
    /// Zero if the frame size is unknown (a [`FmtChunk`] with a zero block
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
    /// 64-bit fields, so both return `None`, as does a [`FmtChunk`] with a
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
        let frame_bytes = self.fmt.frame_bytes() as u64;
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
            .checked_div(self.fmt.frame_bytes() as u64)
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
        // Whether the float path works is a property of the `fmt ` chunk, not of
        // which constructor was used: any chunk describing a format this crate
        // models can be written from floats, however it was built.
        let sample_format = self.fmt.sample_format().ok_or_else(|| {
            WavError::UnsupportedFormat(format!(
                "format code {} is not one this crate can convert to; \
                 use write_raw_interleaved instead",
                self.fmt.format_code
            ))
        })?;
        let frames = src.frames();
        let channels = src.channels();
        // In 64-bit arithmetic: a buffer big enough to overflow a `usize` here
        // is one the capacity check should reject, not wrap around.
        let byte_count = (frames as u64)
            .saturating_mul(channels as u64)
            .saturating_mul(sample_format.bytes_per_sample() as u64);
        self.check_capacity(byte_count)?;
        let mut clipped = 0;
        with_sample_type!(sample_format, S, {
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
    /// RIFF spec requires before a following chunk. Reserved ids (`RIFF`, `fmt `,
    /// `data`, `fact`, `ds64`) are rejected with
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
    /// representation. Reserved ids (`RIFF`, `fmt `, `data`, `fact`, `ds64`) are
    /// rejected with [`WavError::InvalidSpec`](crate::WavError::InvalidSpec).
    pub fn new_with_chunks(inner: W, spec: WavSpec, leading: &[Chunk]) -> Result<Self> {
        WavWriter::builder(spec)?.chunks(leading).open(inner)
    }

    /// Create a seekable RF64 writer, for files that may exceed the 4 GB limit of
    /// plain RIFF.
    ///
    /// The file is written with the `RF64` form id and a `ds64` chunk; the 64-bit
    /// size and sample-count fields are patched by [`finalize`](WavWriter::finalize).
    /// Unlike a plain RIFF writer there is no 4 GB ceiling, so audio writes never
    /// fail on size. RF64 requires a seekable output.
    pub fn new_rf64(inner: W, spec: WavSpec) -> Result<Self> {
        WavWriter::builder(spec)?.rf64().open(inner)
    }

    /// Create a seekable RF64 writer that emits `leading` metadata chunks between
    /// the `fmt ` chunk and the audio data.
    pub fn new_rf64_with_chunks(inner: W, spec: WavSpec, leading: &[Chunk]) -> Result<Self> {
        WavWriter::builder(spec)?.rf64().chunks(leading).open(inner)
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
        let frame_bytes = self.fmt.frame_bytes() as u64;
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
        let frame_bytes = self.fmt.frame_bytes() as u64;
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
                sample_count,
            } => {
                // The 32-bit RIFF and data size fields keep their 0xFFFFFFFF
                // markers; the real 64-bit values go into the ds64 chunk.
                self.inner.seek(SeekFrom::Start(riff_size_offset))?;
                self.inner.write_all(&riff_size.to_le_bytes())?;
                self.inner.seek(SeekFrom::Start(data_size_offset))?;
                self.inner.write_all(&self.data_bytes.to_le_bytes())?;
                // Nothing to patch when the count is unknown: the field was
                // written as zero and stays there.
                let count = match sample_count {
                    Some(Ds64Count::Counted) => Some(frames),
                    Some(Ds64Count::Fixed(samples)) => Some(samples),
                    None => None,
                };
                if let Some(count) = count {
                    self.inner.seek(SeekFrom::Start(sample_count_offset))?;
                    self.inner.write_all(&count.to_le_bytes())?;
                }
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
    /// patched), or if the frame size is unknown (a [`FmtChunk`] with a zero block
    /// alignment).
    pub fn truncate_to_frame(&mut self, frame: usize) -> Result<()> {
        self.ensure_data_open()?;
        if !self.seekable {
            return Err(WavError::InvalidSpec(
                "cannot truncate a streaming writer".to_string(),
            ));
        }
        let frame_bytes = self.fmt.frame_bytes() as u64;
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
