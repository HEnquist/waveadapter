//! Writing audio data to a wav file.

use std::io::{Seek, SeekFrom, Write};

use audioadapter::Adapter;
use audioadapter_sample::readwrite::WriteSamples;
use num_traits::ToPrimitive;
use num_traits::float::FloatCore;

use crate::dispatch::with_sample_type;
use crate::error::Result;
use crate::format::WavSpec;
use crate::header::{self, RIFF_SIZE_OFFSET, write_wav_header};

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
pub struct WavWriter<W: Write> {
    inner: W,
    spec: WavSpec,
    data_bytes: u64,
    seekable: bool,
}

impl<W: Write> WavWriter<W> {
    /// Create a streaming writer.
    ///
    /// The RIFF and data size fields are set to [`u32::MAX`], matching what
    /// players expect from a stream of unknown length. The output does not need
    /// to be seekable.
    pub fn new_streaming(mut inner: W, spec: WavSpec) -> Result<Self> {
        write_wav_header(
            &mut inner,
            spec.channels,
            spec.sample_format,
            spec.sample_rate,
            u32::MAX,
            u32::MAX,
        )?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            seekable: false,
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
    pub fn write_float_buffer<T>(&mut self, src: &dyn Adapter<T>) -> Result<usize>
    where
        T: FloatCore + ToPrimitive,
    {
        let frames = src.frames();
        let channels = src.channels();
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
        self.data_bytes += (frames * channels * self.spec.sample_format.bytes_per_sample()) as u64;
        Ok(clipped)
    }

    /// Write raw interleaved bytes directly to the data chunk.
    ///
    /// The caller is responsible for the bytes being in the file's sample format
    /// and channel interleaving.
    pub fn write_raw_interleaved(&mut self, data: &[u8]) -> Result<()> {
        self.inner.write_all(data)?;
        self.data_bytes += data.len() as u64;
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
    pub fn new(mut inner: W, spec: WavSpec) -> Result<Self> {
        write_wav_header(
            &mut inner,
            spec.channels,
            spec.sample_format,
            spec.sample_rate,
            0,
            0,
        )?;
        Ok(Self {
            inner,
            spec,
            data_bytes: 0,
            seekable: true,
        })
    }

    /// Patch the size fields with the real lengths and return the inner writer.
    ///
    /// For a streaming writer (created with [`WavWriter::new_streaming`]) the
    /// size fields are left at [`u32::MAX`]; only the inner writer is flushed and
    /// returned.
    pub fn finalize(mut self) -> Result<W> {
        self.inner.flush()?;
        if self.seekable {
            let format = self.spec.sample_format;
            let channels = self.spec.channels;
            let data_size = u32::try_from(self.data_bytes).unwrap_or(u32::MAX);
            let riff_size = u32::try_from(header::riff_size(channels, format, self.data_bytes))
                .unwrap_or(u32::MAX);
            self.inner.seek(SeekFrom::Start(RIFF_SIZE_OFFSET))?;
            self.inner.write_all(&riff_size.to_le_bytes())?;
            self.inner
                .seek(SeekFrom::Start(header::data_size_offset(channels, format)))?;
            self.inner.write_all(&data_size.to_le_bytes())?;
            self.inner.seek(SeekFrom::End(0))?;
        }
        Ok(self.inner)
    }
}
