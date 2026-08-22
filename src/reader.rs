//! Reading audio data from a wav file.

use std::io::{ErrorKind, Read, Seek, SeekFrom};

use audioadapter::AdapterMut;
use audioadapter_buffers::owned::InterleavedOwned;
use audioadapter_sample::readwrite::ReadSamples;
use num_traits::float::FloatCore;
use num_traits::{ToPrimitive, Zero};

use crate::dispatch::with_sample_type;
use crate::error::{Result, WavError};
use crate::format::SampleFormat;
use crate::header::{WavParams, read_wav_header};

/// A reader for wav files.
///
/// It parses the header on construction and then reads the interleaved audio
/// data, either converted to floating point samples in an
/// [`AdapterMut`](audioadapter::AdapterMut) buffer, or as raw interleaved bytes.
pub struct WavReader<R: Read + Seek> {
    inner: R,
    params: WavParams,
    /// Total number of frames declared in the header. For streaming files the
    /// declared length is [`u32::MAX`] bytes, so this can be larger than what is
    /// actually present; reading stops cleanly at end of file in that case.
    total_frames: usize,
    frames_pos: usize,
}

impl<R: Read + Seek> WavReader<R> {
    /// Create a new reader, parsing the wav header and positioning the stream at
    /// the start of the audio data.
    pub fn new(mut inner: R) -> Result<Self> {
        let params = read_wav_header(&mut inner)?;
        inner.seek(SeekFrom::Start(params.data_offset as u64))?;
        let frame_bytes = params.frame_bytes();
        // The frame count stays a `usize`: it indexes buffers in memory, so a
        // declared length past what this target can address is clamped rather
        // than overflowing. A zero frame size (an unmodelled format declaring a
        // zero `nBlockAlign`) means no framing exists at all, so the count is
        // zero rather than the clamp, since `usize::MAX` frames would claim the
        // file is enormous when nothing about it can be framed at all.
        let total_frames = if frame_bytes == 0 {
            0
        } else {
            usize::try_from(params.data_length / frame_bytes as u64).unwrap_or(usize::MAX)
        };
        Ok(Self {
            inner,
            params,
            total_frames,
            frames_pos: 0,
        })
    }

    /// The parsed header parameters, including any non-audio chunks.
    pub fn params(&self) -> &WavParams {
        &self.params
    }

    /// The sample format of the audio data, or `None` if the file uses a valid
    /// but unsupported format. In that case the audio can still be read with
    /// [`read_raw_interleaved`](WavReader::read_raw_interleaved); the float read
    /// methods return [`WavError::UnsupportedFormat`](crate::WavError::UnsupportedFormat).
    pub fn sample_format(&self) -> Option<SampleFormat> {
        self.params.sample_format()
    }

    /// The number of channels.
    pub fn channels(&self) -> usize {
        self.params.channels()
    }

    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> usize {
        self.params.sample_rate()
    }

    /// The total number of frames declared in the header.
    ///
    /// This is the declared data length divided by
    /// [`WavParams::frame_bytes`](crate::WavParams::frame_bytes), so for a format
    /// this crate does not interpret it counts whatever `nBlockAlign` describes
    /// rather than audio frames. That makes it zero when such a file declares a
    /// zero `nBlockAlign`: nothing about it can be framed, and the framed read
    /// and seek methods reject it.
    pub fn frames(&self) -> usize {
        self.total_frames
    }

    /// The number of frames read so far.
    pub fn position(&self) -> usize {
        self.frames_pos
    }

    /// The number of frames remaining according to the declared length.
    pub fn remaining(&self) -> usize {
        self.total_frames.saturating_sub(self.frames_pos)
    }

    /// Seek to a frame for random-access reading.
    ///
    /// Positions the stream at the start of frame `frame`, so the next read
    /// begins there. The target is clamped to [`frames`](WavReader::frames), so
    /// seeking past the end leaves the reader at the end with no frames
    /// remaining. Returns [`WavError::InvalidHeader`](crate::WavError::InvalidHeader)
    /// if the frame size is unknown (block alignment is zero).
    ///
    /// For a format this crate does not interpret (`sample_format` is `None`) the
    /// unit follows [`WavParams::frame_bytes`](crate::WavParams::frame_bytes), so it
    /// is whatever `nBlockAlign` describes: a compressed block for ADPCM and GSM, a
    /// single byte for MPEG Layer 3. The seek lands on a multiple of that, which for
    /// a stateful codec is not generally a point a decoder can start from.
    pub fn seek_to_frame(&mut self, frame: usize) -> Result<()> {
        let frame_bytes = self.params.frame_bytes();
        if frame_bytes == 0 {
            return Err(WavError::InvalidHeader(
                "cannot seek: block alignment is zero".to_string(),
            ));
        }
        let frame = frame.min(self.total_frames);
        // In 64-bit arithmetic throughout: `total_frames` is clamped to
        // `usize::MAX` for a file declaring more frames than this target can
        // index, and multiplying that back out overflows a `usize`.
        let offset = self
            .params
            .data_offset
            .saturating_add((frame as u64).saturating_mul(frame_bytes as u64));
        self.inner.seek(SeekFrom::Start(offset))?;
        self.frames_pos = frame;
        Ok(())
    }

    /// Read audio data into a floating point buffer, converting on the fly.
    ///
    /// Reads up to `target.frames()` frames, scaling each sample to the range
    /// -1.0..1.0. Samples are written into the matching channel and frame of
    /// `target`; channels of the file beyond `target.channels()` are read and
    /// discarded. Reading stops early and cleanly if the end of the data is
    /// reached at a frame boundary.
    ///
    /// Returns the number of frames actually read.
    pub fn read_into_float<T>(&mut self, target: &mut dyn AdapterMut<T>) -> Result<usize>
    where
        T: FloatCore + ToPrimitive,
    {
        let format = self.require_sample_format()?;
        let file_channels = self.params.channels();
        let want = target.frames().min(self.remaining());
        let mut produced = 0;
        with_sample_type!(format, S, {
            'outer: for frame in 0..want {
                for ch in 0..file_channels {
                    match self.inner.read_converted::<S, T>() {
                        Ok(value) => {
                            target.write_sample(ch, frame, &value);
                        }
                        Err(err) if err.kind() == ErrorKind::UnexpectedEof && ch == 0 => {
                            break 'outer;
                        }
                        Err(err) => return Err(err.into()),
                    }
                }
                produced += 1;
            }
        });
        self.frames_pos += produced;
        Ok(produced)
    }

    /// Read all remaining audio data into a freshly allocated interleaved
    /// floating point buffer.
    ///
    /// This works for streaming files with an unknown declared length, since it
    /// reads until the end of the data.
    pub fn read_all_to_float<T>(&mut self) -> Result<InterleavedOwned<T>>
    where
        T: FloatCore + ToPrimitive + Zero,
    {
        let format = self.require_sample_format()?;
        let channels = self.params.channels();
        let want = self.remaining();
        let mut data: Vec<T> = Vec::new();
        with_sample_type!(format, S, {
            'outer: for _ in 0..want {
                for ch in 0..channels {
                    match self.inner.read_converted::<S, T>() {
                        Ok(value) => data.push(value),
                        Err(err) if err.kind() == ErrorKind::UnexpectedEof && ch == 0 => {
                            // Drop any partial frame at the very end.
                            data.truncate((data.len() / channels) * channels);
                            break 'outer;
                        }
                        Err(err) => return Err(err.into()),
                    }
                }
            }
        });
        let frames = data.len().checked_div(channels).unwrap_or(0);
        self.frames_pos += frames;
        InterleavedOwned::new_from(data, channels, frames)
            .map_err(|err| WavError::InvalidHeader(format!("buffer size mismatch: {err:?}")))
    }

    /// Read up to `frames` frames of raw interleaved bytes, appending them to
    /// `buf`.
    ///
    /// The bytes are exactly as stored in the file, so each frame is
    /// [`WavParams::frame_bytes`] bytes. This works for any file, including ones
    /// whose format is unsupported by the float path (`sample_format` is `None`),
    /// which is the way to read ADPCM or otherwise unmodeled audio. This is also
    /// the entry point for callers who want to wrap the data with the audioadapter
    /// byte or number adapters themselves. Returns the number of frames read.
    ///
    /// # Examples
    ///
    /// Wrapping the bytes in an adapter that converts to `f32` on access, rather
    /// than copying them into a float buffer up front. The sample type has to
    /// match the format of the file, so check
    /// [`sample_format`](WavReader::sample_format) before picking it:
    ///
    /// ```
    /// # use audioadapter_buffers::owned::InterleavedOwned;
    /// # use waveadapter::{WavSpec, WavWriter};
    /// use std::io::Cursor;
    /// use audioadapter::Adapter;
    /// use audioadapter_buffers::number_to_float::InterleavedNumbers;
    /// use audioadapter_buffers::sample::I16_LE;
    /// use waveadapter::{SampleFormat, WavReader};
    ///
    /// # let spec = WavSpec::new(2, 44100, SampleFormat::I16);
    /// # let mut writer = WavWriter::new(Cursor::new(Vec::new()), spec)?;
    /// # writer.write_float_buffer(&InterleavedOwned::<f32>::new(0.5, 2, 128))?;
    /// # let wav_bytes = writer.finalize()?.into_inner();
    /// // A whole wav file held in memory; a File reads exactly the same way.
    /// let mut reader = WavReader::new(Cursor::new(wav_bytes))?;
    /// assert_eq!(reader.sample_format(), Some(SampleFormat::I16));
    /// let channels = reader.channels();
    ///
    /// let mut bytes = Vec::new();
    /// let frames = reader.read_raw_interleaved(1024, &mut bytes)?;
    ///
    /// let audio = InterleavedNumbers::<&[I16_LE], f32>::new_from_bytes(&bytes, channels, frames)
    ///     .expect("the buffer holds exactly this many frames");
    /// let first: f32 = audio.read_sample(0, 0).unwrap();
    /// assert!((first - 0.5).abs() < 1e-4);
    /// # Ok::<(), waveadapter::WavError>(())
    /// ```
    ///
    /// A format this crate does not model has no matching sample type, so the
    /// bytes are as far as this goes; decode them with whatever does understand
    /// them, using the raw `fmt ` fields of [`params`](WavReader::params) to make
    /// sense of the framing.
    pub fn read_raw_interleaved(&mut self, frames: usize, buf: &mut Vec<u8>) -> Result<usize> {
        let frame_bytes = self.params.frame_bytes();
        if frame_bytes == 0 {
            return Err(WavError::InvalidHeader(
                "cannot read raw frames: block alignment is zero".to_string(),
            ));
        }
        let want = frames.min(self.remaining());
        let start = buf.len();
        buf.resize(start + want * frame_bytes, 0);
        let mut filled = 0;
        while filled < want * frame_bytes {
            match self.inner.read(&mut buf[start + filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(ref err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => return Err(err.into()),
            }
        }
        // Keep only whole frames.
        let frames_read = filled / frame_bytes;
        buf.truncate(start + frames_read * frame_bytes);
        self.frames_pos += frames_read;
        Ok(frames_read)
    }

    /// Read all remaining raw interleaved bytes into a freshly allocated buffer.
    ///
    /// The bulk counterpart to
    /// [`read_raw_interleaved`](WavReader::read_raw_interleaved), for handing a
    /// whole file to a decoder in one go: a format this crate does not model has
    /// to be read this way, and the frame count needed to size a
    /// `read_raw_interleaved` call is exactly what such a file does not state.
    /// Reading stops at the declared end of the data or at the end of the file,
    /// whichever comes first, so it also works for a streaming file whose declared
    /// length is the [`u32::MAX`] placeholder. Only whole frames are kept.
    ///
    /// The buffer grows as the data is read rather than being sized from the
    /// declared length up front, so a header claiming far more data than the file
    /// holds costs nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// # use audioadapter_buffers::owned::InterleavedOwned;
    /// # use waveadapter::{SampleFormat, WavSpec, WavWriter};
    /// # use std::io::Cursor;
    /// use waveadapter::WavReader;
    ///
    /// # let spec = WavSpec::new(2, 44100, SampleFormat::I16);
    /// # let mut writer = WavWriter::new(Cursor::new(Vec::new()), spec)?;
    /// # writer.write_float_buffer(&InterleavedOwned::<f32>::new(0.5, 2, 128))?;
    /// # let wav_bytes = writer.finalize()?.into_inner();
    /// let mut reader = WavReader::new(Cursor::new(wav_bytes))?;
    /// let bytes = reader.read_raw_all()?;
    /// assert_eq!(bytes.len(), 128 * 2 * 2); // frames * channels * 2 bytes
    /// # Ok::<(), waveadapter::WavError>(())
    /// ```
    pub fn read_raw_all(&mut self) -> Result<Vec<u8>> {
        let frame_bytes = self.params.frame_bytes();
        if frame_bytes == 0 {
            return Err(WavError::InvalidHeader(
                "cannot read raw frames: block alignment is zero".to_string(),
            ));
        }
        // Bound the read by the declared length. For a streaming file that is the
        // placeholder, which saturates into "read to end of file".
        let limit = self.remaining().saturating_mul(frame_bytes) as u64;
        let mut buf = Vec::new();
        (&mut self.inner).take(limit).read_to_end(&mut buf)?;
        // Keep only whole frames.
        let frames_read = buf.len() / frame_bytes;
        buf.truncate(frames_read * frame_bytes);
        self.frames_pos += frames_read;
        Ok(buf)
    }

    /// Consume the reader and return the inner stream.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// The interpreted sample format, or an [`UnsupportedFormat`] error if the
    /// file uses a format the float path cannot handle.
    ///
    /// [`UnsupportedFormat`]: crate::WavError::UnsupportedFormat
    fn require_sample_format(&self) -> Result<SampleFormat> {
        self.params.sample_format().ok_or_else(|| {
            WavError::UnsupportedFormat(format!(
                "format code {}, {} bits per sample cannot be read as float; \
                 use read_raw_interleaved instead",
                self.params.fmt.format_code, self.params.fmt.bits_per_sample
            ))
        })
    }
}
