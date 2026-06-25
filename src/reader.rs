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
        let total_frames = params.data_length.checked_div(frame_bytes).unwrap_or(0);
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
        self.params.sample_format
    }

    /// The number of channels.
    pub fn channels(&self) -> usize {
        self.params.channels
    }

    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> usize {
        self.params.sample_rate
    }

    /// The total number of frames declared in the header.
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
    pub fn seek_to_frame(&mut self, frame: usize) -> Result<()> {
        let frame_bytes = self.params.frame_bytes();
        if frame_bytes == 0 {
            return Err(WavError::InvalidHeader(
                "cannot seek: block alignment is zero".to_string(),
            ));
        }
        let frame = frame.min(self.total_frames);
        let offset = self.params.data_offset + frame * frame_bytes;
        self.inner.seek(SeekFrom::Start(offset as u64))?;
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
        let file_channels = self.params.channels;
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
        let channels = self.params.channels;
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
    /// which is the way to read 8-bit or otherwise unmodeled audio. This is also
    /// the entry point for callers who want to wrap the data with the audioadapter
    /// byte or number adapters themselves. Returns the number of frames read.
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

    /// Consume the reader and return the inner stream.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// The interpreted sample format, or an [`UnsupportedFormat`] error if the
    /// file uses a format the float path cannot handle.
    ///
    /// [`UnsupportedFormat`]: crate::WavError::UnsupportedFormat
    fn require_sample_format(&self) -> Result<SampleFormat> {
        self.params.sample_format.ok_or_else(|| {
            WavError::UnsupportedFormat(format!(
                "format code {}, {} bits per sample cannot be read as float; \
                 use read_raw_interleaved instead",
                self.params.format_code, self.params.bits_per_sample
            ))
        })
    }
}
