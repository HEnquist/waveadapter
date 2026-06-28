# waveadapter

Reading and writing wav files into and out of [audioadapter] buffers.

This crate handles the wav container (header parsing and writing, borrowed and generalized from
[CamillaDSP]) and bridges it to the byte and sample handling in the audioadapter family of crates.
Audio data can be read into and written from any `Adapter` / `AdapterMut` buffer as scaled
floating point samples, or moved as raw interleaved bytes for the caller to wrap with the
audioadapter adapters directly.

## Features

- **One-call file helpers**: `read_wav_file(path)` decodes a whole file into floats plus its sample
  rate, and `write_wav_file(path, buffer, rate, format)` writes a buffer out, for when you do not
  need streaming, chunks or random access.
- **audioadapter integration**: read into and write from any `Adapter` / `AdapterMut` buffer
  (interleaved or planar, owned or borrowed), with on-the-fly conversion to and from `f32`/`f64`
  scaled to -1.0..1.0. The write path reports how many samples were clipped.
- **Raw byte passthrough**: move the interleaved sample bytes untouched, to wrap with the
  audioadapter byte/number adapters yourself, or to handle formats this crate does not model.
- **Wide format coverage**: 16-, 24- (both 3-byte packed and 4-byte left-justified), and 32-bit
  integer PCM, plus 32- and 64-bit IEEE float.
- **Any container, even unmodeled formats**: 8-bit PCM, A-law/µ-law, ADPCM and exotic
  `WAVEFORMATEXTENSIBLE` subtypes round-trip as raw bytes, so the crate is a complete WAV container
  library, not just the formats it can decode.
- **Plain and extensible headers**: reads and writes both `WAVEFORMAT`/`WAVEFORMATEX` and
  `WAVEFORMATEXTENSIBLE`, picking the minimal form automatically. The `dwChannelMask` speaker layout
  is read and written.
- **Streaming or seekable**: write to a seekable file (sizes patched on finalize) or straight to a
  pipe with no seeking (`u32::MAX` sizes). Reading handles unknown-length streams, stopping cleanly
  at end of file.
- **Random access**: seek to any frame for reading or writing on a seekable stream.
- **RF64 / BW64 (>4 GB)**: reads both forms, writes RF64, for files past the 4 GB RIFF limit.
- **Chunk passthrough with typed metadata**: every non-audio chunk round-trips verbatim (leading or
  trailing), with a thin typed layer for `LIST`/`INFO` tags, the `bext` Broadcast Audio Extension,
  and `cue ` markers with their `LIST`/`adtl` labels.
- **Robust parsing**: tolerates junk, padding and out-of-order chunks.

## Supported sample formats

Wav data is always little-endian, so only little-endian formats are listed. The names mirror the
audioadapter byte-wrapper sample types.

| `SampleFormat` | Wav format | Bits | Bytes |
| -------------- | ---------- | ---- | ----- |
| `I16`          | PCM        | 16   | 2     |
| `I24_3`        | PCM        | 24   | 3 (packed) |
| `I24_4`        | PCM        | 24   | 4 (left justified) |
| `I32`          | PCM        | 32   | 4     |
| `F32`          | IEEE float | 32   | 4     |
| `F64`          | IEEE float | 64   | 8     |

Both plain `WAVEFORMAT`/`WAVEFORMATEX` and extended `WAVEFORMATEXTENSIBLE` headers are parsed.

When writing, the minimal 16-byte `fmt ` chunk is used by default, and the 40-byte
`WAVEFORMATEXTENSIBLE` form is used in two cases:

- **`I24_4`** (24 valid bits in a 4-byte container) is ambiguous as plain PCM, because the block
  alignment implies a 32-bit sample. It is written as a strict-spec `WAVEFORMATEXTENSIBLE` header,
  with the 32-bit container size in `wBitsPerSample` and 24 in `wValidBitsPerSample`. On read, both
  that strict form and the lenient form (`wBitsPerSample` = 24) are accepted.
- **More than two channels**, following the spec recommendation to use the extensible form once the
  layout is past plain mono/stereo.
- **A non-zero channel mask** in the `WavSpec`, since the mask can only live in the extensible form.

`dwChannelMask` defaults to `0` (unspecified). Set `WavSpec::channel_mask` to write a speaker
layout; a non-zero mask must have exactly one bit set per channel. The crate stores the mask but
does not interpret it, and exposes it on read as `WavParams::channel_mask` (`None` for a plain
header that carries no mask).

## One-call helpers

For the common "just read/write a file" cases, two free functions wrap the reader and writer:

```rust no_run
use waveadapter::{SampleFormat, read_wav_file, write_wav_file};

// Decode a whole file into f32 samples plus the sample rate.
let audio = read_wav_file::<f32, _>("input.wav")?;
println!("{} ch, {} Hz, {} frames", audio.channels(), audio.sample_rate, audio.frames());

// Write a buffer out as 16-bit PCM.
let clipped = write_wav_file("output.wav", &audio.samples, audio.sample_rate, SampleFormat::I16)?;
# let _ = clipped;
# Ok::<(), waveadapter::WavError>(())
```

Reach for `WavReader` / `WavWriter` below when you need streaming, metadata chunks, RF64, raw
formats or random access.

## Reading

```rust no_run
use std::fs::File;
use waveadapter::WavReader;

let mut reader = WavReader::new(File::open("input.wav")?)?;
println!("{} ch, {} Hz, {:?}", reader.channels(), reader.sample_rate(), reader.sample_format());

// Read everything into an owned interleaved float buffer.
let buffer = reader.read_all_to_float::<f32>()?;
# Ok::<(), waveadapter::WavError>(())
```

`read_into_float` fills an existing `AdapterMut` buffer block by block, and `read_raw_interleaved`
hands back the untouched bytes for wrapping with the audioadapter byte/number adapters.
`seek_to_frame` repositions the reader for random access (the reader is always seekable).

## Writing

Two modes are available:

- **Seekable** (`WavWriter::new`): the size fields start as placeholders and are patched with the
  real values by `finalize`, producing a standard-compliant file.
- **Streaming** (`WavWriter::new_streaming`): the size fields are set to `u32::MAX` up front and
  never updated, for pipes and other non-seekable outputs. Finish with `into_inner`.

A seekable writer also supports random access via `seek_to_frame`, to overwrite already-written
audio without shrinking the file.

```rust no_run
use std::fs::File;
use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::{SampleFormat, WavSpec, WavWriter};

let data = InterleavedOwned::<f32>::new(0.0, 2, 128);
let spec = WavSpec::new(2, 44100, SampleFormat::I32);

let mut writer = WavWriter::new(File::create("output.wav")?, spec)?;
let clipped = writer.write_float_buffer(&data)?;
writer.finalize()?;
# Ok::<(), waveadapter::WavError>(())
```

## Examples

The `examples/` directory shows both the float and raw paths for reading and writing. Run any of
them with `cargo run --example <name> -- <file.wav>`.

- **`read_float`** — read a file into an `f32` buffer (converting from whatever the on-disk format
  is) and report a peak level per channel.
- **`read_raw`** — read the untouched interleaved sample bytes and decode the first frame by hand
  with the matching audioadapter-sample byte type.
- **`write_float`** — write a file from an `f32` buffer, letting the writer convert and clip into
  the target format.
- **`write_raw`** — write a file from raw interleaved sample bytes in streaming mode (size fields
  set to `u32::MAX`, no seeking required).

## License

Licensed under either of MIT or Apache-2.0 at your option.

[audioadapter]: https://github.com/HEnquist/audioadapter-rs
[CamillaDSP]: https://github.com/HEnquist/camilladsp
