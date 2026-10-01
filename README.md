# waveadapter

Reading and writing wav files into and out of [audioadapter] buffers.

This crate handles the wav container (header parsing and writing, borrowed and generalized from
[CamillaDSP]) and bridges it to the byte and sample handling in the audioadapter family of crates.
Audio data can be read into and written from any `Adapter` / `AdapterMut` buffer as scaled
floating point samples, or moved as raw interleaved bytes for the caller to wrap with the
audioadapter adapters directly.

## Features

- **One-call file helpers**: `read_wav_file(path)` decodes a whole file into floats plus its sample
  rate, `write_wav_file(path, buffer, rate, format)` writes a buffer out, and
  `write_wav_file_raw(path, bytes, format)` writes bytes that are already encoded, for when you do
  not need streaming, chunks or random access.
- **audioadapter integration**: read into and write from any `Adapter` / `AdapterMut` buffer
  (interleaved or planar, owned or borrowed), with on-the-fly conversion to and from `f32`/`f64`
  scaled to -1.0..1.0. The write path reports how many samples were clipped by the integer
  formats (the float formats keep their headroom and never clip).
- **Raw byte passthrough**: move the interleaved sample bytes untouched, to wrap with the
  audioadapter byte/number adapters yourself, or to handle formats this crate does not model.
- **Wide format coverage**: 8- (unsigned), 16-, 24- (both 3-byte packed and 4-byte left-justified),
  and 32-bit integer PCM, 32- and 64-bit IEEE float, and the G.711 companded telephony formats
  A-law and mu-law.
- **Any container, even unmodeled formats**: ADPCM, GSM and exotic `WAVEFORMATEXTENSIBLE` subtypes
  round-trip *byte for byte*, `fmt ` extension and `fact` count included, so the crate is a
  complete WAV container library, not just the formats it can decode. See
  [building a codec on top](#building-a-codec-on-top).
- **Plain and extensible headers**: reads and writes both `WAVEFORMAT`/`WAVEFORMATEX` and
  `WAVEFORMATEXTENSIBLE`, picking the minimal form automatically. The `dwChannelMask` speaker layout
  is read and written.
- **Streaming or seekable**: write to a seekable file (sizes patched on finalize) or straight to a
  pipe with no seeking. Reading handles unknown-length streams, stopping cleanly at end of file.
- **Crash tolerant writing**: an interrupted file stays readable, and `update_header` keeps the
  sizes current during a long recording.
- **Random access**: seek to any frame for reading or writing on a seekable stream.
- **RF64 / BW64 (>4 GB)**: reads both forms, writes RF64, for files past the 4 GB RIFF limit.
- **Chunk passthrough with typed metadata**: every non-audio chunk round-trips verbatim (leading or
  trailing), with a thin typed layer for `LIST`/`INFO` tags, the `bext` Broadcast Audio Extension,
  `cue ` markers with their `LIST`/`adtl` labels, and `smpl` sampler loop points.
- **Robust parsing**: tolerates junk, padding and out-of-order chunks.

## Supported sample formats

Wav data is always little-endian, so only little-endian formats are listed. The names mirror the
audioadapter byte-wrapper sample types.

| `SampleFormat` | Wav format | Bits | Bytes |
| -------------- | ---------- | ---- | ----- |
| `U8`           | PCM (unsigned) | 8 | 1  |
| `I16`          | PCM        | 16   | 2     |
| `I24_3`        | PCM        | 24   | 3 (packed) |
| `I24_4`        | PCM        | 24   | 4 (left justified) |
| `I32`          | PCM        | 32   | 4     |
| `F32`          | IEEE float | 32   | 4     |
| `F64`          | IEEE float | 64   | 8     |
| `ALAW`         | G.711 A-law  | 8  | 1     |
| `MULAW`        | G.711 mu-law | 8  | 1     |

`U8` is the odd one out: wav 8-bit PCM is unsigned and centered at 128, while every deeper integer
depth is signed. The conversion handles that, so `0` reads back as -1.0 and `255` as +1.0.

`ALAW` and `MULAW` are the companded telephony formats of ITU-T G.711. They also store one byte per
sample, but space the quantization steps logarithmically, so they carry roughly 13 and 14 bits of
dynamic range rather than the 8 of `U8`. Being lossy, they quantize on write: a value written and
read back does not come out unchanged, though re-encoding what was read changes nothing further.
Mu-law is also written μ-law, u-law or ulaw elsewhere; the spelling here follows the wav format tag
`WAVE_FORMAT_MULAW`.

Both plain `WAVEFORMAT`/`WAVEFORMATEX` and extended `WAVEFORMATEXTENSIBLE` headers are parsed.

When writing, plain integer PCM gets the minimal 16-byte `fmt ` chunk. Every other format (float,
A-law, mu-law) gets the 18-byte `WAVEFORMATEX` form with a zero `cbSize`, which the spec requires
for non-PCM data, along with a `fact` chunk carrying the frame count. The 40-byte
`WAVEFORMATEXTENSIBLE` form is used in three cases:

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

For the common "just read/write a file" cases, three free functions wrap the reader and writer:

```rust no_run
use waveadapter::{SampleFormat, WavSpec, read_wav_file, write_wav_file, write_wav_file_raw};

// Decode a whole file into f32 samples plus the sample rate.
let audio = read_wav_file::<f32, _>("input.wav")?;
println!("{} ch, {} Hz, {} frames", audio.channels(), audio.sample_rate, audio.frames());

// Write a buffer out as 16-bit PCM.
let clipped = write_wav_file("output.wav", &audio.samples, audio.sample_rate, SampleFormat::I16)?;
# let _ = clipped;

// Or write bytes that are already in the target format, checked to be whole frames.
let bytes: Vec<u8> = vec![0; 4 * 1024];
write_wav_file_raw("raw.wav", &bytes, WavSpec::new(2, 44100, SampleFormat::I16))?;
# Ok::<(), waveadapter::WavError>(())
```

`write_wav_file_raw` takes a `FmtChunk` just as happily as a `WavSpec`, so it also writes a format
this crate does not model, straight from the chunk a file was read with.

Reach for `WavReader` / `WavWriter` below when you need streaming, metadata chunks, RF64 or random
access.

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
hands back the untouched bytes for wrapping with the audioadapter byte/number adapters, or
`read_raw_all` for all of them at once.
`seek_to_frame` repositions the reader for random access (the reader is always seekable).

## Writing

Two modes are available:

- **Seekable** (`WavWriter::new`): the size fields start as placeholders and are patched with the
  real values by `finalize`, producing a standard-compliant file.
- **Streaming** (`WavWriter::new_streaming`): the size fields are never updated, for pipes and other
  non-seekable outputs. Finish with `into_inner`.

Both write the placeholder as `u32::MAX`, the "runs to the end of the file" marker, so a plain RIFF
file whose writer never reached `finalize` still reads back as the audio that made it to disk. RF64
has no such marker, so for long recordings call `update_header` now and then: it patches the sizes
in place and returns to the write position, leaving a valid file behind at every step.

A seekable writer also supports random access via `seek_to_frame`, to overwrite already-written
audio without shrinking the file. To drop the audio past the cursor instead, call `truncate`
(or `truncate_to_frame` for an explicit point). Shortening a stream is beyond what `Write + Seek`
can do, so these need the inner writer to implement the `Truncate` trait, which `File`,
`Cursor<Vec<u8>>` and `BufWriter` around either of them already do.

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

## Building a codec on top

waveadapter models the container, not every codec that can live inside one. For a format it has no
`SampleFormat` for (ADPCM, GSM, an exotic extensible subtype) it hands you the `fmt ` chunk and the
audio bytes, and takes them back unchanged, so you can put the codec on top without reimplementing
the container.

The `fmt ` chunk is a `FmtChunk`: the six core fields plus `extension`, the format-specific bytes
after `cbSize` that only the codec understands (MS ADPCM keeps its predictor coefficients there,
GSM its `wSamplesPerBlock`). Nothing is recomputed. `byte_rate` in particular is carried rather
than derived, because outside linear PCM it is not `block_align * sample_rate`: GSM 6.10 declares
1625, not 520000.

```rust no_run
use std::fs::File;
use waveadapter::{Fact, WavReader, WavWriter};

let mut reader = WavReader::new(File::open("in.wav")?)?;
if reader.sample_format().is_none() {
    println!("format {:#06x}, decode it yourself", reader.params().fmt.format_code);
}

// `frames()` counts whatever nBlockAlign describes, which for a block-compressed
// format is compressed blocks. The real frame count is in the fact chunk.
let audio = reader.read_raw_all()?;
let frames = reader.params().sample_count();

// Write it back: same fmt chunk, same fact body, byte for byte.
let fact = reader.params().fact.clone().map_or(Fact::None, Fact::Body);
let mut writer = WavWriter::builder(reader.params().fmt.clone())?
    .fact(fact)
    .open(File::create("out.wav")?)?;
writer.write_raw_interleaved(&audio)?;
writer.finalize()?;
# Ok::<(), waveadapter::WavError>(())
```

Metadata chunks come back split by which side of the audio they were on, `chunks_before` and
`chunks_after`, so re-writing a file does not move a trailing `cue ` or `id3 ` to the front. The
chunks waveadapter writes itself (`fmt `, `data`, `fact`, `ds64`) are not in either list, so
passing them straight back is always safe.

## Examples

The `examples/` directory shows both the float and raw paths for reading and writing. Run any of
them with `cargo run --example <name> -- <file.wav>`.

- **`read_float`** — read a file into an `f32` buffer (converting from whatever the on-disk format
  is) and report a peak level per channel.
- **`read_raw`** — read a file waveadapter cannot decode (IMA ADPCM), decode the blocks with a
  codec crate, and report peak and RMS through audioadapter's stats.
- **`write_float`** — write a file from an `f32` buffer, letting the writer convert and clip into
  the target format.
- **`write_raw`** — encode a sine to IMA ADPCM with that same codec crate and write it from a
  hand-built `fmt ` chunk, with the frame count supplied for the `fact` chunk.

The two raw path examples pair waveadapter with
[audio-codec-algorithms](https://crates.io/crates/audio-codec-algorithms), a dev-dependency, since
this crate does no codec work of its own. That seam is what they demonstrate.

## License

Licensed under either of MIT or Apache-2.0 at your option.

[audioadapter]: https://github.com/HEnquist/audioadapter-rs
[CamillaDSP]: https://github.com/HEnquist/camilladsp
