# waveadapter

Reading and writing wav files into and out of [audioadapter] buffers.

This crate handles the wav container (header parsing and writing, borrowed and generalized from
[CamillaDSP]) and bridges it to the byte and sample handling in the audioadapter family of crates.
Audio data can be read into and written from any `Adapter` / `AdapterMut` buffer as scaled
floating point samples, or moved as raw interleaved bytes for the caller to wrap with the
audioadapter adapters directly.

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
