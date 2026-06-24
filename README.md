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

Both plain `WAVEFORMAT` and extended `WAVEFORMATEXTENSIBLE` headers are parsed.

## Reading

```rust
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

## Writing

Two modes are available:

- **Seekable** (`WavWriter::new`): the size fields start as placeholders and are patched with the
  real values by `finalize`, producing a standard-compliant file.
- **Streaming** (`WavWriter::new_streaming`): the size fields are set to `u32::MAX` up front and
  never updated, for pipes and other non-seekable outputs. Finish with `into_inner`.

```rust
use std::fs::File;
use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::{SampleFormat, WavSpec, WavWriter};

let data = InterleavedOwned::<f32>::new(0.0, 2, 128);
let spec = WavSpec { channels: 2, sample_rate: 44100, sample_format: SampleFormat::I32 };

let mut writer = WavWriter::new(File::create("output.wav")?, spec)?;
let clipped = writer.write_float_buffer(&data)?;
writer.finalize()?;
# Ok::<(), waveadapter::WavError>(())
```

## License

Licensed under either of MIT or Apache-2.0 at your option.

[audioadapter]: https://github.com/HEnquist/audioadapter-rs
[CamillaDSP]: https://github.com/HEnquist/camilladsp
