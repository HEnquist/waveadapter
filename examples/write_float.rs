//! Write a wav file from an f32 buffer, converting to the file's sample format.
//!
//! This is the high level path: keep your audio as floats in an audioadapter
//! buffer and let the writer convert and clip into the target format.
//!
//! Run with: `cargo run --example write_float -- output.wav`

use std::f32::consts::TAU;

use audioadapter::AdapterMut;
use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::{SampleFormat, WavSpec, WavWriter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output_float.wav".to_string());

    let channels = 2;
    let sample_rate = 44100;
    let frames = sample_rate; // one second
    let freq = 440.0;

    // Fill an interleaved f32 buffer with a 440 Hz sine, louder on the left.
    let mut buffer = InterleavedOwned::<f32>::new(0.0, channels, frames);
    for frame in 0..frames {
        let t = frame as f32 / sample_rate as f32;
        let sample = (TAU * freq * t).sin();
        buffer.write_sample(0, frame, &(sample * 0.8));
        buffer.write_sample(1, frame, &(sample * 0.4));
    }

    let spec = WavSpec {
        channels,
        sample_rate,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    // Seekable writer: the size fields are patched with the real lengths by
    // finalize, producing a standard-compliant file.
    let mut writer = WavWriter::new(std::fs::File::create(&path)?, spec)?;
    let clipped = writer.write_float_buffer(&buffer)?;
    writer.finalize()?;

    println!("Wrote {frames} frames to {path} ({clipped} samples clipped)");
    Ok(())
}
