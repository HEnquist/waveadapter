//! Write a wav file from raw interleaved sample bytes, in streaming mode.
//!
//! This is the raw path: you already have (or produce) encoded sample bytes and
//! hand them to the writer untouched. Here the bytes are built with
//! audioadapter-sample, but they could come from anywhere. Streaming mode writes
//! the size fields as `u32::MAX`, so the output does not need to be seekable.
//!
//! Run with: `cargo run --example write_raw -- output.wav`

use std::f64::consts::TAU;

use audioadapter_sample::readwrite::WriteSamples;
use audioadapter_sample::sample::I16_LE;
use waveadapter::{SampleFormat, WavSpec, WavWriter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output_raw.wav".to_string());

    let channels = 1;
    let sample_rate = 48000;
    let frames = sample_rate / 2; // half a second
    let freq = 220.0_f64;

    // Produce mono i16 samples and encode them to interleaved little-endian
    // bytes. `write_all_numbers` comes from the audioadapter-sample WriteSamples
    // trait, implemented for any std::io::Write (including a Vec<u8>).
    let mut numbers = Vec::with_capacity(frames);
    for frame in 0..frames {
        let t = frame as f64 / sample_rate as f64;
        let value = (TAU * freq * t).sin() * 0.5;
        numbers.push((value * i16::MAX as f64) as i16);
    }
    let mut bytes = Vec::new();
    bytes.write_all_numbers::<I16_LE>(&numbers)?;

    let spec = WavSpec {
        channels,
        sample_rate,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    let mut writer = WavWriter::new_streaming(std::fs::File::create(&path)?, spec)?;
    writer.write_raw_interleaved(&bytes)?;
    writer.into_inner()?; // flush; streaming leaves the sizes at u32::MAX

    println!("Streamed {frames} frames ({} bytes) to {path}", bytes.len());
    Ok(())
}
