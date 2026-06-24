//! Read the raw interleaved sample bytes from a wav file and decode the first
//! frame by hand.
//!
//! This is the raw path: waveadapter hands you the untouched bytes and you
//! decide how to interpret them. Here we decode them with the matching
//! audioadapter-sample byte type, dispatching on the file's sample format.
//!
//! Run with: `cargo run --example read_raw -- input.wav`

use audioadapter_sample::readwrite::ReadSamples;
use audioadapter_sample::sample::*;
use waveadapter::{SampleFormat, WavReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: read_raw <input.wav>")?;

    let mut reader = WavReader::new(std::fs::File::open(&path)?)?;
    let format = reader.sample_format();
    let channels = reader.channels();

    // Pull the first frame out as raw bytes, exactly as stored in the file.
    let mut bytes = Vec::new();
    let frames_read = reader.read_raw_interleaved(1, &mut bytes)?;
    println!(
        "read {frames_read} frame of {format:?}, {} raw bytes: {bytes:02x?}",
        bytes.len()
    );

    // Decode each channel of that frame to a normalized f32 using the byte type
    // that matches the file's format. `read_converted` comes from the
    // audioadapter-sample ReadSamples trait, implemented for any std::io::Read
    // (a &[u8] here).
    let mut slice = &bytes[..];
    for ch in 0..channels {
        let value: f32 = match format {
            SampleFormat::I16 => slice.read_converted::<I16_LE, f32>()?,
            SampleFormat::I24_3 => slice.read_converted::<I24_LE, f32>()?,
            SampleFormat::I24_4 => slice.read_converted::<I24_4LJ_LE, f32>()?,
            SampleFormat::I32 => slice.read_converted::<I32_LE, f32>()?,
            SampleFormat::F32 => slice.read_converted::<F32_LE, f32>()?,
            SampleFormat::F64 => slice.read_converted::<F64_LE, f32>()?,
        };
        println!("channel {ch}: {value:.6}");
    }
    Ok(())
}
