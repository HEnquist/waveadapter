//! Read a wav file into an f32 buffer and report a peak level per channel.
//!
//! This is the high level path: the reader converts whatever the file's sample
//! format is into normalized floats in the range -1.0..1.0.
//!
//! Run with: `cargo run --example read_float -- input.wav`

use audioadapter::Adapter;
use waveadapter::WavReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: read_float <input.wav>")?;

    let mut reader = WavReader::new(std::fs::File::open(&path)?)?;
    println!(
        "{path}: {} ch, {} Hz, {:?}, {} frames",
        reader.channels(),
        reader.sample_rate(),
        reader.sample_format(),
        reader.frames()
    );

    // Read all remaining frames into an owned interleaved f32 buffer. This also
    // works for streaming files whose declared length is not accurate.
    let buffer = reader.read_all_to_float::<f32>()?;

    for ch in 0..buffer.channels() {
        let mut peak = 0.0_f32;
        for frame in 0..buffer.frames() {
            peak = peak.max(buffer.read_sample(ch, frame).unwrap().abs());
        }
        let dbfs = 20.0 * peak.max(1e-9).log10();
        println!("channel {ch}: peak {peak:.4} ({dbfs:.1} dBFS)");
    }
    Ok(())
}
