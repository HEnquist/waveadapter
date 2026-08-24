//! Write a file in a format this crate does not model, by handing the writer the
//! `fmt ` chunk to use and the encoded bytes to put after it.
//!
//! The format is IMA ADPCM, and waveadapter cannot encode it. The sine wave here
//! is encoded by `audio-codec-algorithms`, one block at a time, and this crate
//! only puts a `fmt ` chunk in front of the result and keeps the sizes straight.
//! That division of labour is the point of the raw path.
//!
//! A `WavSpec` says "I have audio in a format you model, build me a header"; a
//! `FmtChunk` says "I am the codec, write exactly these bytes".
//!
//! Run with: `cargo run --example write_raw -- output.wav`, then read it back
//! with `cargo run --example read_raw -- output.wav`.

use std::f64::consts::TAU;

use audio_codec_algorithms::{AdpcmImaState, encode_adpcm_ima_ms};
use waveadapter::{Fact, FmtChunk, WavWriter};

/// `WAVE_FORMAT_DVI_ADPCM`: four bits per sample, in fixed size blocks.
const IMA_ADPCM: u16 = 0x11;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output_raw.wav".to_string());

    // Mono keeps the block layout simple: stereo interleaves the sample codes in
    // four byte groups per channel rather than one code at a time.
    let channels: u16 = 1;
    let sample_rate: u32 = 48000;
    let freq = 440.0;
    // The usual IMA ADPCM block: 256 bytes per channel, holding a four byte
    // header and then one four bit code per sample frame, so the number of
    // frames it decodes to is fixed by its size.
    let block_align = 256 * channels;
    let samples_per_block = (block_align - 4 * channels) * 2 / channels + 1;
    // Only whole blocks can be written, so half a second rounds up to the next
    // block boundary.
    let blocks = (sample_rate / 2).div_ceil(samples_per_block as u32);
    let frames = blocks * samples_per_block as u32;

    let fmt = FmtChunk {
        format_code: IMA_ADPCM,
        channels,
        sample_rate,
        // Not derivable outside linear PCM, which is why the crate carries this
        // field rather than computing it: whole blocks per second, times the
        // block size, and only the codec knows how many frames a block holds.
        byte_rate: sample_rate * block_align as u32 / samples_per_block as u32,
        block_align,
        bits_per_sample: 4,
        // The format specific bytes after cbSize, wSamplesPerBlock for IMA
        // ADPCM. The writer fills in cbSize from the length.
        extension: Some(samples_per_block.to_le_bytes().to_vec()),
    };

    // `Fact::Auto` can only count frames for a format the crate models, and a
    // block count is not a frame count. So the caller supplies it; without it
    // the file carries no trustworthy frame count at all. A supplied count is
    // written as is and left alone by finalize.
    let mut writer = WavWriter::builder(fmt)?
        .fact(Fact::Samples(frames as u64))
        .open(std::fs::File::create(&path)?)?;

    let sine: Vec<i16> = (0..frames)
        .map(|frame| {
            let t = frame as f64 / sample_rate as f64;
            ((TAU * freq * t).sin() * 0.5 * i16::MAX as f64) as i16
        })
        .collect();

    // Encode one block at a time. The state carries the step size index from one
    // block to the next; its predictor is overwritten by the encoder with the
    // first sample of each block, which is what ends up in the block header.
    let mut states = vec![AdpcmImaState::new(); channels as usize];
    let mut block = vec![0u8; block_align as usize];
    for chunk in sine.chunks(samples_per_block as usize) {
        encode_adpcm_ima_ms(chunk, &mut states, &mut block)
            .map_err(|err| format!("encoding failed: {err:?}"))?;
        writer.write_raw_interleaved(&block)?;
    }
    writer.finalize()?; // patch the size fields with the real lengths

    println!("Wrote {blocks} blocks ({frames} sample frames) of {freq} Hz sine to {path}");
    Ok(())
}
