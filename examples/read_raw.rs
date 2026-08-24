//! Read a file this crate cannot decode, and turn its blocks back into audio
//! with a codec crate.
//!
//! The file is IMA ADPCM. waveadapter hands back the untouched bytes and the
//! `fmt ` chunk that frames them, `audio-codec-algorithms` turns a block into
//! samples, and audioadapter measures the result. Keeping the container, the
//! codec and the buffer in three separate crates is the point of the raw path.
//!
//! This is that path: the caller gets the bytes and decides what they mean. It
//! is the only way to read a format this crate does not model
//! (`sample_format()` is `None`, and the float path refuses).
//!
//! Run with:
//! `cargo run --example read_raw -- tests/wav_variants/ima_adpcm_mono.wav`,
//! or on the file the `write_raw` example produces.

use audio_codec_algorithms::decode_adpcm_ima_ms;
use audioadapter::stats::AdapterStats;
use audioadapter_buffers::direct::InterleavedSlice;
use waveadapter::WavReader;

/// `WAVE_FORMAT_DVI_ADPCM`: four bits per sample, in fixed size blocks.
const IMA_ADPCM: u16 = 0x11;

/// Full scale for the 16 bit samples the decoder produces.
const FULL_SCALE: f64 = 32768.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: read_raw <input.wav>")?;

    let file = std::fs::File::open(&path)?;
    let file_bytes = file.metadata()?.len();
    let mut reader = WavReader::new(file)?;
    let params = reader.params();
    let fmt = params.fmt.clone();
    let channels = params.channels();
    let data_offset = params.data_offset;
    // A streaming file declares the u32::MAX placeholder instead of a real size.
    // `length_is_unknown` is the way to ask: an RF64 file whose real size is
    // exactly that value is not a stream, and the accessor knows the difference.
    let declared_bytes = (!params.length_is_unknown()).then_some(params.data_length);

    println!(
        "format {:#06x}, {channels} ch at {} Hz, {} bits per sample",
        fmt.format_code, fmt.sample_rate, fmt.bits_per_sample
    );

    // For a block compressed format nBlockAlign is a whole compressed block
    // rather than one sample frame, and the raw path frames the bytes off it. So
    // the reader counts blocks here, and the only real frame count is the one
    // the fact chunk carries.
    println!("block align {} bytes", fmt.block_align);
    match declared_bytes {
        Some(_) => println!("{} block(s) of audio data", reader.frames()),
        None => println!("data size not declared: the placeholder that runs to end of file"),
    }
    match params.sample_count() {
        Some(frames) => println!("fact chunk: {frames} sample frames"),
        None => println!("no fact chunk, so the true frame count is unknown"),
    }
    // wSamplesPerBlock sits in the format specific bytes after cbSize, which the
    // crate carries along without interpreting.
    if let Some(extension) = fmt.extension.as_deref().filter(|ext| ext.len() >= 2) {
        let per_block = u16::from_le_bytes([extension[0], extension[1]]);
        println!("fmt extension: {per_block} sample frames per block");
    }

    // One "frame" of the raw path is one whole block for this format, and it
    // comes back exactly as stored.
    let mut block = Vec::new();
    if reader.read_raw_interleaved(1, &mut block)? == 0 {
        println!("\nno audio data");
        return Ok(());
    }
    println!(
        "\nfirst block, {} bytes, starting {:02x?}",
        block.len(),
        &block[..8.min(block.len())]
    );

    if fmt.format_code != IMA_ADPCM || block.len() < 4 * channels {
        println!("not IMA ADPCM, so the bytes are as far as this example goes");
        return Ok(());
    }
    if channels > 2 {
        println!("the decoder handles mono and stereo only");
        return Ok(());
    }

    // Size the buffers up front. The declared data size can be the streaming
    // placeholder or simply too large, so cap it by what the file actually holds;
    // getting it wrong only costs a reallocation.
    let mut decoded = vec![0i16; 2 * block.len() - 7 * channels]; // what the decoder wants
    let available = file_bytes.saturating_sub(data_offset);
    let audio_bytes = declared_bytes.unwrap_or(available).min(available);
    let expected_blocks = (audio_bytes / block.len() as u64) as usize;
    let mut samples = Vec::with_capacity(expected_blocks * decoded.len());

    // Read a block, decode it, stash the samples, repeat.
    let mut blocks = 0;
    while !block.is_empty() {
        decode_adpcm_ima_ms(&block, channels == 2, &mut decoded)
            .map_err(|err| format!("decoding block {blocks} failed: {err:?}"))?;
        samples.extend_from_slice(&decoded);
        blocks += 1;
        block.clear();
        reader.read_raw_interleaved(1, &mut block)?;
    }

    // Now we have ordinary interleaved audio, so wrap it in an audioadapter buffer
    // and let that do the measuring.
    let frames = samples.len() / channels;
    let audio = InterleavedSlice::new(&samples, channels, frames)?;
    println!("\ndecoded {blocks} block(s) into {frames} sample frames");
    for ch in 0..channels {
        let peak = audio.channel_peak(ch) / FULL_SCALE;
        let rms = audio.channel_rms(ch) / FULL_SCALE;
        println!(
            "channel {ch}: peak {peak:.4} ({:.1} dBFS), rms {rms:.4} ({:.1} dBFS)",
            20.0 * peak.max(1e-9).log10(),
            20.0 * rms.max(1e-9).log10()
        );
    }
    Ok(())
}
