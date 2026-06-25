//! Read and decode a `LIST`/`INFO` metadata chunk.
//!
//! waveadapter treats every chunk other than `fmt ` and `data` as an opaque
//! blob, handing it back in `WavReader::params().chunks`. The `metadata` module
//! gives the common `LIST`/`INFO` tag list (title, artist, comment, ...) a typed
//! form: `InfoList::from_chunk` decodes one, `InfoList::to_chunk` builds one.
//!
//! With no argument it synthesizes a small file that carries a `LIST` chunk and
//! decodes that, so it runs standalone. Pass a path to inspect a real file:
//!
//! Run with: `cargo run --example read_list [-- input.wav]`

use std::io::Cursor;

use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::metadata::{self, InfoList};
use waveadapter::{Chunk, SampleFormat, WavReader, WavSpec, WavWriter};

/// Create an in-memory wav file with a `LIST`/`INFO` chunk, returning its bytes.
fn synthesize_demo() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
    };

    let mut info = InfoList::new();
    info.set(metadata::TITLE, "Demo Tone");
    info.set(metadata::ARTIST, "waveadapter");
    info.set(metadata::COMMENT, "synthesized by the read_list example");
    info.set(metadata::SOFTWARE, "waveadapter");
    let list = info.to_chunk();

    let mut writer = WavWriter::new(Cursor::new(Vec::new()), spec)?;
    writer.write_float_buffer(&InterleavedOwned::<f32>::new(0.0, 1, 16))?;
    // A trailing chunk goes after the audio data; LIST/INFO is commonly appended.
    writer.write_chunk(list.id, &list.data)?;
    Ok(writer.finalize()?.into_inner())
}

/// Decode and print the tags from every `LIST`/`INFO` chunk among the parsed chunks.
fn report(chunks: &[Chunk]) {
    let mut found = false;
    for info in chunks.iter().filter_map(InfoList::from_chunk) {
        found = true;
        println!("LIST/INFO tags:");
        for (id, text) in info.iter() {
            println!("  {}: {text}", String::from_utf8_lossy(&id));
        }
    }
    if !found {
        println!(
            "no LIST/INFO chunk found ({} other chunks present)",
            chunks.len()
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::args().nth(1) {
        Some(path) => {
            let reader = WavReader::new(std::fs::File::open(&path)?)?;
            println!("{path}:");
            report(&reader.params().chunks);
        }
        None => {
            println!("(no file given, decoding a synthesized demo file)");
            let reader = WavReader::new(Cursor::new(synthesize_demo()?))?;
            report(&reader.params().chunks);
        }
    }
    Ok(())
}
