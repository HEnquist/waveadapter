//! Replay the fuzzing seeds and crash artifacts on stable.
//!
//! The fuzz targets under `fuzz/fuzz_targets/` need a nightly toolchain, so the
//! checks they perform are mirrored here and driven over a fixed set of files
//! instead of a mutation engine. Discovering new inputs still needs
//! `cargo +nightly fuzz run`, but every input ever found stays locked in as a
//! regression that runs in a plain `cargo test`.
//!
//! Inputs come from two places: the committed `tests/wav_variants/*.wav`
//! fixtures, which seed the fuzzer, and anything libFuzzer has dropped into
//! `fuzz/artifacts/`. Commit crash artifacts when they appear.
//!
//! The check bodies are duplicated from the fuzz targets rather than shared,
//! since a separate crate cannot reach into them. They are short on purpose;
//! keep the two sides in step when either changes.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use waveadapter::{AdtlList, Bext, Cue, InfoList, Smpl, WavReader};

const MAX_RAW_FRAMES: usize = 4096;

/// Every file to replay: the wav fixtures that seed the fuzzer, plus any crash
/// artifacts it has produced.
fn corpus() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect(&root.join("tests/wav_variants"), &mut files);
    collect(&root.join("fuzz/artifacts"), &mut files);
    files.sort();
    files
}

/// Collect every file under `dir`, recursing into subdirectories. A missing
/// directory is not an error: `fuzz/artifacts` only exists once the fuzzer has
/// found something.
fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.file_name().is_some_and(|name| name != ".gitignore") {
            files.push(path);
        }
    }
}

/// Mirrors `fuzz_targets/parse_header.rs`.
fn parse_header(data: &[u8]) {
    if let Ok(params) = waveadapter::header::read_wav_header(Cursor::new(data)) {
        assert!(params.channels() > 0, "a parsed file must have channels");
        if let Some(format) = params.sample_format() {
            assert_eq!(
                params.frame_bytes(),
                params.channels() * format.bytes_per_sample(),
                "frame size must agree with the interpreted format"
            );
            assert!(params.frame_bytes() > 0, "interpreted frames must be sized");
        }
    }
}

/// Mirrors `fuzz_targets/read_audio.rs`.
fn read_audio(data: &[u8]) {
    if let Ok(mut reader) = WavReader::new(Cursor::new(data))
        && reader.read_all_to_float::<f32>().is_ok()
    {
        assert!(
            reader.position() <= reader.frames(),
            "reading past the declared frame count"
        );
    }

    if let Ok(mut reader) = WavReader::new(Cursor::new(data)) {
        let mut buf = Vec::new();
        if let Ok(frames) = reader.read_raw_interleaved(MAX_RAW_FRAMES, &mut buf) {
            assert_eq!(
                buf.len(),
                frames * reader.params().frame_bytes(),
                "raw reads must yield whole frames"
            );
        }
    }

    if let Ok(mut reader) = WavReader::new(Cursor::new(data)) {
        let target = usize::from(data.first().copied().unwrap_or(0));
        if reader.seek_to_frame(target).is_ok() {
            assert!(
                reader.position() <= reader.frames(),
                "seek must stay within the declared frame count"
            );
            let mut buf = Vec::new();
            let _ = reader.read_raw_interleaved(MAX_RAW_FRAMES, &mut buf);
        }
    }
}

/// Mirrors `fuzz_targets/metadata_chunks.rs`.
fn metadata_chunks(data: &[u8]) {
    if let Some(list) = InfoList::from_bytes(data) {
        let again = InfoList::from_bytes(&list.to_bytes())
            .expect("re-decoding an InfoList's own output must succeed");
        assert_eq!(list, again, "InfoList encode/decode is not idempotent");
    }
    if let Some(bext) = Bext::from_bytes(data) {
        let again = Bext::from_bytes(&bext.to_bytes())
            .expect("re-decoding a Bext's own output must succeed");
        assert_eq!(bext, again, "Bext encode/decode is not idempotent");
    }
    if let Some(cue) = Cue::from_bytes(data) {
        let again =
            Cue::from_bytes(&cue.to_bytes()).expect("re-decoding a Cue's own output must succeed");
        assert_eq!(cue, again, "Cue encode/decode is not idempotent");
    }
    if let Some(adtl) = AdtlList::from_bytes(data) {
        let again = AdtlList::from_bytes(&adtl.to_bytes())
            .expect("re-decoding an AdtlList's own output must succeed");
        assert_eq!(adtl, again, "AdtlList encode/decode is not idempotent");
    }
    if let Some(smpl) = Smpl::from_bytes(data) {
        let again = Smpl::from_bytes(&smpl.to_bytes())
            .expect("re-decoding a Smpl's own output must succeed");
        assert_eq!(smpl, again, "Smpl encode/decode is not idempotent");
    }
}

/// Run every corpus file through all three targets, and through a few truncations
/// of each. Cutting a valid file short is the cheapest way to reach the
/// partial-chunk and partial-frame paths without a mutation engine.
#[test]
fn corpus_replays_without_panicking() {
    let files = corpus();
    assert!(!files.is_empty(), "no corpus files found");

    for path in &files {
        let data = fs::read(path).unwrap();

        for cut in [data.len(), data.len() / 2, data.len() / 3, 12, 4, 0] {
            if cut > data.len() {
                continue;
            }
            let slice = &data[..cut];
            // Name the input on failure. Without this a panic points at the
            // check but not at the file or truncation that produced it, which is
            // most of what you need to reproduce it.
            let context = || format!("{} truncated to {cut} bytes", path.display());
            run(&context, || parse_header(slice));
            run(&context, || read_audio(slice));
            run(&context, || metadata_chunks(slice));
        }
    }
}

/// Run one check, reporting which corpus input was being replayed if it panics.
fn run(context: &dyn Fn() -> String, check: impl FnOnce() + std::panic::UnwindSafe) {
    if std::panic::catch_unwind(check).is_err() {
        panic!("replaying {}", context());
    }
}
