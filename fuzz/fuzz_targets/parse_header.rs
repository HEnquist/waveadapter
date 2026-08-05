//! Fuzz the container parser on arbitrary bytes.
//!
//! `read_wav_header` walks every chunk to the end of the file, with all the
//! lengths and offsets taken from the input, so it is the crate's main untrusted
//! input surface. Any input at all must produce either a `WavParams` or a
//! `WavError`, never a panic.
//!
//! Kept in sync with `parse_header` in `tests/fuzz_replay.rs`, which replays the
//! corpus and crash artifacts through the same call on stable.

#![no_main]

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(params) = waveadapter::header::read_wav_header(Cursor::new(data)) {
        // The invariants the reader relies on when it frames the audio. A parse
        // that reports success while breaking these would trip up every caller.
        assert!(params.channels > 0, "a parsed file must have channels");
        if let Some(format) = params.sample_format {
            assert_eq!(
                params.frame_bytes(),
                params.channels * format.bytes_per_sample(),
                "frame size must agree with the interpreted format"
            );
            assert!(params.frame_bytes() > 0, "interpreted frames must be sized");
        }
    }
});
