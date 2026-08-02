//! Fuzz the full read pipeline: parse a header, then pull audio through both the
//! float and the raw path.
//!
//! This covers the frame arithmetic in `reader.rs` that `parse_header` does not
//! reach: the declared-length bookkeeping, the stop-at-EOF handling, the partial
//! trailing frame rule, and seeking.
//!
//! Kept in sync with `read_audio` in `tests/fuzz_replay.rs`.

#![no_main]

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;
use waveadapter::WavReader;

/// Cap the raw read. `read_raw_interleaved` sizes its buffer from the *declared*
/// data length, which the input controls, so an unbounded request here would
/// just measure how large a number the header can name.
const MAX_RAW_FRAMES: usize = 4096;

fuzz_target!(|data: &[u8]| {
    // Float path. Reading stops cleanly at end of data, so a header that
    // over-declares its length must not run away or panic.
    if let Ok(mut reader) = WavReader::new(Cursor::new(data))
        && reader.read_all_to_float::<f32>().is_ok()
    {
        assert!(
            reader.position() <= reader.frames(),
            "reading past the declared frame count"
        );
    }

    // Raw path. This one also accepts formats the float path rejects, so it sees
    // headers the branch above returns early on.
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

    // Seeking, driven by the input itself. The target is clamped to the frame
    // count, so no offset may push the reader into an inconsistent state.
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
});
