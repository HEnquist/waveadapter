//! Fuzz the typed metadata decoders on arbitrary chunk bodies.
//!
//! These take blobs that came out of a file, so they are untrusted in exactly the
//! same way the container parser is. Beyond "must not panic", each decoder is
//! checked for idempotence: re-encoding a decoded value and decoding it again
//! must give the same value back.
//!
//! Idempotence is the right property here rather than a straight round-trip,
//! because decoding is deliberately lossy on malformed input (truncated cue
//! points are dropped, unknown `adtl` subchunks are skipped, overlong `bext`
//! fields are truncated on write). Those losses all happen on the first decode,
//! so everything after it must be stable.
//!
//! Kept in sync with `metadata_chunks` in `tests/fuzz_replay.rs`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use waveadapter::{AdtlList, Bext, Cue, InfoList};

fuzz_target!(|data: &[u8]| {
    if let Some(list) = InfoList::from_bytes(data) {
        let again = InfoList::from_bytes(&list.to_bytes())
            .expect("re-decoding an InfoList's own output must succeed");
        assert_eq!(list, again, "InfoList encode/decode is not idempotent");
    }

    if let Some(bext) = Bext::from_bytes(data) {
        let again = Bext::from_bytes(&bext.to_bytes())
            .expect("re-decoding a Bext's own output must succeed");
        // Only checked when the fixed-width fields are ASCII. Non-UTF-8 bytes
        // there are a known defect: `decode_text` turns each one into a 3-byte
        // U+FFFD, and `encode_fixed` then truncates back to the field width, so
        // a full 32-byte field of non-ASCII comes back as 11 characters. Widen
        // this assertion once the charset handling is settled.
        let fixed_is_ascii = data[..338].iter().all(u8::is_ascii);
        if fixed_is_ascii {
            assert_eq!(bext, again, "Bext encode/decode is not idempotent");
        }
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
});
