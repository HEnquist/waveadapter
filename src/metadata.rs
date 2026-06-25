//! Parsing and generating typed metadata chunks.
//!
//! waveadapter passes every non-audio chunk through as a raw [`Chunk`] blob.
//! This module is a thin typed layer over those blobs for the two common cases,
//! [`InfoList`] (the `LIST`/`INFO` tag list) and [`Bext`] (the Broadcast Audio
//! Extension). Each offers `from_chunk`/`from_bytes` to decode and
//! `to_chunk`/`to_bytes` to build one ready for
//! [`WavWriter::write_chunk`](crate::WavWriter::write_chunk) or a leading-chunk
//! constructor. Anything else stays a raw blob for a caller to interpret.
//!
//! The RIFF `INFO` tag list is the common, simple form of wav metadata (title,
//! artist, comment, ...). It is not a chunk of its own: it is a [`LIST`](Chunk)
//! chunk whose four-character *form type* is `INFO`, followed by a sequence of
//! tag sub-chunks. Each sub-chunk is a 4-byte id, a little-endian 32-bit size,
//! that many bytes of (conventionally NUL-terminated) text, and a pad byte when
//! the size is odd.
//!
//! ```
//! use waveadapter::metadata::{self, InfoList};
//!
//! let mut info = InfoList::new();
//! info.set(metadata::TITLE, "Demo Tone");
//! info.set(metadata::ARTIST, "waveadapter");
//!
//! let chunk = info.to_chunk();
//! assert_eq!(&chunk.id, b"LIST");
//!
//! let parsed = InfoList::from_chunk(&chunk).unwrap();
//! assert_eq!(parsed.get(metadata::TITLE), Some("Demo Tone"));
//! ```

use crate::header::Chunk;

/// The four-character id of the `LIST` chunk that carries an `INFO` list.
pub const LIST_ID: [u8; 4] = *b"LIST";
/// The `LIST` form type for a metadata tag list.
pub const INFO: [u8; 4] = *b"INFO";

/// Title / name of the work (`INAM`).
pub const TITLE: [u8; 4] = *b"INAM";
/// Artist / original performer (`IART`).
pub const ARTIST: [u8; 4] = *b"IART";
/// Product / album the work belongs to (`IPRD`).
pub const PRODUCT: [u8; 4] = *b"IPRD";
/// Comment (`ICMT`).
pub const COMMENT: [u8; 4] = *b"ICMT";
/// Creation date, conventionally `YYYY-MM-DD` (`ICRD`).
pub const CREATION_DATE: [u8; 4] = *b"ICRD";
/// Genre (`IGNR`).
pub const GENRE: [u8; 4] = *b"IGNR";
/// Software that created the file (`ISFT`).
pub const SOFTWARE: [u8; 4] = *b"ISFT";
/// Copyright notice (`ICOP`).
pub const COPYRIGHT: [u8; 4] = *b"ICOP";
/// Engineer (`IENG`).
pub const ENGINEER: [u8; 4] = *b"IENG";
/// Keywords (`IKEY`).
pub const KEYWORDS: [u8; 4] = *b"IKEY";

/// A parsed `LIST`/`INFO` tag list: an ordered sequence of (id, text) tags.
///
/// Order is preserved and duplicate ids are allowed, matching the on-disk
/// layout, so a file can be read and written back without losing or reordering
/// tags. Unknown ids round-trip just like the known ones.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InfoList {
    tags: Vec<([u8; 4], String)>,
}

impl InfoList {
    /// Create an empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode an `INFO` list body: the form type `INFO` followed by the tag
    /// sub-chunks. This is the raw bytes of the `LIST` chunk.
    ///
    /// Returns `None` if the body is not an `INFO` list. Parsing is lenient: a
    /// truncated final sub-chunk is dropped rather than erroring.
    pub fn from_bytes(body: &[u8]) -> Option<Self> {
        if body.len() < 4 || body[0..4] != INFO {
            return None;
        }
        let mut tags = Vec::new();
        let mut pos = 4;
        while pos + 8 <= body.len() {
            let id: [u8; 4] = body[pos..pos + 4].try_into().unwrap();
            let size = u32::from_le_bytes(body[pos + 4..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            if pos + size > body.len() {
                break; // truncated sub-chunk, stop rather than read past the end
            }
            tags.push((id, decode_text(&body[pos..pos + size])));
            pos += size + (size & 1); // step over the pad byte for odd sizes
        }
        Some(Self { tags })
    }

    /// Decode an `INFO` list from a parsed [`Chunk`], as found in
    /// [`WavParams::chunks`](crate::WavParams::chunks).
    ///
    /// Returns `None` if the chunk is not a `LIST` chunk of `INFO` form type.
    pub fn from_chunk(chunk: &Chunk) -> Option<Self> {
        if chunk.id != LIST_ID {
            return None;
        }
        Self::from_bytes(&chunk.data)
    }

    /// Encode the list as a `LIST` chunk body: the form type `INFO` followed by
    /// the tag sub-chunks, each NUL-terminated and padded to an even length.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&INFO);
        for (id, text) in &self.tags {
            let mut value = text.as_bytes().to_vec();
            value.push(0); // NUL terminator
            body.extend_from_slice(id);
            body.extend_from_slice(&(value.len() as u32).to_le_bytes());
            body.extend_from_slice(&value);
            if value.len() % 2 == 1 {
                body.push(0); // pad to an even length
            }
        }
        body
    }

    /// Build a `LIST` [`Chunk`] ready to hand to the writer.
    pub fn to_chunk(&self) -> Chunk {
        Chunk {
            id: LIST_ID,
            data: self.to_bytes(),
        }
    }

    /// The text of the first tag with the given id, if present.
    pub fn get(&self, id: [u8; 4]) -> Option<&str> {
        self.tags
            .iter()
            .find(|(tag_id, _)| *tag_id == id)
            .map(|(_, text)| text.as_str())
    }

    /// Set the first tag with the given id, replacing its text, or append it if
    /// there is none yet.
    pub fn set(&mut self, id: [u8; 4], text: impl Into<String>) {
        match self.tags.iter_mut().find(|(tag_id, _)| *tag_id == id) {
            Some((_, existing)) => *existing = text.into(),
            None => self.tags.push((id, text.into())),
        }
    }

    /// Append a tag, keeping any existing tag with the same id.
    pub fn push(&mut self, id: [u8; 4], text: impl Into<String>) {
        self.tags.push((id, text.into()));
    }

    /// Remove every tag with the given id, returning how many were removed.
    pub fn remove(&mut self, id: [u8; 4]) -> usize {
        let before = self.tags.len();
        self.tags.retain(|(tag_id, _)| *tag_id != id);
        before - self.tags.len()
    }

    /// Iterate the tags in order as (id, text) pairs.
    pub fn iter(&self) -> impl Iterator<Item = ([u8; 4], &str)> + '_ {
        self.tags.iter().map(|(id, text)| (*id, text.as_str()))
    }

    /// The number of tags.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Whether the list has no tags.
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }
}

/// Decode a NUL-terminated, fixed-width or trailing text field: take the bytes
/// up to the first NUL (the rest is padding) and interpret them as text,
/// replacing invalid sequences. Used for both `INFO` values and the fixed-width
/// `bext` string fields.
fn decode_text(raw: &[u8]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

/// Append `s` as a fixed-width NUL-padded field of `width` bytes, truncating the
/// text if it is longer than the field.
fn encode_fixed(out: &mut Vec<u8>, s: &str, width: usize) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(width);
    out.extend_from_slice(&bytes[..n]);
    out.resize(out.len() + (width - n), 0);
}

/// The four-character id of the Broadcast Audio Extension chunk.
pub const BEXT_ID: [u8; 4] = *b"bext";

/// The size of the fixed part of a `bext` chunk, before the variable-length
/// coding history.
const BEXT_FIXED_LEN: usize = 602;

/// A parsed Broadcast Audio Extension (`bext`) chunk, EBU Tech 3285.
///
/// This is the broadcast/production metadata: a timecode reference, originator
/// and timestamp fields, a coding-history log, and (from version 2) EBU R128
/// loudness measurements. The on-disk layout is a fixed 602-byte structure
/// followed by the variable-length [`coding_history`](Bext::coding_history).
///
/// The fields are public so they can be read and modified directly. The string
/// fields have fixed maximum widths on disk and are truncated to fit when
/// written: [`description`](Bext::description) 256 bytes,
/// [`originator`](Bext::originator) and
/// [`originator_reference`](Bext::originator_reference) 32,
/// [`origination_date`](Bext::origination_date) 10 (`yyyy-mm-dd`),
/// [`origination_time`](Bext::origination_time) 8 (`hh:mm:ss`).
///
/// [`umid`](Bext::umid) is meaningful for `version >= 1` and the loudness fields
/// for `version >= 2`; in older files those byte ranges are reserved and read
/// back as zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bext {
    /// Free-text description of the sequence.
    pub description: String,
    /// Name of the originator (often the recorder or application).
    pub originator: String,
    /// Unambiguous reference identifying the originating file.
    pub originator_reference: String,
    /// Origination date, conventionally `yyyy-mm-dd`.
    pub origination_date: String,
    /// Origination time, conventionally `hh:mm:ss`.
    pub origination_time: String,
    /// Timecode of the first sample, as a count of samples since midnight.
    /// Combined low/high 32-bit halves; divide by the sample rate for seconds.
    pub time_reference: u64,
    /// BWF version: 0, 1, or 2.
    pub version: u16,
    /// SMPTE UMID (meaningful for `version >= 1`).
    pub umid: [u8; 64],
    /// Integrated loudness in LUFS times 100 (`version >= 2`).
    pub loudness_value: i16,
    /// Loudness range in LU times 100 (`version >= 2`).
    pub loudness_range: i16,
    /// Maximum true peak level in dBTP times 100 (`version >= 2`).
    pub max_true_peak_level: i16,
    /// Maximum momentary loudness in LUFS times 100 (`version >= 2`).
    pub max_momentary_loudness: i16,
    /// Maximum short-term loudness in LUFS times 100 (`version >= 2`).
    pub max_short_term_loudness: i16,
    /// Free-form history of the coding processes the file went through, one
    /// line per stage (for example `A=PCM,F=48000,W=24,M=stereo,T=original`).
    pub coding_history: String,
}

impl Default for Bext {
    fn default() -> Self {
        Self {
            description: String::new(),
            originator: String::new(),
            originator_reference: String::new(),
            origination_date: String::new(),
            origination_time: String::new(),
            time_reference: 0,
            version: 0,
            umid: [0; 64],
            loudness_value: 0,
            loudness_range: 0,
            max_true_peak_level: 0,
            max_momentary_loudness: 0,
            max_short_term_loudness: 0,
            coding_history: String::new(),
        }
    }
}

impl Bext {
    /// Create a `bext` with all fields zeroed/empty and version 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode a `bext` chunk body.
    ///
    /// Returns `None` if the body is shorter than the fixed 602-byte structure.
    pub fn from_bytes(body: &[u8]) -> Option<Self> {
        if body.len() < BEXT_FIXED_LEN {
            return None;
        }
        let u32_at = |o: usize| u32::from_le_bytes(body[o..o + 4].try_into().unwrap());
        let u16_at = |o: usize| u16::from_le_bytes(body[o..o + 2].try_into().unwrap());
        let i16_at = |o: usize| i16::from_le_bytes(body[o..o + 2].try_into().unwrap());

        let mut umid = [0u8; 64];
        umid.copy_from_slice(&body[348..412]);

        Some(Self {
            description: decode_text(&body[0..256]),
            originator: decode_text(&body[256..288]),
            originator_reference: decode_text(&body[288..320]),
            origination_date: decode_text(&body[320..330]),
            origination_time: decode_text(&body[330..338]),
            time_reference: u32_at(338) as u64 | ((u32_at(342) as u64) << 32),
            version: u16_at(346),
            umid,
            loudness_value: i16_at(412),
            loudness_range: i16_at(414),
            max_true_peak_level: i16_at(416),
            max_momentary_loudness: i16_at(418),
            max_short_term_loudness: i16_at(420),
            coding_history: decode_text(&body[BEXT_FIXED_LEN..]),
        })
    }

    /// Decode a `bext` from a parsed [`Chunk`]. Returns `None` if the chunk is
    /// not a `bext` chunk.
    pub fn from_chunk(chunk: &Chunk) -> Option<Self> {
        if chunk.id != BEXT_ID {
            return None;
        }
        Self::from_bytes(&chunk.data)
    }

    /// Encode the `bext` chunk body: the fixed 602-byte structure followed by
    /// the coding history.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(BEXT_FIXED_LEN + self.coding_history.len());
        encode_fixed(&mut out, &self.description, 256);
        encode_fixed(&mut out, &self.originator, 32);
        encode_fixed(&mut out, &self.originator_reference, 32);
        encode_fixed(&mut out, &self.origination_date, 10);
        encode_fixed(&mut out, &self.origination_time, 8);
        out.extend_from_slice(&(self.time_reference as u32).to_le_bytes());
        out.extend_from_slice(&((self.time_reference >> 32) as u32).to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.umid);
        out.extend_from_slice(&self.loudness_value.to_le_bytes());
        out.extend_from_slice(&self.loudness_range.to_le_bytes());
        out.extend_from_slice(&self.max_true_peak_level.to_le_bytes());
        out.extend_from_slice(&self.max_momentary_loudness.to_le_bytes());
        out.extend_from_slice(&self.max_short_term_loudness.to_le_bytes());
        // Reserved field: pad the fixed part out to its full 602 bytes.
        out.resize(BEXT_FIXED_LEN, 0);
        out.extend_from_slice(self.coding_history.as_bytes());
        out
    }

    /// Build a `bext` [`Chunk`] ready to hand to the writer.
    pub fn to_chunk(&self) -> Chunk {
        Chunk {
            id: BEXT_ID,
            data: self.to_bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_bytes() {
        let mut info = InfoList::new();
        info.set(TITLE, "Demo Tone");
        info.set(ARTIST, "waveadapter");
        // An odd-length value exercises the pad byte.
        info.set(COMMENT, "odd");

        let bytes = info.to_bytes();
        assert_eq!(&bytes[0..4], b"INFO");
        let parsed = InfoList::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, info);
        assert_eq!(parsed.get(ARTIST), Some("waveadapter"));
    }

    #[test]
    fn roundtrips_through_chunk() {
        let mut info = InfoList::new();
        info.set(SOFTWARE, "waveadapter");
        let chunk = info.to_chunk();
        assert_eq!(&chunk.id, b"LIST");
        assert_eq!(InfoList::from_chunk(&chunk), Some(info));
    }

    #[test]
    fn set_replaces_and_push_appends() {
        let mut info = InfoList::new();
        info.set(TITLE, "First");
        info.set(TITLE, "Second");
        assert_eq!(info.len(), 1);
        assert_eq!(info.get(TITLE), Some("Second"));

        info.push(TITLE, "Third");
        assert_eq!(info.len(), 2);
        assert_eq!(info.remove(TITLE), 2);
        assert!(info.is_empty());
    }

    #[test]
    fn non_info_body_is_rejected() {
        assert!(InfoList::from_bytes(b"adtl").is_none());
        assert!(InfoList::from_bytes(b"").is_none());
        let other = Chunk {
            id: *b"bext",
            data: b"INFO".to_vec(),
        };
        assert!(InfoList::from_chunk(&other).is_none());
    }

    #[test]
    fn truncated_subchunk_is_dropped() {
        // "INFO" + a tag claiming 100 bytes but only 3 present.
        let mut body = b"INFO".to_vec();
        body.extend_from_slice(b"INAM");
        body.extend_from_slice(&100u32.to_le_bytes());
        body.extend_from_slice(b"abc");
        let parsed = InfoList::from_bytes(&body).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn unknown_ids_roundtrip() {
        let mut info = InfoList::new();
        info.set(*b"ZZZZ", "custom");
        let parsed = InfoList::from_bytes(&info.to_bytes()).unwrap();
        assert_eq!(parsed.get(*b"ZZZZ"), Some("custom"));
    }

    #[test]
    fn bext_roundtrips() {
        let bext = Bext {
            description: "Scene 1 Take 3".to_string(),
            originator: "MixPre".to_string(),
            originator_reference: "REF-0001".to_string(),
            origination_date: "2026-06-25".to_string(),
            origination_time: "09:30:00".to_string(),
            // A value larger than u32::MAX exercises the low/high split.
            time_reference: 0x1_0000_2345,
            version: 2,
            umid: [7; 64],
            loudness_value: -2300,
            loudness_range: 700,
            max_true_peak_level: -150,
            max_momentary_loudness: -1800,
            max_short_term_loudness: -2000,
            coding_history: "A=PCM,F=48000,W=24,M=stereo,T=original".to_string(),
        };
        let bytes = bext.to_bytes();
        assert_eq!(&bytes[0..14], b"Scene 1 Take 3");
        assert!(
            bytes.len() > BEXT_FIXED_LEN,
            "coding history follows the fixed part"
        );
        assert_eq!(Bext::from_bytes(&bytes), Some(bext));
    }

    #[test]
    fn bext_roundtrips_through_chunk() {
        let bext = Bext {
            originator: "waveadapter".to_string(),
            version: 1,
            ..Bext::new()
        };
        let chunk = bext.to_chunk();
        assert_eq!(&chunk.id, b"bext");
        assert_eq!(Bext::from_chunk(&chunk), Some(bext));
    }

    #[test]
    fn bext_rejects_short_and_wrong_chunk() {
        assert!(Bext::from_bytes(&[0u8; 100]).is_none());
        let not_bext = Chunk {
            id: *b"LIST",
            data: vec![0u8; BEXT_FIXED_LEN],
        };
        assert!(Bext::from_chunk(&not_bext).is_none());
    }

    #[test]
    fn bext_truncates_overlong_fields() {
        let bext = Bext {
            // The date field is 10 bytes; anything longer is truncated.
            origination_date: "2026-06-25T12:00".to_string(),
            ..Bext::new()
        };
        let parsed = Bext::from_bytes(&bext.to_bytes()).unwrap();
        assert_eq!(parsed.origination_date, "2026-06-25");
    }
}
