//! Parsing and generating the `LIST`/`INFO` metadata chunk.
//!
//! The RIFF `INFO` tag list is the common, simple form of wav metadata (title,
//! artist, comment, ...). It is not a chunk of its own: it is a [`LIST`](Chunk)
//! chunk whose four-character *form type* is `INFO`, followed by a sequence of
//! tag sub-chunks. Each sub-chunk is a 4-byte id, a little-endian 32-bit size,
//! that many bytes of (conventionally NUL-terminated) text, and a pad byte when
//! the size is odd.
//!
//! This module is a thin typed layer over the raw [`Chunk`] blobs that the
//! reader and writer pass through: [`InfoList::from_chunk`] decodes one, and
//! [`InfoList::to_chunk`] builds one ready for
//! [`WavWriter::write_chunk`](crate::WavWriter::write_chunk) or a leading-chunk
//! constructor.
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

/// Decode an `INFO` tag value: take the bytes up to the first NUL (the rest is
/// padding) and interpret them as text, replacing invalid sequences.
fn decode_text(raw: &[u8]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
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
}
