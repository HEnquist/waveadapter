//! Push every typed metadata chunk through `WavWriter` and back out through
//! `WavReader`.
//!
//! The unit tests in `metadata.rs` cover encode/decode symmetry at the type
//! level, so the types demonstrably serialise on their own. What they cannot see
//! is the writer: chunk padding, the leading-versus-trailing split, and the two
//! `LIST` forms sitting in the same file. That is what these tests exercise.

use std::io::Cursor;

use waveadapter::metadata::{self, AdtlEntry, AdtlList, Bext, Cue, CuePoint, InfoList, SampleLoop};
use waveadapter::{Chunk, SampleFormat, Smpl, WavReader, WavSpec, WavWriter};

const SAMPLE_RATE: usize = 48_000;

fn spec() -> WavSpec {
    WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    }
}

fn info_list() -> InfoList {
    let mut info = InfoList::new();
    info.set(metadata::TITLE, "Demo Tone");
    info.set(metadata::ARTIST, "waveadapter");
    // An odd-length value exercises the sub-chunk pad byte inside the list.
    info.set(metadata::COMMENT, "odd");
    info
}

fn bext() -> Bext {
    Bext {
        description: "Scene 1 Take 3".to_string(),
        originator: "waveadapter".to_string(),
        origination_date: "2026-06-25".to_string(),
        origination_time: "09:30:00".to_string(),
        time_reference: 0x1_0000_2345,
        version: 2,
        umid: [7; 64],
        loudness_value: -2300,
        // An odd-length coding history makes the whole chunk odd, which is what
        // forces the writer to emit a pad byte before whatever follows.
        coding_history: "A=PCM,F=48000,W=24,M=mono".to_string(),
        ..Bext::new()
    }
}

fn cue() -> Cue {
    Cue {
        points: vec![CuePoint::at(1, 0), CuePoint::at(2, 24_000)],
    }
}

fn adtl() -> AdtlList {
    AdtlList {
        entries: vec![
            AdtlEntry::Label {
                cue_id: 1,
                text: "Intro".to_string(),
            },
            AdtlEntry::Note {
                cue_id: 1,
                text: "odd".to_string(),
            },
            AdtlEntry::LabeledText {
                cue_id: 2,
                sample_length: 24_000,
                purpose: *b"rgn ",
                country: 0,
                language: 9,
                dialect: 1,
                code_page: 0,
                text: "Verse".to_string(),
            },
        ],
    }
}

fn smpl() -> Smpl {
    Smpl {
        loops: vec![SampleLoop::forward(1, 0, 47)],
        // An odd number of trailing bytes, again for the pad byte.
        sampler_data: vec![0xAA, 0xBB, 0xCC],
        ..Smpl::at_rate(SAMPLE_RATE as u32)
    }
}

/// The audio the tests write, as raw little-endian 16-bit frames.
fn audio() -> Vec<u8> {
    (0..48i16).flat_map(|v| v.to_le_bytes()).collect()
}

/// Every chunk with the given id, in file order.
fn by_id<'a>(chunks: &'a [Chunk], id: &[u8; 4]) -> Vec<&'a Chunk> {
    chunks.iter().filter(|c| &c.id == id).collect()
}

#[test]
fn every_typed_chunk_roundtrips_through_a_file() {
    // Both LIST forms are in play: INFO leads, adtl trails. They share an id, so
    // this is also the check that the form type keeps them apart in a real file.
    let leading = vec![info_list().to_chunk(), bext().to_chunk(), cue().to_chunk()];

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_with_chunks(&mut cursor, spec(), &leading).unwrap();
    writer.write_raw_interleaved(&audio()).unwrap();
    writer.write_chunk(*b"LIST", &adtl().to_bytes()).unwrap();
    writer.write_chunk(*b"smpl", &smpl().to_bytes()).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    let chunks = reader.params().chunks.clone();

    let lists = by_id(&chunks, b"LIST");
    assert_eq!(lists.len(), 2, "both LIST chunks survive: {chunks:?}");
    assert_eq!(InfoList::from_chunk(lists[0]), Some(info_list()));
    assert_eq!(AdtlList::from_chunk(lists[1]), Some(adtl()));
    // Neither decoder accepts the other's chunk, so nothing is silently swapped.
    assert!(AdtlList::from_chunk(lists[0]).is_none());
    assert!(InfoList::from_chunk(lists[1]).is_none());

    assert_eq!(Bext::from_chunk(by_id(&chunks, b"bext")[0]), Some(bext()));
    assert_eq!(Cue::from_chunk(by_id(&chunks, b"cue ")[0]), Some(cue()));
    assert_eq!(Smpl::from_chunk(by_id(&chunks, b"smpl")[0]), Some(smpl()));

    // The metadata around it must not disturb the audio.
    let mut got = Vec::new();
    let frames = reader.read_raw_interleaved(48, &mut got).unwrap();
    assert_eq!(frames, 48);
    assert_eq!(got, audio());
}

#[test]
fn odd_length_chunks_keep_their_exact_bytes() {
    // A pad byte after an odd-length chunk belongs to the container, not to the
    // chunk: it must not come back as part of the payload, and it must not shift
    // whatever follows. Alternating odd and even chunks on both sides of the
    // audio puts a pad byte before a leading chunk, before the audio, and before
    // a trailing chunk.
    let bext_bytes = bext().to_bytes();
    let smpl_bytes = smpl().to_bytes();
    assert_eq!(bext_bytes.len() % 2, 1, "the bext body must be odd here");
    assert_eq!(smpl_bytes.len() % 2, 1, "the smpl body must be odd here");

    let leading = vec![bext().to_chunk(), cue().to_chunk()];
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_with_chunks(&mut cursor, spec(), &leading).unwrap();
    // An odd number of data bytes forces a pad byte after the audio too.
    writer.write_raw_interleaved(&audio()[..46]).unwrap();
    writer.write_chunk(*b"smpl", &smpl_bytes).unwrap();
    writer
        .write_chunk(*b"LIST", &info_list().to_bytes())
        .unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let reader = WavReader::new(cursor).unwrap();
    let chunks = &reader.params().chunks;

    assert_eq!(by_id(chunks, b"bext")[0].data, bext_bytes);
    assert_eq!(by_id(chunks, b"smpl")[0].data, smpl_bytes);
    assert_eq!(by_id(chunks, b"cue ")[0].data, cue().to_bytes());
    assert_eq!(by_id(chunks, b"LIST")[0].data, info_list().to_bytes());
    // 46 bytes is 23 whole frames; the trailing half frame is dropped, not read
    // as audio and not confused with the pad byte.
    assert_eq!(reader.frames(), 23);
}

#[test]
fn chunk_order_is_preserved() {
    // Order matters for duplicate ids, since decoding cannot tell two INFO lists
    // apart. Leading chunks come out ahead of the trailing ones either way.
    let mut first = InfoList::new();
    first.set(metadata::TITLE, "First");
    let mut second = InfoList::new();
    second.set(metadata::TITLE, "Second");

    let leading = vec![first.to_chunk(), bext().to_chunk()];
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_with_chunks(&mut cursor, spec(), &leading).unwrap();
    writer.write_raw_interleaved(&audio()).unwrap();
    writer.write_chunk(*b"LIST", &second.to_bytes()).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let reader = WavReader::new(cursor).unwrap();
    let ids: Vec<[u8; 4]> = reader.params().chunks.iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![*b"LIST", *b"bext", *b"LIST"]);

    let lists = by_id(&reader.params().chunks, b"LIST");
    assert_eq!(
        InfoList::from_chunk(lists[0]).unwrap().get(metadata::TITLE),
        Some("First")
    );
    assert_eq!(
        InfoList::from_chunk(lists[1]).unwrap().get(metadata::TITLE),
        Some("Second")
    );
}

#[test]
fn metadata_survives_a_streaming_write() {
    // A streaming writer never patches its sizes, so a reader has to find the
    // leading chunks through the placeholder-sized header instead.
    let leading = vec![info_list().to_chunk(), smpl().to_chunk()];
    let mut writer =
        WavWriter::new_streaming_with_chunks(Cursor::new(Vec::new()), spec(), &leading).unwrap();
    writer.write_raw_interleaved(&audio()).unwrap();
    let mut cursor = writer.into_inner().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    let chunks = reader.params().chunks.clone();
    assert_eq!(
        InfoList::from_chunk(by_id(&chunks, b"LIST")[0]),
        Some(info_list())
    );
    assert_eq!(Smpl::from_chunk(by_id(&chunks, b"smpl")[0]), Some(smpl()));

    let mut got = Vec::new();
    reader.read_raw_interleaved(48, &mut got).unwrap();
    assert_eq!(got, audio());
}

#[test]
fn metadata_survives_an_rf64_write() {
    // RF64 puts a ds64 chunk ahead of the caller's leading chunks and patches
    // sizes in a different place, so the pass-through is worth its own check.
    let leading = vec![bext().to_chunk(), cue().to_chunk()];
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_rf64_with_chunks(&mut cursor, spec(), &leading).unwrap();
    writer.write_raw_interleaved(&audio()).unwrap();
    writer.write_chunk(*b"LIST", &adtl().to_bytes()).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let reader = WavReader::new(cursor).unwrap();
    let chunks = &reader.params().chunks;
    assert_eq!(Bext::from_chunk(by_id(chunks, b"bext")[0]), Some(bext()));
    assert_eq!(Cue::from_chunk(by_id(chunks, b"cue ")[0]), Some(cue()));
    assert_eq!(
        AdtlList::from_chunk(by_id(chunks, b"LIST")[0]),
        Some(adtl())
    );
    assert_eq!(reader.frames(), 48);
}
