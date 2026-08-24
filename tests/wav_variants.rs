//! Reading the battery of wav file variants produced by
//! `generate_wav_variants.py` (committed under `tests/wav_variants/`).
//!
//! The fixtures cover the range of variation a wav parser has to handle, one
//! variation each. Every file we support must parse and decode to the expected
//! shape; the deliberately unsupported ones (IMA ADPCM, which audioadapter has
//! no sample type for, and a header with no data chunk) must be rejected with
//! an error rather than panicking.

use std::path::PathBuf;

use audioadapter::Adapter;
use waveadapter::{SampleFormat, WavReader};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/wav_variants")
        .join(format!("{name}.wav"))
}

struct Expect {
    format: SampleFormat,
    channels: usize,
    /// The number of frames actually decodable from the data, which for the
    /// lying-length cases differs from what the header declares.
    frames: usize,
}

/// The files we expect to read successfully, with the shape we expect to get.
const READABLE: &[(&str, Expect)] = &[
    (
        "baseline_16bit_stereo",
        Expect {
            format: SampleFormat::I16,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "canonical_cd_16bit",
        Expect {
            format: SampleFormat::I16,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "canonical_24bit_48k",
        Expect {
            format: SampleFormat::I24_3,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "int32",
        Expect {
            format: SampleFormat::I32,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "float64",
        Expect {
            format: SampleFormat::F64,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "fmt_size_18_cbsize_zero",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "24bit_packed",
        Expect {
            format: SampleFormat::I24_3,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "24bit_padded",
        Expect {
            format: SampleFormat::I24_4,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "float32",
        Expect {
            format: SampleFormat::F32,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "extensible_24in32_5point1",
        Expect {
            format: SampleFormat::I24_4,
            channels: 6,
            frames: 20,
        },
    ),
    (
        "extensible_float",
        Expect {
            format: SampleFormat::F32,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "float32_waveformatex18",
        Expect {
            format: SampleFormat::F32,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "extensible_16bit",
        Expect {
            format: SampleFormat::I16,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "extensible_24bit_packed",
        Expect {
            format: SampleFormat::I24_3,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "extensible_int32",
        Expect {
            format: SampleFormat::I32,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "extensible_float64",
        Expect {
            format: SampleFormat::F64,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "extensible_24in32_strict",
        Expect {
            format: SampleFormat::I24_4,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "junk_before_fmt",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "list_info_between_fmt_and_data",
        Expect {
            format: SampleFormat::I16,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "data_before_fmt",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "riff_size_too_large",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "data_size_too_large",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "data_size_streaming_placeholder",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "trailing_junk_after_data",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "trailing_chunk_after_data",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "zero_length_data",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 0,
        },
    ),
    (
        "multiple_data_chunks",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "huge_channel_count",
        Expect {
            format: SampleFormat::I16,
            channels: 16,
            frames: 20,
        },
    ),
    (
        "rf64_16bit_stereo",
        Expect {
            format: SampleFormat::I16,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "bw64_16bit_stereo",
        Expect {
            format: SampleFormat::I16,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "rf64_float32_real_size",
        Expect {
            format: SampleFormat::F32,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "rf64_chunk_size_in_table",
        Expect {
            format: SampleFormat::I16,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "mono_8bit_unsigned",
        Expect {
            format: SampleFormat::U8,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "extensible_8bit",
        Expect {
            format: SampleFormat::U8,
            channels: 2,
            frames: 20,
        },
    ),
    // Odd-length data, with and without the RIFF pad byte. Both are 8-bit mono
    // with an odd frame count, which is what makes the payload odd-sized.
    (
        "odd_sized_data_with_pad",
        Expect {
            format: SampleFormat::U8,
            channels: 1,
            frames: 21,
        },
    ),
    (
        "odd_sized_data_missing_pad",
        Expect {
            format: SampleFormat::U8,
            channels: 1,
            frames: 21,
        },
    ),
    // G.711, one companded byte per sample, in both the plain 18-byte
    // WAVEFORMATEX form and the extensible form matched by subtype GUID.
    (
        "mulaw_mono",
        Expect {
            format: SampleFormat::MULAW,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "mulaw_stereo",
        Expect {
            format: SampleFormat::MULAW,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "alaw_mono",
        Expect {
            format: SampleFormat::ALAW,
            channels: 1,
            frames: 20,
        },
    ),
    (
        "alaw_stereo",
        Expect {
            format: SampleFormat::ALAW,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "extensible_alaw",
        Expect {
            format: SampleFormat::ALAW,
            channels: 2,
            frames: 20,
        },
    ),
    (
        "extensible_mulaw",
        Expect {
            format: SampleFormat::MULAW,
            channels: 2,
            frames: 20,
        },
    ),
];

/// Files our reader is expected to reject, with the reason they are unsupported.
const REJECTED: &[(&str, &str)] = &[
    (
        "ima_adpcm_mono",
        "block-compressed ADPCM is not decodable, only readable as raw",
    ),
    (
        "gsm610_mono",
        "GSM 6.10 is not decodable, only readable as raw",
    ),
    (
        "ms_adpcm_stereo",
        "MS ADPCM is not decodable, only readable as raw",
    ),
    (
        "odd_length_fmt_extension",
        "an unassigned format tag is not decodable, only readable as raw",
    ),
    (
        "extensible_too_short",
        "extensible without its subformat GUID names no format we know",
    ),
    ("empty_riff_no_data_chunk", "no data chunk present"),
];

#[test]
fn reads_all_supported_variants() {
    for (name, exp) in READABLE {
        let path = fixture(name);
        let file =
            std::fs::File::open(&path).unwrap_or_else(|e| panic!("opening fixture {name}: {e}"));
        let mut reader =
            WavReader::new(file).unwrap_or_else(|e| panic!("parsing header of {name}: {e}"));

        assert_eq!(
            reader.sample_format(),
            Some(exp.format),
            "{name}: sample format"
        );
        assert_eq!(reader.channels(), exp.channels, "{name}: channel count");

        let buffer = reader
            .read_all_to_float::<f32>()
            .unwrap_or_else(|e| panic!("reading data of {name}: {e}"));
        assert_eq!(buffer.channels(), exp.channels, "{name}: buffer channels");
        assert_eq!(buffer.frames(), exp.frames, "{name}: decoded frame count");
    }
}

#[test]
fn rejects_unsupported_variants() {
    for (name, reason) in REJECTED {
        let path = fixture(name);
        let file =
            std::fs::File::open(&path).unwrap_or_else(|e| panic!("opening fixture {name}: {e}"));
        // Either header parsing or the first read must fail, and it must not panic.
        let result = WavReader::new(file).and_then(|mut r| {
            r.read_all_to_float::<f32>()?;
            Ok(())
        });
        assert!(
            result.is_err(),
            "{name} should be rejected ({reason}) but was read successfully"
        );
    }
}

#[test]
fn unsupported_format_reads_as_raw() {
    // IMA ADPCM has no audioadapter sample type, so the float path rejects it,
    // but it still parses and its audio is readable as raw bytes.
    let file = std::fs::File::open(fixture("ima_adpcm_mono")).unwrap();
    let mut reader = WavReader::new(file).expect("ADPCM file should parse");
    assert_eq!(
        reader.sample_format(),
        None,
        "ADPCM should be uninterpreted"
    );
    assert_eq!(reader.params().fmt.format_code, 0x11);
    assert_eq!(reader.params().fmt.bits_per_sample, 4);
    assert!(reader.params().fmt.block_align >= 1);

    let mut bytes = Vec::new();
    let frames = reader
        .read_raw_interleaved(reader.frames(), &mut bytes)
        .unwrap();
    assert!(frames > 0, "expected to read some raw frames");
    assert_eq!(bytes.len(), frames * reader.params().frame_bytes());
}

#[test]
fn gsm610_survives_the_raw_path() {
    // GSM 6.10 has the most hostile fmt chunk in common use: wBitsPerSample is
    // zero, nBlockAlign is an odd 65, and a cbSize of 2 makes the chunk 20
    // bytes rather than any of the three standard sizes. Nothing about it may
    // be inferred from the bit depth, and the framing has to come from the
    // block alignment alone.
    let file = std::fs::File::open(fixture("gsm610_mono")).unwrap();
    let mut reader = WavReader::new(file).expect("GSM file should parse");

    assert_eq!(reader.sample_format(), None, "GSM is uninterpreted");
    assert_eq!(reader.params().fmt.format_code, 0x31);
    assert_eq!(
        reader.params().fmt.bits_per_sample,
        0,
        "GSM declares zero bits"
    );
    assert_eq!(reader.params().fmt.block_align, 65, "odd block alignment");
    assert_eq!(reader.params().frame_bytes(), 65, "framing off block_align");
    assert_eq!(reader.channels(), 1);
    // 195 bytes of data, three 65-byte blocks. The odd size means the data
    // chunk carries a pad byte that must not be counted as audio.
    assert_eq!(reader.params().data_length, 195);
    assert_eq!(reader.frames(), 3);
    // The fact chunk is parsed, not left opaque, and it is the only place the
    // real frame count lives: `frames()` counts the three compressed blocks.
    assert_eq!(reader.params().fact_samples(), Some(960));
    assert!(!reader.params().chunks().any(|c| &c.id == b"fact"));

    let mut bytes = Vec::new();
    let blocks = reader
        .read_raw_interleaved(reader.frames(), &mut bytes)
        .unwrap();
    assert_eq!(blocks, 3);
    assert_eq!(bytes.len(), 195);
    assert_eq!(bytes[0], 0xD0, "first block starts where it should");
    assert_eq!(bytes[65], 0xD1, "second block is 65 bytes in");
    assert_eq!(bytes[130], 0xD2, "third block is 130 bytes in");
}

/// Read a fixture, write it back out through the codec-facing path, and return
/// the two files' `fmt ` chunk bodies plus their `fact` counts.
/// The `fmt ` bytes and `fact` body of a fixture, and of that fixture rewritten
/// through the raw path: what a byte-exact rewrite has to keep identical.
type Rewritten = (Vec<u8>, Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>);

fn rewrite_raw(name: &str) -> Rewritten {
    use waveadapter::{Fact, WavWriter};

    let mut reader = WavReader::new(std::fs::File::open(fixture(name)).unwrap()).unwrap();
    let mut audio = Vec::new();
    reader
        .read_raw_interleaved(reader.frames(), &mut audio)
        .unwrap();
    let params = reader.params().clone();

    // The whole body, not just the count: a `fact` chunk may carry bytes after
    // it, and a byte-exact rewrite has to put those back too.
    let fact = params.fact.clone().map_or(Fact::None, Fact::Body);
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(params.fmt.clone())
        .unwrap()
        .fact(fact)
        .open(&mut cursor)
        .unwrap();
    writer.write_raw_interleaved(&audio).unwrap();
    writer.finalize().unwrap();

    let rewritten = WavReader::new(std::io::Cursor::new(cursor.into_inner())).unwrap();
    (
        params.fmt.to_bytes().unwrap(),
        rewritten.params().fmt.to_bytes().unwrap(),
        params.fact.clone(),
        rewritten.params().fact.clone(),
    )
}

#[test]
fn unmodeled_formats_rewrite_byte_for_byte() {
    // The point of the whole raw path: a codec built on top of this crate has to
    // be able to read a file, write it back, and have the container come out
    // identical. That means the format-specific `fmt ` extension and the `fact`
    // frame count both survive, neither of which the crate interprets.
    for name in ["gsm610_mono", "ms_adpcm_stereo", "ima_adpcm_mono"] {
        let (original, rewritten, fact_before, fact_after) = rewrite_raw(name);
        assert_eq!(original, rewritten, "{name}: fmt chunk changed on rewrite");
        assert_eq!(fact_before, fact_after, "{name}: fact body changed");
    }
}

#[test]
fn gsm610_rewrite_keeps_the_fields_no_one_can_recompute() {
    // Spelling out the two GSM fields that a fields-and-guesswork writer gets
    // wrong: `nAvgBytesPerSec` is 1625, not block_align * sample_rate (520000),
    // and `wSamplesPerBlock` (320) lives in the extension.
    let (original, rewritten, ..) = rewrite_raw("gsm610_mono");
    assert_eq!(original, rewritten);

    let fmt = waveadapter::FmtChunk::from_bytes(&rewritten).unwrap();
    assert_eq!(fmt.format_code, 0x31);
    assert_eq!(fmt.byte_rate, 1625, "not block_align * sample_rate");
    assert_eq!(fmt.block_align, 65);
    assert_eq!(fmt.bits_per_sample, 0);
    assert_eq!(
        fmt.extension.as_deref(),
        Some(&[0x40, 0x01][..]),
        "wSamplesPerBlock = 320"
    );
}

#[test]
fn odd_length_fmt_extension_keeps_the_file_aligned() {
    // Every standard fmt body is even (16, 18, 40), so the pad byte after an
    // odd one is never exercised by a normal file. Omit it and the parser, which
    // advances past every chunk assuming the pad is there, lands one byte off
    // and misreads everything after the fmt chunk.
    let file = std::fs::File::open(fixture("odd_length_fmt_extension")).unwrap();
    let mut reader = WavReader::new(file).expect("odd fmt extension should parse");
    assert_eq!(reader.params().fmt.format_code, 0x99);
    assert_eq!(
        reader.params().fmt.extension.as_deref(),
        Some(&[0xaa, 0xbb, 0xcc][..])
    );
    assert_eq!(reader.frames(), 20, "the data chunk was still found");

    // And it survives a rewrite, pad byte and all.
    let (original, rewritten, ..) = rewrite_raw("odd_length_fmt_extension");
    assert_eq!(original.len(), 21, "16-byte core, cbSize, three bytes");
    assert_eq!(original, rewritten);

    let mut bytes = Vec::new();
    reader
        .read_raw_interleaved(reader.frames(), &mut bytes)
        .unwrap();
    assert_eq!(bytes, (0..20).collect::<Vec<u8>>());
}

#[test]
fn extensible_without_its_extension_falls_back_to_raw() {
    // Format tag 0xFFFE promises a subformat GUID that is not there. That is a
    // format we cannot interpret, not a broken container, so it degrades to the
    // raw path like any other unknown format rather than failing the parse.
    let file = std::fs::File::open(fixture("extensible_too_short")).unwrap();
    let mut reader = WavReader::new(file).expect("a short extensible header should still parse");
    assert_eq!(reader.sample_format(), None);
    assert_eq!(
        reader.params().channel_mask(),
        None,
        "no extension, no mask"
    );
    assert!(!reader.params().fmt.is_extensible());
    assert_eq!(reader.frames(), 20);

    let mut bytes = Vec::new();
    assert_eq!(
        reader
            .read_raw_interleaved(reader.frames(), &mut bytes)
            .unwrap(),
        20
    );
}

#[test]
fn long_fmt_chunk_is_not_mistaken_for_extensible() {
    // MS ADPCM has a 50-byte fmt chunk: longer than the 40-byte extensible form,
    // but not extensible. Body offset 20 holds `wNumCoef` here, where a
    // WAVEFORMATEXTENSIBLE keeps `dwChannelMask`, so keying the mask off the
    // chunk length alone reports a coefficient count (7) as a speaker layout.
    // Only the format code may decide.
    let file = std::fs::File::open(fixture("ms_adpcm_stereo")).unwrap();
    let reader = WavReader::new(file).expect("MS ADPCM file should parse");

    assert_eq!(reader.sample_format(), None, "MS ADPCM is uninterpreted");
    assert_eq!(reader.params().fmt.format_code, 2);
    assert_eq!(reader.channels(), 2);
    assert_eq!(
        reader.params().channel_mask(),
        None,
        "a non-extensible header carries no channel mask, however long it is"
    );
}

#[test]
fn g711_decodes_to_the_documented_values() {
    // The code words the two laws define for silence and for full scale, so a
    // silent mu-law file does not come back as full scale negative and the
    // sign convention is not flipped. A-law has no code for exact silence, its
    // smallest positive magnitude is 8.
    let cases: &[(&str, u8, f32)] = &[
        ("mulaw_mono", 0xff, 0.0),
        ("mulaw_mono", 0x80, 32124.0 / 32768.0),
        ("mulaw_mono", 0x00, -32124.0 / 32768.0),
        ("alaw_mono", 0xd5, 8.0 / 32768.0),
        ("alaw_mono", 0xaa, 32256.0 / 32768.0),
        ("alaw_mono", 0x2a, -32256.0 / 32768.0),
    ];
    for (name, code, expected) in cases {
        // Rebuild the fixture with a payload of one repeated code word, so the
        // header comes from the generator but the audio is what we choose.
        let original = std::fs::read(fixture(name)).unwrap();
        let data_start = original.len() - 20;
        let mut bytes = original[..data_start].to_vec();
        bytes.extend(std::iter::repeat_n(*code, 20));

        let mut reader = WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        let decoded = reader.read_all_to_float::<f32>().unwrap();
        assert_eq!(decoded.frames(), 20);
        for frame in 0..20 {
            assert_eq!(
                decoded.read_sample(0, frame).unwrap(),
                *expected,
                "{name}: code word {code:#04x}"
            );
        }
    }
}

#[test]
fn every_fixture_is_covered() {
    // Guard against a fixture being added to the generator but forgotten here.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wav_variants");
    let on_disk = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .count();
    assert_eq!(
        on_disk,
        READABLE.len() + REJECTED.len(),
        "number of fixtures on disk does not match the cases covered by these tests"
    );
}

/// Build an RF64 file by hand: 16-bit mono PCM, `frames` frames of audio, a
/// `ds64` chunk declaring `ds64_count` frames, and optionally a legacy `fact`
/// chunk declaring a different one. No generator produces this combination,
/// since the crate's own RF64 writer never emits a `fact` chunk.
fn rf64_with_counts(frames: u32, ds64_count: u64, fact_count: Option<u32>) -> Vec<u8> {
    let data: Vec<u8> = (0..frames * 2).map(|i| i as u8).collect();

    let mut body = Vec::new();
    body.extend_from_slice(&[0u8; 8]); // riffSize, patched below
    body.extend_from_slice(&(data.len() as u64).to_le_bytes()); // dataSize
    body.extend_from_slice(&ds64_count.to_le_bytes()); // sampleCount
    body.extend_from_slice(&0u32.to_le_bytes()); // tableLength

    let mut file = Vec::new();
    file.extend_from_slice(b"RF64");
    file.extend_from_slice(&u32::MAX.to_le_bytes()); // size lives in ds64
    file.extend_from_slice(b"WAVE");

    file.extend_from_slice(b"ds64");
    file.extend_from_slice(&(body.len() as u32).to_le_bytes());
    file.extend_from_slice(&body);

    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&1u16.to_le_bytes()); // mono
    fmt.extend_from_slice(&44100u32.to_le_bytes());
    fmt.extend_from_slice(&88200u32.to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    file.extend_from_slice(b"fmt ");
    file.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    file.extend_from_slice(&fmt);

    if let Some(count) = fact_count {
        file.extend_from_slice(b"fact");
        file.extend_from_slice(&4u32.to_le_bytes());
        file.extend_from_slice(&count.to_le_bytes());
    }

    file.extend_from_slice(b"data");
    file.extend_from_slice(&u32::MAX.to_le_bytes()); // size lives in ds64
    file.extend_from_slice(&data);

    let riff_size = (file.len() - 8) as u64;
    let riff_size_offset = 12 + 8;
    file[riff_size_offset..riff_size_offset + 8].copy_from_slice(&riff_size.to_le_bytes());
    file
}

#[test]
fn rf64_sample_count_comes_from_ds64_not_a_legacy_fact_chunk() {
    // `ds64` is the only one of the two fields that is 64-bit, so on an RF64
    // file it wins: a `fact` chunk there can only ever hold a truncated or
    // stale copy. The 32-bit accessor still reports what the `fact` chunk says,
    // since that is what a byte-exact rewrite has to put back.
    let file = rf64_with_counts(20, 20, Some(9999));
    let reader = WavReader::new(std::io::Cursor::new(file)).unwrap();
    assert_eq!(reader.params().sample_count(), Some(20));
    assert_eq!(reader.params().fact_samples(), Some(9999));
    assert_eq!(reader.frames(), 20);
}

#[test]
fn a_zeroed_ds64_sample_count_falls_back_to_the_fact_chunk() {
    // Some writers fill in the ds64 sizes and leave `sampleCount` at zero. A
    // `fact` chunk is a better answer than a field nobody filled in.
    let file = rf64_with_counts(20, 0, Some(20));
    let reader = WavReader::new(std::io::Cursor::new(file)).unwrap();
    assert_eq!(reader.params().sample_count(), Some(20));

    // With nothing to fall back to, the zero stands: the file did declare it.
    let file = rf64_with_counts(20, 0, None);
    let reader = WavReader::new(std::io::Cursor::new(file)).unwrap();
    assert_eq!(reader.params().sample_count(), Some(0));

    // And a plain RIFF file is unaffected, having no ds64 chunk at all: its
    // count comes from `fact` as before, or is absent when the file has none.
    let reader = WavReader::new(std::fs::File::open(fixture("gsm610_mono")).unwrap()).unwrap();
    assert_eq!(reader.params().sample_count(), Some(960));
    let reader = WavReader::new(std::fs::File::open(fixture("float32")).unwrap()).unwrap();
    assert_eq!(reader.params().sample_count(), None);
}

#[test]
fn a_block_align_that_is_not_a_multiple_of_the_channels_stays_raw() {
    // 16-bit stereo PCM with nBlockAlign = 5 is malformed: the field says five
    // bytes per frame where the depth and channel count say four. Dividing and
    // rounding down calls it I16 and reads four-byte frames, misaligned against
    // the framing the file declares from the second frame on. It is a format we
    // cannot interpret, so it belongs on the raw path with nBlockAlign taken as
    // stated.
    let data: Vec<u8> = (0..20).collect(); // four 5-byte frames
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&2u16.to_le_bytes()); // stereo
    fmt.extend_from_slice(&44100u32.to_le_bytes());
    fmt.extend_from_slice(&220500u32.to_le_bytes());
    fmt.extend_from_slice(&5u16.to_le_bytes()); // nBlockAlign, not 2 * 2
    fmt.extend_from_slice(&16u16.to_le_bytes());

    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&0u32.to_le_bytes()); // patched below
    file.extend_from_slice(b"WAVE");
    file.extend_from_slice(b"fmt ");
    file.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    file.extend_from_slice(&fmt);
    file.extend_from_slice(b"data");
    file.extend_from_slice(&(data.len() as u32).to_le_bytes());
    file.extend_from_slice(&data);
    let riff_size = (file.len() - 8) as u32;
    file[4..8].copy_from_slice(&riff_size.to_le_bytes());

    let mut reader = WavReader::new(std::io::Cursor::new(file)).expect("it should still parse");
    assert_eq!(reader.sample_format(), None, "not I16");
    assert_eq!(reader.params().frame_bytes(), 5, "nBlockAlign as stated");
    assert_eq!(reader.frames(), 4);
    assert!(matches!(
        reader.read_all_to_float::<f32>(),
        Err(waveadapter::WavError::UnsupportedFormat(_))
    ));

    let mut bytes = Vec::new();
    assert_eq!(
        reader
            .read_raw_interleaved(reader.frames(), &mut bytes)
            .unwrap(),
        4
    );
    assert_eq!(bytes, (0..20).collect::<Vec<u8>>());
}

#[test]
fn a_seventeen_byte_fmt_body_comes_back_as_the_bare_core() {
    // 17 bytes is the one fmt length the typed chunk cannot represent: half a
    // cbSize field and nothing after it. Pinning the deliberate choice here, so
    // it reads as a decision rather than an oversight: the file parses, the
    // stray byte is dropped, and the chunk re-encodes as the 16-byte core.
    // Rejecting it would fail a file that is otherwise perfectly readable, and
    // keeping the byte would need a public field for a half-written cbSize,
    // which this crate derives and never stores.
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&44100u32.to_le_bytes());
    fmt.extend_from_slice(&88200u32.to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    fmt.push(0x00); // the lone low byte of cbSize
    assert_eq!(fmt.len(), 17);

    let data: Vec<u8> = (0..40).collect();
    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&0u32.to_le_bytes()); // patched below
    file.extend_from_slice(b"WAVE");
    file.extend_from_slice(b"fmt ");
    file.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    file.extend_from_slice(&fmt);
    file.push(0); // RIFF pad byte after the odd-length body
    file.extend_from_slice(b"data");
    file.extend_from_slice(&(data.len() as u32).to_le_bytes());
    file.extend_from_slice(&data);
    let riff_size = (file.len() - 8) as u32;
    file[4..8].copy_from_slice(&riff_size.to_le_bytes());

    let reader = WavReader::new(std::io::Cursor::new(file)).expect("it should still parse");
    let fmt = &reader.params().fmt;
    assert_eq!(reader.sample_format(), Some(SampleFormat::I16));
    assert_eq!(reader.frames(), 20, "the data chunk was still found");
    assert_eq!(fmt.extension, None, "not a form the spec defines");
    assert_eq!(
        fmt.to_bytes().unwrap().len(),
        16,
        "re-encoded as the bare core"
    );
}

#[test]
fn a_second_ds64_chunk_is_ignored() {
    // The ds64 sizes frame every chunk after them, so a duplicate is not just a
    // stray field: honoring it would re-point the audio and change the frame
    // count. First one wins, as for a second `fmt `, `data` or `fact`.
    let base = rf64_with_counts(20, 20, None);

    // A second ds64 right after the first, claiming a shorter data chunk and a
    // different sample count.
    let mut liar = Vec::new();
    liar.extend_from_slice(&0u64.to_le_bytes()); // riffSize, unused on read
    liar.extend_from_slice(&8u64.to_le_bytes()); // dataSize
    liar.extend_from_slice(&9999u64.to_le_bytes()); // sampleCount
    liar.extend_from_slice(&0u32.to_le_bytes()); // tableLength
    let mut chunk = Vec::new();
    chunk.extend_from_slice(b"ds64");
    chunk.extend_from_slice(&(liar.len() as u32).to_le_bytes());
    chunk.extend_from_slice(&liar);

    let first_ds64_end = 12 + 8 + 28;
    assert_eq!(&base[12..16], b"ds64");
    let mut file = base[..first_ds64_end].to_vec();
    file.extend_from_slice(&chunk);
    file.extend_from_slice(&base[first_ds64_end..]);

    let reader = WavReader::new(std::io::Cursor::new(file)).unwrap();
    assert_eq!(
        reader.params().data_length,
        40,
        "40 bytes, not the liar's 8"
    );
    assert_eq!(reader.frames(), 20);
    assert_eq!(reader.params().sample_count(), Some(20));
    assert!(
        !reader.params().chunks().any(|c| &c.id == b"ds64"),
        "a dropped ds64 is not passed through as an opaque chunk either"
    );
}
