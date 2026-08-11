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
    assert_eq!(reader.params().format_code, 0x11);
    assert_eq!(reader.params().bits_per_sample, 4);
    assert!(reader.params().block_align >= 1);

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
    assert_eq!(reader.params().format_code, 0x31);
    assert_eq!(reader.params().bits_per_sample, 0, "GSM declares zero bits");
    assert_eq!(reader.params().block_align, 65, "odd block alignment");
    assert_eq!(reader.params().frame_bytes(), 65, "framing off block_align");
    assert_eq!(reader.channels(), 1);
    // 195 bytes of data, three 65-byte blocks. The odd size means the data
    // chunk carries a pad byte that must not be counted as audio.
    assert_eq!(reader.params().data_length, 195);
    assert_eq!(reader.frames(), 3);
    // The fact chunk before the audio comes through as an opaque blob.
    assert!(reader.params().chunks.iter().any(|c| &c.id == b"fact"));

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
