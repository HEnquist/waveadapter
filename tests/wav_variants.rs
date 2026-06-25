//! Reading the battery of wav file variants produced by
//! `generate_wav_variants.py` (committed under `tests/wav_variants/`).
//!
//! The fixtures cover the range of variation a wav parser has to handle, one
//! variation each. Every file we support must parse and decode to the expected
//! shape; the deliberately unsupported ones (8-bit audio, which audioadapter has
//! no sample type for, and a header with no data chunk) must be rejected with an
//! error rather than panicking.

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
];

/// Files our reader is expected to reject, with the reason they are unsupported.
const REJECTED: &[(&str, &str)] = &[
    ("mono_8bit_unsigned", "8-bit audio is not supported"),
    ("odd_sized_data_with_pad", "8-bit audio is not supported"),
    ("odd_sized_data_missing_pad", "8-bit audio is not supported"),
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
    // An 8-bit file has no audioadapter sample type, so the float path rejects
    // it, but it still parses and its audio is readable as raw bytes.
    let file = std::fs::File::open(fixture("mono_8bit_unsigned")).unwrap();
    let mut reader = WavReader::new(file).expect("8-bit file should parse");
    assert_eq!(
        reader.sample_format(),
        None,
        "8-bit should be uninterpreted"
    );
    assert_eq!(reader.params().bits_per_sample, 8);
    assert!(reader.params().block_align >= 1);

    let mut bytes = Vec::new();
    let frames = reader
        .read_raw_interleaved(reader.frames(), &mut bytes)
        .unwrap();
    assert!(frames > 0, "expected to read some raw frames");
    assert_eq!(bytes.len(), frames * reader.params().frame_bytes());
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
