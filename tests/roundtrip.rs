//! Round-trip and header tests for waveadapter.

use std::io::Cursor;

use audioadapter::{Adapter, AdapterMut};
use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::header::read_wav_header;
use waveadapter::{Chunk, RawSpec, SampleFormat, WavReader, WavSpec, WavWriter};

fn make_buffer(channels: usize, frames: usize) -> InterleavedOwned<f32> {
    let mut buf = InterleavedOwned::<f32>::new(0.0, channels, frames);
    for frame in 0..frames {
        for ch in 0..channels {
            // A simple ramp, distinct per channel, staying inside -1.0..1.0.
            let value = ((frame as f32) / (frames as f32)) * 0.5 - 0.25 + (ch as f32) * 0.01;
            buf.write_sample(ch, frame, &value);
        }
    }
    buf
}

fn roundtrip(format: SampleFormat, tolerance: f32) {
    let channels = 2;
    let frames = 64;
    let source = make_buffer(channels, frames);

    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: format,
        channel_mask: None,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.channels(), channels);
    assert_eq!(reader.sample_rate(), 48000);
    assert_eq!(reader.sample_format(), Some(format));
    assert_eq!(reader.frames(), frames);

    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);
    assert_eq!(restored.channels(), channels);

    for frame in 0..frames {
        for ch in 0..channels {
            let a = source.read_sample(ch, frame).unwrap();
            let b = restored.read_sample(ch, frame).unwrap();
            assert!(
                (a - b).abs() <= tolerance,
                "format {format:?}: frame {frame} ch {ch}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn roundtrip_all_formats() {
    // Integer formats lose precision according to their bit depth.
    roundtrip(SampleFormat::I16, 1.0 / 32767.0 * 2.0);
    roundtrip(SampleFormat::I24_3, 1.0 / 8_388_607.0 * 2.0);
    roundtrip(SampleFormat::I24_4, 1.0 / 8_388_607.0 * 2.0);
    roundtrip(SampleFormat::I32, 1.0 / 2_147_483_647.0 * 4.0);
    // Float formats are exact for these values.
    roundtrip(SampleFormat::F32, 0.0);
    roundtrip(SampleFormat::F64, 0.0);
}

#[test]
fn streaming_writer_produces_readable_file() {
    let channels = 1;
    let frames = 32;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 44100,
        sample_format: SampleFormat::F32,
        channel_mask: None,
    };

    // Streaming mode leaves the size fields at u32::MAX.
    let mut writer = WavWriter::new_streaming(Cursor::new(Vec::new()), spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    let cursor = writer.into_inner().unwrap();

    // The declared length is bogus, but read_all_to_float reads to EOF.
    let mut reader = WavReader::new(cursor).unwrap();
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);
    for frame in 0..frames {
        let a = source.read_sample(0, frame).unwrap();
        let b = restored.read_sample(0, frame).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn read_into_partial_buffer() {
    let channels = 2;
    let frames = 100;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 44100,
        sample_format: SampleFormat::F32,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let mut reader = WavReader::new(cursor).unwrap();
    let mut chunk = InterleavedOwned::<f32>::new(0.0, channels, 30);
    let got = reader.read_into_float(&mut chunk).unwrap();
    assert_eq!(got, 30);
    assert_eq!(reader.position(), 30);
    assert_eq!(reader.remaining(), 70);
    assert_eq!(
        chunk.read_sample(0, 5).unwrap(),
        source.read_sample(0, 5).unwrap()
    );
}

#[test]
fn header_roundtrip_offsets() {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 44100,
        sample_format: SampleFormat::I32,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    let data = InterleavedOwned::<f32>::new(0.0, 2, 5);
    writer.write_float_buffer(&data).unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let params = read_wav_header(&mut cursor).unwrap();
    assert_eq!(params.sample_format, Some(SampleFormat::I32));
    assert_eq!(params.channels, 2);
    assert_eq!(params.sample_rate, 44100);
    assert_eq!(params.data_offset, 44);
    // 5 frames * 2 channels * 4 bytes.
    assert_eq!(params.data_length, 40);
}

#[test]
fn writes_i24_4_as_strict_extensible() {
    let channels = 2;
    let frames = 8;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 96000,
        sample_format: SampleFormat::I24_4,
        channel_mask: None,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();

    let rd16 = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let rd32 = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    // The fmt chunk is a 40-byte WAVE_FORMAT_EXTENSIBLE, strict-spec form.
    assert_eq!(&bytes[12..16], b"fmt ");
    assert_eq!(rd32(16), 40, "fmt chunk is the 40-byte extensible form");
    assert_eq!(rd16(20), 0xFFFE, "format tag is WAVE_FORMAT_EXTENSIBLE");
    assert_eq!(rd16(22), channels as u16, "channels");
    assert_eq!(rd16(32), 8, "block alignment is channels * 4 bytes");
    assert_eq!(rd16(34), 32, "wBitsPerSample carries the 32-bit container");
    assert_eq!(rd16(36), 22, "cbSize");
    assert_eq!(rd16(38), 24, "wValidBitsPerSample carries the real 24 bits");
    // Extensible (non-PCM tag) means a fact chunk sits between fmt and data.
    assert_eq!(
        &bytes[60..64],
        b"fact",
        "fact chunk follows the 40-byte fmt"
    );
    assert_eq!(rd32(64), 4, "fact body is 4 bytes");
    assert_eq!(rd32(68), frames as u32, "fact carries the frame count");
    assert_eq!(&bytes[72..76], b"data", "data chunk follows the fact chunk");

    // And it reads back as I24_4 with the audio intact and data starting at 80.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::I24_4));
    assert_eq!(reader.channels(), channels);
    assert_eq!(reader.params().data_offset, 80);

    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);
    let tolerance = 1.0 / 8_388_607.0 * 2.0;
    for frame in 0..frames {
        for ch in 0..channels {
            let a = source.read_sample(ch, frame).unwrap();
            let b = restored.read_sample(ch, frame).unwrap();
            assert!(
                (a - b).abs() <= tolerance,
                "frame {frame} ch {ch}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn writes_multichannel_as_extensible() {
    // I16 is normally a plain 16-byte fmt chunk, but more than two channels
    // forces the WAVE_FORMAT_EXTENSIBLE form per the spec recommendation.
    let channels = 4;
    let frames = 8;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();

    let rd16 = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let rd32 = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    assert_eq!(&bytes[12..16], b"fmt ");
    assert_eq!(rd32(16), 40, "fmt chunk is the 40-byte extensible form");
    assert_eq!(rd16(20), 0xFFFE, "format tag is WAVE_FORMAT_EXTENSIBLE");
    assert_eq!(rd16(22), channels as u16, "channels");
    assert_eq!(rd16(32), 8, "block alignment is channels * 2 bytes");
    assert_eq!(rd16(34), 16, "wBitsPerSample carries the 16-bit container");
    assert_eq!(rd16(38), 16, "wValidBitsPerSample matches the container");
    assert_eq!(bytes[44], 1, "subformat GUID is KSDATAFORMAT_SUBTYPE_PCM");
    // Extensible (non-PCM tag) means a fact chunk sits between fmt and data,
    // even though the subformat here is PCM.
    assert_eq!(
        &bytes[60..64],
        b"fact",
        "fact chunk follows the 40-byte fmt"
    );
    assert_eq!(rd32(68), frames as u32, "fact carries the frame count");
    assert_eq!(&bytes[72..76], b"data", "data chunk follows the fact chunk");

    // And it reads back as plain I16 with the audio intact.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::I16));
    assert_eq!(reader.channels(), channels);

    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);
    let tolerance = 1.0 / 32_767.0 * 2.0;
    for frame in 0..frames {
        for ch in 0..channels {
            let a = source.read_sample(ch, frame).unwrap();
            let b = restored.read_sample(ch, frame).unwrap();
            assert!(
                (a - b).abs() <= tolerance,
                "frame {frame} ch {ch}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn float_write_emits_fact_chunk() {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::F32,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    let data = make_buffer(2, 10);
    writer.write_float_buffer(&data).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();

    let rd32 = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    // RIFF/WAVE (12) + fmt chunk (8 + 16). The fact chunk follows the fmt chunk.
    assert_eq!(
        &bytes[36..40],
        b"fact",
        "fact chunk follows the 16-byte fmt"
    );
    assert_eq!(rd32(40), 4, "fact body is 4 bytes");
    assert_eq!(rd32(44), 10, "fact carries the sample-frame count");
    assert_eq!(&bytes[48..52], b"data", "data chunk follows the fact chunk");

    // It reads back cleanly, with the fact chunk surfaced as a raw chunk.
    let cursor = Cursor::new(bytes);
    let reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::F32));
    assert_eq!(reader.frames(), 10);
    assert!(reader.params().chunks.iter().any(|c| &c.id == b"fact"));
}

#[test]
fn pcm_write_has_no_fact_chunk() {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&make_buffer(1, 4)).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();
    // 16-byte fmt chunk is followed directly by the data chunk, no fact.
    assert_eq!(&bytes[36..40], b"data");
}

#[test]
fn leading_and_trailing_chunks_roundtrip() {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    // An odd-length leading chunk exercises the pad byte, a trailing chunk after
    // odd-length data exercises the data pad byte.
    let leading = vec![Chunk {
        id: *b"bext",
        data: vec![1, 2, 3],
    }];

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_with_chunks(&mut cursor, spec, &leading).unwrap();
    // An odd number of data bytes forces a pad byte before the trailing chunk.
    writer.write_raw_interleaved(&[0u8; 5]).unwrap();
    writer.write_chunk(*b"LIST", b"INFOIART").unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let params = read_wav_header(&mut cursor).unwrap();
    assert_eq!(params.channels, 1);
    let ids: Vec<[u8; 4]> = params.chunks.iter().map(|c| c.id).collect();
    assert!(ids.contains(b"bext"), "leading chunk present: {ids:?}");
    assert!(ids.contains(b"LIST"), "trailing chunk present: {ids:?}");
    let bext = params.chunks.iter().find(|c| &c.id == b"bext").unwrap();
    assert_eq!(bext.data, vec![1, 2, 3]);
    let list = params.chunks.iter().find(|c| &c.id == b"LIST").unwrap();
    assert_eq!(list.data, b"INFOIART");
}

#[test]
fn reserved_chunk_id_is_rejected() {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let bad = vec![Chunk {
        id: *b"data",
        data: vec![0],
    }];
    let result = WavWriter::new_with_chunks(Cursor::new(Vec::new()), spec, &bad);
    assert!(matches!(result, Err(waveadapter::WavError::InvalidSpec(_))));
}

#[test]
fn audio_after_trailing_chunk_is_rejected() {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let mut writer = WavWriter::new(Cursor::new(Vec::new()), spec).unwrap();
    writer.write_raw_interleaved(&[0u8; 4]).unwrap();
    writer.write_chunk(*b"LIST", b"x").unwrap();
    let result = writer.write_raw_interleaved(&[0u8; 4]);
    assert!(matches!(result, Err(waveadapter::WavError::InvalidSpec(_))));
}

#[test]
fn rf64_roundtrip() {
    let channels = 2;
    let frames = 50;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::F32,
        channel_mask: None,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_rf64(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();

    let rd32 = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let rd64 = |o: usize| {
        u64::from_le_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
            bytes[o + 4],
            bytes[o + 5],
            bytes[o + 6],
            bytes[o + 7],
        ])
    };

    // RF64 form id with a 0xFFFFFFFF RIFF size, then WAVE and a ds64 chunk.
    assert_eq!(&bytes[0..4], b"RF64");
    assert_eq!(rd32(4), u32::MAX, "RIFF size field is the marker");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[12..16], b"ds64", "ds64 chunk comes first");
    assert_eq!(rd32(16), 28, "ds64 body is 28 bytes (no table)");

    let data_bytes = (frames * channels * 4) as u64;
    let riff_size = rd64(20);
    assert_eq!(rd64(28), data_bytes, "ds64 dataSize");
    assert_eq!(rd64(36), frames as u64, "ds64 sampleCount");
    assert_eq!(rd32(44), 0, "ds64 tableLength is zero");
    assert_eq!(
        riff_size,
        bytes.len() as u64 - 8,
        "ds64 riffSize matches file"
    );

    // fmt follows ds64; no fact chunk is written for RF64. The data chunk's
    // 32-bit size field carries the marker.
    // ds64 ends at 48; the 16-byte-core fmt chunk (8 + 16) runs to 72.
    assert_eq!(&bytes[48..52], b"fmt ");
    assert_eq!(&bytes[72..76], b"data", "data follows fmt, no fact chunk");
    assert_eq!(rd32(76), u32::MAX, "data size field is the marker");

    // It reads back with the resolved 64-bit data length and intact audio.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::F32));
    assert_eq!(reader.channels(), channels);
    assert_eq!(reader.frames(), frames);
    assert_eq!(reader.params().data_length, data_bytes as usize);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);
    for frame in 0..frames {
        for ch in 0..channels {
            assert_eq!(
                source.read_sample(ch, frame).unwrap(),
                restored.read_sample(ch, frame).unwrap()
            );
        }
    }
}

#[test]
fn rf64_with_leading_chunk_roundtrips() {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let leading = vec![Chunk {
        id: *b"bext",
        data: vec![7u8; 10],
    }];
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_rf64_with_chunks(&mut cursor, spec, &leading).unwrap();
    writer.write_raw_interleaved(&[0u8; 8]).unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let params = read_wav_header(&mut cursor).unwrap();
    assert_eq!(params.data_length, 8);
    let bext = params.chunks.iter().find(|c| &c.id == b"bext").unwrap();
    assert_eq!(bext.data, vec![7u8; 10]);
    // The ds64 chunk is consumed by the parser, not surfaced as a raw chunk.
    assert!(!params.chunks.iter().any(|c| &c.id == b"ds64"));
}

#[test]
fn bw64_is_read_like_rf64() {
    // BW64 is structurally identical to RF64. Write an RF64 file and swap the
    // form id to BW64; it must read back the same.
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let source = make_buffer(2, 16);
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_rf64(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    let mut bytes = cursor.into_inner();
    bytes[0..4].copy_from_slice(b"BW64");

    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.channels(), 2);
    assert_eq!(reader.frames(), 16);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), 16);
}

#[test]
fn raw_writer_roundtrips_an_unmodeled_format() {
    // 8-bit unsigned PCM: a valid format this crate does not model. Two channels,
    // so one frame is two bytes.
    let spec = RawSpec {
        format_code: 1,
        channels: 2,
        sample_rate: 22050,
        bits_per_sample: 8,
        block_align: 2,
    };
    let samples: Vec<u8> = (0..32u8).collect();

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_raw(&mut cursor, spec).unwrap();
    writer.write_raw_interleaved(&samples).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    // The format is not interpreted, but the raw fmt fields survive.
    assert_eq!(reader.sample_format(), None);
    assert_eq!(reader.channels(), 2);
    assert_eq!(reader.sample_rate(), 22050);
    assert_eq!(reader.params().format_code, 1);
    assert_eq!(reader.params().bits_per_sample, 8);
    assert_eq!(reader.params().block_align, 2);
    assert_eq!(reader.frames(), 16);

    // No `fact` chunk is written for a raw format.
    assert!(!reader.params().chunks.iter().any(|c| &c.id == b"fact"));

    // The bytes come back untouched through the raw read path.
    let mut out = Vec::new();
    let frames_read = reader.read_raw_interleaved(16, &mut out).unwrap();
    assert_eq!(frames_read, 16);
    assert_eq!(out, samples);
}

#[test]
fn float_read_on_raw_format_errors() {
    let spec = RawSpec {
        format_code: 1,
        channels: 1,
        sample_rate: 8000,
        bits_per_sample: 8,
        block_align: 1,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_raw(&mut cursor, spec).unwrap();
    writer
        .write_raw_interleaved(&[0, 64, 128, 192, 255])
        .unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    assert!(matches!(
        reader.read_all_to_float::<f32>(),
        Err(waveadapter::WavError::UnsupportedFormat(_))
    ));
}

#[test]
fn float_write_on_raw_writer_errors() {
    let spec = RawSpec {
        format_code: 1,
        channels: 1,
        sample_rate: 8000,
        bits_per_sample: 8,
        block_align: 1,
    };
    let mut writer = WavWriter::new_raw(Cursor::new(Vec::new()), spec).unwrap();
    let buf = InterleavedOwned::<f32>::new(0.0, 1, 4);
    assert!(matches!(
        writer.write_float_buffer(&buf),
        Err(waveadapter::WavError::UnsupportedFormat(_))
    ));
}

#[test]
fn invalid_header_is_rejected() {
    let mut cursor = Cursor::new(vec![0u8; 100]);
    assert!(read_wav_header(&mut cursor).is_err());
}

#[test]
fn oversized_spec_is_rejected() {
    // A channel count that does not fit in the 16-bit header field.
    let spec = WavSpec {
        channels: 70_000,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let result = WavWriter::new(Cursor::new(Vec::new()), spec);
    assert!(matches!(result, Err(waveadapter::WavError::InvalidSpec(_))));

    // Zero channels is also rejected.
    let spec = WavSpec {
        channels: 0,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    assert!(matches!(
        WavWriter::new(Cursor::new(Vec::new()), spec),
        Err(waveadapter::WavError::InvalidSpec(_))
    ));
}

#[test]
fn reader_seek_to_frame() {
    let channels = 2;
    let frames = 64;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let mut reader = WavReader::new(cursor).unwrap();
    reader.seek_to_frame(40).unwrap();
    assert_eq!(reader.position(), 40);
    assert_eq!(reader.remaining(), frames - 40);

    let mut target = InterleavedOwned::<f32>::new(0.0, channels, 4);
    let got = reader.read_into_float(&mut target).unwrap();
    assert_eq!(got, 4);
    for frame in 0..4 {
        for ch in 0..channels {
            let a = source.read_sample(ch, 40 + frame).unwrap();
            let b = target.read_sample(ch, frame).unwrap();
            assert!((a - b).abs() <= 1e-4, "frame {frame} ch {ch}: {a} vs {b}");
        }
    }

    // Seeking backwards re-reads from the new position.
    reader.seek_to_frame(0).unwrap();
    assert_eq!(reader.position(), 0);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);

    // Seeking past the end clamps to the frame count.
    reader.seek_to_frame(1000).unwrap();
    assert_eq!(reader.position(), frames);
    assert_eq!(reader.remaining(), 0);
}

#[test]
fn writer_seek_to_frame_overwrites() {
    let channels = 1;
    let frames = 32;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();

    // Overwrite a few frames in the middle, then confirm the file length is
    // unchanged and the data outside the patched region is intact.
    let patch = make_buffer(channels, 4);
    writer.seek_to_frame(10).unwrap();
    writer.write_float_buffer(&patch).unwrap();
    assert_eq!(writer.data_bytes(), (frames * channels * 2) as u64);
    writer.finalize().unwrap();
    cursor.set_position(0);

    let mut reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.frames(), frames);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    for frame in 0..frames {
        let expected = if (10..14).contains(&frame) {
            patch.read_sample(0, frame - 10).unwrap()
        } else {
            source.read_sample(0, frame).unwrap()
        };
        let got = restored.read_sample(0, frame).unwrap();
        assert!(
            (expected - got).abs() <= 1e-4,
            "frame {frame}: {expected} vs {got}"
        );
    }
}

#[test]
fn channel_mask_roundtrips() {
    // Stereo with FRONT_LEFT | FRONT_RIGHT (0x3). A non-zero mask forces the
    // extensible header even for stereo, and reading it back yields the value.
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: Some(0x3),
    };
    let source = make_buffer(2, 16);
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let params = read_wav_header(&mut cursor).unwrap();
    assert_eq!(params.channel_mask, Some(0x3));
    assert_eq!(params.format_code, 0xFFFE);
}

#[test]
fn no_channel_mask_leaves_plain_header() {
    // Without a mask, stereo PCM stays a plain WAVEFORMAT and reports no mask.
    let spec = WavSpec::new(2, 48000, SampleFormat::I16);
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&make_buffer(2, 16)).unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let params = read_wav_header(&mut cursor).unwrap();
    assert_eq!(params.channel_mask, None);
    assert_eq!(params.format_code, 1);
}

#[test]
fn channel_mask_bit_count_mismatch_is_rejected() {
    // Three bits set but only two channels.
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: Some(0x7),
    };
    assert!(matches!(
        WavWriter::new(Cursor::new(Vec::new()), spec),
        Err(waveadapter::WavError::InvalidSpec(_))
    ));

    // A zero mask is always accepted ("unspecified").
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: Some(0),
    };
    assert!(WavWriter::new(Cursor::new(Vec::new()), spec).is_ok());
}
