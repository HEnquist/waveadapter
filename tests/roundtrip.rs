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
    roundtrip(SampleFormat::U8, 1.0 / 127.0 * 2.0);
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
fn writes_8bit_as_unsigned_centered_at_128() {
    // Wav 8-bit PCM is unsigned, unlike every deeper depth, so check the bytes
    // that land on disk rather than only the float roundtrip.
    let spec = WavSpec {
        channels: 1,
        sample_rate: 8000,
        sample_format: SampleFormat::U8,
        channel_mask: None,
    };
    let mut source = InterleavedOwned::<f32>::new(0.0, 1, 3);
    source.write_sample(0, 0, &-1.0);
    source.write_sample(0, 1, &0.0);
    source.write_sample(0, 2, &1.0);

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    // +1.0 clips to 255, the same asymmetry the signed formats have at +1.0.
    assert_eq!(
        writer.write_float_buffer(&source).unwrap(),
        1,
        "the +1.0 sample clips"
    );
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();

    // Plain 16-byte fmt, no fact chunk: 8-bit PCM is unambiguous.
    assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1, "format tag");
    assert_eq!(u16::from_le_bytes([bytes[32], bytes[33]]), 1, "block align");
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 8, "bit depth");
    assert_eq!(&bytes[36..40], b"data");
    assert_eq!(&bytes[44..47], &[0, 128, 255], "unsigned, centered at 128");

    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::U8));
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.read_sample(0, 0).unwrap(), -1.0);
    assert_eq!(restored.read_sample(0, 1).unwrap(), 0.0);
    assert_eq!(restored.read_sample(0, 2).unwrap(), 127.0 / 128.0);
}

#[test]
fn writes_multichannel_8bit_as_extensible() {
    // More than two channels forces the extensible form, so the 8-bit subformat
    // has to survive the GUID round trip too.
    let channels = 4;
    let frames = 8;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 22050,
        sample_format: SampleFormat::U8,
        channel_mask: None,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();

    assert_eq!(
        u16::from_le_bytes([bytes[20], bytes[21]]),
        0xFFFE,
        "format tag is WAVE_FORMAT_EXTENSIBLE"
    );
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 8, "bit depth");
    assert_eq!(
        u16::from_le_bytes([bytes[38], bytes[39]]),
        8,
        "wValidBitsPerSample"
    );

    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::U8));
    assert_eq!(reader.channels(), channels);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);
    let tolerance = 1.0 / 127.0 * 2.0;
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
    // Mu-law: a valid format this crate does not model. Two channels, so one
    // frame is two bytes.
    let spec = RawSpec {
        format_code: 7,
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
    assert_eq!(reader.params().format_code, 7);
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
        format_code: 7,
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
        format_code: 7,
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

#[test]
fn convenience_file_roundtrip() {
    use waveadapter::{WavData, read_wav_file, write_wav_file};

    let channels = 2;
    let frames = 64;
    let source = make_buffer(channels, frames);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "waveadapter_convenience_{}.wav",
        std::process::id()
    ));

    let clipped = write_wav_file(&path, &source, 44100, SampleFormat::I16).unwrap();
    assert_eq!(clipped, 0);

    let audio: WavData<f32> = read_wav_file(&path).unwrap();
    assert_eq!(audio.sample_rate, 44100);
    assert_eq!(audio.channels(), channels);
    assert_eq!(audio.frames(), frames);

    for frame in 0..frames {
        for ch in 0..channels {
            let a = source.read_sample(ch, frame).unwrap();
            let b = audio.samples.read_sample(ch, frame).unwrap();
            // I16 quantization tolerance.
            assert!(
                (a - b).abs() <= 1.0 / 32767.0,
                "frame {frame} ch {ch}: {a} vs {b}"
            );
        }
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn abandoned_riff_writer_is_still_readable() {
    let channels = 2;
    let frames = 48;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 44100,
        sample_format: SampleFormat::F32,
        channel_mask: None,
    };

    // Write audio but never call finalize, standing in for a process that was
    // interrupted mid-recording.
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
        writer.write_float_buffer(&source).unwrap();
    }
    let bytes = cursor.into_inner();

    // The unpatched size fields carry the "runs to end of file" marker, not zero.
    let rd32 = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    assert_eq!(rd32(4), u32::MAX, "RIFF size is the unknown-length marker");

    // So the audio that reached the file reads back intact.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), frames);
    for frame in 0..frames {
        for ch in 0..channels {
            let a = source.read_sample(ch, frame).unwrap();
            let b = restored.read_sample(ch, frame).unwrap();
            assert_eq!(a, b, "frame {frame} ch {ch}");
        }
    }
}

#[test]
fn update_header_makes_an_abandoned_rf64_readable() {
    let channels = 2;
    let half = 32;
    let source = make_buffer(channels, half);
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::F32,
        channel_mask: None,
    };

    // Write a block, update the header, write a second block, then never
    // finalize. RF64 has no unknown-length marker, so only the updated ds64
    // sizes make the file readable.
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new_rf64(&mut cursor, spec).unwrap();
        writer.write_float_buffer(&source).unwrap();
        writer.update_header().unwrap();
        writer.write_float_buffer(&source).unwrap();
    }
    let bytes = cursor.into_inner();

    // Both blocks are on disk, but the header only declares the first.
    let expected_bytes = (half * channels * 4) as u64;
    assert_eq!(
        bytes.len() as u64,
        72 + 8 + 2 * expected_bytes,
        "both blocks were written"
    );
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
    assert_eq!(rd64(28), expected_bytes, "ds64 dataSize from the update");
    assert_eq!(rd64(36), half as u64, "ds64 sampleCount from the update");

    // The declared audio reads back intact.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.frames(), half);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    assert_eq!(restored.frames(), half);
    for frame in 0..half {
        for ch in 0..channels {
            let a = source.read_sample(ch, frame).unwrap();
            let b = restored.read_sample(ch, frame).unwrap();
            assert_eq!(a, b, "frame {frame} ch {ch}");
        }
    }
}

#[test]
fn update_header_resumes_writing_at_the_right_place() {
    let channels = 1;
    let frames = 16;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    // Interleaving updates with writes must not disturb the write cursor: the
    // finalized file has to match one written without any updates.
    let mut updated = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut updated, spec).unwrap();
    writer.update_header().unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.update_header().unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.update_header().unwrap();
    writer.finalize().unwrap();

    let mut plain = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut plain, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();

    assert_eq!(
        updated.into_inner(),
        plain.into_inner(),
        "updating the header must not change the finished file"
    );
}

#[test]
fn rf64_chunk_size_from_ds64_cannot_overflow_the_scan() {
    // Found by the parse_header fuzz target. For RF64 the real chunk size comes
    // from the ds64 table as an unvalidated 64-bit value, so a file can name a
    // size near u64::MAX. The offset arithmetic that checks whether a chunk
    // overruns the file used to overflow before it could reject it: a panic in
    // debug, and in release a wrapped offset that passed the overrun check and
    // went on to request a multi-exabyte allocation.
    let mut file = Vec::new();
    file.extend_from_slice(b"RF64");
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    file.extend_from_slice(b"WAVE");

    // ds64 with a one-entry table claiming a JUNK chunk of u64::MAX bytes.
    file.extend_from_slice(b"ds64");
    file.extend_from_slice(&40u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes()); // riffSize
    file.extend_from_slice(&0u64.to_le_bytes()); // dataSize
    file.extend_from_slice(&0u64.to_le_bytes()); // sampleCount
    file.extend_from_slice(&1u32.to_le_bytes()); // tableLength
    file.extend_from_slice(b"JUNK");
    file.extend_from_slice(&u64::MAX.to_le_bytes());

    // A perfectly good 16-bit stereo fmt and a short data chunk.
    file.extend_from_slice(b"fmt ");
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes()); // PCM
    file.extend_from_slice(&2u16.to_le_bytes()); // channels
    file.extend_from_slice(&44100u32.to_le_bytes());
    file.extend_from_slice(&176400u32.to_le_bytes()); // byte rate
    file.extend_from_slice(&4u16.to_le_bytes()); // block align
    file.extend_from_slice(&16u16.to_le_bytes()); // bits
    file.extend_from_slice(b"data");
    file.extend_from_slice(&4u32.to_le_bytes());
    file.extend_from_slice(&[0u8; 4]);

    // The oversized chunk, whose 0xFFFFFFFF size resolves through the table.
    file.extend_from_slice(b"JUNK");
    file.extend_from_slice(&u32::MAX.to_le_bytes());

    // The bogus length must stop the scan, not blow up, and the chunks found
    // before it must survive.
    let params = read_wav_header(Cursor::new(file)).expect("header should still parse");
    assert_eq!(params.channels, 2);
    assert_eq!(params.sample_rate, 44100);
    assert_eq!(params.sample_format, Some(SampleFormat::I16));
    assert_eq!(params.data_length, 4);
}
