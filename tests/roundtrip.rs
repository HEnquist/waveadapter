//! Round-trip and header tests for waveadapter.

use std::io::Cursor;

use audioadapter::{Adapter, AdapterMut};
use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::header::read_wav_header;
use waveadapter::{Chunk, SampleFormat, WavReader, WavSpec, WavWriter};

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
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.channels(), channels);
    assert_eq!(reader.sample_rate(), 48000);
    assert_eq!(reader.sample_format(), format);
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
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    let data = InterleavedOwned::<f32>::new(0.0, 2, 5);
    writer.write_float_buffer(&data).unwrap();
    writer.finalize().unwrap();
    cursor.set_position(0);

    let params = read_wav_header(&mut cursor).unwrap();
    assert_eq!(params.sample_format, SampleFormat::I32);
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
    assert_eq!(
        &bytes[60..64],
        b"data",
        "data chunk follows the 40-byte fmt"
    );

    // And it reads back as I24_4 with the audio intact and data starting at 68.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), SampleFormat::I24_4);
    assert_eq!(reader.channels(), channels);
    assert_eq!(reader.params().data_offset, 68);

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
    assert_eq!(
        &bytes[60..64],
        b"data",
        "data chunk follows the 40-byte fmt"
    );

    // And it reads back as plain I16 with the audio intact.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), SampleFormat::I16);
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
    assert_eq!(reader.sample_format(), SampleFormat::F32);
    assert_eq!(reader.frames(), 10);
    assert!(reader.params().chunks.iter().any(|c| &c.id == b"fact"));
}

#[test]
fn pcm_write_has_no_fact_chunk() {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
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
    };
    let mut writer = WavWriter::new(Cursor::new(Vec::new()), spec).unwrap();
    writer.write_raw_interleaved(&[0u8; 4]).unwrap();
    writer.write_chunk(*b"LIST", b"x").unwrap();
    let result = writer.write_raw_interleaved(&[0u8; 4]);
    assert!(matches!(result, Err(waveadapter::WavError::InvalidSpec(_))));
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
    };
    let result = WavWriter::new(Cursor::new(Vec::new()), spec);
    assert!(matches!(result, Err(waveadapter::WavError::InvalidSpec(_))));

    // Zero channels is also rejected.
    let spec = WavSpec {
        channels: 0,
        sample_rate: 44100,
        sample_format: SampleFormat::I16,
    };
    assert!(matches!(
        WavWriter::new(Cursor::new(Vec::new()), spec),
        Err(waveadapter::WavError::InvalidSpec(_))
    ));
}
