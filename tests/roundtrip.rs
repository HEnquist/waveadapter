//! Round-trip and header tests for waveadapter.

use std::io::Cursor;

use audioadapter::{Adapter, AdapterMut};
use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::header::read_wav_header;
use waveadapter::{SampleFormat, WavReader, WavSpec, WavWriter};

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
