//! Round-trip and header tests for waveadapter.

use std::io::Cursor;

use audioadapter::{Adapter, AdapterMut};
use audioadapter_buffers::owned::InterleavedOwned;
use waveadapter::header::read_wav_header;
use waveadapter::{Chunk, Fact, FmtChunk, SampleFormat, WavError, WavReader, WavSpec, WavWriter};

/// An IMA ADPCM `fmt ` chunk: a valid format this crate does not model, in the
/// shape a real encoder writes it. The two-byte extension (`wSamplesPerBlock`)
/// makes it the 20-byte form, so it also exercises the verbatim extension
/// bytes; `byte_rate` is not `block_align * sample_rate`, as it never is
/// outside linear PCM.
fn ima_adpcm_fmt(channels: u16, sample_rate: u32) -> FmtChunk {
    let block_align = 256 * channels;
    // Each block spends 4 bytes per channel on the preamble and then packs two
    // samples per byte.
    let samples_per_block = (block_align - 4 * channels) * 2 / channels + 1;
    FmtChunk {
        format_code: 0x11,
        channels,
        sample_rate,
        byte_rate: sample_rate * block_align as u32 / samples_per_block as u32,
        block_align,
        bits_per_sample: 4,
        extension: Some(samples_per_block.to_le_bytes().to_vec()),
    }
}

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
    // The G.711 laws space their steps logarithmically, so the error depends on
    // the magnitude rather than on a fixed bit depth. `make_buffer` stays inside
    // +-0.26, where both laws are in their sixth segment and the step is 512 of
    // the 32768 an i16 spans, so half a step is the bound.
    roundtrip(SampleFormat::ALAW, 256.0 / 32768.0);
    roundtrip(SampleFormat::MULAW, 256.0 / 32768.0);
}

/// Write a buffer as the given format and return the whole file.
fn write_to_bytes(source: &InterleavedOwned<f32>, spec: WavSpec) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(source).unwrap();
    writer.finalize().unwrap();
    cursor.into_inner()
}

#[test]
fn g711_encoding_is_idempotent() {
    // Companding quantizes on the way in, so a float does not survive a
    // roundtrip. What must hold is that the quantization is stable: decoding a
    // file and writing it back out again reproduces the same code words, so
    // repeated read/write cycles do not drift.
    for format in [SampleFormat::ALAW, SampleFormat::MULAW] {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 8000,
            sample_format: format,
            channel_mask: None,
        };
        let first = write_to_bytes(&make_buffer(2, 64), spec);

        let mut reader = WavReader::new(Cursor::new(first.clone())).unwrap();
        let decoded = reader.read_all_to_float::<f32>().unwrap();
        let second = write_to_bytes(&decoded, spec);

        assert_eq!(first, second, "{format:?}: re-encoding changed the bytes");
    }
}

#[test]
fn g711_writes_a_non_pcm_header() {
    for (format, code) in [(SampleFormat::ALAW, 6u16), (SampleFormat::MULAW, 7u16)] {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 8000,
            sample_format: format,
            channel_mask: None,
        };
        let bytes = write_to_bytes(&make_buffer(2, 10), spec);
        let rd16 = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        let rd32 =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

        // The 18-byte WAVEFORMATEX form: body 20..38, cbSize last.
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(rd32(16), 18, "{format:?}: non-PCM needs the cbSize field");
        assert_eq!(rd16(20), code, "{format:?}: format tag");
        assert_eq!(rd16(32), 2, "{format:?}: one byte per channel per frame");
        assert_eq!(rd16(34), 8, "{format:?}: bits per sample");
        assert_eq!(rd16(36), 0, "{format:?}: cbSize is zero");

        // Non-PCM also means a `fact` chunk carrying the frame count.
        assert_eq!(
            &bytes[38..42],
            b"fact",
            "{format:?}: fact chunk follows fmt"
        );
        assert_eq!(rd32(46), 10, "{format:?}: fact frame count");

        // One byte per sample on disk, so ten stereo frames are twenty bytes.
        let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.sample_format(), Some(format));
        assert_eq!(reader.frames(), 10);
        assert_eq!(reader.params().data_length, 20);
        assert_eq!(reader.read_all_to_float::<f32>().unwrap().frames(), 10);
    }
}

#[test]
fn multichannel_g711_uses_the_matching_subtype_guid() {
    // More than two channels forces the extensible form, and the subformat GUID
    // has to follow the format. Writing the PCM GUID here would produce a file
    // that claims to be linear 8-bit and decodes to noise.
    for (format, guid_first_byte) in [(SampleFormat::ALAW, 6u8), (SampleFormat::MULAW, 7u8)] {
        let spec = WavSpec {
            channels: 4,
            sample_rate: 8000,
            sample_format: format,
            channel_mask: None,
        };
        let bytes = write_to_bytes(&make_buffer(4, 8), spec);

        assert_eq!(
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            40,
            "{format:?}: four channels write the extensible form"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[20], bytes[21]]),
            0xFFFE,
            "{format:?}: extensible format tag"
        );
        // Subformat GUID sits at body offset 24, so file offset 44.
        assert_eq!(
            &bytes[44..60],
            &[
                guid_first_byte,
                0,
                0,
                0,
                0,
                0,
                0x10,
                0,
                0x80,
                0,
                0,
                0xaa,
                0,
                0x38,
                0x9b,
                0x71
            ],
            "{format:?}: subformat GUID"
        );

        let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.sample_format(), Some(format));
        assert_eq!(reader.channels(), 4);
        assert_eq!(reader.read_all_to_float::<f32>().unwrap().frames(), 8);
    }
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
    assert_eq!(params.sample_format(), Some(SampleFormat::I32));
    assert_eq!(params.channels(), 2);
    assert_eq!(params.sample_rate(), 44100);
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

    // RIFF/WAVE (12) + fmt chunk (8 + 18, float is non-PCM so it gets the
    // WAVEFORMATEX form with a zero cbSize). The fact chunk follows it.
    assert_eq!(rd32(16), 18, "float fmt body is the 18-byte WAVEFORMATEX");
    // Body runs 20..38; cbSize is the last field of it.
    assert_eq!(
        u16::from_le_bytes([bytes[36], bytes[37]]),
        0,
        "cbSize is zero"
    );
    assert_eq!(
        &bytes[38..42],
        b"fact",
        "fact chunk follows the 18-byte fmt"
    );
    assert_eq!(rd32(42), 4, "fact body is 4 bytes");
    assert_eq!(rd32(46), 10, "fact carries the sample-frame count");
    assert_eq!(&bytes[50..54], b"data", "data chunk follows the fact chunk");

    // It reads back cleanly, with the fact chunk parsed rather than left among
    // the opaque chunks: the writer owns `fact`, so the reader does too, and a
    // caller handing `chunks` straight back cannot end up writing it twice.
    let cursor = Cursor::new(bytes);
    let reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::F32));
    assert_eq!(reader.frames(), 10);
    assert_eq!(reader.params().fact_samples(), Some(10));
    assert!(!reader.params().chunks().any(|c| &c.id == b"fact"));
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
    assert_eq!(params.channels(), 1);
    let ids: Vec<[u8; 4]> = params.chunks().map(|c| c.id).collect();
    assert!(ids.contains(b"bext"), "leading chunk present: {ids:?}");
    assert!(ids.contains(b"LIST"), "trailing chunk present: {ids:?}");
    let bext = params.chunks().find(|c| &c.id == b"bext").unwrap();
    assert_eq!(bext.data, vec![1, 2, 3]);
    let list = params.chunks().find(|c| &c.id == b"LIST").unwrap();
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
    // ds64 ends at 48; the float fmt chunk (8 + 18) runs to 74.
    assert_eq!(&bytes[48..52], b"fmt ");
    assert_eq!(&bytes[74..78], b"data", "data follows fmt, no fact chunk");
    assert_eq!(rd32(78), u32::MAX, "data size field is the marker");

    // It reads back with the resolved 64-bit data length and intact audio.
    let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.sample_format(), Some(SampleFormat::F32));
    assert_eq!(reader.channels(), channels);
    assert_eq!(reader.frames(), frames);
    assert_eq!(reader.params().data_length, data_bytes);
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
fn rf64_sample_count_is_never_a_block_count() {
    // RF64 keeps the sample-frame count in ds64 instead of a `fact` chunk, and
    // the same honesty rule applies: for a format the crate does not model,
    // data_bytes / block_align is a block count, so `Auto` writes nothing and
    // the field keeps the zero it was written with.
    let fmt = ima_adpcm_fmt(2, 22050);
    let blocks = 3usize;
    let audio = vec![0u8; blocks * fmt.block_align as usize];

    let read_sample_count = |bytes: &[u8]| {
        assert_eq!(&bytes[12..16], b"ds64");
        u64::from_le_bytes(bytes[36..44].try_into().unwrap())
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(&fmt)
        .unwrap()
        .rf64()
        .open(&mut cursor)
        .unwrap();
    writer.write_raw_interleaved(&audio).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();
    assert_eq!(
        read_sample_count(&bytes),
        0,
        "no count rather than a block count"
    );

    // A codec that knows the real number says so, and it lands in ds64.
    let samples = 12_345;
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(&fmt)
        .unwrap()
        .fact(Fact::Samples(samples))
        .rf64()
        .open(&mut cursor)
        .unwrap();
    writer.write_raw_interleaved(&audio).unwrap();
    writer.finalize().unwrap();
    let bytes = cursor.into_inner();
    assert_eq!(read_sample_count(&bytes), samples);
    // No `fact` chunk sneaks in alongside it.
    let params = read_wav_header(Cursor::new(&bytes)).unwrap();
    assert_eq!(params.sample_count(), Some(samples));
    assert!(!params.chunks().any(|c| &c.id == b"fact"));

    // A modelled format still gets its counted frames.
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_rf64(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&make_buffer(2, 40)).unwrap();
    writer.finalize().unwrap();
    assert_eq!(read_sample_count(&cursor.into_inner()), 40);
}

#[test]
fn a_supplied_count_reaches_ds64_at_the_full_64_bits() {
    // The ds64 sampleCount field is 64-bit, which is the whole point of RF64:
    // a file too long to state its frame count in a 32-bit `fact` chunk. A
    // supplied count has to survive at that width, and for an unmodelled format
    // supplying it is the only way the file gets a count at all.
    let fmt = ima_adpcm_fmt(2, 22050);
    let audio = vec![0u8; fmt.block_align as usize];
    let samples = u64::from(u32::MAX) + 12_345;

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(&fmt)
        .unwrap()
        .fact(Fact::Samples(samples))
        .rf64()
        .open(&mut cursor)
        .unwrap();
    writer.write_raw_interleaved(&audio).unwrap();
    writer.finalize().unwrap();

    let bytes = cursor.into_inner();
    assert_eq!(&bytes[12..16], b"ds64");
    assert_eq!(
        u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
        samples,
        "written to the ds64 field untruncated"
    );
    let params = read_wav_header(Cursor::new(&bytes)).unwrap();
    assert_eq!(params.sample_count(), Some(samples));

    // The same count on a plain RIFF file has nowhere to go: the `fact` field
    // is 32 bits, so it is refused rather than written truncated.
    let mut cursor = Cursor::new(Vec::new());
    assert!(matches!(
        WavWriter::builder(&fmt)
            .unwrap()
            .fact(Fact::Samples(samples))
            .open(&mut cursor)
            .map(|_| ()),
        Err(WavError::InvalidSpec(_))
    ));

    // And a count that does fit is still written to the `fact` chunk as before.
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(&fmt)
        .unwrap()
        .fact(Fact::Samples(12_345))
        .open(&mut cursor)
        .unwrap();
    writer.write_raw_interleaved(&audio).unwrap();
    writer.finalize().unwrap();
    let params = read_wav_header(Cursor::new(cursor.into_inner())).unwrap();
    assert_eq!(params.fact_samples(), Some(12_345));
    assert_eq!(params.sample_count(), Some(12_345));
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
    let bext = params.chunks().find(|c| &c.id == b"bext").unwrap();
    assert_eq!(bext.data, vec![7u8; 10]);
    // The ds64 chunk is consumed by the parser, not surfaced as a raw chunk.
    assert!(!params.chunks().any(|c| &c.id == b"ds64"));
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
    // IMA ADPCM: a valid format this crate does not model. Nothing interprets
    // the bytes, so a "frame" is whatever the block alignment claims, here one
    // 512-byte block.
    let spec = ima_adpcm_fmt(2, 22050);
    let block = spec.block_align as usize;
    let samples: Vec<u8> = (0..2 * block).map(|i| i as u8).collect();

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(&spec)
        .unwrap()
        .open(&mut cursor)
        .unwrap();
    writer.write_raw_interleaved(&samples).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    // The format is not interpreted, but the raw fmt fields survive.
    assert_eq!(reader.sample_format(), None);
    assert_eq!(reader.channels(), 2);
    assert_eq!(reader.sample_rate(), 22050);
    assert_eq!(reader.params().fmt, spec, "the fmt chunk survives verbatim");
    assert_eq!(reader.frames(), 2);

    // No `fact` chunk is written for a raw format.
    assert!(!reader.params().chunks().any(|c| &c.id == b"fact"));

    // The bytes come back untouched through the raw read path.
    let mut out = Vec::new();
    let frames_read = reader.read_raw_interleaved(2, &mut out).unwrap();
    assert_eq!(frames_read, 2);
    assert_eq!(out, samples);
}

#[test]
fn float_read_on_raw_format_errors() {
    let spec = ima_adpcm_fmt(1, 8000);
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(spec).unwrap().open(&mut cursor).unwrap();
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
    let spec = ima_adpcm_fmt(1, 8000);
    let mut writer = WavWriter::builder(spec)
        .unwrap()
        .open(Cursor::new(Vec::new()))
        .unwrap();
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
    assert_eq!(params.channel_mask(), Some(0x3));
    assert_eq!(params.fmt.format_code, 0xFFFE);
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
    assert_eq!(params.channel_mask(), None);
    assert_eq!(params.fmt.format_code, 1);
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

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("waveadapter_{name}_{}.wav", std::process::id()));
    path
}

/// The raw one-call helper writes the bytes untouched, and the file it produces
/// reads back as ordinary audio.
#[test]
fn convenience_raw_file_roundtrip() {
    use waveadapter::write_wav_file_raw;

    let path = temp_path("convenience_raw");
    // Interleaved 16-bit stereo: a distinct value per sample, so 128 frames.
    let source: Vec<u8> = (0..256u32).flat_map(|n| (n as u16).to_le_bytes()).collect();

    write_wav_file_raw(&path, &source, WavSpec::new(2, 44100, SampleFormat::I16)).unwrap();

    let mut reader = WavReader::new(std::fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(reader.channels(), 2);
    assert_eq!(reader.sample_rate(), 44100);
    assert_eq!(reader.sample_format(), Some(SampleFormat::I16));
    assert_eq!(reader.frames(), 128);
    assert_eq!(reader.read_raw_all().unwrap(), source);

    std::fs::remove_file(&path).unwrap();
}

/// The promise that is actually checked: bytes that do not fill whole frames are
/// rejected instead of producing a `data` size that is not a multiple of
/// `nBlockAlign`.
#[test]
fn convenience_raw_rejects_a_partial_frame() {
    use waveadapter::write_wav_file_raw;

    let path = temp_path("convenience_raw_partial");
    // 10 bytes is two and a half frames of 16-bit stereo.
    let result = write_wav_file_raw(&path, &[0u8; 10], WavSpec::new(2, 44100, SampleFormat::I16));

    match result {
        Err(WavError::InvalidSpec(_)) => {}
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
    assert!(!path.exists(), "no file should be left behind");
}

/// A format the crate does not model goes through the same helper, by handing it
/// the `fmt ` chunk instead of a spec. The audio and the `fmt ` chunk survive
/// byte for byte; the `fact` count does not, which is what the docs say.
#[test]
fn convenience_raw_writes_an_unmodeled_format() {
    use waveadapter::write_wav_file_raw;

    let mut reader =
        WavReader::new(std::fs::File::open("tests/wav_variants/ima_adpcm_mono.wav").unwrap())
            .unwrap();
    assert_eq!(reader.sample_format(), None);
    let fmt = reader.params().fmt.clone();
    let audio = reader.read_raw_all().unwrap();
    assert!(!audio.is_empty());

    let path = temp_path("convenience_raw_adpcm");
    write_wav_file_raw(&path, &audio, fmt.clone()).unwrap();

    let mut rewritten = WavReader::new(std::fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(rewritten.sample_format(), None);
    assert_eq!(rewritten.params().fmt, fmt);
    assert_eq!(rewritten.read_raw_all().unwrap(), audio);
    // Fact::Auto writes nothing for a format it cannot count frames for.
    assert_eq!(rewritten.params().fact_samples(), None);

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
    // Header is RIFF/WAVE (12) + ds64 (8 + 28) + float fmt (8 + 18) = 74.
    let expected_bytes = (half * channels * 4) as u64;
    assert_eq!(
        bytes.len() as u64,
        74 + 8 + 2 * expected_bytes,
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
fn a_hand_built_chunk_with_no_channels_is_rejected() {
    // A FmtChunk is taken as authoritative, but the writer still must not
    // produce a file its own reader refuses, and zero channels is the one field
    // the parser rejects outright.
    let mut fmt = ima_adpcm_fmt(1, 8000);
    fmt.channels = 0;
    assert!(matches!(
        WavWriter::builder(fmt.clone()).map(|_| ()),
        Err(WavError::InvalidSpec(_))
    ));

    // The same chunk with a channel opens, unmodelled format and all.
    fmt.channels = 1;
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(fmt).unwrap().open(&mut cursor).unwrap();
    writer.write_raw_interleaved(&[0; 256]).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.channels(), 1);
    assert_eq!(reader.frames(), 1);
}

#[test]
fn zero_block_alignment_means_no_frames() {
    // An unmodelled format declaring a zero nBlockAlign has no framing at all.
    // The frame count used to come out as usize::MAX, the clamp meant for a
    // file too large to index, which reads as "enormous" instead of "unknown".
    let raw = FmtChunk {
        format_code: 6,
        channels: 1,
        sample_rate: 8000,
        byte_rate: 0,
        block_align: 0,
        bits_per_sample: 8,
        extension: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::builder(raw).unwrap().open(&mut cursor).unwrap();
    writer.write_raw_interleaved(&[0; 16]).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.params().frame_bytes(), 0);
    assert_eq!(reader.frames(), 0);
    assert_eq!(reader.remaining(), 0);
    // Every framed path still rejects it outright rather than guessing.
    let mut buf = Vec::new();
    assert!(matches!(
        reader.read_raw_interleaved(1, &mut buf),
        Err(WavError::InvalidHeader(_))
    ));
    assert!(matches!(
        reader.read_raw_all(),
        Err(WavError::InvalidHeader(_))
    ));
    assert!(matches!(
        reader.seek_to_frame(1),
        Err(WavError::InvalidHeader(_))
    ));
}

#[test]
fn seek_past_an_unindexable_frame_count_cannot_overflow() {
    // An RF64 file can declare a data length filling the whole 64-bit field.
    // Turning the resulting frame count back into a byte offset used to be done
    // in usize arithmetic, which overflows once the product passes the target's
    // pointer width: a panic in debug, and in release a wrapped offset landing
    // somewhere arbitrary in the file. Only a 32-bit target can reach that with
    // these numbers, so what this pins down on a 64-bit one is the boundary
    // itself, that the largest declarable file still seeks and reads cleanly.
    let mut file = Vec::new();
    file.extend_from_slice(b"RF64");
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    file.extend_from_slice(b"WAVE");

    file.extend_from_slice(b"ds64");
    file.extend_from_slice(&28u32.to_le_bytes());
    file.extend_from_slice(&u64::MAX.to_le_bytes()); // riffSize
    file.extend_from_slice(&u64::MAX.to_le_bytes()); // dataSize
    file.extend_from_slice(&0u64.to_le_bytes()); // sampleCount
    file.extend_from_slice(&0u32.to_le_bytes()); // tableLength

    file.extend_from_slice(b"fmt ");
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes()); // PCM
    file.extend_from_slice(&2u16.to_le_bytes()); // channels
    file.extend_from_slice(&44100u32.to_le_bytes());
    file.extend_from_slice(&176400u32.to_le_bytes()); // byte rate
    file.extend_from_slice(&4u16.to_le_bytes()); // block align
    file.extend_from_slice(&16u16.to_le_bytes()); // bits
    file.extend_from_slice(b"data");
    file.extend_from_slice(&u32::MAX.to_le_bytes()); // resolved through ds64
    file.extend_from_slice(&[0u8; 8]);

    let mut reader = WavReader::new(Cursor::new(file)).unwrap();
    let frames = usize::try_from(u64::MAX / 4).unwrap_or(usize::MAX);
    assert_eq!(
        reader.frames(),
        frames,
        "the whole declared length, clamped"
    );
    // The seek clamps to the declared count, and the offset saturates instead
    // of wrapping.
    reader.seek_to_frame(usize::MAX).unwrap();
    assert_eq!(reader.position(), frames);
    // Nothing is there to read, but asking must not panic either.
    let mut buf = Vec::new();
    reader.read_raw_interleaved(1, &mut buf).unwrap();
    assert!(buf.is_empty());
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
    assert_eq!(params.channels(), 2);
    assert_eq!(params.sample_rate(), 44100);
    assert_eq!(params.sample_format(), Some(SampleFormat::I16));
    assert_eq!(params.data_length, 4);
}

#[test]
fn writer_position_and_extent() {
    let channels = 2;
    let frames = 32;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let frame_bytes = channels * 2;

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();

    // A plain 16-bit stereo header: RIFF + fmt + data headers, no fact chunk.
    assert_eq!(writer.data_offset(), 44);
    assert_eq!(writer.frames_written(), 0);
    assert_eq!(writer.position(), 0);

    writer.write_float_buffer(&source).unwrap();
    assert_eq!(writer.frames_written(), frames);
    assert_eq!(writer.position(), frames);
    assert_eq!(writer.data_bytes(), (frames * frame_bytes) as u64);

    // After seeking back, the cursor and the extent are different numbers.
    writer.seek_to_frame(10).unwrap();
    assert_eq!(writer.position(), 10);
    assert_eq!(writer.frames_written(), frames);

    // Overwriting in place moves the cursor but not the extent.
    writer
        .write_float_buffer(&make_buffer(channels, 4))
        .unwrap();
    assert_eq!(writer.position(), 14);
    assert_eq!(writer.frames_written(), frames);

    // The room left is counted from the cursor, up to the 4 GB RIFF ceiling.
    let ceiling = u32::MAX as u64 + 8 - writer.data_offset();
    assert_eq!(
        writer.remaining_frames(),
        Some(((ceiling - (14 * frame_bytes) as u64) / frame_bytes as u64) as usize)
    );

    // A trailing chunk closes the audio, so nothing more can be written.
    writer.write_chunk(*b"JUNK", &[0; 4]).unwrap();
    assert_eq!(writer.remaining_frames(), Some(0));
}

#[test]
fn remaining_frames_is_none_without_a_limit() {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    // Streaming never patches the size fields, and RF64 has 64-bit ones.
    let streaming = WavWriter::new_streaming(Cursor::new(Vec::new()), spec).unwrap();
    assert_eq!(streaming.remaining_frames(), None);

    let rf64 = WavWriter::new_rf64(Cursor::new(Vec::new()), spec).unwrap();
    assert_eq!(rf64.remaining_frames(), None);
}

#[test]
fn writer_forward_seek_fills_with_silence() {
    let channels = 1;
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

    // Skip a gap and write a second block. The gap is silence, and the file has
    // already grown to the seek target before anything is written there.
    writer.seek_to_frame(20).unwrap();
    assert_eq!(writer.frames_written(), 20);
    assert_eq!(writer.position(), 20);
    writer.write_float_buffer(&source).unwrap();
    assert_eq!(writer.frames_written(), 28);
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.frames(), 28);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    for frame in 0..28 {
        let expected = match frame {
            0..8 => source.read_sample(0, frame).unwrap(),
            8..20 => 0.0,
            _ => source.read_sample(0, frame - 20).unwrap(),
        };
        let got = restored.read_sample(0, frame).unwrap();
        assert!(
            (expected - got).abs() <= 1e-4,
            "frame {frame}: {expected} vs {got}"
        );
    }
}

#[test]
fn writer_truncate_drops_data_past_the_cursor() {
    let channels = 2;
    let frames = 32;
    let source = make_buffer(channels, frames);
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let frame_bytes = channels * 2;

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.write_float_buffer(&source).unwrap();

    // Seek back and cut: the extent follows the cursor down instead of staying
    // at the high-water mark.
    writer.seek_to_frame(10).unwrap();
    assert_eq!(writer.frames_written(), frames);
    writer.truncate().unwrap();
    assert_eq!(writer.frames_written(), 10);
    assert_eq!(writer.position(), 10);
    assert_eq!(writer.data_bytes(), (10 * frame_bytes) as u64);

    // Writing continues from the new end.
    writer
        .write_float_buffer(&make_buffer(channels, 4))
        .unwrap();
    assert_eq!(writer.frames_written(), 14);
    let data_offset = writer.data_offset();
    writer.finalize().unwrap();

    // The bytes are really gone, not just excluded by the declared size.
    assert_eq!(
        cursor.get_ref().len() as u64,
        data_offset + (14 * frame_bytes) as u64
    );

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.frames(), 14);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    let second = make_buffer(channels, 4);
    for frame in 0..14 {
        for ch in 0..channels {
            let expected = if frame < 10 {
                source.read_sample(ch, frame).unwrap()
            } else {
                second.read_sample(ch, frame - 10).unwrap()
            };
            let got = restored.read_sample(ch, frame).unwrap();
            assert!(
                (expected - got).abs() <= 1e-4,
                "frame {frame} ch {ch}: {expected} vs {got}"
            );
        }
    }
}

#[test]
fn writer_truncate_to_frame_moves_a_stranded_cursor() {
    let channels = 1;
    let frames = 16;
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer
        .write_float_buffer(&make_buffer(channels, frames))
        .unwrap();

    // Cutting below the cursor drags it back to the new end, so the next write
    // appends there rather than leaving a hole.
    writer.truncate_to_frame(4).unwrap();
    assert_eq!(writer.position(), 4);
    assert_eq!(writer.frames_written(), 4);

    // A cut at or past the end is a no-op, and truncating never grows the file.
    writer.truncate_to_frame(4).unwrap();
    writer.truncate_to_frame(100).unwrap();
    assert_eq!(writer.frames_written(), 4);

    writer
        .write_float_buffer(&make_buffer(channels, 2))
        .unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.frames(), 6);
}

#[test]
fn writer_truncate_rejects_streaming_and_trailing_chunks() {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    // A streaming writer never patches its size fields, so a trim could not be
    // described even though the underlying stream could be shortened.
    let mut streaming = WavWriter::new_streaming(Cursor::new(Vec::new()), spec).unwrap();
    streaming.write_raw_interleaved(&[0; 16]).unwrap();
    assert!(matches!(
        streaming.truncate_to_frame(1),
        Err(WavError::InvalidSpec(_))
    ));

    // Trimming the audio under a trailing chunk would strand it.
    let mut writer = WavWriter::new(Cursor::new(Vec::new()), spec).unwrap();
    writer.write_raw_interleaved(&[0; 16]).unwrap();
    writer.write_chunk(*b"JUNK", &[0; 4]).unwrap();
    assert!(matches!(
        writer.truncate_to_frame(1),
        Err(WavError::InvalidSpec(_))
    ));

    // A raw writer with no block alignment has no frame size to cut on.
    let raw = FmtChunk {
        format_code: 6,
        channels: 1,
        sample_rate: 8000,
        byte_rate: 0,
        block_align: 0,
        bits_per_sample: 8,
        extension: None,
    };
    let mut raw_writer = WavWriter::builder(raw)
        .unwrap()
        .open(Cursor::new(Vec::new()))
        .unwrap();
    raw_writer.write_raw_interleaved(&[0; 16]).unwrap();
    assert!(matches!(
        raw_writer.truncate_to_frame(1),
        Err(WavError::InvalidSpec(_))
    ));
}

#[test]
fn rf64_truncate_updates_the_ds64_sizes() {
    let channels = 2;
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new_rf64(&mut cursor, spec).unwrap();
    writer
        .write_float_buffer(&make_buffer(channels, 32))
        .unwrap();
    writer.truncate_to_frame(12).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.frames(), 12);
    assert_eq!(reader.params().data_length, (12 * channels * 2) as u64);
}

#[test]
fn writer_truncate_on_a_buffered_file() {
    use std::fs::File;
    use std::io::BufWriter;

    let channels = 2;
    let spec = WavSpec {
        channels,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };

    let mut path = std::env::temp_dir();
    path.push(format!("waveadapter_truncate_{}.wav", std::process::id()));

    let file = BufWriter::new(File::create(&path).unwrap());
    let mut writer = WavWriter::new(file, spec).unwrap();
    writer
        .write_float_buffer(&make_buffer(channels, 64))
        .unwrap();
    // Nothing has been flushed yet, so this exercises the BufWriter impl
    // flushing before the file is shortened.
    writer.truncate_to_frame(20).unwrap();
    let data_offset = writer.data_offset();
    writer.finalize().unwrap();

    let on_disk = std::fs::metadata(&path).unwrap().len();
    assert_eq!(on_disk, data_offset + (20 * channels * 2) as u64);

    let audio: waveadapter::WavData<f32> = waveadapter::read_wav_file(&path).unwrap();
    assert_eq!(audio.frames(), 20);

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn writer_forward_seek_alone_extends_the_file() {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48000,
        sample_format: SampleFormat::I16,
        channel_mask: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
    writer.seek_to_frame(16).unwrap();
    writer.finalize().unwrap();

    cursor.set_position(0);
    let mut reader = WavReader::new(cursor).unwrap();
    assert_eq!(reader.frames(), 16);
    let restored = reader.read_all_to_float::<f32>().unwrap();
    for frame in 0..16 {
        for ch in 0..2 {
            assert_eq!(restored.read_sample(ch, frame).unwrap(), 0.0f32);
        }
    }
}
