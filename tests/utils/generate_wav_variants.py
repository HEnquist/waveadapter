#!/usr/bin/env python3
"""
generate_wav_variants.py

Generates a battery of short WAV files covering the range of variation a WAV
parser has to handle: different sample formats and layouts, optional and
out-of-order chunks, and the common real-world spec violations. Each case
isolates roughly one variation so a failing test points at a specific behavior
rather than "something about this file is wrong."

Usage:
    python tests/utils/generate_wav_variants.py [output_dir]

The default output_dir is the sibling tests/wav_variants directory, where the
Rust test suite (tests/wav_variants.rs) expects to find the fixtures. The
generated files are committed, so the test does not depend on Python being
available.
"""

import os
import struct
import sys

SAMPLE_RATE = 8000  # low rate keeps files tiny; irrelevant to header parsing
NUM_FRAMES = 20      # a handful of frames is enough to exercise data parsing


# ---------------------------------------------------------------------------
# Low-level chunk / container helpers
# ---------------------------------------------------------------------------

def chunk(chunk_id: bytes, data: bytes, pad: bool = True) -> bytes:
    """Wrap data in a RIFF chunk: 4-byte id, 4-byte LE size, data, pad byte
    if data length is odd (standard RIFF rule). Set pad=False to deliberately
    omit the pad byte even on odd-length data (a real-world spec violation)."""
    assert len(chunk_id) == 4
    out = chunk_id + struct.pack("<I", len(data)) + data
    if pad and len(data) % 2 == 1:
        out += b"\x00"
    return out


def riff(form_type: bytes, body: bytes, riff_size_override: int = None) -> bytes:
    """Wrap chunks in a RIFF container. body is the concatenation of all
    sub-chunks (after the 4-byte form type). riff_size_override lets you
    write a deliberately wrong RIFF size (lying header) for negative tests."""
    assert len(form_type) == 4
    size = riff_size_override if riff_size_override is not None else 4 + len(body)
    return b"RIFF" + struct.pack("<I", size) + form_type + body


def fmt_pcm(channels=1, sample_rate=SAMPLE_RATE, bits_per_sample=16, fmt_size=16):
    """Standard PCM fmt chunk. fmt_size lets you emit the technically-valid
    18-byte variant (with cbSize=0) some encoders write for plain PCM."""
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data = struct.pack(
        "<HHIIHH",
        1,  # WAVE_FORMAT_PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    if fmt_size == 18:
        data += struct.pack("<H", 0)  # cbSize = 0
    return chunk(b"fmt ", data)


def fmt_g711(format_code, channels=1, sample_rate=SAMPLE_RATE):
    """A-law (format tag 6) or mu-law (format tag 7) fmt chunk. One companded
    byte per sample. Both are non-PCM, so they use the 18-byte WAVEFORMATEX
    form with cbSize = 0."""
    assert format_code in (6, 7)
    block_align = channels
    byte_rate = sample_rate * block_align
    data = struct.pack(
        "<HHIIHH",
        format_code,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        8,
    )
    data += struct.pack("<H", 0)  # cbSize = 0
    return chunk(b"fmt ", data)


def fmt_ima_adpcm(channels=1, sample_rate=SAMPLE_RATE, samples_per_block=505):
    """IMA/DVI ADPCM fmt chunk (format tag 0x11). A block-compressed format
    with 4 bits per sample and a per-block predictor header, so its bytes do
    not divide into per-sample units at all. This crate does not model it, and
    it is here to keep the uninterpreted-format path covered."""
    block_align = 256 * channels
    # Roughly one block per this many sample frames.
    byte_rate = sample_rate * block_align // samples_per_block
    data = struct.pack(
        "<HHIIHH",
        0x11,  # WAVE_FORMAT_DVI_ADPCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        4,
    )
    data += struct.pack("<HH", 2, samples_per_block)  # cbSize = 2, wSamplesPerBlock
    return chunk(b"fmt ", data)


def fmt_ms_adpcm(channels=2, sample_rate=SAMPLE_RATE, samples_per_block=244):
    """MS ADPCM fmt chunk (format tag 2). Its extension is 32 bytes, making the
    whole chunk 50, which lands it *past* the 40-byte extensible size without
    being extensible. Offset 20 of the body, where a WAVEFORMATEXTENSIBLE keeps
    dwChannelMask, is wNumCoef here, so a parser that reads the mask on length
    alone reports a coefficient count as a speaker layout."""
    block_align = 256 * channels
    byte_rate = sample_rate * block_align // samples_per_block
    data = struct.pack(
        "<HHIIHH",
        2,  # WAVE_FORMAT_ADPCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        4,
    )
    # cbSize = 32: wSamplesPerBlock, wNumCoef, then 7 pairs of int16 coefficients.
    coefficients = [
        (256, 0), (512, -256), (0, 0), (192, 64),
        (240, 0), (460, -208), (392, -232),
    ]
    data += struct.pack("<H", 32)
    data += struct.pack("<HH", samples_per_block, len(coefficients))
    for coef1, coef2 in coefficients:
        data += struct.pack("<hh", coef1, coef2)
    return chunk(b"fmt ", data)


def fmt_gsm610(sample_rate=SAMPLE_RATE):
    """MS GSM 6.10 fmt chunk (format tag 0x31), the most hostile fmt chunk in
    common use. Three things are unusual and each one breaks a natural
    assumption:

      * wBitsPerSample is 0, so any "bytes = bits / 8" arithmetic gives zero
      * nBlockAlign is 65, an odd number, so blocks never land on the 2-byte
        boundary RIFF is otherwise built around
      * a cbSize of 2 carries wSamplesPerBlock, making the chunk 20 bytes

    Mono only, and one 65-byte block holds 320 sample frames."""
    block_align = 65
    samples_per_block = 320
    byte_rate = sample_rate * block_align // samples_per_block
    data = struct.pack(
        "<HHIIHH",
        0x31,  # WAVE_FORMAT_GSM610
        1,
        sample_rate,
        byte_rate,
        block_align,
        0,  # wBitsPerSample, genuinely zero for GSM
    )
    data += struct.pack("<HH", 2, samples_per_block)  # cbSize = 2
    return chunk(b"fmt ", data)


def fmt_pcm_padded24(channels=1, sample_rate=SAMPLE_RATE):
    """24-bit samples stored in 4-byte (padded) containers. block_align
    reflects the *actual* on-disk byte width (4), not the bit depth (24)."""
    bits_per_sample = 24
    byte_width = 4
    block_align = channels * byte_width
    byte_rate = sample_rate * block_align
    data = struct.pack(
        "<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits_per_sample
    )
    return chunk(b"fmt ", data)


def fmt_float(channels=1, sample_rate=SAMPLE_RATE, bits_per_sample=32, fmt_size=16):
    """IEEE float fmt chunk. fmt_size=18 emits the WAVEFORMATEX variant with
    cbSize=0, the way many encoders write float wav files."""
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    if fmt_size == 18:
        data += struct.pack("<H", 0)  # cbSize = 0
    return chunk(b"fmt ", data)


# Common speaker channel masks for WAVE_FORMAT_EXTENSIBLE
KSDATAFORMAT_SUBTYPE_PCM = bytes.fromhex("0100000000001000800000aa00389b71")
KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = bytes.fromhex("0300000000001000800000aa00389b71")
KSDATAFORMAT_SUBTYPE_ALAW = bytes.fromhex("0600000000001000800000aa00389b71")
KSDATAFORMAT_SUBTYPE_MULAW = bytes.fromhex("0700000000001000800000aa00389b71")


def fmt_extensible(channels=2, sample_rate=SAMPLE_RATE, bits_per_sample=24,
                    valid_bits_per_sample=24, channel_mask=0x3,
                    sub_format=KSDATAFORMAT_SUBTYPE_PCM, byte_width=None):
    """WAVE_FORMAT_EXTENSIBLE (format tag 0xFFFE), 40-byte fmt chunk.
    byte_width defaults to ceil(bits_per_sample/8) but can be overridden to
    produce packed-vs-padded variants under the extensible format too."""
    if byte_width is None:
        byte_width = (bits_per_sample + 7) // 8
    block_align = channels * byte_width
    byte_rate = sample_rate * block_align
    head = struct.pack(
        "<HHIIHH",
        0xFFFE,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    ext = struct.pack("<HH", 22, valid_bits_per_sample)  # cbSize=22, validBits
    ext += struct.pack("<I", channel_mask)
    ext += sub_format
    return chunk(b"fmt ", head + ext)


def pcm_ramp_data(num_frames, channels, bits_per_sample, byte_width=None,
                   signed=True):
    """A simple ascending ramp per channel, written at the given bit depth.
    byte_width lets you write fewer/more bytes than bits_per_sample implies
    (packed vs padded 24-bit, etc)."""
    if byte_width is None:
        byte_width = (bits_per_sample + 7) // 8
    max_val = (1 << (bits_per_sample - 1)) - 1 if signed else (1 << bits_per_sample) - 1
    out = bytearray()
    for frame in range(num_frames):
        for ch in range(channels):
            val = int(max_val * (frame / max(1, num_frames - 1)) * (1 if ch == 0 else -1))
            raw = val.to_bytes(byte_width, "little", signed=signed)
            out += raw
    return bytes(out)


def float_ramp_data(num_frames, channels, bits_per_sample=32):
    pack = "<f" if bits_per_sample == 32 else "<d"
    out = bytearray()
    for frame in range(num_frames):
        for ch in range(channels):
            val = (frame / max(1, num_frames - 1)) * (1.0 if ch == 0 else -1.0)
            out += struct.pack(pack, val)
    return bytes(out)


def data_chunk(payload: bytes, pad: bool = True) -> bytes:
    return chunk(b"data", payload, pad=pad)


def junk_chunk(size: int, chunk_id=b"JUNK") -> bytes:
    return chunk(chunk_id, b"\x00" * size)


def list_info_chunk(software="Test Generator") -> bytes:
    isft = software.encode("ascii") + b"\x00"
    if len(isft) % 2:
        isft += b"\x00"
    body = b"INFO" + chunk(b"ISFT", isft.rstrip(b"\x00") + b"\x00")
    return chunk(b"LIST", body)


# ---------------------------------------------------------------------------
# RF64 / BW64 (64-bit container) helpers
# ---------------------------------------------------------------------------

DS64_SIZE_MARKER = 0xFFFFFFFF  # 32-bit size field meaning "see ds64"


def data_chunk_marker(payload: bytes) -> bytes:
    """A data chunk whose 32-bit size field is the 0xFFFFFFFF marker; the real
    size is carried by the ds64 chunk. This is the usual real-world RF64 form."""
    out = b"data" + struct.pack("<I", DS64_SIZE_MARKER) + payload
    if len(payload) % 2 == 1:
        out += b"\x00"
    return out


def rf64(body: bytes, data_size: int, sample_count: int, table=(),
         form=b"RF64") -> bytes:
    """Wrap chunks in an RF64/BW64 container. The form id replaces RIFF and its
    32-bit size field is the 0xFFFFFFFF marker; a leading ds64 chunk carries the
    real 64-bit riffSize/dataSize/sampleCount and an optional table of (id, size)
    overrides for other oversized chunks. body is everything after the ds64 chunk
    (typically fmt + data). form is b"RF64" or b"BW64" (structurally identical)."""
    assert len(form) == 4
    table_bytes = b"".join(tid + struct.pack("<Q", sz) for tid, sz in table)
    assert all(len(tid) == 4 for tid, _ in table)
    # riffSize is filled in once the ds64 chunk length (fixed) is known.
    ds64_body = struct.pack("<QQQI", 0, data_size, sample_count, len(table)) + table_bytes
    ds64_chunk = chunk(b"ds64", ds64_body)
    riff_size = 4 + len(ds64_chunk) + len(body)  # bytes after the 8-byte header
    ds64_body = struct.pack("<Q", riff_size) + ds64_body[8:]
    ds64_chunk = chunk(b"ds64", ds64_body)
    return form + struct.pack("<I", DS64_SIZE_MARKER) + b"WAVE" + ds64_chunk + body


# ---------------------------------------------------------------------------
# Individual variant cases
# ---------------------------------------------------------------------------

def case_baseline_16bit_stereo():
    """Sanity baseline: canonical 16-bit PCM stereo, fmt then data, nothing
    unusual. Every other case should be compared against this passing."""
    f = fmt_pcm(channels=2, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 2, 16))
    return riff(b"WAVE", f + d)


def case_canonical_cd_16bit():
    """The most common wav file there is: 16-bit PCM, 44.1 kHz, stereo, plain
    fmt-then-data layout. The everyday happy path at a realistic sample rate."""
    f = fmt_pcm(channels=2, sample_rate=44100, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 2, 16))
    return riff(b"WAVE", f + d)


def case_canonical_24bit_48k():
    """A common high-resolution studio file: 24-bit packed PCM, 48 kHz, stereo,
    plain fmt-then-data layout."""
    f = fmt_pcm(channels=2, sample_rate=48000, bits_per_sample=24)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 2, 24, byte_width=3))
    return riff(b"WAVE", f + d)


def case_int32():
    """Plain 32-bit integer PCM, mono, 44.1 kHz. A normal format that no other
    case in the suite exercises."""
    f = fmt_pcm(channels=1, sample_rate=44100, bits_per_sample=32)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 32))
    return riff(b"WAVE", f + d)


def case_float64():
    """Plain 64-bit (double precision) IEEE float, mono, 44.1 kHz. A normal
    format that no other case in the suite exercises."""
    f = fmt_float(channels=1, sample_rate=44100, bits_per_sample=64)
    d = data_chunk(float_ramp_data(NUM_FRAMES, 1, bits_per_sample=64))
    return riff(b"WAVE", f + d)


def case_fmt_size_18_cbsize_zero():
    """Plain PCM but with the 18-byte fmt chunk variant (cbSize=0 appended).
    Technically valid; some encoders emit this for PCM unnecessarily."""
    f = fmt_pcm(channels=1, bits_per_sample=16, fmt_size=18)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    return riff(b"WAVE", f + d)


def case_24bit_packed():
    """24-bit samples, 3 bytes per sample, no padding byte."""
    f = fmt_pcm(channels=1, bits_per_sample=24)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 24, byte_width=3))
    return riff(b"WAVE", f + d)


def case_24bit_padded():
    """24-bit samples stored in 4-byte slots (high byte zero/sign-extend)."""
    f = fmt_pcm_padded24(channels=1)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 24, byte_width=4))
    return riff(b"WAVE", f + d)


def case_float32():
    f = fmt_float(channels=1, bits_per_sample=32)
    d = data_chunk(float_ramp_data(NUM_FRAMES, 1))
    return riff(b"WAVE", f + d)


def case_extensible_24in32_5point1():
    """WAVE_FORMAT_EXTENSIBLE, 24-bit-in-32-bit container, 6 channels,
    a real-world surround-sound layout."""
    f = fmt_extensible(channels=6, bits_per_sample=24, valid_bits_per_sample=24,
                        channel_mask=0x3F, byte_width=4)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 6, 24, byte_width=4))
    return riff(b"WAVE", f + d)


def case_extensible_float():
    f = fmt_extensible(channels=2, bits_per_sample=32, valid_bits_per_sample=32,
                        channel_mask=0x3, sub_format=KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
                        byte_width=4)
    d = data_chunk(float_ramp_data(NUM_FRAMES, 2))
    return riff(b"WAVE", f + d)


def case_float32_waveformatex18():
    """IEEE float written as the 18-byte WAVEFORMATEX form (cbSize=0), the way
    many encoders emit float wav files. Exercises the 18-byte path for a
    non-PCM format."""
    f = fmt_float(channels=1, bits_per_sample=32, fmt_size=18)
    d = data_chunk(float_ramp_data(NUM_FRAMES, 1))
    return riff(b"WAVE", f + d)


def case_extensible_16bit():
    """WAVE_FORMAT_EXTENSIBLE wrapping plain 16-bit PCM, stereo. Common output
    from Windows and many DAWs even for ordinary 16-bit audio."""
    f = fmt_extensible(channels=2, bits_per_sample=16, valid_bits_per_sample=16,
                        channel_mask=0x3, byte_width=2)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 2, 16))
    return riff(b"WAVE", f + d)


def case_extensible_24bit_packed():
    """WAVE_FORMAT_EXTENSIBLE with packed 24-in-3 PCM (no padding to a 4-byte
    container). Unusual but valid."""
    f = fmt_extensible(channels=1, bits_per_sample=24, valid_bits_per_sample=24,
                        channel_mask=0x4, byte_width=3)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 24, byte_width=3))
    return riff(b"WAVE", f + d)


def case_extensible_int32():
    """WAVE_FORMAT_EXTENSIBLE wrapping 32-bit integer PCM, stereo."""
    f = fmt_extensible(channels=2, bits_per_sample=32, valid_bits_per_sample=32,
                        channel_mask=0x3, byte_width=4)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 2, 32))
    return riff(b"WAVE", f + d)


def case_extensible_float64():
    """WAVE_FORMAT_EXTENSIBLE wrapping 64-bit double precision float, mono."""
    f = fmt_extensible(channels=1, bits_per_sample=64, valid_bits_per_sample=64,
                        channel_mask=0x4, sub_format=KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
                        byte_width=8)
    d = data_chunk(float_ramp_data(NUM_FRAMES, 1, bits_per_sample=64))
    return riff(b"WAVE", f + d)


def case_extensible_24in32_strict():
    """WAVE_FORMAT_EXTENSIBLE 24-in-32 in the strict-spec form: wBitsPerSample
    carries the 32-bit container size and wValidBitsPerSample carries the real
    24 bits. The same audio as extensible_24in32_5point1, which instead reports
    wBitsPerSample=24 (the lenient form)."""
    f = fmt_extensible(channels=2, bits_per_sample=32, valid_bits_per_sample=24,
                        channel_mask=0x3, byte_width=4)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 2, 24, byte_width=4))
    return riff(b"WAVE", f + d)


def case_junk_before_fmt():
    """Unknown chunk before fmt. Parsers that assume fmt is first will fail."""
    j = junk_chunk(8)
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    return riff(b"WAVE", j + f + d)


def case_list_info_between_fmt_and_data():
    """LIST/INFO metadata chunk sandwiched between fmt and data, a very
    common real-world layout from DAWs and encoders."""
    f = fmt_pcm(channels=2, bits_per_sample=16)
    li = list_info_chunk("Ableton Live")
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 2, 16))
    return riff(b"WAVE", f + li + d)


def case_data_before_fmt():
    """data chunk physically precedes fmt. Legal per RIFF (chunk order isn't
    mandated), but trips up parsers that read fmt lazily assuming order."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    return riff(b"WAVE", d + f)


def case_odd_sized_data_with_pad():
    """data payload has an odd length (e.g. 8-bit mono, odd frame count),
    correctly followed by a pad byte per RIFF spec."""
    f = fmt_pcm(channels=1, bits_per_sample=8)
    payload = pcm_ramp_data(NUM_FRAMES + 1, 1, 8, byte_width=1)  # odd length
    assert len(payload) % 2 == 1
    d = data_chunk(payload, pad=True)
    return riff(b"WAVE", f + d)


def case_odd_sized_data_missing_pad():
    """Same as above but the encoder forgot the pad byte: a real-world
    spec violation that some files genuinely have."""
    f = fmt_pcm(channels=1, bits_per_sample=8)
    payload = pcm_ramp_data(NUM_FRAMES + 1, 1, 8, byte_width=1)
    assert len(payload) % 2 == 1
    d = data_chunk(payload, pad=False)
    return riff(b"WAVE", f + d)


def case_riff_size_too_large():
    """RIFF header claims more bytes than the file actually contains
    (lying/truncated header), common from interrupted recordings."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    body = f + d
    return riff(b"WAVE", body, riff_size_override=4 + len(body) + 10_000)


def case_data_size_too_large():
    """data chunk's declared size exceeds the bytes actually present
    (truncated file, or a streaming encoder that wrote a placeholder size)."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    payload = pcm_ramp_data(NUM_FRAMES, 1, 16)
    # Build the data chunk header by hand with an inflated size field.
    d = b"data" + struct.pack("<I", len(payload) + 10_000) + payload
    return riff(b"WAVE", f + d)


def case_data_size_streaming_placeholder():
    """data chunk size written as 0xFFFFFFFF, a known convention some
    streaming/live encoders use when the final size isn't known yet."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    payload = pcm_ramp_data(NUM_FRAMES, 1, 16)
    d = b"data" + struct.pack("<I", 0xFFFFFFFF) + payload
    return riff(b"WAVE", f + d)


def case_trailing_junk_after_data():
    """Extra bytes after the data chunk that aren't a valid chunk at all
    (some tools just append garbage/log text)."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    return riff(b"WAVE", f + d) + b"NOT A CHUNK AT ALL"


def case_trailing_chunk_after_data():
    """A well-formed but unexpected chunk (e.g. cue points) after data."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    cue = chunk(b"cue ", struct.pack("<I", 0))  # empty cue list, valid shape
    return riff(b"WAVE", f + d + cue)


def case_zero_length_data():
    """data chunk present but contains zero frames."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d = data_chunk(b"")
    return riff(b"WAVE", f + d)


def case_multiple_data_chunks():
    """Two data chunks (invalid per spec, but seen from buggy encoders/
    crashed writes that appended a second session)."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d1 = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    d2 = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    return riff(b"WAVE", f + d1 + d2)


def case_mono_8bit_unsigned():
    """8-bit PCM is conventionally unsigned (offset-128), unlike every
    other PCM depth which is signed. Easy place for sign bugs."""
    f = fmt_pcm(channels=1, bits_per_sample=8)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 8, byte_width=1, signed=False))
    return riff(b"WAVE", f + d)


def case_extensible_8bit():
    """8-bit unsigned PCM in the extensible form, the only place the strict
    spec says a channel mask can live."""
    f = fmt_extensible(channels=2, bits_per_sample=8, valid_bits_per_sample=8,
                       channel_mask=0x3, sub_format=KSDATAFORMAT_SUBTYPE_PCM)
    # An unsigned ramp per channel, one rising and one falling around the
    # 128 center point (pcm_ramp_data only makes a rising unsigned ramp).
    payload = bytearray()
    for frame in range(NUM_FRAMES):
        step = int(127 * frame / (NUM_FRAMES - 1))
        payload += bytes([128 + step, 128 - step])
    d = data_chunk(bytes(payload))
    return riff(b"WAVE", f + d)


def g711_ramp_data(num_frames, channels):
    """A spread of G.711 code words. The values are the raw bytes, so what they
    decode to depends on the law; all that matters here is that they cover the
    whole code space rather than one corner of it."""
    out = bytearray()
    for frame in range(num_frames):
        for ch in range(channels):
            step = (frame * 256) // max(1, num_frames)
            out.append((step + ch * 128) % 256)
    return bytes(out)


def case_mulaw_mono():
    """Mu-law (format tag 7), the 18-byte WAVEFORMATEX form most encoders
    write."""
    f = fmt_g711(7, channels=1)
    d = data_chunk(g711_ramp_data(NUM_FRAMES, 1))
    return riff(b"WAVE", f + d)


def case_mulaw_stereo():
    """Mu-law with two channels, to check the framing is one byte per channel
    rather than the two a 16-bit assumption would give."""
    f = fmt_g711(7, channels=2)
    d = data_chunk(g711_ramp_data(NUM_FRAMES, 2))
    return riff(b"WAVE", f + d)


def case_alaw_mono():
    """A-law (format tag 6), the other half of G.711."""
    f = fmt_g711(6, channels=1)
    d = data_chunk(g711_ramp_data(NUM_FRAMES, 1))
    return riff(b"WAVE", f + d)


def case_alaw_stereo():
    """A-law with two channels."""
    f = fmt_g711(6, channels=2)
    d = data_chunk(g711_ramp_data(NUM_FRAMES, 2))
    return riff(b"WAVE", f + d)


def case_extensible_alaw():
    """A-law carried in a WAVEFORMATEXTENSIBLE header, matched by the
    KSDATAFORMAT_SUBTYPE_ALAW GUID rather than the plain format tag."""
    f = fmt_extensible(channels=2, bits_per_sample=8, valid_bits_per_sample=8,
                       channel_mask=0x3, sub_format=KSDATAFORMAT_SUBTYPE_ALAW,
                       byte_width=1)
    d = data_chunk(g711_ramp_data(NUM_FRAMES, 2))
    return riff(b"WAVE", f + d)


def case_extensible_mulaw():
    """Mu-law carried in a WAVEFORMATEXTENSIBLE header."""
    f = fmt_extensible(channels=2, bits_per_sample=8, valid_bits_per_sample=8,
                       channel_mask=0x3, sub_format=KSDATAFORMAT_SUBTYPE_MULAW,
                       byte_width=1)
    d = data_chunk(g711_ramp_data(NUM_FRAMES, 2))
    return riff(b"WAVE", f + d)


def case_ima_adpcm_mono():
    """IMA ADPCM (format tag 0x11): a valid format with no audioadapter sample
    type, so it parses but stays uninterpreted and is readable only as raw
    bytes. This is the fixture guarding that path."""
    f = fmt_ima_adpcm(channels=1)
    # One block's worth of bytes, contents irrelevant since nothing decodes it.
    d = data_chunk(bytes((i * 7) % 256 for i in range(256)))
    return riff(b"WAVE", f + d)


def case_ms_adpcm_stereo():
    """MS ADPCM (format tag 2): a 50-byte fmt chunk, longer than the 40-byte
    extensible form without being extensible. Uninterpreted, raw path only."""
    f = fmt_ms_adpcm(channels=2)
    d = data_chunk(bytes((i * 11) % 256 for i in range(512)))
    return riff(b"WAVE", f + d)


def case_odd_length_fmt_extension():
    """A fmt chunk with a 3-byte extension, so the body is 21 bytes and the
    chunk needs a RIFF pad byte. Every standard fmt body (16, 18, 40) is even,
    so nothing else exercises the padding, and getting it wrong shifts every
    later offset in the file by one."""
    payload = struct.pack(
        "<HHIIHH",
        0x99,  # an unassigned tag: uninterpreted, raw path only
        1,
        SAMPLE_RATE,
        SAMPLE_RATE,
        1,
        8,
    )
    payload += struct.pack("<H", 3) + b"\xaa\xbb\xcc"  # cbSize = 3
    f = chunk(b"fmt ", payload)
    d = data_chunk(bytes(range(NUM_FRAMES)))
    return riff(b"WAVE", f + d)


def case_extensible_too_short():
    """Format tag 0xFFFE but only an 18-byte fmt chunk, so the subformat GUID
    that names the real format is not there. Used to abort the whole parse;
    should degrade to the uninterpreted raw path like any other format we
    cannot make sense of."""
    payload = struct.pack("<HHIIHH", 0xFFFE, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    payload += struct.pack("<H", 0)  # cbSize = 0, no extensible fields at all
    f = chunk(b"fmt ", payload)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, 1, 16))
    return riff(b"WAVE", f + d)


def case_gsm610_mono():
    """MS GSM 6.10 (format tag 0x31): zero bits per sample, an odd 65-byte
    block alignment, and a cbSize extension. Uninterpreted, raw path only.
    Three blocks, so the data chunk is 195 bytes and therefore odd-sized and
    padded, which is the whole point of the odd block alignment."""
    f = fmt_gsm610()
    blocks = 3
    # Real GSM blocks start with the 0xD magic nibble; the rest is irrelevant
    # since nothing decodes it.
    payload = bytearray()
    for block in range(blocks):
        payload.append(0xD0 | block)
        payload += bytes((i * 13 + block) % 256 for i in range(64))
    fact = chunk(b"fact", struct.pack("<I", blocks * 320))
    d = data_chunk(bytes(payload))
    return riff(b"WAVE", f + fact + d)


def case_huge_channel_count():
    """Unusual but spec-legal high channel count (e.g. ambisonics/array
    mics), exercises any hardcoded mono/stereo/5.1 assumptions."""
    channels = 16
    f = fmt_pcm(channels=channels, bits_per_sample=16)
    d = data_chunk(pcm_ramp_data(NUM_FRAMES, channels, 16))
    return riff(b"WAVE", f + d)


def case_empty_riff_no_data_chunk():
    """fmt present, no data chunk at all (e.g. a header-only template, or
    a crashed write that never got to the audio)."""
    f = fmt_pcm(channels=1, bits_per_sample=16)
    return riff(b"WAVE", f)


def case_rf64_16bit_stereo():
    """Canonical RF64: RF64 form id, a ds64 chunk carrying the 64-bit data size
    and sample count, and a data chunk whose 32-bit size is the 0xFFFFFFFF marker.
    The everyday real-world shape of a >4 GB-capable file (kept tiny here)."""
    payload = pcm_ramp_data(NUM_FRAMES, 2, 16)
    f = fmt_pcm(channels=2, bits_per_sample=16)
    d = data_chunk_marker(payload)
    return rf64(f + d, data_size=len(payload), sample_count=NUM_FRAMES)


def case_bw64_16bit_stereo():
    """The same file as rf64_16bit_stereo but with the BW64 form id (ITU-R
    BS.2088), which is structurally identical to RF64 and must read the same."""
    payload = pcm_ramp_data(NUM_FRAMES, 2, 16)
    f = fmt_pcm(channels=2, bits_per_sample=16)
    d = data_chunk_marker(payload)
    return rf64(f + d, data_size=len(payload), sample_count=NUM_FRAMES, form=b"BW64")


def case_rf64_float32_real_size():
    """RF64 float32 mono where the data chunk keeps its real 32-bit size (it fits
    in 32 bits) instead of the marker. ds64 still carries the sizes, but the
    parser must take the non-marker field at face value. Also confirms RF64 needs
    no fact chunk to report its frame count."""
    payload = float_ramp_data(NUM_FRAMES, 1, bits_per_sample=32)
    f = fmt_float(channels=1, bits_per_sample=32)
    d = data_chunk(payload)  # real 32-bit size, not the marker
    return rf64(f + d, data_size=len(payload), sample_count=NUM_FRAMES)


def case_rf64_chunk_size_in_table():
    """RF64 where a non-data chunk (a JUNK filler) is sized through the ds64
    table: its own 32-bit size field is the 0xFFFFFFFF marker and the real size
    is an entry in the ds64 size table. Exercises the table-lookup path that the
    dedicated data/riff fields don't."""
    junk_payload = b"\x00" * 8
    junk = b"JUNK" + struct.pack("<I", DS64_SIZE_MARKER) + junk_payload
    payload = pcm_ramp_data(NUM_FRAMES, 1, 16)
    f = fmt_pcm(channels=1, bits_per_sample=16)
    d = data_chunk_marker(payload)
    return rf64(
        f + junk + d,
        data_size=len(payload),
        sample_count=NUM_FRAMES,
        table=[(b"JUNK", len(junk_payload))],
    )


CASES = {
    "baseline_16bit_stereo": case_baseline_16bit_stereo,
    "canonical_cd_16bit": case_canonical_cd_16bit,
    "canonical_24bit_48k": case_canonical_24bit_48k,
    "int32": case_int32,
    "float64": case_float64,
    "fmt_size_18_cbsize_zero": case_fmt_size_18_cbsize_zero,
    "24bit_packed": case_24bit_packed,
    "24bit_padded": case_24bit_padded,
    "float32": case_float32,
    "extensible_24in32_5point1": case_extensible_24in32_5point1,
    "extensible_float": case_extensible_float,
    "float32_waveformatex18": case_float32_waveformatex18,
    "extensible_16bit": case_extensible_16bit,
    "extensible_24bit_packed": case_extensible_24bit_packed,
    "extensible_int32": case_extensible_int32,
    "extensible_float64": case_extensible_float64,
    "extensible_24in32_strict": case_extensible_24in32_strict,
    "junk_before_fmt": case_junk_before_fmt,
    "list_info_between_fmt_and_data": case_list_info_between_fmt_and_data,
    "data_before_fmt": case_data_before_fmt,
    "odd_sized_data_with_pad": case_odd_sized_data_with_pad,
    "odd_sized_data_missing_pad": case_odd_sized_data_missing_pad,
    "riff_size_too_large": case_riff_size_too_large,
    "data_size_too_large": case_data_size_too_large,
    "data_size_streaming_placeholder": case_data_size_streaming_placeholder,
    "trailing_junk_after_data": case_trailing_junk_after_data,
    "trailing_chunk_after_data": case_trailing_chunk_after_data,
    "zero_length_data": case_zero_length_data,
    "multiple_data_chunks": case_multiple_data_chunks,
    "mono_8bit_unsigned": case_mono_8bit_unsigned,
    "extensible_8bit": case_extensible_8bit,
    "mulaw_mono": case_mulaw_mono,
    "mulaw_stereo": case_mulaw_stereo,
    "alaw_mono": case_alaw_mono,
    "alaw_stereo": case_alaw_stereo,
    "extensible_alaw": case_extensible_alaw,
    "extensible_mulaw": case_extensible_mulaw,
    "ima_adpcm_mono": case_ima_adpcm_mono,
    "gsm610_mono": case_gsm610_mono,
    "odd_length_fmt_extension": case_odd_length_fmt_extension,
    "extensible_too_short": case_extensible_too_short,
    "ms_adpcm_stereo": case_ms_adpcm_stereo,
    "huge_channel_count": case_huge_channel_count,
    "empty_riff_no_data_chunk": case_empty_riff_no_data_chunk,
    "rf64_16bit_stereo": case_rf64_16bit_stereo,
    "bw64_16bit_stereo": case_bw64_16bit_stereo,
    "rf64_float32_real_size": case_rf64_float32_real_size,
    "rf64_chunk_size_in_table": case_rf64_chunk_size_in_table,
}


def main():
    default_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "wav_variants")
    )
    out_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in CASES.items():
        content = fn()
        path = os.path.join(out_dir, f"{name}.wav")
        with open(path, "wb") as fh:
            fh.write(content)
        print(f"wrote {path} ({len(content)} bytes)")
    print(f"\n{len(CASES)} files written to {out_dir}/")


if __name__ == "__main__":
    main()
