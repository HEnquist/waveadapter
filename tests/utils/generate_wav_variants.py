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
    "huge_channel_count": case_huge_channel_count,
    "empty_riff_no_data_chunk": case_empty_riff_no_data_chunk,
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
