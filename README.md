# reac3e

A standalone E-AC-3 encoder written in Rust.

`reac3e` produces raw E-AC-3 syncframes at 48 kHz. It implements the coding
tools needed for high-quality output: KBD-windowed MDCT, block switching,
differential exponent coding, channel coupling, stereo rematrixing, CBR bit
allocation and mantissa packing. The public API is frame- or stream-oriented
and has no container or Dolby-specific metadata dependency.

## Scope

Supported:

- 48 kHz
- `acmod` 1+1, 1/0, 2/0, 3/0, 2/1, 3/1, 2/2 and 3/2
- optional LFE
- 1, 2, 3 or 6 blocks per syncframe
- CBR
- block switching with the A/52 transient detector; enabled by default at up
  to 64 kbps per channel and disabled with `--no-block-switch`
- channel coupling (auto / on / off / start band 0..=15)
- stereo rematrixing
- AHT (adaptive hybrid transform) for stationary frames (experimental, `--aht`)
- the BSI metadata fields that the E-AC-3 syntax exposes (dialnorm, mixing and
  information metadata)

Not supported (these return a clear configuration error):

- AC-3 encoding
- sample rates other than 48 kHz
- spectral extension, enhanced coupling, transient pre-noise
- delta bit allocation, dither
- VBR / ABR
- dependent substreams or layouts above 5.1

## Usage

```rust
use reac3e::{
    Acmod, BsiMetadata, ChannelLayout, ChannelOrder, Coupling, EncoderConfig,
    FrameEncoder, PlanarFrame,
};

let cfg = EncoderConfig {
    layout: ChannelLayout { acmod: Acmod::Stereo, lfe: false },
    bitrate_bps: 192_000,
    dialnorm: -27,
    metadata: BsiMetadata::default(),
    coupling: Coupling::Auto,
    coupling_start: None,
    stereo_rematrixing: true,
    cutoff_hz: None,
};

let mut enc = FrameEncoder::new(cfg)?;
let samples = enc.samples_per_frame();

// One internal channel slice per channel, SMPTE/WAVE order (L, R, ...).
let left = vec![0.0f32; samples];
let right = vec![0.0f32; samples];
let channels: [&[f32]; 2] = [&left, &right];
let frame = enc.encode_frame(&PlanarFrame::new(&channels, ChannelOrder::Smpte))?;
# Ok::<(), reac3e::EncodeError>(())
```

For arbitrary chunk sizes, use `StreamEncoder::push_interleaved` and
`StreamEncoder::finish`. Interleaved input is in SMPTE/WAVE order for the
configured layout.

The encoder has a 256-sample overlap latency, reported by
`FrameEncoder::latency`. The first frame is produced from the input samples
with a zeroed history, matching the conventional E-AC-3 filterbank start-up.

### CLI example

`examples/reac3e.rs` is a small encoder that reads any audio file (or stdin)
ffmpeg can decode, converts it to 48 kHz `f32`, and writes raw E-AC-3
syncframes to a file or stdout.

```sh
cargo run --release --example reac3e -- input.wav output.ec3
cargo run --release --example reac3e -- input.flac output.ec3
cargo run --release --example reac3e -- input.wav - > output.ec3
cat input.wav | cargo run --release --example reac3e -- - - > output.ec3

# Explicit layout and bitrate (kbps)
cargo run --release --example reac3e -- input.wav output.ec3 --layout 5.1 -b 448
```

Input channel counts of 1, 2, 3, 4, 5 and 6 map to mono, stereo, 3.0, 4.0,
5.0 and 5.1 respectively; use `--layout` to select another supported AC-3
layout (for example `2.1`, `3.1` or `4.1`). Channels are expected in WAVE
order. `--dialnorm`, `--coupling`, `--coupling-start`, `--no-rematrixing`,
`--no-block-switch`, `--aht`, `--cutoff-hz` and `--quiet` expose the remaining
encoder options.  The bandwidth cutoff defaults to a bitrate-dependent table;
`--cutoff-hz` overrides it (the nearest representable E-AC-3 bandwidth code is
used). `--coupling-start` overrides the coupling start band (0..=15), which is
roughly 3.5 kHz plus 1.125 kHz per band. Use
`--help` for the full list. The output is raw E-AC-3 syncframes, not a
container format.

### ViSQOL measurement

`tools/measure_visqol.py` measures ViSQOL MOS-LQO with the
[`visqol-python`](https://pypi.org/project/visqol-python/) implementation.

```sh
python3 -m pip install 'visqol-python[accel,fftw]' soundfile numpy

python3 tools/measure_visqol.py reference.wav encoded.ec3
python3 tools/measure_visqol.py reference.flac encoded.mp3
cat reference.wav | python3 tools/measure_visqol.py - encoded.ec3
```

The mean, minimum and per-channel MOS-LQO scores (plus VNSIM) are printed
to stdout; `--quiet` suppresses the per-channel progress lines on stderr.

### Zimtohrli measurement

`tools/measure_zimtohrli.py` measures Zimtohrli distance and MOS using the
[`zimtohrli`](https://pypi.org/project/zimtohrli/) package. Lower distance is
better; the MOS is an approximation derived from it.

```sh
python3 -m pip install zimtohrli soundfile numpy

python3 tools/measure_zimtohrli.py reference.wav encoded.ec3
python3 tools/measure_zimtohrli.py reference.flac encoded.mp3
cat reference.wav | python3 tools/measure_zimtohrli.py - encoded.ec3
```

## Channel orders

- `ChannelOrder::Smpte`: `L R C LFE Ls Rs` for 5.1, with the LFE placed after
  the front channels for smaller layouts.
- `ChannelOrder::Ac3`: the internal AC-3 order, e.g. `L C R Ls Rs LFE`.

## Tests

### Conformance

```sh
cargo test --release --test conformance
```

This suite never compares against FFmpeg's encoder output, so it remains valid
when the bitstream legitimately diverges.

### Perceptual quality

`tests/quality.rs` scores both the candidate and the FFmpeg reference against
the original PCM with Zimtohrli, through `tests/zimtohrli_scores.py`. Install it
first:

```sh
python3 -m pip install zimtohrli
```

Set `PYTHON` to use a specific interpreter. Each case checks:

- an absolute Zimtohrli MOS floor
- mean and per-channel MOS within `MOS_MARGIN` of the FFmpeg reference

```sh
cargo test --release --test quality -- --nocapture
```

Run this in release mode. The output prints per-channel MOS for the candidate
and the FFmpeg reference.

Run everything:

```sh
cargo test --release
```

## Licensing and provenance

LGPL-2.1-or-later. Some algorithms and constant tables are derived from the
FFmpeg project's E-AC-3 encoder. See `NOTICE` for attribution and `LICENSE` for
the full licence text.
