//! Encode an audio file or stream to raw E-AC-3.
//!
//! The input is decoded with ffmpeg and converted to 48 kHz `f32`,
//! then encoded with the `reac3e` stream encoder.
//! Raw E-AC-3 syncframes are written to a file or to stdout.
//!
//! Examples:
//!
//! ```sh
//! cargo run --release --example reac3e -- input.wav output.ec3
//! cargo run --release --example reac3e -- input.wav - > output.ec3
//! cat input.wav | cargo run --release --example reac3e -- - - > output.ec3
//! cargo run --release --example reac3e -- input.wav output.ec3 --layout 5.1 -b 448
//! cargo run --release --example reac3e -- input.flac output.ec3
//! ```

use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter, Write};

use clap::{Parser, ValueEnum};
use ffmpeg::format::Sample;
use ffmpeg::software::resampling;
use ffmpeg_next as ffmpeg;

use reac3e::{
    Acmod, BsiMetadata, ChannelLayout, Coupling, EncoderConfig, SAMPLE_RATE, StreamEncoder,
};

#[derive(Parser)]
#[command(
    name = "reac3e",
    version,
    about = "Encode an audio file or stream to raw E-AC-3"
)]
struct Cli {
    /// Input audio file (WAV, FLAC, ...), or '-' for stdin.
    #[arg(value_name = "INPUT", default_value = "-")]
    input: String,

    /// Output E-AC-3 file, or '-' for stdout.
    #[arg(value_name = "OUTPUT", default_value = "-")]
    output: String,

    /// Target bitrate in kbps (default depends on the channel layout).
    #[arg(short, long, value_name = "KBPS")]
    bitrate: Option<u32>,

    /// Dialogue normalisation level in dB (-31..=-1).
    #[arg(
        short,
        long,
        value_name = "DB",
        default_value_t = -27,
        allow_hyphen_values = true
    )]
    dialnorm: i8,

    /// Channel layout (default: inferred from the input channel count).
    #[arg(short, long, value_enum)]
    layout: Option<LayoutArg>,

    /// Channel coupling mode.
    #[arg(short, long, value_enum, default_value_t = CouplingArg::Auto)]
    coupling: CouplingArg,

    /// Disable stereo rematrixing.
    #[arg(long)]
    no_rematrixing: bool,

    /// Disable block switching (enabled by default).
    #[arg(long)]
    no_block_switch: bool,

    /// Override the coded bandwidth cutoff in Hz (default: automatic).
    #[arg(long, value_name = "HZ")]
    cutoff_hz: Option<u32>,

    /// Suppress informational messages on stderr.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LayoutArg {
    Mono,
    DualMono,
    Stereo,
    #[value(name = "2.1")]
    StereoLfe,
    #[value(name = "3.0")]
    Surround,
    #[value(name = "3.1")]
    SurroundLfe,
    #[value(name = "4.0")]
    Quad,
    #[value(name = "4.1")]
    QuadLfe,
    #[value(name = "5.0")]
    FiveZero,
    #[value(name = "5.1")]
    FiveOne,
}

impl LayoutArg {
    fn layout(self) -> ChannelLayout {
        let (acmod, lfe) = match self {
            Self::Mono => (Acmod::Mono, false),
            Self::DualMono => (Acmod::DualMono, false),
            Self::Stereo => (Acmod::Stereo, false),
            Self::StereoLfe => (Acmod::Stereo, true),
            Self::Surround => (Acmod::Surround, false),
            Self::SurroundLfe => (Acmod::Surround, true),
            Self::Quad => (Acmod::TwoTwo, false),
            Self::QuadLfe => (Acmod::TwoTwo, true),
            Self::FiveZero => (Acmod::ThreeTwo, false),
            Self::FiveOne => (Acmod::ThreeTwo, true),
        };
        ChannelLayout { acmod, lfe }
    }
}

impl From<LayoutArg> for ChannelLayout {
    fn from(value: LayoutArg) -> Self {
        value.layout()
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CouplingArg {
    Auto,
    On,
    Off,
}

impl From<CouplingArg> for Coupling {
    fn from(value: CouplingArg) -> Self {
        match value {
            CouplingArg::Auto => Coupling::Auto,
            CouplingArg::On => Coupling::On,
            CouplingArg::Off => Coupling::Off,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    let decoded = decode_audio(&cli.input)?;
    let channels = decoded.channels;
    let source_rate = decoded.source_rate;
    let input_frames = decoded.frames();
    if input_frames == 0 {
        return Err("input contains no audio samples".into());
    }

    let layout = match cli.layout {
        Some(layout) => layout.layout(),
        None => infer_layout(decoded.channels)?,
    };
    if layout.channels() != decoded.channels {
        return Err(format!(
            "input has {} channels but layout '{}' needs {}",
            decoded.channels,
            layout_name(layout),
            layout.channels()
        )
        .into());
    }

    let bitrate_kbps = cli.bitrate.unwrap_or_else(|| default_bitrate_kbps(layout));
    if bitrate_kbps == 0 {
        return Err("bitrate must be non-zero".into());
    }
    let bitrate_bps = bitrate_kbps
        .checked_mul(1000)
        .ok_or("bitrate is too large")?;

    let samples = decoded.samples;
    let resampled = source_rate != SAMPLE_RATE;

    let config = EncoderConfig {
        layout,
        bitrate_bps,
        dialnorm: cli.dialnorm,
        metadata: BsiMetadata::default(),
        coupling: cli.coupling.into(),
        stereo_rematrixing: !cli.no_rematrixing,
        cutoff_hz: cli.cutoff_hz,
    };

    let mut encoder = StreamEncoder::new(config)?;
    encoder.set_block_switch(!cli.no_block_switch);
    let mut output = open_output(&cli.output)?;
    let chunk_samples = 4096 * channels;
    let mut syncframes = 0u64;
    let mut bytes_written = 0u64;

    for chunk in samples.chunks(chunk_samples) {
        let frames = encoder.push_interleaved(chunk)?;
        syncframes += frames.len() as u64;
        bytes_written += write_frames(&mut output, frames)?;
    }
    let frames = encoder.finish()?;
    syncframes += frames.len() as u64;
    bytes_written += write_frames(&mut output, frames)?;
    output.flush()?;

    if !cli.quiet {
        let source = if cli.input == "-" {
            "stdin"
        } else {
            &cli.input
        };
        let destination = if cli.output == "-" {
            "stdout"
        } else {
            &cli.output
        };
        eprintln!(
            "input: {source}, {} Hz, {} ch, {} frames",
            source_rate, channels, input_frames
        );
        if resampled {
            eprintln!("resampled: {} Hz -> {} Hz", source_rate, SAMPLE_RATE);
        }
        eprintln!(
            "layout: {} ({} ch), bitrate: {} kbps, dialnorm: {} dB, coupling: {:?}, cutoff: {:.0} Hz",
            layout_name(layout),
            layout.channels(),
            bitrate_kbps,
            cli.dialnorm,
            cli.coupling,
            encoder.cutoff_hz()
        );
        eprintln!("output: {destination}, {syncframes} syncframes, {bytes_written} bytes");
    }

    Ok(())
}

fn open_output(path: &str) -> io::Result<Box<dyn Write>> {
    if path == "-" {
        Ok(Box::new(BufWriter::new(io::stdout())))
    } else {
        Ok(Box::new(BufWriter::new(File::create(path)?)))
    }
}

fn write_frames(output: &mut dyn Write, frames: Vec<reac3e::Frame>) -> io::Result<u64> {
    let mut written = 0u64;
    for frame in frames {
        output.write_all(&frame.bytes)?;
        written += frame.bytes.len() as u64;
    }
    Ok(written)
}

fn infer_layout(channels: usize) -> Result<ChannelLayout, Box<dyn Error>> {
    let (acmod, lfe) = match channels {
        1 => (Acmod::Mono, false),
        2 => (Acmod::Stereo, false),
        3 => (Acmod::Surround, false),
        4 => (Acmod::TwoTwo, false),
        5 => (Acmod::ThreeTwo, false),
        6 => (Acmod::ThreeTwo, true),
        _ => {
            return Err(format!(
                "unsupported channel count {channels}; this encoder supports at most 5.1 (6 channels)"
            )
            .into());
        }
    };
    Ok(ChannelLayout { acmod, lfe })
}

fn default_bitrate_kbps(layout: ChannelLayout) -> u32 {
    match (layout.acmod, layout.lfe) {
        (Acmod::Mono, _) => 96,
        (Acmod::DualMono, _) => 192,
        (Acmod::Stereo, false) => 192,
        (Acmod::Stereo, true) => 224,
        (Acmod::Surround, false) => 256,
        (Acmod::Surround, true) => 320,
        (Acmod::TwoOne, _) => 256,
        (Acmod::ThreeOne, false) => 320,
        (Acmod::ThreeOne, true) => 384,
        (Acmod::TwoTwo, false) => 320,
        (Acmod::TwoTwo, true) => 384,
        (Acmod::ThreeTwo, false) => 384,
        (Acmod::ThreeTwo, true) => 448,
    }
}

fn layout_name(layout: ChannelLayout) -> &'static str {
    match (layout.acmod, layout.lfe) {
        (Acmod::Mono, _) => "mono",
        (Acmod::DualMono, _) => "dual-mono",
        (Acmod::Stereo, false) => "stereo",
        (Acmod::Stereo, true) => "2.1",
        (Acmod::Surround, false) => "3.0",
        (Acmod::Surround, true) => "3.1",
        (Acmod::TwoOne, _) => "2/1",
        (Acmod::ThreeOne, false) => "3/1",
        (Acmod::ThreeOne, true) => "3.1-surround",
        (Acmod::TwoTwo, false) => "4.0",
        (Acmod::TwoTwo, true) => "4.1",
        (Acmod::ThreeTwo, false) => "5.0",
        (Acmod::ThreeTwo, true) => "5.1",
    }
}

/// Sample rate of the decoded output.
const TARGET_RATE: u32 = SAMPLE_RATE;
const TARGET_FORMAT: Sample = Sample::F32(ffmpeg::format::sample::Type::Planar);

/// Decoded, interleaved `f32` audio at [`TARGET_RATE`].
struct DecodedAudio {
    samples: Vec<f32>,
    channels: usize,
    /// Sample rate of the input before conversion.
    source_rate: u32,
}

impl DecodedAudio {
    fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }
}

/// Decode `path` (or stdin if `path` is `-`) to interleaved 48 kHz `f32`.
fn decode_audio(path: &str) -> Result<DecodedAudio, Box<dyn Error>> {
    ffmpeg::init()?;
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Error);

    let mut input = if path == "-" {
        let io = ffmpeg::format::context::StreamIo::from_read(io::stdin())?;
        ffmpeg::format::input_from_stream(io, None, None)
            .map_err(|error| format!("stdin: {error}"))?
    } else {
        ffmpeg::format::input(path).map_err(|error| format!("{path}: {error}"))?
    };

    let (stream_index, mut decoder) = {
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| format!("{path}: no audio stream"))?;
        let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
            .decoder()
            .audio()?;
        (stream.index(), decoder)
    };

    let mut converter: Option<Converter> = None;
    let mut samples = Vec::new();

    for (stream, packet) in input.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        drain_decoder(&mut decoder, &mut converter, &mut samples)?;
    }
    decoder.send_eof()?;
    drain_decoder(&mut decoder, &mut converter, &mut samples)?;
    flush_converter(&mut converter, &mut samples)?;

    let converter = converter.ok_or_else(|| format!("{path}: no audio frames decoded"))?;
    if samples.is_empty() {
        return Err(format!("{path}: decoded audio is empty").into());
    }

    Ok(DecodedAudio {
        samples,
        channels: converter.channels,
        source_rate: converter.source_rate,
    })
}

struct Converter {
    context: resampling::Context,
    layout: ffmpeg::ChannelLayout,
    channels: usize,
    source_format: Sample,
    source_rate: u32,
    output: ffmpeg::frame::Audio,
}

fn drain_decoder(
    decoder: &mut ffmpeg::codec::decoder::Audio,
    converter: &mut Option<Converter>,
    samples: &mut Vec<f32>,
) -> Result<(), Box<dyn Error>> {
    loop {
        let mut frame = ffmpeg::frame::Audio::empty();
        match decoder.receive_frame(&mut frame) {
            Ok(()) => handle_frame(&frame, converter, samples)?,
            Err(ffmpeg::Error::Eof) => break,
            Err(ffmpeg::Error::Other { errno }) if errno == libc::EAGAIN => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn handle_frame(
    frame: &ffmpeg::frame::Audio,
    converter: &mut Option<Converter>,
    samples: &mut Vec<f32>,
) -> Result<(), Box<dyn Error>> {
    if frame.rate() == 0 {
        return Err("decoded audio has no sample rate".into());
    }
    if frame.samples() == 0 {
        return Ok(());
    }

    if converter.is_none() {
        let layout = normalized_layout(frame);
        let channels = layout.channels() as usize;
        if channels == 0 {
            return Err("decoded audio has no channels".into());
        }
        let context = resampling::Context::get(
            frame.format(),
            layout,
            frame.rate(),
            TARGET_FORMAT,
            layout,
            TARGET_RATE,
        )?;
        let capacity = ((frame.samples() as f64 * TARGET_RATE as f64 / frame.rate() as f64).ceil()
            as usize)
            .max(1024)
            + 4096;
        let output = ffmpeg::frame::Audio::new(TARGET_FORMAT, capacity, layout);
        *converter = Some(Converter {
            context,
            layout,
            channels,
            source_format: frame.format(),
            source_rate: frame.rate(),
            output,
        });
    }

    let converter = converter.as_mut().expect("converter initialized above");
    if frame.format() != converter.source_format
        || frame.rate() != converter.source_rate
        || frame.channels() as usize != converter.channels
    {
        return Err("input changes audio format mid-stream".into());
    }

    let mut input = frame.clone();
    input.set_channel_layout(converter.layout);
    converter.context.run(&input, &mut converter.output)?;
    append_interleaved(&converter.output, converter.channels, samples)?;
    Ok(())
}

fn flush_converter(
    converter: &mut Option<Converter>,
    samples: &mut Vec<f32>,
) -> Result<(), Box<dyn Error>> {
    let Some(converter) = converter.as_mut() else {
        return Ok(());
    };
    loop {
        let mut output = ffmpeg::frame::Audio::new(TARGET_FORMAT, 1024, converter.layout);
        converter.context.flush(&mut output)?;
        if output.samples() == 0 {
            break;
        }
        append_interleaved(&output, converter.channels, samples)?;
    }
    Ok(())
}

fn append_interleaved(
    frame: &ffmpeg::frame::Audio,
    channels: usize,
    samples: &mut Vec<f32>,
) -> Result<(), Box<dyn Error>> {
    let frames = frame.samples();
    if frames == 0 {
        return Ok(());
    }
    if frame.format() != TARGET_FORMAT {
        return Err("resampler did not return planar f32 samples".into());
    }
    if frame.channels() as usize != channels {
        return Err(format!(
            "resampler returned {} channels, expected {channels}",
            frame.channels()
        )
        .into());
    }
    samples.reserve(frames * channels);
    for i in 0..frames {
        for channel in 0..channels {
            samples.push(frame.plane::<f32>(channel)[i]);
        }
    }
    Ok(())
}

fn normalized_layout(frame: &ffmpeg::frame::Audio) -> ffmpeg::ChannelLayout {
    let layout = frame.channel_layout();
    if layout.is_empty() || layout.channels() <= 0 {
        ffmpeg::ChannelLayout::default(frame.channels() as i32)
    } else {
        layout
    }
}
