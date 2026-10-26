//! Perceptual quality gates.
//!
//! Three gates are applied per case:
//!
//! 1. An absolute MOS floor, so a catastrophic regression is caught even if
//!    FFmpeg regresses by the same amount.
//! 2. A relative gate against FFmpeg encoded from the same PCM: the candidate
//!    may diverge from FFmpeg's bitstream, but it must not be perceptually
//!    worse by more than `MOS_MARGIN`.
//! 3. The same relative gate per channel, so a single-channel regression
//!    cannot hide behind the mean.
//!
//! Requires Python 3 with the official `visqol-python` package:
//! `python3 -m pip install 'visqol-python[accel,fftw]'`.

mod common;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::sync::atomic::{AtomicU64, Ordering};

use common::{Case, decode, encode_ours, encode_reference, init_ffmpeg, stream_bytes};
use ffmpeg_next as ffmpeg;
use reac3e::{Acmod, BsiMetadata, Coupling};

/// The candidate may trail FFmpeg by this much MOS-LQO before the gate fails.
const MOS_MARGIN: f64 = 0.10;

/// Identity ceiling of the audio-mode SVR is about 4.73.
const TRANSPARENT: f64 = 4.60;

struct ChannelScores {
    per_channel: Vec<f64>,
}

impl ChannelScores {
    fn mean(&self) -> f64 {
        self.per_channel.iter().sum::<f64>() / self.per_channel.len() as f64
    }

    fn min(&self) -> f64 {
        self.per_channel
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }
}

/// Deterministic programme material: tones, low-passed noise, amplitude
/// modulation and (optionally) non-periodic transients.  One channel is
/// deliberately quiet to exercise low-level coding.
///
/// Phase is computed in `f64`: `f32` accumulation is not accurate enough for
/// a 2 s tone and would put broadband phase noise into the *reference*.
fn gen_material(case: &Case, transients: bool) -> Vec<f32> {
    let channels = case.channels();
    let duration = case.samples as f64 / 48_000.0;
    let mut out = vec![0.0f32; case.samples * channels];
    let mut state = 0xC0FF_EE11u32;
    let mut low = vec![0.0f64; channels];

    // Non-periodic transients: a periodic click train makes ViSQOL's global
    // alignment lock onto the wrong period and reports a false quality loss.
    let mut events = vec![Vec::new(); channels];
    for (ch, channel_events) in events.iter_mut().enumerate() {
        let mut seed = state ^ (ch as u32).wrapping_mul(0x9E37_79B9);
        for _ in 0..6 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let fraction = ((seed >> 8) as f64) / (1u32 << 24) as f64;
            channel_events.push(0.15 + fraction * (duration - 0.35));
        }
    }

    for n in 0..case.samples {
        let t = n as f64 / 48_000.0;
        for ch in 0..channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let white = ((state >> 8) as f64 / (1u32 << 24) as f64) * 2.0 - 1.0;

            low[ch] = 0.99 * low[ch] + 0.01 * white;

            let envelope = 0.75 + 0.25 * (2.0 * std::f64::consts::PI * 0.37 * t + ch as f64).sin();
            let f0 = 110.0 * 2.0f64.powf((ch % 5) as f64 / 3.0);
            let tone = (2.0 * std::f64::consts::PI * f0 * t).sin() * 0.18
                + (2.0 * std::f64::consts::PI * f0 * 2.9 * t + ch as f64).sin() * 0.06
                + (2.0 * std::f64::consts::PI * f0 * 7.3 * t).sin() * 0.03;

            let mut burst = 0.0;
            if transients {
                for &event in &events[ch] {
                    let dt = t - event;
                    if (0.0..0.030).contains(&dt) {
                        burst += (-dt * 120.0).exp()
                            * (2.0 * std::f64::consts::PI * 3_500.0 * dt).sin()
                            * 0.6;
                    }
                }
            }

            let gain = if channels > 2 && ch == channels - 1 {
                0.08
            } else {
                1.0
            };
            let sample = if case.lfe && ch == 3 {
                (2.0 * std::f64::consts::PI * 48.0 * t).sin() * 0.20
            } else {
                (tone * envelope + low[ch] * 0.5 + white * 0.01 + burst) * gain
            };
            out[n * channels + ch] = sample as f32;
        }
    }
    out
}

fn gen_program(case: &Case) -> Vec<f32> {
    gen_material(case, true)
}

/// Purely tonal material for the transparency test.  Broadband noise is not
/// a fair transparency probe: even a very high E-AC-3 rate spends bits on it.
fn gen_tonal(case: &Case) -> Vec<f32> {
    let channels = case.channels();
    let mut out = vec![0.0f32; case.samples * channels];
    for n in 0..case.samples {
        let t = n as f64 / 48_000.0;
        for ch in 0..channels {
            let f0 = 110.0 * 2.0f64.powf((ch % 5) as f64 / 3.0);
            let envelope = 0.75 + 0.25 * (2.0 * std::f64::consts::PI * 0.37 * t + ch as f64).sin();
            let sample = if case.lfe && ch == 3 {
                (2.0 * std::f64::consts::PI * 48.0 * t).sin() * 0.20
            } else {
                ((2.0 * std::f64::consts::PI * f0 * t).sin() * 0.15
                    + (2.0 * std::f64::consts::PI * f0 * 2.3 * t).sin() * 0.08
                    + (2.0 * std::f64::consts::PI * f0 * 5.1 * t).sin() * 0.05
                    + (2.0 * std::f64::consts::PI * f0 * 9.7 * t).sin() * 0.03)
                    * envelope
            };
            out[n * channels + ch] = sample as f32;
        }
    }
    out
}

fn write_mono_wav(path: &Path, samples: impl Iterator<Item = f32>) {
    let pcm: Vec<i16> = samples
        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
        .collect();
    let data_len = (pcm.len() * 2) as u32;
    let mut writer = BufWriter::new(File::create(path).unwrap());
    writer.write_all(b"RIFF").unwrap();
    writer.write_all(&(36 + data_len).to_le_bytes()).unwrap();
    writer.write_all(b"WAVEfmt ").unwrap();
    writer.write_all(&16u32.to_le_bytes()).unwrap();
    writer.write_all(&1u16.to_le_bytes()).unwrap();
    writer.write_all(&1u16.to_le_bytes()).unwrap();
    writer.write_all(&48_000u32.to_le_bytes()).unwrap();
    writer.write_all(&(48_000u32 * 2).to_le_bytes()).unwrap();
    writer.write_all(&2u16.to_le_bytes()).unwrap();
    writer.write_all(&16u16.to_le_bytes()).unwrap();
    writer.write_all(b"data").unwrap();
    writer.write_all(&data_len.to_le_bytes()).unwrap();
    for sample in pcm {
        writer.write_all(&sample.to_le_bytes()).unwrap();
    }
    writer.flush().unwrap();
}

fn temp_dir() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "reac3e_quality_{}_{}",
        process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

/// Score candidate and FFmpeg PCM against the reference with the official
/// ViSQOL Python implementation.  Both candidates are scored in one helper
/// process to amortize Python startup.
fn visqol_pair(
    reference: &[f32],
    ours: &[f32],
    ffmpeg: &[f32],
    channels: usize,
) -> (ChannelScores, ChannelScores) {
    let frames = (reference.len() / channels)
        .min(ours.len() / channels)
        .min(ffmpeg.len() / channels);
    assert!(frames > 0, "empty comparison");

    let directory = temp_dir();
    let mut reference_paths = Vec::with_capacity(channels);
    let mut ours_paths = Vec::with_capacity(channels);
    let mut ffmpeg_paths = Vec::with_capacity(channels);
    for channel in 0..channels {
        let reference_path = directory.join(format!("reference_{channel}.wav"));
        write_mono_wav(
            &reference_path,
            (0..frames).map(|n| reference[n * channels + channel]),
        );
        let ours_path = directory.join(format!("ours_{channel}.wav"));
        write_mono_wav(
            &ours_path,
            (0..frames).map(|n| ours[n * channels + channel]),
        );
        let ffmpeg_path = directory.join(format!("ffmpeg_{channel}.wav"));
        write_mono_wav(
            &ffmpeg_path,
            (0..frames).map(|n| ffmpeg[n * channels + channel]),
        );
        reference_paths.push(reference_path);
        ours_paths.push(ours_path);
        ffmpeg_paths.push(ffmpeg_path);
    }

    let mut arguments: Vec<PathBuf> = Vec::with_capacity(channels * 4);
    for channel in 0..channels {
        arguments.push(reference_paths[channel].clone());
        arguments.push(ours_paths[channel].clone());
    }
    for channel in 0..channels {
        arguments.push(reference_paths[channel].clone());
        arguments.push(ffmpeg_paths[channel].clone());
    }

    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/visqol_scores.py");
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python3".into());
    let output = Command::new(python)
        .arg(script)
        .args(&arguments)
        .output()
        .unwrap_or_else(|error| panic!("failed to run Python for ViSQOL: {error}"));
    fs::remove_dir_all(&directory).ok();

    if !output.status.success() {
        panic!(
            "official ViSQOL failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8(output.stdout).expect("ViSQOL output is not UTF-8");
    let mut scores = Vec::with_capacity(channels * 2);
    for line in stdout.lines() {
        let mut fields = line.split_whitespace();
        let mos = fields
            .next()
            .unwrap_or_else(|| panic!("malformed ViSQOL output:\n{stdout}"))
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("malformed ViSQOL output:\n{stdout}"));
        scores.push(mos);
    }
    assert_eq!(
        scores.len(),
        channels * 2,
        "unexpected ViSQOL output:\n{stdout}"
    );

    (
        ChannelScores {
            per_channel: scores[..channels].to_vec(),
        },
        ChannelScores {
            per_channel: scores[channels..].to_vec(),
        },
    )
}

fn run_case(case: Case, floor: f64, generate: fn(&Case) -> Vec<f32>) {
    init_ffmpeg();
    let pcm = generate(&case);
    let ours = stream_bytes(encode_ours(&case, &pcm, &BsiMetadata::default()));
    let reference = encode_reference(&case, &pcm, &[]);
    let ours_pcm = decode(&ours, case.channels());
    let ref_pcm = decode(&reference, case.channels());

    let (ours_mos, ref_mos) = visqol_pair(&pcm, &ours_pcm, &ref_pcm, case.channels());

    eprintln!(
        "{}: ours {:.4} (min {:.4})  ffmpeg {:.4} (min {:.4})  delta {:+.4}  ours per-channel [{}]",
        case.name,
        ours_mos.mean(),
        ours_mos.min(),
        ref_mos.mean(),
        ref_mos.min(),
        ours_mos.mean() - ref_mos.mean(),
        ours_mos
            .per_channel
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    assert!(
        ours_mos.mean() >= floor,
        "{}: mean MOS {:.4} below absolute floor {:.4}",
        case.name,
        ours_mos.mean(),
        floor
    );
    assert!(
        ours_mos.min() >= floor - 0.50,
        "{}: worst channel MOS {:.4} below absolute floor {:.4}",
        case.name,
        ours_mos.min(),
        floor
    );
    assert!(
        ours_mos.mean() >= ref_mos.mean() - MOS_MARGIN,
        "{}: mean MOS {:.4} is more than {MOS_MARGIN} below FFmpeg {:.4}",
        case.name,
        ours_mos.mean(),
        ref_mos.mean()
    );
    for (channel, (ours, reference)) in ours_mos
        .per_channel
        .iter()
        .zip(&ref_mos.per_channel)
        .enumerate()
    {
        assert!(
            *ours >= reference - MOS_MARGIN,
            "{}: channel {channel} MOS {ours:.4} is more than {MOS_MARGIN} below FFmpeg {reference:.4}",
            case.name
        );
    }
}

#[test]
fn stereo_high_bitrate_is_transparent() {
    run_case(
        Case {
            name: "quality_stereo_4m",
            acmod: Acmod::Stereo,
            lfe: false,
            layout: ffmpeg::ChannelLayout::STEREO,
            bitrate: 4_000_000,
            coupling: Coupling::Auto,
            rematrix: true,
            samples: 96_000,
        },
        TRANSPARENT,
        gen_tonal,
    );
}

#[test]
fn stereo_192k_matches_ffmpeg_quality() {
    run_case(
        Case {
            name: "quality_stereo_192k",
            acmod: Acmod::Stereo,
            lfe: false,
            layout: ffmpeg::ChannelLayout::STEREO,
            bitrate: 192_000,
            coupling: Coupling::Auto,
            rematrix: true,
            samples: 96_000,
        },
        3.35,
        gen_program,
    );
}

#[test]
fn surround_51_640k_matches_ffmpeg_quality() {
    run_case(
        Case {
            name: "quality_51_640k",
            acmod: Acmod::ThreeTwo,
            lfe: true,
            layout: ffmpeg::ChannelLayout::_5POINT1,
            bitrate: 640_000,
            coupling: Coupling::Auto,
            rematrix: false,
            samples: 96_000,
        },
        3.75,
        gen_program,
    );
}

#[test]
fn surround_51_448k_coupling_off_matches_ffmpeg_quality() {
    run_case(
        Case {
            name: "quality_51_448k_nocpl",
            acmod: Acmod::ThreeTwo,
            lfe: true,
            layout: ffmpeg::ChannelLayout::_5POINT1,
            bitrate: 448_000,
            coupling: Coupling::Off,
            rematrix: false,
            samples: 96_000,
        },
        3.50,
        gen_program,
    );
}
