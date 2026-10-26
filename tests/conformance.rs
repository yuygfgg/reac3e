//! Specification and decoder-conformance checks.
//! - every syncframe is well formed and self-consistent
//! - the CBR rate matches the configured bitrate
//! - an independent decoder accepts the stream and produces the expected
//!   number of channels/samples, at the right level

mod common;

use common::{Case, decode, encode_ours, gen_pcm, split_frames, stream_bytes};
use ffmpeg_next as ffmpeg;
use reac3e::{Acmod, BsiMetadata, Coupling};

fn check(case: Case) {
    let pcm = gen_pcm(&case);
    let bytes = stream_bytes(encode_ours(&case, &pcm, &BsiMetadata::default()));
    let frames = split_frames(&bytes);
    assert!(!frames.is_empty(), "{}: no syncframes", case.name);

    let duration = case.samples as f64 / 48_000.0;
    let actual_bitrate = bytes.len() as f64 * 8.0 / duration;
    let tolerance = case.bitrate as f64 * 0.015;
    assert!(
        (actual_bitrate - case.bitrate as f64).abs() <= tolerance,
        "{}: CBR rate {actual_bitrate:.0} bps vs target {} bps",
        case.name,
        case.bitrate
    );

    let decoded = decode(&bytes, case.channels());
    let decoded_frames = decoded.len() / case.channels();
    assert!(
        decoded_frames >= case.samples,
        "{}: decoder produced {decoded_frames} < {} samples",
        case.name,
        case.samples
    );
    assert!(
        decoded_frames - case.samples <= 1536,
        "{}: decoder produced {} samples, more than one frame of padding",
        case.name,
        decoded_frames
    );

    let input_rms =
        (pcm.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / pcm.len() as f64).sqrt();
    let output_rms =
        (decoded.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / decoded.len() as f64).sqrt();
    let level_error = 20.0 * (output_rms / input_rms).log10();
    assert!(
        level_error.abs() <= 3.0,
        "{}: decoded level changed by {level_error:+.2} dB",
        case.name
    );
}

fn case(
    name: &'static str,
    acmod: Acmod,
    lfe: bool,
    layout: ffmpeg::ChannelLayout,
    bitrate: u32,
    coupling: Coupling,
    rematrix: bool,
) -> Case {
    Case {
        name,
        acmod,
        lfe,
        layout,
        bitrate,
        coupling,
        rematrix,
        // One second: enough for several syncframes at every rate.
        samples: 96_000,
    }
}

#[test]
fn mono_96k() {
    check(case(
        "mono96",
        Acmod::Mono,
        false,
        ffmpeg::ChannelLayout::MONO,
        96_000,
        Coupling::Auto,
        false,
    ));
}

#[test]
fn stereo_192k() {
    check(case(
        "stereo192",
        Acmod::Stereo,
        false,
        ffmpeg::ChannelLayout::STEREO,
        192_000,
        Coupling::Auto,
        true,
    ));
}

#[test]
fn stereo_448k() {
    check(case(
        "stereo448",
        Acmod::Stereo,
        false,
        ffmpeg::ChannelLayout::STEREO,
        448_000,
        Coupling::Off,
        true,
    ));
}

#[test]
fn stereo_one_block_high_bitrate() {
    check(case(
        "stereo4m",
        Acmod::Stereo,
        false,
        ffmpeg::ChannelLayout::STEREO,
        4_000_000,
        Coupling::Auto,
        true,
    ));
}

#[test]
fn surround_51_640k() {
    check(case(
        "surround51_640",
        Acmod::ThreeTwo,
        true,
        ffmpeg::ChannelLayout::_5POINT1,
        640_000,
        Coupling::Auto,
        false,
    ));
}

#[test]
fn surround_51_448k_coupling_off() {
    check(case(
        "surround51_448_nocpl",
        Acmod::ThreeTwo,
        true,
        ffmpeg::ChannelLayout::_5POINT1,
        448_000,
        Coupling::Off,
        false,
    ));
}

#[test]
fn surround_50_no_lfe() {
    check(case(
        "surround50",
        Acmod::ThreeTwo,
        false,
        ffmpeg::ChannelLayout::_5POINT0,
        512_000,
        Coupling::Auto,
        false,
    ));
}
