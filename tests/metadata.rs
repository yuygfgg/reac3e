//! Skip-field/addbsi carriage tests against an independent decoder.
//!
//! The encoder writes Atmos metadata natively.  A decoder must ignore the
//! metadata bits and reproduce the plain stream's PCM exactly, for every
//! syntax reac3e supports (block switching, coupling, all block counts).

mod common;

use common::{decode, init_ffmpeg, split_frames};
use reac3e::{
    Acmod, BsiMetadata, ChannelLayout, ChannelOrder, Coupling, EncoderConfig, Frame, FrameEncoder,
    PlanarFrame, SkipMetadata,
};

fn config(bitrate_bps: u32, coupling: Coupling) -> EncoderConfig {
    EncoderConfig {
        layout: ChannelLayout {
            acmod: Acmod::ThreeTwo,
            lfe: true,
        },
        bitrate_bps,
        dialnorm: -27,
        metadata: BsiMetadata::default(),
        coupling,
        stereo_rematrixing: true,
        cutoff_hz: None,
        coupling_start: None,
    }
}

fn planar(samples_per_frame: usize, frames: usize, transient: bool) -> Vec<Vec<f32>> {
    let mut state = 0x1357_9BDFu32;
    (0..6)
        .map(|ch| {
            (0..samples_per_frame * frames)
                .map(|n| {
                    let t = n as f32 / 48_000.0;
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let noise = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
                    let mut value = (0.12
                        * (2.0 * std::f32::consts::PI * 220.0 * (ch as f32 + 1.0) * t).sin())
                        + noise * 0.01;
                    if transient && n % samples_per_frame < 128 {
                        value += 0.7 * (-((n % samples_per_frame) as f32) / 32.0).exp();
                    }
                    value.clamp(-1.0, 1.0)
                })
                .collect()
        })
        .collect()
}

fn encode(cfg: EncoderConfig, samples: &[Vec<f32>], metadata: Option<&SkipMetadata>) -> Vec<Frame> {
    let mut encoder = FrameEncoder::new(cfg).unwrap();
    let spf = encoder.samples_per_frame();
    let total = samples[0].len();
    let mut frames = Vec::new();
    let mut offset = 0;
    while offset < total {
        let length = (total - offset).min(spf);
        let channels: Vec<&[f32]> = samples
            .iter()
            .map(|channel| &channel[offset..offset + length])
            .collect();
        frames.push(
            encoder
                .encode_frame_with_metadata(
                    &PlanarFrame::new(&channels, ChannelOrder::Smpte),
                    metadata.cloned(),
                )
                .unwrap(),
        );
        offset += length;
    }
    frames
}

fn bytes(frames: &[Frame]) -> Vec<u8> {
    frames
        .iter()
        .flat_map(|frame| frame.bytes.clone())
        .collect()
}

fn check(cfg: EncoderConfig, transient: bool, metadata: SkipMetadata) {
    let reference_encoder = FrameEncoder::new(cfg.clone()).unwrap();
    let spf = reference_encoder.samples_per_frame();
    let samples = planar(spf, 6, transient);

    let plain = encode(cfg.clone(), &samples, None);
    let tagged = encode(cfg, &samples, Some(&metadata));
    assert_eq!(plain.len(), tagged.len());

    // Sizes differ only by the metadata overhead, and every declared size is
    // consistent with the byte stream.
    for frame in &tagged {
        let declared = frame.edits.frame_size;
        assert_eq!(frame.bytes.len(), declared);
        let words = (((frame.bytes[2] & 0x07) as usize) << 8) | frame.bytes[3] as usize;
        assert_eq!((words + 1) * 2, declared);
    }
    let plain_bytes = bytes(&plain);
    let tagged_bytes = bytes(&tagged);
    let plain_frames = split_frames(&plain_bytes);
    let tagged_frames = split_frames(&tagged_bytes);
    assert_eq!(plain_frames.len(), tagged_frames.len());

    init_ffmpeg();
    let plain_pcm = decode(&plain_bytes, 6);
    let tagged_pcm = decode(&tagged_bytes, 6);
    assert_eq!(
        plain_pcm.len(),
        tagged_pcm.len(),
        "metadata changed the decoded sample count"
    );
    assert_eq!(
        plain_pcm, tagged_pcm,
        "metadata changed the decoded PCM; skip-field syntax is misplaced"
    );
}

#[test]
fn skip_field_metadata_is_transparent_to_the_decoder() {
    let meta = SkipMetadata::new(
        &[0x01, 0x10],
        &[0x58, 0x38, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05],
        2,
    );
    check(config(693_000, Coupling::Off), false, meta);
}

#[test]
fn metadata_is_transparent_with_coupling() {
    let meta = SkipMetadata::new(&[0x01, 0x10], &[0x58, 0x38, 0xAA], 1);
    check(config(256_000, Coupling::Auto), false, meta);
}

#[test]
fn metadata_is_transparent_with_block_switching() {
    let meta = SkipMetadata::new(&[0x01, 0x10], &[0x58, 0x38, 0xAA], 0);
    check(config(256_000, Coupling::Off), true, meta);
}

#[test]
fn metadata_is_transparent_for_every_block_count() {
    for bitrate in [693_000u32, 1_500_000, 3_000_000, 5_000_000] {
        let encoder = FrameEncoder::new(config(bitrate, Coupling::Off)).unwrap();
        let target = encoder.samples_per_frame() / 256 - 1;
        let meta = SkipMetadata::new(&[0x01, 0x10], &[0x58, 0x38, 0xAA], target);
        check(config(bitrate, Coupling::Off), false, meta);
    }
}

fn check_post_injection(cfg: EncoderConfig, transient: bool, metadata: SkipMetadata) {
    let encoder = FrameEncoder::new(cfg.clone()).unwrap();
    let spf = encoder.samples_per_frame();
    let samples = planar(spf, 6, transient);
    let plain = encode(cfg, &samples, None);
    let injected: Vec<Frame> = plain
        .iter()
        .map(|frame| reac3e::inject_metadata(frame, &metadata).unwrap())
        .collect();

    for frame in &injected {
        assert_eq!(frame.bytes.len(), frame.edits.frame_size);
        let words = (((frame.bytes[2] & 0x07) as usize) << 8) | frame.bytes[3] as usize;
        assert_eq!((words + 1) * 2, frame.edits.frame_size);
        assert_eq!(frame.edits.addbsi_len, metadata.addbsi.len());
        if !metadata.emdf.is_empty() {
            let start = frame.edits.emdf_start_bit.unwrap();
            // The 9-bit length field precedes the EMDF bytes.
            let length_pos = start - 9;
            let mut bits = Vec::new();
            for &byte in &frame.bytes {
                for shift in (0..8).rev() {
                    bits.push((byte >> shift) & 1);
                }
            }
            let length = bits[length_pos..length_pos + 9]
                .iter()
                .fold(0usize, |value, &bit| (value << 1) | usize::from(bit));
            assert_eq!(length, metadata.emdf.len());
            for (index, &byte) in metadata.emdf.iter().enumerate() {
                let mut value = 0u8;
                for offset in 0..8 {
                    value = (value << 1) | bits[start + index * 8 + offset];
                }
                assert_eq!(value, byte);
            }
        }
    }

    let plain_bytes = bytes(&plain);
    let injected_bytes = bytes(&injected);
    init_ffmpeg();
    let plain_pcm = decode(&plain_bytes, 6);
    let injected_pcm = decode(&injected_bytes, 6);
    assert_eq!(plain_pcm.len(), injected_pcm.len());
    assert_eq!(
        plain_pcm, injected_pcm,
        "post-encode injection changed the decoded PCM"
    );
}

#[test]
fn post_encode_injection_is_transparent() {
    check_post_injection(
        config(693_000, Coupling::Off),
        false,
        SkipMetadata::new(&[0x01, 0x10], &[0x58, 0x38, 0xAA], 2),
    );
    check_post_injection(
        config(256_000, Coupling::Auto),
        false,
        SkipMetadata::new(&[0x01, 0x10], &[0x58, 0x38, 0xAA], 1),
    );
    check_post_injection(
        config(256_000, Coupling::Off),
        true,
        SkipMetadata::new(&[0x01, 0x10], &[0x58, 0x38, 0xAA], 0),
    );
    for bitrate in [1_500_000u32, 3_000_000, 5_000_000] {
        let target = FrameEncoder::new(config(bitrate, Coupling::Off))
            .unwrap()
            .samples_per_frame()
            / 256
            - 1;
        check_post_injection(
            config(bitrate, Coupling::Off),
            false,
            SkipMetadata::new(&[0x01, 0x10], &[0x58, 0x38, 0xAA], target),
        );
    }
}

#[test]
fn post_encode_injection_accepts_addbsi_or_emdf_alone() {
    check_post_injection(
        config(693_000, Coupling::Off),
        false,
        SkipMetadata::new(&[0x01, 0x10], &[], 0),
    );
    check_post_injection(
        config(693_000, Coupling::Off),
        false,
        SkipMetadata::new(&[], &[0x58, 0x38, 0xAA], 2),
    );
}
