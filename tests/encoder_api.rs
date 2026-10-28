//! Public API, configuration and framing tests.

use reac3e::{
    Acmod, AdConverterType, BsiMetadata, ChannelLayout, ChannelOrder, Coupling, EncodeError,
    EncoderConfig, Frame, FrameEncoder, PlanarFrame, RoomType, ServiceType, StreamEncoder,
};

fn config(
    acmod: Acmod,
    lfe: bool,
    bitrate_bps: u32,
    coupling: Coupling,
    stereo_rematrixing: bool,
) -> EncoderConfig {
    EncoderConfig {
        layout: ChannelLayout { acmod, lfe },
        bitrate_bps,
        dialnorm: -27,
        metadata: BsiMetadata::default(),
        coupling,
        stereo_rematrixing,
        cutoff_hz: None,
        coupling_start: None,
    }
}

fn stereo_config() -> EncoderConfig {
    config(Acmod::Stereo, false, 192_000, Coupling::Auto, true)
}

fn surround_config() -> EncoderConfig {
    config(Acmod::ThreeTwo, true, 640_000, Coupling::Auto, false)
}

fn gen_planar(samples: usize, channels: usize) -> Vec<Vec<f32>> {
    let mut state = 0x2468_ACE1u32;
    let mut out = vec![vec![0.0f32; samples]; channels];
    for n in 0..samples {
        let time = n as f32 / 48_000.0;
        for (channel, output) in out.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            let tone = (2.0 * std::f32::consts::PI * 220.0 * (channel as f32 + 1.0) * time).sin();
            output[n] = tone * 0.1 + noise * 0.005;
        }
    }
    out
}

fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    let samples = channels[0].len();
    let mut out = Vec::with_capacity(samples * channels.len());
    for n in 0..samples {
        for channel in channels {
            out.push(channel[n]);
        }
    }
    out
}

fn encode_frames(cfg: &EncoderConfig, planar: &[Vec<f32>], order: ChannelOrder) -> Vec<Frame> {
    let mut encoder = FrameEncoder::new(cfg.clone()).unwrap();
    let samples_per_frame = encoder.samples_per_frame();
    let samples = planar[0].len();
    let mut frames = Vec::new();
    let mut offset = 0usize;
    while offset < samples {
        let length = (samples - offset).min(samples_per_frame);
        let channels: Vec<&[f32]> = planar
            .iter()
            .map(|channel| &channel[offset..offset + length])
            .collect();
        frames.push(
            encoder
                .encode_frame(&PlanarFrame::new(&channels, order))
                .unwrap(),
        );
        offset += length;
    }
    frames
}

fn encode_stream(cfg: &EncoderConfig, planar: &[Vec<f32>], chunk_samples: usize) -> Vec<Frame> {
    let channels = planar.len();
    let interleaved = interleave(planar);
    let mut encoder = StreamEncoder::new(cfg.clone()).unwrap();
    let mut frames = Vec::new();
    for chunk in interleaved.chunks(chunk_samples * channels) {
        frames.extend(encoder.push_interleaved(chunk).unwrap());
    }
    frames.extend(encoder.finish().unwrap());
    frames
}

fn frame_bytes(frames: &[Frame]) -> Vec<u8> {
    frames
        .iter()
        .flat_map(|frame| frame.bytes.clone())
        .collect()
}

fn expect_error(cfg: EncoderConfig) -> EncodeError {
    match FrameEncoder::new(cfg) {
        Ok(_) => panic!("expected configuration to be rejected"),
        Err(error) => error,
    }
}

#[test]
fn stream_and_frame_encoders_agree() {
    let cfg = stereo_config();
    let samples_per_frame = FrameEncoder::new(cfg.clone()).unwrap().samples_per_frame();
    let planar = gen_planar(40 * samples_per_frame + 211, 2);

    let from_frames = encode_frames(&cfg, &planar, ChannelOrder::Smpte);
    let from_stream = encode_stream(&cfg, &planar, 777);

    assert_eq!(
        frame_bytes(&from_frames),
        frame_bytes(&from_stream),
        "frame and stream paths must produce identical bytes"
    );
    assert_eq!(from_frames.len(), from_stream.len());
    assert_eq!(
        from_frames
            .iter()
            .map(|frame| frame.index)
            .collect::<Vec<_>>(),
        (0..from_frames.len() as u64).collect::<Vec<_>>()
    );
}

#[test]
fn chunk_size_does_not_change_stream_output() {
    let cfg = surround_config();
    let planar = gen_planar(17 * 1536 + 17, 6);
    let reference = frame_bytes(&encode_stream(&cfg, &planar, 777));

    for chunk_samples in [1usize, 13, 256, 1536, 4096] {
        let frames = encode_stream(&cfg, &planar, chunk_samples);
        assert_eq!(
            reference,
            frame_bytes(&frames),
            "chunk size {chunk_samples} changed the output"
        );
    }
}

#[test]
fn smpte_and_ac3_channel_orders_agree() {
    let cfg = surround_config();
    let samples_per_frame = FrameEncoder::new(cfg.clone()).unwrap().samples_per_frame();
    let smpte = gen_planar(2 * samples_per_frame + 99, 6);

    // SMPTE: L R C LFE Ls Rs -> AC-3: L C R Ls Rs LFE.
    let ac3_order = [0usize, 2, 1, 4, 5, 3];
    let ac3: Vec<Vec<f32>> = ac3_order.iter().map(|&ch| smpte[ch].clone()).collect();

    let from_smpte = encode_frames(&cfg, &smpte, ChannelOrder::Smpte);
    let from_ac3 = encode_frames(&cfg, &ac3, ChannelOrder::Ac3);

    assert_eq!(frame_bytes(&from_smpte), frame_bytes(&from_ac3));
}

#[test]
fn frame_indices_are_sequential() {
    let cfg = stereo_config();
    let planar = gen_planar(5 * 1536, 2);
    let frames = encode_frames(&cfg, &planar, ChannelOrder::Smpte);

    assert_eq!(frames.len(), 5);
    for (expected, frame) in frames.iter().enumerate() {
        assert_eq!(frame.index, expected as u64);
    }
}

#[test]
fn reported_sizes_match_block_counts() {
    let cases = [
        (4_000_000, 1usize),
        (2_500_000, 2),
        (1_500_000, 3),
        (192_000, 6),
    ];
    for (bitrate, blocks) in cases {
        let cfg = config(Acmod::Stereo, false, bitrate, Coupling::Auto, true);
        let encoder = FrameEncoder::new(cfg).unwrap();
        assert_eq!(encoder.samples_per_frame(), blocks * 256);
        assert_eq!(encoder.latency(), 256);
    }
}

#[test]
fn frame_has_sync_word_and_declared_size() {
    for cfg in [
        stereo_config(),
        surround_config(),
        config(Acmod::Mono, false, 96_000, Coupling::Auto, false),
    ] {
        let channels = cfg.layout.channels();
        let planar = gen_planar(3 * 1536, channels);
        let frames = encode_frames(&cfg, &planar, ChannelOrder::Smpte);
        assert!(!frames.is_empty());

        for frame in frames {
            assert_eq!(&frame.bytes[..2], &[0x0B, 0x77]);
            let words = (((frame.bytes[2] & 0x07) as usize) << 8) | frame.bytes[3] as usize;
            assert_eq!((words + 1) * 2, frame.bytes.len());
        }
    }
}

#[test]
fn cbr_frame_sizes_alternate_by_two_bytes() {
    // A rate that is not an exact multiple of the 6-block frame granularity
    // forces the CBR controller to alternate between the two legal sizes.
    let cfg = config(Acmod::Stereo, false, 192_123, Coupling::Auto, true);
    let planar = gen_planar(30 * 1536, 2);
    let sizes: Vec<usize> = encode_frames(&cfg, &planar, ChannelOrder::Smpte)
        .iter()
        .map(|frame| frame.bytes.len())
        .collect();

    assert!(sizes.windows(2).all(|pair| pair[0].abs_diff(pair[1]) <= 2));
    let minimum = *sizes.iter().min().unwrap();
    let maximum = *sizes.iter().max().unwrap();
    assert_eq!(maximum - minimum, 2);
}

#[test]
fn short_final_frame_is_zero_padded_and_rejects_more_input() {
    let cfg = stereo_config();
    let mut encoder = FrameEncoder::new(cfg).unwrap();
    let samples_per_frame = encoder.samples_per_frame();
    let planar = gen_planar(samples_per_frame - 1, 2);
    let channels: Vec<&[f32]> = planar.iter().map(|channel| channel.as_slice()).collect();
    let frame = encoder
        .encode_frame(&PlanarFrame::new(&channels, ChannelOrder::Smpte))
        .unwrap();
    assert_eq!(&frame.bytes[..2], &[0x0B, 0x77]);
    assert!(encoder.flush().unwrap().is_none());

    let full = gen_planar(samples_per_frame, 2);
    let channels: Vec<&[f32]> = full.iter().map(|channel| channel.as_slice()).collect();
    let error = encoder
        .encode_frame(&PlanarFrame::new(&channels, ChannelOrder::Smpte))
        .unwrap_err();
    assert!(matches!(error, EncodeError::Pcm(_)));
}

#[test]
fn stream_finish_is_empty_without_audio_and_idempotent() {
    let mut encoder = StreamEncoder::new(stereo_config()).unwrap();
    assert!(encoder.finish().unwrap().is_empty());

    let planar = gen_planar(100, 2);
    let interleaved = interleave(&planar);
    assert!(encoder.push_interleaved(&interleaved).unwrap().is_empty());
    assert_eq!(encoder.finish().unwrap().len(), 1);
    assert!(encoder.finish().unwrap().is_empty());
}

#[test]
fn interleaved_push_rejects_partial_sample_frames() {
    let mut encoder = StreamEncoder::new(stereo_config()).unwrap();
    let error = encoder.push_interleaved(&[0.0; 3]).unwrap_err();
    assert!(matches!(error, EncodeError::Pcm(_)));
}

#[test]
fn frame_encoder_validates_channel_count_and_lengths() {
    let cfg = stereo_config();
    let mut encoder = FrameEncoder::new(cfg).unwrap();
    let samples_per_frame = encoder.samples_per_frame();

    let mono = vec![0.0f32; samples_per_frame];
    let mono_channels: [&[f32]; 1] = [&mono];
    let error = encoder
        .encode_frame(&PlanarFrame::new(&mono_channels, ChannelOrder::Smpte))
        .unwrap_err();
    assert!(matches!(error, EncodeError::Pcm(_)));

    let left = vec![0.0f32; samples_per_frame];
    let right = vec![0.0f32; samples_per_frame - 1];
    let channels: [&[f32]; 2] = [&left, &right];
    let error = encoder
        .encode_frame(&PlanarFrame::new(&channels, ChannelOrder::Smpte))
        .unwrap_err();
    assert!(matches!(error, EncodeError::Pcm(_)));

    let left = vec![0.0f32; samples_per_frame + 1];
    let channels: [&[f32]; 2] = [&left, &left];
    let error = encoder
        .encode_frame(&PlanarFrame::new(&channels, ChannelOrder::Smpte))
        .unwrap_err();
    assert!(matches!(error, EncodeError::Pcm(_)));
}

#[test]
fn invalid_configurations_are_rejected() {
    let mut cfg = stereo_config();
    cfg.dialnorm = 0;
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    cfg = stereo_config();
    cfg.dialnorm = -32;
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    cfg = stereo_config();
    cfg.bitrate_bps = 0;
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    cfg = stereo_config();
    cfg.bitrate_bps = 1_000_000_000;
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    let mut cfg = config(Acmod::Mono, false, 96_000, Coupling::Auto, false);
    cfg.metadata.service_type = ServiceType::Karaoke;
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    let mut cfg = surround_config();
    cfg.metadata.service_type = ServiceType::Commentary;
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    let mut cfg = surround_config();
    cfg.metadata.room_type = Some(RoomType::Large);
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    let mut cfg = surround_config();
    cfg.metadata.mixing_level = Some(79);
    cfg.metadata.room_type = Some(RoomType::Large);
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));

    let mut cfg = surround_config();
    cfg.metadata.ltrt_center_mix_level = Some(0.123);
    assert!(matches!(expect_error(cfg), EncodeError::Config(_)));
}

#[test]
fn metadata_defaults_are_accepted() {
    let mut cfg = surround_config();
    cfg.metadata = BsiMetadata {
        copyright: Some(true),
        original: Some(false),
        mixing_level: Some(90),
        room_type: Some(RoomType::Large),
        ad_converter_type: Some(AdConverterType::Hdcd),
        ..BsiMetadata::default()
    };
    assert!(FrameEncoder::new(cfg).is_ok());
}

#[test]
fn stream_finish_does_not_emit_an_extra_frame_at_a_boundary() {
    let cfg = stereo_config();
    let planar = gen_planar(2 * 1536, 2);
    let frames = encode_stream(&cfg, &planar, 777);
    assert_eq!(frames.len(), 2);
}

#[test]
fn long_cbr_stream_hits_the_target_bitrate() {
    const FRAMES: usize = 300;
    let target = 192_123u32;
    let cfg = config(Acmod::Stereo, false, target, Coupling::Auto, true);
    let planar = gen_planar(FRAMES * 1536, 2);
    let frames = encode_frames(&cfg, &planar, ChannelOrder::Smpte);
    assert_eq!(frames.len(), FRAMES);

    let sizes: Vec<usize> = frames.iter().map(|frame| frame.bytes.len()).collect();
    let minimum = *sizes.iter().min().unwrap();
    let maximum = *sizes.iter().max().unwrap();
    assert_eq!(
        maximum - minimum,
        2,
        "only the two legal CBR sizes are used"
    );
    assert!(sizes.contains(&minimum));
    assert!(sizes.contains(&maximum));

    let total_bytes: usize = sizes.iter().sum();
    let duration = (FRAMES * 1536) as f64 / 48_000.0;
    let actual_bitrate = total_bytes as f64 * 8.0 / duration;
    assert!(
        (actual_bitrate - target as f64).abs() < 10.0,
        "actual {actual_bitrate:.3} bps vs target {target}"
    );
}

#[test]
fn exact_bitrate_uses_a_single_frame_size() {
    let cfg = stereo_config();
    let planar = gen_planar(120 * 1536, 2);
    let sizes: Vec<usize> = encode_frames(&cfg, &planar, ChannelOrder::Smpte)
        .iter()
        .map(|frame| frame.bytes.len())
        .collect();
    let minimum = *sizes.iter().min().unwrap();
    let maximum = *sizes.iter().max().unwrap();
    assert_eq!(minimum, maximum);
}
