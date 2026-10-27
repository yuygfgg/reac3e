//! Shared harness for the integration tests.

#![allow(dead_code)]

use ffmpeg::{
    ChannelLayout, Dictionary, Error, Packet,
    codec::{self, decoder, encoder},
    format::{self, Sample},
    frame,
};
use ffmpeg_next as ffmpeg;

use reac3e::{
    Acmod, BsiMetadata, ChannelLayout as Layout, Coupling, EncoderConfig, Frame, StreamEncoder,
};

pub fn init_ffmpeg() {
    // Make the comparison reproducible.
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        ffmpeg::ffi::av_force_cpu_flags(0);
    });
}

pub fn sample_format() -> Sample {
    Sample::F32(format::sample::Type::Planar)
}

pub struct Case {
    pub name: &'static str,
    pub acmod: Acmod,
    pub lfe: bool,
    pub layout: ChannelLayout,
    pub bitrate: u32,
    pub coupling: Coupling,
    pub rematrix: bool,
    pub samples: usize,
}

impl Case {
    pub fn channels(&self) -> usize {
        self.layout.channels() as usize
    }
}

pub fn gen_pcm(case: &Case) -> Vec<f32> {
    let channels = case.channels();
    let mut out = vec![0.0f32; case.samples * channels];
    let mut state = 0x1357_9BDFu32;
    let mut rng = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    for n in 0..case.samples {
        let t = n as f32 / 48_000.0;
        for ch in 0..channels {
            let f = 180.0 * (ch as f32 + 1.0);
            let tone = (2.0 * std::f32::consts::PI * f * t).sin() * 0.12;
            let noise = rng() * 0.01;
            out[n * channels + ch] = tone + noise;
        }
    }
    out
}

pub fn encode_ours(case: &Case, pcm: &[f32], metadata: &BsiMetadata) -> Vec<Frame> {
    let cfg = EncoderConfig {
        layout: Layout {
            acmod: case.acmod,
            lfe: case.lfe,
        },
        bitrate_bps: case.bitrate,
        dialnorm: -27,
        metadata: metadata.clone(),
        coupling: case.coupling,
        stereo_rematrixing: case.rematrix,
        cutoff_hz: None,
    };
    let mut enc = StreamEncoder::new(cfg).unwrap();
    let mut frames = Vec::new();
    let chunk = case.channels() * 777;
    for chunk in pcm.chunks(chunk) {
        frames.extend(enc.push_interleaved(chunk).unwrap());
    }
    frames.extend(enc.finish().unwrap());
    frames
}

fn drain_encoder(enc: &mut encoder::Encoder) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut packet = Packet::empty();
        match enc.receive_packet(&mut packet) {
            Ok(()) => {
                if let Some(data) = packet.data() {
                    out.extend_from_slice(data);
                }
            }
            Err(Error::Eof) => break,
            Err(Error::Other { errno }) if errno == libc::EAGAIN => break,
            Err(e) => panic!("encoder receive_packet: {e}"),
        }
    }
    out
}

pub fn encode_reference(case: &Case, pcm: &[f32], options: &[(&str, &str)]) -> Vec<u8> {
    let codec = encoder::find_by_name("eac3").expect("eac3 encoder");
    let mut ctx = codec::Context::new_with_codec(codec)
        .encoder()
        .audio()
        .unwrap();
    ctx.set_rate(48_000);
    ctx.set_format(sample_format());
    ctx.set_channel_layout(case.layout);
    ctx.set_bit_rate(case.bitrate as usize);
    ctx.set_time_base((1, 48_000));

    let mut dict = Dictionary::new();
    dict.set("dialnorm", "-27");
    match case.coupling {
        Coupling::Auto => {}
        Coupling::On => dict.set("channel_coupling", "1"),
        Coupling::Off => dict.set("channel_coupling", "0"),
    }
    if !case.rematrix {
        dict.set("stereo_rematrixing", "0");
    }
    for &(key, value) in options {
        dict.set(key, value);
    }

    let mut enc = ctx.open_with(dict).unwrap();
    let frame_size = enc.frame_size() as usize;
    let channels = case.channels();
    let mut out = Vec::new();

    let samples_per_frame = frame_size * channels;
    for chunk in pcm.chunks(samples_per_frame) {
        let n = chunk.len() / channels;
        let mut audio = frame::Audio::new(sample_format(), n, case.layout);
        audio.set_rate(48_000);
        for ch in 0..channels {
            let plane = audio.plane_mut::<f32>(ch);
            for i in 0..n {
                plane[i] = chunk[i * channels + ch];
            }
        }
        enc.send_frame(&audio).unwrap();
        out.extend_from_slice(&drain_encoder(&mut enc));
    }
    enc.send_eof().unwrap();
    out.extend_from_slice(&drain_encoder(&mut enc));
    out
}

pub fn split_frames(data: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut offset = 0usize;
    while offset + 6 <= data.len() {
        assert_eq!(&data[offset..offset + 2], &[0x0B, 0x77], "sync word");
        let words = (((data[offset + 2] & 0x07) as usize) << 8) | data[offset + 3] as usize;
        let size = (words + 1) * 2;
        assert!(offset + size <= data.len(), "truncated syncframe");
        frames.push(&data[offset..offset + size]);
        offset += size;
    }
    assert_eq!(offset, data.len(), "trailing bytes after last syncframe");
    frames
}

fn drain_decoder(dec: &mut decoder::Audio, channels: usize) -> Vec<f32> {
    let mut out = Vec::new();
    loop {
        let mut audio = frame::Audio::empty();
        match dec.receive_frame(&mut audio) {
            Ok(()) => {
                assert_eq!(audio.format(), sample_format(), "decoder output format");
                let n = audio.samples();
                assert_eq!(audio.channels() as usize, channels, "decoder channels");
                for i in 0..n {
                    for ch in 0..channels {
                        out.push(audio.plane::<f32>(ch)[i]);
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(Error::Other { errno }) if errno == libc::EAGAIN => break,
            Err(e) => panic!("decoder receive_frame: {e}"),
        }
    }
    out
}

pub fn decode(data: &[u8], channels: usize) -> Vec<f32> {
    let codec = decoder::find_by_name("eac3").expect("eac3 decoder");
    let mut dec = codec::Context::new_with_codec(codec)
        .decoder()
        .open()
        .unwrap()
        .audio()
        .unwrap();
    dec.request_format(sample_format());

    let mut out = Vec::new();
    for frame_data in split_frames(data) {
        let packet = Packet::copy(frame_data);
        dec.send_packet(&packet).unwrap();
        out.extend_from_slice(&drain_decoder(&mut dec, channels));
    }
    dec.send_eof().unwrap();
    out.extend_from_slice(&drain_decoder(&mut dec, channels));
    out
}

pub fn stream_bytes(frames: Vec<Frame>) -> Vec<u8> {
    frames.into_iter().flat_map(|frame| frame.bytes).collect()
}
