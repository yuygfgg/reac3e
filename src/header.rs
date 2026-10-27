//! E-AC-3 BSI and audio-frame header.

use crate::bitwriter::BitWriter;
use crate::encoder::FrameEncoder;
use crate::tables::{AC3_CHMODE_2F2R, AC3_CHMODE_MONO, AC3_CHMODE_STEREO};

pub(crate) fn write_frame_header(enc: &FrameEncoder, w: &mut BitWriter) {
    write_sync_info(enc, w);
    write_mixing_metadata(enc, w);
    write_info_metadata(enc, w);
    write_frame_flags(enc, w);
    write_coupling_strategy(enc, w);
    write_exponent_strategies(enc, w);
    write_snr_offsets(enc, w);
}

fn write_sync_info(enc: &FrameEncoder, w: &mut BitWriter) {
    let cfg = &enc.cfg;

    w.write(16, 0x0B77); // sync word
    w.write(2, 0); // stream type: independent
    w.write(3, 0); // substream id
    w.write(11, (enc.frame_size / 2 - 1) as u32); // frame size in 16-bit words - 1
    w.write(2, 0); // sample rate code: 48 kHz
    w.write(2, cfg.num_blks_code as u32);
    w.write(3, cfg.channel_mode as u32);
    w.write_bool(cfg.lfe_on);
    w.write(5, 16); // bitstream id: E-AC-3
    w.write(5, (-(cfg.dialnorm as i32)) as u32);
    w.write_bool(false); // no compression gain word
}

fn write_mixing_metadata(enc: &FrameEncoder, w: &mut BitWriter) {
    let cfg = &enc.cfg;
    let meta = &cfg.metadata;

    w.write_bool(meta.eac3_mixing_metadata);
    if !meta.eac3_mixing_metadata {
        return;
    }

    if cfg.channel_mode > AC3_CHMODE_STEREO {
        w.write(2, meta.preferred_stereo_downmix as u32);
    }
    if cfg.has_center {
        w.write(3, meta.ltrt_center_mix_level as u32);
        w.write(3, meta.loro_center_mix_level as u32);
    }
    if cfg.has_surround {
        w.write(3, meta.ltrt_surround_mix_level as u32);
        w.write(3, meta.loro_surround_mix_level as u32);
    }
    if cfg.lfe_on {
        w.write_bool(false);
    }
    w.write_bool(false); // no program scale
    w.write_bool(false); // no extended program scale
    w.write(2, 0); // no mixing parameters
    if cfg.channel_mode < AC3_CHMODE_STEREO {
        w.write_bool(false); // no pan info
    }
    w.write_bool(false); // no frame mixing configuration
}

fn write_info_metadata(enc: &FrameEncoder, w: &mut BitWriter) {
    let cfg = &enc.cfg;
    let meta = &cfg.metadata;

    w.write_bool(meta.eac3_info_metadata);
    if !meta.eac3_info_metadata {
        return;
    }

    w.write(3, meta.bitstream_mode as u32);
    w.write_bool(meta.copyright);
    w.write_bool(meta.original);
    if cfg.channel_mode == AC3_CHMODE_STEREO {
        w.write(2, meta.dolby_surround_mode as u32);
        w.write(2, meta.dolby_headphone_mode as u32);
    }
    if cfg.channel_mode >= AC3_CHMODE_2F2R {
        w.write(2, meta.dolby_surround_ex_mode as u32);
    }
    w.write_bool(meta.audio_production_info);
    if meta.audio_production_info {
        w.write(5, (meta.mixing_level - 80) as u32);
        w.write(2, meta.room_type as u32);
        w.write_bool(meta.ad_converter_type != 0);
    }
    w.write_bool(false);
}

fn write_frame_flags(enc: &FrameEncoder, w: &mut BitWriter) {
    let cfg = &enc.cfg;

    if cfg.num_blocks != 6 {
        // Converter sync: one in every six frames.
        w.write_bool(enc.frame_index.is_multiple_of(6));
    }
    w.write_bool(false); // no additional bitstream info

    if cfg.num_blocks == 6 {
        w.write_bool(!enc.use_frame_exp_strategy);
        w.write_bool(false); // AHT disabled
    }
    w.write(2, 0); // SNR offset strategy
    w.write_bool(false); // transient pre-noise disabled
    w.write_bool(enc.blk_sw_syntax); // block switch syntax
    w.write_bool(false); // dither syntax disabled
    w.write_bool(false); // bit allocation model syntax disabled
    w.write_bool(false); // fast gain codes disabled
    w.write_bool(false); // delta bit allocation disabled
    w.write_bool(false); // skip field syntax disabled
    w.write_bool(false); // spectral extension disabled
}

fn write_coupling_strategy(enc: &FrameEncoder, w: &mut BitWriter) {
    let cfg = &enc.cfg;
    if cfg.channel_mode <= AC3_CHMODE_MONO {
        return;
    }

    w.write_bool(enc.blocks[0].cpl_in_use);
    for block in &enc.blocks[1..] {
        w.write_bool(block.new_cpl_strategy);
        if block.new_cpl_strategy {
            w.write_bool(block.cpl_in_use);
        }
    }
}

fn write_exponent_strategies(enc: &FrameEncoder, w: &mut BitWriter) {
    let cfg = &enc.cfg;

    if enc.use_frame_exp_strategy {
        let start = usize::from(!enc.cpl_on);
        for ch in start..=cfg.fbw_channels {
            w.write(5, enc.frame_exp_strategy[ch] as u32);
        }
    } else {
        for (blk, block) in enc.blocks.iter().enumerate() {
            let start = usize::from(!block.cpl_in_use);
            for ch in start..=cfg.fbw_channels {
                w.write(2, enc.exp_strategy[ch][blk].code() as u32);
            }
        }
    }

    if cfg.lfe_on {
        for blk in 0..cfg.num_blocks {
            w.write_bool(enc.exp_strategy[cfg.lfe_channel][blk].code() != 0);
        }
    }

    if cfg.num_blocks != 6 {
        w.write_bool(false);
    } else {
        for ch in 1..=cfg.fbw_channels {
            let value = if enc.use_frame_exp_strategy {
                enc.frame_exp_strategy[ch] as u32
            } else {
                0
            };
            w.write(5, value);
        }
    }
}

fn write_snr_offsets(enc: &FrameEncoder, w: &mut BitWriter) {
    w.write(6, enc.coarse_snr_offset as u32);
    w.write(4, enc.fine_snr_offset[1] as u32);
    if enc.cfg.num_blocks > 1 {
        w.write_bool(false); // no block start info
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{
        Acmod, AdConverterType, BsiMetadata, ChannelLayout, ChannelOrder, Coupling,
        DolbySurroundExMode, DownmixMode, EncoderConfig, RoomType,
    };
    use crate::encoder::FrameEncoder;
    use crate::pcm::PlanarFrame;

    struct BitReader<'a> {
        data: &'a [u8],
        position: usize,
    }

    impl<'a> BitReader<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self { data, position: 0 }
        }

        fn read(&mut self, bits: u32) -> u32 {
            let mut value = 0u32;
            for _ in 0..bits {
                let byte = self.data[self.position / 8];
                let bit = (byte >> (7 - self.position % 8)) & 1;
                value = (value << 1) | bit as u32;
                self.position += 1;
            }
            value
        }

        fn flag(&mut self) -> bool {
            self.read(1) != 0
        }
    }

    fn encode_zero_frame(cfg: EncoderConfig) -> Vec<u8> {
        let mut enc = FrameEncoder::new(cfg).unwrap();
        let silence = vec![0.0f32; enc.samples_per_frame()];
        let channels: Vec<&[f32]> = (0..enc.cfg.channels).map(|_| silence.as_slice()).collect();
        let pcm = PlanarFrame::new(&channels, ChannelOrder::Smpte);
        enc.encode_frame(&pcm).unwrap().bytes
    }

    #[test]
    fn header_starts_with_sync_word() {
        let bytes = encode_zero_frame(EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::Stereo,
                lfe: false,
            },
            bitrate_bps: 192_000,
            dialnorm: -27,
            metadata: Default::default(),
            coupling: Coupling::Auto,
            stereo_rematrixing: true,
            cutoff_hz: None,
        });
        assert_eq!(&bytes[..2], &[0x0B, 0x77]);
    }

    #[test]
    fn header_encodes_layout_and_metadata_fields() {
        let metadata = BsiMetadata {
            copyright: Some(true),
            original: Some(false),
            mixing_level: Some(90),
            room_type: Some(RoomType::Large),
            dolby_surround_ex_mode: Some(DolbySurroundExMode::Off),
            ad_converter_type: Some(AdConverterType::Hdcd),
            preferred_stereo_downmix: Some(DownmixMode::LoRo),
            ltrt_center_mix_level: Some(std::f32::consts::FRAC_1_SQRT_2),
            ltrt_surround_mix_level: Some(std::f32::consts::FRAC_1_SQRT_2),
            loro_center_mix_level: Some(std::f32::consts::FRAC_1_SQRT_2),
            loro_surround_mix_level: Some(std::f32::consts::FRAC_1_SQRT_2),
            ..BsiMetadata::default()
        };
        let bytes = encode_zero_frame(EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::ThreeTwo,
                lfe: true,
            },
            bitrate_bps: 640_000,
            dialnorm: -21,
            metadata,
            coupling: Coupling::Auto,
            stereo_rematrixing: false,
            cutoff_hz: None,
        });

        let mut bits = BitReader::new(&bytes);
        assert_eq!(bits.read(16), 0x0B77);
        assert_eq!(bits.read(2), 0); // stream type
        assert_eq!(bits.read(3), 0); // substream id
        let words = bits.read(11) as usize;
        assert_eq!((words + 1) * 2, bytes.len());
        assert_eq!(bits.read(2), 0); // 48 kHz
        assert_eq!(bits.read(2), 3); // six blocks
        assert_eq!(bits.read(3), 7); // 3/2
        assert!(bits.flag()); // LFE
        assert_eq!(bits.read(5), 16); // E-AC-3
        assert_eq!(bits.read(5), 21); // -dialnorm
        assert!(!bits.flag()); // no compression gain

        assert!(bits.flag()); // mixing metadata
        assert_eq!(bits.read(2), DownmixMode::LoRo as u32);
        assert_eq!(bits.read(3), 4); // Lt/Rt centre = 0.7071
        assert_eq!(bits.read(3), 4); // Lo/Ro centre = 0.7071
        assert_eq!(bits.read(3), 4); // Lt/Rt surround = 0.7071
        assert_eq!(bits.read(3), 4); // Lo/Ro surround = 0.7071
        assert!(!bits.flag()); // no LFE mixing level
        assert!(!bits.flag()); // no program scale
        assert!(!bits.flag()); // no extended program scale
        assert_eq!(bits.read(2), 0); // no mixing parameters
        assert!(!bits.flag()); // no frame mixing configuration

        assert!(bits.flag()); // info metadata
        assert_eq!(bits.read(3), 0); // main service
        assert!(bits.flag()); // copyright
        assert!(!bits.flag()); // original
        assert_eq!(bits.read(2), 1); // Dolby Surround EX off
        assert!(bits.flag()); // audio production info
        assert_eq!(bits.read(5), 10); // mixing level 90
        assert_eq!(bits.read(2), 1); // room type large
        assert!(bits.flag()); // HDCD converter
        assert!(!bits.flag()); // no additional info
    }

    #[test]
    fn converter_sync_bit_is_high_every_sixth_non_six_block_frame() {
        let cfg = EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::Stereo,
                lfe: false,
            },
            bitrate_bps: 1_500_000,
            dialnorm: -27,
            metadata: BsiMetadata::default(),
            coupling: Coupling::Auto,
            stereo_rematrixing: true,
            cutoff_hz: None,
        };
        let mut encoder = FrameEncoder::new(cfg).unwrap();
        assert_eq!(encoder.samples_per_frame(), 3 * 256);
        let silence = vec![0.0f32; encoder.samples_per_frame()];
        let channels: [&[f32]; 2] = [&silence, &silence];

        let mut sync_bits = Vec::new();
        for _ in 0..7 {
            let frame = encoder
                .encode_frame(&PlanarFrame::new(&channels, ChannelOrder::Smpte))
                .unwrap();
            let mut bits = BitReader::new(&frame.bytes);
            bits.read(16);
            bits.read(2);
            bits.read(3);
            bits.read(11);
            bits.read(2);
            assert_eq!(bits.read(2), 2); // three blocks
            bits.read(3);
            bits.read(1);
            bits.read(5);
            bits.read(5);
            bits.read(1);
            assert!(!bits.flag()); // no mixing metadata
            assert!(!bits.flag()); // no info metadata
            sync_bits.push(bits.flag());
        }

        assert_eq!(sync_bits, [true, false, false, false, false, false, true]);
    }

    #[test]
    fn default_metadata_omits_both_metadata_extensions() {
        let bytes = encode_zero_frame(EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::ThreeTwo,
                lfe: true,
            },
            bitrate_bps: 640_000,
            dialnorm: -27,
            metadata: BsiMetadata::default(),
            coupling: Coupling::Auto,
            stereo_rematrixing: false,
            cutoff_hz: None,
        });

        let mut bits = BitReader::new(&bytes);
        bits.read(16);
        bits.read(2);
        bits.read(3);
        bits.read(11);
        bits.read(2);
        bits.read(2);
        bits.read(3);
        bits.read(1);
        bits.read(5);
        bits.read(5);
        bits.read(1);
        assert!(!bits.flag()); // no mixing metadata
        assert!(!bits.flag()); // no info metadata
    }
}
