//! Encoder configuration.
//!
//! Unsupported layouts and metadata combinations are rejected during validation.

use crate::error::EncodeError;
use crate::tables::{
    AC3_BLOCK_SIZE, AC3_CHMODE_2F2R, AC3_CHMODE_STEREO, BANDWIDTH_TAB, BITRATE_TAB, CHANNELS_TAB,
    COUPLING_START_TAB, DEFAULT_CPL_BAND_STRUCT, GAIN_LEVELS,
};

/// The only sample rate supported by the encoder.
pub const SAMPLE_RATE: u32 = 48_000;

/// Audio coding mode (`acmod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acmod {
    /// 1+1 (dual mono)
    DualMono,
    /// 1/0 (centre)
    Mono,
    /// 2/0 (left, right)
    Stereo,
    /// 3/0 (left, centre, right)
    Surround,
    /// 2/1 (left, right, surround)
    TwoOne,
    /// 3/1 (left, centre, right, surround)
    ThreeOne,
    /// 2/2 (left, right, left surround, right surround)
    TwoTwo,
    /// 3/2 (left, centre, right, left surround, right surround)
    ThreeTwo,
}

impl Acmod {
    /// The `acmod` value written in the bitstream.
    pub const fn channel_mode(self) -> u8 {
        match self {
            Self::DualMono => 0,
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround => 3,
            Self::TwoOne => 4,
            Self::ThreeOne => 5,
            Self::TwoTwo => 6,
            Self::ThreeTwo => 7,
        }
    }

    /// Number of full-bandwidth (non-LFE) channels.
    pub const fn full_bandwidth_channels(self) -> usize {
        CHANNELS_TAB[self.channel_mode() as usize] as usize
    }
}

/// Channel layout: audio coding mode plus an optional LFE channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelLayout {
    pub acmod: Acmod,
    pub lfe: bool,
}

impl ChannelLayout {
    /// Total number of channels.
    pub const fn channels(self) -> usize {
        self.acmod.full_bandwidth_channels() + self.lfe as usize
    }
}

/// Order of the planar channels supplied by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    /// SMPTE/WAVE order (`L R C LFE Ls Rs ...`).  LFE is placed after the
    /// front channels, matching the native layout order.
    Smpte,
    /// AC-3 order (`L C R Ls Rs LFE`, with absent channels omitted).
    Ac3,
}

/// Channel coupling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coupling {
    /// Select coupling automatically from the bitrate table; disable it when
    /// the table says it is not beneficial.
    Auto,
    /// Enable coupling even when the automatic table disables it.
    On,
    /// Disable coupling.
    Off,
}

/// Audio service type (`bsmod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ServiceType {
    Main = 0,
    Effects = 1,
    VisuallyImpaired = 2,
    HearingImpaired = 3,
    Commentary = 4,
    Emergency = 5,
    VoiceOver = 6,
    Karaoke = 7,
}

/// Room type for audio production information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoomType {
    NotIndicated = 0,
    Large = 1,
    Small = 2,
}

/// Preferred stereo downmix mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DownmixMode {
    LtRt = 1,
    LoRo = 2,
    Dplii = 3,
}

/// Dolby Surround mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DolbySurroundMode {
    NotIndicated = 0,
    Off = 1,
    On = 2,
}

/// Dolby Surround EX mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DolbySurroundExMode {
    NotIndicated = 0,
    Off = 1,
    On = 2,
    DpliiZ = 3,
}

/// Dolby Headphone mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DolbyHeadphoneMode {
    NotIndicated = 0,
    Off = 1,
    On = 2,
}

/// A/D converter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdConverterType {
    Standard = 0,
    Hdcd = 1,
}

/// BSI metadata supported by the E-AC-3 encoder.
///
/// `None` means "not indicated by the caller"; the encoder then applies the
/// standard defaults (including when the corresponding metadata is omitted).
#[derive(Debug, Clone)]
pub struct BsiMetadata {
    pub service_type: ServiceType,
    pub copyright: Option<bool>,
    pub original: Option<bool>,
    pub mixing_level: Option<u8>,
    pub room_type: Option<RoomType>,
    pub dolby_surround_mode: Option<DolbySurroundMode>,
    pub dolby_headphone_mode: Option<DolbyHeadphoneMode>,
    pub dolby_surround_ex_mode: Option<DolbySurroundExMode>,
    pub ad_converter_type: Option<AdConverterType>,
    pub preferred_stereo_downmix: Option<DownmixMode>,
    pub ltrt_center_mix_level: Option<f32>,
    pub ltrt_surround_mix_level: Option<f32>,
    pub loro_center_mix_level: Option<f32>,
    pub loro_surround_mix_level: Option<f32>,
}

impl Default for BsiMetadata {
    fn default() -> Self {
        Self {
            service_type: ServiceType::Main,
            copyright: None,
            original: None,
            mixing_level: None,
            room_type: None,
            dolby_surround_mode: None,
            dolby_headphone_mode: None,
            dolby_surround_ex_mode: None,
            ad_converter_type: None,
            preferred_stereo_downmix: None,
            ltrt_center_mix_level: None,
            ltrt_surround_mix_level: None,
            loro_center_mix_level: None,
            loro_surround_mix_level: None,
        }
    }
}

/// Full encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub layout: ChannelLayout,
    /// Target bitrate in bits per second.  Any rate accepted by E-AC-3 for the
    /// selected block count is permitted.
    pub bitrate_bps: u32,
    /// Dialogue normalisation level, -31..=-1 dB.
    pub dialnorm: i8,
    pub metadata: BsiMetadata,
    pub coupling: Coupling,
    pub stereo_rematrixing: bool,
}

/// Metadata after validation and defaulting, ready for the bitstream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValidatedMetadata {
    pub eac3_mixing_metadata: bool,
    pub eac3_info_metadata: bool,
    pub preferred_stereo_downmix: u8,
    pub ltrt_center_mix_level: u8,
    pub ltrt_surround_mix_level: u8,
    pub loro_center_mix_level: u8,
    pub loro_surround_mix_level: u8,
    pub bitstream_mode: u8,
    pub copyright: bool,
    pub original: bool,
    pub dolby_surround_mode: u8,
    pub dolby_headphone_mode: u8,
    pub dolby_surround_ex_mode: u8,
    pub audio_production_info: bool,
    pub mixing_level: u8,
    pub room_type: u8,
    pub ad_converter_type: u8,
}

/// Coupling band layout derived from the bitrate and channel mode.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CouplingLayout {
    pub enabled: bool,
    pub num_subbands: usize,
    pub num_bands: usize,
    pub band_sizes: [usize; 18],
    pub start_freq: usize,
    pub end_freq: usize,
}

impl CouplingLayout {
    const DISABLED: Self = Self {
        enabled: false,
        num_subbands: 0,
        num_bands: 0,
        band_sizes: [0; 18],
        start_freq: 0,
        end_freq: 0,
    };
}

/// Configuration after table lookup and validation.
#[derive(Debug, Clone)]
pub(crate) struct ValidatedConfig {
    pub bit_rate: u32,
    pub num_blks_code: u8,
    pub num_blocks: usize,
    pub frame_size_min: usize,
    pub fbw_channels: usize,
    pub channels: usize,
    pub lfe_on: bool,
    pub lfe_channel: usize,
    pub channel_mode: u8,
    pub has_center: bool,
    pub has_surround: bool,
    /// Maps internal AC-3 channel indices (0..channels) to source indices in
    /// the caller's SMPTE order.  For `ChannelOrder::Ac3` this is the
    /// identity; see `pcm.rs`.
    pub channel_map: [usize; 8],
    pub bandwidth_code: usize,
    pub start_freq: [usize; 8],
    pub cpl_enabled: bool,
    pub num_cpl_subbands: usize,
    pub num_cpl_bands: usize,
    pub cpl_band_sizes: [usize; 18],
    pub cpl_start_freq: usize,
    pub cpl_end_freq: usize,
    pub rematrixing_enabled: bool,
    pub dialnorm: i8,
    pub metadata: ValidatedMetadata,
}

/// Table mapping SMPTE/native order to AC-3 order: `[channel_mode][lfe][ch]`.
const AC3_ENC_CHANNEL_MAP: [[[usize; 6]; 2]; 8] = [
    [[0, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
    [[0, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]],
    [[0, 2, 1, 0, 0, 0], [0, 2, 1, 3, 0, 0]],
    [[0, 1, 2, 0, 0, 0], [0, 1, 3, 2, 0, 0]],
    [[0, 2, 1, 3, 0, 0], [0, 2, 1, 4, 3, 0]],
    [[0, 1, 2, 3, 0, 0], [0, 1, 3, 4, 2, 0]],
    [[0, 2, 1, 3, 4, 0], [0, 2, 1, 4, 5, 3]],
];

/// Frame-size codes in increasing block-count order.
const BLOCK_COUNTS: [usize; 4] = [1, 2, 3, 6];

/// The bitrate range permitted for a frame with `num_blocks` blocks.
fn bitrate_limits(num_blocks: usize) -> (i64, i64) {
    let frame_samples = (AC3_BLOCK_SIZE * num_blocks) as i64;
    let sample_rate = SAMPLE_RATE as i64;
    let max = 2048 * sample_rate / frame_samples * 16;
    let min = (sample_rate + frame_samples - 1) / frame_samples * 16;
    (min, max)
}

/// Pick the smallest block count whose maximum bitrate covers `bitrate`.
fn select_block_code(bitrate: i64) -> usize {
    (0..BLOCK_COUNTS.len())
        .rev()
        .find(|&code| bitrate <= bitrate_limits(BLOCK_COUNTS[code]).1)
        .unwrap_or(0)
}

/// Closest AC-3 bitrate-code index, used by the bandwidth/coupling tables.
fn closest_bitrate_code(bitrate: i64) -> usize {
    BITRATE_TAB
        .iter()
        .enumerate()
        .min_by_key(|&(_, &rate)| (rate as i64 * 1000 - bitrate).abs())
        .map_or(0, |(index, _)| index)
}

fn validate_float_option(value: f32, options: &[f32]) -> Option<usize> {
    const FLT_OPTION_THRESHOLD: f32 = 0.01;
    options
        .iter()
        .position(|&x| value < x + FLT_OPTION_THRESHOLD && value > x - FLT_OPTION_THRESHOLD)
}

fn validate_mix_level(
    name: &str,
    value: Option<f32>,
    default_value: usize,
    min_value: usize,
) -> Result<u8, EncodeError> {
    let Some(value) = value else {
        return Ok(default_value as u8);
    };
    let level = validate_float_option(value, &GAIN_LEVELS)
        .ok_or_else(|| EncodeError::Config(format!("invalid {name}: {value}")))?;
    Ok(if level < min_value {
        default_value as u8
    } else {
        level as u8
    })
}

fn coupling_layout(
    coupling: Coupling,
    channel_mode: u8,
    bandwidth_code: usize,
    bitrate_code: usize,
) -> CouplingLayout {
    if channel_mode < AC3_CHMODE_STEREO {
        return CouplingLayout::DISABLED;
    }

    let table_start = COUPLING_START_TAB[(channel_mode - 2) as usize][0][bitrate_code];
    let mut start = match (coupling, table_start) {
        (Coupling::Off, _) => return CouplingLayout::DISABLED,
        (Coupling::Auto, start) if start < 0 => return CouplingLayout::DISABLED,
        (Coupling::Auto, start) => start as i32,
        (Coupling::On, start) if start < 0 => 15,
        (Coupling::On, start) => start as i32,
    };

    // The reference coupling-start table is conservative for 5.1 at
    // 256-512 kbps.  Starting automatic coupling lower (band 6) improved
    // both ViSQOL and segmental SNR on real 5.1 downmixes.  Explicit `On`
    // keeps the reference start so callers can still request the old
    // behaviour.
    if channel_mode == Acmod::ThreeTwo.channel_mode()
        && coupling == Coupling::Auto
        && (12..=16).contains(&bitrate_code)
    {
        start = start.min(6);
    }

    let end_band = bandwidth_code / 4 + 3;
    let max_start = (end_band - 1).min(15) as i32;
    let start = start.clamp(0, max_start) as usize;
    let num_subbands = end_band - start;

    let mut band_sizes = [0usize; 18];
    let mut num_bands = 1;
    band_sizes[0] = 12;
    for &structure in DEFAULT_CPL_BAND_STRUCT
        .iter()
        .take(end_band)
        .skip(start + 1)
    {
        if structure != 0 {
            band_sizes[num_bands - 1] += 12;
        } else {
            band_sizes[num_bands] = 12;
            num_bands += 1;
        }
    }

    CouplingLayout {
        enabled: true,
        num_subbands,
        num_bands,
        band_sizes,
        start_freq: start * 12 + 37,
        end_freq: end_band * 12 + 37,
    }
}

/// Validate the audio service type / channel-count combination.
fn validate_service_type(service_type: ServiceType, channels: usize) -> Result<(), EncodeError> {
    match service_type {
        ServiceType::Karaoke if channels == 1 => Err(EncodeError::Config(
            "karaoke service type is invalid for a mono stream".into(),
        )),
        ServiceType::Commentary | ServiceType::Emergency | ServiceType::VoiceOver
            if channels > 1 =>
        {
            Err(EncodeError::Config(
                "service type is invalid for a multi-channel stream".into(),
            ))
        }
        _ => Ok(()),
    }
}

impl EncoderConfig {
    pub(crate) fn validate(&self) -> Result<ValidatedConfig, EncodeError> {
        let layout = self.layout;
        let channel_mode = layout.acmod.channel_mode();
        let fbw_channels = layout.acmod.full_bandwidth_channels();
        let channels = layout.channels();
        let lfe_on = layout.lfe;
        let lfe_channel = if lfe_on { fbw_channels + 1 } else { 0 };
        let has_center = (channel_mode & 0x01) != 0 && channel_mode != 1;
        let has_surround = (channel_mode & 0x04) != 0;

        if !(-31..=-1).contains(&self.dialnorm) {
            return Err(EncodeError::Config(
                "dialnorm must be between -31 and -1 dB".into(),
            ));
        }
        if self.bitrate_bps == 0 {
            return Err(EncodeError::Config("bitrate must be non-zero".into()));
        }
        if fbw_channels == 0 {
            return Err(EncodeError::Config(
                "layout has no fullband channels".into(),
            ));
        }

        let bitrate = self.bitrate_bps as i64;
        let num_blks_code = select_block_code(bitrate);
        let num_blocks = BLOCK_COUNTS[num_blks_code];
        let (min_br, max_br) = bitrate_limits(num_blocks);
        if !(min_br..=max_br).contains(&bitrate) {
            return Err(EncodeError::Config(format!(
                "invalid bit rate; must be {min_br} to {max_br} for this sample rate"
            )));
        }

        let frame_samples = (AC3_BLOCK_SIZE * num_blocks) as i64;
        let mut words_per_frame = (bitrate / 16) * frame_samples / SAMPLE_RATE as i64;
        if words_per_frame <= 0 || words_per_frame > 2048 {
            return Err(EncodeError::Config(
                "bit rate produces an invalid words-per-frame value".into(),
            ));
        }
        while words_per_frame > 1
            && words_per_frame * SAMPLE_RATE as i64 / (AC3_BLOCK_SIZE * 6) as i64 * 16 > bitrate
        {
            words_per_frame -= 1;
        }
        let frame_size_min = 2 * words_per_frame as usize;

        let bitrate_code = closest_bitrate_code(bitrate);
        let mut bandwidth_code = BANDWIDTH_TAB[fbw_channels - 1][0][bitrate_code] as usize;
        // The AC-3 bandwidth table is a poor fit for low-rate E-AC-3.
        if bitrate_code >= 4 {
            let floor = if fbw_channels <= 2 {
                44
            } else if fbw_channels >= 5 {
                match bitrate_code {
                    0..=5 => 0,
                    6 => 12,
                    7 | 8 => 28,
                    _ => 48,
                }
            } else {
                28
            };
            bandwidth_code = bandwidth_code.max(floor);
        }
        let coupling = coupling_layout(self.coupling, channel_mode, bandwidth_code, bitrate_code);

        let mut start_freq = [0usize; 8];
        if coupling.enabled {
            start_freq[0] = coupling.start_freq;
        }

        let mut channel_map = [0usize; 8];
        let map_row = &AC3_ENC_CHANNEL_MAP[channel_mode as usize][usize::from(lfe_on)];
        channel_map[..channels].copy_from_slice(&map_row[..channels]);

        let metadata = self.validate_metadata(channel_mode, channels, has_center, has_surround)?;

        Ok(ValidatedConfig {
            bit_rate: self.bitrate_bps,
            num_blks_code: num_blks_code as u8,
            num_blocks,
            frame_size_min,
            fbw_channels,
            channels,
            lfe_on,
            lfe_channel,
            channel_mode,
            has_center,
            has_surround,
            channel_map,
            bandwidth_code,
            start_freq,
            cpl_enabled: coupling.enabled,
            num_cpl_subbands: coupling.num_subbands,
            num_cpl_bands: coupling.num_bands,
            cpl_band_sizes: coupling.band_sizes,
            cpl_start_freq: coupling.start_freq,
            cpl_end_freq: coupling.end_freq,
            rematrixing_enabled: self.stereo_rematrixing && channel_mode == AC3_CHMODE_STEREO,
            dialnorm: self.dialnorm,
            metadata,
        })
    }

    fn validate_metadata(
        &self,
        channel_mode: u8,
        channels: usize,
        has_center: bool,
        has_surround: bool,
    ) -> Result<ValidatedMetadata, EncodeError> {
        let m = &self.metadata;

        let metadata_requested = m.preferred_stereo_downmix.is_some()
            || (has_center
                && (m.ltrt_center_mix_level.is_some() || m.loro_center_mix_level.is_some()))
            || (has_surround
                && (m.ltrt_surround_mix_level.is_some() || m.loro_surround_mix_level.is_some()));
        let eac3_mixing_metadata = channel_mode > AC3_CHMODE_STEREO && metadata_requested;

        let audio_production_info =
            m.mixing_level.is_some() || m.room_type.is_some() || m.ad_converter_type.is_some();
        let info_requested = m.service_type != ServiceType::Main
            || m.copyright.is_some()
            || m.original.is_some()
            || (channel_mode == AC3_CHMODE_STEREO
                && (m.dolby_headphone_mode.is_some() || m.dolby_surround_mode.is_some()))
            || (channel_mode >= AC3_CHMODE_2F2R && m.dolby_surround_ex_mode.is_some())
            || audio_production_info;
        let eac3_info_metadata = info_requested;

        validate_service_type(m.service_type, channels)?;

        let mut meta = ValidatedMetadata {
            eac3_mixing_metadata,
            eac3_info_metadata,
            preferred_stereo_downmix: 0,
            ltrt_center_mix_level: 0,
            ltrt_surround_mix_level: 0,
            loro_center_mix_level: 0,
            loro_surround_mix_level: 0,
            bitstream_mode: m.service_type as u8,
            copyright: false,
            original: false,
            dolby_surround_mode: 0,
            dolby_headphone_mode: 0,
            dolby_surround_ex_mode: 0,
            audio_production_info,
            mixing_level: 0,
            room_type: 0,
            ad_converter_type: 0,
        };

        if eac3_mixing_metadata {
            meta.preferred_stereo_downmix = m.preferred_stereo_downmix.map_or(0, |v| v as u8);
            if has_center {
                meta.ltrt_center_mix_level =
                    validate_mix_level("ltrt_center_mix_level", m.ltrt_center_mix_level, 5, 0)?;
                meta.loro_center_mix_level =
                    validate_mix_level("loro_center_mix_level", m.loro_center_mix_level, 5, 0)?;
            }
            if has_surround {
                meta.ltrt_surround_mix_level =
                    validate_mix_level("ltrt_surround_mix_level", m.ltrt_surround_mix_level, 6, 3)?;
                meta.loro_surround_mix_level =
                    validate_mix_level("loro_surround_mix_level", m.loro_surround_mix_level, 6, 3)?;
            }
        }

        if eac3_info_metadata {
            meta.dolby_headphone_mode = m.dolby_headphone_mode.map_or(0, |v| v as u8);
            meta.dolby_surround_ex_mode = m.dolby_surround_ex_mode.map_or(0, |v| v as u8);
            meta.ad_converter_type = m.ad_converter_type.map_or(0, |v| v as u8);
            meta.copyright = m.copyright.unwrap_or(false);
            meta.original = m.original.unwrap_or(true);
            if channel_mode == AC3_CHMODE_STEREO {
                meta.dolby_surround_mode = m.dolby_surround_mode.map_or(0, |v| v as u8);
            }
        }

        if audio_production_info {
            let Some(level) = m.mixing_level else {
                return Err(EncodeError::Config(
                    "mixing_level must be set when room_type or ad_converter_type is set".into(),
                ));
            };
            if !(80..=111).contains(&level) {
                return Err(EncodeError::Config(
                    "mixing level must be between 80 dB and 111 dB".into(),
                ));
            }
            meta.mixing_level = level;
            meta.room_type = m.room_type.map_or(0, |v| v as u8);
        }

        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(acmod: Acmod, lfe: bool, bitrate_bps: u32, coupling: Coupling) -> ValidatedConfig {
        EncoderConfig {
            layout: ChannelLayout { acmod, lfe },
            bitrate_bps,
            dialnorm: -27,
            metadata: BsiMetadata::default(),
            coupling,
            stereo_rematrixing: true,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn block_count_follows_bitrate() {
        let cases = [
            (4_000_000, 0u8, 1usize),
            (2_500_000, 1, 2),
            (1_500_000, 2, 3),
            (192_000, 3, 6),
        ];
        for (bitrate, code, blocks) in cases {
            let cfg = validate(Acmod::Stereo, false, bitrate, Coupling::Auto);
            assert_eq!(cfg.num_blks_code, code);
            assert_eq!(cfg.num_blocks, blocks);
        }
    }

    #[test]
    fn coupling_modes_are_resolved_from_the_bitrate_table() {
        // At 640 kbps the stereo/5.1 coupling table disables automatic
        // coupling, while explicit `On` keeps it available.
        let auto = validate(Acmod::ThreeTwo, true, 640_000, Coupling::Auto);
        assert!(!auto.cpl_enabled);
        assert_eq!(auto.num_cpl_bands, 0);

        let off = validate(Acmod::ThreeTwo, true, 640_000, Coupling::Off);
        assert!(!off.cpl_enabled);

        let on = validate(Acmod::ThreeTwo, true, 640_000, Coupling::On);
        assert!(on.cpl_enabled);
        assert!(on.num_cpl_bands > 0);
        assert!(on.cpl_start_freq >= 37);
        assert!(on.cpl_start_freq < on.cpl_end_freq);
    }

    #[test]
    fn five_one_auto_coupling_starts_at_band_six() {
        for bitrate in [256_000, 320_000, 384_000, 448_000, 512_000] {
            let cfg = validate(Acmod::ThreeTwo, true, bitrate, Coupling::Auto);
            assert!(cfg.cpl_enabled, "coupling should be enabled at {bitrate}");
            assert_eq!(cfg.cpl_start_freq, 6 * 12 + 37, "bitrate {bitrate}");
        }

        let auto_640 = validate(Acmod::ThreeTwo, true, 640_000, Coupling::Auto);
        assert!(!auto_640.cpl_enabled);

        let on_448 = validate(Acmod::ThreeTwo, true, 448_000, Coupling::On);
        assert!(on_448.cpl_enabled);
        assert_eq!(on_448.cpl_start_freq, 12 * 12 + 37);
    }

    #[test]
    fn low_rate_bandwidth_is_not_zero() {
        let stereo_96 = validate(Acmod::Stereo, false, 96_000, Coupling::Auto);
        assert_eq!(stereo_96.bandwidth_code, 44);
        let stereo_192 = validate(Acmod::Stereo, false, 192_000, Coupling::Auto);
        assert!(stereo_192.bandwidth_code >= 48);

        let five_one_96 = validate(Acmod::ThreeTwo, true, 96_000, Coupling::Auto);
        assert_eq!(five_one_96.bandwidth_code, 12);
        let five_one_128 = validate(Acmod::ThreeTwo, true, 128_000, Coupling::Auto);
        assert_eq!(five_one_128.bandwidth_code, 28);
        let five_one_224 = validate(Acmod::ThreeTwo, true, 224_000, Coupling::Auto);
        assert_eq!(five_one_224.bandwidth_code, 48);
    }

    #[test]
    fn coupling_requires_a_layout_with_surround_channels() {
        for acmod in [Acmod::DualMono, Acmod::Mono, Acmod::Stereo] {
            for coupling in [Coupling::Auto, Coupling::On, Coupling::Off] {
                let cfg = validate(acmod, false, 448_000, coupling);
                assert_eq!(
                    cfg.cpl_enabled,
                    coupling == Coupling::On && acmod == Acmod::Stereo
                );
            }
        }
    }

    #[test]
    fn metadata_presence_flags_follow_the_requested_fields() {
        let base = validate(Acmod::ThreeTwo, true, 640_000, Coupling::Auto);
        assert!(!base.metadata.eac3_mixing_metadata);
        assert!(!base.metadata.eac3_info_metadata);

        let cfg = EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::ThreeTwo,
                lfe: true,
            },
            bitrate_bps: 640_000,
            dialnorm: -27,
            metadata: BsiMetadata {
                copyright: Some(true),
                mixing_level: Some(90),
                room_type: Some(RoomType::Large),
                ..BsiMetadata::default()
            },
            coupling: Coupling::Auto,
            stereo_rematrixing: false,
        }
        .validate()
        .unwrap();

        assert!(cfg.metadata.eac3_info_metadata);
        assert!(cfg.metadata.audio_production_info);
        assert_eq!(cfg.metadata.mixing_level, 90);
        assert_eq!(cfg.metadata.room_type, RoomType::Large as u8);
        assert!(cfg.metadata.copyright);
        assert!(cfg.metadata.original);
    }

    #[test]
    fn channel_map_matches_internal_order() {
        let cfg = validate(Acmod::ThreeTwo, true, 640_000, Coupling::Auto);
        assert_eq!(&cfg.channel_map[..6], &[0, 2, 1, 4, 5, 3]);
        assert_eq!(cfg.lfe_channel, 6);
        assert_eq!(cfg.channels, 6);
    }
}
