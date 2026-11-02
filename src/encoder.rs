//! Frame/stream encoder orchestration.

use crate::aht;
use crate::aht_tables::{BITS_VS_HEBAP, HEBAP_TAB};
use crate::alloc;
use crate::analysis::{
    self, BitAllocParameters, EXPONENT_GROUP_TAB, ExpStrategy, encode_exponents_blk_ch,
    extract_exponents,
};
use crate::bitwriter::BitWriter;
use crate::config::{ChannelOrder, EncoderConfig, SAMPLE_RATE, ValidatedConfig, ValidatedMetadata};
use crate::coupling;
use crate::error::EncodeError;
use crate::header;
use crate::mdct::Mdct;
use crate::pack;
use crate::pcm::{PlanarFrame, collect_internal_channels};
use crate::quantize::{self, MantCounts};
use crate::rematrix;
use crate::tables::{
    AC3_BLOCK_SIZE, AC3_CHMODE_2F2R, AC3_CHMODE_MONO, AC3_CHMODE_STEREO, AC3_MAX_BLOCKS,
    AC3_MAX_CHANNELS, AC3_MAX_COEFS, BAP_TAB, EAC3_FRM_EXPSTR, FAST_GAIN_TAB,
};
use crate::transient::TransientDetector;

/// Internal channel slots: coupling channel 0, fullband 1..=fbw, LFE last.
pub(crate) const CH: usize = AC3_MAX_CHANNELS + 1;

const EXP_DIFF_THRESHOLD: i32 = 500;

/// Block switching costs a coupling-off frame and a second transform; below
/// this per-channel rate it has measured benefits on transient material,
/// above it the gain disappears and the coupling loss dominates.
const BLOCK_SWITCH_MAX_BPS_PER_CHANNEL: u32 = 64_000;

/// Largest syncframe: `frmsiz` is 11 bits and counts 16-bit words minus one.
pub(crate) const MAX_FRAME_BYTES: usize = 4096;

/// Validate a [`SkipMetadata`] for a frame with `num_blocks` audio blocks.
pub(crate) fn validate_skip_metadata(
    meta: &SkipMetadata,
    num_blocks: usize,
) -> Result<(), EncodeError> {
    if meta.addbsi.len() > 64 {
        return Err(EncodeError::Config(format!(
            "addbsi is {} bytes; the length field allows at most 64",
            meta.addbsi.len()
        )));
    }
    if meta.emdf.len() > 511 {
        return Err(EncodeError::Config(format!(
            "EMDF is {} bytes; the 9-bit skip-field length allows at most 511",
            meta.emdf.len()
        )));
    }
    if meta.has_skip_field() && meta.target_block >= num_blocks {
        return Err(EncodeError::Config(format!(
            "target block {} is outside the {num_blocks} blocks of this frame",
            meta.target_block
        )));
    }
    Ok(())
}

/// Patch the 11-bit `frmsiz` field (bit 21) after the frame grew.
pub(crate) fn patch_frame_size(bytes: &mut [u8], frame_size: usize) {
    debug_assert!(bytes.len() >= 4);
    debug_assert!((2..=MAX_FRAME_BYTES).contains(&frame_size));
    let frmsiz = (frame_size / 2 - 1) as u32;
    bytes[2] = (bytes[2] & 0xF8) | ((frmsiz >> 8) & 0x07) as u8;
    bytes[3] = (frmsiz & 0xFF) as u8;
}

/// State of a coupling update flag in the E-AC-3 syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CplFlag {
    /// Reuse the previous value.
    Reuse,
    /// A new value follows in the bitstream.
    New,
    /// A new value is implicit in the syntax (no flag bit).
    Implicit,
}

impl CplFlag {
    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::Reuse => 0,
            Self::New => 1,
            Self::Implicit => 2,
        }
    }

    pub(crate) const fn is_reuse(self) -> bool {
        matches!(self, Self::Reuse)
    }

    pub(crate) const fn is_implicit(self) -> bool {
        matches!(self, Self::Implicit)
    }

    pub(crate) const fn from_new(new: bool) -> Self {
        if new { Self::New } else { Self::Reuse }
    }
}

const D15: ExpStrategy = ExpStrategy::D15;
const D25: ExpStrategy = ExpStrategy::D25;
const D45: ExpStrategy = ExpStrategy::D45;

/// Final strategy for a run of reused blocks, by number of blocks per frame.
const EXP_STRATEGY_REUSE_TAB: [[ExpStrategy; 6]; 4] = [
    [D15; 6],
    [D15; 6],
    [D25, D25, D15, D15, D15, D15],
    [D45, D25, D25, D15, D15, D15],
];

fn exp_difference(current: &[u8], previous: &[u8]) -> i32 {
    current
        .iter()
        .zip(previous)
        .map(|(&current, &previous)| (current as i32 - previous as i32).abs())
        .sum()
}

/// Per-block coding state.
pub(crate) struct Block {
    pub mdct_coef: [[f32; AC3_MAX_COEFS]; CH],
    pub fixed_coef: [[i32; AC3_MAX_COEFS]; CH],
    pub exp: [[u8; AC3_MAX_COEFS]; CH],
    pub grouped_exp: [[u8; 128]; CH],
    pub psd: [[i16; AC3_MAX_COEFS]; CH],
    pub band_psd: [[i16; 64]; CH],
    pub mask: [[i16; 64]; CH],
    pub qmant: [[i16; AC3_MAX_COEFS]; CH],
    pub cpl_coord_exp: [[u8; 16]; CH],
    pub cpl_coord_mant: [[u8; 16]; CH],
    pub new_rematrixing_strategy: bool,
    pub num_rematrixing_bands: usize,
    pub rematrixing_flags: [u8; 4],
    pub new_cpl_strategy: bool,
    pub cpl_in_use: bool,
    pub channel_in_cpl: [bool; CH],
    pub num_cpl_channels: usize,
    pub new_cpl_coords: [CplFlag; CH],
    pub cpl_master_exp: [u8; CH],
    pub new_snr_offsets: bool,
    pub new_cpl_leak: CplFlag,
    pub end_freq: [usize; CH],
    /// Write bap_work into the state used for output.
    pub bap: [[u8; AC3_MAX_COEFS]; CH],
}

impl Block {
    pub(crate) fn new(cfg: &ValidatedConfig) -> Self {
        let mut end_freq = [0usize; CH];
        for e in end_freq.iter_mut().take(cfg.fbw_channels + 1).skip(1) {
            *e = cfg.bandwidth_code * 3 + 73;
        }
        if cfg.lfe_on {
            end_freq[cfg.lfe_channel] = 7;
        }
        if cfg.cpl_enabled {
            end_freq[0] = cfg.cpl_end_freq;
        }
        Self {
            mdct_coef: [[0.0; AC3_MAX_COEFS]; CH],
            fixed_coef: [[0; AC3_MAX_COEFS]; CH],
            exp: [[0; AC3_MAX_COEFS]; CH],
            grouped_exp: [[0; 128]; CH],
            psd: [[0; AC3_MAX_COEFS]; CH],
            band_psd: [[0; 64]; CH],
            mask: [[0; 64]; CH],
            qmant: [[0; AC3_MAX_COEFS]; CH],
            cpl_coord_exp: [[0; 16]; CH],
            cpl_coord_mant: [[0; 16]; CH],
            new_rematrixing_strategy: false,
            num_rematrixing_bands: 0,
            rematrixing_flags: [0; 4],
            new_cpl_strategy: false,
            cpl_in_use: false,
            channel_in_cpl: [false; CH],
            num_cpl_channels: 0,
            new_cpl_coords: [CplFlag::Reuse; CH],
            cpl_master_exp: [0; CH],
            new_snr_offsets: false,
            new_cpl_leak: CplFlag::Reuse,
            end_freq,
            bap: [[0; AC3_MAX_COEFS]; CH],
        }
    }
}

/// Channel indices in the order required by the audio-block syntax.
///
/// The coupling channel (index 0) is written after the first coupled full-band
/// channel and before the next one.  The returned slice is `order[..count]`.
pub(crate) fn coded_channel_order(cfg: &ValidatedConfig, block: &Block) -> ([usize; CH], usize) {
    let mut order = [0usize; CH];
    let mut count = 0;

    let first_coupled = block
        .cpl_in_use
        .then(|| (1..=cfg.fbw_channels).find(|&ch| block.channel_in_cpl[ch]))
        .flatten();

    match first_coupled {
        // The coupling channel is only emitted if a channel follows it.
        Some(first) if first < cfg.channels => {
            for ch in 1..=first {
                order[count] = ch;
                count += 1;
            }
            order[count] = 0;
            count += 1;
            for ch in first + 1..=cfg.channels {
                order[count] = ch;
                count += 1;
            }
        }
        _ => {
            for ch in 1..=cfg.channels {
                order[count] = ch;
                count += 1;
            }
        }
    }
    (order, count)
}

/// Atmos metadata embedded in an E-AC-3 frame's skip field.
///
/// Both payloads are written natively by the encoder, so every syntax reac3e
/// supports (any block count, block switching, coupling, rematrixing, AHT) is
/// handled without re-parsing the finished frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkipMetadata {
    /// Raw `addbsi` bytes, written after `addbsie`/`addbsil`.  Empty disables
    /// the `addbsi` field.  At most 64 bytes (the length field is 6 bits).
    pub addbsi: Vec<u8>,
    /// EMDF container bytes, carried in one audio block's skip field.  Empty
    /// disables the skip field.  At most 511 bytes (`skipl` is 9 bits).
    pub emdf: Vec<u8>,
    /// Zero-based audio block that carries `emdf`.
    pub target_block: usize,
}

impl SkipMetadata {
    pub fn new(addbsi: &[u8], emdf: &[u8], target_block: usize) -> Self {
        Self {
            addbsi: addbsi.to_vec(),
            emdf: emdf.to_vec(),
            target_block,
        }
    }

    pub(crate) fn has_skip_field(&self) -> bool {
        !self.emdf.is_empty()
    }

    pub(crate) fn has_addbsi(&self) -> bool {
        !self.addbsi.is_empty()
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.has_skip_field() && !self.has_addbsi()
    }
}

/// Exact bit positions of the fields a metadata injector may patch.
///
/// These are absolute bit offsets from the start of [`Frame::bytes`] and remain
/// valid for the frame that produced them (including any skip-field/addbsi
/// syntax written natively).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameEdits {
    /// Audio blocks per syncframe for this frame.
    pub num_blocks: usize,
    /// Final syncframe size in bytes, including CRC2.
    pub frame_size: usize,
    /// Bit position of the BSI `infomdate` flag and the first bit after its body.
    pub infomdate_bit: usize,
    pub info_end_bit: usize,
    /// Bit position of the BSI `addbsie` flag.
    pub addbsie_bit: usize,
    /// Number of `addbsi` bytes written after the flag (0 = no addbsi field).
    pub addbsi_len: usize,
    /// Bit position of the `audfrm` `skipFieldSyntaxEnabled` flag.
    pub skipfld_bit: usize,
    /// Bit position of each block's skip-field point (before `skiple`, just
    /// before the mantissa data), matching the E-AC-3 `audblk()` syntax.
    pub block_skip_points: Vec<usize>,
    /// Bit position of the first EMDF byte in the target block, or `None`
    /// when the frame carries no skip-field payload.
    pub emdf_start_bit: Option<usize>,
    /// Bit position just after the last written audio/metadata bit, before the
    /// zero padding and the 16-bit CRC2. Used by frame signers to exclude the
    /// padded tail.
    pub audio_end_bit: usize,
}

/// One encoded syncframe.
#[derive(Debug, Clone)]
pub struct Frame {
    pub bytes: Vec<u8>,
    pub index: u64,
    /// Bit positions of the injectable fields in `bytes`.
    pub edits: FrameEdits,
}

/// Running bitrate accounting used to alternate the two legal frame sizes.
#[derive(Debug, Clone, Copy, Default)]
struct FrameBudget {
    bits_written: i64,
    samples_written: i64,
}

impl FrameBudget {
    /// Choose the next frame size, keeping the long-term average on target.
    fn next_frame_size(&mut self, cfg: &ValidatedConfig) -> usize {
        let bit_rate = cfg.bit_rate as i64;
        let sample_rate = SAMPLE_RATE as i64;
        while self.bits_written >= bit_rate && self.samples_written >= sample_rate {
            self.bits_written -= bit_rate;
            self.samples_written -= sample_rate;
        }
        let add_two = self.bits_written * sample_rate < self.samples_written * bit_rate;
        let frame_size = cfg.frame_size_min + 2 * usize::from(add_two);
        self.bits_written += frame_size as i64 * 8;
        self.samples_written += (AC3_BLOCK_SIZE * cfg.num_blocks) as i64;
        frame_size
    }
}

/// Encodes complete E-AC-3 syncframes from planar PCM.
pub struct FrameEncoder {
    pub(crate) cfg: ValidatedConfig,
    mdct: Mdct,
    pub(crate) blocks: Vec<Block>,
    tail: [[f32; AC3_BLOCK_SIZE]; CH],
    pub(crate) exp_strategy: [[ExpStrategy; AC3_MAX_BLOCKS]; CH],
    pub(crate) frame_exp_strategy: [u8; CH],
    pub(crate) use_frame_exp_strategy: bool,
    pub(crate) exp_ref_block: [[u8; AC3_MAX_BLOCKS]; CH],
    bap_work: Vec<[[u8; AC3_MAX_COEFS]; CH]>,
    pub(crate) cpl_on: bool,
    pub(crate) phase_flags: [u8; 18],
    pub(crate) phase_flags_in_use: bool,
    pub(crate) bit_alloc: BitAllocParameters,
    pub(crate) coarse_snr_offset: u8,
    pub(crate) fine_snr_offset: [u8; CH],
    pub(crate) frame_bits_fixed: i32,
    pub(crate) frame_size: usize,
    pub(crate) aht_enabled: bool,
    pub(crate) aht_used: bool,
    pub(crate) aht_channels: [bool; CH],
    pub(crate) aht_coef: [[[f64; 6]; AC3_MAX_COEFS]; CH],
    pub(crate) aht_vq: [[u16; AC3_MAX_COEFS]; CH],
    pub(crate) aht_quant: [[[i32; 6]; AC3_MAX_COEFS]; CH],
    budget: FrameBudget,
    pub(crate) frame_index: u64,
    final_frame_seen: bool,
    pub(crate) blksw: [[bool; AC3_MAX_BLOCKS]; CH],
    pub(crate) blk_sw_syntax: bool,
    block_switch: bool,
    transient: Vec<TransientDetector>,
    /// Metadata for the frame currently being packed.
    pub(crate) frame_meta: Option<SkipMetadata>,
}

impl FrameEncoder {
    pub fn new(config: EncoderConfig) -> Result<Self, EncodeError> {
        let cfg = config.validate()?;
        let num_blocks = cfg.num_blocks;
        let blocks = (0..num_blocks).map(|_| Block::new(&cfg)).collect();
        let bap_work = vec![[[0u8; AC3_MAX_COEFS]; CH]; num_blocks];
        let bit_alloc = BitAllocParameters::eac3_defaults();
        let frame_bits_fixed = count_frame_bits_fixed(&cfg);
        Ok(Self {
            frame_size: cfg.frame_size_min,
            cfg,
            mdct: Mdct::new(),
            blocks,
            tail: [[0.0; AC3_BLOCK_SIZE]; CH],
            exp_strategy: [[ExpStrategy::Reuse; AC3_MAX_BLOCKS]; CH],
            frame_exp_strategy: [0; CH],
            use_frame_exp_strategy: false,
            exp_ref_block: [[0; AC3_MAX_BLOCKS]; CH],
            bap_work,
            cpl_on: false,
            phase_flags: [0; 18],
            phase_flags_in_use: false,
            bit_alloc,
            coarse_snr_offset: 40,
            fine_snr_offset: [0; CH],
            frame_bits_fixed,
            aht_enabled: false,
            aht_used: false,
            aht_channels: [false; CH],
            aht_coef: [[[0.0; 6]; AC3_MAX_COEFS]; CH],
            aht_vq: [[0; AC3_MAX_COEFS]; CH],
            aht_quant: [[[0; 6]; AC3_MAX_COEFS]; CH],
            budget: FrameBudget::default(),
            frame_index: 0,
            final_frame_seen: false,
            blksw: [[false; AC3_MAX_BLOCKS]; CH],
            blk_sw_syntax: false,
            block_switch: true,
            transient: (0..CH).map(|_| TransientDetector::new()).collect(),
            frame_meta: None,
        })
    }

    /// Enable or disable block switching (enabled by default).
    pub fn set_block_switch(&mut self, enabled: bool) {
        self.block_switch = enabled;
    }

    /// Enable or disable the experimental AHT path.
    pub fn set_aht_enabled(&mut self, enabled: bool) {
        self.aht_enabled = enabled;
    }

    /// Samples represented by one syncframe (1536 for six blocks).
    pub fn samples_per_frame(&self) -> usize {
        AC3_BLOCK_SIZE * self.cfg.num_blocks
    }

    /// Actual coded bandwidth cutoff in Hz.
    pub fn cutoff_hz(&self) -> f64 {
        self.cfg.cutoff_hz()
    }

    /// Encoder delay introduced by the initial 256-sample overlap window,
    /// matching the initial overlap padding.
    pub fn latency(&self) -> usize {
        AC3_BLOCK_SIZE
    }

    /// Encode one frame.  `pcm` must contain exactly `samples_per_frame()`
    /// samples per channel; a shorter final frame is accepted and zero padded.
    pub fn encode_frame(&mut self, pcm: &PlanarFrame<'_>) -> Result<Frame, EncodeError> {
        self.encode_frame_with_metadata(pcm, None)
    }

    /// Encode one frame with optional Atmos metadata written into the frame.
    ///
    /// The audio is still allocated against the configured CBR core frame size;
    /// `addbsi` and the skip-field EMDF are appended, so the returned frame is
    /// larger than the core frame by the metadata overhead.  Block switching,
    /// coupling, rematrixing and AHT are all handled because the metadata is
    /// written by the same packer that emits the audio blocks.
    pub fn encode_frame_with_metadata(
        &mut self,
        pcm: &PlanarFrame<'_>,
        metadata: Option<SkipMetadata>,
    ) -> Result<Frame, EncodeError> {
        if self.final_frame_seen {
            return Err(EncodeError::Pcm(
                "cannot encode another frame after a short final frame".into(),
            ));
        }
        if let Some(meta) = &metadata {
            validate_skip_metadata(meta, self.cfg.num_blocks)?;
        }
        let channels = collect_internal_channels(&self.cfg, pcm)?;
        let len = channels[1].len();
        let spf = self.samples_per_frame();
        if len > spf {
            return Err(EncodeError::Pcm(format!(
                "frame contains {len} samples, maximum is {spf}"
            )));
        }
        if len < spf {
            self.final_frame_seen = true;
        }
        let result = self.encode_internal(&channels, len, metadata);
        // Clear even on error so a failed frame cannot leak metadata into the
        // next one.
        self.frame_meta = None;
        result
    }

    /// Flush the final partial frame.  `encode_frame` already zero-pads a
    /// short final frame, so there is no buffered audio left; this always
    /// returns `None` and is provided for API symmetry and idempotence.
    pub fn flush(&mut self) -> Result<Option<Frame>, EncodeError> {
        Ok(None)
    }

    fn encode_internal(
        &mut self,
        channels: &[&[f32]],
        valid: usize,
        metadata: Option<SkipMetadata>,
    ) -> Result<Frame, EncodeError> {
        let spf = self.samples_per_frame();
        self.adjust_frame_size();
        self.frame_meta = metadata.filter(|meta| !meta.is_empty());

        // --- block-switch detection and MDCT ---
        let num_blocks = self.cfg.num_blocks;
        self.blksw = [[false; AC3_MAX_BLOCKS]; CH];
        let mut any_switched = false;
        for (ch, channel) in channels
            .iter()
            .enumerate()
            .take(self.cfg.channels + 1)
            .skip(1)
        {
            let mut input = Vec::with_capacity(AC3_BLOCK_SIZE + spf);
            input.extend_from_slice(&self.tail[ch]);
            input.extend_from_slice(channel);
            if valid < spf {
                input.resize(AC3_BLOCK_SIZE + spf, 0.0);
            }

            let switch_allowed = self.block_switch
                && self.cfg.bit_rate / self.cfg.channels as u32 <= BLOCK_SWITCH_MAX_BPS_PER_CHANNEL;
            if switch_allowed && ch <= self.cfg.fbw_channels {
                for blk in 0..num_blocks {
                    let start = AC3_BLOCK_SIZE + blk * AC3_BLOCK_SIZE;
                    let segment = &input[start..start + AC3_BLOCK_SIZE];
                    self.blksw[ch][blk] = self.transient[ch].detect(segment);
                    any_switched |= self.blksw[ch][blk];
                }
            }

            let mut windowed = [0.0f32; AC3_BLOCK_SIZE * 2];
            for blk in 0..num_blocks {
                let start = blk * AC3_BLOCK_SIZE;
                self.mdct
                    .window_block(&input[start..start + AC3_BLOCK_SIZE * 2], &mut windowed);
                if self.blksw[ch][blk] {
                    let (first, second) = windowed.split_at(AC3_BLOCK_SIZE);
                    self.mdct
                        .process_short(first, second, &mut self.blocks[blk].mdct_coef[ch]);
                } else {
                    self.mdct
                        .process(&windowed, &mut self.blocks[blk].mdct_coef[ch]);
                }
            }
            self.tail[ch].copy_from_slice(&input[spf..spf + AC3_BLOCK_SIZE]);
        }
        self.blk_sw_syntax = any_switched;

        // --- coupling strategy + coupling channel ---
        self.cpl_on = coupling::compute_coupling_strategy(
            &self.cfg,
            &mut self.blocks,
            self.cfg.cpl_enabled && !self.blk_sw_syntax,
        );
        if self.cpl_on {
            coupling::apply_channel_coupling(
                &self.cfg,
                &mut self.blocks,
                &mut self.phase_flags,
                &mut self.phase_flags_in_use,
            );
        } else {
            self.phase_flags_in_use = false;
        }

        // --- rematrixing strategy (uses float coefficients) ---
        rematrix::compute_rematrixing_strategy(
            &self.cfg,
            &mut self.blocks,
            self.cfg.rematrixing_enabled,
        );

        // --- scale to fixed point ---
        let ch_start = if self.cpl_on { 0 } else { 1 };
        for ch in ch_start..=self.cfg.channels {
            for blk in 0..self.cfg.num_blocks {
                let mdct = self.blocks[blk].mdct_coef[ch];
                quantize::float_to_fixed24(&mut self.blocks[blk].fixed_coef[ch], &mdct);
            }
        }

        // --- rematrixing on the fixed-point coefficients ---
        rematrix::apply_rematrixing(&mut self.blocks, self.cfg.rematrixing_enabled);

        // --- exponents ---
        self.extract_exponents();
        self.compute_exp_strategy();
        if self.aht_enabled {
            self.prepare_aht();
        }
        self.encode_exponents();

        // --- bit allocation ---
        self.compute_bit_allocation()?;

        // --- grouped exponents, quantisation, packing ---
        self.group_exponents();
        self.quantize_aht();
        self.quantize_mantissas();

        let mut w = BitWriter::with_capacity(self.frame_size);
        let (addbsie_bit, skipfld_bit, infomdate_bit, info_end_bit) =
            header::write_frame_header(self, &mut w);
        let mut block_skip_points = Vec::with_capacity(self.cfg.num_blocks);
        for blk in 0..self.cfg.num_blocks {
            block_skip_points.push(pack::write_audio_block(self, blk, &mut w));
        }
        let audio_end_bit = w.bits_written();
        let mut bytes = w.into_bytes();

        // Audio is budgeted against the core CBR frame size; metadata grows the
        // frame on top of it.  `addbsi` replaces the 1-bit flag, and every block
        // gains a `skiple` bit (the target block also gets `skipl` + payload).
        let overhead_bits = self.frame_meta.as_ref().map_or(0, |meta| {
            let addbsi = if meta.addbsi.is_empty() {
                0
            } else {
                6 + meta.addbsi.len() * 8
            };
            let skip = if meta.emdf.is_empty() {
                0
            } else {
                self.cfg.num_blocks + 9 + meta.emdf.len() * 8
            };
            addbsi + skip
        });
        let frame_size = if self.frame_meta.is_some() {
            let mut size = (self.frame_size * 8 + overhead_bits).div_ceil(8);
            if !size.is_multiple_of(2) {
                size += 1;
            }
            if size > MAX_FRAME_BYTES {
                self.frame_meta = None;
                return Err(EncodeError::Config(format!(
                    "frame with metadata is {size} bytes; the E-AC-3 limit is {MAX_FRAME_BYTES}"
                )));
            }
            size
        } else {
            self.frame_size
        };
        patch_frame_size(&mut bytes, frame_size);
        let bytes = pack::finalize(bytes, frame_size)?;

        let frame = Frame {
            bytes,
            index: self.frame_index,
            edits: FrameEdits {
                num_blocks: self.cfg.num_blocks,
                frame_size,
                infomdate_bit,
                info_end_bit,
                addbsie_bit,
                addbsi_len: self.frame_meta.as_ref().map_or(0, |meta| meta.addbsi.len()),
                skipfld_bit,
                emdf_start_bit: self
                    .frame_meta
                    .as_ref()
                    .filter(|meta| !meta.emdf.is_empty())
                    .map(|meta| block_skip_points[meta.target_block] + 10),
                audio_end_bit,
                block_skip_points,
            },
        };
        self.frame_meta = None;
        self.frame_index += 1;
        Ok(frame)
    }

    fn adjust_frame_size(&mut self) {
        self.frame_size = self.budget.next_frame_size(&self.cfg);
    }

    fn extract_exponents(&mut self) {
        let start = usize::from(!self.cpl_on);
        let channels = self.cfg.channels;
        for block in &mut self.blocks {
            for ch in start..=channels {
                extract_exponents(&block.fixed_coef[ch], &mut block.exp[ch], AC3_MAX_COEFS);
            }
        }
    }

    fn compute_exp_strategy(&mut self) {
        let start = usize::from(!self.cpl_on);
        for ch in start..=self.cfg.fbw_channels {
            self.compute_channel_exp_strategy(ch);
            self.compact_exp_strategy_runs(ch);
        }

        if self.cfg.lfe_on {
            let ch = self.cfg.lfe_channel;
            self.exp_strategy[ch][0] = ExpStrategy::D15;
            for blk in 1..self.cfg.num_blocks {
                self.exp_strategy[ch][blk] = ExpStrategy::Reuse;
            }
        }

        self.get_frame_exp_strategy();
    }

    fn compute_channel_exp_strategy(&mut self, ch: usize) {
        self.exp_strategy[ch][0] = ExpStrategy::D15;
        for blk in 1..self.cfg.num_blocks {
            self.exp_strategy[ch][blk] = self.block_exp_strategy(ch, blk);
        }
    }

    /// Per-block strategy before reuse runs are compacted into their final
    /// strategy codes.
    fn block_exp_strategy(&self, ch: usize, blk: usize) -> ExpStrategy {
        if ch == 0 {
            if !self.blocks[blk - 1].cpl_in_use {
                return ExpStrategy::D15;
            }
            if !self.blocks[blk].cpl_in_use {
                return ExpStrategy::Reuse;
            }
        } else if self.blocks[blk].channel_in_cpl[ch] != self.blocks[blk - 1].channel_in_cpl[ch] {
            return ExpStrategy::D15;
        }

        let difference = exp_difference(&self.blocks[blk].exp[ch], &self.blocks[blk - 1].exp[ch]);
        let over_threshold = if ch == 0 {
            difference
                > EXP_DIFF_THRESHOLD
                    * (self.blocks[blk].end_freq[ch] - self.cfg.start_freq[ch]) as i32
                    / AC3_MAX_COEFS as i32
        } else {
            difference > EXP_DIFF_THRESHOLD
        };
        if over_threshold {
            ExpStrategy::D15
        } else {
            ExpStrategy::Reuse
        }
    }

    fn compact_exp_strategy_runs(&mut self, ch: usize) {
        let mut blk = 0usize;
        while blk < self.cfg.num_blocks {
            let mut next = blk + 1;
            while next < self.cfg.num_blocks && self.exp_strategy[ch][next].is_reuse() {
                next += 1;
            }
            self.exp_strategy[ch][blk] =
                EXP_STRATEGY_REUSE_TAB[self.cfg.num_blks_code as usize][next - blk - 1];
            blk = next;
        }
    }

    fn get_frame_exp_strategy(&mut self) {
        if self.cfg.num_blocks < 6 {
            self.use_frame_exp_strategy = false;
            return;
        }

        let start = usize::from(!self.cpl_on);
        for ch in start..=self.cfg.fbw_channels {
            let strategies: [u8; 6] = std::array::from_fn(|blk| self.exp_strategy[ch][blk].code());
            let Some(index) = EAC3_FRM_EXPSTR.iter().position(|row| *row == strategies) else {
                self.use_frame_exp_strategy = false;
                return;
            };
            self.frame_exp_strategy[ch] = index as u8;
        }
        self.use_frame_exp_strategy = true;
    }

    fn prepare_aht(&mut self) {
        self.aht_used = false;
        self.aht_channels = [false; CH];
        if self.cfg.num_blocks != 6 || self.blk_sw_syntax {
            return;
        }

        let first_ch = usize::from(!self.cpl_on);
        for ch in first_ch..=self.cfg.fbw_channels {
            if self.cfg.lfe_on && ch == self.cfg.lfe_channel {
                continue;
            }
            let in_cpl = self.blocks[0].channel_in_cpl[ch];
            let end = self.blocks[0].end_freq[ch];
            if self
                .blocks
                .iter()
                .any(|block| block.channel_in_cpl[ch] != in_cpl || block.end_freq[ch] != end)
            {
                continue;
            }
            let start = self.cfg.start_freq[ch];
            if start >= end {
                continue;
            }
            let mut exponents = [0u8; AC3_MAX_COEFS];
            let mut ok = true;
            let mut dc_energy = 0.0f64;
            let mut total_energy = 0.0f64;
            for (bin, exponent_slot) in exponents.iter_mut().enumerate().take(end).skip(start) {
                let mut input = [0.0f64; 6];
                for (blk, value) in input.iter_mut().enumerate() {
                    *value = self.blocks[blk].fixed_coef[ch][bin] as f64;
                }
                let aht = aht::forward(&input);
                dc_energy += aht[0] * aht[0];
                for value in aht {
                    total_energy += value * value;
                }
                let max_abs = aht
                    .iter()
                    .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
                let exponent = if max_abs < 1.0 {
                    24
                } else {
                    23 - max_abs.log2().floor() as i32
                };
                if bin == start && exponent > 15 {
                    ok = false;
                    break;
                }
                *exponent_slot = exponent.clamp(0, 24) as u8;
                self.aht_coef[ch][bin] = aht;
            }
            if !ok || total_energy <= 0.0 || dc_energy / total_energy < 0.1 {
                continue;
            }

            for block in &mut self.blocks {
                block.exp[ch][start..end].copy_from_slice(&exponents[start..end]);
            }
            self.exp_strategy[ch][0] = D15;
            for blk in 1..self.cfg.num_blocks {
                self.exp_strategy[ch][blk] = ExpStrategy::Reuse;
            }
            self.aht_channels[ch] = true;
            self.aht_used = true;
        }

        if self.aht_used {
            self.get_frame_exp_strategy();
        }
    }

    /// Quantise the transformed AHT coefficients once the final bit
    /// allocation (hebap) is known.
    fn quantize_aht(&mut self) {
        let first_ch = usize::from(!self.cpl_on);
        for ch in first_ch..=self.cfg.fbw_channels {
            if !self.aht_channels[ch] {
                continue;
            }
            let start = self.cfg.start_freq[ch];
            let end = self.blocks[0].end_freq[ch];
            for bin in start..end {
                let hebap = self.bap_work[0][ch][bin];
                self.blocks[0].bap[ch][bin] = hebap;
                let exponent = self.blocks[0].exp[ch][bin] as i32;
                let scale = 2f64.powi(exponent - 1);
                let mut target = [0.0f64; 6];
                for (target, value) in target.iter_mut().zip(self.aht_coef[ch][bin]) {
                    *target = value * scale;
                }
                let (vq, mantissa) = aht::quantise(hebap, &target);
                self.aht_vq[ch][bin] = vq;
                self.aht_quant[ch][bin] = mantissa;
            }
        }
    }

    fn encode_exponents(&mut self) {
        let start = usize::from(!self.cpl_on);
        for ch in start..=self.cfg.channels {
            let cpl = ch == 0;
            let mut blk = 0usize;
            while blk < self.cfg.num_blocks {
                if cpl && !self.blocks[blk].cpl_in_use {
                    blk += 1;
                    continue;
                }
                let nb_coefs = self.blocks[blk].end_freq[ch] - self.cfg.start_freq[ch];
                let mut next = blk + 1;
                self.exp_ref_block[ch][blk] = blk as u8;
                while next < self.cfg.num_blocks && self.exp_strategy[ch][next].is_reuse() {
                    self.exp_ref_block[ch][next] = blk as u8;
                    next += 1;
                }
                let reuse_blocks = next - blk - 1;

                // Reused blocks must not use an exponent below the reference.
                if reuse_blocks > 0 {
                    let minimum: [u8; AC3_MAX_COEFS] = std::array::from_fn(|coefficient| {
                        self.blocks[blk..next]
                            .iter()
                            .map(|block| block.exp[ch][coefficient])
                            .min()
                            .expect("reference block is present")
                    });
                    self.blocks[blk].exp[ch] = minimum;
                }

                let strategy = self.exp_strategy[ch][blk];
                encode_exponents_blk_ch(
                    &mut self.blocks[blk].exp[ch],
                    self.cfg.start_freq[ch],
                    nb_coefs,
                    strategy,
                    cpl,
                );
                blk = next;
            }
        }
    }

    fn compute_bit_allocation(&mut self) -> Result<(), EncodeError> {
        let frame_bits = self.count_frame_bits();
        let exponent_bits = self.count_exponent_bits();
        self.bit_alloc_masking();

        let bits_left =
            8 * self.frame_size as i32 - (self.frame_bits_fixed + frame_bits + exponent_bits);
        if bits_left < 0 {
            return Err(EncodeError::BitAllocation);
        }

        let initial_snr = (self.coarse_snr_offset as i32) << 4;
        let previous_snr = ((self.coarse_snr_offset as i32) << 4) | self.fine_snr_offset[1] as i32;
        let chosen = alloc::cbr_search(bits_left, initial_snr, previous_snr, |snr| {
            self.run_bit_alloc(snr) as i32
        })
        .ok_or(EncodeError::BitAllocation)?;

        // Re-run the allocation at the chosen offset to populate the final baps.
        self.run_bit_alloc(chosen);

        let (coarse, fine) = alloc::split_snr_offset(chosen);
        self.coarse_snr_offset = coarse;
        let start = if self.cpl_on { 0 } else { 1 };
        for ch in start..=self.cfg.channels {
            self.fine_snr_offset[ch] = fine;
        }
        Ok(())
    }

    fn run_bit_alloc(&mut self, snr_offset: i32) -> usize {
        let snr_offset = (snr_offset - 240) * 4;
        let start_ch = usize::from(!self.cpl_on);
        let floor = self.bit_alloc.floor;

        for (blk, block) in self.blocks.iter().enumerate() {
            for ch in start_ch..=self.cfg.channels {
                if self.exp_strategy[ch][blk].is_reuse() {
                    continue;
                }
                let table = if self.aht_channels[ch] {
                    &HEBAP_TAB
                } else {
                    &BAP_TAB
                };
                alloc::calc_bap(
                    &block.mask[ch],
                    &block.psd[ch],
                    self.cfg.start_freq[ch],
                    block.end_freq[ch],
                    snr_offset,
                    floor,
                    table,
                    &mut self.bap_work[blk][ch],
                );
            }
        }

        // Reuse blocks point at the reference block's allocation.
        for ch in start_ch..=self.cfg.channels {
            for blk in 0..self.cfg.num_blocks {
                if ch == 0 && !self.blocks[blk].cpl_in_use {
                    continue;
                }
                if self.exp_strategy[ch][blk].is_reuse() {
                    let reference = self.exp_ref_block[ch][blk] as usize;
                    self.bap_work[blk][ch] = self.bap_work[reference][ch];
                }
            }
        }
        self.count_mantissa_bits()
    }

    fn count_mantissa_bits(&self) -> usize {
        let mut mant_cnt = [[0u16; 16]; AC3_MAX_BLOCKS];
        alloc::count_mantissa_bits_init(&mut mant_cnt);
        let max_end_freq = self.cfg.bandwidth_code * 3 + 73;
        let start_ch = usize::from(!self.cfg.cpl_enabled);
        let mut aht_bits = 0usize;
        for (blk, block) in self.blocks.iter().enumerate() {
            for ch in start_ch..=self.cfg.channels {
                if ch == 0 && !block.cpl_in_use {
                    continue;
                }
                if self.aht_channels[ch] {
                    if blk == 0 {
                        for bin in self.cfg.start_freq[ch]..block.end_freq[ch] {
                            let hebap = self.bap_work[0][ch][bin] as usize;
                            if hebap == 0 {
                                continue;
                            }
                            let bits = BITS_VS_HEBAP[hebap] as usize;
                            aht_bits += if hebap < 8 { bits } else { bits * 6 };
                        }
                    }
                    continue;
                }
                let start = self.cfg.start_freq[ch];
                let end = max_end_freq.min(block.end_freq[ch]);
                let len = end - start;
                alloc::update_bap_counts(&mut mant_cnt[blk], &self.bap_work[blk][ch][start..], len);
            }
        }
        aht_bits + alloc::compute_mantissa_size(&mant_cnt)
    }

    fn bit_alloc_masking(&mut self) {
        let fast_gain = FAST_GAIN_TAB[4] as i32;
        let bit_alloc = self.bit_alloc;
        let start_freq = self.cfg.start_freq;
        let lfe_channel = self.cfg.lfe_channel;
        let channels = self.cfg.channels;

        for (blk, block) in self.blocks.iter_mut().enumerate() {
            let start_ch = usize::from(!block.cpl_in_use);
            for (ch, &start) in start_freq
                .iter()
                .enumerate()
                .take(channels + 1)
                .skip(start_ch)
            {
                if self.exp_strategy[ch][blk].is_reuse() {
                    continue;
                }
                let end = block.end_freq[ch];
                analysis::calc_psd(
                    &block.exp[ch],
                    start,
                    end,
                    &mut block.psd[ch],
                    &mut block.band_psd[ch],
                );
                analysis::calc_mask(
                    &bit_alloc,
                    &block.band_psd[ch],
                    start,
                    end,
                    fast_gain,
                    ch == lfe_channel,
                    &mut block.mask[ch],
                );
            }
        }
    }

    fn count_exponent_bits(&self) -> i32 {
        let mut bit_count = 0i32;
        for (blk, block) in self.blocks.iter().enumerate() {
            let start_ch = usize::from(!block.cpl_in_use);
            for ch in start_ch..=self.cfg.channels {
                let strategy = self.exp_strategy[ch][blk];
                if strategy.is_reuse() {
                    continue;
                }
                let cpl = ch == 0;
                let nb_coefs = block.end_freq[ch] - self.cfg.start_freq[ch];
                let nb_groups =
                    EXPONENT_GROUP_TAB[cpl as usize][strategy.table_index()][nb_coefs] as i32;
                bit_count += 4 + nb_groups * 7;
            }
        }
        bit_count
    }

    fn count_frame_bits(&self) -> i32 {
        let mut bits = count_metadata_bits(&self.cfg.metadata, &self.cfg);
        bits += self.count_coupling_strategy_bits();
        bits += self.count_coupling_exponent_bits();
        bits += self.count_aht_bits();
        for (blk, block) in self.blocks.iter().enumerate() {
            bits += self.count_audio_block_bits(blk, block);
        }
        bits
    }

    pub(crate) fn aht_eligible(&self, ch: usize) -> bool {
        for blk in 1..self.cfg.num_blocks {
            if !self.exp_strategy[ch][blk].is_reuse() {
                return false;
            }
            if ch == 0 && self.blocks[blk].new_cpl_strategy {
                return false;
            }
        }
        true
    }

    fn count_aht_bits(&self) -> i32 {
        if !self.aht_used {
            return 0;
        }
        let num_cpl_blocks = self.blocks.iter().filter(|block| block.cpl_in_use).count();
        let start = usize::from(num_cpl_blocks != self.cfg.num_blocks);
        (start..=self.cfg.channels)
            .filter(|&ch| self.aht_eligible(ch))
            .count() as i32
    }

    /// Flag bits that describe where coupling starts and stops.
    fn count_coupling_strategy_bits(&self) -> i32 {
        let cfg = &self.cfg;
        if cfg.channel_mode <= AC3_CHMODE_MONO {
            return 0;
        }
        let mut bits = 1;
        for block in &self.blocks[1..] {
            bits += 1;
            if block.new_cpl_strategy {
                bits += 1;
            }
        }
        bits
    }

    /// Exponent strategy bits for the coupling channel itself.
    fn count_coupling_exponent_bits(&self) -> i32 {
        if !self.cpl_on {
            return 0;
        }
        if self.use_frame_exp_strategy {
            return 5;
        }
        self.blocks
            .iter()
            .map(|block| 2 * i32::from(block.cpl_in_use))
            .sum()
    }

    fn count_audio_block_bits(&self, blk: usize, block: &Block) -> i32 {
        let cfg = &self.cfg;
        let mut bits = 0;

        if self.blk_sw_syntax {
            bits += cfg.fbw_channels as i32;
        }

        if block.new_cpl_strategy && block.cpl_in_use {
            bits += 1;
            if cfg.channel_mode != AC3_CHMODE_STEREO {
                bits += cfg.fbw_channels as i32;
            }
            if cfg.channel_mode == AC3_CHMODE_STEREO {
                bits += 1;
            }
            bits += 4 + 4 + 1;
        }

        if block.cpl_in_use {
            let mut coordinates_exist = false;
            for ch in 1..=cfg.fbw_channels {
                if !block.channel_in_cpl[ch] {
                    continue;
                }
                if !block.new_cpl_coords[ch].is_implicit() {
                    bits += 1;
                }
                if !block.new_cpl_coords[ch].is_reuse() {
                    coordinates_exist = true;
                    bits += 2 + (4 + 4) * cfg.num_cpl_bands as i32;
                }
            }
            if cfg.channel_mode == AC3_CHMODE_STEREO && self.phase_flags_in_use && coordinates_exist
            {
                bits += cfg.num_cpl_bands as i32;
            }
        }

        if cfg.channel_mode == AC3_CHMODE_STEREO {
            if blk > 0 {
                bits += 1;
            }
            if block.new_rematrixing_strategy {
                bits += block.num_rematrixing_bands as i32;
            }
        }

        for ch in 1..=cfg.fbw_channels {
            if !self.exp_strategy[ch][blk].is_reuse() {
                if !block.channel_in_cpl[ch] {
                    bits += 6;
                }
                bits += 2;
            }
            if blk == 0 && self.aht_channels[ch] {
                bits += 2; // gaq_mode
            }
        }

        if block.cpl_in_use {
            if !block.new_cpl_leak.is_implicit() {
                bits += 1;
            }
            if !block.new_cpl_leak.is_reuse() {
                bits += 3 + 3;
            }
        }

        bits
    }

    fn group_exponents(&mut self) {
        for blk in 0..self.cfg.num_blocks {
            let start_ch = usize::from(!self.blocks[blk].cpl_in_use);
            for ch in start_ch..=self.cfg.channels {
                let strategy = self.exp_strategy[ch][blk];
                if strategy.is_reuse() {
                    continue;
                }
                let cpl = ch == 0;
                let group_size = strategy.group_size();
                let num_groups = EXPONENT_GROUP_TAB[cpl as usize][strategy.table_index()]
                    [self.blocks[blk].end_freq[ch] - self.cfg.start_freq[ch]]
                    as usize;

                let block = &mut self.blocks[blk];
                let exponent = &block.exp[ch];
                let mut position = self.cfg.start_freq[ch] - usize::from(cpl);
                let mut previous = exponent[position];
                position += 1;
                block.grouped_exp[ch][0] = previous;

                for group in 1..=num_groups {
                    let mut deltas = [0i32; 3];
                    for delta in &mut deltas {
                        let next = exponent[position];
                        position += group_size;
                        *delta = next as i32 - previous as i32 + 2;
                        previous = next;
                    }
                    debug_assert!(deltas.iter().all(|&delta| (0..=4).contains(&delta)));
                    block.grouped_exp[ch][group] =
                        ((deltas[0] * 5 + deltas[1]) * 5 + deltas[2]) as u8;
                }
            }
        }
    }

    fn quantize_mantissas(&mut self) {
        for blk in 0..self.cfg.num_blocks {
            let (order, count) = coded_channel_order(&self.cfg, &self.blocks[blk]);
            let mut counters = MantCounts::default();
            let mut qmant = [0i16; CH * quantize::QSTRIDE];

            for &ch in &order[..count] {
                if self.aht_channels[ch] {
                    continue;
                }
                let reference = self.exp_ref_block[ch][blk] as usize;
                quantize::quantize_mantissas_blk_ch(
                    &mut counters,
                    ch,
                    quantize::MantissaBlock {
                        fixed_coef: &self.blocks[blk].fixed_coef[ch],
                        exp: &self.blocks[reference].exp[ch],
                        bap: &self.bap_work[blk][ch],
                        qmant: &mut qmant,
                        start_freq: self.cfg.start_freq[ch],
                        end_freq: self.blocks[blk].end_freq[ch],
                    },
                );
            }

            let block = &mut self.blocks[blk];
            for (ch, channel) in block.qmant.iter_mut().enumerate() {
                let base = ch * quantize::QSTRIDE;
                channel.copy_from_slice(&qmant[base..base + quantize::QSTRIDE]);
            }
            block.bap = self.bap_work[blk];
        }
    }
}

fn count_metadata_bits(meta: &ValidatedMetadata, cfg: &ValidatedConfig) -> i32 {
    let mut bits = 0;

    if meta.eac3_mixing_metadata {
        if cfg.channel_mode > AC3_CHMODE_STEREO {
            bits += 2;
        }
        if cfg.has_center {
            bits += 6;
        }
        if cfg.has_surround {
            bits += 6;
        }
        bits += i32::from(cfg.lfe_on);
        bits += 1 + 1 + 2;
        if cfg.channel_mode < AC3_CHMODE_STEREO {
            bits += 1;
        }
        bits += 1;
    }

    if meta.eac3_info_metadata {
        bits += 3 + 1 + 1;
        if cfg.channel_mode == AC3_CHMODE_STEREO {
            bits += 2 + 2;
        }
        if cfg.channel_mode >= AC3_CHMODE_2F2R {
            bits += 2;
        }
        bits += 1;
        if meta.audio_production_info {
            bits += 5 + 2 + 1;
        }
        bits += 1;
    }

    bits
}

fn count_frame_bits_fixed(cfg: &ValidatedConfig) -> i32 {
    // Fixed frame-bit count, evaluated once at construction where the frame
    // exponent strategy has not been selected yet.

    // Sync word and the fixed BSI fields, including the two metadata-presence
    // flags.
    let mut bits = 16 + 35 + 1 + 1;

    // Converter sync, additional bitstream info, frame-level exponent strategy
    // and AHT.
    if cfg.num_blocks != 6 {
        bits += 1;
    }
    bits += 1;
    if cfg.num_blocks == 6 {
        bits += 2;
    }

    // SNR offset strategy and the fixed syntax-disable flags.
    bits += 10;

    // Per-block exponent strategy fields for the full-bandwidth channels.
    bits += cfg.num_blocks as i32 * 2 * cfg.fbw_channels as i32;
    if cfg.lfe_on {
        bits += cfg.num_blocks as i32;
    }

    // Channel-map / frame-exponent-strategy payload.
    if cfg.num_blks_code != 3 {
        bits += 1;
    } else {
        bits += cfg.fbw_channels as i32 * 5;
    }

    // Coarse and fine SNR offsets.
    bits += 10;
    if cfg.num_blocks != 1 {
        bits += 1; // block start info
    }

    // Dynamic range, spectral extension and converter SNR offset per block.
    bits += 3 * cfg.num_blocks as i32;

    bits += 1; // auxiliary data
    bits += 1 + 16; // reserved bit and CRC2
    bits
}

/// Buffers arbitrary interleaved chunks and emits whole syncframes.
///
/// Interleaved input is in SMPTE/WAVE channel order for the configured layout
/// (the same order as [`ChannelOrder::Smpte`]).
pub struct StreamEncoder {
    encoder: FrameEncoder,
    channels: usize,
    pending: Vec<Vec<f32>>,
}

impl StreamEncoder {
    /// Enable or disable block switching.
    pub fn set_block_switch(&mut self, enabled: bool) {
        self.encoder.set_block_switch(enabled);
    }

    /// Enable or disable the experimental AHT path.
    pub fn set_aht_enabled(&mut self, enabled: bool) {
        self.encoder.set_aht_enabled(enabled);
    }

    /// Actual coded bandwidth cutoff in Hz.
    pub fn cutoff_hz(&self) -> f64 {
        self.encoder.cutoff_hz()
    }

    pub fn new(config: EncoderConfig) -> Result<Self, EncodeError> {
        let encoder = FrameEncoder::new(config)?;
        let channels = encoder.cfg.channels;
        Ok(Self {
            encoder,
            channels,
            pending: vec![Vec::new(); channels],
        })
    }

    /// Push interleaved samples.  Any number of whole sample frames is
    /// accepted; incomplete encoder frames remain buffered.
    pub fn push_interleaved(&mut self, pcm: &[f32]) -> Result<Vec<Frame>, EncodeError> {
        self.push_interleaved_with_metadata(pcm, |_| None)
    }

    /// Like [`StreamEncoder::push_interleaved`], but the callback supplies the
    /// metadata for each completed syncframe (called with the frame index).
    pub fn push_interleaved_with_metadata<F>(
        &mut self,
        pcm: &[f32],
        mut metadata_for: F,
    ) -> Result<Vec<Frame>, EncodeError>
    where
        F: FnMut(u64) -> Option<SkipMetadata>,
    {
        if !pcm.len().is_multiple_of(self.channels) {
            return Err(EncodeError::Pcm(format!(
                "interleaved length {} is not a multiple of {} channels",
                pcm.len(),
                self.channels
            )));
        }
        for frame in pcm.chunks_exact(self.channels) {
            for (pending, &sample) in self.pending.iter_mut().zip(frame) {
                pending.push(sample);
            }
        }
        self.drain_full_frames(&mut metadata_for)
    }

    /// Flush the final partial frame (zero padded) and return it.
    pub fn finish(&mut self) -> Result<Vec<Frame>, EncodeError> {
        self.finish_with_metadata(|_| None)
    }

    /// Like [`StreamEncoder::finish`], with per-frame metadata.
    pub fn finish_with_metadata<F>(
        &mut self,
        mut metadata_for: F,
    ) -> Result<Vec<Frame>, EncodeError>
    where
        F: FnMut(u64) -> Option<SkipMetadata>,
    {
        let mut out = Vec::new();
        if !self.pending[0].is_empty() {
            let index = self.encoder.frame_index;
            let slices: Vec<&[f32]> = self.pending.iter().map(|c| c.as_slice()).collect();
            let pcm = PlanarFrame::new(&slices, ChannelOrder::Smpte);
            out.push(
                self.encoder
                    .encode_frame_with_metadata(&pcm, metadata_for(index))?,
            );
            for ch in &mut self.pending {
                ch.clear();
            }
        }
        Ok(out)
    }

    fn drain_full_frames<F>(&mut self, metadata_for: &mut F) -> Result<Vec<Frame>, EncodeError>
    where
        F: FnMut(u64) -> Option<SkipMetadata>,
    {
        let spf = self.encoder.samples_per_frame();
        let mut out = Vec::new();
        while self.pending[0].len() >= spf {
            let index = self.encoder.frame_index;
            let slices: Vec<&[f32]> = self.pending.iter().map(|c| &c[..spf]).collect();
            let pcm = PlanarFrame::new(&slices, ChannelOrder::Smpte);
            out.push(
                self.encoder
                    .encode_frame_with_metadata(&pcm, metadata_for(index))?,
            );
            for ch in &mut self.pending {
                ch.drain(..spf);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Acmod, ChannelLayout, Coupling};

    fn validated(coupling: Coupling) -> ValidatedConfig {
        EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::ThreeTwo,
                lfe: true,
            },
            bitrate_bps: 448_000,
            dialnorm: -27,
            metadata: Default::default(),
            coupling,
            stereo_rematrixing: false,
            cutoff_hz: None,
            coupling_start: None,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn coupling_channel_is_written_after_the_first_coupled_channel() {
        let cfg = validated(Coupling::Off);
        let mut block = Block::new(&cfg);
        block.cpl_in_use = true;
        block.channel_in_cpl[1] = true;
        block.channel_in_cpl[2] = true;

        let (order, count) = coded_channel_order(&cfg, &block);
        assert_eq!(&order[..count], &[1, 0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn coupling_channel_follows_a_later_first_coupled_channel() {
        let cfg = validated(Coupling::Off);
        let mut block = Block::new(&cfg);
        block.cpl_in_use = true;
        block.channel_in_cpl[2] = true;
        block.channel_in_cpl[3] = true;

        let (order, count) = coded_channel_order(&cfg, &block);
        assert_eq!(&order[..count], &[1, 2, 0, 3, 4, 5, 6]);
    }

    #[test]
    fn without_coupling_channels_keep_natural_order() {
        let cfg = validated(Coupling::Off);
        let block = Block::new(&cfg);

        let (order, count) = coded_channel_order(&cfg, &block);
        assert_eq!(&order[..count], &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn transient_sets_short_block_flags() {
        let config = EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::Stereo,
                lfe: false,
            },
            bitrate_bps: 64_000,
            dialnorm: -27,
            metadata: Default::default(),
            coupling: Coupling::Off,
            stereo_rematrixing: false,
            cutoff_hz: None,
            coupling_start: None,
        };
        let mut encoder = FrameEncoder::new(config).unwrap();
        let silence = vec![0.0f32; encoder.samples_per_frame()];
        let silent_channels = [&silence[..], &silence[..]];
        encoder
            .encode_frame(&PlanarFrame::new(&silent_channels, ChannelOrder::Smpte))
            .unwrap();

        let mut transient = vec![0.0f32; encoder.samples_per_frame()];
        for (offset, sample) in transient.iter_mut().take(128).enumerate() {
            let envelope = (-(offset as f64) / 32.0).exp();
            let phase = 2.0 * std::f64::consts::PI * 6_000.0 * offset as f64 / 48_000.0;
            *sample = (0.8 * envelope * phase.sin()) as f32;
        }
        let transient_channels = [&transient[..], &transient[..]];
        encoder
            .encode_frame(&PlanarFrame::new(&transient_channels, ChannelOrder::Smpte))
            .unwrap();

        assert!(encoder.blk_sw_syntax);
        assert!(encoder.blksw[1][0]);
        assert!(encoder.blksw[2][0]);
        assert!(!encoder.blksw[1][1]);
        assert!(!encoder.blksw[2][1]);
    }

    fn metadata_frame(
        encoder: &mut FrameEncoder,
        freq: f64,
        transient: bool,
        metadata: Option<SkipMetadata>,
    ) -> Frame {
        let spf = encoder.samples_per_frame();
        let channels = encoder.cfg.channels;
        let data: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
                (0..spf)
                    .map(|i| {
                        let t = i as f64 / SAMPLE_RATE as f64;
                        let mut value = (0.2
                            * (2.0 * std::f64::consts::PI * (freq + ch as f64 * 97.0) * t).sin())
                            as f32;
                        if transient && i < 128 {
                            let envelope = (-(i as f64) / 32.0).exp();
                            value += (0.7 * envelope) as f32;
                        }
                        value.clamp(-1.0, 1.0)
                    })
                    .collect()
            })
            .collect();
        let slices: Vec<&[f32]> = data.iter().map(|channel| channel.as_slice()).collect();
        encoder
            .encode_frame_with_metadata(&PlanarFrame::new(&slices, ChannelOrder::Smpte), metadata)
            .unwrap()
    }

    fn bits_of(bytes: &[u8]) -> Vec<u8> {
        let mut bits = Vec::with_capacity(bytes.len() * 8);
        for byte in bytes {
            for shift in (0..8).rev() {
                bits.push((byte >> shift) & 1);
            }
        }
        bits
    }

    fn read_bits(bits: &[u8], position: usize, count: usize) -> u32 {
        assert!(position + count <= bits.len());
        bits[position..position + count]
            .iter()
            .fold(0u32, |value, &bit| (value << 1) | u32::from(bit))
    }

    fn check_metadata_frame(frame: &Frame, metadata: &SkipMetadata) {
        let bits = bits_of(&frame.bytes);
        assert_eq!(read_bits(&bits, 0, 16), 0x0B77, "sync word");
        let frmsiz = read_bits(&bits, 21, 11) as usize;
        assert_eq!(frmsiz + 1, frame.bytes.len() / 2, "frmsiz matches bytes");
        assert_eq!(frame.edits.frame_size, frame.bytes.len());

        let end = frame.bytes.len();
        let stored = u16::from_be_bytes([frame.bytes[end - 2], frame.bytes[end - 1]]);
        assert_eq!(stored, crate::pack::crc16(&frame.bytes[2..end - 2]));

        let addbsi_bit = frame.edits.addbsie_bit;
        assert_eq!(
            read_bits(&bits, addbsi_bit, 1),
            u32::from(!metadata.addbsi.is_empty()),
            "addbsie"
        );
        if !metadata.addbsi.is_empty() {
            let len = read_bits(&bits, addbsi_bit + 1, 6) as usize + 1;
            assert_eq!(len, metadata.addbsi.len());
            for (index, &byte) in metadata.addbsi.iter().enumerate() {
                assert_eq!(read_bits(&bits, addbsi_bit + 7 + index * 8, 8) as u8, byte);
            }
        }

        let skip_enabled = !metadata.emdf.is_empty();
        assert_eq!(
            read_bits(&bits, frame.edits.skipfld_bit, 1),
            u32::from(skip_enabled),
            "skipFieldSyntaxEnabled"
        );
        assert_eq!(frame.edits.block_skip_points.len(), frame.edits.num_blocks);
        assert!(
            frame
                .edits
                .block_skip_points
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "skip points must increase"
        );
        if skip_enabled {
            assert_eq!(
                frame.edits.emdf_start_bit,
                Some(frame.edits.block_skip_points[metadata.target_block] + 10)
            );
            for (block, &point) in frame.edits.block_skip_points.iter().enumerate() {
                if block == metadata.target_block {
                    assert_eq!(read_bits(&bits, point, 1), 1, "target skiple");
                    assert_eq!(
                        read_bits(&bits, point + 1, 9) as usize,
                        metadata.emdf.len(),
                        "skipl"
                    );
                    for (index, &byte) in metadata.emdf.iter().enumerate() {
                        assert_eq!(
                            read_bits(&bits, point + 10 + index * 8, 8) as u8,
                            byte,
                            "emdf byte {index}"
                        );
                    }
                } else {
                    assert_eq!(read_bits(&bits, point, 1), 0, "empty skiple");
                }
            }
        }
    }

    fn config(bitrate_bps: u32, coupling: Coupling) -> EncoderConfig {
        EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::ThreeTwo,
                lfe: true,
            },
            bitrate_bps,
            dialnorm: -27,
            metadata: Default::default(),
            coupling,
            stereo_rematrixing: true,
            cutoff_hz: None,
            coupling_start: None,
        }
    }

    fn metadata_at(target_block: usize) -> SkipMetadata {
        SkipMetadata::new(
            &[0x01, 0x10],
            &[0x58, 0x38, 0xAA, 0x55, 0x01, 0x02, 0x03],
            target_block,
        )
    }

    fn metadata() -> SkipMetadata {
        metadata_at(2)
    }

    #[test]
    fn metadata_is_written_into_the_skip_field() {
        let mut encoder = FrameEncoder::new(config(693_000, Coupling::Off)).unwrap();
        let meta = metadata();
        let frame = metadata_frame(&mut encoder, 440.0, false, Some(meta.clone()));
        assert_eq!(frame.edits.num_blocks, 6);
        check_metadata_frame(&frame, &meta);
    }

    #[test]
    fn metadata_grows_the_core_frame_by_the_overhead() {
        let cfg = config(693_000, Coupling::Off);
        let meta = metadata();
        let mut plain_encoder = FrameEncoder::new(cfg.clone()).unwrap();
        let mut tagged_encoder = FrameEncoder::new(cfg).unwrap();
        let plain = metadata_frame(&mut plain_encoder, 440.0, false, None);
        let tagged = metadata_frame(&mut tagged_encoder, 440.0, false, Some(meta.clone()));

        let overhead = 6 + meta.addbsi.len() * 8 + plain.edits.num_blocks + 9 + meta.emdf.len() * 8;
        let mut expected = (plain.bytes.len() * 8 + overhead).div_ceil(8);
        if !expected.is_multiple_of(2) {
            expected += 1;
        }
        assert_eq!(tagged.bytes.len(), expected);
        assert_eq!(plain.bytes.len(), plain.edits.frame_size);

        // Two-pass injection must produce the same final size as the one-pass
        // writer for the same core.
        let injected = crate::inject::inject_metadata(&plain, &meta).unwrap();
        assert_eq!(injected.bytes.len(), tagged.bytes.len());
    }

    #[test]
    fn metadata_works_with_coupling() {
        let mut encoder = FrameEncoder::new(config(256_000, Coupling::Auto)).unwrap();
        let meta = metadata();
        let frame = metadata_frame(&mut encoder, 440.0, false, Some(meta.clone()));
        assert!(encoder.cpl_on, "test must exercise coupling");
        check_metadata_frame(&frame, &meta);
    }

    #[test]
    fn metadata_works_with_block_switching() {
        let mut encoder = FrameEncoder::new(config(256_000, Coupling::Off)).unwrap();
        // A silent lead-in keeps the block-switch detector from firing on
        // history; the transient frame then sets blksw syntax.
        let _ = metadata_frame(&mut encoder, 440.0, false, None);
        let meta = metadata();
        let frame = metadata_frame(&mut encoder, 440.0, true, Some(meta.clone()));
        assert!(encoder.blk_sw_syntax, "test must exercise block switching");
        check_metadata_frame(&frame, &meta);
    }

    #[test]
    fn metadata_works_for_every_block_count() {
        // >1.024 Mbps selects 3 blocks, >2.048 Mbps selects 2, >4.096 Mbps
        // selects 1 in `select_block_code`.
        for (bitrate, expected_blocks) in
            [(693_000, 6), (1_500_000, 3), (3_000_000, 2), (5_000_000, 1)]
        {
            let mut encoder = FrameEncoder::new(config(bitrate, Coupling::Off)).unwrap();
            assert_eq!(encoder.cfg.num_blocks, expected_blocks);
            let meta = metadata_at(expected_blocks - 1);
            let frame = metadata_frame(&mut encoder, 1_000.0, false, Some(meta.clone()));
            assert_eq!(frame.edits.num_blocks, expected_blocks);
            check_metadata_frame(&frame, &meta);
        }
    }

    #[test]
    fn metadata_works_with_stereo_rematrixing() {
        let mut cfg = config(192_000, Coupling::Off);
        cfg.layout = ChannelLayout {
            acmod: Acmod::Stereo,
            lfe: false,
        };
        cfg.stereo_rematrixing = true;
        let mut encoder = FrameEncoder::new(cfg).unwrap();
        assert!(encoder.cfg.rematrixing_enabled);
        let meta = metadata();
        let frame = metadata_frame(&mut encoder, 440.0, false, Some(meta.clone()));
        check_metadata_frame(&frame, &meta);
    }

    #[test]
    fn metadata_works_with_aht() {
        let mut encoder = FrameEncoder::new(config(256_000, Coupling::Off)).unwrap();
        encoder.set_aht_enabled(true);
        encoder.set_block_switch(false);
        let meta = metadata();
        let _ = metadata_frame(&mut encoder, 20.0, false, None);
        let frame = metadata_frame(&mut encoder, 20.0, false, Some(meta.clone()));
        assert!(encoder.aht_used, "test must exercise AHT");
        check_metadata_frame(&frame, &meta);
    }

    #[test]
    fn metadata_works_without_addbsi_or_emdf() {
        let mut encoder = FrameEncoder::new(config(448_000, Coupling::Off)).unwrap();
        let empty = SkipMetadata::new(&[], &[], 0);
        let frame = metadata_frame(&mut encoder, 440.0, false, Some(empty.clone()));
        check_metadata_frame(&frame, &empty);
        // A truly metadata-free frame still reports usable edit positions.
        let plain = metadata_frame(&mut encoder, 440.0, false, None);
        assert_eq!(
            read_bits(&bits_of(&plain.bytes), plain.edits.addbsie_bit, 1),
            0
        );
        assert_eq!(
            read_bits(&bits_of(&plain.bytes), plain.edits.skipfld_bit, 1),
            0
        );
    }

    #[test]
    fn metadata_is_validated() {
        let mut encoder = FrameEncoder::new(config(448_000, Coupling::Off)).unwrap();
        let too_many_addbsi = SkipMetadata::new(&[0; 65], &[], 0);
        let too_large_emdf = SkipMetadata::new(&[], &[0; 512], 0);
        let bad_block = SkipMetadata::new(&[], &[0; 4], 6);
        let silence = vec![0.0f32; encoder.samples_per_frame()];
        let channels = vec![&silence[..]; encoder.cfg.channels];
        let pcm = PlanarFrame::new(&channels, ChannelOrder::Smpte);
        for meta in [too_many_addbsi, too_large_emdf, bad_block] {
            assert!(matches!(
                encoder.encode_frame_with_metadata(&pcm, Some(meta)),
                Err(EncodeError::Config(_))
            ));
        }
    }

    #[test]
    fn stream_encoder_forwards_per_frame_metadata() {
        let mut encoder = StreamEncoder::new(config(693_000, Coupling::Off)).unwrap();
        let spf = encoder.encoder.samples_per_frame();
        let channels = encoder.channels;
        let samples: Vec<f32> = (0..spf)
            .flat_map(|i| {
                (0..channels).map(move |ch| {
                    (0.2 * (2.0 * std::f64::consts::PI * (440.0 + ch as f64 * 100.0) * i as f64
                        / SAMPLE_RATE as f64)
                        .sin()) as f32
                })
            })
            .collect();
        let meta = metadata();
        let frames = encoder
            .push_interleaved_with_metadata(&samples, |_| Some(meta.clone()))
            .unwrap();
        assert_eq!(frames.len(), 1);
        check_metadata_frame(&frames[0], &meta);
    }

    #[test]
    fn coded_channel_order_matches_the_reference_loop() {
        let cfg = validated(Coupling::Off);
        for cpl_in_use in [false, true] {
            for mask in 0u8..(1 << cfg.fbw_channels) {
                let mut block = Block::new(&cfg);
                block.cpl_in_use = cpl_in_use;
                for ch in 1..=cfg.fbw_channels {
                    block.channel_in_cpl[ch] = mask & (1 << (ch - 1)) != 0;
                }

                let (order, count) = coded_channel_order(&cfg, &block);
                assert_eq!(&order[..count], reference_channel_order(&cfg, &block));
            }
        }
    }

    /// The original FFmpeg-style loop, kept as an oracle for the helper.
    fn reference_channel_order(cfg: &ValidatedConfig, block: &Block) -> Vec<usize> {
        let mut order = Vec::new();
        let mut got_cpl = !block.cpl_in_use;
        let mut ch = 1usize;
        let mut previous = 0usize;
        while ch <= cfg.channels {
            if !got_cpl && ch > 1 && block.channel_in_cpl[ch - 1] {
                previous = ch - 1;
                ch = 0;
                got_cpl = true;
            }
            order.push(ch);
            if ch == 0 {
                ch = previous;
            }
            ch += 1;
        }
        order
    }
}
