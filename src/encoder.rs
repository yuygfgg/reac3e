//! Frame/stream encoder orchestration.

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
    AC3_MAX_CHANNELS, AC3_MAX_COEFS, EAC3_FRM_EXPSTR, FAST_GAIN_TAB,
};

/// Internal channel slots: coupling channel 0, fullband 1..=fbw, LFE last.
pub(crate) const CH: usize = AC3_MAX_CHANNELS + 1;

const EXP_DIFF_THRESHOLD: i32 = 500;

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

/// One encoded syncframe.
#[derive(Debug, Clone)]
pub struct Frame {
    pub bytes: Vec<u8>,
    pub index: u64,
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
    budget: FrameBudget,
    pub(crate) frame_index: u64,
    final_frame_seen: bool,
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
            budget: FrameBudget::default(),
            frame_index: 0,
            final_frame_seen: false,
        })
    }

    /// Samples represented by one syncframe (1536 for six blocks).
    pub fn samples_per_frame(&self) -> usize {
        AC3_BLOCK_SIZE * self.cfg.num_blocks
    }

    /// Encoder delay introduced by the initial 256-sample overlap window,
    /// matching the initial overlap padding.
    pub fn latency(&self) -> usize {
        AC3_BLOCK_SIZE
    }

    /// Encode one frame.  `pcm` must contain exactly `samples_per_frame()`
    /// samples per channel; a shorter final frame is accepted and zero padded.
    pub fn encode_frame(&mut self, pcm: &PlanarFrame<'_>) -> Result<Frame, EncodeError> {
        if self.final_frame_seen {
            return Err(EncodeError::Pcm(
                "cannot encode another frame after a short final frame".into(),
            ));
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
        self.encode_internal(&channels, len)
    }

    /// Flush the final partial frame.  `encode_frame` already zero-pads a
    /// short final frame, so there is no buffered audio left; this always
    /// returns `None` and is provided for API symmetry and idempotence.
    pub fn flush(&mut self) -> Result<Option<Frame>, EncodeError> {
        Ok(None)
    }

    fn encode_internal(&mut self, channels: &[&[f32]], valid: usize) -> Result<Frame, EncodeError> {
        let spf = self.samples_per_frame();
        self.adjust_frame_size();

        // --- apply MDCT ---
        let num_blocks = self.cfg.num_blocks;
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
            let mut windowed = [0.0f32; AC3_BLOCK_SIZE * 2];
            for blk in 0..num_blocks {
                let start = blk * AC3_BLOCK_SIZE;
                self.mdct
                    .window_block(&input[start..start + AC3_BLOCK_SIZE * 2], &mut windowed);
                self.mdct
                    .process(&windowed, &mut self.blocks[blk].mdct_coef[ch]);
            }
            self.tail[ch].copy_from_slice(&input[spf..spf + AC3_BLOCK_SIZE]);
        }

        // --- coupling strategy + coupling channel ---
        self.cpl_on =
            coupling::compute_coupling_strategy(&self.cfg, &mut self.blocks, self.cfg.cpl_enabled);
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
        self.encode_exponents();

        // --- bit allocation ---
        self.compute_bit_allocation()?;

        // --- grouped exponents, quantisation, packing ---
        self.group_exponents();
        self.quantize_mantissas();

        let mut w = BitWriter::with_capacity(self.frame_size);
        header::write_frame_header(self, &mut w);
        for blk in 0..self.cfg.num_blocks {
            pack::write_audio_block(self, blk, &mut w);
        }
        let bytes = pack::finalize(w.into_bytes(), self.frame_size)?;

        let frame = Frame {
            bytes,
            index: self.frame_index,
        };
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
                alloc::calc_bap(
                    &block.mask[ch],
                    &block.psd[ch],
                    self.cfg.start_freq[ch],
                    block.end_freq[ch],
                    snr_offset,
                    floor,
                    &mut self.bap_work[blk][ch],
                );
            }
        }

        // Reuse blocks point at the reference block's allocation.
        for ch in start_ch..=self.cfg.channels {
            for (blk, block) in self.blocks.iter().enumerate() {
                if ch == 0 && !block.cpl_in_use {
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
        for (blk, block) in self.blocks.iter().enumerate() {
            for ch in start_ch..=self.cfg.channels {
                if ch == 0 && !block.cpl_in_use {
                    continue;
                }
                let start = self.cfg.start_freq[ch];
                let end = max_end_freq.min(block.end_freq[ch]);
                let len = end - start;
                alloc::update_bap_counts(&mut mant_cnt[blk], &self.bap_work[blk][ch][start..], len);
            }
        }
        alloc::compute_mantissa_size(&mant_cnt)
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
        for (blk, block) in self.blocks.iter().enumerate() {
            bits += self.count_audio_block_bits(blk, block);
        }
        bits
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
        self.drain_full_frames()
    }

    /// Flush the final partial frame (zero padded) and return it.
    pub fn finish(&mut self) -> Result<Vec<Frame>, EncodeError> {
        let mut out = Vec::new();
        if !self.pending[0].is_empty() {
            let slices: Vec<&[f32]> = self.pending.iter().map(|c| c.as_slice()).collect();
            let pcm = PlanarFrame::new(&slices, ChannelOrder::Smpte);
            out.push(self.encoder.encode_frame(&pcm)?);
            for ch in &mut self.pending {
                ch.clear();
            }
        }
        Ok(out)
    }

    fn drain_full_frames(&mut self) -> Result<Vec<Frame>, EncodeError> {
        let spf = self.encoder.samples_per_frame();
        let mut out = Vec::new();
        while self.pending[0].len() >= spf {
            let slices: Vec<&[f32]> = self.pending.iter().map(|c| &c[..spf]).collect();
            let pcm = PlanarFrame::new(&slices, ChannelOrder::Smpte);
            out.push(self.encoder.encode_frame(&pcm)?);
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
