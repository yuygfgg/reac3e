//! Channel coupling.

use crate::analysis::extract_exponents;
use crate::config::ValidatedConfig;
use crate::dsp::sum_square_butterfly;
use crate::encoder::{Block, CH, CplFlag};
use crate::quantize::{COEF_MAX, clip_coefficients, float_to_fixed24};
use crate::tables::{AC3_CHMODE_STEREO, AC3_MAX_BLOCKS};

const NEW_CPL_COORD_THRESHOLD: f32 = 0.03;
const PHASE_BAND_ENERGY_DENOMINATOR: f32 = (1 << 14) as f32;
const PHASE_DIFF_ENERGY_FACTOR: f32 = (1 << 4) as f32;
const MAX_CPL_BANDS: usize = 16;
const PHASE_BANDS: usize = 18;

type CouplingCoordinates = [[[f32; MAX_CPL_BANDS]; CH]; AC3_MAX_BLOCKS];
type BandEnergies = [[[f32; MAX_CPL_BANDS]; CH]; AC3_MAX_BLOCKS];

fn calc_cpl_coord(energy_ch: f32, energy_cpl: f32) -> f32 {
    let mut coord = 0.125f32;
    if energy_cpl > 0.0 {
        coord *= (energy_ch / energy_cpl).sqrt();
    }
    coord.min(COEF_MAX)
}

/// Set per-block coupling flags and channel end frequencies.
/// Returns whether coupling is active for the frame.
pub(crate) fn compute_coupling_strategy(
    cfg: &ValidatedConfig,
    blocks: &mut [Block],
    cpl_enabled: bool,
) -> bool {
    for block in blocks.iter_mut() {
        for ch in 1..=cfg.fbw_channels {
            block.channel_in_cpl[ch] = cpl_enabled;
        }
    }

    let mut got_cpl_snr = false;
    let mut num_cpl_blocks = 0usize;
    for blk in 0..blocks.len() {
        let prev_cpl = blk
            .checked_sub(1)
            .map(|p| (blocks[p].cpl_in_use, blocks[p].channel_in_cpl));
        let block = &mut blocks[blk];
        let mut num = 0usize;
        for ch in 1..=cfg.fbw_channels {
            num += usize::from(block.channel_in_cpl[ch]);
        }
        block.num_cpl_channels = num;
        block.cpl_in_use = num > 1;
        num_cpl_blocks += usize::from(block.cpl_in_use);
        if !block.cpl_in_use {
            block.num_cpl_channels = 0;
            for ch in 1..=cfg.fbw_channels {
                block.channel_in_cpl[ch] = false;
            }
        }

        block.new_cpl_strategy = blk == 0;
        if let Some((_, prev_channels)) = prev_cpl {
            let channels_changed =
                (1..=cfg.fbw_channels).any(|ch| block.channel_in_cpl[ch] != prev_channels[ch]);
            if channels_changed {
                block.new_cpl_strategy = true;
            }
        }
        block.new_cpl_leak = CplFlag::from_new(block.new_cpl_strategy);

        if blk == 0 || (block.cpl_in_use && !got_cpl_snr) {
            block.new_snr_offsets = true;
            if block.cpl_in_use {
                got_cpl_snr = true;
            }
        } else {
            block.new_snr_offsets = false;
        }
    }

    let cpl_on = cpl_enabled && num_cpl_blocks > 0;

    for block in blocks.iter_mut() {
        for ch in 1..=cfg.fbw_channels {
            block.end_freq[ch] = if block.channel_in_cpl[ch] {
                cfg.cpl_start_freq
            } else {
                cfg.bandwidth_code * 3 + 73
            };
        }
    }
    cpl_on
}

/// Detect the stereo phase flags and pre-seed the per-band channel energies.
fn compute_stereo_phase_flags(
    cfg: &ValidatedConfig,
    blocks: &[Block],
    energies: &mut BandEnergies,
    phase_flags_out: &mut [u8; PHASE_BANDS],
    phase_flags_in_use: &mut bool,
) {
    if cfg.channel_mode != AC3_CHMODE_STEREO {
        return;
    }

    let mut phase_flags = [1u8; PHASE_BANDS];
    let mut cpl_blocks = 0usize;

    for (blk, block) in blocks.iter().enumerate() {
        if !block.cpl_in_use {
            continue;
        }
        cpl_blocks += 1;

        let prefix =
            sum_square_butterfly(&block.mdct_coef[1], &block.mdct_coef[2], cfg.cpl_start_freq);
        let mut block_energy = [prefix[0], prefix[1], 0.0f32, 0.0f32];
        let mut band_sums = [[0.0f32; 4]; PHASE_BANDS];
        let mut start = cfg.cpl_start_freq;

        for bnd in 0..cfg.num_cpl_bands {
            let size = cfg.cpl_band_sizes[bnd];
            let sums = sum_square_butterfly(
                &block.mdct_coef[1][start..],
                &block.mdct_coef[2][start..],
                size,
            );
            band_sums[bnd] = sums;
            energies[blk][1][bnd] = sums[0];
            energies[blk][2][bnd] = sums[1];
            block_energy[0] += sums[0];
            block_energy[1] += sums[1];
            start += size;
        }

        let max_energy = block_energy[0].max(block_energy[1]);
        for (bnd, &sums) in band_sums[..cfg.num_cpl_bands].iter().enumerate() {
            let band_energy = sums[0].max(sums[1]);
            let significant = band_energy > max_energy / PHASE_BAND_ENERGY_DENOMINATOR;
            let difference_wins = sums[3] / PHASE_DIFF_ENERGY_FACTOR > sums[2];
            if !(significant && difference_wins) {
                phase_flags[bnd] = 0;
            }
        }
    }

    for bnd in 0..cfg.num_cpl_bands {
        let phase = phase_flags[bnd] != 0 && cpl_blocks > 0;
        phase_flags_out[bnd] = u8::from(phase);
        *phase_flags_in_use |= phase;
    }
}

/// Sum the coupled channels into the coupling channel (index 0).
fn build_coupling_channel(
    cfg: &ValidatedConfig,
    blocks: &mut [Block],
    phase_flags: &[u8; PHASE_BANDS],
    cpl_start: usize,
    num_cpl_coefs: usize,
) {
    for block in blocks.iter_mut() {
        if !block.cpl_in_use {
            continue;
        }

        block.mdct_coef[0][cpl_start..cpl_start + num_cpl_coefs].fill(0.0);
        for ch in 1..=cfg.fbw_channels {
            if !block.channel_in_cpl[ch] {
                continue;
            }
            for i in 0..num_cpl_coefs {
                block.mdct_coef[0][cpl_start + i] += block.mdct_coef[ch][cpl_start + i];
            }
        }

        if cfg.channel_mode == AC3_CHMODE_STEREO {
            let mut start = cfg.cpl_start_freq;
            for (bnd, &band_size) in cfg.cpl_band_sizes[..cfg.num_cpl_bands].iter().enumerate() {
                if phase_flags[bnd] != 0 {
                    for j in 0..band_size {
                        block.mdct_coef[0][start + j] =
                            block.mdct_coef[1][start + j] - block.mdct_coef[2][start + j];
                    }
                }
                start += band_size;
            }
        }

        clip_coefficients(&mut block.mdct_coef[0][cpl_start..], num_cpl_coefs);
    }
}

/// Accumulate per-band energy for the coupling channel and each coupled
/// channel.  Stereo energies were pre-seeded by the phase detector.
fn accumulate_band_energies(cfg: &ValidatedConfig, blocks: &[Block], energies: &mut BandEnergies) {
    let last_ch = if cfg.channel_mode == AC3_CHMODE_STEREO {
        0
    } else {
        cfg.fbw_channels
    };

    let mut bnd = 0usize;
    let mut start = cfg.cpl_start_freq;
    while start < cfg.cpl_end_freq {
        let band_size = cfg.cpl_band_sizes[bnd];
        for (blk, block) in blocks.iter().enumerate() {
            if !block.cpl_in_use {
                continue;
            }
            for (ch, energy) in energies[blk].iter_mut().enumerate().take(last_ch + 1) {
                if ch > 0 && !block.channel_in_cpl[ch] {
                    continue;
                }
                for j in 0..band_size {
                    let value = block.mdct_coef[ch][start + j];
                    energy[bnd] = value.mul_add(value, energy[bnd]);
                }
            }
        }
        start += band_size;
        bnd += 1;
    }
}

/// Per-block coupling coordinates from the band energies.
fn compute_coupling_coordinates(
    cfg: &ValidatedConfig,
    blocks: &[Block],
    energies: &BandEnergies,
    coordinates: &mut CouplingCoordinates,
) {
    for (blk, block) in blocks.iter().enumerate() {
        if !block.cpl_in_use {
            continue;
        }
        for (ch, coordinate) in coordinates[blk]
            .iter_mut()
            .enumerate()
            .take(cfg.fbw_channels + 1)
            .skip(1)
        {
            if !block.channel_in_cpl[ch] {
                continue;
            }
            for (bnd, value) in coordinate.iter_mut().enumerate().take(cfg.num_cpl_bands) {
                *value = calc_cpl_coord(energies[blk][ch][bnd], energies[blk][0][bnd]);
            }
        }
    }
}

/// Decide in which blocks new coupling coordinates are transmitted.
fn decide_coordinate_updates(
    cfg: &ValidatedConfig,
    blocks: &mut [Block],
    coordinates: &CouplingCoordinates,
) {
    for blk in 0..blocks.len() {
        blocks[blk].new_cpl_coords = [CplFlag::Reuse; CH];
        if !blocks[blk].cpl_in_use {
            continue;
        }

        let first_block = blk == 0 || !blocks[blk - 1].cpl_in_use;
        if first_block {
            for ch in 1..=cfg.fbw_channels {
                blocks[blk].new_cpl_coords[ch] = CplFlag::New;
            }
            continue;
        }

        for (ch, coordinate) in coordinates[blk]
            .iter()
            .enumerate()
            .take(cfg.fbw_channels + 1)
            .skip(1)
        {
            if !blocks[blk].channel_in_cpl[ch] {
                continue;
            }
            if !blocks[blk - 1].channel_in_cpl[ch] {
                blocks[blk].new_cpl_coords[ch] = CplFlag::New;
                continue;
            }
            let difference: f32 = coordinates[blk - 1][ch]
                .iter()
                .zip(coordinate)
                .take(cfg.num_cpl_bands)
                .map(|(&previous, &current)| (previous - current).abs())
                .sum::<f32>()
                / cfg.num_cpl_bands as f32;
            if difference > NEW_CPL_COORD_THRESHOLD {
                blocks[blk].new_cpl_coords[ch] = CplFlag::New;
            }
        }
    }
}

/// Fold reused coordinate runs into the block that transmits them.
fn summarize_reused_coordinates(
    cfg: &ValidatedConfig,
    blocks: &[Block],
    energies: &BandEnergies,
    coordinates: &mut CouplingCoordinates,
) {
    for bnd in 0..cfg.num_cpl_bands {
        let mut blk = 0usize;
        while blk < blocks.len() {
            if !blocks[blk].cpl_in_use {
                blk += 1;
                continue;
            }

            let mut next = blk + 1;
            for ch in 1..=cfg.fbw_channels {
                if !blocks[blk].channel_in_cpl[ch] {
                    continue;
                }
                let mut energy_cpl = energies[blk][0][bnd];
                let mut energy_ch = energies[blk][ch][bnd];
                next = blk + 1;
                while next < blocks.len() && blocks[next].new_cpl_coords[ch].is_reuse() {
                    if blocks[next].cpl_in_use {
                        energy_cpl += energies[next][0][bnd];
                        energy_ch += energies[next][ch][bnd];
                    }
                    next += 1;
                }
                coordinates[blk][ch][bnd] = calc_cpl_coord(energy_ch, energy_cpl);
            }
            blk = next;
        }
    }
}

/// Quantise the coupling coordinate exponents and mantissas.
fn quantize_coupling_coordinates(
    cfg: &ValidatedConfig,
    blocks: &mut [Block],
    coordinates: &CouplingCoordinates,
) {
    for blk in 0..blocks.len() {
        if !blocks[blk].cpl_in_use {
            continue;
        }

        let width = cfg.fbw_channels * MAX_CPL_BANDS;
        let mut coords_flat = [0.0f32; 128];
        let mut fixed_flat = [0i32; 128];
        for (ch, coordinate) in coordinates[blk]
            .iter()
            .enumerate()
            .take(cfg.fbw_channels + 1)
            .skip(1)
        {
            let base = (ch - 1) * MAX_CPL_BANDS;
            coords_flat[base..base + MAX_CPL_BANDS].copy_from_slice(coordinate);
        }
        float_to_fixed24(&mut fixed_flat[..width], &coords_flat[..width]);
        let mut exponents = [0u8; 128];
        extract_exponents(&fixed_flat[..width], &mut exponents[..width], width);

        for ch in 1..=cfg.fbw_channels {
            if blocks[blk].new_cpl_coords[ch].is_reuse() {
                continue;
            }
            let base = (ch - 1) * MAX_CPL_BANDS;
            let mut min_exp = exponents[base];
            let mut max_exp = exponents[base];
            for bnd in 1..cfg.num_cpl_bands {
                min_exp = min_exp.min(exponents[base + bnd]);
                max_exp = max_exp.max(exponents[base + bnd]);
            }

            let mut master_exp = ((max_exp as i32 - 15) + 2) / 3;
            master_exp = master_exp.max(0);
            while (min_exp as i32) < master_exp * 3 {
                master_exp -= 1;
            }

            for bnd in 0..cfg.num_cpl_bands {
                let exponent = (exponents[base + bnd] as i32 - master_exp * 3).clamp(0, 15);
                blocks[blk].cpl_coord_exp[ch][bnd] = exponent as u8;
            }
            blocks[blk].cpl_master_exp[ch] = master_exp as u8;

            for bnd in 0..cfg.num_cpl_bands {
                let cpl_exp = blocks[blk].cpl_coord_exp[ch][bnd] as i32;
                let shift = (5 + cpl_exp + master_exp * 3) as u32;
                let mut mantissa = fixed_flat[base + bnd].wrapping_shl(shift) >> 24;
                if cpl_exp == 15 {
                    mantissa >>= 1;
                } else {
                    mantissa -= 16;
                }
                blocks[blk].cpl_coord_mant[ch][bnd] = mantissa as u8;
            }
        }
    }
}

/// Mark the flags that are implicit in the first coupled block.
fn mark_implicit_flags(cfg: &ValidatedConfig, blocks: &mut [Block]) {
    let mut first_cpl_coords = [true; CH];
    for block in blocks.iter_mut() {
        for (ch, first) in first_cpl_coords
            .iter_mut()
            .enumerate()
            .take(cfg.fbw_channels + 1)
            .skip(1)
        {
            if block.channel_in_cpl[ch] {
                if *first {
                    block.new_cpl_coords[ch] = CplFlag::Implicit;
                    *first = false;
                }
            } else {
                *first = true;
            }
        }
    }

    for block in blocks.iter_mut() {
        if block.cpl_in_use {
            block.new_cpl_leak = CplFlag::Implicit;
            break;
        }
    }
}

pub(crate) fn apply_channel_coupling(
    cfg: &ValidatedConfig,
    blocks: &mut [Block],
    phase_flags_out: &mut [u8; PHASE_BANDS],
    phase_flags_in_use: &mut bool,
) {
    let mut coordinates = [[[0.0f32; MAX_CPL_BANDS]; CH]; AC3_MAX_BLOCKS];
    let mut energies = [[[0.0f32; MAX_CPL_BANDS]; CH]; AC3_MAX_BLOCKS];
    *phase_flags_in_use = false;

    let cpl_start_estimate = cfg.cpl_start_freq as isize - 1;
    let num_cpl_coefs = (cfg.num_cpl_subbands * 12 + 1).div_ceil(32) * 32;
    let cpl_start = (256.min(cpl_start_estimate + num_cpl_coefs as isize) - num_cpl_coefs as isize)
        .max(0) as usize;

    compute_stereo_phase_flags(
        cfg,
        blocks,
        &mut energies,
        phase_flags_out,
        phase_flags_in_use,
    );
    build_coupling_channel(cfg, blocks, phase_flags_out, cpl_start, num_cpl_coefs);
    accumulate_band_energies(cfg, blocks, &mut energies);
    compute_coupling_coordinates(cfg, blocks, &energies, &mut coordinates);
    decide_coordinate_updates(cfg, blocks, &coordinates);
    summarize_reused_coordinates(cfg, blocks, &energies, &mut coordinates);
    quantize_coupling_coordinates(cfg, blocks, &coordinates);
    mark_implicit_flags(cfg, blocks);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Acmod, ChannelLayout, Coupling, EncoderConfig};

    fn cfg() -> ValidatedConfig {
        EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::ThreeTwo,
                lfe: true,
            },
            bitrate_bps: 448_000,
            dialnorm: -27,
            metadata: Default::default(),
            coupling: Coupling::Auto,
            stereo_rematrixing: false,
            cutoff_hz: None,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn coupling_coordinates_follow_the_energy_ratio() {
        assert_eq!(calc_cpl_coord(1.0, 0.0), 0.125);
        assert!((calc_cpl_coord(4.0, 1.0) - 0.25).abs() < 1e-7);
        assert_eq!(calc_cpl_coord(1.0e30, 1.0), COEF_MAX);
    }

    #[test]
    fn coupling_strategy_requires_at_least_two_channels() {
        let cfg = cfg();
        let mut blocks = vec![Block::new(&cfg), Block::new(&cfg)];

        assert!(compute_coupling_strategy(&cfg, &mut blocks, true));
        assert!(
            blocks
                .iter()
                .all(|block| block.cpl_in_use && block.num_cpl_channels == cfg.fbw_channels)
        );

        assert!(!compute_coupling_strategy(&cfg, &mut blocks, false));
        assert!(blocks.iter().all(|block| !block.cpl_in_use));
    }
}
