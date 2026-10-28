//! Stereo rematrixing.

use crate::config::ValidatedConfig;
use crate::dsp::sum_square_butterfly;
use crate::encoder::Block;
use crate::tables::{AC3_CHMODE_STEREO, REMATRIX_BAND_TAB};

pub(crate) fn compute_rematrixing_strategy(
    cfg: &ValidatedConfig,
    blocks: &mut [Block],
    enabled: bool,
) {
    if cfg.channel_mode != AC3_CHMODE_STEREO {
        return;
    }

    let mut prev_num = 0usize;
    let mut prev_flags = [0u8; 4];

    for (blk, block) in blocks.iter_mut().enumerate() {
        let mut new_strategy = blk == 0;
        let mut num_bands = 4usize;
        if block.cpl_in_use {
            num_bands -= usize::from(cfg.cpl_start_freq <= 61);
            num_bands -= usize::from(cfg.cpl_start_freq == 37);
            if blk > 0 && num_bands != prev_num {
                new_strategy = true;
            }
        }
        block.num_rematrixing_bands = num_bands;

        let nb_coefs = block.end_freq[1].min(block.end_freq[2]);
        if !enabled {
            prev_num = num_bands;
            prev_flags = block.rematrixing_flags;
            block.new_rematrixing_strategy = new_strategy;
            continue;
        }

        for (bnd, &band_start) in REMATRIX_BAND_TAB[..num_bands].iter().enumerate() {
            let end = nb_coefs.min(REMATRIX_BAND_TAB[bnd + 1]);
            let sum = sum_square_butterfly(
                &block.mdct_coef[1][band_start..],
                &block.mdct_coef[2][band_start..],
                end - band_start,
            );
            let flag = u8::from(sum[2].min(sum[3]) < sum[0].min(sum[1]));
            block.rematrixing_flags[bnd] = flag;
            if blk > 0 && flag != prev_flags[bnd] {
                new_strategy = true;
            }
        }
        block.new_rematrixing_strategy = new_strategy;
        prev_num = num_bands;
        prev_flags = block.rematrixing_flags;
    }
}

pub(crate) fn apply_rematrixing(blocks: &mut [Block], enabled: bool) {
    if !enabled {
        return;
    }

    let mut flags = [0u8; 4];
    for block in blocks.iter_mut() {
        if block.new_rematrixing_strategy {
            flags = block.rematrixing_flags;
        }
        let nb_coefs = block.end_freq[1].min(block.end_freq[2]);
        for bnd in 0..block.num_rematrixing_bands {
            if flags[bnd] == 0 {
                continue;
            }
            let start = REMATRIX_BAND_TAB[bnd];
            let end = nb_coefs.min(REMATRIX_BAND_TAB[bnd + 1]);
            for i in start..end {
                let lt = block.fixed_coef[1][i];
                let rt = block.fixed_coef[2][i];
                block.fixed_coef[1][i] = (lt + rt) >> 1;
                block.fixed_coef[2][i] = (lt - rt) >> 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Acmod, ChannelLayout, Coupling, EncoderConfig};

    fn cfg() -> ValidatedConfig {
        EncoderConfig {
            layout: ChannelLayout {
                acmod: Acmod::Stereo,
                lfe: false,
            },
            bitrate_bps: 192_000,
            dialnorm: -27,
            metadata: Default::default(),
            coupling: Coupling::Off,
            stereo_rematrixing: true,
            cutoff_hz: None,
            coupling_start: None,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn apply_rematrixing_uses_the_last_transmitted_flags() {
        let cfg = cfg();
        let mut blocks = vec![Block::new(&cfg), Block::new(&cfg)];

        for block in &mut blocks {
            block.end_freq[1] = 253;
            block.end_freq[2] = 253;
            block.num_rematrixing_bands = 4;
            for i in 0..253 {
                block.fixed_coef[1][i] = 100;
                block.fixed_coef[2][i] = 40;
            }
        }

        blocks[0].new_rematrixing_strategy = true;
        blocks[0].rematrixing_flags = [1, 0, 0, 0];
        blocks[1].new_rematrixing_strategy = false;
        blocks[1].rematrixing_flags = [0, 0, 0, 0];

        apply_rematrixing(&mut blocks, true);

        for block in &blocks {
            // The first rematrixing band spans coefficients 13..25.
            for i in 13..25 {
                assert_eq!(block.fixed_coef[1][i], 70);
                assert_eq!(block.fixed_coef[2][i], 30);
            }
            // Coefficients outside the flagged band stay untouched.
            assert_eq!(block.fixed_coef[1][0], 100);
            assert_eq!(block.fixed_coef[2][0], 40);
            assert_eq!(block.fixed_coef[1][25], 100);
            assert_eq!(block.fixed_coef[2][25], 40);
        }
    }

    #[test]
    fn apply_rematrixing_is_a_no_op_when_disabled() {
        let cfg = cfg();
        let mut blocks = vec![Block::new(&cfg)];
        blocks[0].new_rematrixing_strategy = true;
        blocks[0].rematrixing_flags = [1, 1, 1, 1];
        blocks[0].num_rematrixing_bands = 4;
        blocks[0].end_freq[1] = 253;
        blocks[0].end_freq[2] = 253;
        for i in 0..253 {
            blocks[0].fixed_coef[1][i] = 100;
            blocks[0].fixed_coef[2][i] = 40;
        }

        apply_rematrixing(&mut blocks, false);

        assert!(blocks[0].fixed_coef[1][..253].iter().all(|&v| v == 100));
        assert!(blocks[0].fixed_coef[2][..253].iter().all(|&v| v == 40));
    }
}
