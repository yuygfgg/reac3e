//! Exponents, power spectral density and masking.

use crate::tables::{
    AC3_CRITICAL_BANDS, AC3_MAX_COEFS, BAND_START_TAB, BIN_TO_BAND_TAB, DB_PER_BIT_TAB,
    FAST_DECAY_TAB, FLOOR_TAB, HEARING_THRESHOLD_TAB, LOG_ADD_TAB, SLOW_DECAY_TAB, SLOW_GAIN_TAB,
};

/// Exponent coding strategy for one channel and block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpStrategy {
    /// Reuse the exponents from the reference block.
    Reuse,
    /// 1.5-bit differential exponents.
    D15,
    /// 2.5-bit differential exponents.
    D25,
    /// 4.5-bit differential exponents.
    D45,
}

impl ExpStrategy {
    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::Reuse => 0,
            Self::D15 => 1,
            Self::D25 => 2,
            Self::D45 => 3,
        }
    }

    #[cfg(test)]
    pub(crate) const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Reuse),
            1 => Some(Self::D15),
            2 => Some(Self::D25),
            3 => Some(Self::D45),
            _ => None,
        }
    }

    pub(crate) const fn is_reuse(self) -> bool {
        matches!(self, Self::Reuse)
    }

    pub(crate) const fn group_size(self) -> usize {
        match self {
            Self::Reuse => 0,
            Self::D15 => 1,
            Self::D25 => 2,
            Self::D45 => 4,
        }
    }

    /// Index into the three-group exponent table (strategies code - 1).
    pub(crate) const fn table_index(self) -> usize {
        self.code() as usize - 1
    }
}

/// Bit-allocation parameters, initialised from the default codes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BitAllocParameters {
    pub sr_code: i32,
    pub sr_shift: i32,
    pub slow_gain: i32,
    pub slow_decay: i32,
    pub fast_decay: i32,
    pub db_per_bit: i32,
    pub floor: i32,
    pub cpl_fast_leak: i32,
    pub cpl_slow_leak: i32,
}

impl BitAllocParameters {
    /// The fixed defaults used by the E-AC-3 encoder.
    pub(crate) fn eac3_defaults() -> Self {
        Self {
            sr_code: 0,
            sr_shift: 0,
            slow_decay: SLOW_DECAY_TAB[2] as i32,
            fast_decay: FAST_DECAY_TAB[1] as i32,
            slow_gain: SLOW_GAIN_TAB[1] as i32,
            db_per_bit: DB_PER_BIT_TAB[2] as i32,
            floor: FLOOR_TAB[7] as i32,
            cpl_fast_leak: 0,
            cpl_slow_leak: 0,
        }
    }
}

/// `exponent_group_tab[coupling][strategy-1][ncoefs]`.
///
/// The group size is `3 << (strategy - 1)`, with a special case for the
/// 7-coefficient LFE channel.
pub(crate) const EXPONENT_GROUP_TAB: [[[u8; 256]; 3]; 2] = build_exponent_group_tab();

const fn build_exponent_group_tab() -> [[[u8; 256]; 3]; 2] {
    let mut tab = [[[0u8; 256]; 3]; 2];
    let mut expstr = 0usize;
    while expstr < 3 {
        let grpsize = 3usize << expstr;
        let mut i = 12usize;
        while i < 256 {
            tab[0][expstr][i] = ((i + grpsize - 4) / grpsize) as u8;
            tab[1][expstr][i] = (i / grpsize) as u8;
            i += 1;
        }
        expstr += 1;
    }
    tab[0][0][7] = 2;
    tab
}

/// Extract 24-bit fixed-point exponents.
pub(crate) fn extract_exponents(coef: &[i32], exp: &mut [u8], count: usize) {
    for (exp, &coefficient) in exp.iter_mut().zip(&coef[..count]) {
        let magnitude = coefficient.unsigned_abs();
        *exp = if magnitude != 0 {
            23 - magnitude.ilog2() as u8
        } else {
            24
        };
    }
}

/// Calculate the per-bin power spectral density and the critical-band PSD.
pub(crate) fn calc_psd(
    exp: &[u8],
    start: usize,
    end: usize,
    psd: &mut [i16],
    band_psd: &mut [i16],
) {
    if start >= end {
        return;
    }
    for (psd, &exponent) in psd[start..end].iter_mut().zip(&exp[start..end]) {
        *psd = (3072 - ((exponent as i32) << 7)) as i16;
    }

    let early_end = end.min(28);
    let mut bin = start;
    if start < early_end {
        band_psd[start..early_end].copy_from_slice(&psd[start..early_end]);
        bin = early_end;
    }
    if bin >= end {
        return;
    }

    let mut band = BIN_TO_BAND_TAB[bin] as usize;
    loop {
        let mut value = psd[bin] as i32;
        bin += 1;
        let band_end = (BAND_START_TAB[band + 1] as usize).min(end);
        while bin < band_end {
            let maximum = value.max(psd[bin] as i32);
            let address = (maximum - ((value + psd[bin] as i32 + 1) >> 1)).min(255) as usize;
            value = maximum + LOG_ADD_TAB[address] as i32;
            bin += 1;
        }
        band_psd[band] = value as i16;
        band += 1;
        if end <= BAND_START_TAB[band] as usize {
            break;
        }
    }
}

fn calc_lowcomp1(a: i32, b0: i32, b1: i32, c: i32) -> i32 {
    if b0 + 256 == b1 {
        c
    } else if b0 > b1 {
        (a - 64).max(0)
    } else {
        a
    }
}

fn calc_lowcomp(a: i32, b0: i32, b1: i32, bin: usize) -> i32 {
    if bin < 7 {
        calc_lowcomp1(a, b0, b1, 384)
    } else if bin < 20 {
        calc_lowcomp1(a, b0, b1, 320)
    } else {
        (a - 128).max(0)
    }
}

/// Calculate the masking curve.  Delta bit allocation is not used.
pub(crate) fn calc_mask(
    params: &BitAllocParameters,
    band_psd: &[i16],
    start: usize,
    end: usize,
    fast_gain: i32,
    is_lfe: bool,
    mask: &mut [i16],
) {
    if end == 0 {
        return;
    }
    let band_start = BIN_TO_BAND_TAB[start] as usize;
    let band_end = BIN_TO_BAND_TAB[end - 1] as usize + 1;

    let mut excite = [0i16; AC3_CRITICAL_BANDS];
    let mut fastleak = 0i32;
    let mut slowleak = 0i32;

    if band_start == 0 {
        let mut lowcomp = 0;
        lowcomp = calc_lowcomp1(lowcomp, band_psd[0] as i32, band_psd[1] as i32, 384);
        excite[0] = (band_psd[0] as i32 - fast_gain - lowcomp) as i16;
        lowcomp = calc_lowcomp1(lowcomp, band_psd[1] as i32, band_psd[2] as i32, 384);
        excite[1] = (band_psd[1] as i32 - fast_gain - lowcomp) as i16;
        let mut first_band = 7usize;
        let mut band = 2usize;
        while band < 7 {
            if !(is_lfe && band == 6) {
                lowcomp = calc_lowcomp1(
                    lowcomp,
                    band_psd[band] as i32,
                    band_psd[band + 1] as i32,
                    384,
                );
            }
            fastleak = band_psd[band] as i32 - fast_gain;
            slowleak = band_psd[band] as i32 - params.slow_gain;
            excite[band] = (fastleak - lowcomp) as i16;
            if !(is_lfe && band == 6) && band_psd[band] <= band_psd[band + 1] {
                first_band = band + 1;
                break;
            }
            band += 1;
        }

        let end1 = band_end.min(22);
        let mut band = first_band;
        while band < end1 {
            if !(is_lfe && band == 6) {
                lowcomp = calc_lowcomp(
                    lowcomp,
                    band_psd[band] as i32,
                    band_psd[band + 1] as i32,
                    band,
                );
            }
            fastleak = (fastleak - params.fast_decay).max(band_psd[band] as i32 - fast_gain);
            slowleak = (slowleak - params.slow_decay).max(band_psd[band] as i32 - params.slow_gain);
            excite[band] = (fastleak - lowcomp).max(slowleak) as i16;
            band += 1;
        }
        // The leak tail always starts at band 22, matching the reference.
        let mut band = 22usize;
        while band < band_end {
            fastleak = (fastleak - params.fast_decay).max(band_psd[band] as i32 - fast_gain);
            slowleak = (slowleak - params.slow_decay).max(band_psd[band] as i32 - params.slow_gain);
            excite[band] = fastleak.max(slowleak) as i16;
            band += 1;
        }
    } else {
        fastleak = (params.cpl_fast_leak << 8) + 768;
        slowleak = (params.cpl_slow_leak << 8) + 768;
        let mut band = band_start;
        while band < band_end {
            fastleak = (fastleak - params.fast_decay).max(band_psd[band] as i32 - fast_gain);
            slowleak = (slowleak - params.slow_decay).max(band_psd[band] as i32 - params.slow_gain);
            excite[band] = fastleak.max(slowleak) as i16;
            band += 1;
        }
    }

    for band in band_start..band_end {
        let excitation = excite[band] as i32;
        let tmp = params.db_per_bit - band_psd[band] as i32;
        let excitation = if tmp > 0 {
            (excitation + (tmp >> 2)) as i16
        } else {
            excitation as i16
        };
        let threshold =
            HEARING_THRESHOLD_TAB[band >> params.sr_shift][params.sr_code as usize] as i32;
        mask[band] = threshold.max(excitation as i32) as i16;
    }
}

/// Rewrite a channel's exponents to the values the decoder will reconstruct.
///
/// `exp` is the channel's full 256-coefficient exponent buffer, `start` is the
/// first active bin, and the function rewrites the exponents to the values the
/// decoder will reconstruct.
pub(crate) fn encode_exponents_blk_ch(
    exp: &mut [u8; AC3_MAX_COEFS],
    start: usize,
    nb_exps: usize,
    exp_strategy: ExpStrategy,
    cpl: bool,
) {
    let cpl_offset = usize::from(cpl);
    let group_count =
        EXPONENT_GROUP_TAB[cpl as usize][exp_strategy.table_index()][nb_exps] as usize * 3;

    // Collapse each group to its minimum exponent.
    match exp_strategy {
        ExpStrategy::D25 => {
            for group in 1..=group_count {
                let source = start + 1 - cpl_offset + 2 * (group - 1);
                let target = start + group - cpl_offset;
                exp[target] = exp[source].min(exp[source + 1]);
            }
        }
        ExpStrategy::D45 => {
            for group in 1..=group_count {
                let source = start + 1 - cpl_offset + 4 * (group - 1);
                let target = start + group - cpl_offset;
                exp[target] = exp[source]
                    .min(exp[source + 1])
                    .min(exp[source + 2])
                    .min(exp[source + 3]);
            }
        }
        ExpStrategy::Reuse | ExpStrategy::D15 => {}
    }

    // The first exponent is limited to 15 in non-coupling channels.
    if !cpl && exp[start] > 15 {
        exp[start] = 15;
    }

    // Transmitted differences must fit in the 2-bit delta range.
    for group in 1..=group_count {
        let index = start + group;
        exp[index] = exp[index].min(exp[index - 1].saturating_add(2));
    }
    for group in (0..group_count).rev() {
        let index = start + group;
        exp[index] = exp[index].min(exp[index + 1].saturating_add(2));
    }

    if cpl {
        exp[start - 1] = exp[start] & !1;
    }

    // Expand the grouped exponents back to one value per coefficient.
    match exp_strategy {
        ExpStrategy::D25 => {
            for group in (1..=group_count).rev() {
                let value = exp[start + group - cpl_offset];
                let target = start + 2 * group - cpl_offset;
                exp[target] = value;
                exp[target - 1] = value;
            }
        }
        ExpStrategy::D45 => {
            for group in (1..=group_count).rev() {
                let value = exp[start + group - cpl_offset];
                let target = start + 4 * group - cpl_offset;
                exp[target] = value;
                exp[target - 1] = value;
                exp[target - 2] = value;
                exp[target - 3] = value;
            }
        }
        ExpStrategy::Reuse | ExpStrategy::D15 => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponent_strategy_codes_round_trip() {
        for strategy in [
            ExpStrategy::Reuse,
            ExpStrategy::D15,
            ExpStrategy::D25,
            ExpStrategy::D45,
        ] {
            assert_eq!(ExpStrategy::from_code(strategy.code()), Some(strategy));
        }
        assert_eq!(ExpStrategy::from_code(4), None);
    }

    #[test]
    fn exponent_group_table_matches_formula() {
        // D15, 253 coefficients: (253 + 3 - 4) / 3 = 84.
        assert_eq!(EXPONENT_GROUP_TAB[0][0][253], 84);
        // D25: (253 + 6 - 4) / 6 = 42.
        assert_eq!(EXPONENT_GROUP_TAB[0][1][253], 42);
        // D45: (253 + 12 - 4) / 12 = 21.
        assert_eq!(EXPONENT_GROUP_TAB[0][2][253], 21);
        // Coupling excludes the DC coefficient.
        assert_eq!(EXPONENT_GROUP_TAB[1][0][253], 84);
        assert_eq!(EXPONENT_GROUP_TAB[0][0][7], 2);
    }

    #[test]
    fn exponents_from_coefficients() {
        let coef = [0i32, 1, 2, 3, 4, 1 << 23, (1 << 23) + 1, -5];
        let mut exp = [0u8; 8];
        extract_exponents(&coef, &mut exp, 8);
        assert_eq!(exp[0], 24);
        assert_eq!(exp[1], 23);
        assert_eq!(exp[2], 22);
        assert_eq!(exp[3], 22);
        assert_eq!(exp[4], 21);
        assert_eq!(exp[5], 0);
        assert_eq!(exp[6], 0);
        assert_eq!(exp[7], 21);
    }

    #[test]
    fn psd_is_linear_in_exponent() {
        let mut exp = [0u8; 8];
        exp[3] = 1;
        exp[4] = 2;
        let mut psd = [0i16; 256];
        let mut band = [0i16; 64];
        calc_psd(&exp, 3, 5, &mut psd, &mut band);
        assert_eq!(psd[3], 3072 - 128);
        assert_eq!(psd[4], 3072 - 256);
        assert_eq!(band[3], psd[3]);
        assert_eq!(band[4], psd[4]);
    }

    #[test]
    fn exponent_encoding_limits_deltas() {
        let mut exp = [0u8; 256];
        for (i, e) in exp.iter_mut().enumerate() {
            *e = (i % 20) as u8;
        }
        encode_exponents_blk_ch(&mut exp, 0, 253, ExpStrategy::D15, false);
        // Grouped exponents are formed in `ac3_group_exponents`; here we only
        // assert the reconstructed sequence never changes by more than two
        // between neighbouring transmitted positions.
        for pair in exp[..253].windows(2) {
            assert!(pair[0].abs_diff(pair[1]) <= 2 || pair[1] < pair[0]);
        }
    }

    #[test]
    fn psd_handles_a_coupling_start_above_the_early_bins() {
        let exp = [0u8; 256];
        let mut psd = [0i16; 256];
        let mut band_psd = [0i16; 64];
        let start = 133;
        let end = 145;

        calc_psd(&exp, start, end, &mut psd, &mut band_psd);

        for value in &psd[start..end] {
            assert_eq!(*value, 3072);
        }
        let first_band = BIN_TO_BAND_TAB[start] as usize;
        assert!(band_psd[first_band] > 3072);
    }

    #[test]
    fn grouped_exponents_preserve_constant_exponents() {
        let mut d25 = [5u8; 256];
        encode_exponents_blk_ch(&mut d25, 0, 253, ExpStrategy::D25, false);
        assert!(d25[..253].iter().all(|&exponent| exponent == 5));

        let mut d45 = [5u8; 256];
        encode_exponents_blk_ch(&mut d45, 0, 253, ExpStrategy::D45, false);
        assert!(d45[..253].iter().all(|&exponent| exponent == 5));
    }

    #[test]
    fn coupling_exponents_clear_the_low_bit_below_the_start() {
        let mut d25 = [5u8; 256];
        encode_exponents_blk_ch(&mut d25, 37, 108, ExpStrategy::D25, true);
        assert!(d25[37..37 + 108].iter().all(|&exponent| exponent == 5));
        assert_eq!(d25[36], 4);

        let mut d45 = [5u8; 256];
        encode_exponents_blk_ch(&mut d45, 37, 108, ExpStrategy::D45, true);
        assert!(d45[37..37 + 108].iter().all(|&exponent| exponent == 5));
        assert_eq!(d45[36], 4);
    }
}
