//! Coefficient scaling and mantissa quantisation.

use crate::tables::AC3_MAX_COEFS;

pub(crate) const COEF_MIN: f32 = -16_777_215.0 / 16_777_216.0;
pub(crate) const COEF_MAX: f32 = 16_777_215.0 / 16_777_216.0;

/// Scale to 24-bit fixed point with round-to-nearest-even.
pub(crate) fn float_to_fixed24(dst: &mut [i32], src: &[f32]) {
    const SCALE: f32 = (1u32 << 24) as f32;
    for (dst, &src) in dst.iter_mut().zip(src) {
        *dst = (src * SCALE).round_ties_even() as i32;
    }
}

/// Clip MDCT coefficients to the coding range.
pub(crate) fn clip_coefficients(coef: &mut [f32], len: usize) {
    for coefficient in &mut coef[..len] {
        *coefficient = coefficient.clamp(COEF_MIN, COEF_MAX);
    }
}

/// Symmetric quantisation on `levels` levels.
#[inline]
pub(crate) fn sym_quant(coefficient: i32, exponent: u32, levels: i32) -> i32 {
    (((levels * coefficient) >> (24 - exponent)) + levels) >> 1
}

/// Asymmetric quantisation on 2^qbits levels.
#[inline]
pub(crate) fn asym_quant(coefficient: i32, exponent: u32, qbits: u32) -> i32 {
    let mut value =
        (((coefficient.wrapping_mul(1i32.wrapping_shl(exponent))) >> (24 - qbits)) + 1) >> 1;
    let maximum = 1i32 << (qbits - 1);
    if value >= maximum {
        value = maximum - 1;
    }
    value
}

/// Channel stride in the flat per-block mantissa buffer.
pub(crate) const QSTRIDE: usize = AC3_MAX_COEFS;

/// Pending mantissas that are packed into a single grouped code.
#[derive(Debug, Clone, Copy, Default)]
struct MantGroup {
    position: (usize, usize),
    count: u8,
}

/// Grouping parameters for the 5-, 7- and 11-level quantisers.
#[derive(Debug, Clone, Copy)]
struct GroupSpec {
    /// Number of quantisation levels.
    levels: i32,
    /// Scale applied to the first mantissa of a group.
    first_scale: i32,
    /// Scale applied to the second mantissa (three-mantissa groups only).
    second_scale: i32,
    /// Whether the group packs three mantissas instead of two.
    three_mantissas: bool,
}

const MANT1_GROUP: GroupSpec = GroupSpec {
    levels: 3,
    first_scale: 9,
    second_scale: 3,
    three_mantissas: true,
};

const MANT2_GROUP: GroupSpec = GroupSpec {
    levels: 5,
    first_scale: 25,
    second_scale: 5,
    three_mantissas: true,
};

const MANT4_GROUP: GroupSpec = GroupSpec {
    levels: 11,
    first_scale: 11,
    second_scale: 1,
    three_mantissas: false,
};

/// Running grouped-mantissa state for one block.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MantCounts {
    mant1: MantGroup,
    mant2: MantGroup,
    mant4: MantGroup,
}

impl MantCounts {
    /// Pack `value` into the group, or stash it until the group is complete.
    ///
    /// `128` is the "no mantissa written here" sentinel used by the bitstream.
    fn push(
        group: &mut MantGroup,
        spec: GroupSpec,
        channel: usize,
        coefficient: usize,
        value: i32,
        qmant: &mut [i16],
    ) -> i32 {
        match group.count {
            0 => {
                group.position = (channel, coefficient);
                group.count = 1;
                value * spec.first_scale
            }
            1 if spec.three_mantissas => {
                let index = group.position.0 * QSTRIDE + group.position.1;
                qmant[index] += (spec.second_scale * value) as i16;
                group.count = 2;
                128
            }
            _ => {
                let index = group.position.0 * QSTRIDE + group.position.1;
                qmant[index] += value as i16;
                group.count = 0;
                128
            }
        }
    }
}

/// One channel's coefficient data for a block.
pub(crate) struct MantissaBlock<'a> {
    pub fixed_coef: &'a [i32],
    pub exp: &'a [u8],
    pub bap: &'a [u8],
    pub qmant: &'a mut [i16],
    pub start_freq: usize,
    pub end_freq: usize,
}

/// Quantise one channel's mantissas for one block.
pub(crate) fn quantize_mantissas_blk_ch(
    counts: &mut MantCounts,
    channel: usize,
    block: MantissaBlock<'_>,
) {
    let MantissaBlock {
        fixed_coef,
        exp,
        bap,
        qmant,
        start_freq,
        end_freq,
    } = block;

    for coefficient in start_freq..end_freq {
        let value = fixed_coef[coefficient];
        let exponent = exp[coefficient] as u32;
        let value = match bap[coefficient] {
            0 => 0,
            1 => {
                let value = sym_quant(value, exponent, MANT1_GROUP.levels);
                MantCounts::push(
                    &mut counts.mant1,
                    MANT1_GROUP,
                    channel,
                    coefficient,
                    value,
                    qmant,
                )
            }
            2 => {
                let value = sym_quant(value, exponent, MANT2_GROUP.levels);
                MantCounts::push(
                    &mut counts.mant2,
                    MANT2_GROUP,
                    channel,
                    coefficient,
                    value,
                    qmant,
                )
            }
            3 => sym_quant(value, exponent, 7),
            4 => {
                let value = sym_quant(value, exponent, MANT4_GROUP.levels);
                MantCounts::push(
                    &mut counts.mant4,
                    MANT4_GROUP,
                    channel,
                    coefficient,
                    value,
                    qmant,
                )
            }
            5 => sym_quant(value, exponent, 15),
            14 => asym_quant(value, exponent, 14),
            15 => asym_quant(value, exponent, 16),
            bits => asym_quant(value, exponent, (bits - 1) as u32),
        };
        qmant[channel * QSTRIDE + coefficient] = value as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed24_rounds_to_even() {
        let mut dst = [0i32; 4];
        float_to_fixed24(
            &mut dst,
            &[
                (0.5 / (1u32 << 24) as f32),
                (1.5 / (1u32 << 24) as f32),
                (-0.5 / (1u32 << 24) as f32),
                (-1.5 / (1u32 << 24) as f32),
            ],
        );
        assert_eq!(dst, [0, 2, 0, -2]);
    }

    #[test]
    fn symmetric_quant_bounds() {
        for coefficient in [-16_000_000, -1, 0, 1, 16_000_000] {
            let value = sym_quant(coefficient, 0, 3);
            assert!((0..3).contains(&value));
        }
    }

    #[test]
    fn grouped_mantissas_pack_three_and_two_codes() {
        let fixed = [1 << 23; 8];
        let exp = [0u8; 8];
        let bap = [1u8, 1, 1, 2, 2, 2, 4, 4];
        let mut qmant = [0i16; QSTRIDE];
        let mut counts = MantCounts::default();

        quantize_mantissas_blk_ch(
            &mut counts,
            0,
            MantissaBlock {
                fixed_coef: &fixed,
                exp: &exp,
                bap: &bap,
                qmant: &mut qmant,
                start_freq: 0,
                end_freq: 8,
            },
        );

        assert_eq!(&qmant[..8], &[26, 128, 128, 93, 128, 128, 96, 128]);
    }

    #[test]
    fn unquantised_coefficients_write_zero() {
        let fixed = [1 << 23; 3];
        let exp = [0u8; 3];
        let bap = [0u8, 3, 0];
        let mut qmant = [7i16; 3];
        let mut counts = MantCounts::default();

        quantize_mantissas_blk_ch(
            &mut counts,
            0,
            MantissaBlock {
                fixed_coef: &fixed,
                exp: &exp,
                bap: &bap,
                qmant: &mut qmant,
                start_freq: 0,
                end_freq: 3,
            },
        );

        assert_eq!(qmant, [0, 5, 0]);
    }
}
