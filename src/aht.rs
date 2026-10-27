//! Adaptive Hybrid Transform (AHT) helpers.
//!
//! AHT applies a 6-point DCT across the six MDCT blocks of a frame before
//! quantisation.  The decoder uses the inverse transform in `idct6`.
//!
//! The encoder must reproduce the decoder's fixed-point pipeline exactly:
//! `pre_mantissa = dequant(...)`, `idct6(pre_mantissa)`, then
//! `coefficient = pre_mantissa >> exponent`.

use crate::aht_tables::{BITS_VS_HEBAP, VQ_TABLES};

/// Rows of the inverse of the decoder's 6-point IDCT (M^T / 6).
const FWD: [[f64; 6]; 6] = [
    [
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
    ],
    [
        0.227_670_893,
        1.0 / 6.0,
        0.061_004_226_6,
        -0.061_004_226_6,
        -1.0 / 6.0,
        -0.227_670_893,
    ],
    [
        0.204_124_138,
        0.0,
        -0.204_124_138,
        -0.204_124_138,
        0.0,
        0.204_124_138,
    ],
    [
        1.0 / 6.0,
        -1.0 / 6.0,
        -1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        -1.0 / 6.0,
    ],
    [
        0.117_851_132,
        -0.235_702_264,
        0.117_851_132,
        0.117_851_132,
        -0.235_702_264,
        0.117_851_132,
    ],
    [
        0.061_004_226_6,
        -1.0 / 6.0,
        0.227_670_893,
        -0.227_670_893,
        1.0 / 6.0,
        -0.061_004_226_6,
    ],
];

/// Decoder's no-GAQ mantissa remap (Table E3.6, Gk=1).
const GAQ_REMAP_1: [i16; 12] = [4681, 2185, 1057, 520, 258, 129, 64, 32, 16, 8, 2, 0];

/// Forward 6-point AHT transform (inverse of the decoder's `idct6`).
pub(crate) fn forward(input: &[f64; 6]) -> [f64; 6] {
    let mut output = [0.0; 6];
    for (k, out) in output.iter_mut().enumerate() {
        *out = FWD[k]
            .iter()
            .zip(input)
            .map(|(coefficient, sample)| coefficient * sample)
            .sum();
    }
    output
}

/// Decoder-compatible fixed-point 6-point IDCT.
#[allow(dead_code)]
pub(crate) fn idct6(pre_mantissa: &mut [i32; 6]) {
    const COEFF_0: i64 = 10_273_905;
    const COEFF_1: i64 = 11_863_283;
    const COEFF_2: i64 = 3_070_444;

    let odd1 = pre_mantissa[1] - pre_mantissa[3] - pre_mantissa[5];
    let even2 = ((pre_mantissa[2] as i64 * COEFF_0) >> 23) as i32;
    let tmp = ((pre_mantissa[4] as i64 * COEFF_1) >> 23) as i32;
    let odd0 = (((pre_mantissa[1] + pre_mantissa[5]) as i64 * COEFF_2) >> 23) as i32;

    let even0 = pre_mantissa[0] + (tmp >> 1);
    let even1 = pre_mantissa[0] - tmp;

    let tmp = even0;
    let even0 = tmp + even2;
    let even2 = tmp - even2;

    let tmp = odd0;
    let odd0 = tmp + pre_mantissa[1] + pre_mantissa[3];
    let odd2 = tmp + pre_mantissa[5] - pre_mantissa[3];

    pre_mantissa[0] = even0 + odd0;
    pre_mantissa[1] = even1 + odd1;
    pre_mantissa[2] = even2 + odd2;
    pre_mantissa[3] = even2 - odd2;
    pre_mantissa[4] = even1 - odd1;
    pre_mantissa[5] = even0 - odd0;
}

/// Quantise one AHT coefficient vector for a given `hebap`.
///
/// `target` is `forward(x) * 2^exponent`, i.e. the decoder's pre-mantissa
/// that reconstructs `x`.  Returns the VQ index for `hebap` 1..=7 (the
/// signed mantissas are unused), or the six signed mantissas for `hebap >= 8`.
pub(crate) fn quantise(hebap: u8, target: &[f64; 6]) -> (u16, [i32; 6]) {
    if hebap == 0 {
        return (0, [0; 6]);
    }
    if hebap < 8 {
        let table = VQ_TABLES[hebap as usize];
        let mut best = 0u16;
        let mut best_error = f64::INFINITY;
        for (index, vector) in table.iter().enumerate() {
            let mut error = 0.0;
            for (value, target) in vector.iter().zip(target) {
                let difference = *value as f64 * 256.0 - *target;
                error += difference * difference;
            }
            if error < best_error {
                best_error = error;
                best = index as u16;
            }
        }
        return (best, [0; 6]);
    }

    let bits = BITS_VS_HEBAP[hebap as usize] as i32;
    let step = (1i64 << (24 - bits)) as f64;
    // The decoder applies this remap after dequantising small mantissas.
    let remap = 1.0 + GAQ_REMAP_1[(hebap - 8) as usize] as f64 / 32768.0;
    let denominator = step * remap;
    let minimum = -(1i64 << (bits - 1));
    let maximum = (1i64 << (bits - 1)) - 1;
    let mut mantissas = [0i32; 6];
    for (mantissa, target) in mantissas.iter_mut().zip(target) {
        let value = (*target / denominator).round() as i64;
        *mantissa = value.clamp(minimum, maximum) as i32;
    }
    (0, mantissas)
}

/// Reconstruct one AHT bin the way the decoder will: apply the VQ/scalar
/// dequantisation (including the no-GAQ remap), the 6-point IDCT, then the
/// exponent shift.
#[allow(dead_code)]
pub(crate) fn reconstruct(hebap: u8, vq: u16, mantissa: &[i32; 6], exponent: i32) -> [i32; 6] {
    let mut pre = [0i32; 6];
    if hebap == 0 {
        return pre;
    }
    if hebap < 8 {
        let table = VQ_TABLES[hebap as usize];
        let vector = table.get(vq as usize).copied().unwrap_or([0i16; 6]);
        for (value, coefficient) in pre.iter_mut().zip(vector) {
            *value = coefficient as i32 * 256;
        }
    } else {
        let bits = BITS_VS_HEBAP[hebap as usize] as i32;
        let step = 1i64 << (24 - bits);
        let remap = GAQ_REMAP_1[(hebap - 8) as usize] as i64;
        for (value, mantissa) in pre.iter_mut().zip(mantissa) {
            let mut dequant = *mantissa as i64 * step;
            dequant += (remap * dequant) >> 15;
            *value = dequant.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
    idct6(&mut pre);
    for value in &mut pre {
        *value >>= exponent;
    }
    pre
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_idct_round_trip() {
        let input = [1.0e6, -2.0e6, 3.0e6, -4.0e6, 5.0e6, -6.0e6];
        let aht = forward(&input);
        let mut mantissa = [0i32; 6];
        for (mantissa, value) in mantissa.iter_mut().zip(aht) {
            *mantissa = value.round() as i32;
        }
        let mut reconstructed = mantissa;
        idct6(&mut reconstructed);
        for (got, want) in reconstructed.iter().zip(input) {
            assert!((*got as f64 - want).abs() < 4.0, "{got} vs {want}");
        }
    }
}
