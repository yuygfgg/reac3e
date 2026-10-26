//! Bit allocation pointers, mantissa bit counting and the CBR SNR search.

use crate::tables::{
    AC3_MAX_BLOCKS, AC3_MAX_COEFS, BAND_START_TAB, BAP_BITS, BAP_TAB, BIN_TO_BAND_TAB,
};

/// Assign a bit-allocation pointer to every coefficient in a band range.
pub(crate) fn calc_bap(
    mask: &[i16],
    psd: &[i16],
    start: usize,
    end: usize,
    snr_offset: i32,
    floor: i32,
    bap: &mut [u8],
) {
    if snr_offset == -960 {
        bap[..AC3_MAX_COEFS].fill(0);
        return;
    }

    let mask_offset = |band: usize| {
        let offset = (mask[band] as i32 - snr_offset - floor).max(0) & 0x1FE0;
        offset + floor
    };

    let mut bin = start;
    while bin < end.min(28) {
        let address = ((psd[bin] as i32 - mask_offset(bin)) >> 5).clamp(0, 63) as usize;
        bap[bin] = BAP_TAB[address];
        bin += 1;
    }
    if bin >= end {
        return;
    }

    let mut band = BIN_TO_BAND_TAB[bin] as usize;
    loop {
        let offset = mask_offset(band);
        band += 1;
        let band_end = (BAND_START_TAB[band] as usize).min(end);
        while bin < band_end {
            let address = ((psd[bin] as i32 - offset) >> 5).clamp(0, 63) as usize;
            bap[bin] = BAP_TAB[address];
            bin += 1;
        }
        if end <= band_end {
            break;
        }
    }
}

/// Accumulate mantissa counts for one block.
pub(crate) fn update_bap_counts(mant_cnt: &mut [u16; 16], bap: &[u8], len: usize) {
    for &b in &bap[..len] {
        mant_cnt[b as usize] += 1;
    }
}

/// Initialise the padded mantissa counts.
pub(crate) fn count_mantissa_bits_init(mant_cnt: &mut [[u16; 16]; AC3_MAX_BLOCKS]) {
    for blk in mant_cnt.iter_mut() {
        *blk = [0; 16];
        blk[1] = 2;
        blk[2] = 2;
        blk[4] = 1;
    }
}

/// Compute the number of mantissa bits for a frame.
pub(crate) fn compute_mantissa_size(mant_cnt: &[[u16; 16]; AC3_MAX_BLOCKS]) -> usize {
    let mut bits = 0usize;
    for blk in mant_cnt {
        bits += (blk[1] as usize / 3) * 5;
        bits += ((blk[2] as usize / 3) + (blk[4] as usize >> 1)) * 7;
        bits += blk[3] as usize * 3;
        for bap in 5..16 {
            bits += blk[bap] as usize * BAP_BITS[bap] as usize;
        }
    }
    bits
}

/// The CBR SNR-offset search.  `bits_for` runs one bit-allocation pass for the
/// given SNR offset and returns the number of mantissa bits required.
///
/// Returns the chosen SNR offset, or `None` if even SNR 0 does not fit.
pub(crate) fn cbr_search(
    bits_left: i32,
    initial_snr: i32,
    previous_snr: i32,
    mut bits_for: impl FnMut(i32) -> i32,
) -> Option<i32> {
    if bits_left < 0 {
        return None;
    }

    // If the previous frame used 1023, try to keep it.
    if previous_snr == 1023 && bits_for(1023) <= bits_left {
        return Some(1023);
    }

    let mut snr_offset = initial_snr;
    while snr_offset >= 0 && bits_for(snr_offset) > bits_left {
        snr_offset -= 64;
    }
    if snr_offset < 0 {
        return None;
    }

    let mut snr_incr = 64;
    while snr_incr > 0 {
        while snr_offset + snr_incr <= 1023 && bits_for(snr_offset + snr_incr) <= bits_left {
            snr_offset += snr_incr;
        }
        snr_incr >>= 2;
    }
    Some(snr_offset)
}

/// Convert an SNR offset to the `coarse_snr_offset` / `fine_snr_offset` split
/// written to the bitstream.
pub(crate) fn split_snr_offset(snr: i32) -> (u8, u8) {
    ((snr >> 4) as u8, (snr & 0xF) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mantissa_size_matches_grouping() {
        let mut cnt = [[0u16; 16]; AC3_MAX_BLOCKS];
        count_mantissa_bits_init(&mut cnt);
        // 3 bap=1 mantissas = 5 bits; 2 bap=4 = 7 bits.
        cnt[0][1] = 3;
        cnt[0][4] = 2;
        let bits = compute_mantissa_size(&cnt);
        assert_eq!(bits, 12);
    }

    #[test]
    fn snr_search_is_monotonic() {
        // A synthetic allocator: 1000 bits at SNR 0, 0 bits at SNR 1023.
        let bits_for = |snr: i32| 1000 - snr;
        let chosen = cbr_search(400, 640, 0, bits_for).unwrap();
        assert!(bits_for(chosen) <= 400);
        assert!(chosen + 1 > 1023 || bits_for(chosen + 1) > 400);
    }

    #[test]
    fn snr_search_rejects_negative_budget_and_exhausted_search() {
        assert_eq!(cbr_search(-1, 640, 0, |_| 0), None);
        assert_eq!(cbr_search(10, 640, 0, |_| 11), None);
    }

    #[test]
    fn snr_search_keeps_a_previous_full_offset_when_it_fits() {
        let chosen = cbr_search(100, 0, 1023, |snr| if snr == 1023 { 100 } else { 0 });
        assert_eq!(chosen, Some(1023));
    }

    #[test]
    fn zero_snr_offset_zeroes_every_bap() {
        let mask = [0i16; 256];
        let psd = [0i16; 256];
        let mut bap = [9u8; 256];
        calc_bap(&mask, &psd, 0, 256, -960, 0, &mut bap);
        assert!(bap.iter().all(|&value| value == 0));
    }

    #[test]
    fn mantissa_counts_accumulate_by_bap() {
        let mut counts = [0u16; 16];
        update_bap_counts(&mut counts, &[0, 1, 1, 4, 15], 5);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[4], 1);
        assert_eq!(counts[15], 1);
        assert_eq!(counts.iter().sum::<u16>(), 5);
    }
}
