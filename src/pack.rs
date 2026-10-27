//! Audio-block packing, frame padding and CRC2.

use crate::aht_tables::BITS_VS_HEBAP;
use crate::analysis::EXPONENT_GROUP_TAB;
use crate::bitwriter::BitWriter;
use crate::encoder::{FrameEncoder, coded_channel_order};
use crate::error::EncodeError;
use crate::tables::{AC3_CHMODE_STEREO, BAP_BITS};

pub(crate) fn write_audio_block(enc: &FrameEncoder, blk: usize, w: &mut BitWriter) {
    let cfg = &enc.cfg;
    let block = &enc.blocks[blk];

    if enc.blk_sw_syntax {
        for ch in 1..=cfg.fbw_channels {
            w.write_bool(enc.blksw[ch][blk]);
        }
    }

    w.write_bool(false); // dynamic range codes disabled
    w.write_bool(false); // spectral extension disabled

    if block.new_cpl_strategy && block.cpl_in_use {
        w.write_bool(false); // enhanced coupling disabled
        if cfg.channel_mode != AC3_CHMODE_STEREO {
            for ch in 1..=cfg.fbw_channels {
                w.write_bool(block.channel_in_cpl[ch]);
            }
        }
        if cfg.channel_mode == AC3_CHMODE_STEREO {
            w.write_bool(enc.phase_flags_in_use);
        }
        let start_sub = (cfg.cpl_start_freq - 37) / 12;
        let end_sub = (cfg.cpl_end_freq - 37) / 12;
        w.write(4, start_sub as u32);
        w.write(4, (end_sub - 3) as u32);
        w.write_bool(false); // default coupling band structure
    }

    if block.cpl_in_use {
        let mut cpl_coords_exist = false;
        for ch in 1..=cfg.fbw_channels {
            if !block.channel_in_cpl[ch] {
                continue;
            }
            if !block.new_cpl_coords[ch].is_implicit() {
                w.write_bool(block.new_cpl_coords[ch].code() != 0);
            }
            if !block.new_cpl_coords[ch].is_reuse() {
                cpl_coords_exist = true;
                w.write(2, block.cpl_master_exp[ch] as u32);
                for bnd in 0..cfg.num_cpl_bands {
                    w.write(4, block.cpl_coord_exp[ch][bnd] as u32);
                    w.write(4, block.cpl_coord_mant[ch][bnd] as u32);
                }
            }
        }
        if cfg.channel_mode == AC3_CHMODE_STEREO && enc.phase_flags_in_use && cpl_coords_exist {
            for bnd in 0..cfg.num_cpl_bands {
                w.write_bool(enc.phase_flags[bnd] != 0);
            }
        }
    }

    if cfg.channel_mode == AC3_CHMODE_STEREO {
        if blk > 0 {
            w.write_bool(block.new_rematrixing_strategy);
        }
        if block.new_rematrixing_strategy {
            for bnd in 0..block.num_rematrixing_bands {
                w.write_bool(block.rematrixing_flags[bnd] != 0);
            }
        }
    }

    for ch in 1..=cfg.fbw_channels {
        if !enc.exp_strategy[ch][blk].is_reuse() && !block.channel_in_cpl[ch] {
            w.write(6, cfg.bandwidth_code as u32);
        }
    }

    let start = usize::from(!block.cpl_in_use);
    for ch in start..=cfg.channels {
        let strategy = enc.exp_strategy[ch][blk];
        if strategy.is_reuse() {
            continue;
        }
        let cpl = ch == 0;
        w.write(4, (block.grouped_exp[ch][0] >> u8::from(cpl)) as u32);
        let nb_groups = EXPONENT_GROUP_TAB[cpl as usize][strategy.table_index()]
            [block.end_freq[ch] - cfg.start_freq[ch]] as usize;
        for i in 1..=nb_groups {
            w.write(7, block.grouped_exp[ch][i] as u32);
        }
        if ch != cfg.lfe_channel && !cpl {
            w.write(2, 0); // gain range info
        }
    }

    w.write_bool(false); // no converter SNR offset

    if block.cpl_in_use {
        if !block.new_cpl_leak.is_implicit() {
            w.write_bool(block.new_cpl_leak.code() != 0);
        }
        if !block.new_cpl_leak.is_reuse() {
            w.write(3, enc.bit_alloc.cpl_fast_leak as u32);
            w.write(3, enc.bit_alloc.cpl_slow_leak as u32);
        }
    }

    let (order, count) = coded_channel_order(cfg, block);
    for &ch in &order[..count] {
        if enc.aht_channels[ch] {
            if blk == 0 {
                w.write(2, 0); // no GAQ
                for i in cfg.start_freq[ch]..block.end_freq[ch] {
                    let hebap = block.bap[ch][i] as usize;
                    if hebap == 0 {
                        continue;
                    }
                    let bits = BITS_VS_HEBAP[hebap] as u32;
                    if hebap < 8 {
                        w.write(bits, enc.aht_vq[ch][i] as u32);
                    } else {
                        for value in enc.aht_quant[ch][i] {
                            w.write_signed(bits, value);
                        }
                    }
                }
            }
            continue;
        }
        for i in cfg.start_freq[ch]..block.end_freq[ch] {
            let q = block.qmant[ch][i];
            let b = block.bap[ch][i];
            match b {
                0 => {}
                1 | 2 | 4 => {
                    if q != 128 {
                        let bits = if b == 1 { 5 } else { 7 };
                        w.write(bits, q as u32);
                    }
                }
                _ => w.write_signed(BAP_BITS[b as usize] as u32, q as i32),
            }
        }
    }
}

/// CRC-16/ANSI over E-AC-3 frame bytes, with the polynomial in the MSB-first
/// representation used by the bitstream.
pub(crate) fn crc16(data: &[u8]) -> u16 {
    let mut crc: u32 = 0;
    for &byte in data {
        for i in (0..8).rev() {
            let b = ((byte >> i) & 1) as u32;
            let top = (crc >> 15) & 1;
            crc = if b != top {
                ((crc << 1) ^ 0x8005) & 0xFFFF
            } else {
                (crc << 1) & 0xFFFF
            };
        }
    }
    crc as u16
}

/// Pad a frame to its target size, append CRC2, and apply the reserved-bit
/// rule that keeps CRC2 from matching the sync word.
pub(crate) fn finalize(mut bytes: Vec<u8>, frame_size: usize) -> Result<Vec<u8>, EncodeError> {
    if frame_size < 4 {
        return Err(EncodeError::Internal("frame size is too small".into()));
    }
    if bytes.len() > frame_size - 2 {
        return Err(EncodeError::Internal(format!(
            "frame payload {} bytes exceeds {} bytes",
            bytes.len(),
            frame_size - 2
        )));
    }
    bytes.resize(frame_size - 2, 0);
    let mut crc = crc16(&bytes[2..]);
    if crc == 0x0B77 {
        bytes[frame_size - 3] ^= 0x01;
        crc ^= 0x8005;
    }
    bytes.push((crc >> 8) as u8);
    bytes.push(crc as u8);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc16_matches_buypass_check_value() {
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }

    #[test]
    fn finalize_pads_and_appends_crc() {
        let bytes = finalize(vec![0u8; 10], 16).unwrap();
        assert_eq!(bytes.len(), 16);
        assert_eq!(bytes[10..14], [0, 0, 0, 0]);
        assert_eq!(&bytes[14..], &crc16(&bytes[2..14]).to_be_bytes());
    }

    #[test]
    fn finalize_rejects_tiny_and_overfull_frames() {
        assert!(matches!(
            finalize(Vec::new(), 3),
            Err(EncodeError::Internal(_))
        ));
        assert!(matches!(
            finalize(vec![0u8; 11], 12),
            Err(EncodeError::Internal(_))
        ));
    }

    #[test]
    fn finalize_flips_the_reserved_bit_when_crc_equals_the_sync_word() {
        const FRAME_SIZE: usize = 8;
        let mut payload = vec![0u8; FRAME_SIZE - 2];

        let mut collision = None;
        'search: for high in 0..=255u8 {
            for low in 0..=255u8 {
                payload[FRAME_SIZE - 4] = high;
                payload[FRAME_SIZE - 3] = low;
                if crc16(&payload[2..]) == 0x0B77 {
                    collision = Some((high, low));
                    break 'search;
                }
            }
        }
        let (high, low) = collision.expect("a CRC collision must exist");
        payload[FRAME_SIZE - 4] = high;
        payload[FRAME_SIZE - 3] = low;

        let mut expected = payload.clone();
        expected[FRAME_SIZE - 3] ^= 1;
        let mut crc = crc16(&payload[2..]);
        crc ^= 0x8005;
        expected.push((crc >> 8) as u8);
        expected.push(crc as u8);

        assert_eq!(finalize(payload, FRAME_SIZE).unwrap(), expected);
    }
}
