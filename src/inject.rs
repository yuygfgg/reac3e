//! Post-encode Atmos metadata injection.
//!
//! [`inject_metadata`] inserts `addbsi` and an EMDF skip field into a frame
//! produced by [`crate::FrameEncoder`], using the exact bit positions recorded
//! in [`crate::FrameEdits`].  No audio-block parsing is involved, so it works
//! for every syntax the encoder emits (block switching, coupling, all block
//! counts, AHT).
//!
//! This is the two-pass counterpart of
//! [`crate::FrameEncoder::encode_frame_with_metadata`]: encode the plain core,
//! analyse it, then inject the metadata.

use crate::bitwriter::BitWriter;
use crate::encoder::{
    Frame, FrameEdits, MAX_FRAME_BYTES, SkipMetadata, patch_frame_size, validate_skip_metadata,
};
use crate::error::EncodeError;
use crate::pack;

/// Inject metadata into an already-encoded, metadata-free frame.
///
/// The frame is grown by the `addbsi`/skip-field overhead; `frmsiz` and CRC2 are
/// recomputed.  Audio bits are copied verbatim, so decoded PCM is unchanged.
pub fn inject_metadata(frame: &Frame, metadata: &SkipMetadata) -> Result<Frame, EncodeError> {
    let edits = &frame.edits;
    validate_skip_metadata(metadata, edits.num_blocks)?;
    if edits.addbsi_len != 0 || edits.emdf_start_bit.is_some() {
        return Err(EncodeError::Config(
            "frame already carries addbsi or an EMDF skip field".into(),
        ));
    }
    if frame.bytes.len() < 4 || frame.bytes.len() != edits.frame_size {
        return Err(EncodeError::Internal(
            "frame bytes do not match the recorded frame size".into(),
        ));
    }

    let original_data_bits = (frame.bytes.len() - 2) * 8;
    let addbsi = &metadata.addbsi;
    let emdf = &metadata.emdf;
    let addbsi_bits = if addbsi.is_empty() {
        0
    } else {
        6 + addbsi.len() * 8
    };
    let skip_enabled = !emdf.is_empty();

    let mut writer = BitWriter::with_capacity(frame.bytes.len() + addbsi.len() + emdf.len() + 16);
    let mut cursor = 0usize;

    // BSI addbsi: replace the single `addbsie` bit with flag + length + bytes.
    writer.copy_from(&frame.bytes, cursor, edits.addbsie_bit - cursor);
    cursor = edits.addbsie_bit;
    if addbsi.is_empty() {
        writer.write(1, 0);
    } else {
        writer.write(1, 1);
        writer.write(6, (addbsi.len() - 1) as u32);
        for &byte in addbsi {
            writer.write(8, u32::from(byte));
        }
    }
    cursor += 1;

    // audfrm skipFieldSyntaxEnabled: a single-bit replacement.
    writer.copy_from(&frame.bytes, cursor, edits.skipfld_bit - cursor);
    cursor = edits.skipfld_bit;
    writer.write(1, u32::from(skip_enabled));
    cursor += 1;

    let mut new_points = Vec::with_capacity(edits.num_blocks);
    for (block, &point) in edits.block_skip_points.iter().enumerate() {
        if point < cursor {
            return Err(EncodeError::Internal(
                "skip point ordering is inconsistent".into(),
            ));
        }
        writer.copy_from(&frame.bytes, cursor, point - cursor);
        cursor = point;
        new_points.push(writer.bits_written());
        if skip_enabled {
            if block == metadata.target_block {
                writer.write(1, 1); // skiple
                writer.write(9, emdf.len() as u32); // skipl
                for &byte in emdf {
                    writer.write(8, u32::from(byte));
                }
            } else {
                writer.write(1, 0); // skiple
            }
        }
    }

    // Copy the remaining audio/padding bits up to the original CRC2.
    if cursor > original_data_bits {
        return Err(EncodeError::Internal(
            "skip points extend past the frame data".into(),
        ));
    }
    writer.copy_from(&frame.bytes, cursor, original_data_bits - cursor);

    let skip_bits = if skip_enabled {
        edits.num_blocks + 9 + emdf.len() * 8
    } else {
        0
    };
    let audio_end_bit = edits.audio_end_bit + addbsi_bits + skip_bits;

    let mut frame_size = (writer.bits_written() + 16).div_ceil(8);
    if !frame_size.is_multiple_of(2) {
        frame_size += 1;
    }
    if frame_size > MAX_FRAME_BYTES {
        return Err(EncodeError::Config(format!(
            "frame with metadata is {frame_size} bytes; the E-AC-3 limit is {MAX_FRAME_BYTES}"
        )));
    }

    let mut bytes = writer.into_bytes();
    patch_frame_size(&mut bytes, frame_size);
    let bytes = pack::finalize(bytes, frame_size)?;

    Ok(Frame {
        bytes,
        index: frame.index,
        edits: FrameEdits {
            num_blocks: edits.num_blocks,
            frame_size,
            infomdate_bit: edits.infomdate_bit,
            info_end_bit: edits.info_end_bit,
            addbsie_bit: edits.addbsie_bit,
            addbsi_len: addbsi.len(),
            skipfld_bit: edits.skipfld_bit + addbsi_bits,
            emdf_start_bit: skip_enabled.then(|| new_points[metadata.target_block] + 10),
            audio_end_bit,
            block_skip_points: new_points,
        },
    })
}
