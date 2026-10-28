//! Planar PCM input and channel-order handling.

use crate::config::{ChannelOrder, ValidatedConfig};
use crate::error::EncodeError;

/// A planar frame of `f32` samples.
///
/// `channels[i]` is one channel; all channels must have the same length.  The
/// length need not be a multiple of the encoder's frame size when using
/// [`crate::StreamEncoder`], but [`crate::FrameEncoder::encode_frame`] expects
/// exactly one full frame (or a shorter final frame, which is zero padded).
#[derive(Debug, Clone, Copy)]
pub struct PlanarFrame<'a> {
    pub channels: &'a [&'a [f32]],
    pub order: ChannelOrder,
}

impl<'a> PlanarFrame<'a> {
    pub fn new(channels: &'a [&'a [f32]], order: ChannelOrder) -> Self {
        Self { channels, order }
    }
}

/// Return the source index in `PlanarFrame::channels` for each internal AC-3
/// channel index `1..=channels`.
pub(crate) fn internal_source_map(cfg: &ValidatedConfig, order: ChannelOrder) -> [usize; 8] {
    let mut map = [0usize; 8];
    for (ch, slot) in map.iter_mut().enumerate().take(cfg.channels + 1).skip(1) {
        *slot = match order {
            ChannelOrder::Ac3 => ch - 1,
            ChannelOrder::Smpte => cfg.channel_map[ch - 1],
        };
    }
    map
}

/// Validate a planar frame and return its channels arranged in AC-3 internal
/// order.  The borrowed slices are reused directly; no audio is copied.
///
/// Index 0 (the coupling channel) and unused trailing slots are empty slices.
pub(crate) fn collect_internal_channels<'a>(
    cfg: &ValidatedConfig,
    pcm: &'a PlanarFrame<'a>,
) -> Result<Vec<&'a [f32]>, EncodeError> {
    if pcm.channels.len() != cfg.channels {
        return Err(EncodeError::Pcm(format!(
            "expected {} channels, got {}",
            cfg.channels,
            pcm.channels.len()
        )));
    }
    let len = pcm.channels.first().map_or(0, |c| c.len());
    if pcm.channels.iter().any(|c| c.len() != len) {
        return Err(EncodeError::Pcm(
            "all channels must have the same number of samples".into(),
        ));
    }

    let map = internal_source_map(cfg, pcm.order);
    let mut out = vec![&[][..]; cfg.channels + 1];
    for (ch, slot) in out.iter_mut().enumerate().skip(1) {
        *slot = pcm.channels[map[ch]];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Acmod, ChannelLayout, EncoderConfig};

    fn cfg(acmod: Acmod, lfe: bool) -> ValidatedConfig {
        EncoderConfig {
            layout: ChannelLayout { acmod, lfe },
            bitrate_bps: 448_000,
            dialnorm: -27,
            metadata: Default::default(),
            coupling: crate::config::Coupling::Auto,
            stereo_rematrixing: true,
            cutoff_hz: None,
            coupling_start: None,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn smpte_51_maps_to_ac3_order() {
        let c = cfg(Acmod::ThreeTwo, true);
        // SMPTE: L R C LFE Ls Rs -> AC-3 internal 1..6: L C R Ls Rs LFE
        let l = [1.0f32];
        let frame = PlanarFrame {
            channels: &[&l, &l, &l, &l, &l, &l],
            order: ChannelOrder::Smpte,
        };
        let d = collect_internal_channels(&c, &frame).unwrap();
        assert_eq!(d.len(), 7);
        let map = internal_source_map(&c, ChannelOrder::Smpte);
        assert_eq!(&map[1..=6], &[0, 2, 1, 4, 5, 3]);
    }

    #[test]
    fn ac3_order_is_identity() {
        let c = cfg(Acmod::ThreeTwo, true);
        let map = internal_source_map(&c, ChannelOrder::Ac3);
        assert_eq!(&map[1..=6], &[0, 1, 2, 3, 4, 5]);
    }
}
