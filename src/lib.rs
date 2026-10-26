//! `reac3e`: a standalone E-AC-3 encoder.
//!
//! The encoder implements the E-AC-3 coding tools needed for high-quality
//! 48 kHz output: MDCT, exponent coding, channel coupling, rematrixing,
//! bit allocation and mantissa packing.
//!
//! The encoder is fixed at 48 kHz.  It produces raw E-AC-3 syncframes with no
//! container framing.
//!
//! # Licensing and provenance
//!
//! LGPL-2.1-or-later.  Some algorithms and constant tables are derived from
//! the FFmpeg E-AC-3 encoder; see `NOTICE` and `LICENSE` for attribution and
//! the full licence text.

#![forbid(unsafe_code)]

mod alloc;
mod analysis;
mod bitwriter;
mod config;
mod coupling;
mod dsp;
mod encoder;
mod error;
mod header;
mod kbd;
mod mdct;
mod pack;
mod pcm;
mod quantize;
mod rematrix;
mod tables;
mod transient;

pub use config::{
    Acmod, AdConverterType, BsiMetadata, ChannelLayout, ChannelOrder, Coupling, DolbyHeadphoneMode,
    DolbySurroundExMode, DolbySurroundMode, DownmixMode, EncoderConfig, RoomType, SAMPLE_RATE,
    ServiceType,
};
pub use encoder::{Frame, FrameEncoder, StreamEncoder};
pub use error::EncodeError;
pub use pcm::PlanarFrame;
