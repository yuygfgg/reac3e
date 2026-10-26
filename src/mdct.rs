//! MDCT used by the E-AC-3 encoder.
//!
//! The transform turns 512 windowed samples into 256 coefficients with a
//! scale of `-2 / 512`:
//!
//! ```text
//! X[k] = scale * sum_{n=0}^{511} x[n] cos(pi/(4*256) * (2n+1+256)(2k+1))
//! ```

use std::sync::Arc;

use rustfft::num_complex::Complex32 as C;
use rustfft::{Fft, FftPlanner};

use crate::kbd::kbd_window;
use crate::tables::{AC3_BLOCK_SIZE, AC3_MAX_COEFS};

const FFT_LEN: usize = 1024;
const WINDOW_LEN: usize = AC3_BLOCK_SIZE * 2;

/// MDCT transform plus the 512-tap symmetric KBD window.
pub(crate) struct Mdct {
    fft: Arc<dyn Fft<f32>>,
    scratch: Vec<C>,
    buf: Vec<C>,
    cos: [f32; AC3_MAX_COEFS],
    sin: [f32; AC3_MAX_COEFS],
    scale: f32,
    window: Vec<f32>,
}

impl Mdct {
    pub(crate) fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_LEN);
        let n = AC3_BLOCK_SIZE as f64;
        let scale = -2.0 / (2.0 * n);
        let mut cos = [0.0f32; AC3_MAX_COEFS];
        let mut sin = [0.0f32; AC3_MAX_COEFS];
        for (k, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
            let q = (2 * k + 1) as f64;
            let theta = std::f64::consts::PI * q * (n + 1.0) / (4.0 * n);
            *c = theta.cos() as f32;
            *s = theta.sin() as f32;
        }
        Self {
            fft,
            scratch: vec![C::default(); FFT_LEN],
            buf: vec![C::default(); FFT_LEN],
            cos,
            sin,
            scale: scale as f32,
            window: kbd_window(5.0, AC3_BLOCK_SIZE),
        }
    }

    /// The 256-tap KBD half window used for both halves of the MDCT input.
    #[cfg(test)]
    pub(crate) fn window(&self) -> &[f32] {
        &self.window
    }

    /// Apply the KBD window to a 512-sample block.
    pub(crate) fn window_block(&self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), WINDOW_LEN);
        debug_assert_eq!(out.len(), WINDOW_LEN);

        let (first, second) = out.split_at_mut(AC3_BLOCK_SIZE);
        for ((out, &input), &window) in first.iter_mut().zip(input).zip(&self.window) {
            *out = input * window;
        }
        for ((out, &input), &window) in second
            .iter_mut()
            .zip(&input[AC3_BLOCK_SIZE..])
            .zip(self.window.iter().rev())
        {
            *out = input * window;
        }
    }

    /// Window and transform one 512-sample block into 256 MDCT coefficients.
    pub(crate) fn process(&mut self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), WINDOW_LEN);
        debug_assert_eq!(out.len(), AC3_MAX_COEFS);

        for (dst, &src) in self.buf.iter_mut().zip(input.iter()) {
            *dst = C::new(src, 0.0);
        }
        for dst in self.buf.iter_mut().skip(WINDOW_LEN) {
            *dst = C::default();
        }

        self.fft
            .process_with_scratch(&mut self.buf, &mut self.scratch);

        for (k, o) in out.iter_mut().enumerate() {
            let q = 2 * k + 1;
            let re = self.buf[q].re;
            let im = self.buf[q].im;
            *o = (re * self.cos[k] + im * self.sin[k]) * self.scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(input: &[f32]) -> Vec<f32> {
        let n = AC3_BLOCK_SIZE as f64;
        let scale = -2.0 / (2.0 * n);
        let mut out = vec![0.0f32; AC3_MAX_COEFS];
        for (k, o) in out.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (j, &x) in input.iter().enumerate() {
                let a = (2 * j + 1 + AC3_BLOCK_SIZE) as f64 * (2 * k + 1) as f64;
                sum += x as f64 * (a * std::f64::consts::PI / (4.0 * n)).cos();
            }
            *o = (sum * scale) as f32;
        }
        out
    }

    #[test]
    fn matches_direct_cosine_sum() {
        let mut mdct = Mdct::new();
        let mut state = 0x1234_5678u32;
        let mut input = [0.0f32; WINDOW_LEN];
        for x in &mut input {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *x = ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
        }
        let mut got = [0.0f32; AC3_MAX_COEFS];
        mdct.process(&input, &mut got);
        let want = naive(&input);
        let mut max_err = 0.0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 2e-5, "max error {max_err}");
    }

    #[test]
    fn window_is_applied_to_both_halves() {
        let mdct = Mdct::new();
        let input = [1.0f32; WINDOW_LEN];
        let mut out = [0.0f32; WINDOW_LEN];
        mdct.window_block(&input, &mut out);
        assert_eq!(&out[..AC3_BLOCK_SIZE], mdct.window());
        let mut rev = mdct.window().to_vec();
        rev.reverse();
        assert_eq!(&out[AC3_BLOCK_SIZE..], &rev[..]);
    }
}
