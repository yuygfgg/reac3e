//! A/52 §8.2.2 transient detector for block switching.

const SILENCE_THRESHOLD: f64 = 100.0 / 32768.0;
const T1: f64 = 0.1;
const T2: f64 = 0.075;
const T3: f64 = 0.05;

#[derive(Clone, Copy, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Per-channel state.  The filter memory and the previous peak tree carry
/// across 256-sample segments, so one instance belongs to one continuous
/// channel stream.
pub(crate) struct TransientDetector {
    stages: [Biquad; 2],
    prev1: f64,
    prev2: f64,
    prev3: f64,
    first: bool,
}

impl TransientDetector {
    pub(crate) fn new() -> Self {
        let fs = 48_000.0f64;
        let w0 = 2.0 * std::f64::consts::PI * 8000.0 / fs;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * (0.5f64).sqrt());
        let a0 = 1.0 + alpha;
        let stage = Biquad {
            b0: (1.0 + cos_w0) / 2.0 / a0,
            b1: -(1.0 + cos_w0) / a0,
            b2: (1.0 + cos_w0) / 2.0 / a0,
            a1: -2.0 * cos_w0 / a0,
            a2: (1.0 - alpha) / a0,
            ..Biquad::default()
        };
        Self {
            stages: [stage; 2],
            prev1: 0.0,
            prev2: 0.0,
            prev3: 0.0,
            first: true,
        }
    }

    /// Feed the 256 new PCM samples of one block.
    pub(crate) fn detect(&mut self, pcm: &[f32]) -> bool {
        debug_assert!(pcm.len() >= 256);
        let suppress = self.first;
        self.first = false;
        let mut filtered = [0.0f64; 256];
        for (dst, &sample) in filtered.iter_mut().zip(&pcm[..256]) {
            let mut x = sample as f64;
            for stage in &mut self.stages {
                x = stage.process(x);
            }
            *dst = x;
        }

        let peak = |slice: &[f64]| slice.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let p3 = [
            peak(&filtered[0..64]),
            peak(&filtered[64..128]),
            peak(&filtered[128..192]),
            peak(&filtered[192..256]),
        ];
        let p2 = [p3[0].max(p3[1]), p3[2].max(p3[3])];
        let p1 = p2[0].max(p2[1]);

        let mut transient = false;
        if p1 >= SILENCE_THRESHOLD {
            if p1 * T1 > self.prev1 {
                transient = true;
            }
            if p2[0] * T2 > self.prev2 {
                transient = true;
            }
            if p2[1] * T2 > p2[0] {
                transient = true;
            }
            if p3[0] * T3 > self.prev3 {
                transient = true;
            }
            for k in 1..p3.len() {
                if p3[k] * T3 > p3[k - 1] {
                    transient = true;
                }
            }
        }

        self.prev1 = p1;
        self.prev2 = p2[1];
        self.prev3 = p3[3];
        !suppress && transient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_click_after_silence() {
        let mut detector = TransientDetector::new();
        let silence = [0.0f32; 256];
        for _ in 0..4 {
            let _ = detector.detect(&silence);
        }
        let mut click = [0.0f32; 256];
        for sample in click.iter_mut().take(64) {
            *sample = 0.8;
        }
        assert!(detector.detect(&click));
    }
}
