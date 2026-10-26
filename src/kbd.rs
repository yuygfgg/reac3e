//! Kaiser-Bessel derived window.

// Reference coefficients are kept at full source precision.
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

const P1: [f64; 15] = [
    -2.2335582639474375249e+15,
    -5.5050369673018427753e+14,
    -3.2940087627407749166e+13,
    -8.4925101247114157499e+11,
    -1.1912746104985237192e+10,
    -1.0313066708737980747e+08,
    -5.9545626019847898221e+05,
    -2.4125195876041896775e+03,
    -7.0935347449210549190e+00,
    -1.5453977791786851041e-02,
    -2.5172644670688975051e-05,
    -3.0517226450451067446e-08,
    -2.6843448573468483278e-11,
    -1.5982226675653184646e-14,
    -5.2487866627945699800e-18,
];

const Q1: [f64; 6] = [
    -2.2335582639474375245e+15,
    7.8858692566751002988e+12,
    -1.2207067397808979846e+10,
    1.0377081058062166144e+07,
    -4.8527560179962773045e+03,
    1.0,
];

const P2: [f64; 7] = [
    -2.2210262233306573296e-04,
    1.3067392038106924055e-02,
    -4.4700805721174453923e-01,
    5.5674518371240761397e+00,
    -2.3517945679239481621e+01,
    3.1611322818701131207e+01,
    -9.6090021968656180000e+00,
];

const Q2: [f64; 8] = [
    -5.5194330231005480228e-04,
    3.2547697594819615062e-02,
    -1.1151759188741312645e+00,
    1.3982595353892851542e+01,
    -6.0228002066743340583e+01,
    8.5539563258012929600e+01,
    -3.1446690275135491500e+01,
    1.0,
];

fn eval_poly(coeff: &[f64], x: f64) -> f64 {
    let mut sum = coeff[coeff.len() - 1];
    for &c in coeff[..coeff.len() - 1].iter().rev() {
        sum *= x;
        sum += c;
    }
    sum
}

/// Modified Bessel function of the first kind, order zero.
///
/// Uses the minimax rational approximation from Blair and Edwards,
/// Chalk River Report AECL-4928, 1974.
pub(crate) fn bessel_i0(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        let y = 1.0 / x - 1.0 / 15.0;
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        let factor = x.exp() / x.sqrt();
        factor * r
    }
}

/// Generate a Kaiser-Bessel derived window of `n` samples (half of the
/// symmetric MDCT window).  E-AC-3 uses `alpha = 5.0` and `n = 256`.
pub(crate) fn kbd_window(alpha: f32, n: usize) -> Vec<f32> {
    let n2 = n / 2;
    let mut temp = vec![0.0f64; n2 + 1];
    let mut sum = 0.0f64;
    let alpha = alpha as f64;
    let alpha2 = 4.0 * (alpha * PI / n as f64) * (alpha * PI / n as f64);

    for (i, t) in temp.iter_mut().enumerate().take(n2 + 1) {
        let tmp = i as f64 * (n - i) as f64 * alpha2;
        *t = bessel_i0(tmp.sqrt());
        sum += *t * (1.0 + u8::from(i != 0 && i < n2) as f64);
    }
    let scale = 1.0 / (sum + 1.0);

    let mut window = vec![0.0f32; n];
    let mut acc = 0.0f64;
    for i in 0..=n2 {
        acc += temp[i];
        window[i] = (acc * scale).sqrt() as f32;
    }
    for i in n2 + 1..n {
        acc += temp[n - i];
        window[i] = (acc * scale).sqrt() as f32;
    }
    window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_reference_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008).abs() < 1e-12);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_446).abs() < 1e-7);
        assert!((bessel_i0(20.0) - 4.355_828_314_863_939e7).abs() < 1.0);
    }

    #[test]
    fn kbd_window_shape() {
        let w = kbd_window(5.0, 256);
        assert_eq!(w.len(), 256);
        // The window starts at zero and rises monotonically to one at the top.
        assert!(w[0] < 1e-3);
        assert!((w[255] - 1.0).abs() < 1e-6);
        assert!(w.windows(2).all(|p| p[0] <= p[1]));
        // The two halves are not the same sequence length; the MDCT uses a
        // symmetric 512-tap window built from w and its reverse.
        assert!((w[128] - 0.707_106_77).abs() < 0.1);
    }
}
