//! Small signal-processing helpers shared by the coding tools.

/// Sum of squares for a stereo pair: `[ΣL², ΣR², Σ(L+R)², Σ(L-R)²]`.
///
/// `a` and `b` must each contain at least `len` samples.
pub(crate) fn sum_square_butterfly(a: &[f32], b: &[f32], len: usize) -> [f32; 4] {
    let mut sum = [0.0f32; 4];
    for (&left, &right) in a.iter().zip(b).take(len) {
        let mid = left + right;
        let side = left - right;
        sum[0] = left.mul_add(left, sum[0]);
        sum[1] = right.mul_add(right, sum[1]);
        sum[2] = mid.mul_add(mid, sum[2]);
        sum[3] = side.mul_add(side, sum[3]);
    }
    sum
}
