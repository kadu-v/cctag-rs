//! The two median flavours used upstream.

use std::cmp::Ordering;

#[inline]
fn cmp(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

/// `numerical::medianRef(v)`: `sort(v); v[v.size()/2]` (upper median for even
/// sizes). The input order is destroyed, like upstream.
pub fn median_ref(v: &mut [f32]) -> f32 {
    let n = v.len();
    let k = n / 2;
    let (_, m, _) = v.select_nth_unstable_by(k, cmp);
    *m
}

/// `identification::computeMedian(vec)`: partial sort of the smallest
/// `n/2 + 1` elements; odd → middle, even → mean of the two middles computed
/// in `double` and narrowed to float.
pub fn compute_median(v: &[f32], scratch: &mut Vec<f32>) -> f32 {
    let n = v.len();
    let s = n / 2 + 1;
    scratch.clear();
    scratch.extend_from_slice(v);
    // elements 0..s of the sorted order
    scratch.select_nth_unstable_by(s - 1, cmp);
    let hi = scratch[s - 1];
    if n % 2 == 1 {
        return hi;
    }
    // need sorted[s-2] = max of the first s-1 elements
    let lo = scratch[..s - 1]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    ((hi as f64 + lo as f64) / 2.0) as f32
}
