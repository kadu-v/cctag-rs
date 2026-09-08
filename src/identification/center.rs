//! Imaged-centre / homography refinement (Identification.cpp:213-277, 714-1154).

use super::cut::{ImageCut, pixel_bilinear};
use crate::geometry::conditioner::{condition, conditioner_from_ellipse};
use crate::geometry::{Ellipse, Point2};
use crate::image::GrayImage;
use crate::linalg::Mat3;
use crate::params::Params;
use crate::robust::median::compute_median;

#[inline(always)]
fn apply_homography(h: &Mat3, x: f32, y: f32) -> (f32, f32) {
    let u = h.at(0, 0) * x + h.at(0, 1) * y + h.at(0, 2);
    let v = h.at(1, 0) * x + h.at(1, 1) * y + h.at(1, 2);
    let w = h.at(2, 0) * x + h.at(2, 1) * y + h.at(2, 2);
    (u / w, v / w)
}

/// `extractSignalUsingHomography`: note the `out_of_bounds` flag is only ever
/// set (never cleared) and stale samples remain in the signal — as upstream.
pub fn extract_signal_using_homography(
    cut: &mut ImageCut,
    src: &GrayImage,
    h: &Mat3,
    h_inv: &Mat3,
) {
    let (bpx, bpy) = apply_homography(h_inv, cut.stop.x, cut.stop.y);
    let (x_start, y_start) = if cut.begin_sig != 0.0 {
        (bpx * cut.begin_sig, bpy * cut.begin_sig)
    } else {
        (0.0, 0.0)
    };
    let (x_stop, y_stop) = if cut.end_sig != 1.0 {
        (bpx * cut.end_sig, bpy * cut.end_sig)
    } else {
        (bpx, bpy)
    };
    let n = cut.signal.len();
    let step_x = (x_stop - x_start) / (n as f32 - 1.0);
    let step_y = (y_stop - y_start) / (n as f32 - 1.0);
    let mut x = x_start;
    let mut y = y_start;
    let cols_m1 = (src.w as i32 - 1) as f32;
    let rows_m1 = (src.h as i32 - 1) as f32;
    for i in 0..n {
        let (xr, yr) = apply_homography(h, x, y);
        if xr >= 0.0 && xr < cols_m1 && yr >= 0.0 && yr < rows_m1 {
            cut.signal[i] = pixel_bilinear(src, xr, yr);
        } else {
            cut.out_of_bounds = true;
        }
        x += step_x;
        y += step_y;
    }
}

/// `getSignals`.
pub fn get_signals(cuts: &mut [ImageCut], h: &Mat3, src: &GrayImage) {
    let h_inv = h.inverse();
    for c in cuts.iter_mut() {
        extract_signal_using_homography(c, src, h, &h_inv);
    }
}

/// `computeHomographyFromEllipseAndImagedCenter` (Identification.cpp:735-794).
/// NaNs from `sqrt` of negative arguments propagate silently, as upstream.
pub fn compute_homography_from_ellipse_and_imaged_center(
    ellipse: &Ellipse,
    center: Point2,
) -> Mat3 {
    let (canonic, t_can, t_inv_can) = ellipse.canonic_form();
    let (xc, yc) = apply_homography(&t_can, center.x, center.y);
    let q11 = canonic.at(0, 0);
    let q22 = canonic.at(1, 1);
    let q33 = canonic.at(2, 2);
    let mut h = Mat3::ZERO;
    h.set(0, 0, q33);
    h.set(1, 0, 0.0);
    h.set(2, 0, -q11 * xc);
    h.set(0, 1, q22 * xc * yc);
    h.set(1, 1, -q11 * xc * xc - q33);
    h.set(2, 1, q22 * yc);
    h.set(0, 2, -q33 * xc);
    h.set(1, 2, -q33 * yc);
    h.set(2, 2, -q33);
    let d0 = (q22 * q33 / q11 * (q11 * xc * xc + q22 * yc * yc + q33)).sqrt();
    let d1 = q33;
    let d2 = (-q22 * (q11 * xc * xc + q33)).sqrt();
    let diag = [d0, d1, d2];
    for i in 0..3 {
        for j in 0..3 {
            let v = h.at(i, j) * diag[j];
            h.set(i, j, v);
        }
    }
    t_inv_can.mul(&h)
}

/// `costFunctionGlob`: mean over readable cut pairs of the summed squared
/// sample differences. `std::pow(float, 2)` is evaluated in double and the
/// accumulation narrows back to float at every step.
pub fn cost_function_glob(h: &Mat3, cuts: &mut [ImageCut], src: &GrayImage) -> Option<f32> {
    get_signals(cuts, h, src);
    let mut res = 0.0f32;
    let mut n_pairs = 0usize;
    let n = cuts.len();
    for i in 0..n.saturating_sub(1) {
        for j in i + 1..n {
            if !cuts[i].out_of_bounds && !cuts[j].out_of_bounds {
                let a = &cuts[i].signal;
                let b = &cuts[j].signal;
                for k in 0..b.len() {
                    let d = (a[k] - b[k]) as f64;
                    res = (res as f64 + d * d) as f32;
                }
                n_pairs += 1;
            }
        }
    }
    if n_pairs == 0 {
        None
    } else {
        Some(res / n_pairs as f32)
    }
}

/// `getNearbyPoints` (GRID pattern only).
pub fn get_nearby_points(
    ellipse: &Ellipse,
    center: Point2,
    neighbour_size: f32,
    grid_n: usize,
    frozen_mean_ab: Option<f32>,
) -> Vec<Point2> {
    let t = conditioner_from_ellipse(ellipse, frozen_mean_ab);
    let t_inv = t.inverse();
    // transformedEllipse = projectiveTransform(mInvT, ellipse) => matrix = mInvT^T * C * mInvT
    let te = Ellipse::from_matrix(t_inv.sandwich(&ellipse.matrix)).unwrap_or_default();
    let ns = neighbour_size * te.a.max(te.b);
    let cc = condition(center, &t);
    let width = ns;
    let half = width / 2.0;
    let step = width / (grid_n as f32 - 1.0);
    let mut pts = Vec::with_capacity(grid_n * grid_n);
    for i in 0..grid_n {
        for j in 0..grid_n {
            pts.push(Point2::new(
                cc.x - half + i as f32 * step,
                cc.y - half + j as f32 * step,
            ));
        }
    }
    for p in pts.iter_mut() {
        *p = condition(*p, &t_inv);
    }
    pts
}

/// Extract the rectified signal of one cut for homography `h` into `out`
/// (same arithmetic as [`extract_signal_using_homography`]), returning whether
/// any sample fell out of bounds. Samples out of bounds are left untouched.
#[inline]
fn extract_signal_into(
    cut: &ImageCut,
    src: &GrayImage,
    h: &Mat3,
    h_inv: &Mat3,
    out: &mut [f32],
) -> bool {
    let (bpx, bpy) = apply_homography(h_inv, cut.stop.x, cut.stop.y);
    let (x_start, y_start) = if cut.begin_sig != 0.0 {
        (bpx * cut.begin_sig, bpy * cut.begin_sig)
    } else {
        (0.0, 0.0)
    };
    let (x_stop, y_stop) = if cut.end_sig != 1.0 {
        (bpx * cut.end_sig, bpy * cut.end_sig)
    } else {
        (bpx, bpy)
    };
    let n = out.len();
    let step_x = (x_stop - x_start) / (n as f32 - 1.0);
    let step_y = (y_stop - y_start) / (n as f32 - 1.0);
    let mut x = x_start;
    let mut y = y_start;
    let cols_m1 = (src.w as i32 - 1) as f32;
    let rows_m1 = (src.h as i32 - 1) as f32;
    let mut oob = false;
    for o in out.iter_mut() {
        let (xr, yr) = apply_homography(h, x, y);
        if xr >= 0.0 && xr < cols_m1 && yr >= 0.0 && yr < rows_m1 {
            *o = pixel_bilinear(src, xr, yr);
        } else {
            oob = true;
        }
        x += step_x;
        y += step_y;
    }
    oob
}

/// Scratch for the batched grid evaluation.
#[derive(Default)]
pub struct CenterScratch {
    /// signals laid out `[cut][sample][grid]`
    sig: Vec<f32>,
    /// `[grid][cut]` readable flags (after the sticky update at that grid point)
    readable: Vec<bool>,
    hs: Vec<(Mat3, Mat3)>,
    res: Vec<f64>,
    valid: Vec<bool>,
    tmp: Vec<f32>,
    n_pairs: Vec<u32>,
}

/// `imageCenterOptimizationGlob`: one 5x5 grid pass. Returns `false` when no
/// grid point produced a readable cut pair.
///
/// Bit-exact with the upstream sequential evaluation, but organised so that the
/// 25 grid points' cost chains (`res = float(double(res) + d*d)`, a strictly
/// sequential dependency chain upstream) are advanced in lock-step, which turns
/// a latency-bound loop into a throughput-bound one.
#[allow(clippy::too_many_arguments)]
pub fn image_center_optimization_glob(
    h: &mut Mat3,
    cuts: &mut [ImageCut],
    center: &mut Point2,
    min_res: &mut f32,
    neighbour_size: f32,
    src: &GrayImage,
    outer: &Ellipse,
    params: &Params,
    frozen_mean_ab: Option<f32>,
    sc: &mut CenterScratch,
) -> bool {
    let nearby = get_nearby_points(
        outer,
        *center,
        neighbour_size,
        params.imaged_center_n_grid_sample,
        frozen_mean_ab,
    );
    let g = nearby.len();
    let nc = cuts.len();
    let ns = if nc > 0 { cuts[0].signal.len() } else { 0 };
    *min_res = f32::MAX;

    // 1. homographies (and inverses) for every grid point
    sc.hs.clear();
    for p in &nearby {
        let th = compute_homography_from_ellipse_and_imaged_center(outer, *p);
        sc.hs.push((th, th.inverse()));
    }
    // 2. signals for every grid point, sequentially, with the sticky out-of-bounds flags
    sc.sig.resize(nc * ns * g, 0.0);
    sc.readable.resize(g * nc, false);
    sc.tmp.resize(ns, 0.0);
    let tmp = &mut sc.tmp;
    for (gi, (th, th_inv)) in sc.hs.iter().enumerate() {
        for (ci, cut) in cuts.iter_mut().enumerate() {
            // start from the cut's current (stale) samples, exactly like upstream
            tmp.copy_from_slice(&cut.signal);
            let oob = extract_signal_into(cut, src, th, th_inv, tmp);
            cut.signal.copy_from_slice(tmp);
            if oob {
                cut.out_of_bounds = true;
            }
            sc.readable[gi * nc + ci] = !cut.out_of_bounds;
            for (k, v) in tmp.iter().enumerate() {
                sc.sig[(ci * ns + k) * g + gi] = *v;
            }
        }
    }
    // 3. cost chains in lock-step over the grid points
    sc.res.clear();
    sc.res.resize(g, 0.0);
    sc.n_pairs.resize(g, 0);
    sc.n_pairs.fill(0);
    let n_pairs = &mut sc.n_pairs;
    sc.valid.clear();
    sc.valid.resize(g, false);
    for i in 0..nc.saturating_sub(1) {
        for j in i + 1..nc {
            let mut any = false;
            for gi in 0..g {
                let v = sc.readable[gi * nc + i] && sc.readable[gi * nc + j];
                sc.valid[gi] = v;
                any |= v;
                if v {
                    n_pairs[gi] += 1;
                }
            }
            if !any {
                continue;
            }
            let base_i = i * ns * g;
            let base_j = j * ns * g;
            if g == 25 {
                accumulate_25(
                    &sc.sig[base_i..base_i + ns * g],
                    &sc.sig[base_j..base_j + ns * g],
                    &sc.valid,
                    &mut sc.res,
                );
                continue;
            }
            for k in 0..ns {
                let a = &sc.sig[base_i + k * g..base_i + k * g + g];
                let b = &sc.sig[base_j + k * g..base_j + k * g + g];
                for gi in 0..g {
                    // res = (float)((double)res + pow(d, 2)); the f32 rounding at every
                    // step is what upstream does, kept per grid point.
                    let d = (a[gi] - b[gi]) as f64;
                    let r = (sc.res[gi] + d * d) as f32;
                    sc.res[gi] = if sc.valid[gi] { r as f64 } else { sc.res[gi] };
                }
            }
        }
    }
    // 4. argmin (strict <, grid order)
    let mut has_solution = false;
    let mut opt_point = Point2::default();
    let mut opt_h = Mat3::ZERO;
    for gi in 0..g {
        if n_pairs[gi] == 0 {
            continue;
        }
        has_solution = true;
        let res = (sc.res[gi] as f32) / n_pairs[gi] as f32;
        if res < *min_res {
            *min_res = res;
            opt_point = nearby[gi];
            opt_h = sc.hs[gi].0;
        }
    }
    *center = opt_point;
    *h = opt_h;
    has_solution
}

// Each lane owns a grid point, so SIMD never reassociates a reduction.
// Compute invalid lanes locally and discard them once per pair; this removes
// the validity branch from every sample while preserving the sticky flags.
fn accumulate_25(a: &[f32], b: &[f32], valid: &[bool], res: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 25, 0);
    assert_eq!(valid.len(), 25);
    assert_eq!(res.len(), 25);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 has NEON; lengths checked above ensure every 2-lane
        // load/store is in bounds. The last (25th) lane is scalar.
        unsafe {
            accumulate_25_neon(a, b, valid, res);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    accumulate_25_scalar(a, b, valid, res);
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn accumulate_25_scalar(a: &[f32], b: &[f32], valid: &[bool], res: &mut [f64]) {
    let mut r: [f64; 25] = res.try_into().unwrap();
    for (a, b) in a.chunks_exact(25).zip(b.chunks_exact(25)) {
        for gi in 0..25 {
            let d = (a[gi] - b[gi]) as f64;
            r[gi] = (r[gi] + d * d) as f32 as f64;
        }
    }
    for gi in 0..25 {
        if valid[gi] {
            res[gi] = r[gi];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_25_neon(a: &[f32], b: &[f32], valid: &[bool], res: &mut [f64]) {
    use std::arch::aarch64::*;
    // SAFETY: caller checks all lengths; q*2+1 <= 23 and chunks have 25 entries.
    unsafe {
        let mut r: [float64x2_t; 12] = core::array::from_fn(|q| vld1q_f64(res.as_ptr().add(q * 2)));
        let mut tail = res[24];
        for (a, b) in a.chunks_exact(25).zip(b.chunks_exact(25)) {
            for q in 0..12 {
                let diff = vsub_f32(
                    vld1_f32(a.as_ptr().add(q * 2)),
                    vld1_f32(b.as_ptr().add(q * 2)),
                );
                let d = vcvt_f64_f32(diff);
                // Separate multiply/add, then narrow every term, exactly as
                // float(double(res) + double(float(a-b))^2). No FMA.
                r[q] = vcvt_f64_f32(vcvt_f32_f64(vaddq_f64(r[q], vmulq_f64(d, d))));
            }
            let d = (a[24] - b[24]) as f64;
            tail = (tail + d * d) as f32 as f64;
        }
        let mut out = [0.0; 25];
        for q in 0..12 {
            vst1q_f64(out.as_mut_ptr().add(q * 2), r[q]);
        }
        out[24] = tail;
        for gi in 0..25 {
            if valid[gi] {
                res[gi] = out[gi];
            }
        }
    }
}

/// `refineConicFamilyGlob` (Identification.cpp:809-961).
pub fn refine_conic_family_glob(
    h: &mut Mat3,
    center: &mut Point2,
    cuts: &mut [ImageCut],
    src: &GrayImage,
    outer: &Ellipse,
    params: &Params,
    residual: &mut f32,
) -> bool {
    let mut neighbour_size = params.imaged_center_neighbour_size;
    let grid_n = params.imaged_center_n_grid_sample;
    let max_semi = outer.a.max(outer.b);
    let frozen: Option<f32> = None; // upstream's `static meanAB` quirk is not reproduced
    let mut sc = CenterScratch::default();
    while ((neighbour_size * max_semi) as f64) > 0.02 {
        if image_center_optimization_glob(
            h,
            cuts,
            center,
            residual,
            neighbour_size,
            src,
            outer,
            params,
            frozen,
            &mut sc,
        ) {
            neighbour_size /= ((grid_n - 1) / 2) as f32;
        } else {
            return false;
        }
    }
    get_signals(cuts, h, src);

    let correct: Vec<usize> = (0..cuts.len())
        .filter(|&i| !cuts[i].out_of_bounds)
        .collect();
    let signal_size = cuts[0].signal.len();
    let mut bar_code = vec![0.0f32; signal_size];
    let mut along = vec![0.0f32; correct.len()];
    let mut scratch = Vec::new();
    for s in 0..signal_size {
        for (k, &ci) in correct.iter().enumerate() {
            along[k] = cuts[ci].signal[s];
        }
        bar_code[s] = compute_median(&along, &mut scratch);
    }
    let v_min = bar_code.iter().copied().fold(f32::INFINITY, f32::min);
    let v_max = bar_code.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let magnitude = v_max - v_min;
    *residual = residual.sqrt() / magnitude;
    // `if (residual > 2.7f) return false; else return true;` — a NaN residual counts as converged.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let converged = !(*residual > 2.7);
    converged
}

/// Straightforward sequential evaluation of one grid pass (upstream structure:
/// `getSignals` + pairwise cost per grid point). Kept as the reference for the
/// batched implementation; the two must agree bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn image_center_optimization_glob_reference(
    h: &mut Mat3,
    cuts: &mut [ImageCut],
    center: &mut Point2,
    min_res: &mut f32,
    neighbour_size: f32,
    src: &GrayImage,
    outer: &Ellipse,
    params: &Params,
    frozen_mean_ab: Option<f32>,
) -> bool {
    let nearby = get_nearby_points(
        outer,
        *center,
        neighbour_size,
        params.imaged_center_n_grid_sample,
        frozen_mean_ab,
    );
    *min_res = f32::MAX;
    let mut has_solution = false;
    let mut opt_point = Point2::default();
    let mut opt_h = Mat3::ZERO;
    for p in nearby {
        let th = compute_homography_from_ellipse_and_imaged_center(outer, p);
        if let Some(res) = cost_function_glob(&th, cuts, src) {
            has_solution = true;
            if res < *min_res {
                *min_res = res;
                opt_point = p;
                opt_h = th;
            }
        }
    }
    *center = opt_point;
    *h = opt_h;
    has_solution
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_grid_matches_scalar_for_masks_and_lengths() {
        let mut seed = 912u32;
        for ns in [0, 1, 7, 99, 100, 101] {
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / 65536.0
            };
            let a: Vec<_> = (0..ns * 25).map(|_| next()).collect();
            let b: Vec<_> = (0..ns * 25).map(|_| next()).collect();
            for mask in [0u32, u32::MAX, 0x1555555, 0x1f00ff] {
                let valid: Vec<_> = (0..25).map(|i| (mask >> i) & 1 != 0).collect();
                let mut expected: Vec<_> = (0..25).map(|i| (i as f32 * 1.3) as f64).collect();
                let mut actual = expected.clone();
                accumulate_25_scalar(&a, &b, &valid, &mut expected);
                accumulate_25(&a, &b, &valid, &mut actual);
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "ns={ns} mask={mask:x}"
                );
            }
        }
    }
}
