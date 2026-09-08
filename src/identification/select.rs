//! `selectCutCheapUniform`, `outerEdgeRefinement`, `convImageCut`
//! (Identification.cpp:543-705), plus the Boost.Accumulators statistics they use.

use super::cut::{ImageCut, cut_interpolated};
use crate::geometry::raster::point_on_ellipse;
use crate::geometry::{DirectedPoint, Ellipse, Point2};
use crate::image::GrayImage;

/// `accumulator_set<float, features<tag::variance>>`: Boost's *immediate*
/// variance, with `tag::mean` (lazy `sum / count`) as dependency. Returns
/// `(mean, variance)` with the exact recurrence
/// `var = var * (n-1) / n + (x - mean_n)^2 / (n-1)` for `n > 1`.
pub fn boost_mean_variance(xs: &[f32]) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    let mut var = 0.0f32;
    for &x in xs {
        sum += x;
        cnt += 1;
        if cnt > 1 {
            let mean = sum / cnt as f32;
            let tmp = x - mean;
            var = var * (cnt as f32 - 1.0) / cnt as f32 + tmp * tmp / (cnt as f32 - 1.0);
        }
    }
    let mean = if cnt > 0 { sum / cnt as f32 } else { f32::NAN };
    (mean, var)
}

/// `convImageCut`: edge-clamped 1-D correlation; returns `(max, argmax)`.
fn conv_image_cut(kernel: &[f32; 9], signal: &[f32]) -> (f32, f32) {
    let n = signal.len() as isize;
    let half: isize = 4;
    let mut best = f32::NEG_INFINITY;
    let mut best_i = 0usize;
    for i in 0..n {
        let mut tmp = 0.0f32;
        for (j, &k) in kernel.iter().enumerate() {
            let r = i - half + j as isize;
            let v = if r < 0 {
                signal[0]
            } else if r >= n {
                signal[(n - 1) as usize]
            } else {
                signal[r as usize]
            };
            tmp += v * k;
        }
        // std::max_element keeps the first maximum
        if tmp > best {
            best = tmp;
            best_i = i as usize;
        }
    }
    (best, best_i as f32)
}

const KERNEL_A: [f32; 9] = [
    -0.0000, -0.0003, -0.1065, -0.7863, 0.0, 0.7863, 0.1065, 0.0003, 0.0000,
];
const KERNEL_B: [f32; 9] = [
    -0.0044, -0.0540, -0.2376, -0.3450, 0.0, 0.3450, 0.2376, 0.0540, 0.0044,
];
const KERNEL_C: [f32; 9] = [
    -0.0366, -0.1113, -0.1801, -0.1594, 0.0, 0.1594, 0.1801, 0.1113, 0.0366,
];

/// `outerEdgeRefinement` (Identification.cpp:599-652): sub-pixel refinement of
/// `cut.stop` along its gradient direction.
pub fn outer_edge_refinement(
    cut: &mut ImageCut,
    src: &GrayImage,
    scale: f32,
    n_samples: usize,
) -> bool {
    let cut_len = 3.0 * 2.0f32.sqrt() * scale;
    let half = cut_len / 2.0;
    let gn = (cut.stop.dx * cut.stop.dx + cut.stop.dy * cut.stop.dy).sqrt();
    let gx = cut.stop.dx / gn;
    let gy = cut.stop.dy / gn;
    let hwx = half * gx;
    let hwy = half * gy;
    let p_start = Point2::new(cut.stop.x - hwx, cut.stop.y - hwy);
    let p_stop = DirectedPoint::new(
        cut.stop.x + half * gx,
        cut.stop.y + half * gy,
        cut.stop.dx,
        cut.stop.dy,
    );
    let mut c = ImageCut::new(p_start, p_stop, 0.0, 1.0, n_samples);
    cut_interpolated(&mut c, src);
    if c.out_of_bounds {
        return false;
    }
    // std::map<float,float> res; insert keeps the first value for equal keys; take the largest key.
    let mut entries: Vec<(f32, f32)> = Vec::with_capacity(3);
    for k in [&KERNEL_A, &KERNEL_B, &KERNEL_C] {
        let (v, loc) = conv_image_cut(k, &c.signal);
        if !entries.iter().any(|e| e.0 == v) {
            entries.push((v, loc));
        }
    }
    let mut best = entries[0];
    for e in &entries[1..] {
        if e.0 > best.0 {
            best = *e;
        }
    }
    let max_location = best.1;
    let step = cut_len / (n_samples as f32 - 1.0);
    cut.stop = DirectedPoint::new(
        p_start.x + step * max_location * gx,
        p_start.y + step * max_location * gy,
        cut.stop.dx,
        cut.stop.dy,
    );
    true
}

/// `selectCutCheapUniform` (Identification.cpp:543-596).
pub fn select_cut_cheap_uniform(
    select_size: usize,
    outer: &Ellipse,
    mut collected: Vec<ImageCut>,
    src: &GrayImage,
    scale: f32,
    n_refine: usize,
) -> Vec<ImageCut> {
    let select_size = select_size.min(collected.len());
    let var_cuts: Vec<f32> = collected
        .iter()
        .map(|c| boost_mean_variance(&c.signal).1)
        .collect();
    let var_max = var_cuts.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut ind_to_add: Vec<usize> = Vec::with_capacity(var_cuts.len());
    for (i, cut) in collected.iter_mut().enumerate() {
        let p = point_on_ellipse(outer, cut.stop.pos());
        cut.stop = DirectedPoint::new(p.x, p.y, cut.stop.dx, cut.stop.dy);
        if outer_edge_refinement(cut, src, scale, n_refine) && var_cuts[i] / var_max > 0.5 {
            ind_to_add.push(i);
        }
    }
    let step = (ind_to_add.len() as f32 / select_size as f32).max(1.0);
    let mut selected = Vec::with_capacity(select_size);
    let mut k = 0usize;
    loop {
        let idx = (k as f32 * step) as usize;
        if idx < ind_to_add.len() && selected.len() < select_size {
            selected.push(collected[ind_to_add[idx]].clone());
        } else {
            break;
        }
        k += 1;
    }
    selected
}
