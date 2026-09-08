//! `ImageCut`, `getPixelBilinear`, `cutInterpolated`, `collectCuts`.

use crate::geometry::{DirectedPoint, Point2};
use crate::image::GrayImage;

#[derive(Clone, Debug, PartialEq)]
pub struct ImageCut {
    pub start: Point2,
    pub stop: DirectedPoint,
    pub signal: Vec<f32>,
    pub out_of_bounds: bool,
    pub begin_sig: f32,
    pub end_sig: f32,
}

impl ImageCut {
    pub fn new(
        start: Point2,
        stop: DirectedPoint,
        begin_sig: f32,
        end_sig: f32,
        n_samples: usize,
    ) -> Self {
        ImageCut {
            start,
            stop,
            signal: vec![0.0; n_samples],
            out_of_bounds: false,
            begin_sig,
            end_sig,
        }
    }
}

/// `getPixelBilinear` (Identification.hpp:195-220): truncating `(int)` casts and
/// the trailing `/2` are part of the algorithm's calibration.
#[inline(always)]
pub fn pixel_bilinear(src: &GrayImage, x: f32, y: f32) -> f32 {
    let px = x as i32 as usize;
    let py = y as i32 as usize;
    let w = src.w;
    let i = py * w + px;
    let d = &src.data;
    let p1 = d[i] as f32;
    let p2 = d[i + 1] as f32;
    let p3 = d[i + w] as f32;
    let p4 = d[i + w + 1] as f32;
    let fx = x - px as f32;
    let fy = y - py as f32;
    let fx1 = 1.0 - fx;
    let fy1 = 1.0 - fy;
    let w1 = fx1 * fy1;
    let w2 = fx * fy1;
    let w3 = fx1 * fy;
    let w4 = fx * fy;
    (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4) / 2.0
}

/// `cutInterpolated` (Identification.cpp:393-456).
pub fn cut_interpolated(cut: &mut ImageCut, src: &GrayImage) {
    let diff_x = cut.stop.x - cut.start.x;
    let diff_y = cut.stop.y - cut.start.y;
    let (x_start, y_start) = if cut.begin_sig != 0.0 {
        (
            cut.start.x + diff_x * cut.begin_sig,
            cut.start.y + diff_y * cut.begin_sig,
        )
    } else {
        (cut.start.x, cut.start.y)
    };
    let (x_stop, y_stop) = if cut.end_sig != 1.0 {
        (
            cut.start.x + diff_x * cut.end_sig,
            cut.start.y + diff_y * cut.end_sig,
        )
    } else {
        (cut.stop.x, cut.stop.y)
    };
    let n = cut.signal.len();
    let step_x = (x_stop - x_start) / (n as f32 - 1.0);
    let step_y = (y_stop - y_start) / (n as f32 - 1.0);
    let mut x = x_start;
    let mut y = y_start;
    let cols_m1 = (src.w as i32 - 1) as f32;
    let rows_m1 = (src.h as i32 - 1) as f32;
    for i in 0..n {
        if x >= 1.0 && x < cols_m1 && y >= 1.0 && y < rows_m1 {
            cut.signal[i] = pixel_bilinear(src, x, y);
        } else {
            cut.out_of_bounds = true;
            break;
        }
        x += step_x;
        y += step_y;
    }
}

/// `collectCuts`: one cut per outer point, dropping those out of bounds.
pub fn collect_cuts(
    src: &GrayImage,
    center: Point2,
    outer_points: &[DirectedPoint],
    n_samples: usize,
    begin_sig: f32,
) -> Vec<ImageCut> {
    let mut cuts = Vec::with_capacity(outer_points.len());
    for op in outer_points {
        let mut cut = ImageCut::new(center, *op, begin_sig, 1.0, n_samples);
        cut_interpolated(&mut cut, src);
        if !cut.out_of_bounds {
            cuts.push(cut);
        }
    }
    cuts
}
