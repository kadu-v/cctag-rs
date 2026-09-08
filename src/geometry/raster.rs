//! `geometry/EllipseFromPoints.cpp`: points on ellipses, line intersections and
//! the perimeter rasterisation estimate.

use super::ellipse::Ellipse;
use super::point::Point2;

const PI: f32 = std::f32::consts::PI;

/// `pointOnEllipse(ellipse, p)`: approximate radial projection of `p` onto the ellipse.
pub fn point_on_ellipse(e: &Ellipse, p: Point2) -> Point2 {
    let x = p.x - e.center.x;
    let y = p.y - e.center.y;
    let (s, c) = (e.angle.sin(), e.angle.cos());
    let mut tx = x * c + y * s;
    let mut ty = -x * s + y * c;
    let cs = (tx * tx / (e.a * e.a) + ty * ty / (e.b * e.b)).sqrt();
    tx /= cs;
    ty /= cs;
    Point2::new(tx * c - ty * s + e.center.x, tx * s + ty * c + e.center.y)
}

/// `ellipsePoint(ellipse, theta)`.
#[inline]
pub fn ellipse_point(e: &Ellipse, theta: f32) -> (f32, f32) {
    let x = e.a * theta.cos();
    let y = e.b * theta.sin();
    let (s, c) = (e.angle.sin(), e.angle.cos());
    (x * c - y * s + e.center.x, x * s + y * c + e.center.y)
}

/// `computeIntermediatePoints`: the four axis-extreme points rounded to ints
/// (`boost::math::round` = half away from zero).
pub fn intermediate_points(e: &Ellipse) -> [(i32, i32); 4] {
    let (s, c) = (e.angle.sin(), e.angle.cos());
    let a1 = -e.b * s - e.b * c;
    let b1 = -e.a * c + e.a * s;
    let t11 = (-a1).atan2(b1);
    let t12 = t11 + PI;
    let a2 = -e.b * s + e.b * c;
    let b2 = -e.a * c - e.a * s;
    let t21 = (-a2).atan2(b2);
    let t22 = t21 + PI;
    let r = |t: f32| {
        let (x, y) = ellipse_point(e, t);
        (x.round() as i32, y.round() as i32)
    };
    [r(t11), r(t12), r(t21), r(t22)]
}

/// `rasterizeEllipsePerimeter`: returns `(diff1 + diff2) * 2` truncated to an integer.
pub fn rasterize_ellipse_perimeter(e: &Ellipse) -> usize {
    let [p11, p12, _p21, p22] = intermediate_points(e);
    let diff1 = {
        let mx = (p22.0 - p11.0).abs() as f32;
        let my = (p22.1 - p11.1).abs() as f32;
        if mx > my { mx } else { my }
    };
    let diff2 = {
        let mx = (p12.0 - p22.0).abs() as f32;
        let my = (p12.1 - p22.1).abs() as f32;
        if mx > my { mx } else { my }
    };
    ((diff1 + diff2) * 2.0) as usize
}

/// `intersectEllipseWithLine(ellipse, v, horizontal)`: 0, 1 or 2 roots (ascending).
pub fn intersect_ellipse_with_line(e: &Ellipse, v: f32, horizontal: bool) -> ([f32; 2], usize) {
    let m = &e.matrix;
    let (a, b, c) = if horizontal {
        (
            m.at(0, 0),
            2.0 * (v * m.at(0, 1) + m.at(0, 2)),
            m.at(1, 1) * (v * v) + 2.0 * v * m.at(2, 1) + m.at(2, 2),
        )
    } else {
        (
            m.at(1, 1),
            2.0 * (v * m.at(0, 1) + m.at(1, 2)),
            m.at(0, 0) * (v * v) + 2.0 * v * m.at(0, 2) + m.at(2, 2),
        )
    };
    let disc = (b * b) / 4.0 - a * c;
    if disc > 0.0 {
        let sd = disc.sqrt();
        ([(-b / 2.0 - sd) / a, (-b / 2.0 + sd) / a], 2)
    } else if disc == 0.0 {
        ([-b / (2.0 * a), 0.0], 1)
    } else {
        ([0.0, 0.0], 0)
    }
}
