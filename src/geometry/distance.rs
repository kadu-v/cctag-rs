//! `geometry/Distance.hpp/.cpp`.

use super::ellipse::Ellipse;
use crate::edge::EdgePoint;
use crate::linalg::Mat3;

/// `distancePoints2D` for two edge points (short coordinates → float).
#[inline(always)]
pub fn distance_points_2d_i(p1: &EdgePoint, p2: &EdgePoint) -> f32 {
    let dx = (p2.x - p1.x) as f32;
    let dy = (p2.y - p1.y) as f32;
    (dx * dx + dy * dy).sqrt()
}

/// `distancePoints2D` on float coordinates.
#[inline(always)]
pub fn distance_points_2d(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    (dx * dx + dy * dy).sqrt()
}

/// `distancePointEllipseScalar(p, Q)`: squared Sampson distance of a
/// homogeneous point `[x, y, 1]` to the conic `Q` (Distance.cpp:18-41).
#[inline]
pub fn distance_point_ellipse_scalar(x: f32, y: f32, q: &Mat3) -> f32 {
    let w = 1.0f32;
    let aux = [x * x, 2.0 * x * y, 2.0 * x, y * y, 2.0 * y, 1.0];
    let tmp1 = x * q.at(0, 0) + y * q.at(0, 1) + w * q.at(0, 2);
    let tmp2 = x * q.at(0, 1) + y * q.at(1, 1) + w * q.at(1, 2);
    let denom = tmp1 * tmp1 + tmp2 * tmp2;
    let ql = [
        q.at(0, 0),
        q.at(0, 1),
        q.at(0, 2),
        q.at(1, 1),
        q.at(1, 2),
        q.at(2, 2),
    ];
    // Eigen `Vector6f::dot`: 4-lane packet reduction then the two remaining terms:
    // ((p0 + p1) + (p2 + p3)) + (p4 + p5)
    let p: [f32; 6] = core::array::from_fn(|i| aux[i] * ql[i]);
    let dot = ((p[0] + p[1]) + (p[2] + p[3])) + (p[4] + p[5]);
    (dot * dot) / denom
}

#[inline]
pub fn distance_point_ellipse(x: f32, y: f32, e: &Ellipse) -> f32 {
    distance_point_ellipse_scalar(x, y, &e.matrix)
}
