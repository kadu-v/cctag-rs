//! Inline helpers from `EllipseGrowing.hpp` and `computeHull`.

use super::ellipse::{Ellipse, GeomError};
use super::point::Point2;
use crate::linalg::Mat3;

#[inline(always)]
fn quad(p: &[f32; 3], m: &Mat3) -> f32 {
    // p.dot(M * p): the product uses Eigen's tree order per coefficient, the dot
    // of two contiguous 3-vectors is vectorised with a 2-lane packet: (p0+p1)+p2.
    let mp = m.mul_vec(p);
    (p[0] * mp[0] + p[1] * mp[1]) + p[2] * mp[2]
}

/// `isInEllipse(ellipse, p)`: `(p'Qp) * (c'Qc) > 0`.
#[inline]
pub fn is_in_ellipse(e: &Ellipse, p: Point2) -> bool {
    let s1 = quad(&p.hom(), &e.matrix);
    let s2 = quad(&e.center.hom(), &e.matrix);
    s1 * s2 > 0.0
}

/// `isOverlappingEllipses`.
pub fn is_overlapping_ellipses(e1: &Ellipse, e2: &Ellipse) -> bool {
    is_in_ellipse(e1, e2.center) || is_in_ellipse(e2, e1.center)
}

/// `isInHull(qIn, qOut, p)`: `(p'Qin p) * (p'Qout p) < 0`.
#[inline]
pub fn is_in_hull(q_in: &Ellipse, q_out: &Ellipse, x: f32, y: f32) -> bool {
    let p = [x, y, 1.0];
    let s1 = quad(&p, &q_in.matrix);
    let s2 = quad(&p, &q_out.matrix);
    s1 * s2 < 0.0
}

/// `isOnTheSameSide(p1, p2, line)`.
#[inline]
pub fn is_on_the_same_side(p1: Point2, p2: Point2, line: &[f32; 3]) -> bool {
    let a = (p1.x * line[0] + p1.y * line[1]) + line[2];
    let b = (p2.x * line[0] + p2.y * line[1]) + line[2];
    a * b > 0.0
}

/// `computeHull(ellipse, delta) -> (qIn, qOut)`.
pub fn compute_hull(e: &Ellipse, delta: f32) -> Result<(Ellipse, Ellipse), GeomError> {
    let q_in = Ellipse::from_params(
        e.center,
        (e.a - delta).max(0.001),
        (e.b - delta).max(0.001),
        e.angle,
    )?;
    let q_out = Ellipse::from_params(e.center, e.a + delta, e.b + delta, e.angle)?;
    Ok((q_in, q_out))
}
