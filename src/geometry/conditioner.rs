//! `optimization/conditioner.hpp`.

use super::ellipse::Ellipse;
use super::point::Point2;
use crate::linalg::Mat3;

/// `conditionerFromEllipse`. Upstream declares `meanAB` as `static const`, so
/// it is frozen to the *first* ellipse seen by the process; we compute it per
/// call (`frozen_mean_ab` lets callers reproduce the upstream behaviour by
/// passing the value to freeze).
pub fn conditioner_from_ellipse(e: &Ellipse, frozen_mean_ab: Option<f32>) -> Mat3 {
    let sqrt2 = 2.0f32.sqrt();
    let mean_ab = frozen_mean_ab.unwrap_or((e.a + e.b) / 2.0);
    Mat3([
        [sqrt2 / mean_ab, 0.0, -sqrt2 * e.center.x / mean_ab],
        [0.0, sqrt2 / mean_ab, -sqrt2 * e.center.y / mean_ab],
        [0.0, 0.0, 1.0],
    ])
}

/// `condition(point, mT)`: apply the homography and normalise.
#[inline]
pub fn condition(p: Point2, t: &Mat3) -> Point2 {
    let c = t.mul_vec(&p.hom());
    Point2::new(c[0] / c[2], c[1] / c[2])
}
