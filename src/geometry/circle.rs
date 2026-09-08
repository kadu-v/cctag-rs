//! `numerical::geometry::Circle` (geometry/Circle.hpp/.cpp).

use super::ellipse::{Ellipse, GeomError};
use super::point::Point2;

/// `Circle(r)` centred at the origin.
pub fn circle(r: f32) -> Result<Ellipse, GeomError> {
    Ellipse::from_params(Point2::new(0.0, 0.0), r, r, 0.0)
}

/// `Circle(center, r)`.
pub fn circle_at(center: Point2, r: f32) -> Result<Ellipse, GeomError> {
    Ellipse::from_params(center, r, r, 0.0)
}
