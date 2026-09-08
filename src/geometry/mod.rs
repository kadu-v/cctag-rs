//! Geometry primitives (`geometry/*`, `EllipseGrowing.hpp` inline helpers,
//! `optimization/conditioner.hpp`).

pub mod circle;
pub mod conditioner;
pub mod distance;
pub mod ellipse;
pub mod hull;
pub mod point;
pub mod raster;

pub use ellipse::{Ellipse, GeomError};
pub use point::{DirectedPoint, Point2};
