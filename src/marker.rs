//! The detected marker (`CCTag.hpp/.cpp`) and its status codes.

use crate::geometry::ellipse::scale as scale_ellipse;
use crate::geometry::hull::is_overlapping_ellipses;
use crate::geometry::{DirectedPoint, Ellipse, GeomError, Point2};
use crate::linalg::Mat3;

/// Marker status (`cctag::status`, CCTag.hpp:416-426). `IdReliable` is the only
/// valid detection; `TooFewOuterPoints` and `NoCollectedCuts` share the code -1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Status {
    /// Not yet identified (initial `_status = 0`).
    Undetermined = 0,
    IdReliable = 1,
    /// -1: too few outer points *or* no collected cuts.
    TooFewOuterPointsOrNoCuts = -1,
    NoSelectedCuts = -2,
    OptiHasDiverged = -3,
    IdNotReliable = -4,
    Degenerate = -5,
}

impl Status {
    pub fn code(self) -> i32 {
        self as i32
    }
    pub fn from_code(c: i32) -> Status {
        match c {
            1 => Status::IdReliable,
            -1 => Status::TooFewOuterPointsOrNoCuts,
            -2 => Status::NoSelectedCuts,
            -3 => Status::OptiHasDiverged,
            -4 => Status::IdNotReliable,
            -5 => Status::Degenerate,
            _ => Status::Undetermined,
        }
    }
    pub fn is_reliable(self) -> bool {
        self == Status::IdReliable
    }
}

/// `CCTag::_radiusRatiosInit`.
pub const RADIUS_RATIOS_INIT: [f32; 5] = [
    29.0 / 9.0,
    29.0 / 13.0,
    29.0 / 17.0,
    29.0 / 21.0,
    29.0 / 25.0,
];

#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    /// 0-based index into the bank; -1 when not identified.
    pub id: i32,
    pub status: Status,
    /// Imaged centre (original image scale after reprojection / refinement).
    pub center: Point2,
    /// Detection quality (sum of gradient norms × scale), overwritten by
    /// `1 / residual` after identification.
    pub quality: f32,
    /// Homography cctag-plane → pixel plane (zero until identified).
    pub homography: Mat3,
    /// Outer ellipse in pyramid-level coordinates (centre shifted by +0.5).
    pub outer_ellipse: Ellipse,
    /// Outer ellipse in original-image coordinates.
    pub rescaled_outer_ellipse: Ellipse,
    pub rescaled_outer_points: Vec<DirectedPoint>,
    /// Reconstructed ring ellipses + outer ellipse (after identification).
    pub ellipses: Vec<Ellipse>,
    pub radius_ratios: Vec<f32>,
    pub n_circles: usize,
    pub pyramid_level: usize,
    pub scale: f32,
    /// Edge points per ring, outermost last (pyramid-level coordinates).
    pub points: Vec<Vec<DirectedPoint>>,
}

impl Marker {
    /// The `CCTag(id, centerImg, points, outerEllipse, H, level, scale, quality)` constructor.
    pub fn new(
        center: Point2,
        points: Vec<Vec<DirectedPoint>>,
        outer_ellipse: Ellipse,
        pyramid_level: usize,
        scale: f32,
        quality: f32,
    ) -> Result<Marker, GeomError> {
        let mut outer = outer_ellipse;
        outer.set_center(Point2::new(outer.center.x + 0.5, outer.center.y + 0.5))?;
        let rescaled = scale_ellipse(&outer, scale)?;
        Ok(Marker {
            id: -1,
            status: Status::Undetermined,
            center,
            quality,
            homography: Mat3::ZERO,
            outer_ellipse: outer,
            rescaled_outer_ellipse: rescaled,
            rescaled_outer_points: Vec::new(),
            ellipses: Vec::new(),
            radius_ratios: RADIUS_RATIOS_INIT.to_vec(),
            n_circles: RADIUS_RATIOS_INIT.len() + 1,
            pyramid_level,
            scale,
            points,
        })
    }

    pub fn x(&self) -> f32 {
        self.center.x
    }
    pub fn y(&self) -> f32 {
        self.center.y
    }

    /// `CCTag::isEqual`: overlap of the two centre circles of radius `b/2`.
    pub fn is_equal(&self, other: &Marker) -> bool {
        let shrink = |e: &Ellipse| -> Option<Ellipse> {
            let mut c = *e;
            let b = c.b;
            c.set_a(b * 0.5).ok()?;
            c.set_b(b * 0.5).ok()?;
            Some(c)
        };
        match (
            shrink(&self.rescaled_outer_ellipse),
            shrink(&other.rescaled_outer_ellipse),
        ) {
            (Some(a), Some(b)) => is_overlapping_ellipses(&a, &b),
            _ => false,
        }
    }
}
