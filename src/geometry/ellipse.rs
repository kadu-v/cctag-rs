//! `numerical::geometry::Ellipse` (geometry/Ellipse.cpp).

use super::point::{DirectedPoint, Point2};
use crate::linalg::Mat3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GeomError {
    #[error("Semi axes must be real positive!")]
    NegativeAxis,
    #[error("Singular matrix!")]
    Singular,
    #[error("fit_solver: the input points appear to be linearly dependent")]
    NotInvertible,
    #[error("fit_solver: degeneracy")]
    Degeneracy,
    #[error("to_ellipse_2: singularity 1")]
    Singularity1,
    #[error("to_ellipse_2: singularity 2")]
    Singularity2,
    #[error("Degenerate ellipse after fitEllipse => line or point.")]
    DegenerateAxes,
    #[error("fitEllipse: at least 5 points are needed to estimate an ellipse")]
    TooFewPoints,
    #[error("Degenerate circle in circleFitting, radius is negative")]
    DegenerateCircle,
    #[error("Normalization of an infinite point !")]
    InfinitePoint,
}

/// Ellipse kept in both representations (conic matrix and centre/axes/angle),
/// synchronised like upstream.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Ellipse {
    pub matrix: Mat3,
    pub center: Point2,
    pub a: f32,
    pub b: f32,
    pub angle: f32,
}

#[inline(always)]
fn sign(v: f32) -> f32 {
    // boost::math::sign: -1, 0, +1
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl Ellipse {
    /// `Ellipse(const Matrix3f&)` — computes the parameters from the conic.
    pub fn from_matrix(matrix: Mat3) -> Result<Ellipse, GeomError> {
        let mut e = Ellipse {
            matrix,
            ..Default::default()
        };
        e.compute_parameters()?;
        Ok(e)
    }

    /// `Ellipse(center, a, b, angle)`.
    pub fn from_params(center: Point2, a: f32, b: f32, angle: f32) -> Result<Ellipse, GeomError> {
        if a < 0.0 || b < 0.0 {
            return Err(GeomError::NegativeAxis);
        }
        let mut e = Ellipse {
            matrix: Mat3::ZERO,
            center,
            a,
            b,
            angle,
        };
        e.compute_matrix()?;
        Ok(e)
    }

    pub fn set_matrix(&mut self, matrix: Mat3) -> Result<(), GeomError> {
        self.matrix = matrix;
        self.compute_parameters()
    }

    pub fn set_center(&mut self, c: Point2) -> Result<(), GeomError> {
        self.center = c;
        self.compute_matrix()
    }

    pub fn set_a(&mut self, a: f32) -> Result<(), GeomError> {
        if a < 0.0 {
            return Err(GeomError::NegativeAxis);
        }
        self.a = a;
        self.compute_matrix()
    }

    pub fn set_b(&mut self, b: f32) -> Result<(), GeomError> {
        if b < 0.0 {
            return Err(GeomError::NegativeAxis);
        }
        self.b = b;
        self.compute_matrix()
    }

    pub fn set_angle(&mut self, angle: f32) -> Result<(), GeomError> {
        self.angle = angle;
        self.compute_matrix()
    }

    /// `Ellipse::transform(mT)` = `Ellipse(mT^T * C * mT)`.
    pub fn transform(&self, t: &Mat3) -> Result<Ellipse, GeomError> {
        Ellipse::from_matrix(t.sandwich(&self.matrix))
    }

    /// `computeParameters()` (Ellipse.cpp:103-156).
    fn compute_parameters(&mut self) -> Result<(), GeomError> {
        let m = &self.matrix;
        let par0 = m.at(0, 0);
        let par1 = 2.0 * m.at(0, 1);
        let par2 = m.at(1, 1);
        let par3 = 2.0 * m.at(0, 2);
        let par4 = 2.0 * m.at(1, 2);
        let par5 = m.at(2, 2);

        let thetarad = 0.5 * par1.atan2(par0 - par2);
        let cost = thetarad.cos();
        let sint = thetarad.sin();
        let sin_sq = sint * sint;
        let cos_sq = cost * cost;
        let cos_sin = sint * cost;

        let ao = par5;
        let au = par3 * cost + par4 * sint;
        let av = -par3 * sint + par4 * cost;
        let auu = par0 * cos_sq + par2 * sin_sq + par1 * cos_sin;
        let avv = par0 * sin_sq + par2 * cos_sq - par1 * cos_sin;

        if auu == 0.0 || avv == 0.0 {
            self.center = Point2::new(0.0, 0.0);
            self.a = 0.0;
            self.b = 0.0;
            self.angle = 0.0;
        } else {
            let tu = -au / (2.0 * auu);
            let tv = -av / (2.0 * avv);
            let wc = ao - auu * tu * tu - avv * tv * tv;
            self.center = Point2::new(tu * cost - tv * sint, tu * sint + tv * cost);
            let ru = -wc / auu;
            let rv = -wc / avv;
            let a_aux = ru.abs().sqrt() * sign(ru);
            let b_aux = rv.abs().sqrt() * sign(rv);
            if a_aux < 0.0 || b_aux < 0.0 {
                return Err(GeomError::NegativeAxis);
            }
            self.a = a_aux;
            self.b = b_aux;
            self.angle = thetarad;
        }
        Ok(())
    }

    /// `computeMatrix()` (Ellipse.cpp:216-241).
    fn compute_matrix(&mut self) -> Result<(), GeomError> {
        let (s, c) = (self.angle.sin(), self.angle.cos());
        let tmp = Mat3([
            [c, -s, self.center.x],
            [s, c, self.center.y],
            [0.0, 0.0, 1.0],
        ]);
        let mut diag = Mat3::IDENTITY;
        diag.set(0, 0, 1.0 / (self.a * self.a));
        diag.set(1, 1, 1.0 / (self.b * self.b));
        diag.set(2, 2, -1.0);
        let inv = tmp.inverse_checked().ok_or(GeomError::Singular)?;
        let m = diag.mul(&inv);
        self.matrix = inv.transpose().mul(&m);
        Ok(())
    }

    /// `getCanonicForm(mCanonic, mTprimal, mTdual)` (Ellipse.cpp:161-214).
    pub fn canonic_form(&self) -> (Mat3, Mat3, Mat3) {
        let m = &self.matrix;
        let q1 = m.at(0, 0);
        let q2 = m.at(0, 1);
        let q3 = m.at(0, 2);
        let q4 = m.at(1, 1);
        let q5 = m.at(1, 2);
        let q6 = m.at(2, 2);
        let par1 = q1;
        let par2 = 2.0 * q2;
        let par3 = q4;
        let par4 = 2.0 * q3;
        let par5 = 2.0 * q5;
        let thetarad = 0.5 * par2.atan2(par1 - par3);
        let cost = thetarad.cos();
        let sint = thetarad.sin();
        let sin_sq = sint * sint;
        let cos_sq = cost * cost;
        let cos_sin = sint * cost;
        let au = par4 * cost + par5 * sint;
        let av = -par4 * sint + par5 * cost;
        let auu = par1 * cos_sq + par3 * sin_sq + par2 * cos_sin;
        let avv = par1 * sin_sq + par3 * cos_sq - par2 * cos_sin;
        let tu = -au / (2.0 * auu);
        let tv = -av / (2.0 * avv);
        let uc = tu * cost - tv * sint;
        let vc = tu * sint + tv * cost;

        let qt1 = cost * (cost * q1 + q2 * sint) + sint * (cost * q2 + q4 * sint);
        let qt2 = cost * (cost * q2 + q4 * sint) - sint * (cost * q1 + q2 * sint);
        let qt3 =
            cost * q3 + q5 * sint + uc * (cost * q1 + q2 * sint) + vc * (cost * q2 + q4 * sint);
        let qt4 = cost * (cost * q4 - q2 * sint) - sint * (cost * q2 - q1 * sint);
        let qt5 =
            cost * q5 - q3 * sint + uc * (cost * q2 - q1 * sint) + vc * (cost * q4 - q2 * sint);
        let qt6 =
            q6 + uc * (q3 + q1 * uc + q2 * vc) + vc * (q5 + q2 * uc + q4 * vc) + q3 * uc + q5 * vc;

        let canonic = Mat3([[qt1, qt2, qt3], [qt2, qt4, qt5], [qt3, qt5, qt6]]);
        let primal = Mat3([
            [cost, sint, -cost * uc - sint * vc],
            [-sint, cost, sint * uc - cost * vc],
            [0.0, 0.0, cost * cost + sint * sint],
        ]);
        let dual = Mat3([[cost, -sint, uc], [sint, cost, vc], [0.0, 0.0, 1.0]]);
        (canonic, primal, dual)
    }
}

/// `scale(ellipse, rescaled, s)` (Ellipse.cpp:243-249): the setters are applied
/// one after another (centre, a, b, angle), each recomputing the matrix.
pub fn scale(e: &Ellipse, s: f32) -> Result<Ellipse, GeomError> {
    let mut r = Ellipse::default();
    r.set_center(Point2::new(e.center.x * s, e.center.y * s))?;
    r.set_a(e.a * s)?;
    r.set_b(e.b * s)?;
    r.set_angle(e.angle)?;
    Ok(r)
}

/// `getSortedOuterPoints` (Ellipse.cpp:259-298): sort by polar angle around
/// the ellipse centre, then subsample to at most `requested` points.
pub fn sorted_outer_points(
    e: &Ellipse,
    points: &[DirectedPoint],
    requested: usize,
) -> Vec<DirectedPoint> {
    let mut angles: Vec<(f32, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| ((p.y - e.center.y).atan2(p.x - e.center.x), i))
        .collect();
    // std::sort on pairs: lexicographic (angle, index); total order for finite floats.
    angles.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let n_outer = requested.min(points.len());
    let step = (points.len() as f32 / (n_outer as f32 - 1.0)).max(1.0);
    let mut res = Vec::with_capacity(n_outer);
    let mut k = 0usize;
    loop {
        let i = (k as f32 * step) as usize;
        if i < angles.len() {
            res.push(points[angles[i].1]);
        } else {
            break;
        }
        k += 1;
    }
    res
}
