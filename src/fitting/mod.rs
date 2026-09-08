//! Ellipse / circle fitting (`Fitting.cpp`): Halíř–Flusser direct least
//! squares (`fit_solver` + `to_ellipse`), algebraic circle fit, `innerProdMin`.

use crate::edge::{EdgeIdx, EdgePointCollection};
use crate::geometry::distance::distance_points_2d_i;
use crate::geometry::{Ellipse, GeomError, Point2};
use crate::linalg::{Mat3, eig2_sym, eig3_general, min_eigenvector_sym4};

const PI: f32 = std::f32::consts::PI;

/// Conic coefficients `[a, b, c, d, e, f]` (offset-centred) plus the offset.
struct Conic {
    coef: [f32; 6],
    offset: (f32, f32),
}

fn fit_solver(pts: &[(f32, f32)]) -> Result<Conic, GeomError> {
    // get_offset
    let n = pts.len() as f32;
    let mut ox = 0.0f32;
    let mut oy = 0.0f32;
    for &(x, y) in pts {
        ox += x;
        oy += y;
    }
    ox /= n;
    oy /= n;
    // scatter matrices S1 = D1'D1, S2 = D1'D2, S3 = D2'D2 with D1 = [x², xy, y²], D2 = [x, y, 1]
    let mut s1 = [[0.0f32; 3]; 3];
    let mut s2 = [[0.0f32; 3]; 3];
    let mut s3 = [[0.0f32; 3]; 3];
    for &(x, y) in pts {
        let px = x - ox;
        let py = y - oy;
        let q = [px * px, px * py, py * py];
        let l = [px, py, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] += q[i] * q[j];
                s2[i][j] += q[i] * l[j];
                s3[i][j] += l[i] * l[j];
            }
        }
    }
    let s1 = Mat3(s1);
    let s2 = Mat3(s2);
    let s3 = Mat3(s3);
    let s3_inv = s3.inverse_checked().ok_or(GeomError::NotInvertible)?;
    // T = -S3^-1 * S2^T
    let t = s3_inv.mul(&s2.transpose()).scale(-1.0);
    // M = C1^-1 * (S1 + S2*T)
    let mut s1t = s1;
    let s2t = s2.mul(&t);
    for i in 0..3 {
        for j in 0..3 {
            s1t.0[i][j] += s2t.0[i][j];
        }
    }
    let c1inv = Mat3([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]]);
    let m = c1inv.mul(&s1t);
    let ev = eig3_general(&m);
    let eps = f32::EPSILON;
    let mut min_value = f32::MAX;
    let mut imin: Option<usize> = None;
    for (i, (_, v)) in ev.iter().enumerate() {
        let cond = 4.0 * v[0] * v[2] - v[1] * v[1];
        if cond > eps && cond < min_value {
            imin = Some(i);
            min_value = cond;
        }
    }
    let imin = imin.ok_or(GeomError::Degeneracy)?;
    let a1 = ev[imin].1;
    let a2 = t.mul_vec(&a1);
    Ok(Conic {
        coef: [a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]],
        offset: (ox, oy),
    })
}

/// `to_ellipse` (Fitting.cpp:157-204).
fn to_ellipse(conic: &Conic) -> Result<Ellipse, GeomError> {
    let eps = f32::EPSILON;
    let mut coef = conic.coef;
    let mut idet = coef[0] * coef[2] - coef[1] * coef[1] / 4.0;
    idet = if idet > eps { 1.0 / idet } else { 0.0 };
    let scale = (idet / 4.0).sqrt();
    if scale < eps {
        return Err(GeomError::Singularity1);
    }
    for c in coef.iter_mut() {
        *c *= scale;
    }
    let (aa, bb, cc, dd, ee, mut ff) = (coef[0], coef[1], coef[2], coef[3], coef[4], coef[5]);
    let cx = (-dd * cc + ee * bb / 2.0) * 2.0;
    let cy = (-aa * ee + dd * bb / 2.0) * 2.0;
    ff += aa * cx * cx + bb * cx * cy + cc * cy * cy + dd * cx + ee * cy;
    if ff.abs() < eps {
        return Err(GeomError::Singularity2);
    }
    let s00 = aa / -ff;
    let s01 = (bb / 2.0) / -ff;
    let s11 = cc / -ff;
    let (vals, u) = eig2_sym(s00, s01, s11);
    let center = Point2::new(cx + conic.offset.0, cy + conic.offset.1);
    let r0 = (1.0 / vals[0]).sqrt();
    let r1 = (1.0 / vals[1]).sqrt();
    // angle = pi - atan2(U(0,1), U(1,1)); u is [col][row]
    let angle = PI - u[1][0].atan2(u[1][1]);
    if r0 <= 0.0 || r1 <= 0.0 {
        return Err(GeomError::DegenerateAxes);
    }
    Ellipse::from_params(center, r0, r1, angle)
}

/// `fitEllipse` / `ellipseFitting` on generic float points (≥5 required).
pub fn fit_ellipse_xy(pts: &[(f32, f32)]) -> Result<Ellipse, GeomError> {
    if pts.len() < 5 {
        return Err(GeomError::TooFewPoints);
    }
    let conic = fit_solver(pts)?;
    to_ellipse(&conic)
}

/// `ellipseFitting(e, std::vector<EdgePoint*>)`.
pub fn ellipse_fitting_idx(
    coll: &EdgePointCollection,
    idx: &[EdgeIdx],
    scratch: &mut Vec<(f32, f32)>,
) -> Result<Ellipse, GeomError> {
    if idx.len() < 5 {
        return Err(GeomError::TooFewPoints);
    }
    scratch.clear();
    scratch.extend(idx.iter().map(|&i| {
        let p = coll.point(i);
        (p.xf(), p.yf())
    }));
    fit_ellipse_xy(scratch)
}

/// `circleFitting` (Fitting.cpp:334-360): null vector of `[x, y, 1, x²+y²]`.
/// Upstream uses `JacobiSVD` on the n×4 matrix; we use the normal equations in
/// f64 and the smallest eigenvector, which agrees to tolerance.
pub fn circle_fitting_idx(
    coll: &EdgePointCollection,
    idx: &[EdgeIdx],
) -> Result<Ellipse, GeomError> {
    let mut s = [[0.0f64; 4]; 4];
    for &i in idx {
        let p = coll.point(i);
        let x = p.xf() as f64;
        let y = p.yf() as f64;
        let row = [x, y, 1.0, x * x + y * y];
        for a in 0..4 {
            for b in 0..4 {
                s[a][b] += row[a] * row[b];
            }
        }
    }
    let v = min_eigenvector_sym4(&s);
    let xc = (-0.5 * v[0] / v[3]) as f32;
    let yc = (-0.5 * v[1] / v[3]) as f32;
    let radius = (xc * xc + yc * yc - (v[2] / v[3]) as f32).sqrt();
    if radius <= 0.0 {
        return Err(GeomError::DegenerateCircle);
    }
    Ellipse::from_params(Point2::new(xc, yc), radius, radius, 0.0)
}

/// `innerProdMin` (Fitting.cpp:227-318). Returns the minimum inner product of
/// normalised gradients found (early exit at `thr`), plus the two farthest points
/// (unused by callers, kept for fidelity of the early-return semantics).
pub fn inner_prod_min(coll: &EdgePointCollection, filtered: &[EdgeIdx], thr: f32) -> f32 {
    if filtered.is_empty() {
        return 1.1;
    }
    let p0 = coll.point(filtered[0]);
    let ng = (p0.gx() * p0.gx() + p0.gy() * p0.gy()).sqrt();
    let gx0 = p0.gx() / ng;
    let gy0 = p0.gy() / ng;
    let mut min = 1.1f32;
    let mut dist_max = 0.0f32;
    let mut p_angle1: Option<EdgeIdx> = None;
    let mut p1 = filtered[0];
    for &ci in &filtered[1..] {
        let pc = coll.point(ci);
        let ng = (pc.gx() * pc.gx() + pc.gy() * pc.gy()).sqrt();
        let gx = pc.gx() / ng;
        let gy = pc.gy() / ng;
        let ip = gx0 * gx + gy0 * gy;
        if ip <= thr {
            return ip;
        }
        if ip < min {
            min = ip;
            p_angle1 = Some(ci);
        }
        let d = distance_points_2d_i(p0, pc);
        if d > dist_max {
            dist_max = d;
            p1 = ci;
        }
    }
    // Upstream dereferences pAngle1 unconditionally here; with ≥2 points and all
    // inner products in (thr, 1.1) it is always set.
    let pa = match p_angle1 {
        Some(i) => coll.point(i),
        None => return min,
    };
    let ng = (pa.gx() * pa.gx() + pa.gy() * pa.gy()).sqrt();
    let gxm = pa.gx() / ng;
    let gym = pa.gy() / ng;
    min = 1.0;
    dist_max = 0.0;
    let p1p = *coll.point(p1);
    for &ci in filtered {
        let pc = coll.point(ci);
        let ng = (pc.gx() * pc.gx() + pc.gy() * pc.gy()).sqrt();
        let gx = pc.gx() / ng;
        let gy = pc.gy() / ng;
        let ip = gxm * gx + gym * gy;
        if ip <= thr {
            return ip;
        }
        if ip < min {
            min = ip;
        }
        let d = distance_points_2d_i(&p1p, pc);
        if d > dist_max {
            dist_max = d;
        }
    }
    min
}
