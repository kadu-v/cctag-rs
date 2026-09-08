//! `outlierRemoval` (Vote.cpp:413-575) and `isAnotherSegment` (Vote.cpp:577-741).

use super::median::median_ref;
use super::rng::Pcg32;
use crate::edge::{EdgeIdx, EdgePointCollection};
use crate::fitting::{ellipse_fitting_idx, fit_ellipse_xy};
use crate::geometry::DirectedPoint;
use crate::geometry::distance::distance_point_ellipse;
use crate::geometry::raster::rasterize_ellipse_perimeter;
use crate::geometry::{Ellipse, GeomError};
use crate::growing::add_candidate_flow_to_cctag;
use crate::linalg::{Mat3, lu5_solve};
use crate::params::Weight;
use crate::util::EpochSet;

/// Reusable scratch for the RANSAC loops.
#[derive(Default)]
pub struct OutlierScratch {
    pts: Vec<(f32, f32)>,
    weights: Vec<f32>,
    dist: Vec<f32>,
    fit: Vec<(f32, f32)>,
}

/// LMedS ellipse fit + inlier selection. Returns the filtered indices (subset of
/// `children` in order) and `Some(SmFinal)` when at least one inlier was kept
/// (upstream leaves `SmFinal` untouched otherwise).
pub fn outlier_removal(
    coll: &EdgePointCollection,
    children: &[EdgeIdx],
    threshold: f32,
    weighted: Weight,
    max_size: usize,
    rng: &mut Pcg32,
    scratch: &mut OutlierScratch,
) -> (Vec<EdgeIdx>, Option<f32>) {
    let mut filtered = Vec::with_capacity(children.len());
    let n_sub = children.len().min(max_size);
    if n_sub < 5 {
        return (filtered, None);
    }
    let step = children.len() as f32 / n_sub as f32;
    let f = 1.0f32;
    let pts = &mut scratch.pts;
    let weights = &mut scratch.weights;
    pts.clear();
    weights.clear();
    let mut k = 0usize;
    for (i_edge, &ci) in children.iter().enumerate() {
        if i_edge == (k as f32 * step) as usize {
            k += 1;
            let p = coll.point(ci);
            pts.push((p.xf(), p.yf()));
            match weighted {
                Weight::InvGrad => weights.push(255.0 / p.norm_grad()),
                Weight::InvSquareGrad => weights.push(255.0 / p.norm_grad() * p.norm_grad()),
                _ => {}
            }
        }
    }

    let mut qm: Option<Ellipse> = None;
    let mut sm = 10000000.0f32;
    let mut counter = 0usize;
    let mut perm = [0i32; 5];
    let dist = &mut scratch.dist;
    while counter < 70 {
        rng.rand_5_k(&mut perm, pts.len());
        let mut a = [[0.0f32; 5]; 5];
        let mut b = [0.0f32; 5];
        for i in 0..5 {
            let (x, y) = pts[perm[i] as usize];
            a[i][0] = x * x;
            a[i][1] = 2.0 * x * y;
            a[i][2] = y * y;
            a[i][3] = 2.0 * f * x;
            a[i][4] = 2.0 * f * y;
            b[i] = -f * f;
        }
        let Some(t) = lu5_solve(&a, &b) else {
            counter += 1;
            continue;
        };
        if t[0] * t[2] - t[1] * t[1] > 0.0 {
            let q = Mat3([[t[0], t[1], t[3]], [t[1], t[2], t[4]], [t[3], t[4], 1.0]]);
            // try { Ellipse q(Q); ... } catch(...) {}  — a throw does not touch `counter`.
            let Ok(qe) = Ellipse::from_matrix(q) else {
                continue;
            };
            let ratio = qe.a / qe.b;
            if ratio < 0.04 || ratio > 25.0 {
                counter += 1;
                continue;
            }
            dist.clear();
            for &(x, y) in pts.iter() {
                dist.push(distance_point_ellipse(x, y, &qe));
            }
            if weighted != Weight::None {
                for (d, w) in dist.iter_mut().zip(weights.iter()) {
                    *d *= *w;
                }
            }
            let s = median_ref(dist);
            if s < sm {
                counter = 0;
                qm = Some(qe);
                sm = s;
            } else {
                counter += 1;
            }
        } else {
            counter += 1;
        }
    }

    let qm = qm.unwrap_or_default();
    let mut v_dist_final: Vec<f32> = Vec::with_capacity(children.len());
    for &ci in children {
        let p = coll.point(ci);
        let d = distance_point_ellipse(p.xf(), p.yf(), &qm);
        let dist_final = match weighted {
            Weight::None => d,
            Weight::InvGrad => d * 255.0 / p.norm_grad(),
            Weight::InvSquareGrad => d * 255.0 / (p.norm_grad() * p.norm_grad()),
            Weight::InvSqrtGrad => f32::MAX,
        };
        if dist_final < threshold * sm {
            filtered.push(ci);
            v_dist_final.push(dist_final);
        }
    }
    if v_dist_final.is_empty() {
        return (filtered, None);
    }
    let sm_final = median_ref(&mut v_dist_final);
    (filtered, Some(sm_final))
}

/// Inputs describing the "other" candidate for `isAnotherSegment`.
pub struct OtherCandidate<'a> {
    pub outer_ellipse_points: &'a [EdgeIdx],
    pub filtered_children: &'a [EdgeIdx],
}

/// Outcome of `isAnotherSegment`.
pub enum AnotherSegment {
    /// Merged: new outer ellipse, merged outer points; `cctag_points` were filled.
    Merged {
        outer_ellipse: Ellipse,
        outer_points: Vec<EdgeIdx>,
    },
    NotMerged,
}

/// `isAnotherSegment` (Vote.cpp:577-741). `fitEllipse` failures outside the
/// inner `try` propagate as `Err` (upstream: exception aborts the whole
/// loop-three iteration).
#[allow(clippy::too_many_arguments)]
pub fn is_another_segment(
    coll: &EdgePointCollection,
    outer_ellipse: &Ellipse,
    outer_points: &[EdgeIdx],
    other: &OtherCandidate,
    cctag_points: &mut Vec<Vec<DirectedPoint>>,
    num_circles: usize,
    thr_median_distance_ellipse: f32,
    rng: &mut Pcg32,
    aux: &mut EpochSet,
    scratch: &mut OutlierScratch,
) -> Result<AnotherSegment, GeomError> {
    let another_points = other.outer_ellipse_points;
    let mut qm: Option<Ellipse> = None;
    let mut sm = f32::MAX;

    let dist = &mut scratch.dist;
    dist.clear();
    for &i in outer_points {
        let p = coll.point(i);
        dist.push(distance_point_ellipse(p.xf(), p.yf(), outer_ellipse));
    }
    let s_ref = median_ref(dist);

    let mut cnt = 0usize;
    let mut perm = [0i32; 5];
    let mut points: Vec<(f32, f32)> = Vec::with_capacity(8);
    let mut another_dist: Vec<f32> = Vec::with_capacity(another_points.len());
    while cnt < 100 {
        points.clear();
        rng.rand_5_k(&mut perm, outer_points.len());
        for k in 0..4 {
            let p = coll.point(outer_points[perm[k] as usize]);
            points.push((p.xf(), p.yf()));
        }
        rng.rand_5_k(&mut perm, another_points.len());
        for k in 0..4 {
            let p = coll.point(another_points[perm[k] as usize]);
            points.push((p.xf(), p.yf()));
        }
        // fitEllipse outside the try: propagate errors.
        let e_toto = fit_ellipse_xy(&points)?;
        // try { Ellipse q(eToto.matrix()); ... } catch(...) { ++cnt; }
        let q = match Ellipse::from_matrix(e_toto.matrix) {
            Ok(q) => q,
            Err(_) => {
                cnt += 1;
                continue;
            }
        };
        let ratio = q.a / q.b;
        if (ratio as f64) < 0.12 || (ratio as f64) > 8.0 {
            cnt += 1;
            continue;
        }
        dist.clear();
        for &i in outer_points {
            let p = coll.point(i);
            dist.push(distance_point_ellipse(p.xf(), p.yf(), &q));
        }
        let s1 = median_ref(dist);
        another_dist.clear();
        for &i in another_points {
            let p = coll.point(i);
            another_dist.push(distance_point_ellipse(p.xf(), p.yf(), &q));
        }
        let s2 = median_ref(&mut another_dist);
        let s = s1 + s2;
        if s < sm {
            cnt = 0;
            qm = Some(q);
            sm = s;
        } else {
            cnt += 1;
        }
    }
    let _ = qm;

    let thr = 6.0f32;
    if sm < thr * s_ref {
        let mut temp_points: Vec<EdgeIdx> = outer_points.to_vec();
        temp_points.extend_from_slice(another_points);
        // ellipseFitting outside any try: propagate.
        let outer_temp = ellipse_fitting_idx(coll, &temp_points, &mut scratch.fit)?;
        let quality = temp_points.len() as f32 / rasterize_ellipse_perimeter(&outer_temp) as f32;
        if (quality as f64) < 1.1 {
            let mut v_dist_final: Vec<f32> = Vec::with_capacity(temp_points.len());
            for &i in &temp_points {
                let p = coll.point(i);
                v_dist_final.push(distance_point_ellipse(p.xf(), p.yf(), &outer_temp));
            }
            let sm_final = median_ref(&mut v_dist_final);
            if sm_final < thr_median_distance_ellipse {
                if add_candidate_flow_to_cctag(
                    coll,
                    other.filtered_children,
                    another_points,
                    &outer_temp,
                    cctag_points,
                    num_circles,
                    aux,
                ) {
                    return Ok(AnotherSegment::Merged {
                        outer_ellipse: outer_temp,
                        outer_points: temp_points,
                    });
                }
                return Ok(AnotherSegment::NotMerged);
            }
        } else {
            return Ok(AnotherSegment::NotMerged);
        }
    }
    Ok(AnotherSegment::NotMerged)
}
