//! `cctagMultiresDetection` (Multiresolution.cpp:195-413) and `update`.

use super::ExecMode;
use super::from_edges::{FromEdgesScratch, detect_from_edges};
use crate::edge::{EdgeIdx, EdgePointCollection, NO_EDGE, from_canny::edges_points_from_canny};
use crate::fitting::ellipse_fitting_idx;
use crate::geometry::hull::compute_hull;
use crate::geometry::raster::intersect_ellipse_with_line;
use crate::geometry::{DirectedPoint, Ellipse, Point2};
use crate::marker::Marker;
use crate::params::{Params, Weight};
use crate::pyramid::ImagePyramid;
use crate::robust::Pcg32;
use crate::robust::outlier::outlier_removal;
use crate::timing::StageTimer;
use crate::vote::vote::{sort_seeds, vote};

#[derive(Default)]
pub struct MultiresWorkspace {
    pub collections: Vec<EdgePointCollection>,
    pub scratch: Vec<FromEdgesScratch>,
}

/// `update(markers, markerToAdd)`: merge overlapping reliable markers keeping
/// the best quality.
pub fn update(markers: &mut Vec<Marker>, to_add: Marker) {
    let mut flag = false;
    for cur in markers.iter_mut() {
        if cur.status.code() > 0 && to_add.status.code() > 0 && cur.is_equal(&to_add) {
            if to_add.quality > cur.quality {
                *cur = to_add.clone();
            }
            flag = true;
        }
    }
    if !flag {
        markers.push(to_add);
    }
}

fn intersect_line_to_two_ellipses(
    y: i32,
    q_in: &Ellipse,
    q_out: &Ellipse,
    coll: &EdgePointCollection,
    out: &mut Vec<EdgeIdx>,
) -> bool {
    let (io, no) = intersect_ellipse_with_line(q_out, y as f32, true);
    let (ii, ni) = intersect_ellipse_with_line(q_in, y as f32, true);
    let w = coll.bound_w as i32;
    let mut scan = |x0: i32, x1: i32| {
        for x in x0..=x1 {
            let e = coll.at(x, y);
            if e != NO_EDGE {
                let p = coll.point(e);
                let cx = q_in.center.x - p.xf();
                let cy = q_in.center.y - p.yf();
                if p.gx() * cx + p.gy() * cy < 0.0 {
                    out.push(e);
                }
            }
        }
    };
    if no == 2 && ni == 2 {
        let b1 = 0.max(io[0] as i32);
        let e1 = (w - 1).min(ii[0] as i32);
        let b2 = 0.max(ii[1] as i32);
        let e2 = (w - 1).min(io[1] as i32);
        scan(b1, e1);
        scan(b2, e2);
    } else if no == 2 && ni <= 1 {
        let b = 0.max(io[0] as i32);
        let e = (w - 1).min(io[1] as i32);
        scan(b, e);
    } else if no == 1 && ni == 0 {
        if io[0] >= 0.0 && io[0] < coll.bound_w as f32 {
            // edgeCollection(intersectionsOut[0], y): float → int conversion
            scan(io[0] as i32, io[0] as i32);
        }
    } else {
        return false;
    }
    true
}

fn select_edge_points_in_elliptic_hull(
    coll: &EdgePointCollection,
    outer: &Ellipse,
    scale: f32,
) -> Option<Vec<EdgeIdx>> {
    let (q_in, q_out) = compute_hull(outer, scale).ok()?;
    let yc = outer.center.y;
    let max_y = (yc as i32).max(0);
    let min_y = (yc as i32).min(coll.bound_h as i32 - 1);
    let mut pts = Vec::new();
    let mut y = max_y;
    while y < coll.bound_h as i32 {
        if !intersect_line_to_two_ellipses(y, &q_in, &q_out, coll, &mut pts) {
            break;
        }
        y += 1;
    }
    let mut y = min_y;
    while y >= 0 {
        if !intersect_line_to_two_ellipses(y, &q_in, &q_out, coll, &mut pts) {
            break;
        }
        y -= 1;
    }
    Some(pts)
}

/// `cctagMultiresDetection`.
pub fn multires_detection(
    pyr: &ImagePyramid,
    params: &Params,
    mode: ExecMode,
    rng: &mut Pcg32,
    ws: &mut MultiresWorkspace,
    mut timer: Option<&mut StageTimer>,
) -> Vec<Marker> {
    let n = params.number_of_processed_multires_layers;
    let full_w = pyr.level(0).src.w;
    let full_h = pyr.level(0).src.h;
    ws.collections.resize_with(n, EdgePointCollection::default);
    ws.scratch.resize_with(n, FromEdgesScratch::default);

    let mut markers: Vec<Marker> = Vec::new();
    for i in (0..n).rev() {
        let level = pyr.level(i);
        let coll = &mut ws.collections[i];
        coll.reset(level.src.w, level.src.h, full_w, full_h);
        let t0 = std::time::Instant::now();
        edges_points_from_canny(coll, &level.edges, &level.dx, &level.dy);
        if let Some(t) = timer.as_deref_mut() {
            t.record(&format!("L{i}/edges"), t0.elapsed());
        }
        let t1 = std::time::Instant::now();
        let mut seeds = vote(coll, params);
        sort_seeds(coll, &mut seeds);
        if let Some(t) = timer.as_deref_mut() {
            t.record(&format!("L{i}/vote"), t1.elapsed());
            t.count(&format!("L{i}/edge_points"), coll.len());
            t.count(&format!("L{i}/seeds"), seeds.len());
        }
        let t2 = std::time::Instant::now();
        let scale = 2f32.powi(i as i32);
        let found = detect_from_edges(
            coll,
            &seeds,
            level.src.h,
            i,
            scale,
            params,
            mode,
            rng,
            &mut ws.scratch[i],
        );
        if let Some(t) = timer.as_deref_mut() {
            t.record(&format!("L{i}/from_edges"), t2.elapsed());
            t.count(&format!("L{i}/markers"), found.len());
        }
        markers.extend(found);
    }

    // Project markers from the top of the pyramid to the bottom.
    let t3 = std::time::Instant::now();
    let coll0 = &ws.collections[0];
    let num_circles = params.n_crowns * 2;
    let _ = num_circles;
    let mut fit_scratch = Vec::new();
    for m in markers.iter_mut() {
        if m.pyramid_level > 0 {
            let scale = m.scale;
            let rescaled = m.rescaled_outer_ellipse;
            let Some(in_hull) = select_edge_points_in_elliptic_hull(coll0, &rescaled, scale) else {
                continue;
            };
            if in_hull.len() < 5 {
                continue;
            }
            let (pts, _) = outlier_removal(
                coll0,
                &in_hull,
                20.0,
                Weight::None,
                60,
                rng,
                &mut ws.scratch[0].outlier,
            );
            if pts.len() < 5 {
                continue;
            }
            if let Ok(e) = ellipse_fitting_idx(coll0, &pts, &mut fit_scratch) {
                let dp: Vec<DirectedPoint> = pts
                    .iter()
                    .map(|&i| {
                        let p = coll0.point(i);
                        DirectedPoint::new(p.xf(), p.yf(), p.gx(), p.gy())
                    })
                    .collect();
                m.center = Point2::new(m.center.x * scale, m.center.y * scale);
                m.rescaled_outer_ellipse = e;
                m.rescaled_outer_points = dp;
            }
        } else {
            m.rescaled_outer_points = m.points.last().cloned().unwrap_or_default();
        }
    }
    if let Some(t) = timer {
        t.record("reprojection", t3.elapsed());
    }
    markers
}
