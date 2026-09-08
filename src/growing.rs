//! `EllipseGrowing.cpp`: `addCandidateFlowtoCCTag`, `isGoodEGPoints`,
//! `ellipseGrowingInit`, `connectedPoint`, `ellipseHull`, `ellipseGrowing2`.

use crate::edge::{EdgeIdx, EdgePointCollection, NO_EDGE};
use crate::fitting::{circle_fitting_idx, ellipse_fitting_idx, inner_prod_min};
use crate::geometry::hull::{compute_hull, is_in_ellipse, is_in_hull, is_on_the_same_side};
use crate::geometry::{DirectedPoint, Ellipse, GeomError, Point2};
use crate::util::EpochSet;

/// `addCandidateFlowtoCCTag` (EllipseGrowing.cpp:79-202). `aux` replaces the
/// shared `_processedAux` bit-set (per call; upstream clears it at the end).
pub fn add_candidate_flow_to_cctag(
    coll: &EdgePointCollection,
    filtered_children: &[EdgeIdx],
    outer_points: &[EdgeIdx],
    outer_ellipse: &Ellipse,
    cctag_points: &mut Vec<Vec<DirectedPoint>>,
    num_circles: usize,
    aux: &mut EpochSet,
) -> bool {
    // `cctagPoints.resize(numCircles)` upstream: existing rings (from a merged
    // segment) are kept and appended to; only the failure paths clear.
    cctag_points.resize(num_circles, Vec::new());
    {
        let last = cctag_points.last_mut().unwrap();
        last.reserve(outer_points.len());
        for &e in outer_points {
            let p = coll.point(e);
            last.push(DirectedPoint::new(p.xf(), p.yf(), p.gx(), p.gy()));
        }
    }
    aux.begin(coll.len());
    let cx = outer_ellipse.center.x;
    let cy = outer_ellipse.center.y;
    let mut n_gradient_out = 0usize;
    let mut n_added = 0usize;

    for &start in filtered_children {
        let mut dir = -1i32;
        let mut p = start;
        let sp = coll.point(start);
        let outer_point = Point2::new(sp.xf(), sp.yf());
        let a = outer_point.x - cx;
        let b = outer_point.y - cy;
        let line = [a, b, -a * cx - b * cy];
        for j in 1..num_circles {
            p = if dir == -1 {
                coll.before(p)
            } else {
                coll.after(p)
            };
            if p == NO_EDGE {
                // Upstream would dereference a null pointer here; the flow
                // structure guarantees links exist for voted points.
                cctag_points.clear();
                return false;
            }
            if !aux.contains(p as usize) {
                aux.insert(p as usize);
                let pp = coll.point(p);
                let norm_grad = (pp.gx() * pp.gx() + pp.gy() * pp.gy()).sqrt();
                let gex = pp.gx() / norm_grad;
                let gey = pp.gy() / norm_grad;
                let mut tx = cx - pp.xf();
                let mut ty = cy - pp.yf();
                let d = (tx * tx + ty * ty).sqrt();
                tx /= d;
                ty /= d;
                let to_add = DirectedPoint::new(pp.xf(), pp.yf(), pp.gx(), pp.gy());
                if is_in_ellipse(outer_ellipse, to_add.pos())
                    && is_on_the_same_side(outer_point, to_add.pos(), &line)
                {
                    if ((-dir) as f32 * (gex * tx + gey * ty) < -0.5) && (j >= num_circles - 2) {
                        n_gradient_out += 1;
                    }
                    cctag_points[num_circles - j - 1].push(to_add);
                    if j >= num_circles - 2 {
                        n_added += 1;
                    }
                } else {
                    cctag_points.clear();
                    return false;
                }
            }
            dir = -dir;
        }
    }
    if n_gradient_out as f32 / n_added as f32 > 0.5 {
        cctag_points.clear();
        return false;
    }
    true
}

/// `isGoodEGPoints`.
pub fn is_good_eg_points(coll: &EdgePointCollection, filtered: &[EdgeIdx]) -> bool {
    const THR_COS_DIFF_MAX: f32 = 0.25;
    inner_prod_min(coll, filtered, THR_COS_DIFF_MAX) <= THR_COS_DIFF_MAX
}

/// `ellipseGrowingInit`: returns `(ellipse, goodInit)`.
pub fn ellipse_growing_init(
    coll: &EdgePointCollection,
    filtered: &[EdgeIdx],
    fit_scratch: &mut Vec<(f32, f32)>,
) -> Result<(Ellipse, bool), GeomError> {
    if is_good_eg_points(coll, filtered) {
        Ok((ellipse_fitting_idx(coll, filtered, fit_scratch)?, true))
    } else {
        Ok((circle_fitting_idx(coll, filtered)?, false))
    }
}

const XOFF: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const YOFF: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];

/// `connectedPoint` (recursive DFS upstream) with an explicit frame stack that
/// reproduces the recursive visit order exactly.
fn connected_point(
    pts: &mut Vec<EdgeIdx>,
    mask: u64,
    coll: &EdgePointCollection,
    processed: &mut [u64],
    q_in: &Ellipse,
    q_out: &Ellipse,
    x: i32,
    y: i32,
    stack: &mut Vec<(i32, i32, u8)>,
) {
    let start = coll.at(x, y);
    debug_assert!(start != NO_EDGE);
    processed[start as usize] |= mask;
    stack.clear();
    stack.push((x, y, 0));
    let ccx = q_in.center.x;
    let ccy = q_in.center.y;
    while let Some(frame) = stack.last_mut() {
        if frame.2 >= 8 {
            stack.pop();
            continue;
        }
        let i = frame.2 as usize;
        frame.2 += 1;
        let sx = frame.0 + XOFF[i];
        let sy = frame.1 + YOFF[i];
        if !coll.in_bounds(sx, sy) {
            continue;
        }
        let e = coll.at(sx, sy);
        if e == NO_EDGE {
            continue;
        }
        let ep = coll.point(e);
        if is_in_hull(q_in, q_out, ep.xf(), ep.yf()) && (processed[e as usize] & mask) == 0 {
            let eox = ccx - ep.xf();
            let eoy = ccy - ep.yf();
            if ep.gx() * eox + ep.gy() * eoy < 0.0 {
                pts.push(e);
                processed[e as usize] |= mask;
                stack.push((sx, sy, 0));
            }
        }
    }
}

/// `ellipseHull`.
fn ellipse_hull(
    coll: &EdgePointCollection,
    processed: &mut [u64],
    pts: &mut Vec<EdgeIdx>,
    ellipse: &Ellipse,
    delta: f32,
    mask: u64,
    stack: &mut Vec<(i32, i32, u8)>,
) -> Result<(), GeomError> {
    let (q_in, q_out) = compute_hull(ellipse, delta)?;
    let init = pts.len();
    for i in 0..init {
        let p = *coll.point(pts[i]);
        connected_point(
            pts, mask, coll, processed, &q_in, &q_out, p.x as i32, p.y as i32, stack,
        );
    }
    Ok(())
}

fn count_in_hull(
    coll: &EdgePointCollection,
    pts: &[EdgeIdx],
    q_in: &Ellipse,
    q_out: &Ellipse,
) -> i32 {
    let mut n = 0;
    for &i in pts {
        let p = coll.point(i);
        if is_in_hull(q_in, q_out, p.xf(), p.yf()) {
            n += 1;
        }
    }
    n
}

/// Scratch for `ellipse_growing2`.
#[derive(Default)]
pub struct GrowScratch {
    pub stack: Vec<(i32, i32, u8)>,
    pub fit: Vec<(f32, f32)>,
}

/// `ellipseGrowing2` (EllipseGrowing.cpp:403-509). `processed` is the shared
/// per-point run mask (`EdgePoint::_processed`); `run_id` selects the bit.
pub fn ellipse_growing2(
    coll: &EdgePointCollection,
    processed: &mut [u64],
    filtered: &[EdgeIdx],
    ellipse: &mut Ellipse,
    hull_width: f32,
    run_id: usize,
    good_init: bool,
    scratch: &mut GrowScratch,
) -> Result<Vec<EdgeIdx>, GeomError> {
    let mask = 1u64 << run_id;
    let mut pts: Vec<EdgeIdx> = Vec::with_capacity(filtered.len() * 3);
    for &c in filtered {
        pts.push(c);
        processed[c as usize] |= mask;
    }
    let mut last_size: i32 = 0;
    let mut n_iter = 0usize;

    if !good_init {
        let mut new_size = pts.len() as i32;
        let mut max_nb = new_size;
        let mut n_iter_max = 0usize;
        let mut sets: Vec<Vec<EdgeIdx>> = Vec::with_capacity(6);
        let mut ells: Vec<Ellipse> = Vec::with_capacity(6);
        sets.push(pts.clone());
        ells.push(*ellipse);
        n_iter += 1;
        while new_size - last_size > 0 {
            let (q_in, q_out) = compute_hull(ellipse, hull_width)?;
            last_size = count_in_hull(coll, &pts, &q_in, &q_out);
            ellipse_hull(
                coll,
                processed,
                &mut pts,
                ellipse,
                hull_width,
                mask,
                &mut scratch.stack,
            )?;
            sets.push(pts.clone());
            ells.push(*ellipse);
            *ellipse = circle_fitting_idx(coll, &pts)?;
            let (q_in, q_out) = compute_hull(ellipse, hull_width)?;
            new_size = count_in_hull(coll, &pts, &q_in, &q_out);
            if new_size > max_nb {
                max_nb = new_size;
                n_iter_max = n_iter;
            }
            n_iter += 1;
        }
        pts = sets[n_iter_max].clone();
        *ellipse = ells[n_iter_max];
        for set in &sets {
            for &p in set {
                processed[p as usize] &= !mask;
            }
        }
        for &p in &pts {
            processed[p as usize] |= mask;
        }
    }
    let mut last_size: usize = 0;
    *ellipse = ellipse_fitting_idx(coll, &pts, &mut scratch.fit)?;
    while pts.len() - last_size > 0 {
        last_size = pts.len();
        ellipse_hull(
            coll,
            processed,
            &mut pts,
            ellipse,
            hull_width,
            mask,
            &mut scratch.stack,
        )?;
        *ellipse = ellipse_fitting_idx(coll, &pts, &mut scratch.fit)?;
    }
    Ok(pts)
}
