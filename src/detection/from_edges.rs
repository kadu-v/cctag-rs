//! `cctagDetectionFromEdges` (Detection.cpp:567-692) and its three loops.

use super::ExecMode;
use super::candidate::Candidate;
use crate::edge::{EdgeIdx, EdgePointCollection};
use crate::geometry::circle::circle_at;
use crate::geometry::distance::distance_point_ellipse;
use crate::geometry::hull::{compute_hull, is_in_ellipse, is_in_hull};
use crate::geometry::raster::rasterize_ellipse_perimeter;
use crate::geometry::{DirectedPoint, Ellipse, GeomError, Point2};
use crate::growing::{
    GrowScratch, add_candidate_flow_to_cctag, ellipse_growing_init, ellipse_growing2,
};
use crate::marker::Marker;
use crate::params::{K_WEIGHT, Params};
use crate::robust::Pcg32;
use crate::robust::median::median_ref;
use crate::robust::outlier::{
    AnotherSegment, OtherCandidate, OutlierScratch, is_another_segment, outlier_removal,
};
use crate::util::EpochSet;
use crate::vote::linking::{VisitedMap, children_of, edge_linking};

/// Per-level scratch buffers.
#[derive(Default)]
pub struct FromEdgesScratch {
    pub visited: VisitedMap,
    pub outlier: OutlierScratch,
    pub grow: GrowScratch,
    pub aux: EpochSet,
}

/// Loop one: `constructFlowComponentFromSeed` for the first
/// `min(seeds, max(rows/2, maximumNbSeeds))` seeds, sequential in seed order.
/// Returns candidates sorted by descending `averageReceivedVote` (ties keep seed
/// order, matching `lower_bound` insertion in the serialised upstream build).
pub fn loop_one(
    coll: &mut EdgePointCollection,
    seeds: &[EdgeIdx],
    rows: usize,
    params: &Params,
    scratch: &mut FromEdgesScratch,
) -> Vec<Candidate> {
    let n_max = (rows / 2).max(params.maximum_nb_seeds);
    let n = seeds.len().min(n_max);
    let mut out: Vec<Candidate> = Vec::new();
    for &seed in &seeds[..n] {
        if coll.test_processed_in(seed) {
            continue;
        }
        let link = edge_linking(
            coll,
            seed,
            params.window_size_on_inner_elliptic_segment,
            params.average_vote_min,
            &mut scratch.visited,
        );
        for &m in &link.marks {
            coll.set_processed_in(m, true);
        }
        let mut n_received: i32 = 0;
        let mut n_voted: i32 = 0;
        for &p in &link.segment {
            let vs = coll.voters_size(p) as i32;
            n_received += vs;
            if vs > 0 {
                n_voted += 1;
            }
        }
        let avg = (n_received * n_received) as f32 / n_voted as f32;
        let cand = Candidate {
            seed,
            convex_edge_segment: link.segment,
            average_received_vote: avg,
            ..Default::default()
        };
        // lower_bound with comparator (c1.avg > c2.avg): first position where !(existing.avg > new.avg)
        let pos = out.partition_point(|c| c.average_received_vote > avg);
        out.insert(pos, cand);
    }
    out
}

/// Result of `completeFlowComponent` for one candidate.
pub enum Complete {
    Accepted(Candidate),
    Rejected,
}

/// Loop two, part A: `childrenOf` + `outlierRemoval` (consumes RNG).
pub fn loop_two_a(
    coll: &EdgePointCollection,
    cand: &mut Candidate,
    params: &Params,
    rng: &mut Pcg32,
    scratch: &mut OutlierScratch,
) -> bool {
    let children = children_of(coll, &cand.convex_edge_segment);
    if children.len() < params.min_points_segment_candidate {
        return false;
    }
    cand.score = children.len() as i32;
    let (filtered, _sm) = outlier_removal(
        coll,
        &children,
        params.thresh_robust_estimation_of_outer_ellipse,
        K_WEIGHT,
        60,
        rng,
        scratch,
    );
    cand.filtered_children = filtered;
    cand.filtered_children.len() >= 5
}

/// Loop two, part B: label assignment (`Detection.cpp:156-188`), sequential.
pub fn loop_two_b_label(
    coll: &mut EdgePointCollection,
    cand: &mut Candidate,
    n_segment_out: &mut usize,
) {
    let mut common: i64 = -1;
    for &p in &cand.filtered_children {
        if coll.n_segment_out[p as usize] != -1 {
            common = coll.n_segment_out[p as usize] as i64;
            break;
        }
    }
    let label = if common == -1 {
        let l = *n_segment_out;
        *n_segment_out += 1;
        l
    } else {
        common as usize
    };
    for &p in &cand.filtered_children {
        coll.n_segment_out[p as usize] = label as i32;
    }
    cand.n_label = label;
}

/// Loop two, part C: ellipse growing and the acceptance tests.
pub fn loop_two_c(
    coll: &EdgePointCollection,
    processed: &mut [u64],
    cand: &mut Candidate,
    run_id: usize,
    params: &Params,
    grow: &mut GrowScratch,
) -> Result<bool, GeomError> {
    let (mut ellipse, good_init) =
        ellipse_growing_init(coll, &cand.filtered_children, &mut grow.fit)?;
    let pts = ellipse_growing2(
        coll,
        processed,
        &cand.filtered_children,
        &mut ellipse,
        params.ellipse_growing_elliptic_hull_width,
        run_id,
        good_init,
        grow,
    )?;
    cand.outer_ellipse = ellipse;
    cand.outer_ellipse_points = pts;

    let mut v_dist: Vec<f32> = cand
        .outer_ellipse_points
        .iter()
        .map(|&p| {
            let pp = coll.point(p);
            distance_point_ellipse(pp.xf(), pp.yf(), &cand.outer_ellipse)
        })
        .collect();
    let sm_final = median_ref(&mut v_dist);
    if sm_final > params.thr_median_distance_ellipse {
        return Ok(false);
    }
    let quality = cand.outer_ellipse_points.len() as f32
        / rasterize_ellipse_perimeter(&cand.outer_ellipse) as f32;
    if (quality as f64) > 1.1 {
        return Ok(false);
    }
    let ratio = cand.outer_ellipse.a / cand.outer_ellipse.b;
    if (ratio as f64) < 0.05 || (ratio as f64) > 20.0 {
        return Ok(false);
    }
    Ok(true)
}

/// Loop three: `cctagDetectionFromEdgesLoopTwoIteration` (Detection.cpp:378-565).
#[allow(clippy::too_many_arguments)]
pub fn loop_three(
    coll: &EdgePointCollection,
    cands: &[Candidate],
    i_cand: usize,
    pyramid_level: usize,
    scale: f32,
    params: &Params,
    rng: &mut Pcg32,
    scratch: &mut FromEdgesScratch,
) -> Option<Marker> {
    // catch(...) upstream: a failing fit drops the candidate
    loop_three_inner(
        coll,
        cands,
        i_cand,
        pyramid_level,
        scale,
        params,
        rng,
        scratch,
    )
    .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn loop_three_inner(
    coll: &EdgePointCollection,
    cands: &[Candidate],
    i_cand: usize,
    pyramid_level: usize,
    scale: f32,
    params: &Params,
    rng: &mut Pcg32,
    scratch: &mut FromEdgesScratch,
) -> Result<Option<Marker>, GeomError> {
    let cand = &cands[i_cand];
    let mut outer_points: Vec<EdgeIdx> = cand.outer_ellipse_points.clone();
    let mut outer_ellipse = cand.outer_ellipse;
    let mut cctag_points: Vec<Vec<DirectedPoint>> = Vec::new();
    let num_circles = params.n_crowns * 2;

    let mut quality =
        outer_points.len() as f32 / rasterize_ellipse_perimeter(&outer_ellipse) as f32;
    if params.search_for_another_segment && (quality as f64) > 0.25 && (quality as f64) < 0.7 {
        // flowComponentAssembling
        let seed = cand.seed;
        let seed_pt = coll.point(seed);
        let seed_flow = coll.flow_length[seed as usize];
        let area = circle_at(Point2::new(seed_pt.xf(), seed_pt.yf()), seed_flow * 2.5)?;
        let mut score: i32 = -1;
        let mut i_max = 0usize;
        for (i, other) in cands.iter().enumerate() {
            if i == i_cand {
                continue;
            }
            if cand.n_label != other.n_label {
                let of = coll.flow_length[other.seed as usize];
                let r = of / seed_flow;
                if (r as f64) > 0.666 && (r as f64) < 1.5 {
                    let osp = coll.point(other.seed);
                    if is_in_ellipse(&area, Point2::new(osp.xf(), osp.yf())) && other.score > score
                    {
                        score = other.score;
                        i_max = i;
                    }
                }
            }
        }
        if score > 0 {
            let sel = &cands[i_max];
            let other = OtherCandidate {
                outer_ellipse_points: &sel.outer_ellipse_points,
                filtered_children: &sel.filtered_children,
            };
            match is_another_segment(
                coll,
                &outer_ellipse,
                &outer_points,
                &other,
                &mut cctag_points,
                num_circles,
                params.thr_median_distance_ellipse,
                rng,
                &mut scratch.aux,
                &mut scratch.outlier,
            )? {
                AnotherSegment::Merged {
                    outer_ellipse: e,
                    outer_points: p,
                } => {
                    outer_ellipse = e;
                    outer_points = p;
                    quality = outer_points.len() as f32
                        / rasterize_ellipse_perimeter(&outer_ellipse) as f32;
                }
                AnotherSegment::NotMerged => {}
            }
        }
    }

    if !add_candidate_flow_to_cctag(
        coll,
        &cand.filtered_children,
        &cand.outer_ellipse_points,
        &outer_ellipse,
        &mut cctag_points,
        num_circles,
        &mut scratch.aux,
    ) {
        return Ok(None);
    }

    let rescale = Ellipse::from_params(
        outer_ellipse.center,
        outer_ellipse.a * scale,
        outer_ellipse.b * scale,
        outer_ellipse.angle,
    )?;
    let real_perimeter = rasterize_ellipse_perimeter(&rescale) as i32;
    let real_size = quality * real_perimeter as f32;
    let q = quality as f64;
    let rs = real_size as f64;
    if (q <= 0.35 && rs >= 300.0)
        || (quality <= 0.45 && rs >= 200.0 && rs < 300.0)
        || (quality <= 0.5 && rs >= 100.0 && rs < 200.0)
        || (quality <= 0.5 && rs >= 70.0 && rs < 100.0)
        || (quality <= 0.96 && rs >= 50.0 && rs < 70.0)
        || rs < 50.0
    {
        return Ok(None);
    }

    let ratio = outer_ellipse.a / outer_ellipse.b;
    if (ratio as f64) > 8.0 || (ratio as f64) < 0.125 {
        return Ok(None);
    }

    let (q_in, q_out) = compute_hull(&outer_ellipse, 3.6)?;
    for &p in &outer_points {
        let pp = coll.point(p);
        if !is_in_hull(&q_in, &q_out, pp.xf(), pp.yf()) {
            return Ok(None);
        }
    }

    let mut quality2 = 0.0f32;
    for &p in &outer_points {
        quality2 += coll.point(p).norm_grad();
    }
    quality2 *= scale;

    let marker = Marker::new(
        outer_ellipse.center,
        cctag_points,
        outer_ellipse,
        pyramid_level,
        scale,
        quality2,
    )?;
    Ok(Some(marker))
}

/// Full `cctagDetectionFromEdges` for one level (sequential semantics).
/// In `Fast` mode the RNG-consuming parts still run in candidate order; the
/// parallel split is applied at the multires level (see `multires.rs`).
pub fn detect_from_edges(
    coll: &mut EdgePointCollection,
    seeds: &[EdgeIdx],
    rows: usize,
    pyramid_level: usize,
    scale: f32,
    params: &Params,
    _mode: ExecMode,
    rng: &mut Pcg32,
    scratch: &mut FromEdgesScratch,
) -> Vec<Marker> {
    if seeds.is_empty() {
        return Vec::new();
    }
    let mut cands = loop_one(coll, seeds, rows, params, scratch);
    let n2 = cands.len().min(params.maximum_nb_candidates_loop_two);
    cands.truncate(n2);

    // Loop two
    let mut n_segment_out = 0usize;
    let mut accepted: Vec<Candidate> = Vec::with_capacity(n2);
    let mut processed = std::mem::take(&mut coll.processed);
    for (run_id, cand) in cands.iter_mut().enumerate() {
        if !loop_two_a(coll, cand, params, rng, &mut scratch.outlier) {
            continue;
        }
        loop_two_b_label(coll, cand, &mut n_segment_out);
        if let Ok(true) = loop_two_c(
            coll,
            &mut processed,
            cand,
            run_id,
            params,
            &mut scratch.grow,
        ) {
            accepted.push(cand.clone())
        }
    }
    coll.processed = processed;

    // Loop three
    let mut markers = Vec::new();
    for i in 0..accepted.len() {
        if let Some(m) = loop_three(
            coll,
            &accepted,
            i,
            pyramid_level,
            scale,
            params,
            rng,
            scratch,
        ) {
            markers.push(m);
        }
    }
    markers
}
