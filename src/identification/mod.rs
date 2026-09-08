//! Marker identification (`Identification.cpp`): cut collection/selection,
//! imaged-centre refinement by grid search, and ID reading.

pub mod center;
pub mod cut;
pub mod orazio;
pub mod select;

use crate::bank::Bank;
use crate::detection::ExecMode;
use crate::geometry::Ellipse;
use crate::geometry::circle::circle;
use crate::geometry::ellipse::sorted_outer_points;
use crate::image::GrayImage;
use crate::marker::{Marker, Status};
use crate::params::Params;
use cut::{ImageCut, collect_cuts};

/// `identify_step_1`: returns the selected cuts, or the failure status.
pub fn identify_step_1(
    m: &Marker,
    src: &GrayImage,
    params: &Params,
) -> Result<Vec<ImageCut>, Status> {
    let ellipse = &m.rescaled_outer_ellipse;
    let outer_points = sorted_outer_points(
        ellipse,
        &m.rescaled_outer_points,
        params.n_samples_outer_ellipse,
    );
    if outer_points.len() < 5 {
        return Err(Status::TooFewOuterPointsOrNoCuts);
    }
    let start_sig = match params.n_crowns {
        3 => 1.0 - (2 * params.n_crowns - 1) as f32 * 0.15,
        4 => 0.26,
        _ => 0.0,
    };
    let cuts = collect_cuts(
        src,
        ellipse.center,
        &outer_points,
        params.sample_cut_length,
        start_sig,
    );
    if cuts.is_empty() {
        return Err(Status::TooFewOuterPointsOrNoCuts);
    }
    let selected = select::select_cut_cheap_uniform(
        params.num_cuts_in_ident_step,
        ellipse,
        cuts,
        src,
        m.scale,
        params.num_samples_outer_edge_points_refinement,
    );
    if selected.is_empty() {
        return Err(Status::NoSelectedCuts);
    }
    Ok(selected)
}

/// `identify_step_2`: refines centre/homography in place, sets quality, id,
/// radius ratios and ring ellipses; returns the final status.
pub fn identify_step_2(
    m: &mut Marker,
    cuts: &mut [ImageCut],
    bank: &Bank,
    src: &GrayImage,
    params: &Params,
) -> Status {
    let ellipse = m.rescaled_outer_ellipse;
    let mut residual = f32::MAX;
    let converged = center::refine_conic_family_glob(
        &mut m.homography,
        &mut m.center,
        cuts,
        src,
        &ellipse,
        params,
        &mut residual,
    );
    m.quality = 1.0 / residual;
    if !converged {
        return Status::OptiHasDiverged;
    }
    let radius_ratios = bank.markers();
    let v_score = orazio::orazio_distance_robust(radius_ratios, cuts);
    let mut max_size = 0usize;
    let mut i_max = 0usize;
    for (i, l) in v_score.iter().enumerate() {
        if l.len() > max_size {
            i_max = i;
            max_size = l.len();
        }
    }
    let mut score = 0.0f32;
    if let Some(l) = v_score.get(i_max) {
        for &p in l {
            score += p;
        }
        score /= l.len() as f32;
    } else {
        score = f32::NAN;
    }
    m.id = i_max as i32;
    if let Some(rr) = radius_ratios.get(i_max) {
        m.radius_ratios = rr.clone();
    }
    // Ring ellipses from the homography.
    let inv_h = m.homography.inverse();
    let mut ellipses = Vec::with_capacity(m.radius_ratios.len() + 1);
    for &r in &m.radius_ratios {
        let c = match circle(1.0 / r) {
            Ok(c) => c,
            Err(_) => return Status::Degenerate,
        };
        match Ellipse::from_matrix(inv_h.transpose().mul(&c.matrix).mul(&inv_h)) {
            Ok(e) => ellipses.push(e),
            Err(_) => return Status::Degenerate,
        }
    }
    ellipses.push(m.rescaled_outer_ellipse);
    m.ellipses = ellipses;
    if score > params.min_ident_proba {
        Status::IdReliable
    } else {
        Status::IdNotReliable
    }
}

/// Identify one marker end to end (steps 1 and 2), setting its status.
pub fn identify_marker(m: &mut Marker, src: &GrayImage, bank: &Bank, params: &Params) {
    match identify_step_1(m, src, params) {
        Err(s) => m.status = s,
        Ok(mut cuts) => {
            let s = identify_step_2(m, &mut cuts, bank, src, params);
            m.status = s;
        }
    }
}

/// Identification driver over all markers (`Detection.cpp:872-964`).
pub fn identify_all(
    markers: &mut [Marker],
    src: &GrayImage,
    bank: &Bank,
    params: &Params,
    mode: ExecMode,
) {
    match mode {
        ExecMode::Parity => {
            for m in markers.iter_mut() {
                identify_marker(m, src, bank, params);
            }
        }
        ExecMode::Fast => {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                markers
                    .par_iter_mut()
                    .for_each(|m| identify_marker(m, src, bank, params));
            }
            #[cfg(not(feature = "parallel"))]
            for m in markers.iter_mut() {
                identify_marker(m, src, bank, params);
            }
        }
    }
}
