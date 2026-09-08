//! Writer for the upstream `regression` tool's `FileLog` XML (boost
//! `xml_oarchive` layout, `applications/regression/TestLog.h`), so that the
//! upstream `regression --compare` can be used as an independent oracle.

use crate::marker::Marker;
use crate::params::Params;
use std::fmt::Write;

fn f(v: f32) -> String {
    // boost writes floats with %.9e
    format!("{:.9e}", v as f64).replace("e", "e").to_string()
}

fn fmt_boost_float(v: f32) -> String {
    // boost: std::scientific with precision 9 -> "9.999999776e-03"
    let s = format!("{:.9e}", v as f64); // Rust: "9.999999776e-3"
    let (m, e) = s.split_once('e').unwrap();
    let (sign, digits) = if let Some(d) = e.strip_prefix('-') {
        ("-", d)
    } else {
        ("+", e)
    };
    format!("{m}e{sign}{:0>2}", digits)
}

/// One frame of a FileLog.
pub struct FrameLog<'a> {
    pub frame: usize,
    pub elapsed_seconds: f32,
    pub markers: &'a [Marker],
}

/// Serialise a `FileLog` for `filename` with the given parameters and frames.
pub fn write_filelog(filename: &str, params: &Params, frames: &[FrameLog]) -> String {
    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n<!DOCTYPE boost_serialization>\n<boost_serialization signature=\"serialization::archive\" version=\"20\">\n");
    s.push_str("<FileLog class_id=\"0\" tracking_level=\"0\" version=\"0\">\n");
    let _ = writeln!(s, "\t<filename>{filename}</filename>");
    s.push_str("\t<parameters class_id=\"1\" tracking_level=\"0\" version=\"0\">\n");
    let p = params;
    let fields: Vec<(&str, String)> = vec![
        ("_cannyThrLow", fmt_boost_float(p.canny_thr_low)),
        ("_cannyThrHigh", fmt_boost_float(p.canny_thr_high)),
        ("_distSearch", p.dist_search.to_string()),
        (
            "_thrGradientMagInVote",
            p.thr_gradient_mag_in_vote.to_string(),
        ),
        ("_angleVoting", fmt_boost_float(p.angle_voting)),
        ("_ratioVoting", fmt_boost_float(p.ratio_voting)),
        ("_averageVoteMin", fmt_boost_float(p.average_vote_min)),
        (
            "_thrMedianDistanceEllipse",
            fmt_boost_float(p.thr_median_distance_ellipse),
        ),
        ("_maximumNbSeeds", p.maximum_nb_seeds.to_string()),
        (
            "_maximumNbCandidatesLoopTwo",
            p.maximum_nb_candidates_loop_two.to_string(),
        ),
        ("_nCrowns", p.n_crowns.to_string()),
        (
            "_minPointsSegmentCandidate",
            p.min_points_segment_candidate.to_string(),
        ),
        (
            "_minVotesToSelectCandidate",
            p.min_votes_to_select_candidate.to_string(),
        ),
        (
            "_threshRobustEstimationOfOuterEllipse",
            fmt_boost_float(p.thresh_robust_estimation_of_outer_ellipse),
        ),
        (
            "_ellipseGrowingEllipticHullWidth",
            fmt_boost_float(p.ellipse_growing_elliptic_hull_width),
        ),
        (
            "_windowSizeOnInnerEllipticSegment",
            p.window_size_on_inner_elliptic_segment.to_string(),
        ),
        (
            "_numberOfMultiresLayers",
            p.number_of_multires_layers.to_string(),
        ),
        (
            "_numberOfProcessedMultiresLayers",
            p.number_of_processed_multires_layers.to_string(),
        ),
        (
            "_nSamplesOuterEllipse",
            p.n_samples_outer_ellipse.to_string(),
        ),
        ("_numCutsInIdentStep", p.num_cuts_in_ident_step.to_string()),
        (
            "_numSamplesOuterEdgePointsRefinement",
            p.num_samples_outer_edge_points_refinement.to_string(),
        ),
        ("_cutsSelectionTrials", p.cuts_selection_trials.to_string()),
        ("_sampleCutLength", p.sample_cut_length.to_string()),
        (
            "_imagedCenterNGridSample",
            p.imaged_center_n_grid_sample.to_string(),
        ),
        (
            "_imagedCenterNeighbourSize",
            fmt_boost_float(p.imaged_center_neighbour_size),
        ),
        ("_minIdentProba", fmt_boost_float(p.min_ident_proba)),
        ("_useLMDif", (p.use_lm_dif as u8).to_string()),
        (
            "_searchForAnotherSegment",
            (p.search_for_another_segment as u8).to_string(),
        ),
        ("_writeOutput", (p.write_output as u8).to_string()),
        ("_doIdentification", (p.do_identification as u8).to_string()),
        ("_maxEdges", p.max_edges.to_string()),
        ("_useCuda", "0".to_string()),
        ("_pinnedCounters", "100".to_string()),
        ("_pinnedNearbyPoints", "60".to_string()),
    ];
    for (k, v) in fields {
        let _ = writeln!(s, "\t\t<{k}>{v}</{k}>");
    }
    s.push_str("\t</parameters>\n");
    s.push_str("\t<frameLogs class_id=\"2\" tracking_level=\"0\" version=\"0\">\n");
    let _ = writeln!(s, "\t\t<count>{}</count>", frames.len());
    s.push_str("\t\t<item_version>0</item_version>\n");
    for (fi, fr) in frames.iter().enumerate() {
        if fi == 0 {
            s.push_str("\t\t<item class_id=\"3\" tracking_level=\"0\" version=\"0\">\n");
        } else {
            s.push_str("\t\t<item>\n");
        }
        let _ = writeln!(s, "\t\t\t<frame>{}</frame>", fr.frame);
        let _ = writeln!(
            s,
            "\t\t\t<elapsedTime>{}</elapsedTime>",
            fmt_boost_float(fr.elapsed_seconds)
        );
        if fi == 0 {
            s.push_str("\t\t\t<tags class_id=\"4\" tracking_level=\"0\" version=\"0\">\n");
        } else {
            s.push_str("\t\t\t<tags>\n");
        }
        let _ = writeln!(s, "\t\t\t\t<count>{}</count>", fr.markers.len());
        s.push_str("\t\t\t\t<item_version>0</item_version>\n");
        for (mi, m) in fr.markers.iter().enumerate() {
            if fi == 0 && mi == 0 {
                s.push_str("\t\t\t\t<item class_id=\"5\" tracking_level=\"0\" version=\"0\">\n");
            } else {
                s.push_str("\t\t\t\t<item>\n");
            }
            let _ = writeln!(s, "\t\t\t\t\t<id>{}</id>", m.id);
            let _ = writeln!(s, "\t\t\t\t\t<status>{}</status>", m.status.code());
            let _ = writeln!(s, "\t\t\t\t\t<x>{}</x>", fmt_boost_float(m.x()));
            let _ = writeln!(s, "\t\t\t\t\t<y>{}</y>", fmt_boost_float(m.y()));
            let _ = writeln!(
                s,
                "\t\t\t\t\t<quality>{}</quality>",
                fmt_boost_float(m.quality)
            );
            s.push_str("\t\t\t\t</item>\n");
        }
        s.push_str("\t\t\t</tags>\n\t\t</item>\n");
    }
    s.push_str("\t</frameLogs>\n</FileLog>\n</boost_serialization>\n");
    let _ = f; // keep helper for potential future use
    s
}
