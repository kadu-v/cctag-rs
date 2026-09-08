//! The batched grid evaluation must be bit-identical to the sequential reference.
#![cfg(feature = "png")]
mod common;

use cctag::identification::center::{
    CenterScratch, image_center_optimization_glob, image_center_optimization_glob_reference,
};
use cctag::identification::identify_step_1;
use cctag::{Detector, ExecMode};

#[test]
fn batched_equals_reference() {
    let img = common::root().join("CCTag/sample/01.png");
    if !img.exists() {
        return;
    }
    let gray = common::load_gray(&img);
    let mut det = Detector::with_crowns(3, ExecMode::Parity).unwrap();
    let mut p = det.params.clone();
    p.do_identification = false;
    det.params = p.clone();
    let markers = det.detect_raw(&gray);
    let params = cctag::Params::new(3);
    for (mi, m) in markers.iter().enumerate() {
        let Ok(cuts0) = identify_step_1(m, &gray, &params) else {
            continue;
        };
        let outer = m.rescaled_outer_ellipse;
        let max_semi = outer.a.max(outer.b);
        let mut cuts_a = cuts0.clone();
        let mut cuts_b = cuts0.clone();
        let (mut ha, mut hb) = (m.homography, m.homography);
        let (mut ca, mut cb) = (m.center, m.center);
        let (mut ra, mut rb) = (f32::MAX, f32::MAX);
        let mut ns = params.imaged_center_neighbour_size;
        let mut sc = CenterScratch::default();
        let mut pass = 0;
        while ((ns * max_semi) as f64) > 0.02 {
            let oka = image_center_optimization_glob_reference(
                &mut ha,
                &mut cuts_a,
                &mut ca,
                &mut ra,
                ns,
                &gray,
                &outer,
                &params,
                None,
            );
            let okb = image_center_optimization_glob(
                &mut hb,
                &mut cuts_b,
                &mut cb,
                &mut rb,
                ns,
                &gray,
                &outer,
                &params,
                None,
                &mut sc,
            );
            assert_eq!(oka, okb, "marker {mi} pass {pass}: solution flag");
            assert_eq!(
                (ca.x.to_bits(), ca.y.to_bits()),
                (cb.x.to_bits(), cb.y.to_bits()),
                "marker {mi} pass {pass}: centre {ca:?} vs {cb:?} (res {ra} vs {rb})"
            );
            assert_eq!(
                ra.to_bits(),
                rb.to_bits(),
                "marker {mi} pass {pass}: residual {ra} vs {rb}"
            );
            assert_eq!(ha, hb, "marker {mi} pass {pass}: homography");
            for (k, (x, y)) in cuts_a.iter().zip(cuts_b.iter()).enumerate() {
                assert_eq!(
                    x.out_of_bounds, y.out_of_bounds,
                    "marker {mi} pass {pass} cut {k}: oob flag"
                );
                assert_eq!(
                    x.signal, y.signal,
                    "marker {mi} pass {pass} cut {k}: signal"
                );
            }
            if !oka {
                break;
            }
            ns /= 2.0;
            pass += 1;
        }
    }
}
