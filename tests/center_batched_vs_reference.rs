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

#[test]
fn grid_sizes_sticky_bounds_and_stale_samples_match_reference() {
    use cctag::geometry::{DirectedPoint, Ellipse, Point2};
    use cctag::identification::cut::ImageCut;
    use cctag::linalg::Mat3;
    let gray =
        cctag::image::GrayImage::from_vec(65, 49, (0..65 * 49).map(|i| (i * 37) as u8).collect());
    for grid in [3, 5, 7, 9] {
        for (cx, cy) in [(32.0, 24.0), (2.0, 2.0), (63.0, 47.0)] {
            let outer = Ellipse::from_params(Point2::new(cx, cy), 20.0, 15.0, 0.3).unwrap();
            for nc in [0, 1, 2, 5] {
                let mut a: Vec<_> = (0..nc)
                    .map(|i| {
                        let mut c = ImageCut::new(
                            Point2::new(cx, cy),
                            DirectedPoint::new(cx + 20.0 - 7.0 * i as f32, cy + 15.0, 1.0, 0.0),
                            0.1,
                            0.9,
                            17,
                        );
                        c.signal.fill(13.0 + i as f32);
                        c.out_of_bounds = i == 1;
                        c
                    })
                    .collect();
                let mut b = a.clone();
                let (mut ha, mut hb) = (Mat3::IDENTITY, Mat3::IDENTITY);
                let (mut ca, mut cb) = (outer.center, outer.center);
                let (mut ra, mut rb) = (0.0, 0.0);
                let mut p = cctag::Params::new(3);
                p.imaged_center_n_grid_sample = grid;
                let mut sc = CenterScratch::default();
                for ns in [0.2, 0.1, 0.05] {
                    let oka = image_center_optimization_glob_reference(
                        &mut ha, &mut a, &mut ca, &mut ra, ns, &gray, &outer, &p, None,
                    );
                    let okb = image_center_optimization_glob(
                        &mut hb, &mut b, &mut cb, &mut rb, ns, &gray, &outer, &p, None, &mut sc,
                    );
                    assert_eq!(oka, okb);
                    assert_eq!(
                        [ca.x.to_bits(), ca.y.to_bits(), ra.to_bits()],
                        [cb.x.to_bits(), cb.y.to_bits(), rb.to_bits()]
                    );
                    assert_eq!(
                        ha.0.map(|r| r.map(f32::to_bits)),
                        hb.0.map(|r| r.map(f32::to_bits))
                    );
                    for (a, b) in a.iter().zip(&b) {
                        assert_eq!(a.out_of_bounds, b.out_of_bounds);
                        assert_eq!(
                            a.signal.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.signal.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}
