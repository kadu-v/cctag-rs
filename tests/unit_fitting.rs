//! Port of `src/cctag/test/fitEllipse.cpp` (Boost.Test suite `test_ellipseFitting`).

use cctag::fitting::fit_ellipse_xy;
use cctag::geometry::{Ellipse, GeomError};

/// `check_pt_in_ellipse` as written upstream (its own polar test).
fn check_pt_in_ellipse(pt: (f32, f32), el: &Ellipse) -> bool {
    let to = (pt.0 - el.center.x, pt.1 - el.center.y);
    let pt_angle = (to.1 as f64).atan2(to.0 as f64);
    let el_angle = el.angle as f64;
    let x_dist = el.a as f64 * (pt_angle + el_angle).cos();
    let y_dist = el.b as f64 * (pt_angle + el_angle).sin();
    let el_dist = x_dist.hypot(y_dist);
    ((to.0 as f64).hypot(to.1 as f64)) < el_dist
}

fn fit_and_check(pts: &[(f32, f32)]) -> bool {
    let e = fit_ellipse_xy(pts).expect("fit");
    let n = pts.len() as f32;
    let mut c = (0.0f32, 0.0f32);
    for p in pts {
        c.0 += p.0 / n;
        c.1 += p.1 / n;
    }
    check_pt_in_ellipse(c, &e)
}

const PIXEL_INT: [(f32, f32); 12] = [
    (327., 317.),
    (328., 316.),
    (329., 315.),
    (330., 314.),
    (331., 314.),
    (332., 314.),
    (333., 315.),
    (333., 316.),
    (333., 317.),
    (333., 318.),
    (333., 319.),
    (333., 320.),
];
const FLOAT1: [(f32, f32); 10] = [
    (924.784, 764.160),
    (928.388, 615.903),
    (847.400, 888.014),
    (929.406, 741.675),
    (904.564, 825.605),
    (926.742, 760.746),
    (863.479, 873.406),
    (910.987, 808.863),
    (929.145, 744.976),
    (917.474, 791.823),
];
const GT: [(f32, f32); 21] = [
    (5.0, 0.0),
    (4.7553, 0.9271),
    (4.0451, 1.7634),
    (2.9389, 2.4271),
    (1.5451, 2.8532),
    (0.0, 3.0),
    (-1.5451, 2.8532),
    (-2.9389, 2.4271),
    (-4.0451, 1.7634),
    (-4.7553, 0.9271),
    (-5.0, 0.0),
    (-4.7553, -0.9271),
    (-4.0451, -1.7634),
    (-2.9389, -2.4271),
    (-1.5451, -2.8532),
    (-0.0, -3.0),
    (1.5451, -2.8532),
    (2.9389, -2.4271),
    (4.0451, -1.7634),
    (4.7553, -0.9271),
    (5.0, 0.0),
];

#[test]
fn test_pixel_int() {
    assert!(fit_and_check(&PIXEL_INT));
}

#[test]
fn test_pixel_int2() {
    assert!(fit_and_check(&PIXEL_INT));
}

#[test]
fn test_float1() {
    assert!(fit_and_check(&FLOAT1));
}

#[test]
fn test_float2() {
    assert!(fit_and_check(&FLOAT1));
}

#[test]
fn test_with_gt() {
    assert!(fit_and_check(&GT));
    let e = fit_ellipse_xy(&GT).unwrap();
    // Stronger than upstream (which only re-fits): the axes must be 5 and 3.
    let (lo, hi) = if e.a < e.b { (e.a, e.b) } else { (e.b, e.a) };
    assert!(
        (hi - 5.0).abs() < 1e-3 && (lo - 3.0).abs() < 1e-3,
        "axes {} {}",
        e.a,
        e.b
    );
    assert!(e.center.x.abs() < 1e-3 && e.center.y.abs() < 1e-3);
}

#[test]
fn test_throw_5points() {
    for i in 0..5 {
        let pts = vec![(5.0f32, 0.0f32); i];
        assert_eq!(fit_ellipse_xy(&pts).err(), Some(GeomError::TooFewPoints));
    }
}

#[test]
fn test_throw_degenerate_same_point() {
    let pts = vec![(5.0f32, 0.0f32); 5];
    assert!(fit_ellipse_xy(&pts).is_err());
}

#[test]
fn test_throw_degenerate_aligned_points() {
    let pts: Vec<(f32, f32)> = (1..=5).map(|i| (5.0, i as f32)).collect();
    assert!(fit_ellipse_xy(&pts).is_err());
}

// test_throw_repeated_points is disabled upstream as well: Eigen's
// computeInverseWithCheck reports the scatter matrix as invertible.

#[test]
fn test_throw_degenerate_points() {
    let pts = [
        (-0.636353, 5.27272),
        (0.363647, 4.27272),
        (0.363647, 1.27272),
        (0.363647, 2.27272),
        (0.363647, 3.27272),
        (0.363647, 0.27272),
        (0.363647, -0.72728),
        (0.363647, -1.72728),
        (-0.636353, -3.72728),
        (-0.636353, -5.72728),
        (-0.636353, -4.72728),
    ];
    assert!(fit_ellipse_xy(&pts).is_err());
}
