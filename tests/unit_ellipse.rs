//! Port of `src/cctag/geometry/test/ellipse.cpp` (Boost.Test suite `test_ellipse`).

use cctag::geometry::{Ellipse, Point2};
use cctag::linalg::Mat3;

const PTS: [(f32, f32); 21] = [
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

fn quad(p: [f32; 3], m: &Mat3) -> f32 {
    let mp = m.mul_vec(&p);
    p[0] * mp[0] + p[1] * mp[1] + p[2] * mp[2]
}

#[test]
fn test_ellipse_representation() {
    let (a_gt, b_gt, cx, cy) = (5.0f32, 3.0f32, 16.0f32, -4.0f32);
    let pi = std::f32::consts::PI;
    let n = 10;
    for i in 0..n {
        let angle = i as f32 * pi / n as f32;
        let el = Ellipse::from_params(Point2::new(cx, cy), a_gt, b_gt, angle).unwrap();
        assert!((a_gt - el.a).abs() < 1e-5);
        assert!((b_gt - el.b).abs() < 1e-5);
        assert!((angle - el.angle).abs() < 1e-5);
        assert!((cx - el.center.x).abs() < 1e-5);
        assert!((cy - el.center.y).abs() < 1e-5);

        let (c, s) = (angle.cos(), angle.sin());
        let t = Mat3([[c, -s, cx], [s, c, cy], [0.0, 0.0, 1.0]]);
        // p' T' C T p == 0
        let tct = t.transpose().mul(&el.matrix).mul(&t);
        for &(x, y) in &PTS {
            let r = quad([x, y, 1.0], &tct);
            assert!(r.abs() < 1e-3, "angle {angle}: residual {r}");
        }
        // from matrix: axes swapped, angle 90 degrees away
        let fm = Ellipse::from_matrix(el.matrix).unwrap();
        assert!((fm.b - a_gt).abs() < 1e-3, "fm.b {}", fm.b);
        assert!((fm.a - b_gt).abs() < 1e-3, "fm.a {}", fm.a);
        assert!((fm.angle - angle).abs() - pi / 2.0 < 1e-3);

        let (canonic, primal, dual) = el.canonic_form();
        let id = primal.mul(&dual);
        for r in 0..3 {
            for c2 in 0..3 {
                let v = id.at(r, c2);
                if r == c2 {
                    assert!((v - 1.0).abs() < 1e-3)
                } else {
                    assert!(v.abs() < 1e-3)
                }
            }
        }
        for &(x, y) in &PTS {
            let r = quad([y, x, 1.0], &canonic);
            assert!(r.abs() < 1e-3, "canonic residual {r}");
        }
        let fc = Ellipse::from_matrix(canonic).unwrap();
        assert!((fc.b - a_gt).abs() < 1e-3);
        assert!((fc.a - b_gt).abs() < 1e-3);
        assert!(fc.angle.abs() < 1e-4);
        assert!(fc.center.x.abs() < 1e-4 && fc.center.y.abs() < 1e-4);
    }
}
