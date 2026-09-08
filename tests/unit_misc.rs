//! Unit tests for the small numerical building blocks whose exact semantics
//! matter for parity (see PORTING_NOTES.md).

use cctag::bank::Bank;
use cctag::filter::thinning::thin;
use cctag::identification::cut::{ImageCut, pixel_bilinear};
use cctag::identification::orazio::orazio_distance_robust;
use cctag::identification::select::boost_mean_variance;
use cctag::image::Plane;
use cctag::robust::median::{compute_median, median_ref};

#[test]
fn bank_matches_generator_radii() {
    // markersToPrint/generators/cctag3.txt lists radii for a radius-100 marker,
    // e.g. "90 80 70 60 50" for id 0; the bank stores 100 / r in the same order.
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/CCTag/markersToPrint/generators/cctag3.txt"
    ));
    let Ok(text) = text else { return };
    let bank = Bank::builtin(3).unwrap();
    for (id, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let radii: Vec<f32> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(radii.len(), 5);
        let row = &bank.markers()[id];
        // bank ratios are listed from the smallest radius (largest ratio) outwards
        for (k, r) in radii.iter().rev().enumerate() {
            assert!(
                (row[k] - 100.0 / r).abs() < 1e-5,
                "id {id} k {k}: {} vs {}",
                row[k],
                100.0 / r
            );
        }
    }
    let ids = Bank::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/CCTag/markersToPrint/generators/3Crowns/ids.txt"
    ))
    .unwrap();
    assert_eq!(ids.markers(), bank.markers());
}

#[test]
fn bank_parser_accepts_fractions() {
    let b = Bank::parse("2/1 5/3 1.428571\n\n10/9\n").unwrap();
    assert_eq!(b.len(), 2);
    assert_eq!(b.markers()[0], vec![2.0, 5.0 / 3.0, 1.428571]);
    assert_eq!(b.markers()[1], vec![10.0 / 9.0]);
    assert!(Bank::builtin(5).is_err());
}

#[test]
fn thinning_lut_patterns() {
    // A 3-pixel wide vertical bar is thinned to its centre column; border rows/cols untouched.
    let (w, h) = (9usize, 9usize);
    let mut img = Plane::<u8>::new(w, h);
    for y in 1..h - 1 {
        for x in 3..6 {
            img.set(x, y, 255);
        }
    }
    let mut temp = Plane::<u8>::new(0, 0);
    thin(&mut img, &mut temp);
    for y in 2..h - 2 {
        assert_eq!(img.at(3, y), 0, "left column should be removed at y={y}");
        assert_eq!(img.at(4, y), 255, "centre column kept at y={y}");
    }
    // An isolated pixel survives (no neighbours: LUT index 16 -> 255).
    let mut img = Plane::<u8>::new(5, 5);
    img.set(2, 2, 255);
    thin(&mut img, &mut temp);
    assert_eq!(img.at(2, 2), 255);
}

#[test]
fn pixel_bilinear_halves_and_truncates() {
    let img = Plane::from_vec(3, 3, vec![0u8, 100, 200, 50, 150, 250, 0, 0, 0]);
    // exactly on pixel (1,1): value 150 / 2
    assert_eq!(pixel_bilinear(&img, 1.0, 1.0), 75.0);
    // halfway between (0,0)=0 and (1,0)=100 -> 50 / 2
    assert_eq!(pixel_bilinear(&img, 0.5, 0.0), 25.0);
}

#[test]
fn medians() {
    let mut v = [3.0f32, 1.0, 2.0, 4.0];
    assert_eq!(median_ref(&mut v), 3.0); // upper median for even sizes
    let mut v = [3.0f32, 1.0, 2.0];
    assert_eq!(median_ref(&mut v), 2.0);
    let mut s = Vec::new();
    assert_eq!(compute_median(&[3.0, 1.0, 2.0, 4.0], &mut s), 2.5);
    assert_eq!(compute_median(&[3.0, 1.0, 2.0], &mut s), 2.0);
    // even case is computed in double then narrowed
    let a = 0.1f32;
    let b = 0.7f32;
    assert_eq!(
        compute_median(&[b, a], &mut s),
        ((a as f64 + b as f64) / 2.0) as f32
    );
}

#[test]
fn boost_variance_recurrence_matches_population_variance() {
    let xs: Vec<f32> = (0..100).map(|i| ((i * 7919) % 257) as f32 * 0.37).collect();
    let (mean, var) = boost_mean_variance(&xs);
    let m: f64 = xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64;
    let v: f64 = xs.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / xs.len() as f64;
    assert!(((mean as f64 - m) / m).abs() < 1e-5);
    assert!(((var as f64 - v) / v).abs() < 1e-4, "{var} vs {v}");
}

#[test]
fn orazio_identifies_ideal_signals() {
    let bank = Bank::builtin(3).unwrap();
    let n = 100;
    let begin_sig = 0.25f32;
    for (id, ratios) in bank.markers().iter().enumerate() {
        // Ideal rectified signal: black (0) / white (127.5, i.e. 255/2) rings following the template.
        let mut cuts = Vec::new();
        for _ in 0..22 {
            let mut cut = ImageCut::new(Default::default(), Default::default(), begin_sig, 1.0, n);
            let step = (1.0 - begin_sig) / (n as f32 - 1.0);
            let mut x = begin_sig;
            for s in cut.signal.iter_mut() {
                let k = ratios.iter().filter(|&&r| 1.0 / r <= x).count();
                *s = if k % 2 == 1 { 0.0 } else { 127.5 };
                x += step;
            }
            cuts.push(cut);
        }
        let v = orazio_distance_robust(bank.markers(), &cuts);
        let (best, _) = v.iter().enumerate().max_by_key(|(_, l)| l.len()).unwrap();
        assert_eq!(best, id, "ideal signal of id {id} identified as {best}");
    }
}
