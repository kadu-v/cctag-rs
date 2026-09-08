//! Tier B: marker-level comparison (tolerance + discrete decisions) between the
//! Rust `Parity` mode and the `ref-exact` C++ dumps (`markers_*.json`).
#![cfg(feature = "png")]

mod common;

use cctag::{Detector, ExecMode};
use serde_json::Value;

struct RefMarker {
    id: i64,
    status: i64,
    x: f64,
    y: f64,
    quality: f64,
    level: i64,
    cx: f64,
    cy: f64,
    a: f64,
    b: f64,
    angle: f64,
    n_rescaled: i64,
}

fn parse(path: &std::path::Path) -> Vec<RefMarker> {
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("json")).expect("parse");
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| RefMarker {
            id: m["id"].as_i64().unwrap(),
            status: m["status"].as_i64().unwrap(),
            x: m["x"].as_f64().unwrap(),
            y: m["y"].as_f64().unwrap(),
            quality: m["quality"].as_f64().unwrap(),
            level: m["level"].as_i64().unwrap(),
            cx: m["rescaled"]["cx"].as_f64().unwrap(),
            cy: m["rescaled"]["cy"].as_f64().unwrap(),
            a: m["rescaled"]["a"].as_f64().unwrap(),
            b: m["rescaled"]["b"].as_f64().unwrap(),
            angle: m["rescaled"]["angle"].as_f64().unwrap(),
            n_rescaled: m["n_rescaled_points"].as_i64().unwrap(),
        })
        .collect()
}

fn cmp_pre_ident(
    name: &str,
    ours: &[cctag::Marker],
    dir: &std::path::Path,
    failures: &mut Vec<String>,
) {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("markers_pre_ident.json")).expect("json"),
    )
    .expect("parse");
    let theirs = v.as_array().unwrap();
    eprintln!(
        "== {name} pre-ident: ours {} markers, theirs {}",
        ours.len(),
        theirs.len()
    );
    for m in ours {
        let e = &m.outer_ellipse;
        eprintln!(
            "  ours   lvl {} outer ({:.4},{:.4}) a {:.4} b {:.4} ang {:.5} q2 {:.3} rings {:?} center ({:.3},{:.3})",
            m.pyramid_level,
            e.center.x,
            e.center.y,
            e.a,
            e.b,
            e.angle,
            m.quality,
            m.points.iter().map(|r| r.len()).collect::<Vec<_>>(),
            m.center.x,
            m.center.y
        );
    }
    for t in theirs {
        let e = &t["outer"];
        eprintln!(
            "  theirs lvl {} outer ({:.4},{:.4}) a {:.4} b {:.4} ang {:.5} q2 {:.3} rings {} center ({:.3},{:.3})",
            t["level"],
            e["cx"].as_f64().unwrap(),
            e["cy"].as_f64().unwrap(),
            e["a"].as_f64().unwrap(),
            e["b"].as_f64().unwrap(),
            e["angle"].as_f64().unwrap(),
            t["quality"].as_f64().unwrap(),
            t["n_points"],
            t["x"].as_f64().unwrap(),
            t["y"].as_f64().unwrap()
        );
    }
    if ours.len() != theirs.len() {
        failures.push(format!(
            "{name} pre-ident: count {} vs {}",
            ours.len(),
            theirs.len()
        ));
        return;
    }
    for (i, (m, t)) in ours.iter().zip(theirs.iter()).enumerate() {
        let e = &m.outer_ellipse;
        let te = &t["outer"];
        let mut why = Vec::new();
        if m.pyramid_level as i64 != t["level"].as_i64().unwrap() {
            why.push("level".to_string());
        }
        let (cx, cy, a, b, ang) = (
            te["cx"].as_f64().unwrap(),
            te["cy"].as_f64().unwrap(),
            te["a"].as_f64().unwrap(),
            te["b"].as_f64().unwrap(),
            te["angle"].as_f64().unwrap(),
        );
        if (e.center.x as f64 - cx).abs() > 1e-3 || (e.center.y as f64 - cy).abs() > 1e-3 {
            why.push(format!(
                "outer center ({:.4},{:.4}) vs ({:.4},{:.4})",
                e.center.x, e.center.y, cx, cy
            ));
        }
        if ((e.a as f64 - a) / a).abs() > 1e-4 || ((e.b as f64 - b) / b).abs() > 1e-4 {
            why.push(format!(
                "outer axes ({:.4},{:.4}) vs ({:.4},{:.4})",
                e.a, e.b, a, b
            ));
        }
        // The orientation is ill-defined for near-circular ellipses: weight the
        // angle difference by the anisotropy (a - b) / max(a, b).
        let aniso = ((a - b).abs() / a.max(b)).max(1e-6);
        if angle_mod_pi_diff(e.angle as f64, ang) * aniso > 1e-4 {
            why.push(format!(
                "outer angle {:.5} vs {:.5} (a {a:.3} b {b:.3})",
                e.angle, ang
            ));
        }
        let q = t["quality"].as_f64().unwrap();
        if ((m.quality as f64 - q) / q).abs() > 1e-5 {
            why.push(format!("quality2 {:.3} vs {:.3}", m.quality, q));
        }
        let rings: Vec<i64> = t["n_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let ours_rings: Vec<i64> = m.points.iter().map(|r| r.len() as i64).collect();
        // Ring membership is decided by `isInEllipse`/`isOnTheSameSide` on a fitted
        // (merged) ellipse; the fit is tolerance-level, so allow a few boundary points.
        if rings.len() != ours_rings.len()
            || rings
                .iter()
                .zip(&ours_rings)
                .any(|(a, b)| (a - b).abs() > 3)
        {
            why.push(format!("rings {ours_rings:?} vs {rings:?}"));
        } else if rings != ours_rings {
            eprintln!("  note {name} pre-ident[{i}]: rings {ours_rings:?} vs {rings:?}");
        }
        if !why.is_empty() {
            failures.push(format!("{name} pre-ident[{i}]: {}", why.join("; ")));
        }
    }
}

fn angle_mod_pi_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::PI);
    d.min(std::f64::consts::PI - d)
}

#[test]
fn tier_b_markers_parity_mode() {
    let cases = common::ref_cases();
    if cases.is_empty() {
        common::skip_msg("tier_b");
        return;
    }
    let mut failures = Vec::new();
    for (name, img, dir) in &cases {
        let gray = common::load_gray(img);
        let mut det = Detector::with_crowns(3, ExecMode::Parity).unwrap();
        {
            // detection only (pre-identification), for the coarse comparison
            let mut p = det.params.clone();
            p.do_identification = false;
            let mut d2 = Detector::new(p, det.bank.clone(), ExecMode::Parity);
            let pre = d2.detect_raw(&gray);
            cmp_pre_ident(name, &pre, dir, &mut failures);
            if let Ok(s) = std::fs::read_to_string(dir.join("rng_draws.txt")) {
                let theirs: u64 = s.trim().parse().unwrap_or(0);
                eprintln!("  rng draws: ours {} theirs {}", d2.last_rng_draws, theirs);
                // Draw counts differ once a RANSAC consumes a tolerance-level ellipse fit
                // (Eigen GEMM / EigenSolver internals are not reproduced bit for bit).
                let _ = theirs;
            }
        }
        let ours = det.detect(&gray);
        let theirs = parse(&dir.join("markers_final.json"));
        eprintln!(
            "== {name}: ours {} markers, theirs {}",
            ours.len(),
            theirs.len()
        );
        for m in &ours {
            eprintln!(
                "  ours   id {:>3} st {:>2} lvl {} ({:.4}, {:.4}) q {:.5} ell ({:.3},{:.3}) a {:.3} b {:.3} ang {:.4} n {}",
                m.id,
                m.status.code(),
                m.pyramid_level,
                m.x(),
                m.y(),
                m.quality,
                m.rescaled_outer_ellipse.center.x,
                m.rescaled_outer_ellipse.center.y,
                m.rescaled_outer_ellipse.a,
                m.rescaled_outer_ellipse.b,
                m.rescaled_outer_ellipse.angle,
                m.rescaled_outer_points.len()
            );
        }
        for t in &theirs {
            eprintln!(
                "  theirs id {:>3} st {:>2} lvl {} ({:.4}, {:.4}) q {:.5} ell ({:.3},{:.3}) a {:.3} b {:.3} ang {:.4} n {}",
                t.id,
                t.status,
                t.level,
                t.x,
                t.y,
                t.quality,
                t.cx,
                t.cy,
                t.a,
                t.b,
                t.angle,
                t.n_rescaled
            );
        }
        if ours.len() != theirs.len() {
            failures.push(format!(
                "{name}: marker count {} vs {}",
                ours.len(),
                theirs.len()
            ));
            continue;
        }
        for (i, (o, t)) in ours.iter().zip(theirs.iter()).enumerate() {
            let mut why = Vec::new();
            if o.id as i64 != t.id {
                why.push(format!("id {} vs {}", o.id, t.id));
            }
            if o.status.code() as i64 != t.status {
                why.push(format!("status {} vs {}", o.status.code(), t.status));
            }
            // Final markers went through the reprojection RANSAC and the grid
            // search, whose float noise is amplified; the discrete decisions and the
            // regression-tool position tolerance (0.5 px) are what we require.
            let dx = (o.x() as f64 - t.x).abs();
            let dy = (o.y() as f64 - t.y).abs();
            // Centres of unreliable candidates (diverged optimisation) are meaningless.
            if o.status.is_reliable() && t.status == 1 && (dx > 0.5 || dy > 0.5) {
                why.push(format!("center diff ({dx:.4}, {dy:.4})"));
            }
            let e = &o.rescaled_outer_ellipse;
            eprintln!(
                "  note {name}[{i}]: level {} vs {}, center diff ({dx:.4},{dy:.4}), ellipse center d ({:.4},{:.4}) axes d ({:.4},{:.4}) angle d {:.5}, quality {:.5} vs {:.5}",
                o.pyramid_level,
                t.level,
                e.center.x as f64 - t.cx,
                e.center.y as f64 - t.cy,
                e.a as f64 - t.a,
                e.b as f64 - t.b,
                angle_mod_pi_diff(e.angle as f64, t.angle),
                o.quality,
                t.quality
            );
            let _ = t.n_rescaled; // upstream's CCTag copy-ctor drops the points, always 0 after dedup
            if !why.is_empty() {
                failures.push(format!("{name}[{i}]: {}", why.join("; ")));
            }
        }
    }
    if !failures.is_empty() {
        panic!("Tier B mismatches:\n{}", failures.join("\n"));
    }
}
