//! Tier C: behavioural checks — determinism (same input twice, thread count
//! independence) and agreement with the upstream C++ build's output.
#![cfg(feature = "png")]

mod common;

use cctag::{Detector, ExecMode, Marker};

fn same(a: &[Marker], b: &[Marker]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.id == y.id
                && x.status == y.status
                && x.x().to_bits() == y.x().to_bits()
                && x.y().to_bits() == y.y().to_bits()
                && x.quality.to_bits() == y.quality.to_bits()
                && x.homography == y.homography
        })
}

fn sample_images() -> Vec<std::path::PathBuf> {
    ["01", "02"]
        .iter()
        .map(|n| common::root().join(format!("CCTag/sample/{n}.png")))
        .filter(|p| p.exists())
        .collect()
}

#[test]
fn deterministic_across_runs() {
    for img in sample_images() {
        let gray = common::load_gray(&img);
        for mode in [ExecMode::Parity, ExecMode::Fast] {
            let mut det = Detector::with_crowns(3, mode).unwrap();
            let a = det.detect(&gray);
            let b = det.detect(&gray);
            let mut det2 = Detector::with_crowns(3, mode).unwrap();
            let c = det2.detect(&gray);
            assert!(
                same(&a, &b),
                "{mode:?}: two runs of the same detector differ on {}",
                img.display()
            );
            assert!(
                same(&a, &c),
                "{mode:?}: two detectors differ on {}",
                img.display()
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn fast_mode_independent_of_thread_count() {
    for img in sample_images() {
        let gray = common::load_gray(&img);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut det = Detector::with_crowns(3, ExecMode::Fast).unwrap();
                det.detect(&gray)
            })
        };
        let a = run(1);
        let b = run(16);
        assert!(
            same(&a, &b),
            "Fast mode results depend on the thread count for {}",
            img.display()
        );
    }
}

/// Agreement with the unmodified upstream library (`build/upstream/driver/cctag_ref --json`),
/// using the semantics of the upstream `regression` tool: same number of
/// candidates, same reliable ids, centres within 0.5 px. Skipped when the JSON
/// oracle files are absent (`testdata/ref/<name>/markers_upstream.json`).
#[test]
fn agrees_with_upstream_oracle() {
    let cases = common::ref_cases();
    let mut checked = 0;
    let mut failures = Vec::new();
    for (name, img, dir) in &cases {
        let path = dir.join("markers_upstream.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        checked += 1;
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let theirs: Vec<(i64, i64, f64, f64)> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m["id"].as_i64().unwrap(),
                    m["status"].as_i64().unwrap(),
                    m["x"].as_f64().unwrap(),
                    m["y"].as_f64().unwrap(),
                )
            })
            .collect();
        let gray = common::load_gray(img);
        for mode in [ExecMode::Parity, ExecMode::Fast] {
            let mut det = Detector::with_crowns(3, mode).unwrap();
            let ours = det.detect(&gray);
            if ours.len() != theirs.len() {
                failures.push(format!(
                    "{name} {mode:?}: candidate count {} vs {}",
                    ours.len(),
                    theirs.len()
                ));
                continue;
            }
            let mut o: Vec<_> = ours
                .iter()
                .filter(|m| m.status.is_reliable())
                .map(|m| (m.id as i64, m.x() as f64, m.y() as f64))
                .collect();
            let mut t: Vec<_> = theirs
                .iter()
                .filter(|m| m.1 == 1)
                .map(|m| (m.0, m.2, m.3))
                .collect();
            o.sort_by_key(|m| m.0);
            t.sort_by_key(|m| m.0);
            if o.len() != t.len() || o.iter().zip(&t).any(|(a, b)| a.0 != b.0) {
                failures.push(format!(
                    "{name} {mode:?}: reliable ids {:?} vs {:?}",
                    o.iter().map(|m| m.0).collect::<Vec<_>>(),
                    t.iter().map(|m| m.0).collect::<Vec<_>>()
                ));
                continue;
            }
            for (a, b) in o.iter().zip(&t) {
                let (dx, dy) = ((a.1 - b.1).abs(), (a.2 - b.2).abs());
                eprintln!("{name} {mode:?} id {}: d = ({dx:.3}, {dy:.3})", a.0);
                if dx > 0.5 || dy > 0.5 {
                    failures.push(format!(
                        "{name} {mode:?} id {}: centre differs by ({dx:.3}, {dy:.3}) px",
                        a.0
                    ));
                }
            }
        }
    }
    if checked == 0 {
        eprintln!(
            "SKIP agrees_with_upstream_oracle: no markers_upstream.json (run tools/cpp-ref/gen_upstream_oracle.sh)"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
