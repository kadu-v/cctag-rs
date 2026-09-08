//! Exact snapshots captured from 3c582d3 before optimizing. Float values are
//! encoded as IEEE bits, including signed zero and NaNs. Never auto-update.
#![cfg(all(feature = "png", feature = "synth"))]
mod common;
use cctag::geometry::{DirectedPoint, Ellipse};
use cctag::image::GrayImage;
use cctag::{Bank, Detector, ExecMode, Marker, Params};
use serde_json::{Value, json};

fn hash(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
fn points(p: &[DirectedPoint]) -> Value {
    json!(
        p.iter()
            .map(|p| [p.x, p.y, p.dx, p.dy].map(f32::to_bits))
            .collect::<Vec<_>>()
    )
}
fn ellipse(e: &Ellipse) -> Value {
    json!([
        json!(e.matrix.0.map(|r| r.map(f32::to_bits))),
        json!([e.center.x, e.center.y, e.a, e.b, e.angle].map(f32::to_bits))
    ])
}
fn markers(ms: &[Marker]) -> Value {
    json!(ms.iter().map(|m| json!({
        "id": m.id, "status": m.status.code(), "center": [m.x().to_bits(), m.y().to_bits()],
        "quality": m.quality.to_bits(), "homography": m.homography.0.map(|r| r.map(f32::to_bits)),
        "outer": ellipse(&m.outer_ellipse), "rescaled_outer": ellipse(&m.rescaled_outer_ellipse),
        "rescaled_points": hash(points(&m.rescaled_outer_points).to_string().bytes()),
        "ellipses": m.ellipses.iter().map(ellipse).collect::<Vec<_>>(),
        "ratios": m.radius_ratios.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "circles": m.n_circles, "level": m.pyramid_level, "scale": m.scale.to_bits(),
        "points": m.points.iter().map(|p| hash(points(p).to_string().bytes())).collect::<Vec<_>>()
    })).collect::<Vec<_>>())
}
fn cases() -> Vec<(String, GrayImage, Params)> {
    let mut out = Vec::new();
    for name in ["01", "02"] {
        out.push((
            name.into(),
            common::load_gray(&common::root().join(format!("CCTag/sample/{name}.png"))),
            Params::new(3),
        ));
    }
    let mut paths: Vec<_> = std::fs::read_dir(common::root().join("testdata/synth"))
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 8, "all synthetic fixtures are required");
    for p in paths {
        out.push((
            p.file_stem().unwrap().to_str().unwrap().into(),
            common::load_gray(&p),
            Params::new(3),
        ));
    }
    for crowns in [3, 4] {
        let bank = Bank::builtin(crowns).unwrap();
        let scene = [cctag::synth::SceneMarker {
            id: 0,
            cx: 159.2,
            cy: 121.7,
            radius: 70.0,
        }];
        let img = cctag::synth::render_scene(321, 243, &bank, &scene, 0.8, 2.0, 9);
        for grid in [5, 7, 9] {
            let mut p = Params::new(crowns);
            p.imaged_center_n_grid_sample = grid;
            out.push((format!("odd_crowns{crowns}_grid{grid}"), img.clone(), p));
        }
    }
    out.push((
        "blank".into(),
        GrayImage::filled(127, 95, 255),
        Params::new(3),
    ));
    out
}
fn stages(img: &GrayImage, p: &Params) -> Value {
    use cctag::edge::{EdgePointCollection, from_canny::edges_points_from_canny};
    use cctag::filter::canny::{CannyWorkspace, recoded_canny, thresholds};
    use cctag::vote::vote::{sort_seeds, vote};
    let mut pyr = cctag::pyramid::ImagePyramid::new();
    pyr.build(img, p);
    json!(pyr.levels.iter().map(|l| {
        let mut edges = GrayImage::default();
        let (lo, hi) = thresholds(p.canny_thr_low, p.canny_thr_high);
        recoded_canny(&l.dx, &l.dy, lo, hi, &mut edges, &mut CannyWorkspace::default());
        let mut coll = EdgePointCollection::new(l.src.w, l.src.h, img.w, img.h);
        edges_points_from_canny(&mut coll, &l.edges, &l.dx, &l.dy);
        let mut seeds = vote(&mut coll, p); sort_seeds(&coll, &mut seeds);
        json!({"size": [l.src.w, l.src.h], "gray": hash(l.src.data.iter().copied()),
            "dx": hash(l.dx.data.iter().flat_map(|v| v.to_le_bytes())), "dy": hash(l.dy.data.iter().flat_map(|v| v.to_le_bytes())),
            "canny": hash(edges.data), "thinned": hash(l.edges.data.iter().copied()),
            "edges_vote": hash(format!("{coll:?}").bytes()), "seeds": seeds})
    }).collect::<Vec<_>>())
}
fn snapshot(img: &GrayImage, params: &Params, mode: ExecMode, det: &mut Detector) -> Value {
    det.params = params.clone();
    det.mode = mode;
    det.params.do_identification = false;
    let raw = det.detect_raw(img);
    let draws = det.last_rng_draws;
    det.params.do_identification = true;
    let result = det.detect(img);
    json!({"raw": markers(&raw), "final": markers(&result), "rng": draws})
}
#[test]
fn matches_pre_optimization_bits() {
    let golden: Value =
        serde_json::from_str(include_str!("../testdata/optimization_baseline.json")).unwrap();
    let cases = cases();
    for threads in [1, 4, 16] {
        let check = || {
            for mode in [ExecMode::Fast, ExecMode::Parity] {
                // Reuse detectors across image sizes, and repeat each image.
                let mut dets = [
                    Detector::with_crowns(3, mode).unwrap(),
                    Detector::with_crowns(4, mode).unwrap(),
                ];
                for (name, img, p) in &cases {
                    let crowns = if name.starts_with("odd_crowns4") {
                        4
                    } else {
                        3
                    };
                    let det = &mut dets[crowns - 3];
                    for _ in 0..2 {
                        assert_eq!(
                            snapshot(img, p, mode, det),
                            golden[name]["detection"],
                            "{name} {mode:?} {threads}T"
                        );
                    }
                    if mode == ExecMode::Fast {
                        assert_eq!(
                            stages(img, p),
                            golden[name]["stages"],
                            "{name} stages {threads}T"
                        );
                    }
                }
            }
        };
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(check);
        #[cfg(not(feature = "parallel"))]
        {
            check();
            break;
        }
    }
}
#[test]
#[ignore = "only capture on the unmodified baseline commit"]
fn record_original_baseline() {
    use std::process::Command;
    assert!(
        String::from_utf8(
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap()
                .stdout
        )
        .unwrap()
        .starts_with("3c582d3")
    );
    assert!(
        Command::new("git")
            .args(["diff", "--quiet", "HEAD", "--", "src"])
            .status()
            .unwrap()
            .success(),
        "never overwrite the oracle from optimized source"
    );
    let mut output = serde_json::Map::new();
    for (name, img, p) in cases() {
        let crowns = if name.starts_with("odd_crowns4") {
            4
        } else {
            3
        };
        let mut det = Detector::with_crowns(crowns, ExecMode::Parity).unwrap();
        output.insert(name, json!({"stages": stages(&img, &p), "detection": snapshot(&img, &p, ExecMode::Parity, &mut det)}));
    }
    std::fs::write(
        common::root().join("testdata/optimization_baseline.json"),
        serde_json::to_string_pretty(&output).unwrap() + "\n",
    )
    .unwrap();
}
