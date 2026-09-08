//! Per-stage micro-benchmarks on sample/01.png (level 0 unless noted).
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("stages");
    group.sample_size(20);
    #[cfg(feature = "png")]
    {
        use cctag::edge::{EdgePointCollection, from_canny::edges_points_from_canny};
        use cctag::filter::{canny, dog, thinning};
        use cctag::image::{I16Plane, Plane, resize};
        use cctag::params::Params;
        use cctag::pyramid::ImagePyramid;
        use cctag::vote::vote::{sort_seeds, vote};
        let path = format!("{}/CCTag/sample/01.png", env!("CARGO_MANIFEST_DIR"));
        let Ok(gray) = cctag::image::gray::load_gray(&path) else {
            return;
        };
        let params = Params::new(3);
        let mut dx = I16Plane::new(0, 0);
        let mut dy = I16Plane::new(0, 0);
        group.bench_function("L0/downscale", |b| b.iter(|| resize::downscale2(&gray)));
        group.bench_function("L0/gradient_separable", |b| {
            b.iter(|| dog::derivatives(&gray, &mut dx, &mut dy))
        });
        dog::derivatives(&gray, &mut dx, &mut dy);
        let mut edges = Plane::<u8>::new(0, 0);
        let mut ws = canny::CannyWorkspace::default();
        group.bench_function("L0/canny", |b| {
            b.iter(|| canny::recoded_canny(&dx, &dy, 2, 10, &mut edges, &mut ws))
        });
        canny::recoded_canny(&dx, &dy, 2, 10, &mut edges, &mut ws);
        let mut temp = Plane::<u8>::new(0, 0);
        let thinned = {
            let mut e = edges.clone();
            thinning::thin(&mut e, &mut temp);
            e
        };
        group.bench_function("L0/thinning", |b| {
            b.iter(|| {
                let mut e = edges.clone();
                thinning::thin(&mut e, &mut temp)
            })
        });
        let mut pyr = ImagePyramid::new();
        group.bench_function("pyramid/all_levels", |b| {
            b.iter(|| pyr.build(&gray, &params))
        });
        let mut coll = EdgePointCollection::new(gray.w, gray.h, gray.w, gray.h);
        group.bench_function("L0/collect_edges", |b| {
            b.iter(|| {
                coll.reset(gray.w, gray.h, gray.w, gray.h);
                edges_points_from_canny(&mut coll, &thinned, &dx, &dy)
            })
        });
        group.bench_function("L0/vote", |b| {
            b.iter(|| {
                coll.reset(gray.w, gray.h, gray.w, gray.h);
                edges_points_from_canny(&mut coll, &thinned, &dx, &dy);
                let mut seeds = vote(&mut coll, &params);
                sort_seeds(&coll, &mut seeds);
                seeds.len()
            })
        });
        // detection without identification (multires incl. loops + reprojection)
        let mut p2 = params.clone();
        p2.do_identification = false;
        let mut det =
            cctag::Detector::new(p2, cctag::Bank::builtin(3).unwrap(), cctag::ExecMode::Fast);
        group.bench_function("multires/detect_only", |b| b.iter(|| det.detect(&gray)));
        // identification alone
        let markers = det.detect_raw(&gray);
        let bank = cctag::Bank::builtin(3).unwrap();
        group.bench_function("identification/all_markers_serial", |b| {
            b.iter(|| {
                let mut ms = markers.clone();
                cctag::identification::identify_all(
                    &mut ms,
                    &pyr.level(0).src,
                    &bank,
                    &params,
                    cctag::ExecMode::Parity,
                );
                ms
            })
        });
        group.bench_function("identification/all_markers_parallel", |b| {
            b.iter(|| {
                let mut ms = markers.clone();
                cctag::identification::identify_all(
                    &mut ms,
                    &pyr.level(0).src,
                    &bank,
                    &params,
                    cctag::ExecMode::Fast,
                );
                ms
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_stages);
criterion_main!(benches);
