use criterion::{Criterion, criterion_group, criterion_main};

fn bench_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e");
    group.sample_size(10);
    #[cfg(feature = "png")]
    {
        use cctag::{Detector, ExecMode};
        for name in ["01", "02"] {
            let path = format!("{}/CCTag/sample/{name}.png", env!("CARGO_MANIFEST_DIR"));
            let Ok(gray) = cctag::image::gray::load_gray(&path) else {
                continue;
            };
            for (mode, label) in [(ExecMode::Fast, "fast"), (ExecMode::Parity, "parity")] {
                let mut det = Detector::with_crowns(3, mode).unwrap();
                group.bench_function(format!("{name}/{label}"), |b| b.iter(|| det.detect(&gray)));
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_e2e);
criterion_main!(benches);
