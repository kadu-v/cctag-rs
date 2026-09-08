mod support;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_e2e(c: &mut Criterion) {
    let threads = support::threads();
    let mut group = c.benchmark_group("e2e");
    group.sample_size(10);
    #[cfg(feature = "png")]
    {
        use cctag::{Detector, ExecMode};
        for name in ["01", "02"] {
            let path = format!("{}/CCTag/sample/{name}.png", env!("CARGO_MANIFEST_DIR"));
            let gray = cctag::image::gray::load_gray(&path).expect("required benchmark image");
            for (mode, label) in [(ExecMode::Fast, "fast"), (ExecMode::Parity, "parity")] {
                let mut det = Detector::with_crowns(3, mode).unwrap();
                group.bench_function(
                    format!("{name}/{label}/{}x{}/threads={threads}", gray.w, gray.h),
                    |b| b.iter(|| det.detect(&gray)),
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_e2e);
criterion_main!(benches);
