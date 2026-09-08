# cctag-rs

Pure-Rust port of the CPU detection pipeline of
[CCTag](https://github.com/alicevision/CCTag) (concentric-circle fiducial
markers, Calvet et al., CVPR 2016), upstream v1.0.4. No OpenCV / Eigen / Boost /
TBB at runtime.

```rust
use cctag::{Detector, ExecMode};

let gray = cctag::image::gray::load_gray("CCTag/sample/01.png")?; // feature "png"
let mut det = Detector::with_crowns(3, ExecMode::Fast)?;
for m in det.detect(&gray).iter().filter(|m| m.status.is_reliable()) {
    println!("id {} at ({:.2}, {:.2})", m.id, m.x(), m.y());
}
```

* `ExecMode::Fast` — parallel (rayon), deterministic for any thread count.
* `ExecMode::Parity` — sequential RNG consumption identical to the upstream
  `CCTAG_SERIALIZE` build (used by the parity tests).

Features: `parallel` (default, rayon), `png` (`image` crate loader for the
example/tests), `xml`, `refdata` (reader for the C++ parity dumps), `synth`
(synthetic marker renderer used by the tests).

## Layout

| Path | Contents |
|---|---|
| `src/filter`, `src/pyramid.rs` | resize, 9×9 derivative-of-Gaussian, recoded Canny, LUT thinning |
| `src/edge`, `src/vote` | edge points, Bresenham links, voting, convex edge linking |
| `src/geometry`, `src/linalg`, `src/fitting`, `src/robust` | ellipses, Eigen-order-faithful small linear algebra, Halíř–Flusser fit, PCG32, LMedS |
| `src/growing.rs`, `src/detection` | ellipse growing, the three detection loops, multi-resolution + reprojection, de-duplication |
| `src/identification` | image cuts, sub-pixel outer edge refinement, grid-search centre/homography, ID reading |
| `examples/detect.rs` | minimal CLI (`--parity --timings --iters N --threads N --filelog out.xml`) |
| `tests/` | ported Boost suites, unit tests, Tier A/B/C parity tests, determinism, synthetic scenes |
| `benches/` | criterion end-to-end and per-stage benchmarks |
| `tools/cpp-ref/` | scripts, patches and a headless driver to build/run the C++ reference |

## Verifying against the C++ reference

```bash
tools/cpp-ref/build.sh all            # build/{upstream,perf-serial,ref-exact} (+ driver)
tools/cpp-ref/gen_refs.sh             # ref-exact stage dumps -> testdata/ref/<image>/
tools/cpp-ref/gen_upstream_oracle.sh  # unmodified upstream output -> markers_upstream.json
cargo test --features png,synth,refdata
tools/cpp-ref/regression_compare.sh   # upstream `regression --compare` on our FileLog XML
cargo bench --features png                 # explicit 1-thread baseline
CCTAG_BENCH_THREADS=16 cargo bench --features png
```

For repeatable single-thread timing, exclude warm-up iterations:

```bash
cargo run --release --features png --example detect -- \
  CCTag/sample/01.png --threads 1 --warmup 5 --iters 30 --timings
cargo test --release --all-features
cargo test --release --no-default-features --features png,synth,refdata,xml
```

The detector reuses serial gradient buffers and uses AArch64 NEON for gradient,
Canny magnitude, and 25-point identification costs. Scalar fallbacks retain
portability. Pre-optimization snapshots check intermediate results and final
marker float bits; the optimizations require no API or parameter changes.

See `PORTING_NOTES.md` (what is reproduced bit-for-bit, what is tolerance-level
and why) and `BENCHMARKS.md`.

License: MPL-2.0 (same as upstream).
