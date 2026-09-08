# Benchmarks — cctag-rs vs upstream C++ (CPU)

For the latest Rust-to-Rust optimization measurements, see the
[2026-09-09 comparison](#2026-09-09-rust-to-rust-single-thread-optimization) below.

Machine: Apple M3 Max (16 cores: 12P + 4E), macOS, Rust 1.89 (`--release`,
`lto = "fat"`, `codegen-units = 1`), Apple clang, OpenCV 4.13 / Boost 1.90 /
Eigen 3.4.0 / oneTBB 2022 from Homebrew. Inputs: `CCTag/sample/01.png`,
`02.png` (1920×1440), 3 crowns, default parameters.

C++ builds (`tools/cpp-ref/build.sh`): `upstream` = pristine v1.0.4 with TBB,
static, `-O3`; `perf-serial` = `-DCCTAG_SERIALIZE=ON` (the only reproducible
upstream configuration). Thread count fixed with `tbb::global_control` **and**
`cv::setNumThreads` (`tools/cpp-ref/driver/cctag_ref`). Timings are in-process
medians (`std::chrono::steady_clock` / `std::time::Instant`) after warm-up;
Rust: `examples/detect.rs --iters 20`, C++: `cctag_ref --warmup 3 --iters 15`.

All measured configurations produce matching output (Tier C: same reliable ids,
centres within 0.5 px; see `PORTING_NOTES.md`).

## End to end (`cctagDetection` / `Detector::detect`), median ms

| Image | C++ upstream 1T | Rust Fast 1T | ratio | C++ upstream 16T | Rust Fast 16T | ratio | C++ perf-serial 1T | Rust Parity 1T | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 01.png | 223.3 | **108.0** | 0.48 | 397.5 | **30.2** | 0.08 | 252.6 | **107.9** | 0.43 |
| 02.png | 250.9 | **115.2** | 0.46 | 457.0 | **32.4** | 0.07 | 290.4 | **115.4** | 0.40 |

Upstream gets *slower* with 16 threads: its only parallel loops are the three
TBB loops over ≤ 40 candidates plus the per-cut `parallel_for` in
`orazioDistanceRobust`, whose scheduling overhead dominates, while OpenCV's own
threading of `filter2D`/`resize` contends with TBB.

## Per stage, 01.png, median ms

| Stage | C++ upstream 1T | Rust 1T | ratio | C++ upstream 16T | Rust 16T |
|---|---|---|---|---|---|
| pyramid (4 levels: resize, 9×9 DoG, Canny, thinning) | 67.7 | 41.0 | 0.61 | 73.4 | 10.7 |
| multi-resolution detection (edges, vote, loops 1–3, reprojection) | 21.9 | 13.7 | 0.63 | 14.7 | 11.4 |
| identification (cuts, grid search, ID reading) | 136.6 | 50.2 | 0.37 | 346.7 | 7.1 |
| L0 collect edge points | 1.50 | 1.15 | 0.77 | 1.51 | – |
| L0 vote (links + field lines + CSR) | 3.38 | 2.49 | 0.74 | 3.42 | – |
| L0 detection from edges (loops 1–3) | 8.12 | 3.81 | 0.47 | 1.80 | – |

Rust per-stage numbers come from `examples/detect.rs --timings` (single run
inside the full pipeline), C++ from `cctag_ref --mode stages`.

### Where the time goes and what was done about it

* **Identification** dominated both implementations. Upstream evaluates the
  grid-search cost as one strictly sequential chain
  `res = (float)((double)res + d*d)` over ~23 000 terms per grid point
  (latency bound, ~12 cycles/term). The port keeps the exact arithmetic but
  advances the 25 grid points' chains in lock-step (`identification/center.rs`),
  which is throughput bound: 132 ms → 50 ms at one thread, bit-identical
  results (`tests/center_batched_vs_reference.rs`). Markers are identified in
  parallel in `Fast` mode.
* **Pyramid**: the 9×9 derivative kernel is exactly rank-1, so the gradient is
  computed with four 9-tap passes in f64 (row-blocked, cache resident) instead
  of two 81-tap correlations; levels are filtered in parallel and rows of the
  gradient in parallel blocks. Canny and thinning are sequential (Tier A exact).
* **Edge collection / vote**: 8-byte edge points, SoA per-point state, CSR
  voters built with a counting sort; the link and field-line phases are pure and
  run in parallel, the order-dependent vote accumulation is sequential.
* **No per-frame page faults**: upstream allocates ~2 GB of fixed arrays per
  level (`EdgePointCollection`) and relies on lazy commit; the port sizes
  buffers from the image and reuses them.

## Acceptance (plan §6)

| Criterion | Result |
|---|---|
| Rust Fast e2e ≤ C++ upstream e2e at 1T and 16T, both images | ✅ 0.46–0.48× (1T), 0.07–0.08× (16T) |
| every stage ≤ 1.2× C++ at equal thread count | ✅ all stages < 1.0× |
| Rust Parity 1T ≤ C++ perf-serial 1T | ✅ 0.40–0.43× |
| stability p90/median ≤ 1.15 at 1T | ✅ (max/median 1.03–1.04 over 20 runs) |
| outputs agree (Tier C) for every measured configuration | ✅ |
| Rust peak RSS < 300 MB | ✅ 88 MB |

## Memory

`/usr/bin/time -l` peak RSS, `01.png`, Rust Fast 16T: **88 MB**
(the pyramid, edge maps and identification scratch for a 1920×1440 image stay
well under 100 MB; upstream's fixed `EdgePointCollection` arrays alone are
≈ 2 GB virtual per level, committed lazily).

## Criterion micro-benchmarks

`cargo bench --features png` (`benches/stages.rs`, `benches/e2e.rs`) — see the
table appended below after each run.

### `cargo bench --features png --bench stages` (01.png, Fast mode, default rayon pool = 16 threads)

| Benchmark | median |
|---|---|
| L0 downscale (1920×1440 → 960×720) | 0.46 ms |
| L0 gradient (separable f64, row-blocked, parallel) | 2.32 ms |
| L0 Canny (magnitude + NMS + hysteresis, sequential) | 4.70 ms |
| L0 thinning (2 LUT passes, sequential) | 2.17 ms |
| pyramid, all 4 levels | 10.58 ms |
| L0 collect edge points | 1.08 ms |
| L0 vote (links + field lines + CSR, links/field lines parallel) | 2.09 ms |
| multi-resolution detection without identification | 22.40 ms |
| identification, 9 markers, serial (`Parity`) | 50.81 ms |
| identification, 9 markers, parallel over markers (`Fast`) | 7.01 ms |

## Independent oracle: upstream `regression --compare` (`tools/cpp-ref/regression_compare.sh`)

`regression --gen-ref` (unmodified upstream, one process per image) vs our
FileLog XML, `--epsilon 0.5`:

| Image | upstream vs upstream | upstream vs Rust |
|---|---|---|
| 01.png | PASSED | PASSED |
| 02.png | PASSED | PASSED |
| s01_clean_r60, s02_clean_r60_b | *unreadable* | *unreadable* |
| s03_clean_r60_c, s05_small_r30, s06_noisy_r50, s07_blur_r70, s08_mixed | PASSED | PASSED |
| s04_clean_r60_d | PASSED | FAILED: different # of tags (an extra *unreliable* candidate in that upstream run) |

*unreadable*: those scenes contain a diverged (status −3) candidate whose quality
is `1/FLT_MAX` (a denormal); boost's `xml_iarchive` cannot read the file the
upstream tool itself wrote (`input stream error`). The strict candidate-count
check also trips on upstream's own run-to-run variance in the number of
unreliable candidates (observed on `02.png` as well, see `PORTING_NOTES.md`).
The reliable markers agree in every readable case.

### `cargo bench --features png --bench e2e` (default rayon pool = 16 threads)

| Benchmark | median |
|---|---|
| 01.png Fast | 33.0 ms |
| 01.png Parity (pyramid/vote parallel, RNG-consuming loops and identification sequential) | 77.7 ms |
| 02.png Fast | 32.4 ms |
| 02.png Parity | 79.8 ms |

## 2026-09-09: Rust-to-Rust single-thread optimization

Comparison with the original Rust port, commit `3c582d3`. macOS 26.5 arm64,
Rust 1.89.0, the same release profile and `png` + default `parallel` features,
1920×1440 samples, 3 crowns, unchanged default parameters. Each process reuses
one detector, warms up for 5 detections, then measures 30. Baseline and optimized
executables alternate for three sets (the second set reverses order); no other
build, test, or benchmark runs concurrently. Image loading and detector creation
are outside the measured interval. Stage timers are enabled for both versions.

Table entries are the median of the three per-process medians / p90 values,
not percentiles pooled across processes. Raw runs, stage measurements, binary
SHA-256 hashes, and peak RSS are in
[`benches/results/optimization-2026-09-09.json`](benches/results/optimization-2026-09-09.json).

| Image | Mode | Threads | Before median ms | After median ms | Reduction | Before p90 ms | After p90 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 01.png | Fast | 1 | 106.211 | 81.632 | 23.1% | 106.929 | 81.977 |
| 02.png | Fast | 1 | 113.311 | 88.748 | 21.7% | 113.778 | 89.225 |
| 01.png | Fast | 4 | 44.824 | 38.430 | 14.3% | 45.481 | 38.895 |
| 02.png | Fast | 4 | 46.669 | 40.327 | 13.6% | 47.142 | 40.981 |
| 01.png | Fast | 16 | 29.513 | 26.798 | 9.2% | 30.232 | 27.103 |
| 02.png | Fast | 16 | 31.618 | 28.632 | 9.4% | 31.956 | 29.036 |
| 01.png | Parity | 1 | 106.500 | 81.812 | 23.2% | 107.154 | 82.720 |
| 02.png | Parity | 1 | 113.388 | 88.791 | 21.7% | 113.978 | 89.281 |
| 01.png | Parity | 4 | 78.859 | 61.878 | 21.5% | 79.624 | 62.369 |
| 02.png | Parity | 4 | 81.822 | 64.797 | 20.8% | 82.662 | 65.581 |
| 01.png | Parity | 16 | 73.637 | 57.579 | 21.8% | 73.943 | 58.065 |
| 02.png | Parity | 16 | 75.940 | 60.062 | 20.9% | 76.344 | 60.562 |

The primary 1-thread Fast cases improve **23.1% and 21.7%**, exceeding the 20%
target. Both modes improve at every measured thread count; Fast 16T improves
9.2–9.4%. Worst optimized 1T p90/median across individual runs is **1.020**.

### End-to-end stage attribution, Fast 1T

| Image | Stage | Before ms | After ms |
|---|---|---:|---:|
| 01.png | pyramid | 41.204 | 32.376 |
| 01.png | multires | 13.839 | 13.872 |
| 01.png | identification | 51.116 | 35.408 |
| 02.png | pyramid | 43.825 | 34.992 |
| 02.png | multires | 17.772 | 17.828 |
| 02.png | identification | 51.672 | 35.947 |

The serial gradient retains a nine-row ring per pyramid level and avoids
recomputing interior horizontal rows at block boundaries. AArch64 NEON evaluates
adjacent pixels with the original tap order. Canny magnitude uses NEON sqrt and
ties-to-even conversion. Identification holds the 25 independent cost chains in
SIMD registers, retains each f32 rounding step, and moves validity selection out
of the sample loop. Temporary signal/count arrays are reused, and thinning only
clears its required scratch border. Other grid sizes and architectures retain
generic/scalar paths. No approximation, FMA contraction, parameter tuning, or
precision reduction is required.

### Memory and correctness

Peak RSS (`/usr/bin/time -l`, including PNG load and process overhead), maximum
of three runs, Fast 1T: 01.png **71.05 → 71.00 MiB**, 02.png **74.28 → 74.67 MiB**.
The largest optimized RSS in the entire matrix is **92.72 MiB**. The small
additional retained workspace on 02.png is included in these measurements.

All final marker output matches the baseline in all 72 benchmark processes.
Exact regression snapshots also match in both modes at 1/4/16 threads and with
`parallel` disabled: 17 cases, including 3/4 crowns and 5/7/9 grids, repeated
calls and detector reuse. Final coordinates, quality, homographies and ellipses
match at the bit level; coordinate error is **0 px**, so the allowed 0.0001 px
exception is unused. Intermediate image planes, edge/vote state and seeds match.
All ten C++ reference cases and ten upstream JSON oracles were present; the
existing Tier A/B/C tolerances were not changed.

Validation: release tests with all features (36 passed, only the explicit
baseline recorder ignored); release tests without `parallel` (35 passed);
Clippy all targets/all features with warnings denied; x86_64-unknown-linux-gnu
library cross-check for the scalar fallback. Other CPU performance is unmeasured.

### Reproduce the comparison

Build an isolated copy of the baseline with the same measurement CLI (warm-up,
p90, stage medians); its detector source remains at `3c582d3`:

```bash
mkdir -p target/baseline-src
git archive 3c582d3 | tar -x -C target/baseline-src
cp examples/detect.rs target/baseline-src/examples/detect.rs
cargo build --release --features png --example detect \
  --manifest-path target/baseline-src/Cargo.toml --target-dir target/baseline-build
cargo build --release --features png --example detect
uv run --no-project python tools/benchmark_compare.py \
  target/baseline-build/release/examples/detect target/release/examples/detect \
  --output target/comparison.json
```

The comparison script requires permission to read process resource statistics
on macOS (`/usr/bin/time -l`). It fails on command errors or changed marker output.

Criterion now defaults to **one thread**, with image/mode/thread identifiers:
`CCTAG_BENCH_THREADS=16 cargo bench --features png` selects 16 threads. Stage
benchmarks exclude thinning/identification input clones and vote input collection
from the timed section. The detection-only benchmark explicitly includes pyramid
construction. Earlier Criterion tables above describe the original port and its
old default-thread measurement setup.

### Criterion stage spot-check, 01.png, 1T

Exploratory `--quick` estimates (single sequential baseline/current sweep;
end-to-end acceptance uses the three-set measurements above):

| Stage | Before ms | After ms |
|---|---:|---:|
| L0/gradient_separable | 23.222 | 16.429 |
| L0/canny | 4.678 | 4.524 |
| L0/thinning | 2.115 | 2.075 |
| identification/parity | 50.947 | 35.327 |

All ten stage benchmarks completed. The gradient micro-benchmark includes its
compatibility wrapper allocation; the end-to-end pyramid reuses retained scratch.
Raw estimates: [`benches/results/stages-2026-09-09.json`](benches/results/stages-2026-09-09.json).
