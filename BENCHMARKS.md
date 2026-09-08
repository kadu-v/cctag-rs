# Benchmarks — cctag-rs vs upstream C++ (CPU)

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
