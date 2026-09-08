# Porting notes — CCTag C++ (v1.0.4, commit 7102144) → `cctag-rs`

This crate is a pure-Rust port of the **CPU** detection pipeline of
[alicevision/CCTag](https://github.com/alicevision/CCTag) (MPL-2.0). The CUDA
code path is dead upstream (`CCTAG_WITH_CUDA` is never defined for the sources),
so the CPU path is the only behaviour to reproduce.

## Execution modes

| Mode | Threads | RNG | Use |
|---|---|---|---|
| `ExecMode::Parity` | rayon only for pure, order-independent stages (gradient, links, field lines) | one PCG32 stream seeded 271828, consumed in the same order as the upstream `CCTAG_SERIALIZE` build | parity tests |
| `ExecMode::Fast` | as above plus identification in parallel over markers | same stream (loops 2/3 and the reprojection are sequential in both modes today) | production |

Both modes are deterministic for any thread count (tests `tier_c_e2e`).

## Verification tiers

* **Tier A (bit-exact, `tests/tier_a_exact.rs`)** against the `ref-exact` C++
  build (`tools/cpp-ref`, patched: direct double-precision derivative filter,
  zeroed thinning buffer, stable seed sort, RNG reset, `-ffp-contract=off`):
  gray, pyramid levels, `dx`/`dy`, Canny, thinned edges, edge-point list,
  before/after links, voters CSR, `flowLength`, seeds, loop-1 segments and
  children. Only `dx`/`dy` are allowed to differ by ±1 on ≤ 1e-5 of the samples
  (separable vs direct f64 summation order at rounding ties); everything
  downstream must still be identical.
* **Tier B (tolerance + decisions, `tests/tier_b_markers.rs`)**: pre-identification
  markers (level, outer ellipse to 1e-3 px / 1e-4 rel, quality to 1e-5 rel, ring
  point counts ±3) and final markers (id, status, centre ≤ 0.5 px for reliable
  markers).
* **Tier C (behaviour, `tests/tier_c_e2e.rs`, `tools/cpp-ref/regression_compare.sh`)**:
  determinism, thread-count independence, agreement with the unmodified upstream
  build (same candidate count, same reliable ids, centres within 0.5 px — the
  semantics of the upstream `regression --compare` tool), and the upstream
  `regression` tool itself run on our FileLog XML.

Upstream itself is not deterministic: with TBB the same image gives centres that
vary by up to ~0.9 px between runs and occasionally an extra unreliable candidate
(measured 5 runs on `sample/01.png`, `02.png`). Our Tier C tolerance is therefore
not tighter than upstream's own run-to-run spread.

## Why the back half is tolerance-level, not bit-exact

* Eigen evaluates `D1.transpose() * D1` (Halíř–Flusser scatter matrices) with
  its blocked GEMM kernel and the 3x3 eigenproblem with `EigenSolver`; those
  summation orders are not reproduced. Ellipse fits therefore agree to ~1e-6
  relative, not bitwise.
* Every RANSAC (`outlierRemoval`, `isAnotherSegment`, the level-0 reprojection)
  stops after N consecutive non-improvements of a median that depends on such a
  fit, so the number of RNG draws diverges slightly (e.g. 93836 vs 96546 on
  `01.png`) once a fit result differs in the last bit. Results stay within
  tolerance; the RNG draw count is reported by the Tier B test, not asserted.
* Identification amplifies ellipse differences of 1e-3 px into centre
  differences of up to ~0.3 px; where the rescaled ellipse is identical (level-0
  markers) the refined centre agrees to ~1e-4 px.

## Eigen arithmetic that *is* reproduced

* `Matrix3f * Matrix3f` / `Matrix3f * Vector3f` coefficient order:
  `a0*b0 + (a1*b1 + a2*b2)` (strided row, unrolled tree). For
  `A.transpose() * B` the row is contiguous and vectorised: `(a0*b0 + a1*b1) + a2*b2`
  (`Mat3::mul`, `Mat3::mul_tn`, `Mat3::mul_vec`).
* `Vector3f::dot` = `(p0 + p1) + p2`; `Vector6f::dot` (Sampson distance) =
  `((p0 + p1) + (p2 + p3)) + (p4 + p5)`.
* 3x3 inverse: cofactor form with `computeInverseWithCheck`'s threshold
  `dummy_precision() = 1e-5`.
* 5x5 `PartialPivLU`: unblocked LU with first-max pivoting, fully unrolled
  triangular solves whose row dot products use the unrolled binary tree
  (verified bit-exact against Eigen on a test system).
* `boost::accumulators` `tag::variance` is the *immediate* recurrence
  `var = var*(n-1)/n + (x-mean_n)^2/(n-1)` with the lazy mean `sum/n`.

## Reproduced quirks (behaviour-affecting)

* `getPixelBilinear` divides by 2 and truncates coordinates with `(int)`.
* `ImageCut::outOfBounds` is sticky across grid points and passes; out-of-bounds
  samples keep their stale values.
* `costFunctionGlob`: `res += std::pow(float, 2)` promotes to double and narrows
  to float at every step; the mean is over cut *pairs*.
* `computeMedian` (even size) averages in double and narrows; `medianRef` is the
  upper median (`v[n/2]`).
* `while (neighbourSize * maxSemiAxis > 0.02)` compares in double;
  `neighbourSize /= (gridNSample-1)/2` is an integer division.
* `sortedId[v] = idc` (`std::map<float,...>`) collapses equal scores, last id wins;
  `outerEdgeRefinement`'s map keeps the first of equal keys.
* `_thrGradientMagInVote` is a no-op upstream (re-reads the origin gradient).
* `voteMax / 14` integer division; `_averageReceivedVote = (n*n) / nVoted`.
* `cctagPoints.resize(numCircles)` keeps rings added by a merged segment.
* `isAnotherSegment` draws 5 indices and uses 4; `fitEllipse` failures there
  abort the whole loop-three iteration (outside the inner `try`).
* Bounds: `cutInterpolated` needs `x >= 1`, `extractSignalUsingHomography` `x >= 0`.
* Literal types: comparisons with unsuffixed double literals (`0.7`, `0.25`,
  `0.05`, `20`, `0.666`, `1.5`, `0.12`, `1.1`, `0.02`, `300.0`) are done in f64.
* `CCTag` constructor shifts the outer ellipse centre by +0.5 px but not
  `centerImg`; level>0 markers get `centerImg *= scale` only if the reprojection fit succeeds.
* Walk bounds (`gradientDirectionDescent`, `edgeLinking`, `connectedPoint`)
  use the full-resolution image size at every pyramid level.
* Upstream's `CCTag` copy constructor drops `_rescaledOuterEllipsePoints` and
  `_idSet` is never filled; `idSet` is not exposed.

## Fixed (result-neutral) or deliberately different

* `EdgePointCollection::set_bit` uses `v[i/4]` (injective but 8x wasteful) —
  replaced by a normal bit-set.
* Thinning's intermediate buffer border is uninitialised upstream; defined as 0
  here (the parity build is patched the same way).
* Seeds are sorted with a stable sort (upstream `std::sort` leaves tie order
  implementation-defined; parity build patched to `stable_sort`).
* `connectedPoint` recursion → explicit stack with identical visit order.
* `conditionerFromEllipse`'s `static const float meanAB` (frozen to the first
  ellipse of the process) → computed per call.
* Fixed 6144² / 2²⁴ arrays → sized from the image.
* `cv::filter2D` is a DFT on Apple silicon (kernel area 81 ≥ 50 threshold); the
  parity reference and this port use a direct f64 correlation. Differences to the
  DFT output are ±1 LSB on a small fraction of pixels.
* `cv::resize` exact-2x path (`(a+b+c+d+2)>>2`) reproduced; odd image sizes fall
  back to plain bilinear (OpenCV's fixed-point bilinear is not reproduced).
* `orazioDistanceRobust` runs over cuts in order (upstream: TBB, mutex order).

## Not ported (dead upstream)

`SubPixEdgeOptimizer` (OPT++), `CCTagFlowComponent` / `DataSerialization`
(serialize-only), `cuda/`, `orazioDistance`, `refineConicFamily`, `blurImageCut`,
`costSelectCutFun`, `CCTagMarkersBank::identify`, `Parameters::Override`.

## Upstream `regression` tool caveats

* Its `FileLog` XML stores `quality = 1/residual`; a diverged candidate has
  `residual = FLT_MAX`, so the quality is a denormal that boost's XML reader
  rejects (`input stream error`) — the tool cannot read some of its own outputs.
* `TestRunner` deletes its destination directory on construction, and the RNG
  state carries over between images processed in one run, so
  `tools/cpp-ref/regression_compare.sh` runs one process per image.
* The comparison requires the *total* candidate count (including unreliable
  ones) to match; upstream's own runs differ in that count occasionally.

## Result-preserving single-thread optimizations (2026-09-09)

The serial gradient uses a nine-row circular buffer retained per pyramid level,
removing repeated interior-row filtering at 16-row block boundaries. Parallel
filtering retains independent blocks. AArch64 NEON processes neighboring pixels
in the horizontal f64 filter and Canny magnitude calculation. The 25-point centre
search uses one SIMD lane per grid point, with separate multiply/add and the
original f32 narrowing after every cost term. Other grid sizes retain the generic
path, and other architectures use scalar kernels. No detection parameters,
thresholds, RNG consumption, or tie-breaking rules change.

`tests/optimization_regression.rs` compares against snapshots captured from
`3c582d3` on aarch64-apple-darwin with Rust 1.89. The 17 cases include both sample
images, all eight stored synthetic scenes, odd-sized 3/4-crown scenes with 5/7/9
point grids per axis, and a blank image. It checks pre-identification and final
markers (float fields encoded by IEEE bits), RNG draws, pyramid planes, Canny,
thinning, edge collection, vote state and seeds. Large arrays are represented by
stable FNV-1a digests. Both modes, 1/4/16 threads, repeated calls and detector reuse
across sizes agree exactly. There is no floating-point tolerance exception in
this change; measured final coordinate differences are zero. The snapshot's
reference-capture test is explicitly ignored and refuses to run on modified
production source; missing required images are errors.

`tests/kernel_regression.rs` retains the original kernels as independent oracles
for empty/tiny images, borders, tails, thresholds, and reused scratch. The centre
reference tests also exercise different grids, sticky out-of-bounds flags, stale
samples and no readable pairs. Magnitude tests cover the full i16 range with
several gradient pairings and every eight-lane tail length. Existing Tier A/B/C
C++ comparisons and their tolerances are unchanged. Exact Rust snapshots are a
regression corpus, not a guarantee of bit identity across different toolchains or
floating-point environments; non-AArch64 execution performance has not been
measured here.
