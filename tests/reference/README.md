These kernels are frozen from commit `3c582d3` for differential testing:

- `dog.rs`: original separable, 16-row blocked implementation (its unrelated
  unit tests are omitted).
- `canny.rs`, `thinning.rs`, `thinning_lut.rs`: original front-end kernels.

Do not optimize these copies alongside the production kernels. They intentionally
keep the original arithmetic, initialization, and traversal order. The integration
test supplies `crate::image` from the library, so no second detector API is exposed.
