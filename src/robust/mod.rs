//! Robust estimation helpers: PCG32 RNG (`Statistic.cpp`), medians
//! (`Statistic.hpp`, `Identification.hpp`), LMedS outlier removal and segment
//! assembling (`Vote.cpp:413-741`).

pub mod median;
pub mod outlier;
pub mod rng;

pub use median::{compute_median, median_ref};
pub use rng::Pcg32;
