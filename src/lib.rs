//! Pure-Rust port of the CCTag CPU detection pipeline
//! (alicevision/CCTag v1.0.4, MPL-2.0).
//!
//! Entry point: [`Detector::detect`]. See `PORTING_NOTES.md` for the list of
//! upstream quirks that are deliberately reproduced and bugs that were fixed.
//!
//! Lints: constants are copied verbatim from the C++ sources (more digits than
//! an `f32` holds), and index loops mirror the upstream code on purpose.
#![allow(
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_range_contains
)]

pub mod bank;
mod bank_tables;
pub mod detection;
pub mod edge;
pub mod filter;
pub mod fitting;
pub mod geometry;
pub mod growing;
pub mod identification;
pub mod image;
pub mod linalg;
pub mod marker;
pub mod observer;
pub mod params;
pub mod pyramid;
#[cfg(feature = "refdata")]
pub mod refdata;
pub mod robust;
#[cfg(feature = "synth")]
pub mod synth;
pub mod timing;
pub mod vote;

pub use bank::Bank;
pub use detection::{Detector, ExecMode};
pub use geometry::ellipse::Ellipse;
pub use image::Plane;
pub use marker::{Marker, Status};
pub use params::Params;
pub mod filelog;
pub mod util;
