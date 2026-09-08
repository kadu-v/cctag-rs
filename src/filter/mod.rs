//! Image filters of the front-end: derivative-of-Gaussian gradients, the
//! recoded OpenCV-1.x Canny, and LUT thinning.

pub mod canny;
pub mod dog;
pub mod thinning;
mod thinning_lut;
