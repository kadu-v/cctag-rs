//! Edge points and their collection (`EdgePoint.hpp`, `Types.hpp/.cpp`, `Canny.cpp`).

pub mod collection;
pub mod from_canny;
pub mod point;

pub use collection::{EdgeIdx, EdgePointCollection, NO_EDGE};
pub use point::EdgePoint;
