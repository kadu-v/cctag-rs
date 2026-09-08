//! Voting (`Bresenham.cpp`, `Vote.cpp:57-411`): field-line links, votes,
//! seeds, convex edge linking and children collection.

pub mod descent;
pub mod linking;
#[allow(clippy::module_inception)]
pub mod vote;

pub use linking::{children_of, edge_linking};
pub use vote::vote;
