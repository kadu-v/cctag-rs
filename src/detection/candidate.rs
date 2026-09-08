//! `Candidate.hpp`.

use crate::edge::EdgeIdx;
use crate::geometry::Ellipse;
use std::collections::VecDeque;

#[derive(Clone, Debug, Default)]
pub struct Candidate {
    pub seed: EdgeIdx,
    pub convex_edge_segment: VecDeque<EdgeIdx>,
    pub outer_ellipse_points: Vec<EdgeIdx>,
    pub outer_ellipse: Ellipse,
    pub filtered_children: Vec<EdgeIdx>,
    pub score: i32,
    pub n_label: usize,
    pub average_received_vote: f32,
}
