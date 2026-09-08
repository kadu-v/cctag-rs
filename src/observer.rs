//! Hook for stage dumps used by the parity tests (`examples/dump_stages.rs`).
//! The default implementation is a no-op and compiles away.

use crate::edge::EdgePointCollection;
use crate::pyramid::ImagePyramid;

pub trait StageObserver {
    fn on_pyramid(&mut self, _pyr: &ImagePyramid) {}
    fn on_edges(&mut self, _level: usize, _coll: &EdgePointCollection) {}
    fn on_vote(&mut self, _level: usize, _coll: &EdgePointCollection, _seeds: &[i32]) {}
}

pub struct NoopObserver;
impl StageObserver for NoopObserver {}
