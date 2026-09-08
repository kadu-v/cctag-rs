//! `edgeLinking` / `edgeLinkingDir` / `childrenOf` (Vote.cpp:248-411).

use crate::edge::{EdgeIdx, EdgePointCollection, NO_EDGE};
use std::collections::VecDeque;

const EDGE_NOT_FOUND: i32 = -1;
const CONVEXITY_LOST: i32 = -2;
const MAX_LENGTH: usize = 100;
const PI: f32 = std::f32::consts::PI; // boost::math::constants::pi<float>()

/// Marks for `processed_in` produced by a linking run: applied by the caller
/// (sequentially in seed order) so that the walk itself is side-effect free.
#[derive(Clone, Debug, Default)]
pub struct LinkResult {
    pub segment: VecDeque<EdgeIdx>,
    /// Points to `set_processed_in(true)`.
    pub marks: Vec<EdgeIdx>,
}

/// Visited-set replacement for `boost::flat_set<packxy>`: an epoch-stamped
/// array over the map. Reusable across seeds without clearing.
#[derive(Clone, Debug, Default)]
pub struct VisitedMap {
    stamp: Vec<u32>,
    epoch: u32,
    w: usize,
}

impl VisitedMap {
    pub fn new(w: usize, h: usize) -> Self {
        VisitedMap {
            stamp: vec![0; w * h],
            epoch: 0,
            w,
        }
    }
    pub fn ensure(&mut self, w: usize, h: usize) {
        if self.stamp.len() != w * h || self.w != w {
            self.stamp.clear();
            self.stamp.resize(w * h, 0);
            self.w = w;
            self.epoch = 0;
        }
    }
    fn begin(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
    }
    #[inline(always)]
    fn insert(&mut self, x: i32, y: i32) {
        self.stamp[y as usize * self.w + x as usize] = self.epoch;
    }
    #[inline(always)]
    fn contains(&self, x: i32, y: i32) -> bool {
        self.stamp[y as usize * self.w + x as usize] == self.epoch
    }
}

/// `edgeLinking(edgeCollection, segment, pmax, window, averageVoteMin)`.
/// Does not mutate the collection; the `processed_in` marks upstream would set
/// are returned in `LinkResult::marks` (including the seed itself).
pub fn edge_linking(
    coll: &EdgePointCollection,
    pmax: EdgeIdx,
    window: usize,
    average_vote_min: f32,
    visited: &mut VisitedMap,
) -> LinkResult {
    let mut res = LinkResult::default();
    visited.ensure(coll.map_w, coll.map_h);
    visited.begin();
    res.segment.push_back(pmax);
    res.marks.push(pmax);
    let p = coll.point(pmax);
    visited.insert(p.x as i32, p.y as i32);
    edge_linking_dir(coll, visited, pmax, 1, &mut res, window, average_vote_min);
    edge_linking_dir(coll, visited, pmax, -1, &mut res, window, average_vote_min);
    res
}

fn edge_linking_dir(
    coll: &EdgePointCollection,
    visited: &mut VisitedMap,
    mut p: EdgeIdx,
    dir: i32,
    res: &mut LinkResult,
    window: usize,
    average_vote_min: f32,
) {
    let mut phi: VecDeque<f32> = VecDeque::with_capacity(window + 1);
    let mut i = 0usize;
    let mut found = true;
    let mut stop = 0i32;
    let mut average_vote = coll.voters_size(p) as f32;
    const XOFF: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
    let yoff: [i32; 8] = [0, -dir, -dir, -dir, 0, dir, dir, dir];

    while i < MAX_LENGTH && found && average_vote >= average_vote_min {
        let pt = coll.point(p);
        let angle = (pt.gy().atan2(pt.gx()) + 2.0 * PI) % (2.0 * PI);
        phi.push_back(angle);
        if phi.len() > window {
            phi.pop_front();
        }
        // boost::math::round = half away from zero
        let shifting = (((angle + PI / 4.0) / (2.0 * PI)) * 8.0).round() as i32 - 1;

        let mut j = 0i32;
        stop = 0;
        while stop == 0 {
            if j >= 8 {
                stop = EDGE_NOT_FOUND;
            } else {
                let k = if dir == 1 {
                    ((8 - shifting + j) % 8) as usize
                } else {
                    ((shifting + j) % 8) as usize
                };
                let sx = pt.x as i32 + XOFF[k];
                let sy = pt.y as i32 + yoff[k];
                if coll.in_bounds(sx, sy) && coll.at(sx, sy) != NO_EDGE && !visited.contains(sx, sy)
                {
                    let d = *phi.back().unwrap() - *phi.front().unwrap();
                    if phi.len() == window {
                        if dir as f32 * d.sin() < 0.0 {
                            stop = CONVEXITY_LOST;
                        }
                    } else {
                        let s = dir as f32 * d.sin();
                        let c = d.cos();
                        if (s < -0.707 && c > 0.0) || (s < 0.0 && c < 0.0) {
                            stop = CONVEXITY_LOST;
                        }
                    }
                    if stop == 0 {
                        visited.insert(pt.x as i32, pt.y as i32);
                        p = coll.at(sx, sy);
                        if dir > 0 {
                            res.segment.push_back(p);
                        } else {
                            res.segment.push_front(p);
                        }
                        let sz = res.segment.len() as f32;
                        average_vote =
                            (average_vote * sz + coll.voters_size(p) as f32) / (sz + 1.0);
                        stop = 1;
                    }
                    // processed.insert(packxy(p->x(), p->y())) — p may have just moved
                    let cp = coll.point(p);
                    visited.insert(cp.x as i32, cp.y as i32);
                }
            }
            j += 1;
        }
        found = stop == 1;
        i += 1;
    }

    if i == MAX_LENGTH || stop == CONVEXITY_LOST {
        if res.segment.len() > window {
            let n = res.segment.len() - window;
            for &q in res.segment.iter().take(n) {
                res.marks.push(q);
            }
        }
    } else if stop == EDGE_NOT_FOUND {
        for &q in res.segment.iter() {
            res.marks.push(q);
        }
    }
}

/// `childrenOf` (Vote.cpp:395-411): all voters of segment points whose vote
/// count is at least `voteMax / 14` (integer division).
pub fn children_of(coll: &EdgePointCollection, segment: &VecDeque<EdgeIdx>) -> Vec<EdgeIdx> {
    let mut vote_max: usize = 1;
    for &e in segment {
        vote_max = vote_max.max(coll.voters_size(e));
    }
    let thr = vote_max / 14;
    let mut children = Vec::new();
    for &e in segment {
        let v = coll.voters(e);
        if !v.is_empty() && v.len() >= thr {
            children.extend_from_slice(v);
        }
    }
    children
}
