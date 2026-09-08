//! `EdgePointCollection` (Types.hpp/.cpp) with dynamic sizing.
//!
//! Upstream allocates fixed 6144^2 / 2^24 sized arrays per level and relies on
//! lazy page commit; we size everything from the actual image and point count.
//! The `set_bit`/`test_bit` aliasing bug (`v[i/4]` instead of `v[i/32]`) is not
//! reproduced: it only affected the two "processed" bit-sets whose semantics are
//! per-point booleans (see PORTING_NOTES.md).

use super::point::EdgePoint;

pub type EdgeIdx = i32;
pub const NO_EDGE: EdgeIdx = -1;

#[derive(Clone, Debug, Default)]
pub struct EdgePointCollection {
    /// Dimensions of the edge map (the level image).
    pub map_w: usize,
    pub map_h: usize,
    /// Bounds used by the walks (`shape()` upstream). Upstream sizes every
    /// collection with the *full-resolution* image, so walks at coarser levels
    /// are bounded by the full-res size while the map lookups outside the level
    /// image simply find no point. We keep both to stay exact.
    pub bound_w: usize,
    pub bound_h: usize,
    /// `map_w * map_h`, -1 = no point.
    pub edge_map: Vec<EdgeIdx>,
    pub points: Vec<EdgePoint>,
    /// `[before, after]` per point.
    pub links: Vec<[EdgeIdx; 2]>,
    /// CSR offsets (n+1) into `voters`.
    pub voters_off: Vec<u32>,
    pub voters: Vec<EdgeIdx>,
    // mutable per-point state (EdgePoint fields upstream)
    pub flow_length: Vec<f32>,
    pub is_max: Vec<i32>,
    pub n_segment_out: Vec<i32>,
    /// `_processed` per-runId bit mask.
    pub processed: Vec<u64>,
    processed_in: Vec<u64>,
    processed_aux: Vec<u64>,
}

impl EdgePointCollection {
    /// `EdgePointCollection(w, h)` where `(w, h)` is the map size; `bound` is
    /// the `shape()` used by the Bresenham/linking bounds checks.
    pub fn new(map_w: usize, map_h: usize, bound_w: usize, bound_h: usize) -> Self {
        let mut c = EdgePointCollection::default();
        c.reset(map_w, map_h, bound_w, bound_h);
        c
    }

    pub fn reset(&mut self, map_w: usize, map_h: usize, bound_w: usize, bound_h: usize) {
        self.map_w = map_w;
        self.map_h = map_h;
        self.bound_w = bound_w;
        self.bound_h = bound_h;
        self.edge_map.clear();
        self.edge_map.resize(map_w * map_h, NO_EDGE);
        self.points.clear();
        self.links.clear();
        self.voters_off.clear();
        self.voters.clear();
        self.flow_length.clear();
        self.is_max.clear();
        self.n_segment_out.clear();
        self.processed.clear();
        self.processed_in.clear();
        self.processed_aux.clear();
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// `add_point(x, y, dx, dy)`.
    #[inline]
    pub fn add_point(&mut self, x: usize, y: usize, dx: i16, dy: i16) -> EdgeIdx {
        let imap = y * self.map_w + x;
        debug_assert_eq!(self.edge_map[imap], NO_EDGE, "point already exists");
        let idx = self.points.len() as EdgeIdx;
        self.edge_map[imap] = idx;
        self.points.push(EdgePoint::new(x as i32, y as i32, dx, dy));
        self.links.push([NO_EDGE, NO_EDGE]);
        self.flow_length.push(0.0);
        self.is_max.push(-1);
        self.n_segment_out.push(-1);
        self.processed.push(0);
        idx
    }

    /// Finalise bit-sets after all points were added.
    pub fn finish_points(&mut self) {
        let words = self.points.len().div_ceil(64);
        self.processed_in.clear();
        self.processed_in.resize(words, 0);
        self.processed_aux.clear();
        self.processed_aux.resize(words, 0);
    }

    /// Point index at `(x, y)` or `NO_EDGE`. `(x, y)` must be inside the map.
    #[inline(always)]
    pub fn at(&self, x: i32, y: i32) -> EdgeIdx {
        if x < 0 || y < 0 || x as usize >= self.map_w || y as usize >= self.map_h {
            return NO_EDGE;
        }
        self.edge_map[y as usize * self.map_w + x as usize]
    }

    /// Bounds check used by upstream walks (`shape()`).
    #[inline(always)]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.bound_w && (y as usize) < self.bound_h
    }

    #[inline(always)]
    pub fn point(&self, i: EdgeIdx) -> &EdgePoint {
        &self.points[i as usize]
    }

    #[inline(always)]
    pub fn before(&self, i: EdgeIdx) -> EdgeIdx {
        self.links[i as usize][0]
    }

    #[inline(always)]
    pub fn after(&self, i: EdgeIdx) -> EdgeIdx {
        self.links[i as usize][1]
    }

    #[inline(always)]
    pub fn voters(&self, i: EdgeIdx) -> &[EdgeIdx] {
        let i = i as usize;
        &self.voters[self.voters_off[i] as usize..self.voters_off[i + 1] as usize]
    }

    #[inline(always)]
    pub fn voters_size(&self, i: EdgeIdx) -> usize {
        let i = i as usize;
        (self.voters_off[i + 1] - self.voters_off[i]) as usize
    }

    /// `create_voter_lists`: build the CSR from `(chosen, voter)` pairs given in
    /// voting order (stable counting sort, so each list keeps insertion order).
    pub fn create_voter_lists(&mut self, pairs: &[(EdgeIdx, EdgeIdx)]) {
        let n = self.points.len();
        self.voters_off.clear();
        self.voters_off.resize(n + 1, 0);
        for &(c, _) in pairs {
            self.voters_off[c as usize + 1] += 1;
        }
        for i in 0..n {
            self.voters_off[i + 1] += self.voters_off[i];
        }
        self.voters.clear();
        self.voters.resize(pairs.len(), NO_EDGE);
        let mut cursor: Vec<u32> = self.voters_off[..n].to_vec();
        for &(c, v) in pairs {
            let slot = &mut cursor[c as usize];
            self.voters[*slot as usize] = v;
            *slot += 1;
        }
    }

    #[inline(always)]
    pub fn test_processed_in(&self, i: EdgeIdx) -> bool {
        let i = i as usize;
        (self.processed_in[i / 64] >> (i % 64)) & 1 == 1
    }

    #[inline(always)]
    pub fn set_processed_in(&mut self, i: EdgeIdx, f: bool) {
        let i = i as usize;
        if f {
            self.processed_in[i / 64] |= 1u64 << (i % 64);
        } else {
            self.processed_in[i / 64] &= !(1u64 << (i % 64));
        }
    }

    #[inline(always)]
    pub fn test_processed_aux(&self, i: EdgeIdx) -> bool {
        let i = i as usize;
        (self.processed_aux[i / 64] >> (i % 64)) & 1 == 1
    }

    #[inline(always)]
    pub fn set_processed_aux(&mut self, i: EdgeIdx, f: bool) {
        let i = i as usize;
        if f {
            self.processed_aux[i / 64] |= 1u64 << (i % 64);
        } else {
            self.processed_aux[i / 64] &= !(1u64 << (i % 64));
        }
    }
}
