//! Small utilities.

/// Epoch-stamped set over `0..n` indices: O(1) clear via epoch bump.
#[derive(Clone, Debug, Default)]
pub struct EpochSet {
    stamp: Vec<u32>,
    epoch: u32,
}

impl EpochSet {
    pub fn new(n: usize) -> Self {
        EpochSet {
            stamp: vec![0; n],
            epoch: 0,
        }
    }
    /// Make sure the set covers `n` items and start a fresh (empty) epoch.
    pub fn begin(&mut self, n: usize) {
        if self.stamp.len() < n {
            self.stamp.clear();
            self.stamp.resize(n, 0);
            self.epoch = 0;
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
    }
    #[inline(always)]
    pub fn insert(&mut self, i: usize) {
        self.stamp[i] = self.epoch;
    }
    #[inline(always)]
    pub fn remove(&mut self, i: usize) {
        self.stamp[i] = 0;
    }
    #[inline(always)]
    pub fn contains(&self, i: usize) -> bool {
        self.stamp[i] == self.epoch
    }
}
