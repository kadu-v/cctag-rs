//! Image pyramid (`ImagePyramid.cpp`, `Level.cpp`): per level the (down-scaled)
//! source, the `i16` derivatives and the thinned Canny edge map.

use crate::filter::canny::{self, CannyWorkspace};
use crate::filter::{dog, thinning};
use crate::image::{GrayImage, I16Plane, Plane};
use crate::params::Params;

#[derive(Clone, Debug, Default)]
pub struct Level {
    pub level: usize,
    pub src: GrayImage,
    pub dx: I16Plane,
    pub dy: I16Plane,
    pub edges: Plane<u8>,
}

#[derive(Debug, Default)]
struct LevelScratch {
    temp: Plane<u8>,
    canny_ws: CannyWorkspace,
    derivative_ws: dog::DerivativeWorkspace,
}

#[derive(Debug, Default)]
pub struct ImagePyramid {
    pub levels: Vec<Level>,
    scratch: Vec<LevelScratch>,
}

impl ImagePyramid {
    pub fn new() -> Self {
        Self::default()
    }

    /// `ImagePyramid(w, h, nLevels)` + `build(src, thrLow, thrHigh)`.
    /// Level dimensions are successive integer halvings of the input size.
    /// The down-scaling chain is sequential; the per-level filtering (gradient,
    /// Canny, thinning) is independent across levels and runs in parallel.
    pub fn build(&mut self, src: &GrayImage, params: &Params) {
        let n = params.number_of_processed_multires_layers;
        let (low, high) = canny::thresholds(params.canny_thr_low, params.canny_thr_high);
        self.levels.resize_with(n, Level::default);
        self.scratch.resize_with(n, LevelScratch::default);
        for i in 0..n {
            if i == 0 {
                let lvl = &mut self.levels[0];
                lvl.level = 0;
                lvl.src.reset(src.w, src.h, 0);
                lvl.src.data.copy_from_slice(&src.data);
            } else {
                let (prev, rest) = self.levels.split_at_mut(i);
                let lvl = &mut rest[0];
                lvl.level = i;
                crate::image::resize::downscale2_into(&prev[i - 1].src, &mut lvl.src);
            }
        }
        let filter = |lvl: &mut Level, sc: &mut LevelScratch| {
            dog::derivatives_with_workspace(
                &lvl.src,
                &mut lvl.dx,
                &mut lvl.dy,
                &mut sc.derivative_ws,
            );
            canny::recoded_canny(
                &lvl.dx,
                &lvl.dy,
                low,
                high,
                &mut lvl.edges,
                &mut sc.canny_ws,
            );
            thinning::thin(&mut lvl.edges, &mut sc.temp);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.levels
                .par_iter_mut()
                .zip(self.scratch.par_iter_mut())
                .for_each(|(l, s)| filter(l, s));
        }
        #[cfg(not(feature = "parallel"))]
        for (l, s) in self.levels.iter_mut().zip(self.scratch.iter_mut()) {
            filter(l, s);
        }
    }

    pub fn level(&self, i: usize) -> &Level {
        &self.levels[i]
    }
}
