//! Differential tests use frozen pre-optimization kernels, independent of the
//! new implementation. Small sizes exercise border and SIMD-tail handling.
pub use cctag::image;
#[allow(dead_code, clippy::excessive_precision, clippy::needless_range_loop)]
mod reference {
    pub mod canny;
    pub mod dog;
    pub mod thinning;
    pub mod thinning_lut;
}
use cctag::filter::{canny, dog, thinning};
use cctag::image::{GrayImage, I16Plane};
#[test]
fn kernels_match_original_with_reused_buffers() {
    let run = || {
        let (mut ax, mut ay, mut bx, mut by) = (
            I16Plane::default(),
            I16Plane::default(),
            I16Plane::default(),
            I16Plane::default(),
        );
        let (mut ae, mut be, mut at, mut bt) = (
            GrayImage::default(),
            GrayImage::default(),
            GrayImage::default(),
            GrayImage::default(),
        );
        let mut aws = canny::CannyWorkspace::default();
        let mut bws = reference::canny::CannyWorkspace::default();
        let mut seed = 123u32;
        for w in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 25, 31, 32, 33, 65] {
            for h in [0, 1, 2, 3, 8, 17, 33] {
                let data = (0..w * h)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        (seed >> 24) as u8
                    })
                    .collect();
                let img = GrayImage::from_vec(w, h, data);
                for _ in 0..2 {
                    dog::derivatives(&img, &mut ax, &mut ay);
                    reference::dog::derivatives(&img, &mut bx, &mut by);
                    assert_eq!(ax, bx, "dx {w}x{h}");
                    assert_eq!(ay, by, "dy {w}x{h}");
                    for (lo, hi) in [(2, 10), (0, 0), (10, 20), (100, 200)] {
                        canny::recoded_canny(&ax, &ay, lo, hi, &mut ae, &mut aws);
                        reference::canny::recoded_canny(&bx, &by, lo, hi, &mut be, &mut bws);
                        assert_eq!(ae, be, "Canny {w}x{h} {lo}/{hi}");
                        at.reset(w, h, 123);
                        bt.reset(w, h, 123);
                        thinning::thin(&mut ae, &mut at);
                        reference::thinning::thin(&mut be, &mut bt);
                        assert_eq!(ae, be, "thinning {w}x{h}");
                        assert_eq!(at, bt, "thinning scratch {w}x{h}");
                    }
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    for n in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(run);
    }
    #[cfg(not(feature = "parallel"))]
    run();
}
