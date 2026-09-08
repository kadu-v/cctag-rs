//! 9x9 derivative-of-Gaussian gradient, equivalent to the two `cv::filter2D`
//! calls in `filter/cvRecode.cpp:74-92`.
//!
//! The upstream kernel `kerneldX` (9x9 float, correlation, `BORDER_REPLICATE`,
//! output `CV_16S` via `saturate_cast<short>`) is exactly rank-1:
//! `K[r][c] = g[r] * d[c]` with `g[k] = exp(-k^2/2)` (unnormalised Gaussian,
//! sigma = 1) and `d[k] = k * exp(-k^2/2) / pi`. `kerneldY` is its transpose.
//! We therefore compute `dx = G_y * D_x` and `dy = D_y * G_x` separably.
//!
//! Note on parity: OpenCV runs `filter2D` for a 9x9 kernel through its DFT
//! path on machines without SSE3 (Apple silicon), so *no* spatial implementation
//! reproduces it bit for bit. The parity reference (`tools/cpp-ref` patch 0001)
//! replaces the OpenCV call with a direct double-precision correlation; this
//! module matches that reference exactly (see `derivatives_direct` for the
//! literal 81-tap formulation used in tests).

use crate::image::{GrayImage, I16Plane};

/// The upstream 9x9 kernel for the x derivative, verbatim (cvRecode.cpp:74-83).
pub const KERNEL_DX: [[f32; 9]; 9] = {
    const R0: [f32; 9] = [
        -0.000000143284235,
        -0.000003558691641,
        -0.000028902492951,
        -0.000064765993382,
        0.0,
        0.000064765993382,
        0.000028902492951,
        0.000003558691641,
        0.000000143284235,
    ];
    const R1: [f32; 9] = [
        -0.000004744922188,
        -0.000117847682078,
        -0.000957119116802,
        -0.002144755142391,
        0.0,
        0.002144755142391,
        0.000957119116802,
        0.000117847682078,
        0.000004744922188,
    ];
    const R2: [f32; 9] = [
        -0.000057804985902,
        -0.001435678675203,
        -0.011660097860113,
        -0.026128466569370,
        0.0,
        0.026128466569370,
        0.011660097860113,
        0.001435678675203,
        0.000057804985902,
    ];
    const R3: [f32; 9] = [
        -0.000259063973527,
        -0.006434265427174,
        -0.052256933138740,
        -0.117099663048638,
        0.0,
        0.117099663048638,
        0.052256933138740,
        0.006434265427174,
        0.000259063973527,
    ];
    const R4: [f32; 9] = [
        -0.000427124283626,
        -0.010608310271112,
        -0.086157117207395,
        -0.193064705260108,
        0.0,
        0.193064705260108,
        0.086157117207395,
        0.010608310271112,
        0.000427124283626,
    ];
    [R0, R1, R2, R3, R4, R3, R2, R1, R0]
};

const HALF: usize = 4;

#[inline(always)]
fn clamp_idx(i: isize, n: usize) -> usize {
    if i < 0 {
        0
    } else if i as usize >= n {
        n - 1
    } else {
        i as usize
    }
}

/// `saturate_cast<short>(cvRound(v))`: round half to even, then clamp.
#[inline(always)]
fn to_i16(v: f64) -> i16 {
    let r = v.round_ties_even();
    if r < i16::MIN as f64 {
        i16::MIN
    } else if r > i16::MAX as f64 {
        i16::MAX
    } else {
        r as i16
    }
}

/// Direct 81-tap correlation in f64 with replicate border, exactly what the
/// parity reference does. Slow; used for tests and as a fallback oracle.
pub fn derivatives_direct(src: &GrayImage, dx: &mut I16Plane, dy: &mut I16Plane) {
    let (w, h) = (src.w, src.h);
    dx.reset(w, h, 0);
    dy.reset(w, h, 0);
    for y in 0..h {
        for x in 0..w {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            for r in 0..9 {
                let yy = clamp_idx(y as isize + r as isize - HALF as isize, h);
                for c in 0..9 {
                    let xx = clamp_idx(x as isize + c as isize - HALF as isize, w);
                    let p = src.data[yy * w + xx] as f64;
                    sx += p * KERNEL_DX[r][c] as f64;
                    sy += p * KERNEL_DX[c][r] as f64;
                }
            }
            dx.data[y * w + x] = to_i16(sx);
            dy.data[y * w + x] = to_i16(sy);
        }
    }
}

/// Separable implementation. Accumulates in f64 like the parity reference; the
/// separable split changes the summation order, which is verified to be within
/// the f64→i16 rounding margin (see tests).
///
/// Rows are processed in blocks: the horizontal passes for the `block + 8` rows
/// a block needs go to a small ring buffer, so the f64 intermediates never leave
/// the cache. Blocks are independent and run in parallel when `parallel` is on.
pub fn derivatives(src: &GrayImage, dx: &mut I16Plane, dy: &mut I16Plane) {
    let (w, h) = (src.w, src.h);
    dx.reset(w, h, 0);
    dy.reset(w, h, 0);
    if w == 0 || h == 0 {
        return;
    }
    let d: [f64; 9] = core::array::from_fn(|c| KERNEL_DX[4][c] as f64);
    let g: [f64; 9] = core::array::from_fn(|r| KERNEL_DX[r][5] as f64 / d[5]);
    const BLOCK: usize = 16;
    let n_blocks = h.div_ceil(BLOCK);
    let work = |b: usize, odx: &mut [i16], ody: &mut [i16]| {
        let y0 = b * BLOCK;
        let y1 = (y0 + BLOCK).min(h);
        let rows = y1 - y0;
        // horizontal passes for rows y0-4 .. y1+3 (clamped), stored in a local buffer
        let nrows = rows + 8;
        let mut hx = vec![0.0f64; nrows * w];
        let mut hg = vec![0.0f64; nrows * w];
        for (ri, yy) in (y0 as isize - 4..(y1 + 4) as isize).enumerate() {
            let ys = clamp_idx(yy, h);
            let row = &src.data[ys * w..(ys + 1) * w];
            horizontal_pass(
                row,
                w,
                &d,
                &g,
                &mut hx[ri * w..(ri + 1) * w],
                &mut hg[ri * w..(ri + 1) * w],
            );
        }
        let mut accx = vec![0.0f64; w];
        let mut accy = vec![0.0f64; w];
        for (oy, y) in (y0..y1).enumerate() {
            accx.fill(0.0);
            accy.fill(0.0);
            for r in 0..9 {
                // row y + r - 4 is at local index (y - y0) + r
                let ri = (y - y0) + r;
                let gx = g[r];
                let dyk = d[r];
                let rx = &hx[ri * w..(ri + 1) * w];
                let rg = &hg[ri * w..(ri + 1) * w];
                for x in 0..w {
                    accx[x] += rx[x] * gx;
                    accy[x] += rg[x] * dyk;
                }
            }
            let ox = &mut odx[oy * w..(oy + 1) * w];
            let oy_ = &mut ody[oy * w..(oy + 1) * w];
            for x in 0..w {
                ox[x] = to_i16(accx[x]);
                oy_[x] = to_i16(accy[x]);
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        dx.data
            .par_chunks_mut(BLOCK * w)
            .zip(dy.data.par_chunks_mut(BLOCK * w))
            .enumerate()
            .for_each(|(b, (cx, cy))| work(b, cx, cy));
    }
    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..n_blocks {
            let s0 = b * BLOCK * w;
            let s1 = ((b + 1) * BLOCK * w).min(w * h);
            let (cx, cy) = (&mut dx.data[s0..s1], &mut dy.data[s0..s1]);
            work(b, cx, cy);
        }
    }
    let _ = n_blocks;
}

#[inline]
fn horizontal_pass(
    row: &[u8],
    w: usize,
    d: &[f64; 9],
    g: &[f64; 9],
    ox: &mut [f64],
    og: &mut [f64],
) {
    if w >= 9 {
        // interior
        for x in HALF..w - HALF {
            let mut sx = 0.0;
            let mut sg = 0.0;
            for c in 0..9 {
                let p = row[x + c - HALF] as f64;
                sx += p * d[c];
                sg += p * g[c];
            }
            ox[x] = sx;
            og[x] = sg;
        }
    }
    // borders (replicate)
    let border = |x: usize, ox: &mut [f64], og: &mut [f64]| {
        let mut sx = 0.0;
        let mut sg = 0.0;
        for c in 0..9 {
            let xx = clamp_idx(x as isize + c as isize - HALF as isize, w);
            let p = row[xx] as f64;
            sx += p * d[c];
            sg += p * g[c];
        }
        ox[x] = sx;
        og[x] = sg;
    };
    let lo = HALF.min(w);
    for x in 0..lo {
        border(x, ox, og);
    }
    let hi = w.saturating_sub(HALF);
    for x in hi.max(lo)..w {
        border(x, ox, og);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_is_rank_one() {
        let d: [f64; 9] = core::array::from_fn(|c| KERNEL_DX[4][c] as f64);
        let g: [f64; 9] = core::array::from_fn(|r| KERNEL_DX[r][5] as f64 / d[5]);
        for r in 0..9 {
            for c in 0..9 {
                let k = KERNEL_DX[r][c] as f64;
                let p = g[r] * d[c];
                assert!(
                    (k - p).abs() <= 1e-6 * k.abs().max(1e-7),
                    "K[{r}][{c}] = {k} vs {p}"
                );
            }
        }
        // g[k] = exp(-k^2/2), d[k] = k*exp(-k^2/2)/pi
        for k in 0..9i32 {
            let t = (k - 4) as f64;
            let ge = (-t * t / 2.0).exp();
            let de = t * (-t * t / 2.0).exp() / std::f64::consts::PI;
            assert!((g[k as usize] - ge).abs() < 1e-6, "g[{k}]");
            assert!((d[k as usize] - de).abs() < 1e-7, "d[{k}]");
        }
    }

    #[test]
    fn separable_matches_direct() {
        // pseudo-random small image
        let (w, h) = (37, 23);
        let mut s: u32 = 12345;
        let data: Vec<u8> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        let img = GrayImage::from_vec(w, h, data);
        let (mut a, mut b, mut c, mut d) = (
            I16Plane::new(0, 0),
            I16Plane::new(0, 0),
            I16Plane::new(0, 0),
            I16Plane::new(0, 0),
        );
        derivatives_direct(&img, &mut a, &mut b);
        derivatives(&img, &mut c, &mut d);
        assert_eq!(a, c);
        assert_eq!(b, d);
    }
}
