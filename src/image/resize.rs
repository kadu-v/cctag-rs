//! Pyramid down-scaling equivalent to `cv::resize(src, dst, Size(w/2, h/2))`
//! with the default `INTER_LINEAR` (Level.cpp:67).
//!
//! When both scale factors are exactly 2 OpenCV routes `INTER_LINEAR` to its
//! `INTER_AREA` fast path, which for `u8` is the rounded 2x2 box average
//! `(a + b + c + d + 2) >> 2`. That is the only case exercised by the pipeline
//! for even-sized inputs (upstream halves the dimensions with integer division).
//! Odd dimensions fall back to generic bilinear sampling (OpenCV's fixed-point
//! bilinear is *not* reproduced there; see PORTING_NOTES.md).

use super::GrayImage;

/// Down-scale to `(w/2, h/2)` (integer division), like `ImagePyramid` does.
pub fn downscale2(src: &GrayImage) -> GrayImage {
    let dw = src.w / 2;
    let dh = src.h / 2;
    let mut dst = GrayImage::new(dw, dh);
    downscale2_into(src, &mut dst);
    dst
}

pub fn downscale2_into(src: &GrayImage, dst: &mut GrayImage) {
    let dw = src.w / 2;
    let dh = src.h / 2;
    if dst.w != dw || dst.h != dh {
        dst.reset(dw, dh, 0);
    }
    if src.w == 2 * dw && src.h == 2 * dh {
        box2x2(src, dst);
    } else {
        bilinear_generic(src, dst);
    }
}

fn box2x2(src: &GrayImage, dst: &mut GrayImage) {
    let sw = src.w;
    for y in 0..dst.h {
        let r0 = &src.data[(2 * y) * sw..(2 * y + 1) * sw];
        let r1 = &src.data[(2 * y + 1) * sw..(2 * y + 2) * sw];
        let out = dst.row_mut(y);
        for (x, o) in out.iter_mut().enumerate() {
            let s =
                r0[2 * x] as u16 + r0[2 * x + 1] as u16 + r1[2 * x] as u16 + r1[2 * x + 1] as u16;
            *o = ((s + 2) >> 2) as u8;
        }
    }
}

/// Generic bilinear with OpenCV's pixel-centre convention (`(dx+0.5)*scale-0.5`),
/// float arithmetic, round-to-nearest. Not bit-exact with OpenCV's fixed-point path.
fn bilinear_generic(src: &GrayImage, dst: &mut GrayImage) {
    let sx = src.w as f32 / dst.w as f32;
    let sy = src.h as f32 / dst.h as f32;
    for y in 0..dst.h {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(src.h - 1);
        let y1 = (y0 + 1).min(src.h - 1);
        let wy = fy - y0 as f32;
        for x in 0..dst.w {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(src.w - 1);
            let x1 = (x0 + 1).min(src.w - 1);
            let wx = fx - x0 as f32;
            let p00 = src.at(x0, y0) as f32;
            let p10 = src.at(x1, y0) as f32;
            let p01 = src.at(x0, y1) as f32;
            let p11 = src.at(x1, y1) as f32;
            let v = p00 * (1.0 - wx) * (1.0 - wy)
                + p10 * wx * (1.0 - wy)
                + p01 * (1.0 - wx) * wy
                + p11 * wx * wy;
            dst.set(x, y, (v + 0.5) as u8);
        }
    }
}
