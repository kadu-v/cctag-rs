//! Two-pass LUT thinning (`filter/thinning.cpp`).
//!
//! Upstream leaves the one-pixel border of the intermediate buffer
//! uninitialised (`Level.cpp:40`), which makes the second pass read garbage at
//! the innermost ring. We define that border as 0 (the parity reference is
//! patched the same way). The border of `inout` itself is never written by
//! either pass and keeps the Canny output, exactly as upstream.

use super::thinning_lut::{LUTTHIN1, LUTTHIN2};
use crate::image::Plane;

pub fn thin(inout: &mut Plane<u8>, temp: &mut Plane<u8>) {
    if temp.w != inout.w || temp.h != inout.h {
        temp.reset(inout.w, inout.h, 0);
    } else {
        // Border must be 0 (see module docs); interior is fully overwritten.
        temp.fill(0);
    }
    image_iter(inout, temp, &LUTTHIN1);
    image_iter(temp, inout, &LUTTHIN2);
}

fn image_iter(inp: &Plane<u8>, out: &mut Plane<u8>, lut: &[u8; 512]) {
    let w = inp.w;
    let h = inp.h;
    if w < 3 || h < 3 {
        return;
    }
    for y in 1..h - 1 {
        let rm1 = &inp.data[(y - 1) * w..y * w];
        let r0 = &inp.data[y * w..(y + 1) * w];
        let rp1 = &inp.data[(y + 1) * w..(y + 2) * w];
        let o = &mut out.data[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            if r0[x] == 0 {
                o[x] = 0;
            } else {
                let ind = (rm1[x - 1] == 255) as usize
                    + ((rm1[x] == 255) as usize) * 8
                    + ((rm1[x + 1] == 255) as usize) * 64
                    + ((r0[x - 1] == 255) as usize) * 2
                    + ((r0[x] == 255) as usize) * 16
                    + ((r0[x + 1] == 255) as usize) * 128
                    + ((rp1[x - 1] == 255) as usize) * 4
                    + ((rp1[x] == 255) as usize) * 32
                    + ((rp1[x + 1] == 255) as usize) * 256;
                o[x] = lut[ind];
            }
        }
    }
}
