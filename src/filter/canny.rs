//! Port of the recoded OpenCV-1.x Canny in `filter/cvRecode.cpp:107-414`
//! (magnitude + non-maxima suppression + hysteresis), operating on the `i16`
//! derivative planes produced by [`super::dog`].

use crate::image::{I16Plane, Plane};

/// `TG22 = (int)(tan(22.5deg) * (1 << 15) + 0.5)`.
const TG22: i32 = 13573;
const CANNY_SHIFT: u32 = 15;

/// Scratch buffers reused across calls.
#[derive(Debug, Default)]
pub struct CannyWorkspace {
    /// `(w+2) * (h+2)` map: 0 = candidate, 1 = not edge, 2 = edge.
    map: Vec<u8>,
    /// Three rows of `w + 2` magnitudes (ring buffer).
    mag: Vec<i32>,
    stack: Vec<u32>,
}

/// Thresholds as computed by `Level::setLevel` + `cvRecodedCanny`:
/// `low = floor(thr_low * 256)`, `high = floor(thr_high * 256)` (after sorting).
pub fn thresholds(thr_low: f32, thr_high: f32) -> (i32, i32) {
    let a = (thr_low * 256.0) as f64;
    let b = (thr_high * 256.0) as f64;
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    (lo.floor() as i32, hi.floor() as i32)
}

/// Full Canny: writes `edges` as 0/255.
pub fn recoded_canny(
    dx: &I16Plane,
    dy: &I16Plane,
    low: i32,
    high: i32,
    edges: &mut Plane<u8>,
    ws: &mut CannyWorkspace,
) {
    let (w, h) = (dx.w, dx.h);
    assert_eq!((dy.w, dy.h), (w, h));
    edges.reset(w, h, 0);
    if w == 0 || h == 0 {
        return;
    }
    let mapstep = w + 2;
    ws.map.clear();
    ws.map.resize(mapstep * (h + 2), 0);
    ws.mag.clear();
    ws.mag.resize(3 * mapstep, 0);
    ws.stack.clear();

    let map = &mut ws.map;
    // map rows 0 and h+1 = 1
    map[..mapstep].fill(1);
    map[mapstep * (h + 1)..].fill(1);

    // ring buffer offsets into ws.mag: buf[k] starts at k*mapstep; element j at offset (j+1)
    let mut buf = [0usize, mapstep, 2 * mapstep];
    let mag = &mut ws.mag;
    // memset(mag_buf[0], 0)
    mag[buf[0]..buf[0] + mapstep].fill(0);

    for i in 0..=h {
        // _mag = mag_buf[(i > 0) + 1] + 1
        let cur = if i > 0 { buf[2] } else { buf[1] };
        if i < h {
            mag[cur] = 0;
            mag[cur + w + 1] = 0;
            let rdx = dx.row(i);
            let rdy = dy.row(i);
            let out = &mut mag[cur + 1..cur + 1 + w];
            for j in 0..w {
                let x = rdx[j] as i32;
                let y = rdy[j] as i32;
                let m = ((x as f32) * (x as f32) + (y as f32) * (y as f32)).sqrt();
                out[j] = m.round_ties_even() as i32;
            }
        } else {
            mag[cur..cur + mapstep].fill(0);
        }
        if i == 0 {
            continue;
        }
        // NMS on row i-1 using mag rows buf[0] (i-2), buf[1] (i-1), buf[2] (i)
        let map_row = mapstep * i; // _map = map + mapstep*i + 1 ; we index map_row + 1 + j
        map[map_row] = 1;
        map[map_row + w + 1] = 1;
        let c = buf[1] + 1; // central row, element j at c + j
        let magstep1 = buf[2] as isize - buf[1] as isize; // next row
        let magstep2 = buf[0] as isize - buf[1] as isize; // prev row
        let rdx = dx.row(i - 1);
        let rdy = dy.row(i - 1);
        let mut prev_flag = false;
        for j in 0..w {
            let mut x = rdx[j] as i32;
            let mut y = rdy[j] as i32;
            let s = x ^ y;
            let m = mag[c + j];
            x = x.abs();
            y = y.abs();
            let mp = map_row + 1 + j;
            if m > low {
                let tg22x = x * TG22;
                let tg67x = tg22x + ((x + x) << CANNY_SHIFT);
                y <<= CANNY_SHIFT;
                let idx = |off: isize| (c as isize + j as isize + off) as usize;
                let local_max = if y < tg22x {
                    Some(m > mag[idx(-1)] && m >= mag[idx(1)])
                } else if y > tg67x {
                    Some(m > mag[idx(magstep2)] && m >= mag[idx(magstep1)])
                } else {
                    let s = if s < 0 { -1 } else { 1 };
                    Some(m > mag[idx(magstep2 - s)] && m > mag[idx(magstep1 + s)])
                };
                if local_max == Some(true) {
                    if m > high && !prev_flag && map[mp - mapstep] != 2 {
                        map[mp] = 2;
                        ws.stack.push(mp as u32);
                        prev_flag = true;
                    } else {
                        map[mp] = 0;
                    }
                    continue;
                }
            }
            prev_flag = false;
            map[mp] = 1;
        }
        buf.rotate_left(1); // (0,1,2) -> (1,2,0): mag_buf[0]=old[1], [1]=old[2], [2]=old[0]
    }

    // hysteresis
    let ms = mapstep as u32;
    while let Some(p) = ws.stack.pop() {
        let p = p as usize;
        let mut push = |q: usize| {
            if map[q] == 0 {
                map[q] = 2;
                ws.stack.push(q as u32);
            }
        };
        push(p - 1);
        push(p + 1);
        push(p - ms as usize - 1);
        push(p - ms as usize);
        push(p - ms as usize + 1);
        push(p + ms as usize - 1);
        push(p + ms as usize);
        push(p + ms as usize + 1);
    }

    // final pass
    for i in 0..h {
        let mrow = &map[mapstep * (i + 1) + 1..mapstep * (i + 1) + 1 + w];
        let out = edges.row_mut(i);
        for j in 0..w {
            out[j] = 0u8.wrapping_sub(mrow[j] >> 1);
        }
    }
}
