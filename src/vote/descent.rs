//! `gradientDirectionDescent` (Bresenham.cpp:33-209): walk a Bresenham line
//! from an edge point along `dir * gradient` until another edge point is hit.
//!
//! Upstream re-reads the gradient at the *origin* point inside the
//! `thrGradient` block, so `dir` is unchanged and the block is a no-op; we keep
//! the walk semantics (two steps before the first lookup, neighbour probe on a
//! miss) without that dead code.

use crate::edge::{EdgeIdx, EdgePointCollection, NO_EDGE};

#[inline(always)]
fn sign_i(v: f32) -> i32 {
    // boost::math::sign<int>(float): the float is converted to int first.
    let i = v as i32;
    if i > 0 {
        1
    } else if i < 0 {
        -1
    } else {
        0
    }
}

/// `updateXY(dx, dy, x, y, e, stpX, stpY)`.
#[inline(always)]
fn update_xy(
    dx: f32,
    dy: f32,
    x: &mut i32,
    y: &mut i32,
    e: &mut f32,
    stp_x: &mut i32,
    stp_y: &mut i32,
) {
    let a = (dy / dx).abs();
    *stp_x = sign_i(dx);
    *stp_y = sign_i(dy);
    *e += a;
    *x += *stp_x;
    if *e >= 0.5 {
        *y += *stp_y;
        *e -= 1.0;
    }
}

/// Returns the index of the first edge point hit, or `NO_EDGE`.
pub fn gradient_direction_descent(
    coll: &EdgePointCollection,
    p: EdgeIdx,
    dir: i32,
    nmax: usize,
) -> EdgeIdx {
    let pt = coll.point(p);
    let mut e = 0.0f32;
    let dx = dir as f32 * pt.gx();
    let dy = dir as f32 * pt.gy();
    let adx = dx.abs();
    let ady = dy.abs();
    let mut n: usize = 0;
    let mut stp_x = 0i32;
    let mut stp_y = 0i32;
    let mut x = pt.x as i32;
    let mut y = pt.y as i32;

    if ady > adx {
        update_xy(dy, dx, &mut y, &mut x, &mut e, &mut stp_y, &mut stp_x);
        n += 1;
        update_xy(dy, dx, &mut y, &mut x, &mut e, &mut stp_y, &mut stp_x);
        n += 1;
        if coll.in_bounds(x, y) {
            let r = coll.at(x, y);
            if r != NO_EDGE {
                return r;
            }
        } else {
            return NO_EDGE;
        }
        while n <= nmax {
            update_xy(dy, dx, &mut y, &mut x, &mut e, &mut stp_y, &mut stp_x);
            n += 1;
            if coll.in_bounds(x, y) {
                let r = coll.at(x, y);
                if r != NO_EDGE {
                    return r;
                }
                if coll.in_bounds(x, y - stp_y) {
                    let r = coll.at(x, y - stp_y);
                    if r != NO_EDGE {
                        return r;
                    }
                } else {
                    return NO_EDGE;
                }
            } else {
                return NO_EDGE;
            }
        }
    } else {
        update_xy(dx, dy, &mut x, &mut y, &mut e, &mut stp_x, &mut stp_y);
        n += 1;
        update_xy(dx, dy, &mut x, &mut y, &mut e, &mut stp_x, &mut stp_y);
        n += 1;
        if coll.in_bounds(x, y) {
            let r = coll.at(x, y);
            if r != NO_EDGE {
                return r;
            }
        } else {
            return NO_EDGE;
        }
        while n <= nmax {
            update_xy(dx, dy, &mut x, &mut y, &mut e, &mut stp_x, &mut stp_y);
            n += 1;
            if coll.in_bounds(x, y) {
                let r = coll.at(x, y);
                if r != NO_EDGE {
                    return r;
                }
                if coll.in_bounds(x - stp_x, y) {
                    let r = coll.at(x - stp_x, y);
                    if r != NO_EDGE {
                        return r;
                    }
                } else {
                    return NO_EDGE;
                }
            } else {
                return NO_EDGE;
            }
        }
    }
    NO_EDGE
}
