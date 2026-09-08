//! `edgesPointsFromCanny` (Canny.cpp:15-33): raster-scan the thinned edge map.

use super::EdgePointCollection;
use crate::image::{I16Plane, Plane};

pub fn edges_points_from_canny(
    coll: &mut EdgePointCollection,
    edges: &Plane<u8>,
    dx: &I16Plane,
    dy: &I16Plane,
) {
    let (w, h) = (edges.w, edges.h);
    debug_assert_eq!((coll.map_w, coll.map_h), (w, h));
    for y in 0..h {
        let er = edges.row(y);
        let xr = dx.row(y);
        let yr = dy.row(y);
        for x in 0..w {
            if er[x] == 255 {
                coll.add_point(x, y, xr[x], yr[x]);
            }
        }
    }
    coll.finish_points();
}
