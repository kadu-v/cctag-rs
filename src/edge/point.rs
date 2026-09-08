/// An edge pixel with its (unnormalised) image gradient, as extracted from the
/// `i16` derivative planes (`EdgePoint.hpp:24-82`). The mutable per-point state
/// (`_flowLength`, `_isMax`, `_nSegmentOut`, `_processed`) lives in
/// [`super::EdgePointCollection`] as separate arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct EdgePoint {
    pub x: i16,
    pub y: i16,
    pub dx: i16,
    pub dy: i16,
}

impl EdgePoint {
    #[inline(always)]
    pub fn new(x: i32, y: i32, dx: i16, dy: i16) -> Self {
        EdgePoint {
            x: x as i16,
            y: y as i16,
            dx,
            dy,
        }
    }
    /// `EdgePoint::dX()` — gradient as float.
    #[inline(always)]
    pub fn gx(&self) -> f32 {
        self.dx as f32
    }
    #[inline(always)]
    pub fn gy(&self) -> f32 {
        self.dy as f32
    }
    /// `EdgePoint::normGradient()` = `sqrt(dx*dx + dy*dy)` in f32.
    #[inline(always)]
    pub fn norm_grad(&self) -> f32 {
        let (gx, gy) = (self.gx(), self.gy());
        (gx * gx + gy * gy).sqrt()
    }
    #[inline(always)]
    pub fn xf(&self) -> f32 {
        self.x as f32
    }
    #[inline(always)]
    pub fn yf(&self) -> f32 {
        self.y as f32
    }
    /// Unnormalised dot product of the two gradients (`p.gradient().dot(q.gradient())`).
    #[inline(always)]
    pub fn grad_dot(&self, o: &EdgePoint) -> f32 {
        self.gx() * o.gx() + self.gy() * o.gy()
    }
}
