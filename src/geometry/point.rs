/// `Point2d<Eigen::Vector3f>` with `w == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Point2 { x, y }
    }
    #[inline(always)]
    pub fn hom(&self) -> [f32; 3] {
        [self.x, self.y, 1.0]
    }
}

/// `DirectedPoint2d<Eigen::Vector3f>`: a point plus its (unnormalised) gradient.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DirectedPoint {
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
}

impl DirectedPoint {
    #[inline(always)]
    pub const fn new(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        DirectedPoint { x, y, dx, dy }
    }
    #[inline(always)]
    pub fn pos(&self) -> Point2 {
        Point2::new(self.x, self.y)
    }
    #[inline(always)]
    pub fn hom(&self) -> [f32; 3] {
        [self.x, self.y, 1.0]
    }
}
