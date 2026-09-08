//! Minimal dense image planes (row-major, no padding).

pub mod gray;
pub mod resize;

/// A dense 2-D plane with `w * h` elements, row-major, stride == width.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Plane<T> {
    pub w: usize,
    pub h: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Plane<T> {
    pub fn new(w: usize, h: usize) -> Self {
        Plane {
            w,
            h,
            data: vec![T::default(); w * h],
        }
    }

    pub fn filled(w: usize, h: usize, v: T) -> Self {
        Plane {
            w,
            h,
            data: vec![v; w * h],
        }
    }

    pub fn from_vec(w: usize, h: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), w * h, "plane data length mismatch");
        Plane { w, h, data }
    }

    /// Re-initialise to `v`, keeping the allocation (used by the workspace).
    pub fn fill(&mut self, v: T) {
        self.data.fill(v);
    }

    /// Resize (contents undefined afterwards except for `fill`).
    pub fn reset(&mut self, w: usize, h: usize, v: T) {
        self.w = w;
        self.h = h;
        self.data.clear();
        self.data.resize(w * h, v);
    }

    #[inline(always)]
    pub fn at(&self, x: usize, y: usize) -> T {
        self.data[y * self.w + x]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        self.data[y * self.w + x] = v;
    }

    #[inline(always)]
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.w..(y + 1) * self.w]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        &mut self.data[y * self.w..(y + 1) * self.w]
    }
}

pub type GrayImage = Plane<u8>;
pub type I16Plane = Plane<i16>;
