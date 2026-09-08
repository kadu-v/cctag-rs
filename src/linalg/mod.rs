//! Small dense linear algebra used by the pipeline (3x3 conics/homographies,
//! 2x2 symmetric eigen, 5x5 LU, 3x3 general eigen, 4x4 symmetric eigen).
//! Written to mirror the Eigen routines upstream relies on closely enough for
//! tolerance-level parity (see PORTING_NOTES.md).

pub type Vec3 = [f32; 3];

/// Row-major 3x3 f32 matrix.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Mat3(pub [[f32; 3]; 3]);

impl Mat3 {
    pub const ZERO: Mat3 = Mat3([[0.0; 3]; 3]);
    pub const IDENTITY: Mat3 = Mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    #[inline(always)]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.0[r][c]
    }
    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.0[r][c] = v;
    }
    pub fn transpose(&self) -> Mat3 {
        let m = &self.0;
        Mat3([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }
    /// `A * B` for two plain (column-major) Eigen matrices. Eigen evaluates each
    /// coefficient as `A.row(i).cwiseProduct(B.col(j)).sum()`; the strided row
    /// prevents vectorisation, so the 3-term sum is the unrolled tree
    /// `a0*b0 + (a1*b1 + a2*b2)`.
    pub fn mul(&self, o: &Mat3) -> Mat3 {
        let mut r = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = self.0[i][0] * o.0[0][j]
                    + (self.0[i][1] * o.0[1][j] + self.0[i][2] * o.0[2][j]);
            }
        }
        Mat3(r)
    }
    /// `A^T * B` written upstream as `A.transpose() * B`: the lhs row is a
    /// contiguous column of `A`, so Eigen vectorises the 3-term reduction with a
    /// 2-lane packet: `(a0*b0 + a1*b1) + a2*b2`.
    pub fn mul_tn(&self, o: &Mat3) -> Mat3 {
        let mut r = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = (self.0[0][i] * o.0[0][j] + self.0[1][i] * o.0[1][j])
                    + self.0[2][i] * o.0[2][j];
            }
        }
        Mat3(r)
    }
    /// `M * v` (Eigen tree order per coefficient, see [`Mat3::mul`]).
    pub fn mul_vec(&self, v: &Vec3) -> Vec3 {
        let m = &self.0;
        [
            m[0][0] * v[0] + (m[0][1] * v[1] + m[0][2] * v[2]),
            m[1][0] * v[0] + (m[1][1] * v[1] + m[1][2] * v[2]),
            m[2][0] * v[0] + (m[2][1] * v[1] + m[2][2] * v[2]),
        ]
    }
    /// `A^T * B * A` (conic transform), evaluated like `A.transpose() * B * A`.
    pub fn sandwich(&self, b: &Mat3) -> Mat3 {
        self.mul_tn(b).mul(self)
    }
    pub fn scale(&self, s: f32) -> Mat3 {
        let mut r = *self;
        for row in r.0.iter_mut() {
            for v in row.iter_mut() {
                *v *= s;
            }
        }
        r
    }

    /// Eigen `cofactor_3x3<i,j>`.
    #[inline(always)]
    fn cofactor(&self, i: usize, j: usize) -> f32 {
        let m = &self.0;
        let i1 = (i + 1) % 3;
        let i2 = (i + 2) % 3;
        let j1 = (j + 1) % 3;
        let j2 = (j + 2) % 3;
        m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]
    }

    /// Eigen `Matrix3f::determinant()` (cofactor expansion along column 0).
    pub fn determinant(&self) -> f32 {
        let m = &self.0;
        self.cofactor(0, 0) * m[0][0]
            + self.cofactor(1, 0) * m[1][0]
            + self.cofactor(2, 0) * m[2][0]
    }

    fn inverse_with_det(&self) -> (Mat3, f32) {
        let m = &self.0;
        let c00 = self.cofactor(0, 0);
        let c10 = self.cofactor(1, 0);
        let c20 = self.cofactor(2, 0);
        let det = c00 * m[0][0] + c10 * m[1][0] + c20 * m[2][0];
        let invdet = 1.0 / det;
        let mut r = [[0.0f32; 3]; 3];
        r[0][0] = c00 * invdet;
        r[0][1] = c10 * invdet;
        r[0][2] = c20 * invdet;
        r[1][0] = self.cofactor(0, 1) * invdet;
        r[1][1] = self.cofactor(1, 1) * invdet;
        r[1][2] = self.cofactor(2, 1) * invdet;
        r[2][0] = self.cofactor(0, 2) * invdet;
        r[2][1] = self.cofactor(1, 2) * invdet;
        r[2][2] = self.cofactor(2, 2) * invdet;
        (Mat3(r), det)
    }

    /// Eigen `Matrix3f::inverse()` (no check; may produce inf/NaN).
    pub fn inverse(&self) -> Mat3 {
        self.inverse_with_det().0
    }

    /// Eigen `computeInverseWithCheck` with the default threshold
    /// `NumTraits<float>::dummy_precision() = 1e-5`.
    pub fn inverse_checked(&self) -> Option<Mat3> {
        let (inv, det) = self.inverse_with_det();
        if det.abs() > 1e-5 { Some(inv) } else { None }
    }
}

/// Partial-pivot LU solve of a 5x5 system (Eigen `A.lu().solve(b)` with
/// `PartialPivLU`). Returns `None` when `A.determinant() == 0` exactly, which
/// is how upstream guards the call (Vote.cpp:490).
pub fn lu5_solve(a: &[[f32; 5]; 5], b: &[f32; 5]) -> Option<[f32; 5]> {
    let mut lu = *a;
    let mut perm = [0usize, 1, 2, 3, 4];
    let mut det = 1.0f32;
    for k in 0..5 {
        // pivot: max |lu[i][k]| for i >= k (first max wins, like Eigen's maxCoeff)
        let mut piv = k;
        let mut best = lu[k][k].abs();
        for i in k + 1..5 {
            let v = lu[i][k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if piv != k {
            lu.swap(piv, k);
            perm.swap(piv, k);
            det = -det;
        }
        let p = lu[k][k];
        det *= p;
        if p != 0.0 {
            for i in k + 1..5 {
                lu[i][k] /= p;
            }
            for i in k + 1..5 {
                let f = lu[i][k];
                if f != 0.0 {
                    for j in k + 1..5 {
                        lu[i][j] -= f * lu[k][j];
                    }
                }
            }
        }
    }
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    // Eigen's fixed-size triangular solves are fully unrolled: each unknown is
    // resolved with a dot product of a (strided) matrix row and the rhs
    // segment, reduced with the unrolled binary tree used by `redux_novec_unroller`.
    #[inline(always)]
    fn tree_sum(v: &[f32]) -> f32 {
        match v.len() {
            0 => 0.0,
            1 => v[0],
            2 => v[0] + v[1],
            3 => v[0] + (v[1] + v[2]),
            _ => {
                let h = v.len() / 2;
                tree_sum(&v[..h]) + tree_sum(&v[h..])
            }
        }
    }
    // forward: L y = P b (unit lower)
    let mut y = [0.0f32; 5];
    for d in 0..5 {
        let mut v = b[perm[d]];
        if d > 0 {
            let prods: Vec<f32> = (0..d).map(|j| lu[d][j] * y[j]).collect();
            v -= tree_sum(&prods);
        }
        y[d] = v;
    }
    // backward: U x = y
    let mut x = y;
    for k in 0..5 {
        let d = 4 - k;
        if k > 0 {
            let prods: Vec<f32> = (d + 1..5).map(|j| lu[d][j] * x[j]).collect();
            x[d] -= tree_sum(&prods);
        }
        x[d] /= lu[d][d];
    }
    Some(x)
}

/// Eigen-decomposition of a symmetric 2x2 matrix `[[a, b], [b, c]]` mirroring
/// `JacobiSVD<Matrix2f>(S, ComputeFullU)` for SPD input: returns
/// `(singular values descending, U)` with `U` column-major as `[[u00,u10],[u01,u11]]`
/// (i.e. `u[col][row]`).
pub fn eig2_sym(a: f32, b: f32, c: f32) -> ([f32; 2], [[f32; 2]; 2]) {
    // Jacobi rotation diagonalising S in f64, then order descending.
    let (a64, b64, c64) = (a as f64, b as f64, c as f64);
    let (cs, sn) = if b64 == 0.0 {
        (1.0f64, 0.0f64)
    } else {
        let tau = (c64 - a64) / (2.0 * b64);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let t = if tau == 0.0 { 1.0 } else { t };
        let cs = 1.0 / (1.0 + t * t).sqrt();
        (cs, t * cs)
    };
    // eigenvalues
    let l0 = a64 - (if b64 == 0.0 { 0.0 } else { sn / cs * b64 });
    let l1 = c64 + (if b64 == 0.0 { 0.0 } else { sn / cs * b64 });
    // eigenvectors: v0 = (cs, -sn), v1 = (sn, cs)
    let mut vals = [l0, l1];
    let mut vecs = [[cs, -sn], [sn, cs]];
    if vals[1] > vals[0] {
        vals.swap(0, 1);
        vecs.swap(0, 1);
    }
    // Eigen's JacobiSVD returns non-negative singular values; for SPD input they
    // coincide with the eigenvalues. Make the first column have positive x like
    // Eigen's convention is irrelevant for the angle mod pi.
    (
        [vals[0] as f32, vals[1] as f32],
        [
            [vecs[0][0] as f32, vecs[0][1] as f32],
            [vecs[1][0] as f32, vecs[1][1] as f32],
        ],
    )
}

/// Real eigen-decomposition of a general 3x3 matrix (Eigen `EigenSolver<Matrix3f>`,
/// taking `.real()` of the results). Returns three `(eigenvalue_re, unit eigenvector_re)`.
/// Computed in f64 via the characteristic cubic + null-space cross products.
pub fn eig3_general(m: &Mat3) -> [(f32, Vec3); 3] {
    let a: [[f64; 3]; 3] = core::array::from_fn(|i| core::array::from_fn(|j| m.0[i][j] as f64));
    // characteristic polynomial: λ^3 - tr λ^2 + c1 λ - det = 0
    let tr = a[0][0] + a[1][1] + a[2][2];
    let c1 = a[0][0] * a[1][1] - a[0][1] * a[1][0] + a[0][0] * a[2][2] - a[0][2] * a[2][0]
        + a[1][1] * a[2][2]
        - a[1][2] * a[2][1];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let roots = cubic_roots(-tr, c1, -det);
    let mut out = [(0.0f32, [0.0f32; 3]); 3];
    for (k, lam) in roots.iter().enumerate() {
        let v = null_vector_3(&a, *lam);
        out[k] = (lam.re as f32, [v[0] as f32, v[1] as f32, v[2] as f32]);
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct C64 {
    re: f64,
    im: f64,
}
impl C64 {
    fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Roots of x^3 + p x^2 + q x + r (all three, complex allowed), refined by Newton.
fn cubic_roots(p: f64, q: f64, r: f64) -> [C64; 3] {
    // depressed cubic t^3 + a t + b with x = t - p/3
    let a = q - p * p / 3.0;
    let b = 2.0 * p * p * p / 27.0 - p * q / 3.0 + r;
    let disc = (b * b) / 4.0 + (a * a * a) / 27.0;
    let shift = -p / 3.0;
    let mut roots = if disc <= 0.0 {
        // three real roots (trigonometric)
        let m = 2.0 * (-a / 3.0).max(0.0).sqrt();
        let arg = if m == 0.0 {
            0.0
        } else {
            (-4.0 * b / (m * m * m)).clamp(-1.0, 1.0)
        };
        let theta = arg.acos() / 3.0;
        let two_pi_3 = 2.0 * std::f64::consts::PI / 3.0;
        [
            C64::new(m * theta.cos() + shift, 0.0),
            C64::new(m * (theta - two_pi_3).cos() + shift, 0.0),
            C64::new(m * (theta + two_pi_3).cos() + shift, 0.0),
        ]
    } else {
        let sd = disc.sqrt();
        let u = (-b / 2.0 + sd).cbrt();
        let v = (-b / 2.0 - sd).cbrt();
        let t1 = u + v;
        let re = -t1 / 2.0;
        let im = (u - v) * 3f64.sqrt() / 2.0;
        [
            C64::new(t1 + shift, 0.0),
            C64::new(re + shift, im),
            C64::new(re + shift, -im),
        ]
    };
    // Newton refinement on the original cubic (complex arithmetic)
    for root in roots.iter_mut() {
        for _ in 0..3 {
            let x = *root;
            let f = x
                .mul(x)
                .mul(x)
                .add(C64::new(p, 0.0).mul(x).mul(x))
                .add(C64::new(q, 0.0).mul(x))
                .add(C64::new(r, 0.0));
            let df = C64::new(3.0, 0.0)
                .mul(x)
                .mul(x)
                .add(C64::new(2.0 * p, 0.0).mul(x))
                .add(C64::new(q, 0.0));
            let n2 = df.norm2();
            if n2 == 0.0 {
                break;
            }
            // x -= f/df
            let conj = C64::new(df.re, -df.im);
            let quot = f.mul(conj);
            *root = x.sub(C64::new(quot.re / n2, quot.im / n2));
        }
        if root.im.abs() < 1e-12 * (1.0 + root.re.abs()) {
            root.im = 0.0;
        }
    }
    roots
}

/// Unit null vector of `(A - λI)` (real part for complex λ), via the largest
/// cross product of two rows, as a general 3x3 eigenvector.
fn null_vector_3(a: &[[f64; 3]; 3], lam: C64) -> [f64; 3] {
    let m: [[C64; 3]; 3] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let d = if i == j { lam } else { C64::new(0.0, 0.0) };
            C64::new(a[i][j], 0.0).sub(d)
        })
    });
    let cross = |r0: &[C64; 3], r1: &[C64; 3]| -> [C64; 3] {
        [
            r0[1].mul(r1[2]).sub(r0[2].mul(r1[1])),
            r0[2].mul(r1[0]).sub(r0[0].mul(r1[2])),
            r0[0].mul(r1[1]).sub(r0[1].mul(r1[0])),
        ]
    };
    let cands = [
        cross(&m[0], &m[1]),
        cross(&m[1], &m[2]),
        cross(&m[0], &m[2]),
    ];
    let mut best = cands[0];
    let mut bn = 0.0;
    for c in cands.iter() {
        let n = c[0].norm2() + c[1].norm2() + c[2].norm2();
        if n > bn {
            bn = n;
            best = *c;
        }
    }
    if bn == 0.0 {
        // degenerate (λ with multiplicity): fall back to a unit basis vector
        return [1.0, 0.0, 0.0];
    }
    // Eigen normalises the complex eigenvector to unit norm; we take the real
    // part of that normalised vector.
    let scale = 1.0 / bn.sqrt();
    let mut v = [best[0].re * scale, best[1].re * scale, best[2].re * scale];
    if lam.im == 0.0 {
        // make deterministic sign (Eigen's sign is arbitrary; conic is homogeneous)
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 0.0 {
            v = [v[0] / n, v[1] / n, v[2] / n];
        }
    }
    v
}

/// Eigenvector of the smallest eigenvalue of a symmetric 4x4 matrix
/// (cyclic Jacobi in f64). Used for the least-squares circle fit.
pub fn min_eigenvector_sym4(s: &[[f64; 4]; 4]) -> [f64; 4] {
    let mut a = *s;
    let mut v = [[0.0f64; 4]; 4];
    for i in 0..4 {
        v[i][i] = 1.0;
    }
    for _sweep in 0..50 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in p + 1..4 {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..4 {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut imin = 0;
    for i in 1..4 {
        if a[i][i] < a[imin][imin] {
            imin = i;
        }
    }
    [v[0][imin], v[1][imin], v[2][imin], v[3][imin]]
}
