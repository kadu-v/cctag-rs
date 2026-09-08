//! PCG32 (`pcg_engines::setseq_xsh_rr_64_32`) as used by `rand_5_k`
//! (Statistic.cpp:16-29): seeded with `pcg32 rng(271828)`.

pub const PCG_MULT: u64 = 6364136223846793005;
pub const PCG_DEFAULT_INC: u64 = 1442695040888963407;
pub const UPSTREAM_SEED: u64 = 271828;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
    /// Number of `bounded()` draws so far (parity diagnostics).
    pub draws: u64,
}

impl Pcg32 {
    /// `pcg32(seed)` with the default stream (increment).
    pub fn new(seed: u64) -> Self {
        Self::with_increment(seed, PCG_DEFAULT_INC)
    }

    /// `pcg32(seed, stream)`: increment = `(stream << 1) | 1`.
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        Self::with_increment(seed, (stream << 1) | 1)
    }

    fn with_increment(seed: u64, inc: u64) -> Self {
        // state_ = bump(seed + increment)
        let mut r = Pcg32 {
            state: seed.wrapping_add(inc),
            inc,
            draws: 0,
        };
        r.state = r.bump(r.state);
        r
    }

    /// The generator upstream uses (`static thread_local pcg32 rng(271828)`).
    pub fn upstream() -> Self {
        Self::new(UPSTREAM_SEED)
    }

    #[inline(always)]
    fn bump(&self, s: u64) -> u64 {
        s.wrapping_mul(PCG_MULT).wrapping_add(self.inc)
    }

    /// `operator()`: XSH-RR output of the *old* state, then advance.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = self.bump(old);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// `operator()(bound)` = `pcg_extras::bounded_rand`: rejection sampling.
    #[inline]
    pub fn bounded(&mut self, bound: u32) -> u32 {
        let threshold = (0u32.wrapping_sub(bound)) % bound;
        self.draws += 1;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % bound;
            }
        }
    }

    /// `rand_5_k(perm, N)`: 5 distinct indices in `0..N`.
    pub fn rand_5_k(&mut self, perm: &mut [i32; 5], n: usize) {
        debug_assert!(n >= 5, "rand_5_k needs at least 5 candidates");
        let n = n as u32;
        for i in 0..5 {
            loop {
                let r = self.bounded(n) as i32;
                if !perm[..i].contains(&r) {
                    perm[i] = r;
                    break;
                }
            }
        }
    }
}
