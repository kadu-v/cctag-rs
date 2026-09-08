//! PCG32 parity with the upstream `pcg32 rng(271828)` (values dumped by
//! `cctag_ref --selftest-rng`), plus a cross-check against `rand_pcg`.

use cctag::robust::Pcg32;

const RAW: [u32; 16] = [
    3450899053, 2037256299, 1972518263, 3697269208, 155652559, 1413146261, 910878575, 1366771542,
    3834716864, 2046911032, 2717343650, 488262595, 901038725, 1203219447, 104434048, 1938827001,
];

#[test]
fn raw_sequence_matches_upstream() {
    let mut r = Pcg32::upstream();
    for (i, &v) in RAW.iter().enumerate() {
        assert_eq!(r.next_u32(), v, "raw[{i}]");
    }
}

#[test]
fn bounded_matches_upstream() {
    let cases: [(u32, [u32; 16]); 5] = [
        (7, [0, 1, 2, 3, 6, 2, 5, 3, 6, 5, 0, 2, 6, 3, 5, 6]),
        (
            60,
            [
                13, 39, 23, 28, 19, 41, 35, 42, 44, 52, 50, 55, 5, 27, 28, 21,
            ],
        ),
        (
            150,
            [
                103, 99, 113, 58, 109, 11, 125, 42, 14, 82, 50, 145, 125, 147, 148, 51,
            ],
        ),
        (
            1000,
            [
                53, 299, 263, 208, 559, 261, 575, 542, 864, 32, 650, 595, 725, 447, 48, 1,
            ],
        ),
        (
            24963,
            [
                13933, 906, 16892, 24241, 8254, 15794, 3668, 22329, 656, 19921, 21248, 11278,
                24203, 2847, 13819, 717,
            ],
        ),
    ];
    for (n, exp) in cases {
        let mut r = Pcg32::upstream();
        for (i, &v) in exp.iter().enumerate() {
            assert_eq!(r.bounded(n), v, "bounded({n})[{i}]");
        }
    }
}

#[test]
fn rand_5_k_matches_upstream() {
    let mut r = Pcg32::upstream();
    let exp: [[i32; 5]; 4] = [
        [13, 39, 23, 28, 19],
        [41, 35, 42, 44, 52],
        [50, 55, 5, 27, 28],
        [21, 17, 40, 6, 11],
    ];
    let mut perm = [0i32; 5];
    for (k, e) in exp.iter().enumerate() {
        r.rand_5_k(&mut perm, 60);
        assert_eq!(perm, *e, "rand_5_k call {k}");
    }
}

#[test]
fn matches_rand_pcg_crate() {
    use rand_core::RngCore;
    // rand_pcg::Pcg32::new(state, stream): increment = (stream << 1) | 1
    let mut theirs = rand_pcg::Pcg32::new(271828, cctag::robust::rng::PCG_DEFAULT_INC >> 1);
    let mut ours = Pcg32::upstream();
    for _ in 0..1000 {
        assert_eq!(ours.next_u32(), theirs.next_u32());
    }
}
