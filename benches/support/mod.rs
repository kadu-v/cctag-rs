pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        let n = std::env::var("CCTAG_BENCH_THREADS")
            .map_or(1, |s| s.parse().expect("CCTAG_BENCH_THREADS"));
        assert!(n > 0);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .unwrap();
        n
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}
