//! Minimal detection sample:
//!   cargo run --release --features png --example detect -- <image.png> [--parity] [--timings] [--iters N] [--threads N] [--filelog out.xml]
//! Prints one line per marker: `x y id status quality level`. With `--filelog`
//! also writes the upstream `regression` tool's FileLog XML for the image.

use cctag::{Detector, ExecMode};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path: Option<String> = None;
    let mut mode = ExecMode::Fast;
    let mut timings = false;
    let mut iters = 1usize;
    let mut threads: Option<usize> = None;
    let mut n_crowns = 3usize;
    let mut filelog: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--parity" => mode = ExecMode::Parity,
            "--timings" => timings = true,
            "--iters" => {
                i += 1;
                iters = args[i].parse().expect("iters");
            }
            "--threads" => {
                i += 1;
                threads = Some(args[i].parse().expect("threads"));
            }
            "--filelog" => {
                i += 1;
                filelog = Some(args[i].clone());
            }
            "-n" | "--nbrings" => {
                i += 1;
                n_crowns = args[i].parse().expect("nbrings");
            }
            a => path = Some(a.to_string()),
        }
        i += 1;
    }
    let path =
        path.expect("usage: detect <image.png> [--parity] [--timings] [--iters N] [--threads N]");
    #[cfg(feature = "parallel")]
    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("rayon pool");
    }
    let _ = threads;
    let gray = cctag::image::gray::load_gray(&path).expect("load image");
    let mut det = Detector::with_crowns(n_crowns, mode).expect("bank");
    let mut timer = cctag::timing::StageTimer::new();
    let mut markers = Vec::new();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        timer.clear();
        let t0 = Instant::now();
        markers = det.detect_timed(&gray, if timings { Some(&mut timer) } else { None });
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("#frame 0");
    println!("Detected {} candidates", markers.len());
    for m in &markers {
        println!(
            "{} {} {} {} {} {}",
            m.x(),
            m.y(),
            m.id,
            m.status.code(),
            m.quality,
            m.pyramid_level
        );
    }
    let reliable = markers.iter().filter(|m| m.status.is_reliable()).count();
    println!("{reliable} markers detected and identified");
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    println!(
        "time: median {med:.3} ms, min {:.3} ms, max {:.3} ms over {} iters ({:?})",
        times[0],
        times[times.len() - 1],
        iters,
        mode
    );
    if cfg!(debug_assertions) {
        println!("WARNING: debug build, timings are not meaningful");
    }
    if timings {
        print!("{}", timer.report());
    }
    if let Some(out) = filelog {
        let abs = std::fs::canonicalize(&path)
            .map(|p| p.display().to_string())
            .unwrap_or(path.clone());
        let frame = cctag::filelog::FrameLog {
            frame: 0,
            elapsed_seconds: (med / 1000.0) as f32,
            markers: &markers,
        };
        std::fs::write(
            &out,
            cctag::filelog::write_filelog(&abs, &det.params, &[frame]),
        )
        .expect("write filelog");
    }
}
