//! Generate the curated synthetic test set (PNG) under testdata/synth/.
//!   cargo run --release --features png,synth --example synth_gen
use cctag::Bank;
use cctag::synth::{SceneMarker, render_scene};

fn main() {
    let root = env!("CARGO_MANIFEST_DIR");
    let out = format!("{root}/testdata/synth");
    std::fs::create_dir_all(&out).unwrap();
    let bank = Bank::builtin(3).unwrap();
    let (w, h) = (640usize, 480usize);
    // (name, ids, radius, blur, noise)
    let scenes: Vec<(&str, Vec<usize>, f32, f32, f32)> = vec![
        ("s01_clean_r60", (0..8).collect(), 60.0, 1.0, 0.0),
        ("s02_clean_r60_b", (8..16).collect(), 60.0, 1.0, 0.0),
        ("s03_clean_r60_c", (16..24).collect(), 60.0, 1.0, 0.0),
        ("s04_clean_r60_d", (24..32).collect(), 60.0, 1.0, 0.0),
        ("s05_small_r30", vec![1, 9, 17, 25, 30, 4], 30.0, 0.7, 2.0),
        ("s06_noisy_r50", vec![2, 10, 18, 26, 31, 7], 50.0, 1.2, 8.0),
        ("s07_blur_r70", vec![3, 11, 19, 27], 70.0, 2.5, 3.0),
        ("s08_mixed", vec![5, 13, 21, 29, 0, 15], 45.0, 1.0, 5.0),
    ];
    for (name, ids, radius, blur, noise) in scenes {
        let cols = 4;
        let ms: Vec<SceneMarker> = ids
            .iter()
            .enumerate()
            .map(|(k, &id)| SceneMarker {
                id,
                cx: 80.0 + (k % cols) as f32 * 160.0 + (k as f32 * 3.7) % 11.0,
                cy: 80.0 + (k / cols) as f32 * 160.0 + (k as f32 * 5.3) % 13.0,
                radius,
            })
            .collect();
        let img = render_scene(w, h, &bank, &ms, blur, noise, 11);
        let path = format!("{out}/{name}.png");
        cctag::image::gray::save_gray(&img, &path).unwrap();
        println!("wrote {path}");
    }
}
