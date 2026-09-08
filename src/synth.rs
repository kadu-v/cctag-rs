//! Synthetic CCTag renderer (mirrors `markersToPrint/generators/generate.py`):
//! a black disc of radius `R` with alternating white/black rings at
//! `R / ratio` for each radius ratio of the bank entry, supersampled.

use crate::image::GrayImage;

/// Render marker `ratios` (descending radius ratios, e.g. a bank row) centred at
/// `(cx, cy)` with outer radius `r` into `img` (white background must be set by
/// the caller). `ss` is the supersampling factor per axis.
pub fn draw_marker(img: &mut GrayImage, cx: f32, cy: f32, r: f32, ratios: &[f32], ss: usize) {
    // ring radii from outermost: R (black), R/ratio[last] (white), ... alternating
    let mut radii: Vec<f32> = vec![r];
    for &q in ratios.iter().rev() {
        radii.push(r / q);
    }
    let x0 = ((cx - r - 2.0).floor().max(0.0)) as usize;
    let y0 = ((cy - r - 2.0).floor().max(0.0)) as usize;
    let x1 = ((cx + r + 2.0).ceil() as usize).min(img.w.saturating_sub(1));
    let y1 = ((cy + r + 2.0).ceil() as usize).min(img.h.saturating_sub(1));
    let inv = 1.0 / ss as f32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let mut acc = 0.0f32;
            for sy in 0..ss {
                for sx in 0..ss {
                    let px = x as f32 + (sx as f32 + 0.5) * inv;
                    let py = y as f32 + (sy as f32 + 0.5) * inv;
                    let d = ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt();
                    // count rings with radius > d: odd -> black
                    let k = radii.iter().filter(|&&rr| d < rr).count();
                    acc += if k % 2 == 1 { 0.0 } else { 255.0 };
                }
            }
            let v = acc / (ss * ss) as f32;
            let cur = img.at(x, y) as f32;
            // composite: marker paints over background
            let inside =
                ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt() <= r + 1.0;
            if inside {
                img.set(x, y, v.round().clamp(0.0, 255.0) as u8);
            } else {
                img.set(x, y, cur as u8);
            }
        }
    }
}

/// Separable Gaussian blur (odd kernel from `sigma`, replicate border), used to
/// soften synthetic edges like a printed/photographed marker.
pub fn gaussian_blur(img: &GrayImage, sigma: f32) -> GrayImage {
    if sigma <= 0.0 {
        return img.clone();
    }
    let half = (3.0 * sigma).ceil() as i32;
    let k: Vec<f32> = (-half..=half)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / sum).collect();
    let (w, h) = (img.w, img.h);
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x as i32 + i as i32 - half).clamp(0, w as i32 - 1) as usize;
                acc += img.data[y * w + xx] as f32 * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let yy = (y as i32 + i as i32 - half).clamp(0, h as i32 - 1) as usize;
                acc += tmp[yy * w + x] * kv;
            }
            out.data[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Deterministic additive Gaussian noise (Box–Muller on a PCG32 stream).
pub fn add_noise(img: &mut GrayImage, sigma: f32, seed: u64) {
    if sigma <= 0.0 {
        return;
    }
    let mut rng = crate::robust::Pcg32::with_stream(seed, 42);
    let mut i = 0;
    while i < img.data.len() {
        let u1 = (rng.next_u32() as f64 + 1.0) / 4294967297.0;
        let u2 = rng.next_u32() as f64 / 4294967296.0;
        let r = (-2.0 * u1.ln()).sqrt();
        let z0 = r * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = r * (2.0 * std::f64::consts::PI * u2).sin();
        for z in [z0, z1] {
            if i < img.data.len() {
                let v = img.data[i] as f64 + z * sigma as f64;
                img.data[i] = v.round().clamp(0.0, 255.0) as u8;
                i += 1;
            }
        }
    }
}

/// A test scene: markers placed on a grid over a white background.
pub struct SceneMarker {
    pub id: usize,
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
}

/// Render `markers` (bank rows by id) into a `w x h` image with optional blur/noise.
pub fn render_scene(
    w: usize,
    h: usize,
    bank: &crate::bank::Bank,
    markers: &[SceneMarker],
    blur_sigma: f32,
    noise_sigma: f32,
    seed: u64,
) -> GrayImage {
    let mut img = GrayImage::filled(w, h, 255);
    for m in markers {
        draw_marker(&mut img, m.cx, m.cy, m.radius, &bank.markers()[m.id], 4);
    }
    let mut img = gaussian_blur(&img, blur_sigma);
    add_noise(&mut img, noise_sigma, seed);
    img
}
