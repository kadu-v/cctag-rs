//! Synthetic markers: every 3-crown id must be detected and identified at the
//! expected position, in both execution modes.
#![cfg(feature = "synth")]

use cctag::synth::{SceneMarker, render_scene};
use cctag::{Bank, Detector, ExecMode, Params};

fn scene(
    ids: &[usize],
    radius: f32,
    blur: f32,
    noise: f32,
) -> (cctag::image::GrayImage, Vec<SceneMarker>) {
    let bank = Bank::builtin(3).unwrap();
    let (w, h) = (640usize, 480usize);
    let cols = 4;
    let mut ms = Vec::new();
    for (k, &id) in ids.iter().enumerate() {
        let cx = 80.0 + (k % cols) as f32 * 160.0 + (k as f32 * 3.7) % 11.0;
        let cy = 80.0 + (k / cols) as f32 * 160.0 + (k as f32 * 5.3) % 13.0;
        ms.push(SceneMarker { id, cx, cy, radius });
    }
    (render_scene(w, h, &bank, &ms, blur, noise, 7), ms)
}

fn check(
    mode: ExecMode,
    ids: &[usize],
    radius: f32,
    blur: f32,
    noise: f32,
    tol: f32,
) -> Vec<String> {
    let (img, ms) = scene(ids, radius, blur, noise);
    let mut det = Detector::new(Params::new(3), Bank::builtin(3).unwrap(), mode);
    let found = det.detect(&img);
    let mut errs = Vec::new();
    for m in &ms {
        let hit = found
            .iter()
            .filter(|f| f.status.is_reliable())
            .find(|f| ((f.x() - m.cx).powi(2) + (f.y() - m.cy).powi(2)).sqrt() < tol);
        match hit {
            None => errs.push(format!("{mode:?} r={radius} blur={blur} noise={noise}: id {} at ({:.1},{:.1}) not detected", m.id, m.cx, m.cy)),
            Some(f) if f.id as usize != m.id => errs.push(format!("{mode:?} r={radius}: id {} misread as {}", m.id, f.id)),
            Some(_) => {}
        }
    }
    let n_rel = found.iter().filter(|f| f.status.is_reliable()).count();
    if n_rel > ms.len() {
        errs.push(format!(
            "{mode:?} r={radius}: {} reliable markers for {} drawn",
            n_rel,
            ms.len()
        ));
    }
    errs
}

#[test]
fn all_32_ids_detected_clean() {
    let mut errs = Vec::new();
    for chunk in (0..32usize).collect::<Vec<_>>().chunks(12) {
        for mode in [ExecMode::Parity, ExecMode::Fast] {
            errs.extend(check(mode, chunk, 60.0, 1.0, 0.0, 2.0));
        }
    }
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

#[test]
fn ids_detected_with_noise_and_scales() {
    let mut errs = Vec::new();
    for (r, blur, noise) in [(35.0, 0.8, 5.0), (60.0, 1.5, 10.0), (70.0, 0.5, 3.0)] {
        errs.extend(check(
            ExecMode::Fast,
            &[0, 5, 13, 21, 27, 31],
            r,
            blur,
            noise,
            2.5,
        ));
    }
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}
