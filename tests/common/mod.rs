//! Shared helpers for the parity tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Reference dump directories that exist (`testdata/ref/<name>/`), paired with
/// the source image path.
pub fn ref_cases() -> Vec<(String, PathBuf, PathBuf)> {
    let mut out = Vec::new();
    let refdir = root().join("testdata/ref");
    let Ok(rd) = std::fs::read_dir(&refdir) else {
        return out;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let dir = e.path();
        if !dir.join("gray.bin").exists() {
            continue;
        }
        let img = [
            root().join(format!("CCTag/sample/{name}.png")),
            root().join(format!("testdata/synth/{name}.png")),
        ]
        .into_iter()
        .find(|p| p.exists());
        if let Some(img) = img {
            out.push((name, img, dir));
        }
    }
    out.sort();
    out
}

pub fn skip_msg(what: &str) {
    eprintln!(
        "SKIP {what}: no reference dumps in testdata/ref (run tools/cpp-ref/build.sh ref-exact && tools/cpp-ref/gen_refs.sh)"
    );
}

pub fn load_gray(p: &Path) -> cctag::image::GrayImage {
    cctag::image::gray::load_gray(p).expect("load png")
}
