//! Colour to gray conversion matching OpenCV `cvtColor(BGR2GRAY)` for 8-bit input.
//!
//! OpenCV uses fixed-point coefficients with 14 fractional bits:
//! `gray = (R*4899 + G*9617 + B*1868 + (1<<13)) >> 14` (R=0.299, G=0.587, B=0.114).

use super::GrayImage;

const R_COEF: u32 = 4899;
const G_COEF: u32 = 9617;
const B_COEF: u32 = 1868;
const SHIFT: u32 = 14;

#[inline(always)]
fn to_gray(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * R_COEF + g as u32 * G_COEF + b as u32 * B_COEF + (1 << (SHIFT - 1))) >> SHIFT)
        as u8
}

/// Interleaved RGB (as decoded from PNG) to gray.
pub fn rgb_to_gray(rgb: &[u8], w: usize, h: usize) -> GrayImage {
    assert_eq!(rgb.len(), w * h * 3);
    let data = rgb
        .chunks_exact(3)
        .map(|p| to_gray(p[0], p[1], p[2]))
        .collect();
    GrayImage { w, h, data }
}

/// Interleaved RGBA to gray (alpha ignored).
pub fn rgba_to_gray(rgba: &[u8], w: usize, h: usize) -> GrayImage {
    assert_eq!(rgba.len(), w * h * 4);
    let data = rgba
        .chunks_exact(4)
        .map(|p| to_gray(p[0], p[1], p[2]))
        .collect();
    GrayImage { w, h, data }
}

/// Load a PNG (or any format supported by the `image` crate) as gray, the same
/// way the upstream apps do (`imread` colour then `cvtColor(BGR2GRAY)`).
#[cfg(feature = "png")]
pub fn load_gray(path: impl AsRef<std::path::Path>) -> Result<GrayImage, image::ImageError> {
    let img = image::open(path)?;
    let w = img.width() as usize;
    let h = img.height() as usize;
    Ok(match img {
        image::DynamicImage::ImageLuma8(g) => GrayImage {
            w,
            h,
            data: g.into_raw(),
        },
        image::DynamicImage::ImageRgb8(c) => rgb_to_gray(c.as_raw(), w, h),
        image::DynamicImage::ImageRgba8(c) => rgba_to_gray(c.as_raw(), w, h),
        other => rgb_to_gray(other.to_rgb8().as_raw(), w, h),
    })
}

/// Save a gray plane as PNG.
#[cfg(feature = "png")]
pub fn save_gray(
    img: &GrayImage,
    path: impl AsRef<std::path::Path>,
) -> Result<(), image::ImageError> {
    image::GrayImage::from_raw(img.w as u32, img.h as u32, img.data.clone())
        .expect("plane dims")
        .save(path)
}
