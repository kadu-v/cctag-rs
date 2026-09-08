//! Reader/writer for the stage-dump format shared with the C++ parity harness
//! (`tools/cpp-ref/patches/0005-parity-dump.patch`).
//!
//! Layout: little-endian. Planes: 16-byte header `magic(4) dtype(4) w(4) h(4)`
//! followed by `w*h` elements. Record files: `u32 count` then fixed records.

use std::io::{self, Read, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"CCTD";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    U8 = 1,
    I16 = 2,
    I32 = 3,
    F32 = 4,
}

pub fn write_plane<T: bytemuck_like::Pod>(
    path: &Path,
    dtype: DType,
    w: usize,
    h: usize,
    data: &[T],
) -> io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&(dtype as u32).to_le_bytes())?;
    f.write_all(&(w as u32).to_le_bytes())?;
    f.write_all(&(h as u32).to_le_bytes())?;
    f.write_all(bytemuck_like::bytes_of(data))?;
    Ok(())
}

pub fn read_plane_raw(path: &Path) -> io::Result<(DType, usize, usize, Vec<u8>)> {
    let mut f = std::fs::File::open(path)?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr)?;
    if &hdr[..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let dt = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    let w = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let dtype = match dt {
        1 => DType::U8,
        2 => DType::I16,
        3 => DType::I32,
        4 => DType::F32,
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad dtype")),
    };
    let mut data = Vec::new();
    f.read_to_end(&mut data)?;
    Ok((dtype, w, h, data))
}

pub fn read_u8_plane(path: &Path) -> io::Result<crate::image::Plane<u8>> {
    let (dt, w, h, data) = read_plane_raw(path)?;
    if dt != DType::U8 || data.len() != w * h {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a u8 plane"));
    }
    Ok(crate::image::Plane::from_vec(w, h, data))
}

pub fn read_i16_plane(path: &Path) -> io::Result<crate::image::Plane<i16>> {
    let (dt, w, h, data) = read_plane_raw(path)?;
    if dt != DType::I16 || data.len() != w * h * 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an i16 plane",
        ));
    }
    let v: Vec<i16> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(crate::image::Plane::from_vec(w, h, v))
}

/// Read a record file of `i32`s: `u32 count` + `count * n_fields` i32.
pub fn read_i32_records(path: &Path, n_fields: usize) -> io::Result<Vec<Vec<i32>>> {
    let data = std::fs::read(path)?;
    if data.len() < 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short file"));
    }
    let count = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    let body = &data[4..];
    if body.len() != count * n_fields * 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("record size mismatch in {}", path.display()),
        ));
    }
    Ok(body
        .chunks_exact(n_fields * 4)
        .map(|rec| {
            rec.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect())
}

pub fn write_i32_records(path: &Path, recs: &[Vec<i32>]) -> io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(recs.len() as u32).to_le_bytes())?;
    for r in recs {
        for v in r {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Minimal POD-to-bytes helper (avoids a dependency).
pub mod bytemuck_like {
    pub trait Pod: Copy {}
    impl Pod for u8 {}
    impl Pod for i16 {}
    impl Pod for i32 {}
    impl Pod for f32 {}
    pub fn bytes_of<T: Pod>(v: &[T]) -> &[u8] {
        // SAFETY: T is a plain-old-data numeric type; little-endian host assumed.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
}
