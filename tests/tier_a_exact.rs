//! Tier A: bit-exact comparison of the front-end and voting stages against the
//! `ref-exact` C++ dumps (see tools/cpp-ref).
#![cfg(all(feature = "png", feature = "refdata"))]

mod common;

use cctag::edge::{EdgePointCollection, from_canny::edges_points_from_canny};
use cctag::params::Params;
use cctag::pyramid::ImagePyramid;
use cctag::refdata::{read_i16_plane, read_i32_records, read_u8_plane};
use cctag::vote::linking::{VisitedMap, children_of, edge_linking};
use cctag::vote::vote::{sort_seeds, vote};
use std::path::Path;

fn diff_count<T: PartialEq>(a: &[T], b: &[T]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            n += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    (n + a.len().abs_diff(b.len()), first)
}

fn check_plane_u8(
    name: &str,
    ours: &cctag::image::Plane<u8>,
    dir: &Path,
    file: &str,
    failures: &mut Vec<String>,
) {
    let theirs = read_u8_plane(&dir.join(file)).expect(file);
    assert_eq!((ours.w, ours.h), (theirs.w, theirs.h), "{name}: dims");
    let (n, first) = diff_count(&ours.data, &theirs.data);
    if n > 0 {
        let f = first.unwrap();
        failures.push(format!(
            "{name}: {n} differing pixels, first at ({}, {}) ours={} theirs={}",
            f % ours.w,
            f / ours.w,
            ours.data[f],
            theirs.data[f]
        ));
    }
}

fn check_plane_i16(
    name: &str,
    ours: &cctag::image::Plane<i16>,
    dir: &Path,
    file: &str,
    failures: &mut Vec<String>,
) {
    let theirs = read_i16_plane(&dir.join(file)).expect(file);
    assert_eq!((ours.w, ours.h), (theirs.w, theirs.h), "{name}: dims");
    let (n, first) = diff_count(&ours.data, &theirs.data);
    if n > 0 {
        let f = first.unwrap();
        let mut hist = std::collections::BTreeMap::new();
        for (a, b) in ours.data.iter().zip(theirs.data.iter()) {
            let d = *a as i32 - *b as i32;
            if d != 0 {
                *hist.entry(d).or_insert(0usize) += 1;
            }
        }
        // The reference is a direct 81-tap f64 correlation; our separable f64
        // evaluation rounds differently only at near-.5 ties: allow |d| <= 1 on
        // at most 1e-5 of the samples. Everything downstream must still be exact.
        let max_abs = hist.keys().map(|d| d.abs()).max().unwrap_or(0);
        let budget = (ours.data.len() as f64 * 1e-5).ceil() as usize;
        if max_abs > 1 || n > budget {
            failures.push(format!(
                "{name}: {n} differing samples, first at ({}, {}) ours={} theirs={} hist={hist:?}",
                f % ours.w,
                f / ours.w,
                ours.data[f],
                theirs.data[f]
            ));
        } else {
            eprintln!(
                "note {name}: {n} samples differ by +-1 (rounding ties), within budget {budget}"
            );
        }
    }
}

#[test]
fn tier_a_front_end_and_vote() {
    let cases = common::ref_cases();
    if cases.is_empty() {
        common::skip_msg("tier_a");
        return;
    }
    let mut failures = Vec::new();
    for (name, img, dir) in cases
        .iter()
        .map(|(n, i, d)| (n.as_str(), i.as_path(), d.as_path()))
    {
        let gray = common::load_gray(img);
        let params = Params::new(3);
        check_plane_u8(
            &format!("{name}/gray"),
            &gray,
            dir,
            "gray.bin",
            &mut failures,
        );
        let mut pyr = ImagePyramid::new();
        pyr.build(&gray, &params);
        for i in 0..params.number_of_processed_multires_layers {
            let lvl = pyr.level(i);
            check_plane_u8(
                &format!("{name}/L{i}/src"),
                &lvl.src,
                dir,
                &format!("L{i}_src.bin"),
                &mut failures,
            );
            check_plane_i16(
                &format!("{name}/L{i}/dx"),
                &lvl.dx,
                dir,
                &format!("L{i}_dx.bin"),
                &mut failures,
            );
            check_plane_i16(
                &format!("{name}/L{i}/dy"),
                &lvl.dy,
                dir,
                &format!("L{i}_dy.bin"),
                &mut failures,
            );
            check_plane_u8(
                &format!("{name}/L{i}/edges"),
                &lvl.edges,
                dir,
                &format!("L{i}_edges.bin"),
                &mut failures,
            );

            // edge points, links, voters, seeds
            let mut coll = EdgePointCollection::new(lvl.src.w, lvl.src.h, gray.w, gray.h);
            edges_points_from_canny(&mut coll, &lvl.edges, &lvl.dx, &lvl.dy);
            let mut seeds = vote(&mut coll, &params);
            sort_seeds(&coll, &mut seeds);

            let pts = read_i32_records(&dir.join(format!("L{i}_points.bin")), 4).unwrap();
            let ours: Vec<[i32; 4]> = coll
                .points
                .iter()
                .map(|p| [p.x as i32, p.y as i32, p.dx as i32, p.dy as i32])
                .collect();
            let theirs: Vec<[i32; 4]> = pts.iter().map(|r| [r[0], r[1], r[2], r[3]]).collect();
            let (n, first) = diff_count(&ours, &theirs);
            if n > 0 {
                failures.push(format!(
                    "{name}/L{i}/points: {} vs {} points, {n} diffs, first {:?}",
                    ours.len(),
                    theirs.len(),
                    first.map(|f| (ours.get(f), theirs.get(f)))
                ));
                continue; // downstream comparisons meaningless
            }
            let links = read_i32_records(&dir.join(format!("L{i}_links.bin")), 2).unwrap();
            let theirs: Vec<[i32; 2]> = links.iter().map(|r| [r[0], r[1]]).collect();
            let (n, first) = diff_count(&coll.links, &theirs);
            if n > 0 {
                let f = first.unwrap();
                failures.push(format!(
                    "{name}/L{i}/links: {n} diffs, first at point {f} {:?} ours={:?} theirs={:?}",
                    coll.points[f], coll.links[f], theirs[f]
                ));
            }
            let voff: Vec<i32> = read_i32_records(&dir.join(format!("L{i}_voters_off.bin")), 1)
                .unwrap()
                .into_iter()
                .map(|r| r[0])
                .collect();
            let ours_off: Vec<i32> = coll.voters_off.iter().map(|&v| v as i32).collect();
            let (n, first) = diff_count(&ours_off, &voff);
            if n > 0 {
                failures.push(format!(
                    "{name}/L{i}/voters_off: {n} diffs, first at {first:?}"
                ));
            }
            let vl: Vec<i32> = read_i32_records(&dir.join(format!("L{i}_voters.bin")), 1)
                .unwrap()
                .into_iter()
                .map(|r| r[0])
                .collect();
            let (n, first) = diff_count(&coll.voters, &vl);
            if n > 0 {
                failures.push(format!("{name}/L{i}/voters: {n} diffs, first at {first:?}"));
            }
            let fl: Vec<i32> = read_i32_records(&dir.join(format!("L{i}_flow_length.bin")), 1)
                .unwrap()
                .into_iter()
                .map(|r| r[0])
                .collect();
            let ours_fl: Vec<i32> = coll
                .flow_length
                .iter()
                .map(|v| v.to_bits() as i32)
                .collect();
            let (n, first) = diff_count(&ours_fl, &fl);
            if n > 0 {
                let f = first.unwrap();
                failures.push(format!(
                    "{name}/L{i}/flow_length: {n} diffs, first at {f}: ours={} theirs={}",
                    coll.flow_length[f],
                    f32::from_bits(fl[f] as u32)
                ));
            }
            let sd = read_i32_records(&dir.join(format!("L{i}_seeds.bin")), 2).unwrap();
            let theirs: Vec<[i32; 2]> = sd.iter().map(|r| [r[0], r[1]]).collect();
            let ours: Vec<[i32; 2]> = seeds
                .iter()
                .map(|&s| [s, coll.is_max[s as usize]])
                .collect();
            let (n, first) = diff_count(&ours, &theirs);
            if n > 0 {
                failures.push(format!(
                    "{name}/L{i}/seeds: {} vs {} seeds, {n} diffs, first at {first:?}",
                    ours.len(),
                    theirs.len()
                ));
                continue;
            }

            // loop-1 replica: segments + children
            let flat: Vec<i32> = read_i32_records(&dir.join(format!("L{i}_loop1.bin")), 1)
                .unwrap()
                .into_iter()
                .map(|r| r[0])
                .collect();
            let mut theirs_cands: Vec<(i32, f32, Vec<i32>, Vec<i32>)> = Vec::new();
            let mut k = 0;
            while k < flat.len() {
                let seed = flat[k];
                let avg = f32::from_bits(flat[k + 1] as u32);
                let ns = flat[k + 2] as usize;
                let nc = flat[k + 3] as usize;
                let seg = flat[k + 4..k + 4 + ns].to_vec();
                let ch = flat[k + 4 + ns..k + 4 + ns + nc].to_vec();
                theirs_cands.push((seed, avg, seg, ch));
                k += 4 + ns + nc;
            }
            let n_max = (lvl.src.h / 2).max(params.maximum_nb_seeds);
            let n_seeds = seeds.len().min(n_max);
            let mut visited = VisitedMap::new(coll.map_w, coll.map_h);
            let mut ours_cands: Vec<(i32, f32, Vec<i32>, Vec<i32>)> = Vec::new();
            for &seed in &seeds[..n_seeds] {
                if coll.test_processed_in(seed) {
                    continue;
                }
                let link = edge_linking(
                    &coll,
                    seed,
                    params.window_size_on_inner_elliptic_segment,
                    params.average_vote_min,
                    &mut visited,
                );
                for &m in &link.marks {
                    coll.set_processed_in(m, true);
                }
                let mut nr = 0i32;
                let mut nv = 0i32;
                for &p in &link.segment {
                    let vs = coll.voters_size(p) as i32;
                    nr += vs;
                    if vs > 0 {
                        nv += 1;
                    }
                }
                let avg = (nr * nr) as f32 / nv as f32;
                let ch = children_of(&coll, &link.segment);
                ours_cands.push((seed, avg, link.segment.iter().copied().collect(), ch));
            }
            if ours_cands.len() != theirs_cands.len() {
                failures.push(format!(
                    "{name}/L{i}/loop1: {} vs {} candidates",
                    ours_cands.len(),
                    theirs_cands.len()
                ));
            }
            for (ci, (o, t)) in ours_cands.iter().zip(theirs_cands.iter()).enumerate() {
                if o.0 != t.0 || o.1.to_bits() != t.1.to_bits() || o.2 != t.2 || o.3 != t.3 {
                    failures.push(format!("{name}/L{i}/loop1[{ci}]: seed {} vs {}, avg {} vs {}, seg {} vs {} pts, children {} vs {}", o.0, t.0, o.1, t.1, o.2.len(), t.2.len(), o.3.len(), t.3.len()));
                    break;
                }
            }
        }
    }
    if !failures.is_empty() {
        panic!("Tier A mismatches:\n{}", failures.join("\n"));
    }
}
