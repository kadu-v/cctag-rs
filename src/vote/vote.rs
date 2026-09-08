//! `vote()` (Vote.cpp:57-246).

use super::descent::gradient_direction_descent;
use crate::edge::{EdgeIdx, EdgePointCollection, NO_EDGE};
use crate::geometry::distance::distance_points_2d_i;
use crate::params::Params;

/// Phase A: before/after links for every point (pure per point).
pub fn compute_links(coll: &mut EdgePointCollection, dist_search: usize) {
    let n = coll.len();
    let mut links = std::mem::take(&mut coll.links);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let c: &EdgePointCollection = coll;
        links
            .par_iter_mut()
            .enumerate()
            .with_min_len(4096)
            .for_each(|(i, l)| {
                let i = i as EdgeIdx;
                l[0] = gradient_direction_descent(c, i, -1, dist_search);
                l[1] = gradient_direction_descent(c, i, 1, dist_search);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..n {
            let ii = i as EdgeIdx;
            links[i][0] = gradient_direction_descent(coll, ii, -1, dist_search);
            links[i][1] = gradient_direction_descent(coll, ii, 1, dist_search);
        }
    }
    let _ = n;
    coll.links = links;
}

/// Phase B for one point: field-line traversal (Vote.cpp:100-224).
/// Returns `(chosen, total_distance)` if a winner was found.
#[inline]
fn field_line(coll: &EdgePointCollection, p: EdgeIdx, params: &Params) -> Option<(EdgeIdx, f32)> {
    let pp = coll.point(p);
    let mut current = coll.before(p);
    if current == NO_EDGE {
        return None;
    }
    let mut chosen = NO_EDGE;
    let mut v_dist: [f32; 8] = [0.0; 8];
    let mut nd = 0usize;
    let mut total = 0.0f32;
    let ratio = params.ratio_voting;
    let angle_voting = params.angle_voting;

    let cos_diff = -pp.grad_dot(coll.point(current));
    if cos_diff >= angle_voting {
        let last = distance_points_2d_i(pp, coll.point(current));
        v_dist[nd] = last;
        nd += 1;
        total += last;
        let mut i = 1usize;
        while i < params.n_crowns {
            chosen = NO_EDGE;
            let mut target = coll.after(current);
            if target == NO_EDGE {
                break;
            }
            let cur_pt = coll.point(current);
            let cos_diff = -coll.point(target).grad_dot(cur_pt);
            if cos_diff >= angle_voting {
                let dist = distance_points_2d_i(coll.point(target), cur_pt);
                v_dist[nd] = dist;
                nd += 1;
                total += dist;
                let mut flag = true;
                if nd > 1 {
                    for a in 0..nd {
                        for b in a + 1..nd {
                            flag = (v_dist[a] <= v_dist[b] * ratio)
                                && (v_dist[b] <= v_dist[a] * ratio)
                                && flag;
                        }
                    }
                }
                if flag {
                    current = target;
                    target = coll.before(current);
                    if target == NO_EDGE {
                        break;
                    }
                    let cur_pt = coll.point(current);
                    let cos_diff = -coll.point(target).grad_dot(cur_pt);
                    if cos_diff >= angle_voting {
                        let dist = distance_points_2d_i(coll.point(target), cur_pt);
                        v_dist[nd] = dist;
                        nd += 1;
                        total += dist;
                        for a in 0..nd {
                            for b in a + 1..nd {
                                flag = (v_dist[a] <= v_dist[b] * ratio)
                                    && (v_dist[b] <= v_dist[a] * ratio)
                                    && flag;
                            }
                        }
                        if flag {
                            current = target;
                            chosen = current;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
            i += 1;
        }
    }
    if chosen != NO_EDGE {
        Some((chosen, total))
    } else {
        None
    }
}

/// Full vote: links, field lines, voters CSR, `flow_length`, `is_max`, seeds.
/// Seeds are returned in discovery order (upstream sorts them afterwards).
pub fn vote(coll: &mut EdgePointCollection, params: &Params) -> Vec<EdgeIdx> {
    if params.angle_voting != 0.0 {
        panic!("thrVotingAngle must be equal to 0 or edge points gradients have to be normalized.");
    }
    compute_links(coll, params.dist_search);
    let n = coll.len();

    // Phase B (pure): winner per point.
    let winners: Vec<Option<(EdgeIdx, f32)>> = {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let c: &EdgePointCollection = coll;
            (0..n)
                .into_par_iter()
                .with_min_len(4096)
                .map(|i| field_line(c, i as EdgeIdx, params))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n)
                .map(|i| field_line(coll, i as EdgeIdx, params))
                .collect()
        }
    };

    // Phase C (sequential, point order): votes, flow length running mean, seeds.
    let mut pairs: Vec<(EdgeIdx, EdgeIdx)> = Vec::with_capacity(n / 2);
    let mut counts: Vec<u32> = vec![0; n];
    let mut seeds: Vec<EdgeIdx> = Vec::with_capacity(n / 2);
    let min_votes = params.min_votes_to_select_candidate;
    for (i, w) in winners.iter().enumerate() {
        if let Some((chosen, total)) = *w {
            let c = chosen as usize;
            pairs.push((chosen, i as EdgeIdx));
            counts[c] += 1;
            let sz = counts[c] as f32;
            coll.flow_length[c] = (coll.flow_length[c] * (sz - 1.0) + total) / sz;
            if counts[c] as usize >= min_votes {
                if coll.is_max[c] == -1 {
                    seeds.push(chosen);
                }
                coll.is_max[c] = counts[c] as i32;
            }
        }
    }
    coll.create_voter_lists(&pairs);
    seeds
}

/// `std::sort(seeds, receivedMoreVoteThan)` — we use a *stable* sort by
/// descending vote count (ties keep discovery order); upstream's unstable sort
/// leaves tie order implementation-defined (parity reference is patched to
/// `stable_sort`).
pub fn sort_seeds(coll: &EdgePointCollection, seeds: &mut [EdgeIdx]) {
    if seeds.len() > 1 {
        seeds.sort_by(|a, b| coll.is_max[*b as usize].cmp(&coll.is_max[*a as usize]));
    }
}
