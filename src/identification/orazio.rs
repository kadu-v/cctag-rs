//! `orazioDistanceRobust` (Identification.cpp:50-190) — ID reading.

use super::cut::ImageCut;
use super::select::boost_mean_variance;

/// `dis(sig, val, mub, muw, varSubS)`.
#[inline(always)]
fn dis(sig: f32, val: f32, mub: f32, muw: f32, var: f32) -> f32 {
    if val == -1.0 {
        let d = (sig - mub).max(0.0);
        d * d / (2.0 * var)
    } else {
        let d = (sig - muw).min(0.0);
        d * d / (2.0 * var)
    }
}

/// Per-cut vote: for each in-bounds cut, the best matching id and its score.
/// Returns `vScore`: for each bank id, the list of scores of cuts that voted for it
/// (cut order, deterministic).
pub fn orazio_distance_robust(rr_bank: &[Vec<f32>], cuts: &[ImageCut]) -> Vec<Vec<f32>> {
    let mut v_score: Vec<Vec<f32>> = vec![Vec::new(); rr_bank.len()];
    if cuts.is_empty() || rr_bank.is_empty() {
        return v_score;
    }
    let n = cuts[0].signal.len();
    let mut digit = vec![0.0f32; n];
    for cut in cuts {
        if cut.out_of_bounds {
            continue;
        }
        let sig = &cut.signal;
        let (median_sig, var_sig) = boost_mean_variance(&sig[30.min(sig.len())..]);
        let mut sum_inf = 0.0f32;
        let mut n_inf = 0usize;
        let mut sum_sup = 0.0f32;
        let mut n_sup = 0usize;
        let mut do_acc = false;
        for &v in sig {
            if !do_acc && v < median_sig {
                do_acc = true;
            }
            if do_acc {
                if v < median_sig {
                    sum_inf += v;
                    n_inf += 1;
                } else {
                    sum_sup += v;
                    n_sup += 1;
                }
            }
        }
        let muw = sum_sup / n_sup as f32;
        let mub = sum_inf / n_inf as f32;
        let step_x = (cut.end_sig - cut.begin_sig) / (n as f32 - 1.0);

        // std::map<float, MarkerID> sortedId — keyed by score, last insert wins on
        // equal keys. We keep (score, id) pairs with that replacement rule.
        let mut sorted: Vec<(f32, i32)> = Vec::with_capacity(rr_bank.len());
        for (idc, ratios) in rr_bank.iter().enumerate() {
            let mut x = cut.begin_sig;
            for d in digit.iter_mut() {
                let mut ldum = 0i64;
                for &j in ratios {
                    if 1.0 / j <= x {
                        ldum += 1;
                    }
                }
                *d = -((ldum % 2) as f32) * 2.0 + 1.0;
                x += step_x;
            }
            let mut distance = 0.0f32;
            for i in 0..n {
                distance += dis(sig[i], digit[i], mub, muw, var_sig);
            }
            let v = (-distance).exp();
            if let Some(e) = sorted.iter_mut().find(|e| e.0 == v) {
                e.1 = idc as i32;
            } else {
                sorted.push((v, idc as i32));
            }
        }
        // best = largest key
        let mut best = sorted[0];
        for e in &sorted[1..] {
            if e.0 > best.0 {
                best = *e;
            }
        }
        v_score[best.1 as usize].push(best.0);
    }
    v_score
}
