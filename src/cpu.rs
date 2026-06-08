//! CPU (host) training path: a rayon-parallel oblivious-tree histogram GBDT, selected with
//! `--device cpu`. Runs with no GPU/CUDA context. It produces the same model arrays
//! (`feat`/`thr`/`leafval`) as the GPU path, so `Booster::predict_cpu` scores it unchanged.
//!
//! Levers ported from the standalone CPU experiment: histogram subtraction, column-major
//! `apply`, a bandwidth-friendly thread cap, branchless vectorizable gain-eval, and
//! row-sampling (`--subsample` / GOSS). Both objectives are handled through per-row
//! gradient/hessian, so the same kernels serve L2 and logistic.

use rayon::prelude::*;

use crate::config::{Config, Objective};

/// Privatized per-node histograms `[node][feature][bin]` of (gsum, hsum). With `odd_only`,
/// accumulate only rows whose node index is odd, keyed by `node >> 1` (the parent index) --
/// the build half of histogram subtraction.
fn build_hist(
    rows: &[u32], nt: usize, index_nodes: usize, nf: usize, nbins: usize,
    node: &[u32], g: &[f32], h: &[f32], bins: &[u8], odd_only: bool,
) -> Vec<f32> {
    let m = rows.len();
    let hsz = index_nodes * nf * nbins;
    let chunk = m.div_ceil(nt);
    (0..nt)
        .into_par_iter()
        .map(|ti| {
            let lo = ti * chunk;
            let hi = ((ti + 1) * chunk).min(m);
            let mut loc = vec![0f32; hsz * 2];
            if lo >= hi {
                return loc;
            }
            for &ru in &rows[lo..hi] {
                let r = ru as usize;
                let nv = node[r];
                let idx_node = if odd_only {
                    if nv & 1 == 0 {
                        continue;
                    }
                    (nv >> 1) as usize
                } else {
                    nv as usize
                };
                let (gr, hr) = (g[r], h[r]);
                let off = idx_node * nf * nbins;
                let rb = &bins[r * nf..r * nf + nf];
                for f in 0..nf {
                    let idx = (off + f * nbins + rb[f] as usize) * 2;
                    loc[idx] += gr;
                    loc[idx + 1] += hr;
                }
            }
            loc
        })
        .reduce(
            || vec![0f32; hsz * 2],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += *y);
                a
            },
        )
}

/// Gradient-based one-side sampling: keep the `top` fraction of rows by |gradient|, sample
/// `other` of the rest, and amplify the sampled rows' gradient AND hessian by (1-top)/other
/// so both histogram sums stay unbiased. Fills `out` with the chosen row indices.
// Fixed salt folded into the GOSS sampling RNG.
const GOSS_SALT: u64 = 0x0042_4552_455A_4B41;

fn goss_sample(g: &mut [f32], h: &mut [f32], top: f64, other: f64, seed: u64, out: &mut Vec<u32>) {
    let n = g.len();
    let topn = ((top * n as f64) as usize).clamp(1, n);
    let mut absg: Vec<f32> = g.par_iter().map(|x| x.abs()).collect();
    let k = (topn - 1).min(n - 1);
    absg.select_nth_unstable_by(k, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let thresh = absg[k];
    let keep_prob = (other / (1.0 - top).max(1e-9)).clamp(0.0, 1.0);
    let fact = ((1.0 - top) / other.max(1e-9)) as f32;
    let cs = n.div_ceil(rayon::current_num_threads());
    let parts: Vec<Vec<u32>> = g
        .par_chunks_mut(cs)
        .zip(h.par_chunks_mut(cs))
        .enumerate()
        .map(|(ci, (gc, hc))| {
            let base = (ci * cs) as u64;
            let mut local = Vec::with_capacity(gc.len() / 2);
            for i in 0..gc.len() {
                let r = base + i as u64;
                hc[i] = 1.0;
                if gc[i].abs() >= thresh {
                    local.push(r as u32);
                } else {
                    let mut hh = (r ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ GOSS_SALT)
                        .wrapping_mul(0xD6E8_FEB8_6659_FD93);
                    hh ^= hh >> 32;
                    if ((hh >> 11) as f64 / (1u64 << 53) as f64) < keep_prob {
                        gc[i] *= fact;
                        hc[i] = fact;
                        local.push(r as u32);
                    }
                }
            }
            local
        })
        .collect();
    out.clear();
    for p in parts {
        out.extend_from_slice(&p);
    }
}

/// Trained CPU model arrays (same layout the GPU path fills).
pub struct CpuFit {
    pub feat: Vec<u32>,
    pub thr: Vec<f32>,
    pub leafval: Vec<f32>,
    pub fi_gain: Vec<f32>,
    pub fi_count: Vec<f32>,
}

/// Train on the host. `xtr` is row-major, already target-encoded; `boundaries` is the
/// `nf*(n_bins-1)` cut table from `compute_boundaries`; `base_score` matches the GPU path.
pub fn train(
    xtr: &[f32], ytr: &[f32], n: usize, nf: usize, boundaries: &[f32], base_score: f32, cfg: &Config,
) -> CpuFit {
    let nbins = cfg.n_bins;
    let nb1 = nbins - 1;
    let depth = cfg.depth;
    let leaves = 1usize << depth;
    let lambda = cfg.lambda as f64;
    let lr = cfg.learning_rate as f64;
    let min_child = cfg.min_child_h as f64;
    let n_trees = cfg.n_trees;
    let avail = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    // Use every core on small machines; on big ones use half (the histogram build is
    // bandwidth-bound) capped at 32. The old `clamp(8, 32)` forced 8 threads even on a
    // 2-core box -> 4x oversubscription and a big slowdown. Override via RUSTWOOD_THREADS.
    let nt = std::env::var("RUSTWOOD_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&t| t > 0)
        .unwrap_or_else(|| if avail <= 16 { avail } else { (avail / 2).min(32) })
        .max(1);

    let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().expect("rayon pool");
    pool.install(|| {
        // Bin once: row-major for the build, column-major for the apply (sequential reads).
        let mut bins = vec![0u8; n * nf];
        bins.par_chunks_mut(nf).enumerate().for_each(|(r, row)| {
            for f in 0..nf {
                let v = xtr[r * nf + f];
                let bf = &boundaries[f * nb1..f * nb1 + nb1];
                row[f] = bf.partition_point(|&c| c < v).min(nb1) as u8;
            }
        });
        let mut bins_col = vec![0u8; n * nf];
        bins_col.par_chunks_mut(n).enumerate().for_each(|(f, col)| {
            for (r, b) in col.iter_mut().enumerate() {
                *b = bins[r * nf + f];
            }
        });

        let mut pred = vec![base_score; n];
        let mut node = vec![0u32; n];
        let mut hess = vec![1.0f32; n];
        let mut feat = vec![0u32; n_trees * depth];
        let mut thr = vec![0f32; n_trees * depth];
        let mut leafval = vec![0f32; n_trees * leaves];
        let mut fi_gain = vec![0f32; nf];
        let mut fi_count = vec![0f32; nf];
        let all_rows: Vec<u32> = (0..n as u32).collect();
        let stride = if cfg.subsample < 0.999 {
            (1.0 / cfg.subsample as f64).round().max(1.0) as usize
        } else {
            1
        };
        let logistic = cfg.objective == Objective::Logistic;

        for t in 0..n_trees {
            // gradients / hessians for this round
            let mut g = vec![0f32; n];
            if logistic {
                g.par_iter_mut().zip(&pred).zip(&ytr[..n]).zip(hess.par_iter_mut()).for_each(
                    |(((gi, &p), &y), hi)| {
                        let pr = 1.0 / (1.0 + (-p).exp());
                        *gi = pr - y;
                        *hi = (pr * (1.0 - pr)).max(1e-6);
                    },
                );
            } else {
                g.par_iter_mut().zip(&pred).zip(&ytr[..n]).for_each(|((gi, &p), &y)| *gi = p - y);
                hess.iter_mut().for_each(|x| *x = 1.0);
            }
            node.iter_mut().for_each(|x| *x = 0);

            // row set for histograms (split-finding); leaf values use the same set.
            let mut samp_buf: Vec<u32> = Vec::new();
            let samp: &[u32] = if cfg.goss_top > 0.0 {
                goss_sample(&mut g, &mut hess, cfg.goss_top as f64, cfg.goss_other as f64,
                            t as u64, &mut samp_buf);
                &samp_buf
            } else if stride > 1 {
                samp_buf.extend((t % stride..n).step_by(stride).map(|x| x as u32));
                &samp_buf
            } else {
                &all_rows
            };

            let mut prev_hist: Vec<f32> = Vec::new();
            for level in 0..depth {
                let n_nodes = 1usize << level;
                let strideh = nf * nbins;
                let hist = if level == 0 || !cfg.subtract {
                    build_hist(samp, nt, n_nodes, nf, nbins, &node, &g, &hess, &bins, false)
                } else {
                    let prev_nodes = n_nodes / 2;
                    let odd =
                        build_hist(samp, nt, prev_nodes, nf, nbins, &node, &g, &hess, &bins, true);
                    let mut hh = vec![0f32; n_nodes * strideh * 2];
                    for i in 0..prev_nodes {
                        let parent = &prev_hist[i * strideh * 2..(i + 1) * strideh * 2];
                        let osl = &odd[i * strideh * 2..(i + 1) * strideh * 2];
                        let (lo, hi) = hh.split_at_mut((2 * i + 1) * strideh * 2);
                        let even = &mut lo[(2 * i) * strideh * 2..];
                        let oddc = &mut hi[..strideh * 2];
                        for x in 0..strideh * 2 {
                            oddc[x] = osl[x];
                            even[x] = parent[x] - osl[x];
                        }
                    }
                    hh
                };

                // best (feature, bin) over all nodes; gain loop is branchless -> vectorizes.
                let (best_gain, best_f, best_b) = (0..nf)
                    .into_par_iter()
                    .map(|f| {
                        let nsplit = nb1;
                        let mut gain = vec![0f64; nsplit];
                        let mut pg = vec![0f64; nbins];
                        let mut ph = vec![0f64; nbins];
                        for nd in 0..n_nodes {
                            let hb = (nd * nf + f) * nbins;
                            let (mut gl, mut hl) = (0f64, 0f64);
                            for b in 0..nbins {
                                gl += hist[(hb + b) * 2] as f64;
                                hl += hist[(hb + b) * 2 + 1] as f64;
                                pg[b] = gl;
                                ph[b] = hl;
                            }
                            let (gt, ht) = (gl, hl);
                            let parent = gt * gt / (ht + lambda);
                            for b in 0..nsplit {
                                let (gl, hl) = (pg[b], ph[b]);
                                let (gr, hr) = (gt - gl, ht - hl);
                                let valid = ((hl >= min_child) & (hr >= min_child)) as i32 as f64;
                                gain[b] +=
                                    valid * (gl * gl / (hl + lambda) + gr * gr / (hr + lambda) - parent);
                            }
                        }
                        let mut bg = f64::NEG_INFINITY;
                        let mut bb = 0usize;
                        for (b, &gv) in gain.iter().enumerate() {
                            if gv > bg {
                                bg = gv;
                                bb = b;
                            }
                        }
                        (bg, f, bb)
                    })
                    .reduce(|| (f64::NEG_INFINITY, 0, 0), |a, b| if b.0 > a.0 { b } else { a });

                feat[t * depth + level] = best_f as u32;
                thr[t * depth + level] = boundaries[best_f * nb1 + best_b];
                fi_gain[best_f] += best_gain.max(0.0) as f32;
                fi_count[best_f] += 1.0;

                let col = &bins_col[best_f * n..best_f * n + n];
                node.par_iter_mut().zip(col).for_each(|(nd, &b)| {
                    *nd = *nd * 2 + (b as usize > best_b) as u32;
                });
                prev_hist = hist;
            }

            // leaf values (Newton) over the sampled rows; prediction update over all rows.
            let m = samp.len();
            let lchunk = m.div_ceil(nt);
            let (lg, lc) = (0..nt)
                .into_par_iter()
                .map(|ti| {
                    let lo = ti * lchunk;
                    let hi = ((ti + 1) * lchunk).min(m);
                    let mut lg = vec![0f64; leaves];
                    let mut lc = vec![0f64; leaves];
                    if lo >= hi {
                        return (lg, lc);
                    }
                    for &ru in &samp[lo..hi] {
                        let r = ru as usize;
                        let l = node[r] as usize;
                        lg[l] += g[r] as f64;
                        lc[l] += hess[r] as f64;
                    }
                    (lg, lc)
                })
                .reduce(
                    || (vec![0f64; leaves], vec![0f64; leaves]),
                    |mut a, b| {
                        for i in 0..leaves {
                            a.0[i] += b.0[i];
                            a.1[i] += b.1[i];
                        }
                        a
                    },
                );
            for l in 0..leaves {
                leafval[t * leaves + l] = (-lr * lg[l] / (lc[l] + lambda)) as f32;
            }
            let lv = &leafval[t * leaves..(t + 1) * leaves];
            pred.par_iter_mut().zip(&node).for_each(|(p, &nd)| *p += lv[nd as usize]);
        }

        CpuFit { feat, thr, leafval, fi_gain, fi_count }
    })
}
