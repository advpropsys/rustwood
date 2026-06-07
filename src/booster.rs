//! GPU oblivious-tree gradient booster.
//!
//! The whole boosting round is enqueued on a single CUDA stream with **no host
//! synchronization** inside the loop: gradients, histograms, split selection
//! (`argmax_split`), leaf values and the chosen `(feature, threshold)` are all
//! computed and stored device-side. The host only syncs once at the end and copies
//! the finished model out. Device buffers are preallocated once and re-zeroed with
//! a kernel each level — no per-level allocation churn.
//!
//! A `profile` flag turns on per-kernel CUDA-stream-synchronized timing (used to
//! produce the flamegraph breakdown); it serializes the loop and is not used for
//! the headline throughput numbers.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::memory::{memcpy_dtoh_async, memcpy_htod_async};
use cuda_core::{CudaContext, DeviceBuffer, DriverError, LaunchConfig, PinnedHostBuffer};

use crate::config::{Config, Device, Objective};
use crate::data::{Dataset, compute_boundaries};
use crate::gpu_kernels::kernels;

type Err = Box<dyn std::error::Error>;

/// Kernel categories, in pipeline order — used for the profile breakdown / flamegraph.
pub const CATEGORIES: [&str; 9] = [
    "gradients",
    "zero_hist",
    "build_hist",
    "split_gain",
    "argmax_split",
    "apply_split",
    "leaf_hist",
    "leaf_values",
    "update_pred",
];

pub struct Booster {
    ctx: Option<Arc<CudaContext>>,
    cfg: Config,
    n_features: usize,
    base_score: f32,
    feat: Vec<u32>,
    thr: Vec<f32>,
    leafval: Vec<f32>,
    encoders: crate::encoding::Encoders,
    gmean: f32,
    fi_gain: Vec<f32>,
    fi_count: Vec<f32>,
    leafvar: Vec<f32>,
}

/// Inverse standard-normal CDF (Acklam's rational approximation). Used to turn the
/// GOSS top-rate into a half-normal gradient threshold multiplier.
fn norm_ppf(p: f64) -> f64 {
    let a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
             1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0];
    let b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
             6.680131188771972e1, -1.328068155288572e1];
    let c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
             -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
    let d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
             3.754408661907416e0];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Transpose row-major `x[i*nf+f]` into column-major `col[f*n+i]`.
fn column_major(x: &[f32], n: usize, nf: usize) -> Vec<f32> {
    let mut col = vec![0.0f32; n * nf];
    for i in 0..n {
        let row = &x[i * nf..i * nf + nf];
        for f in 0..nf {
            col[f * n + i] = row[f];
        }
    }
    col
}

#[inline]
fn elems(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

#[inline]
fn one_block_256() -> LaunchConfig {
    LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

/// Result of a profiled fit: total GPU loop time plus per-category nanoseconds.
pub struct Profile {
    pub total: Duration,
    pub per_category_ns: Vec<(String, u128)>,
}

impl Booster {
    pub fn new(cfg: Config) -> Result<Self, DriverError> {
        // CPU device runs GPU-free: no CUDA context is created.
        let ctx = match cfg.device {
            Device::Cpu => None,
            Device::Gpu => Some(CudaContext::new(cfg.gpu)?),
        };
        Ok(Self { ctx, cfg, n_features: 0, base_score: 0.0, feat: Vec::new(), thr: Vec::new(),
            leafval: Vec::new(), encoders: Vec::new(), gmean: 0.0,
            fi_gain: Vec::new(), fi_count: Vec::new(), leafvar: Vec::new() })
    }

    fn gpu_ctx(&self) -> &Arc<CudaContext> {
        self.ctx.as_ref().expect("GPU context unavailable (device = cpu)")
    }

    /// Construct a booster that reuses an existing CUDA context (the `--serve` worker keeps
    /// one resident so the ~400 ms init is paid once, not per fit).
    pub fn with_ctx(cfg: Config, ctx: Arc<CudaContext>) -> Self {
        Self { ctx: Some(ctx), cfg, n_features: 0, base_score: 0.0, feat: Vec::new(),
            thr: Vec::new(), leafval: Vec::new(), encoders: Vec::new(), gmean: 0.0,
            fi_gain: Vec::new(), fi_count: Vec::new(), leafvar: Vec::new() }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn fit(&mut self, ds: &Dataset) -> Result<Duration, Err> {
        if self.cfg.device == Device::Cpu {
            return self.fit_cpu(ds);
        }
        Ok(self.fit_inner(ds, false)?.total)
    }

    /// Host training path (no GPU). Mirrors `fit_inner`'s host prep, then builds the
    /// oblivious forest on the CPU and stores the same `feat`/`thr`/`leafval` model.
    fn fit_cpu(&mut self, ds: &Dataset) -> Result<Duration, Err> {
        let n = ds.n_train;
        let nf = ds.n_features;
        self.n_features = nf;
        let t0 = std::time::Instant::now();

        let mut xtr = ds.x_train.clone();
        if !self.cfg.categorical.is_empty() {
            let (enc, gm) = crate::encoding::fit_transform_train(
                &mut xtr, n, nf, &ds.y_train, &self.cfg.categorical, self.cfg.te_smoothing, 5,
                self.cfg.cat_hash_buckets);
            self.encoders = enc;
            self.gmean = gm;
        }
        let boundaries = compute_boundaries(
            &xtr, n, nf, self.cfg.n_bins, self.cfg.bin_sample, self.cfg.unique_bins);
        let mean = (ds.y_train.iter().sum::<f32>() / n as f32).clamp(1e-6, 1.0 - 1e-6);
        self.base_score = match self.cfg.objective {
            Objective::Logistic => (mean / (1.0 - mean)).ln(),
            Objective::SquaredError => mean,
        };

        let fit = crate::cpu::train(&xtr, &ds.y_train, n, nf, &boundaries, self.base_score, &self.cfg);
        self.feat = fit.feat;
        self.thr = fit.thr;
        self.leafval = fit.leafval;
        self.fi_gain = fit.fi_gain;
        self.fi_count = fit.fi_count;
        Ok(t0.elapsed())
    }

    pub fn fit_profiled(&mut self, ds: &Dataset) -> Result<Profile, Err> {
        self.fit_inner(ds, true)
    }

    fn fit_inner(&mut self, ds: &Dataset, profile: bool) -> Result<Profile, Err> {
        let stream = self.gpu_ctx().default_stream();
        let module = kernels::load(self.gpu_ctx())?;

        let n = ds.n_train;
        let nf = ds.n_features;
        let nb = self.cfg.n_bins;
        let depth = self.cfg.depth;
        let n_leaves = 1usize << depth;
        let max_groups = 1usize << (depth - 1);
        let lambda = self.cfg.lambda;
        let lr = self.cfg.learning_rate;
        self.n_features = nf;

        // Cap replication by a ~6 GiB budget for the largest (deepest-level) histogram.
        let max_hist = max_groups * nf * nb;
        let budget_elems = (6usize << 30) / (2 * std::mem::size_of::<f32>());
        let replicas = self.cfg.replicas.clamp(1, (budget_elems / max_hist).max(1));

        // --- one-time host prep: categorical target-encoding (OOF) + binning ---
        let mut xtr = ds.x_train.clone();
        if !self.cfg.categorical.is_empty() {
            let (enc, gm) = crate::encoding::fit_transform_train(
                &mut xtr, n, nf, &ds.y_train, &self.cfg.categorical, self.cfg.te_smoothing, 5,
                self.cfg.cat_hash_buckets);
            self.encoders = enc;
            self.gmean = gm;
        }
        let boundaries = compute_boundaries(&xtr, n, nf, nb, self.cfg.bin_sample, self.cfg.unique_bins);
        let xcol = column_major(&xtr, n, nf);
        let xcol_dev = DeviceBuffer::from_host(&stream, &xcol)?;
        let bound_dev = DeviceBuffer::from_host(&stream, &boundaries)?;
        let mut bins_dev = DeviceBuffer::<u8>::zeroed(&stream, n * nf)?;
        module.bin_features(&stream, elems(n * nf), &xcol_dev, &bound_dev, &mut bins_dev,
            n as u32, nf as u32, nb as u32)?;
        drop(xcol_dev);

        // Optional 4-bit bin packing (n_bins<=16): 2 rows/byte to cut bin-read bandwidth.
        let pack4_on = self.cfg.pack4 && nb <= 16;
        let half = (n + 1) / 2;
        let packed_dev = if pack4_on {
            let mut p = DeviceBuffer::<u8>::zeroed(&stream, nf * half)?;
            module.pack_bins4(&stream, elems(nf * half), &bins_dev, &mut p,
                n as u32, nf as u32, half as u32)?;
            Some(p)
        } else {
            None
        };
        let hist_bins: &DeviceBuffer<u8> = packed_dev.as_ref().unwrap_or(&bins_dev);
        let packed_flag = if pack4_on { 1u32 } else { 0u32 };
        let bin_stride = if pack4_on { half } else { n } as u32;

        let y_dev = DeviceBuffer::from_host(&stream, &ds.y_train)?;
        let mean = (ds.y_train.iter().sum::<f32>() / n as f32).clamp(1e-6, 1.0 - 1e-6);
        self.base_score = match self.cfg.objective {
            Objective::Logistic => (mean / (1.0 - mean)).ln(),
            Objective::SquaredError => mean,
        };
        let mut pred_dev = DeviceBuffer::from_host(&stream, &vec![self.base_score; n])?;

        // --- preallocate all working buffers once ---
        let mut g_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let mut h_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let mut leaf_dev = DeviceBuffer::<u32>::zeroed(&stream, n)?;
        // Two histogram buffer sets (A/B) ping-ponged across levels so the parent
        // level's raw histograms survive for subtraction.
        let mut hist_a_g = DeviceBuffer::<f32>::zeroed(&stream, replicas * max_groups * nf * nb)?;
        let mut hist_a_h = DeviceBuffer::<f32>::zeroed(&stream, replicas * max_groups * nf * nb)?;
        let mut hist_b_g = DeviceBuffer::<f32>::zeroed(&stream, replicas * max_groups * nf * nb)?;
        let mut hist_b_h = DeviceBuffer::<f32>::zeroed(&stream, replicas * max_groups * nf * nb)?;
        // Transient f16 accumulation buffers (opt-in `--features f16-hist`; the f16 atomic
        // is not in stock cuda-oxide).
        #[cfg(feature = "f16-hist")]
        let f16_len = if self.cfg.f16_hist { replicas * max_groups * nf * nb } else { 1 };
        #[cfg(feature = "f16-hist")]
        let mut hist16_g = DeviceBuffer::<f16>::zeroed(&stream, f16_len)?;
        #[cfg(feature = "f16-hist")]
        let mut hist16_h = DeviceBuffer::<f16>::zeroed(&stream, f16_len)?;
        // Transient fixed-point i32 accumulation buffers (used only with --int-hist).
        let int_len = if self.cfg.int_hist { replicas * max_groups * nf * nb } else { 1 };
        let mut histi_g = DeviceBuffer::<i32>::zeroed(&stream, int_len)?;
        let mut histi_h = DeviceBuffer::<i32>::zeroed(&stream, int_len)?;
        let mut gains = DeviceBuffer::<f32>::zeroed(&stream, nf * nb)?;
        let split_ft = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
        let mut leaf_g = DeviceBuffer::<f32>::zeroed(&stream, replicas * n_leaves)?;
        let mut leaf_h = DeviceBuffer::<f32>::zeroed(&stream, replicas * n_leaves)?;
        let mut leaf_g2 = DeviceBuffer::<f32>::zeroed(&stream, replicas * n_leaves)?;
        let leafvar_dev = DeviceBuffer::<f32>::zeroed(&stream, self.cfg.n_trees * n_leaves)?;
        let feat_dev = DeviceBuffer::<u32>::zeroed(&stream, self.cfg.n_trees * depth)?;
        let thr_dev = DeviceBuffer::<f32>::zeroed(&stream, self.cfg.n_trees * depth)?;
        let leafval_dev = DeviceBuffer::<f32>::zeroed(&stream, self.cfg.n_trees * n_leaves)?;
        let fi_gain_dev = DeviceBuffer::<f32>::zeroed(&stream, nf)?;
        let fi_count_dev = DeviceBuffer::<f32>::zeroed(&stream, nf)?;
        let mut monotone = vec![0i32; nf];
        for (i, &m) in self.cfg.monotone.iter().take(nf).enumerate() {
            monotone[i] = m;
        }
        let monotone_dev = DeviceBuffer::from_host(&stream, &monotone)?;

        // GOSS setup (gradient-based one-side sampling).
        let goss_on = self.cfg.goss_top > 0.0 && self.cfg.goss_top < 1.0;
        let goss_q = norm_ppf(1.0 - self.cfg.goss_top as f64 / 2.0) as f32;
        let goss_amplify = (1.0 - self.cfg.goss_top) / self.cfg.goss_other.max(1e-6);
        let mut goss_stats = DeviceBuffer::<f32>::zeroed(&stream, 2)?;
        let mut goss_thresh = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

        let mut timings: HashMap<&'static str, u128> = HashMap::new();
        stream.synchronize()?;
        let t0 = Instant::now();
        let mut last = t0;
        // After a launch, attribute its (serialized) wall time to `cat` when profiling.
        macro_rules! mark {
            ($cat:expr) => {
                if profile {
                    stream.synchronize()?;
                    let now = Instant::now();
                    *timings.entry($cat).or_insert(0) += (now - last).as_nanos();
                    last = now;
                }
            };
        }

        for tree in 0..self.cfg.n_trees {
            // Per-tree seed drives reproducible row/feature subsampling.
            let tree_seed = (tree as u32).wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
            match self.cfg.objective {
                Objective::Logistic => module.grad_logistic(&stream, elems(n), &pred_dev, &y_dev, &mut g_dev, &mut h_dev, n as u32)?,
                Objective::SquaredError => module.grad_l2(&stream, elems(n), &pred_dev, &y_dev, &mut g_dev, &mut h_dev, n as u32)?,
            }
            // Experimental: winsorize hessians at the p/(1-p) tails before histogramming.
            if self.cfg.clamp_hessian > 0.0 {
                let z = norm_ppf(1.0 - self.cfg.clamp_hessian as f64) as f32;
                module.zero2_f32(&stream, elems(2), &mut goss_stats, &mut goss_thresh, 2)?;
                module.hess_stats(&stream, elems(n), &h_dev, &goss_stats, n as u32)?;
                module.hess_clamp(&stream, elems(n), &mut h_dev, &goss_stats, n as u32, z, 1e-6)?;
            }
            // GOSS: reweight g/h in place (top-|grad| kept, rest sampled+amplified or zeroed).
            if goss_on {
                module.zero2_f32(&stream, elems(2), &mut goss_stats, &mut goss_thresh, 2)?;
                module.goss_stats(&stream, elems(n), &g_dev, &goss_stats, n as u32)?;
                module.goss_thresh(&stream, elems(1), &goss_stats, &goss_thresh, n as u32, goss_q)?;
                module.goss_apply(&stream, elems(n), &mut g_dev, &mut h_dev, &goss_thresh,
                    goss_amplify, self.cfg.goss_other, tree_seed ^ 0x00A5_A5A5, n as u32)?;
            }
            module.zero_u32(&stream, elems(n), &mut leaf_dev, n as u32)?;
            mark!("gradients");

            for d in 0..depth {
                let groups = 1usize << d;
                let hist_len = groups * nf * nb;
                let total_hist = replicas * hist_len;
                let even_level = d % 2 == 0;
                // Shared-mem path when this level's histogram fits a 32 KB block scratchpad.
                let smem = self.cfg.smem_hist && groups * nb <= 4096;
                #[cfg(feature = "f16-hist")]
                let f16 = self.cfg.f16_hist && !smem;
                #[cfg(not(feature = "f16-hist"))]
                let f16 = false;
                let int = self.cfg.int_hist && !smem && !f16;
                // Zero: the transient int/f16 buffer, or the current f32 buffer (parent kept).
                #[cfg(feature = "f16-hist")]
                let did_zero_f16 = if f16 {
                    module.zero2_f16(&stream, elems(total_hist), &mut hist16_g, &mut hist16_h, total_hist as u32)?;
                    true
                } else {
                    false
                };
                #[cfg(not(feature = "f16-hist"))]
                let did_zero_f16 = false;
                if did_zero_f16 {
                } else if int {
                    module.zero2_i32(&stream, elems(total_hist), &mut histi_g, &mut histi_h, total_hist as u32)?;
                } else {
                    let zero_len = if smem { hist_len } else { total_hist };
                    if even_level {
                        module.zero2_f32(&stream, elems(zero_len), &mut hist_a_g, &mut hist_a_h, zero_len as u32)?;
                    } else {
                        module.zero2_f32(&stream, elems(zero_len), &mut hist_b_g, &mut hist_b_h, zero_len as u32)?;
                    }
                }
                mark!("zero_hist");

                // cur = this level's histograms (f32); prev = parent level's (raw, retained).
                let (cur_g, cur_h, prev_g, prev_h) = if even_level {
                    (&hist_a_g, &hist_a_h, &hist_b_g, &hist_b_h)
                } else {
                    (&hist_b_g, &hist_b_h, &hist_a_g, &hist_a_h)
                };
                // Levels >= 1 build only the odd (right) children, then derive even ones.
                let use_sub = self.cfg.subtract && d >= 1;
                let odd_only = if use_sub { 1u32 } else { 0u32 };
                // f16 build path (opt-in): folded to the f32 `cur` buffer.
                #[cfg(feature = "f16-hist")]
                let did_build_f16 = if f16 {
                    module.build_hist_f16(&stream, elems(n), hist_bins, &leaf_dev, &g_dev, &h_dev,
                        &hist16_g, &hist16_h, n as u32, nf as u32, nb as u32,
                        replicas as u32, hist_len as u32, self.cfg.subsample, tree_seed, odd_only,
                        packed_flag, bin_stride)?;
                    module.reduce2_f16to32(&stream, elems(hist_len), &hist16_g, &hist16_h,
                        cur_g, cur_h, replicas as u32, hist_len as u32)?;
                    true
                } else {
                    false
                };
                #[cfg(not(feature = "f16-hist"))]
                let did_build_f16 = false;
                if did_build_f16 {
                } else if smem {
                    let n_tiles = ((n + 8191) / 8192).clamp(1, 4096) as u32;
                    let cfg_smem = LaunchConfig {
                        grid_dim: (nf as u32, n_tiles, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    module.build_hist_smem(&stream, cfg_smem, hist_bins, &leaf_dev, &g_dev, &h_dev,
                        cur_g, cur_h, n as u32, nf as u32, nb as u32, groups as u32, n_tiles,
                        self.cfg.subsample, tree_seed, odd_only, packed_flag, bin_stride)?;
                } else if int {
                    // Fixed-point i32 atomics into the transient buffer, rescaled to f32 cur.
                    module.build_hist_int(&stream, elems(n), hist_bins, &leaf_dev, &g_dev, &h_dev,
                        &histi_g, &histi_h, n as u32, nf as u32, nb as u32,
                        replicas as u32, hist_len as u32, self.cfg.subsample, tree_seed, odd_only,
                        packed_flag, bin_stride, self.cfg.int_scale)?;
                    module.reduce2_int_to_f32(&stream, elems(hist_len), &histi_g, &histi_h,
                        cur_g, cur_h, replicas as u32, hist_len as u32, 1.0 / self.cfg.int_scale)?;
                } else {
                    module.build_hist(&stream, elems(n), hist_bins, &leaf_dev, &g_dev, &h_dev,
                        cur_g, cur_h, n as u32, nf as u32, nb as u32,
                        replicas as u32, hist_len as u32, self.cfg.subsample, tree_seed, odd_only,
                        packed_flag, bin_stride)?;
                    if replicas > 1 {
                        module.reduce2(&stream, elems(hist_len), cur_g, cur_h, replicas as u32, hist_len as u32)?;
                    }
                }
                if use_sub {
                    let prev_groups = 1usize << (d - 1);
                    module.subtract_even(&stream, elems(prev_groups * nf * nb), cur_g, cur_h,
                        prev_g, prev_h, prev_groups as u32, nf as u32, nb as u32)?;
                }
                mark!("build_hist");

                module.split_gain(&stream, elems(nf * nb), cur_g, cur_h, &mut gains,
                    nf as u32, nb as u32, groups as u32, lambda, self.cfg.min_child_h,
                    self.cfg.colsample, tree_seed, &monotone_dev)?;
                mark!("split_gain");

                let k = (tree * depth + d) as u32;
                module.argmax_split(&stream, one_block_256(), &gains, &bound_dev, &feat_dev,
                    &thr_dev, &split_ft, &fi_gain_dev, &fi_count_dev, k, nf as u32, nb as u32)?;
                mark!("argmax_split");

                module.apply_split(&stream, elems(n), hist_bins, &mut leaf_dev, &split_ft,
                    n as u32, packed_flag, bin_stride)?;
                mark!("apply_split");
            }

            let total_leaf = replicas * n_leaves;
            module.zero2_f32(&stream, elems(total_leaf), &mut leaf_g, &mut leaf_h, total_leaf as u32)?;
            module.zero_f32(&stream, elems(total_leaf), &mut leaf_g2, total_leaf as u32)?;
            module.leaf_hist(&stream, elems(n), &leaf_dev, &g_dev, &h_dev, &leaf_g, &leaf_h, &leaf_g2,
                n as u32, n_leaves as u32, replicas as u32, self.cfg.subsample, tree_seed)?;
            if replicas > 1 {
                module.reduce2(&stream, elems(n_leaves), &leaf_g, &leaf_h, replicas as u32, n_leaves as u32)?;
                module.reduce2(&stream, elems(n_leaves), &leaf_g2, &leaf_g2, replicas as u32, n_leaves as u32)?;
            }
            mark!("leaf_hist");

            let base = (tree * n_leaves) as u32;
            module.compute_leaf_values(&stream, elems(n_leaves), &leaf_g, &leaf_h, &leaf_g2,
                &leafval_dev, &leafvar_dev, base, lr, lambda, n_leaves as u32)?;
            mark!("leaf_values");

            module.update_pred(&stream, elems(n), &leaf_dev, &leafval_dev, base, &mut pred_dev, n as u32)?;
            mark!("update_pred");
        }

        stream.synchronize()?;
        let total = t0.elapsed();

        self.feat = feat_dev.to_host_vec(&stream)?;
        self.thr = thr_dev.to_host_vec(&stream)?;
        self.leafval = leafval_dev.to_host_vec(&stream)?;
        self.fi_gain = fi_gain_dev.to_host_vec(&stream)?;
        self.fi_count = fi_count_dev.to_host_vec(&stream)?;
        if self.cfg.pgbm {
            self.leafvar = leafvar_dev.to_host_vec(&stream)?;
        }

        let per_category_ns = CATEGORIES.iter().map(|c| (c.to_string(), *timings.get(c).unwrap_or(&0))).collect();
        Ok(Profile { total, per_category_ns })
    }

    pub fn base_score(&self) -> f32 {
        self.base_score
    }

    /// Per-feature importance: total split gain and number of times chosen.
    pub fn feature_importance(&self) -> (&[f32], &[f32]) {
        (&self.fi_gain, &self.fi_count)
    }

    /// Score `n` row-major rows. Returns `(raw_margins, gpu_inference_time)`.
    pub fn predict(&self, x: &[f32], n: usize) -> Result<(Vec<f32>, Duration), Err> {
        let stream = self.gpu_ctx().default_stream();
        let module = kernels::load(self.gpu_ctx())?;
        let nf = self.n_features;

        // Apply categorical encoding to the test features (no-op if none configured).
        let mut xt = x.to_vec();
        if !self.cfg.categorical.is_empty() {
            crate::encoding::transform(&mut xt, n, nf, &self.cfg.categorical, &self.encoders,
                self.gmean, self.cfg.cat_hash_buckets);
        }
        let x_dev = DeviceBuffer::from_host(&stream, &xt)?;
        let feat_dev = DeviceBuffer::from_host(&stream, &self.feat)?;
        let thr_dev = DeviceBuffer::from_host(&stream, &self.thr)?;
        let leafval_dev = DeviceBuffer::from_host(&stream, &self.leafval)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

        stream.synchronize()?;
        let t0 = Instant::now();
        module.predict(&stream, elems(n), &x_dev, &feat_dev, &thr_dev, &leafval_dev, &mut out_dev,
            n as u32, nf as u32, self.cfg.depth as u32, self.cfg.n_trees as u32, self.base_score)?;
        stream.synchronize()?;
        let dt = t0.elapsed();
        Ok((out_dev.to_host_vec(&stream)?, dt))
    }

    /// PGBM predictive standard deviation per row (sqrt of summed per-leaf variances).
    pub fn predict_std(&self, x: &[f32], n: usize) -> Result<Vec<f32>, Err> {
        let stream = self.gpu_ctx().default_stream();
        let module = kernels::load(self.gpu_ctx())?;
        let nf = self.n_features;
        let mut xt = x.to_vec();
        if !self.cfg.categorical.is_empty() {
            crate::encoding::transform(&mut xt, n, nf, &self.cfg.categorical, &self.encoders,
                self.gmean, self.cfg.cat_hash_buckets);
        }
        let x_dev = DeviceBuffer::from_host(&stream, &xt)?;
        let feat_dev = DeviceBuffer::from_host(&stream, &self.feat)?;
        let thr_dev = DeviceBuffer::from_host(&stream, &self.thr)?;
        let leafvar_dev = DeviceBuffer::from_host(&stream, &self.leafvar)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        module.predict_var(&stream, elems(n), &x_dev, &feat_dev, &thr_dev, &leafvar_dev,
            &mut out_dev, n as u32, nf as u32, self.cfg.depth as u32, self.cfg.n_trees as u32)?;
        let var = out_dev.to_host_vec(&stream)?;
        Ok(var.iter().map(|&v| v.max(0.0).sqrt()).collect())
    }

    /// Host-side (CPU) inference — the right path for single-row / small-batch latency,
    /// since it avoids the GPU launch+transfer+sync floor. The oblivious forest is a
    /// branchless `depth`-bit index per tree, so this is a few hundred cheap ops/row.
    pub fn predict_cpu(&self, x: &[f32], n: usize, out: &mut [f32]) {
        let nf = self.n_features;
        let depth = self.cfg.depth;
        let n_leaves = 1usize << depth;
        for i in 0..n {
            let row = &x[i * nf..i * nf + nf];
            let mut acc = self.base_score;
            for t in 0..self.cfg.n_trees {
                let mut leaf = 0usize;
                let kbase = t * depth;
                for d in 0..depth {
                    let k = kbase + d;
                    let bit = (row[self.feat[k] as usize] > self.thr[k]) as usize;
                    leaf = leaf * 2 + bit;
                }
                acc += self.leafval[t * n_leaves + leaf];
            }
            out[i] = acc;
        }
    }

    /// True end-to-end inference latency per call, model resident on the GPU. For each
    /// batch size, times `repeats` full predict calls (host->device copy of the batch,
    /// kernel, device->host copy of the result, completion sync) and returns the mean
    /// per-call nanoseconds. Per-row latency is `per_call / batch`.
    pub fn bench_latency(
        &self,
        x: &[f32],
        n_rows: usize,
        batches: &[usize],
        repeats: usize,
    ) -> Result<Vec<(usize, f64)>, Err> {
        let stream = self.gpu_ctx().default_stream();
        let module = kernels::load(self.gpu_ctx())?;
        let nf = self.n_features;
        // Model uploaded once and kept resident (as a real inference server would).
        let feat_dev = DeviceBuffer::from_host(&stream, &self.feat)?;
        let thr_dev = DeviceBuffer::from_host(&stream, &self.thr)?;
        let leafval_dev = DeviceBuffer::from_host(&stream, &self.leafval)?;
        stream.synchronize()?;

        let mut out = Vec::new();
        for &breq in batches {
            let b = breq.min(n_rows);
            if b == 0 {
                continue;
            }
            let mut call = |reps: usize| -> Result<(), Err> {
                for _ in 0..reps {
                    let x_dev = DeviceBuffer::from_host(&stream, &x[0..b * nf])?;
                    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, b)?;
                    module.predict(&stream, elems(b), &x_dev, &feat_dev, &thr_dev, &leafval_dev,
                        &mut out_dev, b as u32, nf as u32, self.cfg.depth as u32,
                        self.cfg.n_trees as u32, self.base_score)?;
                    let _ = out_dev.to_host_vec(&stream)?; // forces D2H completion
                }
                Ok(())
            };
            call(3)?; // warmup
            let t0 = Instant::now();
            call(repeats)?;
            out.push((b, t0.elapsed().as_nanos() as f64 / repeats as f64));
        }
        Ok(out)
    }

    /// Peak inference throughput sweep: for each row count, allocate device input/output
    /// (no per-call H2D — data resident), launch predict `repeats` times with spin-sync,
    /// and report mean ns/row. Stops at the first allocation that fails (VRAM ceiling).
    /// Returns `(rows, ns_per_row, ok)`.
    pub fn bench_throughput(&self, rows: &[usize], repeats: usize) -> Result<Vec<(usize, f64, bool)>, Err> {
        let stream = self.gpu_ctx().default_stream();
        let module = kernels::load(self.gpu_ctx())?;
        let nf = self.n_features;
        let feat_dev = DeviceBuffer::from_host(&stream, &self.feat)?;
        let thr_dev = DeviceBuffer::from_host(&stream, &self.thr)?;
        let leafval_dev = DeviceBuffer::from_host(&stream, &self.leafval)?;
        let s = stream.cu_stream();

        let mut res = Vec::new();
        for &nrow in rows {
            if nrow > u32::MAX as usize {
                res.push((nrow, f64::NAN, false));
                continue;
            }
            let mut x_dev = match DeviceBuffer::<f32>::zeroed(&stream, nrow * nf) {
                Ok(b) => b,
                Err(_) => { res.push((nrow, f64::NAN, false)); break; }
            };
            let mut out_dev = match DeviceBuffer::<f32>::zeroed(&stream, nrow) {
                Ok(b) => b,
                Err(_) => { res.push((nrow, f64::NAN, false)); break; }
            };
            // Populate with varied pseudo-random features so rows reach different
            // leaves (realistic scattered gather), not a uniform all-same-leaf buffer.
            module.fill_pseudo(&stream, elems(nrow * nf), &mut x_dev, (nrow * nf) as u32, 0x9E37)?;
            stream.synchronize()?;
            let mut once = || -> Result<(), Err> {
                module.predict(&stream, elems(nrow), &x_dev, &feat_dev, &thr_dev, &leafval_dev,
                    &mut out_dev, nrow as u32, nf as u32, self.cfg.depth as u32,
                    self.cfg.n_trees as u32, self.base_score)?;
                while unsafe { cuda_bindings::cuStreamQuery(s) } != 0 {}
                Ok(())
            };
            once()?;
            once()?;
            let t0 = Instant::now();
            for _ in 0..repeats {
                once()?;
            }
            let per_call = t0.elapsed().as_nanos() as f64 / repeats as f64;
            res.push((nrow, per_call / nrow as f64, true));
        }
        Ok(res)
    }

    /// EXPERIMENTAL low-latency GPU path. Three tricks vs `bench_latency`:
    ///   1. device + pinned-host buffers preallocated once and reused (no per-call
    ///      `cudaMalloc`, which is the dominant ~tens-of-µs cost),
    ///   2. pinned host staging for faster, truly-async H2D/D2H,
    ///   3. **spin-sync**: busy-poll `cuStreamQuery` instead of a blocking
    ///      `cuStreamSynchronize`, trading a core for the lowest possible wakeup.
    pub fn bench_latency_fast(
        &self,
        x: &[f32],
        n_rows: usize,
        batches: &[usize],
        repeats: usize,
    ) -> Result<Vec<(usize, f64)>, Err> {
        let stream = self.gpu_ctx().default_stream();
        let module = kernels::load(self.gpu_ctx())?;
        let nf = self.n_features;
        let feat_dev = DeviceBuffer::from_host(&stream, &self.feat)?;
        let thr_dev = DeviceBuffer::from_host(&stream, &self.thr)?;
        let leafval_dev = DeviceBuffer::from_host(&stream, &self.leafval)?;

        let maxb = batches.iter().copied().max().unwrap_or(1).min(n_rows).max(1);
        let x_dev = DeviceBuffer::<f32>::zeroed(&stream, maxb * nf)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, maxb)?;
        let mut pin_in = PinnedHostBuffer::<f32>::zeroed(self.gpu_ctx(), maxb * nf)?;
        let mut pin_out = PinnedHostBuffer::<f32>::zeroed(self.gpu_ctx(), maxb)?;
        let s = stream.cu_stream();
        let dptr_x = x_dev.cu_deviceptr();
        let dptr_out = out_dev.cu_deviceptr();
        let fb = std::mem::size_of::<f32>();
        stream.synchronize()?;

        let mut out = Vec::new();
        for &breq in batches {
            let b = breq.min(n_rows).max(1);
            pin_in.as_mut_slice()[..b * nf].copy_from_slice(&x[..b * nf]);
            let mut once = || -> Result<(), Err> {
                unsafe { memcpy_htod_async(dptr_x, pin_in.as_ptr(), b * nf * fb, s)? };
                module.predict(&stream, elems(b), &x_dev, &feat_dev, &thr_dev, &leafval_dev,
                    &mut out_dev, b as u32, nf as u32, self.cfg.depth as u32,
                    self.cfg.n_trees as u32, self.base_score)?;
                unsafe { memcpy_dtoh_async(pin_out.as_mut_ptr(), dptr_out, b * fb, s)? };
                // spin-sync: busy-poll until the stream drains. cuStreamQuery returns
                // CUresult (u32); 0 == CUDA_SUCCESS, 600 == CUDA_ERROR_NOT_READY.
                while unsafe { cuda_bindings::cuStreamQuery(s) } != 0 {}
                Ok(())
            };
            for _ in 0..5 {
                once()?;
            }
            let t0 = Instant::now();
            for _ in 0..repeats {
                once()?;
            }
            out.push((b, t0.elapsed().as_nanos() as f64 / repeats as f64));
        }
        Ok(out)
    }
}

// --- .rwood model serialization: compact little-endian binary, raw-array blit I/O ---

#[inline]
fn as_bytes<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn wu32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn warr<W: Write, T: Copy>(w: &mut W, s: &[T]) -> std::io::Result<()> {
    wu32(w, s.len() as u32)?;
    w.write_all(as_bytes(s))
}
fn ru32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn rf32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn rvec_f32<R: Read>(r: &mut R) -> std::io::Result<Vec<f32>> {
    let len = ru32(r)? as usize;
    let mut v = vec![0f32; len];
    r.read_exact(unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 4) })?;
    Ok(v)
}
fn rvec_u32<R: Read>(r: &mut R) -> std::io::Result<Vec<u32>> {
    let len = ru32(r)? as usize;
    let mut v = vec![0u32; len];
    r.read_exact(unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * 4) })?;
    Ok(v)
}

impl Booster {
    /// Serialize the trained model to a `.rwood` binary: a small header, raw f32/u32 array
    /// blits (feat/thr/leafval), then the target-encoder maps. Little-endian.
    pub fn save_model(&self, path: &str) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(b"RWD1")?;
        wu32(&mut w, 1)?; // format version
        wu32(&mut w, self.n_features as u32)?;
        wu32(&mut w, self.cfg.n_trees as u32)?;
        wu32(&mut w, self.cfg.depth as u32)?;
        wu32(&mut w, matches!(self.cfg.objective, Objective::Logistic) as u32)?;
        w.write_all(&self.base_score.to_le_bytes())?;
        w.write_all(&self.gmean.to_le_bytes())?;
        w.write_all(&self.cfg.cat_hash_buckets.to_le_bytes())?;
        wu32(&mut w, self.cfg.categorical.len() as u32)?;
        for &c in &self.cfg.categorical {
            wu32(&mut w, c as u32)?;
        }
        warr(&mut w, &self.feat)?;
        warr(&mut w, &self.thr)?;
        warr(&mut w, &self.leafval)?;
        wu32(&mut w, self.encoders.len() as u32)?;
        for map in &self.encoders {
            wu32(&mut w, map.len() as u32)?;
            for (&k, &v) in map {
                w.write_all(&k.to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.flush()
    }

    /// Load a `.rwood` model into a CPU-device Booster (no GPU context). Score with
    /// `predict_host`.
    pub fn load_model(path: &str) -> std::io::Result<Booster> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"RWD1" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a .rwood file"));
        }
        let _ver = ru32(&mut r)?;
        let n_features = ru32(&mut r)? as usize;
        let n_trees = ru32(&mut r)? as usize;
        let depth = ru32(&mut r)? as usize;
        let logistic = ru32(&mut r)? == 1;
        let base_score = rf32(&mut r)?;
        let gmean = rf32(&mut r)?;
        let mut hb = [0u8; 8];
        r.read_exact(&mut hb)?;
        let cat_hash_buckets = u64::from_le_bytes(hb);
        let n_cat = ru32(&mut r)? as usize;
        let mut categorical = Vec::with_capacity(n_cat);
        for _ in 0..n_cat {
            categorical.push(ru32(&mut r)? as usize);
        }
        let feat = rvec_u32(&mut r)?;
        let thr = rvec_f32(&mut r)?;
        let leafval = rvec_f32(&mut r)?;
        let n_enc = ru32(&mut r)? as usize;
        let mut encoders: crate::encoding::Encoders = Vec::with_capacity(n_enc);
        for _ in 0..n_enc {
            let ml = ru32(&mut r)? as usize;
            let mut map = HashMap::with_capacity(ml);
            for _ in 0..ml {
                let mut kb = [0u8; 8];
                r.read_exact(&mut kb)?;
                map.insert(i64::from_le_bytes(kb), rf32(&mut r)?);
            }
            encoders.push(map);
        }
        let mut cfg = Config::default();
        cfg.n_trees = n_trees;
        cfg.depth = depth;
        cfg.objective = if logistic { Objective::Logistic } else { Objective::SquaredError };
        cfg.categorical = categorical;
        cfg.cat_hash_buckets = cat_hash_buckets;
        cfg.device = Device::Cpu;
        Ok(Booster {
            ctx: None, cfg, n_features, base_score, feat, thr, leafval, encoders, gmean,
            fi_gain: Vec::new(), fi_count: Vec::new(), leafvar: Vec::new(),
        })
    }

    /// Host prediction: apply target encoding (if any), then the branchless oblivious
    /// traversal. GPU-free -- the right path for a loaded model.
    pub fn predict_host(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        if self.cfg.categorical.is_empty() {
            self.predict_cpu(x, n, &mut out);
        } else {
            let mut xt = x.to_vec();
            crate::encoding::transform(&mut xt, n, self.n_features, &self.cfg.categorical,
                &self.encoders, self.gmean, self.cfg.cat_hash_buckets);
            self.predict_cpu(&xt, n, &mut out);
        }
        out
    }
}
