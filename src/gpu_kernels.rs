//! GPU kernels for histogram-based oblivious-tree gradient boosting.
//!
//! Compiled to PTX (sm_103 / Blackwell B300) by `rustc-codegen-cuda`.
//!
//! The whole boosting round runs on-device with **no host synchronization**: split
//! selection (`argmax_split`), leaf values (`compute_leaf_values`) and the chosen
//! `(feature, threshold)` are all written into device-resident buffers, so the host
//! just enqueues kernels on one stream and copies the finished model out once.
//!
//! Histogram accumulation uses device-scope `f32` atomics via the
//! `&[f32] -> DeviceAtomicF32` pointer-cast pattern (each bin is shared across
//! threads), matching the repo's `atomics` example.

use cuda_device::atomic::{AtomicOrdering, BlockAtomicF32, DeviceAtomicF32, DeviceAtomicI32};
// f16 atomics are not in stock cuda-oxide; the f16 histogram path is opt-in via `f16-hist`.
#[cfg(feature = "f16-hist")]
use cuda_device::atomic::DeviceAtomicF16;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

/// Internal accumulator precision for histogram cumulation / gain / replica folding.
/// f32 by default (storage is always f32); `--features f64-hist` switches the
/// arithmetic to f64 to harden against rounding/cancellation at very large N.
#[cfg(feature = "f64-hist")]
pub type Acc = f64;
#[cfg(not(feature = "f64-hist"))]
pub type Acc = f32;

#[inline(always)]
fn atomic_add_f32(slice: &[f32], idx: usize, val: f32) {
    unsafe {
        let cell = &*(slice.as_ptr().add(idx) as *const DeviceAtomicF32);
        cell.fetch_add(val, AtomicOrdering::Relaxed);
    }
}

/// Half-precision atomic add (hardware `atom.add.noftz.f16`). Halves the atomic write
/// traffic of the histogram build; precision is recovered by folding to f32 in reduce.
#[cfg(feature = "f16-hist")]
#[inline(always)]
fn atomic_add_f16(slice: &[f16], idx: usize, val: f16) {
    unsafe {
        let cell = &*(slice.as_ptr().add(idx) as *const DeviceAtomicF16);
        cell.fetch_add(val, AtomicOrdering::Relaxed);
    }
}

/// 32-bit integer atomic add (`atom.add.u32`/`s32` — the fastest atomic path). Used for
/// the fixed-point histogram, where gradients/hessians are scaled to integers.
#[inline(always)]
fn atomic_add_i32(slice: &[i32], idx: usize, val: i32) {
    unsafe {
        let cell = &*(slice.as_ptr().add(idx) as *const DeviceAtomicI32);
        cell.fetch_add(val, AtomicOrdering::Relaxed);
    }
}

/// Round an f32 to the nearest i32 (round-half-away-from-zero).
#[inline(always)]
fn round_i32(x: f32) -> i32 {
    if x >= 0.0 { (x + 0.5) as i32 } else { (x - 0.5) as i32 }
}

/// Deterministic hash of `(id, seed)` -> [0, 1). Used for stochastic row/feature
/// subsampling so the same subset is reproducible across a tree's levels.
/// Fixed salts mixed into the sampling RNG and the pseudo-data generator, decorrelating
/// draws from contiguous thread indices. Deliberate build constants — do not change.
const SAMPLE_SALT: u32 = 0x455A_4B41;
const GEN_SALT: u32 = 0x5052_4F50;

#[inline(always)]
fn hash01(id: u32, seed: u32) -> f32 {
    let mut x = id.wrapping_mul(2654435761).wrapping_add(seed ^ SAMPLE_SALT);
    x ^= x >> 15;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x & 0x00FF_FFFF) as f32 / 16_777_215.0
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Fill `buf[0..n]` with pseudo-random values in roughly [-4, 4) (a hash of the
    /// index), so throughput benchmarks exercise realistic, divergent leaf paths
    /// instead of a uniform buffer that lets every row hit the same cached leaf.
    #[kernel]
    pub fn fill_pseudo(mut buf: DisjointSlice<f32>, n: u32, seed: u32) {
        let id = thread::index_1d().get();
        if id >= n as usize {
            return;
        }
        let mut x = (id as u32).wrapping_mul(2654435761).wrapping_add(seed ^ GEN_SALT);
        x ^= x >> 15;
        x = x.wrapping_mul(2246822519);
        x ^= x >> 13;
        let u = (x & 0x00FF_FFFF) as f32 / 16_777_215.0; // [0,1)
        if let Some(slot) = buf.get_mut(thread::index_1d()) {
            *slot = (u - 0.5) * 8.0;
        }
    }

    /// Zero the first `n` elements of one `f32` buffer.
    #[kernel]
    pub fn zero_f32(mut buf: DisjointSlice<f32>, n: u32) {
        let idx = thread::index_1d();
        if idx.get() >= n as usize {
            return;
        }
        if let Some(slot) = buf.get_mut(idx) {
            *slot = 0.0;
        }
    }

    /// Zero the first `n` elements of two `f32` buffers in one launch.
    #[kernel]
    pub fn zero2_f32(mut a: DisjointSlice<f32>, mut b: DisjointSlice<f32>, n: u32) {
        if thread::index_1d().get() >= n as usize {
            return;
        }
        if let Some(s) = a.get_mut(thread::index_1d()) {
            *s = 0.0;
        }
        if let Some(s) = b.get_mut(thread::index_1d()) {
            *s = 0.0;
        }
    }

    /// Shared-memory privatized histogram (HFT mechanical-sympathy: accumulate in the
    /// L1-speed scratchpad, flush once to DRAM). One block per (feature, row-tile):
    /// builds that feature's `[groups*n_bins]` histogram in shared memory with on-chip
    /// `.cta` atomics, then atomic-flushes to global. Requires `groups*n_bins <= 4096`.
    /// Grid = (n_features, n_tiles, 1), block = 256.
    #[kernel]
    pub fn build_hist_smem(
        bins: &[u8],
        leaf: &[u32],
        g: &[f32],
        h: &[f32],
        hist_g: &[f32],
        hist_h: &[f32],
        n: u32,
        n_features: u32,
        n_bins: u32,
        groups: u32,
        n_tiles: u32,
        subsample: f32,
        tree_seed: u32,
        odd_only: u32,
        packed: u32,
        bin_row_stride: u32,
    ) {
        static mut SG: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut SH: SharedArray<f32, 4096> = SharedArray::UNINIT;

        let f = thread::blockIdx_x() as usize;
        let tile = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let nb = n_bins as usize;
        let hsize = groups as usize * nb;

        // Zero the shared histogram.
        let mut s = tid;
        while s < hsize {
            unsafe {
                SG[s] = 0.0;
                SH[s] = 0.0;
            }
            s += bdim;
        }
        thread::sync_threads();

        // Accumulate this tile's rows into shared memory.
        let nn = n as usize;
        let rs = bin_row_stride as usize;
        let chunk = (nn + n_tiles as usize - 1) / n_tiles as usize;
        let start = tile * chunk;
        let mut end = start + chunk;
        if end > nn {
            end = nn;
        }
        let mut i = start + tid;
        while i < end {
            let keep_sub = subsample >= 1.0 || hash01(i as u32, tree_seed) < subsample;
            if keep_sub {
                let grp = leaf[i] as usize;
                if !(odd_only == 1 && (grp & 1) == 0) {
                    let bin = if packed == 1 {
                        let byte = bins[f * rs + (i >> 1)];
                        (if i & 1 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }) as usize
                    } else {
                        bins[f * rs + i] as usize
                    };
                    let off = grp * nb + bin;
                    unsafe {
                        let pg = &SG[off] as *const f32 as *const BlockAtomicF32;
                        (*pg).fetch_add(g[i], AtomicOrdering::Relaxed);
                        let ph = &SH[off] as *const f32 as *const BlockAtomicF32;
                        (*ph).fetch_add(h[i], AtomicOrdering::Relaxed);
                    }
                }
            }
            i += bdim;
        }
        thread::sync_threads();

        // Flush shared -> global (only non-empty bins).
        let nf = n_features as usize;
        let mut s2 = tid;
        while s2 < hsize {
            let vg = unsafe { SG[s2] };
            let vh = unsafe { SH[s2] };
            if vg != 0.0 || vh != 0.0 {
                let grp = s2 / nb;
                let bin = s2 % nb;
                let goff = (grp * nf + f) * nb + bin;
                atomic_add_f32(hist_g, goff, vg);
                atomic_add_f32(hist_h, goff, vh);
            }
            s2 += bdim;
        }
    }

    /// Zero the first `n` elements of two `i32` buffers in one launch.
    #[kernel]
    pub fn zero2_i32(mut a: DisjointSlice<i32>, mut b: DisjointSlice<i32>, n: u32) {
        if thread::index_1d().get() >= n as usize {
            return;
        }
        if let Some(s) = a.get_mut(thread::index_1d()) {
            *s = 0;
        }
        if let Some(s) = b.get_mut(thread::index_1d()) {
            *s = 0;
        }
    }

    /// Fixed-point variant of `build_hist`: gradients/hessians are scaled by `scale`,
    /// rounded to i32, and accumulated with integer atomics (the fastest atomic path).
    /// `reduce2_int_to_f32` folds and rescales back to f32. Same semantics otherwise.
    #[kernel]
    pub fn build_hist_int(
        bins: &[u8],
        leaf: &[u32],
        g: &[f32],
        h: &[f32],
        hist_g: &[i32],
        hist_h: &[i32],
        n: u32,
        n_features: u32,
        n_bins: u32,
        n_replicas: u32,
        replica_stride: u32,
        subsample: f32,
        tree_seed: u32,
        odd_only: u32,
        packed: u32,
        bin_row_stride: u32,
        scale: f32,
    ) {
        let i = thread::index_1d().get();
        let nn = n as usize;
        if i >= nn {
            return;
        }
        if subsample < 1.0 && hash01(i as u32, tree_seed) >= subsample {
            return;
        }
        let grp = leaf[i] as usize;
        if odd_only == 1 && (grp & 1) == 0 {
            return;
        }
        let gq = round_i32(g[i] * scale);
        let hq = round_i32(h[i] * scale);
        let nf = n_features as usize;
        let nb = n_bins as usize;
        let rs = bin_row_stride as usize;
        let rbase = (thread::blockIdx_x() % n_replicas) as usize * replica_stride as usize;
        let row_base = grp * nf;
        let mut f = 0usize;
        while f < nf {
            let bin = if packed == 1 {
                let byte = bins[f * rs + (i >> 1)];
                (if i & 1 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }) as usize
            } else {
                bins[f * rs + i] as usize
            };
            let off = rbase + (row_base + f) * nb + bin;
            atomic_add_i32(hist_g, off, gq);
            atomic_add_i32(hist_h, off, hq);
            f += 1;
        }
    }

    /// Fold `n_replicas` i32 histogram copies into f32 outputs, rescaling by `inv_scale`.
    /// Accumulates in f32 to avoid i32 overflow when summing replicas.
    #[kernel]
    pub fn reduce2_int_to_f32(
        src_g: &[i32],
        src_h: &[i32],
        dst_g: &[f32],
        dst_h: &[f32],
        n_replicas: u32,
        stride: u32,
        inv_scale: f32,
    ) {
        let i = thread::index_1d().get();
        let st = stride as usize;
        if i >= st {
            return;
        }
        let mut sg = 0.0f32;
        let mut sh = 0.0f32;
        let mut r = 0usize;
        while r < n_replicas as usize {
            sg += src_g[r * st + i] as f32;
            sh += src_h[r * st + i] as f32;
            r += 1;
        }
        unsafe {
            *(dst_g.as_ptr() as *mut f32).add(i) = sg * inv_scale;
            *(dst_h.as_ptr() as *mut f32).add(i) = sh * inv_scale;
        }
    }

    /// Zero the first `n` elements of two `f16` buffers in one launch.
    #[kernel]
    #[cfg(feature = "f16-hist")]
    pub fn zero2_f16(mut a: DisjointSlice<f16>, mut b: DisjointSlice<f16>, n: u32) {
        if thread::index_1d().get() >= n as usize {
            return;
        }
        if let Some(s) = a.get_mut(thread::index_1d()) {
            *s = 0.0f16;
        }
        if let Some(s) = b.get_mut(thread::index_1d()) {
            *s = 0.0f16;
        }
    }

    /// f16 variant of `build_hist`: accumulate the gradient/hessian histograms with
    /// half-precision atomics (half the atomic write bandwidth). Storage is f16; the
    /// values are folded back to f32 by `reduce2_f16to32`. Same semantics/params as
    /// `build_hist` otherwise.
    #[kernel]
    #[cfg(feature = "f16-hist")]
    pub fn build_hist_f16(
        bins: &[u8],
        leaf: &[u32],
        g: &[f32],
        h: &[f32],
        hist_g: &[f16],
        hist_h: &[f16],
        n: u32,
        n_features: u32,
        n_bins: u32,
        n_replicas: u32,
        replica_stride: u32,
        subsample: f32,
        tree_seed: u32,
        odd_only: u32,
        packed: u32,
        bin_row_stride: u32,
    ) {
        let i = thread::index_1d().get();
        let nn = n as usize;
        if i >= nn {
            return;
        }
        if subsample < 1.0 && hash01(i as u32, tree_seed) >= subsample {
            return;
        }
        let grp = leaf[i] as usize;
        if odd_only == 1 && (grp & 1) == 0 {
            return;
        }
        let gi = g[i] as f16;
        let hi = h[i] as f16;
        let nf = n_features as usize;
        let nb = n_bins as usize;
        let rs = bin_row_stride as usize;
        let rbase = (thread::blockIdx_x() % n_replicas) as usize * replica_stride as usize;
        let row_base = grp * nf;
        let mut f = 0usize;
        while f < nf {
            let bin = if packed == 1 {
                let byte = bins[f * rs + (i >> 1)];
                (if i & 1 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }) as usize
            } else {
                bins[f * rs + i] as usize
            };
            let off = rbase + (row_base + f) * nb + bin;
            atomic_add_f16(hist_g, off, gi);
            atomic_add_f16(hist_h, off, hi);
            f += 1;
        }
    }

    /// Fold `n_replicas` f16 histogram copies (each `stride` long) into f32 outputs,
    /// accumulating in f32 to recover precision.
    #[kernel]
    #[cfg(feature = "f16-hist")]
    pub fn reduce2_f16to32(
        src_g: &[f16],
        src_h: &[f16],
        dst_g: &[f32],
        dst_h: &[f32],
        n_replicas: u32,
        stride: u32,
    ) {
        let i = thread::index_1d().get();
        let st = stride as usize;
        if i >= st {
            return;
        }
        let mut sg = 0.0f32;
        let mut sh = 0.0f32;
        let mut r = 0usize;
        while r < n_replicas as usize {
            sg += src_g[r * st + i] as f32;
            sh += src_h[r * st + i] as f32;
            r += 1;
        }
        unsafe {
            *(dst_g.as_ptr() as *mut f32).add(i) = sg;
            *(dst_h.as_ptr() as *mut f32).add(i) = sh;
        }
    }

    /// Fold `n_replicas` copies of two buffers (each `stride` long) into replica 0.
    #[kernel]
    pub fn reduce2(a: &[f32], b: &[f32], n_replicas: u32, stride: u32) {
        let i = thread::index_1d().get();
        let st = stride as usize;
        if i >= st {
            return;
        }
        let mut sa = a[i] as Acc;
        let mut sb = b[i] as Acc;
        let mut r = 1usize;
        while r < n_replicas as usize {
            sa += a[r * st + i] as Acc;
            sb += b[r * st + i] as Acc;
            r += 1;
        }
        unsafe {
            *(a.as_ptr() as *mut f32).add(i) = sa as f32;
            *(b.as_ptr() as *mut f32).add(i) = sb as f32;
        }
    }

    /// Set the first `n` elements of a `u32` buffer to zero.
    #[kernel]
    pub fn zero_u32(mut buf: DisjointSlice<u32>, n: u32) {
        let idx = thread::index_1d();
        if idx.get() >= n as usize {
            return;
        }
        if let Some(slot) = buf.get_mut(idx) {
            *slot = 0;
        }
    }

    /// SPIKE: shared-memory histogram via BlockAtomicF32. One block; each thread adds
    /// 1.0 into shared bin `bins[i]`, then writes the shared histogram to `out`. If this
    /// returns correct counts, cuda-oxide lowers `.cta` atomics to `atom.shared`.
    #[kernel]
    pub fn smem_spike(bins: &[u32], out: &[f32], n: u32, nbins: u32) {
        static mut SH: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        if tid < nbins as usize {
            unsafe {
                SH[tid] = 0.0;
            }
        }
        thread::sync_threads();
        let i = thread::index_1d().get();
        if i < n as usize {
            let b = bins[i] as usize;
            unsafe {
                let p = &SH[b] as *const f32 as *const BlockAtomicF32;
                (*p).fetch_add(1.0, AtomicOrdering::Relaxed);
            }
        }
        thread::sync_threads();
        if tid < nbins as usize {
            unsafe {
                *(out.as_ptr() as *mut f32).add(tid) = SH[tid];
            }
        }
    }

    /// Quantize features into bins via binary search over per-feature boundaries.
    /// `xcol` is column-major (`xcol[f*n+i]`); `boundaries` is `f*(n_bins-1)+j`.
    /// `bin = #{ boundaries[f][j] < v }`, so split `bin > t` == `v > boundaries[f][t]`.
    #[kernel]
    pub fn bin_features(
        xcol: &[f32],
        boundaries: &[f32],
        mut bins: DisjointSlice<u8>,
        n: u32,
        n_features: u32,
        n_bins: u32,
    ) {
        let idx = thread::index_1d();
        let id = idx.get();
        if id >= (n * n_features) as usize {
            return;
        }
        let nn = n as usize;
        let f = id / nn;
        let nb1 = (n_bins - 1) as usize;
        let v = xcol[id];
        let base = f * nb1;
        let mut lo = 0usize;
        let mut hi = nb1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if boundaries[base + mid] < v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let b = if lo > nb1 { nb1 } else { lo };
        if let Some(slot) = bins.get_mut(idx) {
            *slot = b as u8;
        }
    }

    /// Hessian-clamp step 1: accumulate sum(h) into `stats[0]` and sum(h^2) into `stats[1]`.
    #[kernel]
    pub fn hess_stats(h: &[f32], stats: &[f32], n: u32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let v = h[i];
        atomic_add_f32(stats, 0, v);
        atomic_add_f32(stats, 1, v * v);
    }

    /// Hessian-clamp step 2 (experimental): winsorize h to [mean - z*std, mean + z*std],
    /// with z = Phi^-1(1-p). Approximates clamping the lowest/highest p fraction (assuming
    /// roughly-normal hessians). Lower bound floored at `eps` so h stays positive.
    #[kernel]
    pub fn hess_clamp(mut h: DisjointSlice<f32>, stats: &[f32], n: u32, z: f32, eps: f32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let mean = stats[0] / n as f32;
        let var = stats[1] / n as f32 - mean * mean;
        let std = if var > 0.0 { var.sqrt() } else { 0.0 };
        let mut lo = mean - z * std;
        if lo < eps {
            lo = eps;
        }
        let hi = mean + z * std;
        if let Some(he) = h.get_mut(thread::index_1d()) {
            let v = *he;
            *he = if v < lo { lo } else if v > hi { hi } else { v };
        }
    }

    /// GOSS step 1: accumulate sum(|g|) into `stats[0]` (count is known = n).
    #[kernel]
    pub fn goss_stats(g: &[f32], stats: &[f32], n: u32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let v = g[i];
        atomic_add_f32(stats, 0, if v < 0.0 { -v } else { v });
    }

    /// GOSS step 2: turn mean(|g|) into a top-rate threshold assuming half-normal
    /// gradients: sigma = mean|g| / sqrt(2/pi); thresh = sigma * q, q = Phi^-1(1-a/2).
    #[kernel]
    pub fn goss_thresh(stats: &[f32], thresh: &[f32], n: u32, q: f32) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mean_abs = stats[0] / n as f32;
        let sigma = mean_abs / 0.7978845608f32;
        unsafe {
            *(thresh.as_ptr() as *mut f32) = sigma * q;
        }
    }

    /// GOSS step 3: reweight gradients/hessians in place. Rows with |g| >= thresh are
    /// kept (weight 1); a fraction `other_rate` of the rest are kept and up-weighted by
    /// `amplify = (1-top_rate)/other_rate`; the remainder are zeroed out (excluded).
    #[kernel]
    pub fn goss_apply(
        mut g: DisjointSlice<f32>,
        mut h: DisjointSlice<f32>,
        thresh: &[f32],
        amplify: f32,
        other_rate: f32,
        seed: u32,
        n: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let t = thresh[0];
        let mut w = 0.0f32;
        if let Some(ge) = g.get_mut(thread::index_1d()) {
            let ag = if *ge < 0.0 { -*ge } else { *ge };
            w = if ag >= t {
                1.0
            } else if hash01(i as u32, seed) < other_rate {
                amplify
            } else {
                0.0
            };
            *ge *= w;
        }
        if let Some(he) = h.get_mut(thread::index_1d()) {
            *he *= w;
        }
    }

    /// Pack column-major u8 bins (values 0..15) to 4-bit, 2 rows per byte:
    /// `packed[f*half + k] = bins[f*n + 2k] | (bins[f*n + 2k+1] << 4)`.
    #[kernel]
    pub fn pack_bins4(bins: &[u8], mut packed: DisjointSlice<u8>, n: u32, n_features: u32, half: u32) {
        let idx = thread::index_1d();
        let id = idx.get();
        if id >= (n_features * half) as usize {
            return;
        }
        let f = id / half as usize;
        let k = id % half as usize;
        let nn = n as usize;
        let lo = bins[f * nn + 2 * k] & 0x0F;
        let hi = if 2 * k + 1 < nn { bins[f * nn + 2 * k + 1] & 0x0F } else { 0 };
        if let Some(slot) = packed.get_mut(idx) {
            *slot = lo | (hi << 4);
        }
    }

    /// Squared-error gradient/hessian: `g = pred - y`, `h = 1`.
    #[kernel]
    pub fn grad_l2(
        pred: &[f32],
        y: &[f32],
        mut g: DisjointSlice<f32>,
        mut h: DisjointSlice<f32>,
        n: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        if let Some(ge) = g.get_mut(thread::index_1d()) {
            *ge = pred[i] - y[i];
        }
        if let Some(he) = h.get_mut(thread::index_1d()) {
            *he = 1.0;
        }
    }

    /// Logistic gradient/hessian: `p = sigmoid(pred)`, `g = p - y`, `h = max(p(1-p), eps)`.
    #[kernel]
    pub fn grad_logistic(
        pred: &[f32],
        y: &[f32],
        mut g: DisjointSlice<f32>,
        mut h: DisjointSlice<f32>,
        n: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let p = 1.0f32 / (1.0f32 + (-pred[i]).exp());
        if let Some(ge) = g.get_mut(thread::index_1d()) {
            *ge = p - y[i];
        }
        if let Some(he) = h.get_mut(thread::index_1d()) {
            let hv = p * (1.0f32 - p);
            *he = if hv < 1e-6 { 1e-6 } else { hv };
        }
    }

    /// Per-(group, feature, bin) gradient/hessian histograms with privatization by
    /// replication: each block accumulates into replica `blockIdx % n_replicas`,
    /// cutting global-atomic contention ~`n_replicas`x. Layout within a replica is
    /// `((group*F)+f)*B + bin`; replicas are `replica_stride = groups*F*B` apart.
    /// Call `reduce_replicas` afterwards to fold replicas into replica 0.
    #[kernel]
    pub fn build_hist(
        bins: &[u8],
        leaf: &[u32],
        g: &[f32],
        h: &[f32],
        hist_g: &[f32],
        hist_h: &[f32],
        n: u32,
        n_features: u32,
        n_bins: u32,
        n_replicas: u32,
        replica_stride: u32,
        subsample: f32,
        tree_seed: u32,
        odd_only: u32,
        packed: u32,
        bin_row_stride: u32,
    ) {
        let i = thread::index_1d().get();
        let nn = n as usize;
        if i >= nn {
            return;
        }
        // Stochastic row subsampling: deterministic per (row, tree).
        if subsample < 1.0 && hash01(i as u32, tree_seed) >= subsample {
            return;
        }
        let grp = leaf[i] as usize;
        // Histogram subtraction: at levels >= 1 only build the odd (right) children;
        // even children are derived as parent - odd.
        if odd_only == 1 && (grp & 1) == 0 {
            return;
        }
        let gi = g[i];
        let hi = h[i];
        let nf = n_features as usize;
        let nb = n_bins as usize;
        let rs = bin_row_stride as usize;
        let rbase = (thread::blockIdx_x() % n_replicas) as usize * replica_stride as usize;
        let row_base = grp * nf;
        let mut f = 0usize;
        while f < nf {
            // 4-bit packed (2 rows/byte) or plain u8 bin read.
            let bin = if packed == 1 {
                let byte = bins[f * rs + (i >> 1)];
                (if i & 1 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }) as usize
            } else {
                bins[f * rs + i] as usize
            };
            let off = rbase + (row_base + f) * nb + bin;
            atomic_add_f32(hist_g, off, gi);
            atomic_add_f32(hist_h, off, hi);
            f += 1;
        }
    }

    /// Derive even (left) child histograms by subtraction: for each parent group `g`
    /// (level d-1) and each (feature,bin) element, `cur[2g] = prev[g] - cur[2g+1]`.
    /// `prev`/`cur` are the reduced (replica-0) raw histograms. Only the odd children
    /// of `cur` were built directly; this fills the even ones.
    #[kernel]
    pub fn subtract_even(
        cur_g: &[f32],
        cur_h: &[f32],
        prev_g: &[f32],
        prev_h: &[f32],
        prev_groups: u32,
        n_features: u32,
        n_bins: u32,
    ) {
        let id = thread::index_1d().get();
        let fb = (n_features * n_bins) as usize;
        if id >= prev_groups as usize * fb {
            return;
        }
        let g = id / fb;
        let e = id % fb;
        let even = (2 * g) * fb + e;
        let odd = (2 * g + 1) * fb + e;
        let pg = prev_g[g * fb + e];
        let ph = prev_h[g * fb + e];
        unsafe {
            *(cur_g.as_ptr() as *mut f32).add(even) = pg - cur_g[odd];
            *(cur_h.as_ptr() as *mut f32).add(even) = ph - cur_h[odd];
        }
    }

    /// In-place inclusive prefix sum of the histograms along the bin axis, per
    /// (group, feature) segment of length `n_bins`. After this, `hist_*[base+t]`
    /// is the left-side cumulative sum and `hist_*[base+B-1]` the segment total,
    /// so `split_gain` evaluates each candidate in O(1).
    #[kernel]
    pub fn prefix_scan(hist_g: &[f32], hist_h: &[f32], n_features: u32, n_bins: u32, n_groups: u32) {
        let id = thread::index_1d().get();
        let nf = n_features as usize;
        let nb = n_bins as usize;
        if id >= (n_groups as usize) * nf {
            return;
        }
        let base = id * nb;
        let mut accg = 0.0f32;
        let mut acch = 0.0f32;
        let mut b = 0usize;
        while b < nb {
            accg += hist_g[base + b];
            acch += hist_h[base + b];
            unsafe {
                *(hist_g.as_ptr() as *mut f32).add(base + b) = accg;
                *(hist_h.as_ptr() as *mut f32).add(base + b) = acch;
            }
            b += 1;
        }
    }

    /// Total XGBoost-style gain for every (feature, threshold), summed over all
    /// current groups (the same split serves every group => oblivious tree). Reads
    /// the prefix-scanned histograms, so each candidate is O(n_groups).
    /// Writes `gains[f*B + t]`; the last bin is an invalid threshold (sentinel).
    #[kernel]
    pub fn split_gain(
        hist_g: &[f32],
        hist_h: &[f32],
        mut gains: DisjointSlice<f32>,
        n_features: u32,
        n_bins: u32,
        n_groups: u32,
        lambda: f32,
        min_child_h: f32,
        colsample: f32,
        tree_seed: u32,
        monotone: &[i32],
    ) {
        let idx = thread::index_1d();
        let id = idx.get();
        let nf = n_features as usize;
        let nb = n_bins as usize;
        if id >= nf * nb {
            return;
        }
        let f = id / nb;
        let t = id % nb;
        // Stochastic feature subsampling: drop columns not selected for this tree.
        let dropped = colsample < 1.0 && hash01(f as u32, tree_seed ^ 0x5bd1_e995) >= colsample;
        if dropped || t + 1 >= nb {
            if let Some(slot) = gains.get_mut(idx) {
                *slot = -1e30;
            }
            return;
        }

        let ng = n_groups as usize;
        // Accumulate in `Acc` (f32 by default, f64 with --features f64-hist). Storage
        // stays f32, so this only changes arithmetic precision, not memory.
        let lam = lambda as Acc;
        let mch = min_child_h as Acc;
        let mut total_gain = 0.0 as Acc;
        let mut feasible = false;
        let mut gl_tot = 0.0 as Acc;
        let mut hl_tot = 0.0 as Acc;
        let mut gr_tot = 0.0 as Acc;
        let mut hr_tot = 0.0 as Acc;
        let mut grp = 0usize;
        while grp < ng {
            let base = (grp * nf + f) * nb;
            // Self-cumulate over bins (histograms are kept raw for subtraction).
            let mut gl = 0.0 as Acc;
            let mut hl = 0.0 as Acc;
            let mut gt = 0.0 as Acc;
            let mut ht = 0.0 as Acc;
            let mut b = 0usize;
            while b < nb {
                let gv = hist_g[base + b] as Acc;
                let hv = hist_h[base + b] as Acc;
                gt += gv;
                ht += hv;
                if b <= t {
                    gl += gv;
                    hl += hv;
                }
                b += 1;
            }
            let gr = gt - gl;
            let hr = ht - hl;
            if hl >= mch && hr >= mch {
                feasible = true;
            }
            gl_tot += gl;
            hl_tot += hl;
            gr_tot += gr;
            hr_tot += hr;
            total_gain += gl * gl / (hl + lam) + gr * gr / (hr + lam) - gt * gt / (ht + lam);
            grp += 1;
        }
        // Monotonic constraint: child weight w = -G/(H+lambda) must respect direction.
        let m = monotone[f];
        if m != 0 {
            let wl = -gl_tot / (hl_tot + lam);
            let wr = -gr_tot / (hr_tot + lam);
            if (m > 0 && wl > wr) || (m < 0 && wl < wr) {
                feasible = false;
            }
        }
        if let Some(slot) = gains.get_mut(idx) {
            *slot = if feasible { total_gain as f32 } else { -1e30 };
        }
    }

    /// Single-block argmax over `gains` (length `F*B`). Writes the winning split's
    /// `feature` into `split_ft[0]`, `bin threshold` into `split_ft[1]`, and records
    /// it in the device-resident model at slot `k`: `feat[k]`, `thr[k]` (raw value
    /// looked up from `boundaries`). Launch with one block of 256 threads.
    #[kernel]
    pub fn argmax_split(
        gains: &[f32],
        boundaries: &[f32],
        feat: &[u32],
        thr: &[f32],
        split_ft: &[u32],
        fi_gain: &[f32],
        fi_count: &[f32],
        k: u32,
        n_features: u32,
        n_bins: u32,
    ) {
        static mut SG: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut SI: SharedArray<u32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let total = (n_features * n_bins) as usize;

        let mut best_g = -1e30f32;
        let mut best_i = 0u32;
        let mut i = tid;
        while i < total {
            let g = gains[i];
            if g > best_g {
                best_g = g;
                best_i = i as u32;
            }
            i += 256;
        }
        unsafe {
            SG[tid] = best_g;
            SI[tid] = best_i;
        }
        thread::sync_threads();

        let mut s = 128usize;
        while s > 0 {
            if tid < s {
                unsafe {
                    if SG[tid + s] > SG[tid] {
                        SG[tid] = SG[tid + s];
                        SI[tid] = SI[tid + s];
                    }
                }
            }
            thread::sync_threads();
            s /= 2;
        }

        if tid == 0 {
            let nb = n_bins as usize;
            let idx = unsafe { SI[0] } as usize;
            let bf = idx / nb;
            let bt = idx % nb;
            unsafe {
                let sp = split_ft.as_ptr() as *mut u32;
                *sp = bf as u32;
                *sp.add(1) = bt as u32;
                *(feat.as_ptr() as *mut u32).add(k as usize) = bf as u32;
                *(thr.as_ptr() as *mut f32).add(k as usize) =
                    boundaries[bf * (n_bins as usize - 1) + bt];
                // Accumulate feature importance (argmax launches are serial on the
                // stream, so a plain read-modify-write by thread 0 is race-free).
                let g = SG[0];
                if g > -1e29 {
                    let pg = (fi_gain.as_ptr() as *mut f32).add(bf);
                    *pg += g;
                    let pc = (fi_count.as_ptr() as *mut f32).add(bf);
                    *pc += 1.0;
                }
            }
        }
    }

    /// Append one oblivious split bit to every row's leaf index, reading the chosen
    /// `(feature, threshold)` from `split_ft` (set by `argmax_split`).
    #[kernel]
    pub fn apply_split(
        bins: &[u8],
        mut leaf: DisjointSlice<u32>,
        split_ft: &[u32],
        n: u32,
        packed: u32,
        bin_row_stride: u32,
    ) {
        let i = thread::index_1d().get();
        let nn = n as usize;
        if i >= nn {
            return;
        }
        let best_f = split_ft[0] as usize;
        let best_t = split_ft[1];
        let rs = bin_row_stride as usize;
        let bin = if packed == 1 {
            let byte = bins[best_f * rs + (i >> 1)];
            (if i & 1 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }) as u32
        } else {
            bins[best_f * rs + i] as u32
        };
        let bit = if bin > best_t { 1u32 } else { 0u32 };
        if let Some(slot) = leaf.get_mut(thread::index_1d()) {
            let old = *slot;
            *slot = old * 2 + bit;
        }
    }

    /// Per-leaf gradient/hessian sums (atomic), privatized by replication into
    /// `leaf_*[replica*n_leaves + leaf]`. Fold with `reduce_replicas(stride=n_leaves)`.
    #[kernel]
    pub fn leaf_hist(
        leaf: &[u32],
        g: &[f32],
        h: &[f32],
        leaf_g: &[f32],
        leaf_h: &[f32],
        leaf_g2: &[f32],
        n: u32,
        n_leaves: u32,
        n_replicas: u32,
        subsample: f32,
        tree_seed: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // Same row subset as build_hist so leaf values match the chosen splits.
        if subsample < 1.0 && hash01(i as u32, tree_seed) >= subsample {
            return;
        }
        let rbase = (thread::blockIdx_x() % n_replicas) as usize * n_leaves as usize;
        let l = rbase + leaf[i] as usize;
        let gi = g[i];
        atomic_add_f32(leaf_g, l, gi);
        atomic_add_f32(leaf_h, l, h[i]);
        atomic_add_f32(leaf_g2, l, gi * gi); // sum of g^2 for per-leaf gradient variance (PGBM)
    }

    /// Leaf values into the device model: `leafval[base + l] = -lr*G/(H+lambda)`.
    #[kernel]
    pub fn compute_leaf_values(
        leaf_g: &[f32],
        leaf_h: &[f32],
        leaf_g2: &[f32],
        leafval: &[f32],
        leafvar: &[f32],
        base: u32,
        lr: f32,
        lambda: f32,
        n_leaves: u32,
    ) {
        let l = thread::index_1d().get();
        if l >= n_leaves as usize {
            return;
        }
        let g = leaf_g[l];
        let cnt = leaf_h[l]; // = sample count for squared-error (h == 1)
        let val = -lr * g / (cnt + lambda);
        // PGBM: within-leaf gradient variance -> standard error of the leaf step.
        let var_g = if cnt > 1.0 {
            let m = g / cnt;
            let v = leaf_g2[l] / cnt - m * m;
            if v > 0.0 { v } else { 0.0 }
        } else {
            0.0
        };
        // Aleatoric (residual) variance contribution of this tree's leaf. Summed over
        // the forest this approximates the predictive variance (PGBM-style propagation).
        let leaf_var = lr * lr * var_g;
        unsafe {
            *(leafval.as_ptr() as *mut f32).add(base as usize + l) = val;
            *(leafvar.as_ptr() as *mut f32).add(base as usize + l) = leaf_var;
        }
    }

    /// PGBM inference: total predictive variance per row = sum of per-leaf variances
    /// across the forest (independent-tree approximation). Std = sqrt(out_var).
    #[kernel]
    pub fn predict_var(
        x: &[f32],
        feat: &[u32],
        thr: &[f32],
        leafvar: &[f32],
        mut out_var: DisjointSlice<f32>,
        n: u32,
        n_features: u32,
        depth: u32,
        n_trees: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let nf = n_features as usize;
        let d = depth as usize;
        let nleaves = 1usize << d;
        let mut acc = 0.0f32;
        let mut t = 0usize;
        while t < n_trees as usize {
            let mut leaf = 0usize;
            let mut dd = 0usize;
            while dd < d {
                let kk = t * d + dd;
                let bit = if x[i * nf + feat[kk] as usize] > thr[kk] { 1usize } else { 0usize };
                leaf = leaf * 2 + bit;
                dd += 1;
            }
            acc += leafvar[t * nleaves + leaf];
            t += 1;
        }
        if let Some(oe) = out_var.get_mut(thread::index_1d()) {
            *oe = acc;
        }
    }

    /// Add this tree's leaf values (at `leafval[base + leaf[i]]`) into running margins.
    #[kernel]
    pub fn update_pred(
        leaf: &[u32],
        leafval: &[f32],
        base: u32,
        mut pred: DisjointSlice<f32>,
        n: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let l = leaf[i] as usize;
        if let Some(pe) = pred.get_mut(thread::index_1d()) {
            *pe += leafval[base as usize + l];
        }
    }

    /// Batch inference over the whole forest from raw (unbinned) row-major features.
    #[kernel]
    pub fn predict(
        x: &[f32],
        feat: &[u32],
        thr: &[f32],
        leafval: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        n_features: u32,
        depth: u32,
        n_trees: u32,
        base: f32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let nf = n_features as usize;
        let d = depth as usize;
        let nleaves = 1usize << d;
        let mut acc = base;
        let mut t = 0usize;
        while t < n_trees as usize {
            let mut leaf = 0usize;
            let mut dd = 0usize;
            while dd < d {
                let kk = t * d + dd;
                let fcol = feat[kk] as usize;
                let bit = if x[i * nf + fcol] > thr[kk] { 1usize } else { 0usize };
                leaf = leaf * 2 + bit;
                dd += 1;
            }
            acc += leafval[t * nleaves + leaf];
            t += 1;
        }
        if let Some(oe) = out.get_mut(thread::index_1d()) {
            *oe = acc;
        }
    }
}
