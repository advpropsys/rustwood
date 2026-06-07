//! Categorical feature handling via smoothed target (mean) encoding.
//!
//! Integer category codes carry no meaningful order, so quantile-binning them as
//! numbers is near-useless. Target encoding replaces each code with a smoothed mean
//! of the label for that category, giving the feature an order aligned with the
//! target — after which ordinary numeric binning is informative.
//!
//! Training rows are encoded **out-of-fold** (K-fold): a row's value uses category
//! statistics computed from the *other* folds only, which avoids the target leakage
//! that plain full-data target encoding would introduce. The test set uses the
//! full-train statistics. This mirrors the idea behind CatBoost's ordered statistics.

use std::collections::HashMap;

/// Per-categorical-feature map from integer code -> full-train encoded value.
pub type Encoders = Vec<HashMap<i64, f32>>;

#[inline]
fn code(v: f32) -> i64 {
    v.round() as i64
}

/// Map a raw category code to a key. With `hash_buckets > 0`, apply the hashing trick
/// (collapse the code into one of `hash_buckets` buckets) to bound cardinality before
/// target encoding — useful for very-high-cardinality features.
#[inline]
fn key_of(v: f32, hash_buckets: u64) -> i64 {
    let c = code(v);
    if hash_buckets == 0 {
        return c;
    }
    let mut x = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % hash_buckets) as i64
}

/// Build encoders from train data AND transform `x` (row-major) in place using
/// out-of-fold statistics. Returns `(encoders, global_mean)` for encoding test data.
pub fn fit_transform_train(
    x: &mut [f32],
    n: usize,
    n_features: usize,
    y: &[f32],
    categorical: &[usize],
    smoothing: f32,
    k_folds: usize,
    hash_buckets: u64,
) -> (Encoders, f32) {
    let k = k_folds.max(2);
    let a = smoothing as f64;
    let gmean = (y.iter().map(|&v| v as f64).sum::<f64>() / n as f64) as f32;
    let gm = gmean as f64;

    let mut encoders: Encoders = Vec::with_capacity(categorical.len());
    for &c in categorical {
        // Per category: (sum, count) for each fold.
        let mut per: HashMap<i64, Vec<(f64, u64)>> = HashMap::new();
        for i in 0..n {
            let key = key_of(x[i * n_features + c], hash_buckets);
            let folds = per.entry(key).or_insert_with(|| vec![(0.0, 0); k]);
            let f = i % k;
            folds[f].0 += y[i] as f64;
            folds[f].1 += 1;
        }

        // Full-train map for the test set.
        let mut full: HashMap<i64, f32> = HashMap::with_capacity(per.len());
        for (key, folds) in &per {
            let s: f64 = folds.iter().map(|p| p.0).sum();
            let cnt: u64 = folds.iter().map(|p| p.1).sum();
            full.insert(*key, ((s + a * gm) / (cnt as f64 + a)) as f32);
        }

        // Out-of-fold transform of the training column.
        for i in 0..n {
            let key = key_of(x[i * n_features + c], hash_buckets);
            let folds = &per[&key];
            let f = i % k;
            let s: f64 = folds.iter().map(|p| p.0).sum::<f64>() - folds[f].0;
            let cnt: f64 = folds.iter().map(|p| p.1).sum::<u64>() as f64 - folds[f].1 as f64;
            x[i * n_features + c] = if cnt > 0.0 {
                ((s + a * gm) / (cnt + a)) as f32
            } else {
                gmean
            };
        }
        encoders.push(full);
    }
    (encoders, gmean)
}

/// Apply previously-fit encoders to new data (row-major), in place. Unseen codes
/// fall back to the global train mean.
pub fn transform(
    x: &mut [f32],
    n: usize,
    n_features: usize,
    categorical: &[usize],
    encoders: &Encoders,
    gmean: f32,
    hash_buckets: u64,
) {
    for (ci, &c) in categorical.iter().enumerate() {
        let map = &encoders[ci];
        for i in 0..n {
            let key = key_of(x[i * n_features + c], hash_buckets);
            x[i * n_features + c] = *map.get(&key).unwrap_or(&gmean);
        }
    }
}
