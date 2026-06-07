//! Dataset loading and per-feature bin-boundary estimation.
//!
//! The Python harness (`scripts/gen_data.py`) writes flat little-endian `f32`
//! blobs plus a tiny `meta.json`. Features are stored row-major
//! (`x[i * n_features + f]`), matching scikit-learn's default memory order.

use std::fs;
use std::path::Path;

pub struct Dataset {
    pub n_train: usize,
    pub n_test: usize,
    pub n_features: usize,
    /// Row-major train features, length `n_train * n_features`.
    pub x_train: Vec<f32>,
    pub y_train: Vec<f32>,
    /// Row-major test features, length `n_test * n_features`.
    pub x_test: Vec<f32>,
    pub y_test: Vec<f32>,
}

fn read_f32_blob(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len() % 4 == 0, "{} not f32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Minimal extractor for `"key": <integer>` out of the flat meta.json.
fn meta_int(text: &str, key: &str) -> usize {
    let pat = format!("\"{key}\"");
    let start = text
        .find(&pat)
        .unwrap_or_else(|| panic!("meta.json missing {key}"));
    let after = &text[start + pat.len()..];
    let digits: String = after
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits
        .parse()
        .unwrap_or_else(|e| panic!("bad int for {key} (got {digits:?}): {e}"))
}

impl Dataset {
    pub fn load(dir: &str) -> Self {
        let dir = Path::new(dir);
        let meta = fs::read_to_string(dir.join("meta.json")).expect("read meta.json");
        let n_train = meta_int(&meta, "n_train");
        let n_test = meta_int(&meta, "n_test");
        let n_features = meta_int(&meta, "n_features");

        let x_train = read_f32_blob(&dir.join("x_train.bin"));
        let y_train = read_f32_blob(&dir.join("y_train.bin"));
        let x_test = read_f32_blob(&dir.join("x_test.bin"));
        let y_test = read_f32_blob(&dir.join("y_test.bin"));

        assert_eq!(x_train.len(), n_train * n_features, "x_train size mismatch");
        assert_eq!(y_train.len(), n_train, "y_train size mismatch");
        assert_eq!(x_test.len(), n_test * n_features, "x_test size mismatch");
        assert_eq!(y_test.len(), n_test, "y_test size mismatch");

        Self {
            n_train,
            n_test,
            n_features,
            x_train,
            y_train,
            x_test,
            y_test,
        }
    }
}

/// Estimate `n_bins - 1` quantile boundaries per feature from a strided sample.
///
/// Returns a flat buffer `boundaries[f * (n_bins - 1) + j]`, ascending in `j`.
/// Mirrors how LightGBM/XGBoost pick split candidates from quantiles.
/// With `unique = true`, boundaries are placed at midpoints between *distinct*
/// sampled values (deduplicated), and unused slots are padded with `+inf`. This
/// stops low-cardinality features (e.g. categoricals) from wasting bins on repeated
/// thresholds — every bin then carries a distinct value range.
pub fn compute_boundaries(
    x_train: &[f32],
    n: usize,
    n_features: usize,
    n_bins: usize,
    sample: usize,
    unique: bool,
) -> Vec<f32> {
    let n_bound = n_bins - 1;
    let mut boundaries = vec![0.0f32; n_features * n_bound];
    let stride = (n / sample.max(1)).max(1);

    let mut scratch: Vec<f32> = Vec::with_capacity(n / stride + 1);
    for f in 0..n_features {
        scratch.clear();
        let mut i = 0;
        while i < n {
            scratch.push(x_train[i * n_features + f]);
            i += stride;
        }
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let base = f * n_bound;

        if unique {
            // Distinct sorted values -> midpoint candidate splits between them.
            let mut uniq: Vec<f32> = Vec::with_capacity(scratch.len());
            for &v in &scratch {
                if uniq.last().map_or(true, |&u| u != v) {
                    uniq.push(v);
                }
            }
            let cand: Vec<f32> = uniq.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
            for j in 0..n_bound {
                boundaries[base + j] = if cand.is_empty() {
                    f32::INFINITY
                } else if cand.len() <= n_bound {
                    // Use every distinct split; pad the rest so those bins stay empty.
                    *cand.get(j).unwrap_or(&f32::INFINITY)
                } else {
                    // More distinct values than bins: subsample candidates evenly.
                    let pos = ((j + 1) * cand.len()) / (n_bound + 1);
                    cand[pos.min(cand.len() - 1)]
                };
            }
        } else {
            let m = scratch.len();
            for j in 0..n_bound {
                let q = (j + 1) as f64 / n_bins as f64;
                let pos = (q * (m - 1) as f64).round() as usize;
                boundaries[base + j] = scratch[pos.min(m - 1)];
            }
            // Keep boundaries monotone even with heavy ties.
            for j in 1..n_bound {
                if boundaries[base + j] <= boundaries[base + j - 1] {
                    boundaries[base + j] = boundaries[base + j - 1];
                }
            }
        }
    }
    boundaries
}
