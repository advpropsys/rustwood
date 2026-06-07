//! Evaluation metrics computed on the host from raw model scores.

/// ROC AUC via the rank-sum (Mann–Whitney U) identity, with average ranks for ties.
pub fn auc(scores: &[f32], labels: &[f32]) -> f64 {
    let n = scores.len();
    assert_eq!(n, labels.len());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Assign average ranks (1-based) to tied score groups.
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        let avg = ((i + 1) + j) as f64 / 2.0; // mean of ranks i+1..=j
        for k in i..j {
            ranks[order[k]] = avg;
        }
        i = j;
    }

    let mut sum_pos_ranks = 0.0f64;
    let mut n_pos = 0.0f64;
    for idx in 0..n {
        if labels[idx] > 0.5 {
            sum_pos_ranks += ranks[idx];
            n_pos += 1.0;
        }
    }
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Binary log-loss from raw margins (applies the logistic link internally).
pub fn logloss(scores: &[f32], labels: &[f32]) -> f64 {
    const EPS: f64 = 1e-7;
    let mut acc = 0.0f64;
    for (&s, &y) in scores.iter().zip(labels) {
        let p = (1.0 / (1.0 + (-s as f64).exp())).clamp(EPS, 1.0 - EPS);
        acc += -(y as f64 * p.ln() + (1.0 - y as f64) * (1.0 - p).ln());
    }
    acc / scores.len() as f64
}

/// Root mean squared error between raw scores and targets.
pub fn rmse(scores: &[f32], labels: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for (&s, &y) in scores.iter().zip(labels) {
        let d = s as f64 - y as f64;
        acc += d * d;
    }
    (acc / scores.len() as f64).sqrt()
}

/// Mean absolute error between raw scores and targets.
pub fn mae(scores: &[f32], labels: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for (&s, &y) in scores.iter().zip(labels) {
        acc += (s as f64 - y as f64).abs();
    }
    acc / scores.len() as f64
}

/// Coefficient of determination R² = 1 − SSE/SST.
pub fn r2(scores: &[f32], labels: &[f32]) -> f64 {
    let n = labels.len() as f64;
    let mean = labels.iter().map(|&y| y as f64).sum::<f64>() / n;
    let mut sse = 0.0f64;
    let mut sst = 0.0f64;
    for (&s, &y) in scores.iter().zip(labels) {
        let yd = y as f64;
        sse += (s as f64 - yd).powi(2);
        sst += (yd - mean).powi(2);
    }
    if sst == 0.0 { f64::NAN } else { 1.0 - sse / sst }
}

/// Classification accuracy at a 0.0 margin threshold (i.e. p = 0.5).
pub fn accuracy(scores: &[f32], labels: &[f32]) -> f64 {
    let mut correct = 0usize;
    for (&s, &y) in scores.iter().zip(labels) {
        let pred = if s > 0.0 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.5 {
            correct += 1;
        }
    }
    correct as f64 / scores.len() as f64
}
