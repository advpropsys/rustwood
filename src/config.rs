//! Training configuration and command-line parsing.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Objective {
    /// Binary logistic; report AUC / logloss.
    Logistic,
    /// Squared error; report RMSE.
    SquaredError,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub data_dir: String,
    pub n_trees: usize,
    pub depth: usize,
    pub learning_rate: f32,
    pub n_bins: usize,
    pub lambda: f32,
    pub min_child_h: f32,
    /// Fraction of rows sampled per tree (stochastic boosting regularizer).
    pub subsample: f32,
    /// Fraction of features considered per tree.
    pub colsample: f32,
    /// Place bin boundaries at midpoints of distinct values (good for categoricals).
    pub unique_bins: bool,
    /// Column indices to treat as categorical (out-of-fold target encoding).
    pub categorical: Vec<usize>,
    /// Smoothing strength for target encoding.
    pub te_smoothing: f32,
    /// Hashing-trick buckets for categoricals (0 = use raw codes / no hashing).
    pub cat_hash_buckets: u64,
    /// Per-feature monotone direction (+1/0/-1); empty = none.
    pub monotone: Vec<i32>,
    /// Use histogram subtraction (build odd children, derive even). Faster; tiny f32 drift.
    pub subtract: bool,
    /// GOSS top-gradient keep rate (a); 0 disables GOSS.
    pub goss_top: f32,
    /// GOSS sampling rate for small-gradient rows (b).
    pub goss_other: f32,
    /// Experimental: winsorize hessians at the p / (1-p) tails (0 = off).
    pub clamp_hessian: f32,
    /// Pack feature bins to 4-bit (2/byte) when n_bins<=16 to cut histogram bin-read
    /// bandwidth (sm100+ intent; the packing itself is integer/arch-agnostic).
    pub pack4: bool,
    /// PGBM mode: also produce per-row predictive variance / intervals (regression).
    pub pgbm: bool,
    /// Use shared-memory privatized histograms when groups*n_bins<=4096 (faster).
    pub smem_hist: bool,
    /// Accumulate histograms with f16 atomics (atom.add.noftz.f16), fold to f32 in reduce.
    pub f16_hist: bool,
    /// Accumulate histograms with fixed-point i32 atomics (fastest atomic path).
    pub int_hist: bool,
    /// Fixed-point scale for --int-hist (gradient*scale rounded to i32).
    pub int_scale: f32,
    pub objective: Objective,
    /// Rows sampled (strided) to estimate per-feature bin boundaries.
    pub bin_sample: usize,
    pub gpu: usize,
    /// Histogram privatization factor (global accumulator copies) to cut atomic
    /// contention. Capped by a memory budget at runtime.
    pub replicas: usize,
    /// If non-empty, run a profiled fit and write a per-kernel ns breakdown here.
    pub profile_out: String,
    /// If true, after training run a per-batch end-to-end inference latency sweep.
    pub latency_bench: bool,
    /// If true, run a device-resident peak-throughput sweep up to the VRAM ceiling.
    pub throughput_bench: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: "data".to_string(),
            n_trees: 500,
            depth: 6,
            learning_rate: 0.1,
            n_bins: 256,
            lambda: 1.0,
            min_child_h: 1e-3,
            subsample: 1.0,
            colsample: 1.0,
            unique_bins: false,
            categorical: Vec::new(),
            te_smoothing: 10.0,
            cat_hash_buckets: 0,
            monotone: Vec::new(),
            subtract: true,
            goss_top: 0.0,
            goss_other: 0.1,
            clamp_hessian: 0.0,
            pack4: false,
            pgbm: false,
            smem_hist: false,
            f16_hist: false,
            int_hist: false,
            int_scale: 16384.0,
            objective: Objective::Logistic,
            bin_sample: 200_000,
            gpu: 0,
            replicas: 64,
            profile_out: String::new(),
            latency_bench: false,
            throughput_bench: false,
        }
    }
}

impl Config {
    /// Parse `--key value` flags; unknown keys abort with a usage message.
    pub fn from_args() -> Self {
        let mut cfg = Config::default();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let key = args[i].as_str();
            let val = || {
                args.get(i + 1)
                    .unwrap_or_else(|| panic!("missing value for {key}"))
            };
            match key {
                "--data" => cfg.data_dir = val().clone(),
                "--trees" => cfg.n_trees = val().parse().unwrap(),
                "--depth" => cfg.depth = val().parse().unwrap(),
                "--lr" => cfg.learning_rate = val().parse().unwrap(),
                "--bins" => cfg.n_bins = val().parse().unwrap(),
                "--lambda" => cfg.lambda = val().parse().unwrap(),
                "--min-child" => cfg.min_child_h = val().parse().unwrap(),
                "--subsample" => cfg.subsample = val().parse().unwrap(),
                "--colsample" => cfg.colsample = val().parse().unwrap(),
                "--unique-bins" => cfg.unique_bins = matches!(val().as_str(), "1" | "true"),
                "--te-smoothing" => cfg.te_smoothing = val().parse().unwrap(),
                "--cat-hash-buckets" => cfg.cat_hash_buckets = val().parse().unwrap(),
                "--subtract" => cfg.subtract = matches!(val().as_str(), "1" | "true"),
                "--goss-top" => cfg.goss_top = val().parse().unwrap(),
                "--goss-other" => cfg.goss_other = val().parse().unwrap(),
                "--clamp-hessian" => cfg.clamp_hessian = val().parse().unwrap(),
                "--pack4" => cfg.pack4 = matches!(val().as_str(), "1" | "true"),
                "--pgbm" => cfg.pgbm = matches!(val().as_str(), "1" | "true"),
                "--smem-hist" => cfg.smem_hist = matches!(val().as_str(), "1" | "true"),
                "--f16-hist" => cfg.f16_hist = matches!(val().as_str(), "1" | "true"),
                "--int-hist" => cfg.int_hist = matches!(val().as_str(), "1" | "true"),
                "--int-scale" => cfg.int_scale = val().parse().unwrap(),
                "--monotone" => {
                    cfg.monotone = val()
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse().expect("bad monotone value"))
                        .collect()
                }
                "--categorical" => {
                    cfg.categorical = val()
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse().expect("bad categorical index"))
                        .collect()
                }
                "--bin-sample" => cfg.bin_sample = val().parse().unwrap(),
                "--gpu" => cfg.gpu = val().parse().unwrap(),
                "--replicas" => cfg.replicas = val().parse().unwrap(),
                "--profile-out" => cfg.profile_out = val().clone(),
                // Takes a value (1/true) to keep the simple key-value parser intact.
                "--latency-bench" => cfg.latency_bench = matches!(val().as_str(), "1" | "true"),
                "--throughput-bench" => cfg.throughput_bench = matches!(val().as_str(), "1" | "true"),
                "--objective" => {
                    cfg.objective = match val().as_str() {
                        "logistic" | "binary" => Objective::Logistic,
                        "l2" | "rmse" | "reg" => Objective::SquaredError,
                        other => panic!("unknown objective {other}"),
                    }
                }
                other => panic!("unknown flag {other}"),
            }
            i += 2;
        }
        assert!(cfg.n_bins >= 2 && cfg.n_bins <= 256, "n_bins must be in 2..=256");
        assert!(cfg.depth >= 1 && cfg.depth <= 12, "depth must be in 1..=12");
        cfg
    }
}
