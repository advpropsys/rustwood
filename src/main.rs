//! `rustwood` — histogram-based oblivious-tree gradient boosting on NVIDIA B300
//! (sm_103), with every GPU kernel written in pure Rust and compiled to PTX by
//! NVlabs `cuda-oxide`.
//!
//! Run (from the cuda-oxide repo root):
//!   cargo oxide run rustwood --arch sm_103 --release -- \
//!       --data crates/rustc-codegen-cuda/examples/rustwood/data \
//!       --trees 500 --depth 6 --lr 0.1 --bins 256 --objective logistic

// The f16 histogram path needs the nightly `f16` type + an f16 atomic not in stock
// cuda-oxide; both are opt-in via `--features f16-hist`.
#![cfg_attr(feature = "f16-hist", feature(f16))]

mod booster;
mod cpu;
mod config;
mod data;
mod encoding;
#[cfg(feature = "gpu")]
mod gpu_kernels;
mod metrics;

use booster::Booster;
use config::{Config, Device, Objective};
use data::Dataset;

/// Validate that shared-memory BlockAtomicF32 works (lowers to `atom.shared`).
#[cfg(feature = "gpu")]
fn smem_spike_test() {
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = gpu_kernels::kernels::load(&ctx).expect("load");
    let nbins = 16u32;
    let n = 256usize;
    // Deterministic bins; expected count per bin = n/nbins.
    let bins: Vec<u32> = (0..n).map(|i| (i as u32) % nbins).collect();
    let bins_dev = DeviceBuffer::from_host(&stream, &bins).expect("bins");
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, nbins as usize).expect("out");
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    module.smem_spike(&stream, cfg, &bins_dev, &out_dev, n as u32, nbins).expect("launch");
    let got = out_dev.to_host_vec(&stream).expect("copy");
    let expected = (n as f32) / (nbins as f32);
    let ok = got.iter().all(|&c| (c - expected).abs() < 0.5);
    println!("smem_spike: counts={got:?} expected={expected}");
    println!("shared-memory atomics: {}", if ok { "WORK ✓" } else { "BROKEN ✗" });
}

#[cfg(not(feature = "gpu"))]
fn smem_spike_test() {
    eprintln!("--smem-test requires a binary built with the 'gpu' feature");
    std::process::exit(2);
}

fn main() {
    if std::env::args().any(|a| a == "--smem-test") {
        smem_spike_test();
        return;
    }
    #[cfg(feature = "gpu")]
    let cfg = Config::from_args();
    #[cfg(not(feature = "gpu"))]
    let mut cfg = Config::from_args();
    #[cfg(not(feature = "gpu"))]
    if !cfg.load_model.is_empty() {
        cfg.device = Device::Cpu;
    }
    if cfg.serve {
        serve(cfg.gpu);
        return;
    }

    #[cfg(not(feature = "gpu"))]
    if cfg.device == Device::Gpu && cfg.load_model.is_empty() {
        eprintln!("this binary was built without the 'gpu' feature; use --device cpu");
        std::process::exit(2);
    }

    let ds = Dataset::load(&cfg.data_dir);
    println!("=== rustwood: oblivious-tree GBDT ({:?}) ===", cfg.device);
    println!(
        "data: train={} test={} features={}",
        ds.n_train, ds.n_test, ds.n_features
    );
    println!(
        "params: trees={} depth={} lr={} bins={} lambda={} objective={:?}",
        cfg.n_trees, cfg.depth, cfg.learning_rate, cfg.n_bins, cfg.lambda, cfg.objective
    );

    let objective = cfg.objective;
    let profile_out = cfg.profile_out.clone();
    let dump_pred = cfg.dump_pred.clone();
    #[cfg(feature = "gpu")]
    let pgbm = cfg.pgbm && cfg.device == Device::Gpu;
    let latency_bench = cfg.latency_bench && cfg.device == Device::Gpu;
    let throughput_bench = cfg.throughput_bench && cfg.device == Device::Gpu;
    #[cfg(not(feature = "gpu"))]
    if latency_bench || throughput_bench || cfg.pgbm {
        eprintln!("GPU-only options require a binary built with the 'gpu' feature");
        std::process::exit(2);
    }
    let device = cfg.device;
    let save_model = cfg.save_model.clone();

    // --load-model: load a .rwood model and predict; no training, no GPU.
    if !cfg.load_model.is_empty() {
        let t0 = std::time::Instant::now();
        let booster = Booster::load_model(&cfg.load_model).expect("load .rwood model");
        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let scores = booster.predict_host(&ds.x_test, ds.n_test);
        let pred_ms = t1.elapsed().as_secs_f64() * 1e3;
        if !cfg.dump_pred.is_empty() {
            let mut bytes = Vec::with_capacity(scores.len() * 4);
            for &s in &scores {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            std::fs::write(&cfg.dump_pred, &bytes).expect("write dump-pred");
        }
        println!("loaded {} in {load_ms:.3} ms; predicted {} rows in {pred_ms:.3} ms",
            cfg.load_model, ds.n_test);
        let line = match booster.config().objective {
            Objective::Logistic => format!(
                "RESULT objective=logistic load_ms={load_ms:.4} pred_ms={pred_ms:.4} auc={:.6} logloss={:.6} accuracy={:.6}",
                metrics::auc(&scores, &ds.y_test), metrics::logloss(&scores, &ds.y_test),
                metrics::accuracy(&scores, &ds.y_test)),
            Objective::SquaredError => format!(
                "RESULT objective=l2 load_ms={load_ms:.4} pred_ms={pred_ms:.4} rmse={:.6} mae={:.6} r2={:.6}",
                metrics::rmse(&scores, &ds.y_test), metrics::mae(&scores, &ds.y_test),
                metrics::r2(&scores, &ds.y_test)),
        };
        println!("\n{line}");
        return;
    }

    let mut booster = Booster::new(cfg).expect("init CUDA context / load PTX module");

    let train_time = if profile_out.is_empty() {
        booster.fit(&ds).expect("training failed")
    } else {
        let prof = booster.fit_profiled(&ds).expect("profiled training failed");
        let mut body = String::from("{\n  \"total_ns\": ");
        body.push_str(&prof.total.as_nanos().to_string());
        body.push_str(",\n  \"categories\": {\n");
        for (i, (name, ns)) in prof.per_category_ns.iter().enumerate() {
            let comma = if i + 1 < prof.per_category_ns.len() { "," } else { "" };
            body.push_str(&format!("    \"{name}\": {ns}{comma}\n"));
        }
        body.push_str("  }\n}\n");
        std::fs::write(&profile_out, body).expect("write profile");
        println!("wrote kernel profile -> {profile_out}");
        prof.total
    };

    if !save_model.is_empty() {
        let t = std::time::Instant::now();
        booster.save_model(&save_model).expect("save .rwood model");
        let sz = std::fs::metadata(&save_model).map(|m| m.len()).unwrap_or(0);
        println!("saved model -> {save_model} ({sz} bytes, {:.3} ms)", t.elapsed().as_secs_f64() * 1e3);
    }

    #[cfg(feature = "gpu")]
    if latency_bench {
        let batches = [1usize, 8, 64, 512, 4096, 32768, 262144];
        let res = booster
            .bench_latency(&ds.x_test, ds.n_test, &batches, 300)
            .expect("latency bench failed");
        println!("\n--- inference latency (GPU end-to-end, model resident) ---");
        for (b, per_call_ns) in &res {
            println!(
                "LATENCY batch={b} per_call_ns={per_call_ns:.1} per_row_ns={:.3}",
                per_call_ns / *b as f64
            );
        }

        // EXPERIMENTAL fast GPU path: reused buffers + pinned host + spin-sync.
        let fast = booster
            .bench_latency_fast(&ds.x_test, ds.n_test, &batches, 300)
            .expect("fast latency bench failed");
        println!("\n--- inference latency (GPU fast path: reuse+pinned+spin-sync) ---");
        for (b, per_call_ns) in &fast {
            println!(
                "LATENCY_FAST batch={b} per_call_ns={per_call_ns:.1} per_row_ns={:.3}",
                per_call_ns / *b as f64
            );
        }

        // CPU path: same batch sweep, host-side oblivious traversal.
        println!("\n--- inference latency (CPU, host oblivious traversal) ---");
        let mut scratch = vec![0.0f32; ds.n_test];
        for &(b, _) in &res {
            for _ in 0..5 {
                booster.predict_cpu(&ds.x_test, b, &mut scratch);
            }
            let reps = if b <= 64 { 5000 } else { 200 };
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                booster.predict_cpu(&ds.x_test, b, &mut scratch);
            }
            let per_call = t0.elapsed().as_nanos() as f64 / reps as f64;
            println!(
                "LATENCY_CPU batch={b} per_call_ns={per_call:.1} per_row_ns={:.3}",
                per_call / b as f64
            );
        }
    }

    #[cfg(feature = "gpu")]
    if throughput_bench {
        let rows = [1_000_000usize, 10_000_000, 50_000_000, 100_000_000, 250_000_000,
                    500_000_000, 1_000_000_000, 2_000_000_000];
        let res = booster.bench_throughput(&rows, 10).expect("throughput bench failed");
        println!("\n--- peak inference throughput (device-resident) ---");
        for (n, ns_per_row, ok) in &res {
            if *ok {
                println!(
                    "THRU rows={n} ns_per_row={ns_per_row:.4} rows_per_s={:.3e} input_GiB={:.2}",
                    1e9 / ns_per_row,
                    (*n as f64 * ds.n_features as f64 * 4.0) / (1u64 << 30) as f64
                );
            } else {
                println!("THRU rows={n} ALLOCATION_FAILED (VRAM/index ceiling)");
            }
        }
    }

    let (scores, infer_time) = if device == Device::Cpu {
        let mut s = vec![0.0f32; ds.n_test];
        let t0 = std::time::Instant::now();
        booster.predict_cpu(&ds.x_test, ds.n_test, &mut s);
        (s, t0.elapsed())
    } else {
        #[cfg(feature = "gpu")]
        {
            booster.predict(&ds.x_test, ds.n_test).expect("prediction failed")
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("GPU prediction is unavailable without the 'gpu' feature")
        }
    };

    // GPU runs: time the host oblivious traversal and verify it matches the GPU kernel
    // bit-for-bit (same forest, same f32 accumulation).
    if device == Device::Gpu {
        let mut cpu = vec![0.0f32; ds.n_test];
        let t0 = std::time::Instant::now();
        booster.predict_cpu(&ds.x_test, ds.n_test, &mut cpu);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        let maxdiff = scores.iter().zip(&cpu).map(|(g, c)| (g - c).abs()).fold(0.0f32, f32::max);
        println!("CPU_PRED_MS={cpu_ms:.4} CPU_GPU_MAXDIFF={maxdiff:.3e}");
    }

    // Persist raw per-row test predictions for external callers.
    // For SquaredError these are the regression outputs; for Logistic they are raw margins
    // (apply the logistic link downstream to obtain probabilities).
    if !dump_pred.is_empty() {
        let mut bytes = Vec::with_capacity(scores.len() * 4);
        for &s in &scores {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(&dump_pred, &bytes).expect("write dump-pred");
        println!("wrote {} test predictions -> {dump_pred}", scores.len());
    }

    let cfg = booster.config();
    let train_s = train_time.as_secs_f64();
    let cells = ds.n_train as f64 * cfg.n_trees as f64;
    println!("\n--- timing ---");
    println!("train ({device:?})      : {train_s:.3} s");
    println!("  throughput     : {:.2} M rows*trees / s", cells / train_s / 1e6);
    println!("  per-tree        : {:.3} ms", train_s * 1e3 / cfg.n_trees as f64);
    println!(
        "inference        : {:.3} ms for {} rows ({:.1} M rows/s)",
        infer_time.as_secs_f64() * 1e3,
        ds.n_test,
        ds.n_test as f64 / infer_time.as_secs_f64() / 1e6
    );

    println!("\n--- test metrics ---");
    let pred_ms = infer_time.as_secs_f64() * 1e3;
    let result_line = match objective {
        Objective::Logistic => {
            let auc = metrics::auc(&scores, &ds.y_test);
            let ll = metrics::logloss(&scores, &ds.y_test);
            let acc = metrics::accuracy(&scores, &ds.y_test);
            println!("AUC      : {auc:.6}");
            println!("logloss  : {ll:.6}");
            println!("accuracy : {acc:.6}");
            format!("RESULT objective=logistic train_s={train_s:.6} pred_ms={pred_ms:.6} auc={auc:.6} logloss={ll:.6} accuracy={acc:.6}")
        }
        Objective::SquaredError => {
            let rmse = metrics::rmse(&scores, &ds.y_test);
            let mae = metrics::mae(&scores, &ds.y_test);
            let r2 = metrics::r2(&scores, &ds.y_test);
            println!("RMSE     : {rmse:.6}");
            println!("MAE      : {mae:.6}");
            println!("R2       : {r2:.6}");
            format!("RESULT objective=l2 train_s={train_s:.6} pred_ms={pred_ms:.6} rmse={rmse:.6} mae={mae:.6} r2={r2:.6}")
        }
    };
    println!("\n{result_line}");

    #[cfg(feature = "gpu")]
    if pgbm {
        let std = booster.predict_std(&ds.x_test, ds.n_test).expect("pgbm variance failed");
        let mean_std = std.iter().map(|&s| s as f64).sum::<f64>() / std.len() as f64;
        // Empirical coverage of the nominal 95% interval (pred +/- 1.96 sigma).
        let mut covered = 0usize;
        for ((&p, &y), &s) in scores.iter().zip(&ds.y_test).zip(&std) {
            if (p - y).abs() <= 1.96 * s {
                covered += 1;
            }
        }
        println!("\n--- PGBM predictive intervals ---");
        println!("mean sigma          : {mean_std:.4}");
        println!("95% interval coverage: {:.1}% (nominal 95%)", 100.0 * covered as f64 / std.len() as f64);
    }

    let (fi_gain, fi_count) = booster.feature_importance();
    if !fi_gain.is_empty() {
        let total: f64 = fi_gain.iter().map(|&g| g as f64).sum();
        let mut idx: Vec<usize> = (0..fi_gain.len()).collect();
        idx.sort_by(|&a, &b| fi_gain[b].partial_cmp(&fi_gain[a]).unwrap());
        println!("\n--- top features (by gain) ---");
        for &f in idx.iter().take(8) {
            if fi_gain[f] <= 0.0 {
                break;
            }
            println!(
                "  f{f:<3} gain={:>12.2} ({:>5.1}%)  splits={}",
                fi_gain[f],
                100.0 * fi_gain[f] as f64 / total.max(1e-9),
                fi_count[f] as u64
            );
        }
    }
}

/// Persistent worker: read tab-separated arg lines from stdin, run each as a fit or a
/// load+predict reusing one resident CUDA context, and reply with a status line + `__END__`.
/// Lets a client pay the ~400 ms CUDA init once instead of per call.
#[cfg(feature = "gpu")]
fn serve(gpu: usize) {
    use std::io::{BufRead, Write};
    let mut ctx: Option<std::sync::Arc<cuda_core::CudaContext>> = None;
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if line == "QUIT" {
            break;
        }
        let tokens: Vec<String> = line.split('\t').map(String::from).collect();
        let resp = serve_handle(tokens, gpu, &mut ctx);
        let _ = writeln!(out, "{resp}\n__END__");
        let _ = out.flush();
    }
}

#[cfg(feature = "gpu")]
fn serve_handle(
    tokens: Vec<String>, gpu: usize, ctx: &mut Option<std::sync::Arc<cuda_core::CudaContext>>,
) -> String {
    let cfg = Config::from_tokens(tokens);
    let ds = Dataset::load(&cfg.data_dir);
    if !cfg.load_model.is_empty() {
        let booster = match Booster::load_model(&cfg.load_model) {
            Ok(b) => b,
            Err(e) => return format!("ERROR load {e}"),
        };
        let scores = booster.predict_host(&ds.x_test, ds.n_test);
        if !cfg.dump_pred.is_empty() {
            let mut bytes = Vec::with_capacity(scores.len() * 4);
            for &s in &scores {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            if let Err(e) = std::fs::write(&cfg.dump_pred, &bytes) {
                return format!("ERROR dump {e}");
            }
        }
        return format!("OK n={}", scores.len());
    }
    // fit: CPU has no context; GPU reuses the resident one (lazily created once).
    let mut booster = if cfg.device == Device::Cpu {
        match Booster::new(cfg.clone()) {
            Ok(b) => b,
            Err(e) => return format!("ERROR new {e}"),
        }
    } else {
        if ctx.is_none() {
            match cuda_core::CudaContext::new(gpu) {
                Ok(c) => *ctx = Some(c),
                Err(e) => return format!("ERROR ctx {e}"),
            }
        }
        Booster::with_ctx(cfg.clone(), ctx.clone().unwrap())
    };
    let t = match booster.fit(&ds) {
        Ok(t) => t,
        Err(e) => return format!("ERROR fit {e}"),
    };
    if !cfg.save_model.is_empty() {
        if let Err(e) = booster.save_model(&cfg.save_model) {
            return format!("ERROR save {e}");
        }
    }
    format!("OK train_s={:.6}", t.as_secs_f64())
}

#[cfg(not(feature = "gpu"))]
fn serve(_gpu: usize) {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if line == "QUIT" {
            break;
        }
        let tokens: Vec<String> = line.split('\t').map(String::from).collect();
        let resp = serve_handle_cpu(tokens);
        let _ = writeln!(out, "{resp}\n__END__");
        let _ = out.flush();
    }
}

#[cfg(not(feature = "gpu"))]
fn serve_handle_cpu(tokens: Vec<String>) -> String {
    let mut cfg = Config::from_tokens(tokens);
    if !cfg.load_model.is_empty() {
        cfg.device = Device::Cpu;
    }
    if cfg.device == Device::Gpu {
        return "ERROR this binary was built without the 'gpu' feature; use --device cpu".to_string();
    }
    let ds = Dataset::load(&cfg.data_dir);
    if !cfg.load_model.is_empty() {
        let booster = match Booster::load_model(&cfg.load_model) {
            Ok(b) => b,
            Err(e) => return format!("ERROR load {e}"),
        };
        let scores = booster.predict_host(&ds.x_test, ds.n_test);
        if !cfg.dump_pred.is_empty() {
            let mut bytes = Vec::with_capacity(scores.len() * 4);
            for &s in &scores {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            if let Err(e) = std::fs::write(&cfg.dump_pred, &bytes) {
                return format!("ERROR dump {e}");
            }
        }
        return format!("OK n={}", scores.len());
    }
    let mut booster = match Booster::new(cfg.clone()) {
        Ok(b) => b,
        Err(e) => return format!("ERROR new {e}"),
    };
    let t = match booster.fit(&ds) {
        Ok(t) => t,
        Err(e) => return format!("ERROR fit {e}"),
    };
    if !cfg.save_model.is_empty() {
        if let Err(e) = booster.save_model(&cfg.save_model) {
            return format!("ERROR save {e}");
        }
    }
    format!("OK train_s={:.6}", t.as_secs_f64())
}
