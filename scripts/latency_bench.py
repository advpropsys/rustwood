#!/usr/bin/env python3
"""True end-to-end inference latency sweep: how long does it take to score a batch of
B rows, from B=1 (single-row latency) up to large batches (throughput regime)?

Every library is measured the same way: model trained once and kept loaded, then
`repeats` full `predict(X[:B])` calls timed with `perf_counter_ns` (the call blocks
until results are back on the host). rustwood is measured by its own `--latency-bench`
(model resident on the GPU, host->device + kernel + device->host + sync per call).

Outputs <out>/infer_latency_perrow.png and <out>/infer_latency_percall.png.
"""
import argparse
import json
import os
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
style.apply()

N_TREES, MAX_DEPTH, LR, L2, MAX_BIN = 100, 6, 0.1, 1.0, 256
BATCHES = [1, 8, 64, 512, 4096, 32768, 131072]
COLORS = {
    "rustwood-GPU (B300)": "#D55E00",
    "rustwood-GPU-fast (reuse+spin)": "#E69F00",
    "rustwood-CPU": "#9467bd",
    "XGBoost-GPU": "#0072B2",
    "LightGBM-CPU": "#009E73",
    "baseline (CPU)": "#CC79A7",
}


def reps_for(b):
    return 1000 if b <= 64 else (300 if b <= 4096 else (50 if b <= 32768 else 15))


def sweep(predict_fn, X, batches):
    res = []
    for b in batches:
        b = min(b, len(X))
        xb = np.ascontiguousarray(X[:b])
        for _ in range(5):
            predict_fn(xb)
        r = reps_for(b)
        t0 = time.perf_counter_ns()
        for _ in range(r):
            predict_fn(xb)
        per_call = (time.perf_counter_ns() - t0) / r
        res.append((b, per_call))
    return res


def run_rustwood(rustwood_bin, data_dir):
    out = subprocess.run(
        [rustwood_bin, "--data", data_dir, "--objective", "l2", "--trees", str(N_TREES),
         "--depth", str(MAX_DEPTH), "--lr", str(LR), "--bins", str(MAX_BIN),
         "--lambda", str(L2), "--replicas", "64", "--latency-bench", "1"],
        capture_output=True, text=True)
    gpu, fast, cpu = [], [], []
    for line in out.stdout.splitlines():
        kv = dict(t.split("=") for t in line.split()[1:]) if line.startswith("LATENCY") else {}
        if line.startswith("LATENCY_CPU"):
            cpu.append((int(kv["batch"]), float(kv["per_call_ns"])))
        elif line.startswith("LATENCY_FAST"):
            fast.append((int(kv["batch"]), float(kv["per_call_ns"])))
        elif line.startswith("LATENCY"):
            gpu.append((int(kv["batch"]), float(kv["per_call_ns"])))
    if not gpu:
        print(out.stdout[-1500:], out.stderr[-1500:])
    return gpu, fast, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/rustwood_bench/Synthetic_500K")
    ap.add_argument("--rustwood-bin", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    nf = meta["n_features"]
    Xtr = np.fromfile(os.path.join(args.data_dir, "x_train.bin"), np.float32).reshape(-1, nf)
    ytr = np.fromfile(os.path.join(args.data_dir, "y_train.bin"), np.float32)
    Xte = np.fromfile(os.path.join(args.data_dir, "x_test.bin"), np.float32).reshape(-1, nf)

    series = {}

    print("rustwood ...", flush=True)
    gpu, fast, cpu = run_rustwood(args.rustwood_bin, args.data_dir)
    series["rustwood-GPU (B300)"] = gpu
    series["rustwood-GPU-fast (reuse+spin)"] = fast
    series["rustwood-CPU"] = cpu

    print("xgboost ...", flush=True)
    import xgboost as xgb
    xm = xgb.XGBRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=LR,
                          reg_lambda=L2, tree_method="hist", device="cuda", max_bin=MAX_BIN)
    xm.fit(Xtr, ytr)
    series["XGBoost-GPU"] = sweep(lambda xb: xm.predict(xb), Xte, BATCHES)

    print("lightgbm ...", flush=True)
    import lightgbm as lgb
    lm = lgb.LGBMRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, num_leaves=2 ** MAX_DEPTH,
                           learning_rate=LR, reg_lambda=L2, max_bin=255, verbose=-1)
    lm.fit(Xtr, ytr)
    series["LightGBM-CPU"] = sweep(lambda xb: lm.predict(xb), Xte, BATCHES)

    print("baseline ...", flush=True)
    from baseline import BaselineRegressor
    rm = BaselineRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=LR,
                         n_bins=MAX_BIN, l2=L2, backend="rust", random_state=42)
    rm.fit(Xtr, ytr)
    series["baseline (CPU)"] = sweep(lambda xb: rm.predict(xb), Xte, BATCHES)

    # ---- per-row latency (ns/row) vs batch ----
    plt.figure(figsize=(8.4, 5.4))
    for name, res in series.items():
        if res:
            xs = [b for b, _ in res]
            ys = [c / b for b, c in res]
            plt.loglog(xs, ys, "o-", color=COLORS[name], label=name, lw=2.2, ms=7)
    plt.axhline(1000, color=style.FAINT, ls=(0, (4, 3)), lw=1.0, zorder=1)
    plt.text(1.15, 1150, "1 µs / row", fontsize=8.5, color=style.MUTED)
    plt.xlabel("batch size (rows scored per call)")
    plt.ylabel("amortized latency (ns / row)")
    style.title(plt.gca(), "Inference latency vs batch size",
                "true end-to-end · CPU path serves single rows in <1 µs · GPU wins at large batch")
    style.loggrid(plt.gca())
    plt.legend()
    plt.savefig(os.path.join(args.out, "infer_latency_perrow.png"))
    plt.close()

    # ---- per-call latency (us) vs batch ----
    plt.figure(figsize=(8.4, 5.4))
    for name, res in series.items():
        if res:
            xs = [b for b, _ in res]
            ys = [c / 1e3 for _, c in res]
            plt.loglog(xs, ys, "o-", color=COLORS[name], label=name, lw=2.2, ms=7)
    plt.xlabel("batch size (rows scored per call)")
    plt.ylabel("per-call latency (µs)")
    style.title(plt.gca(), "Inference latency per call vs batch size",
                "true end-to-end · lower is better")
    style.loggrid(plt.gca())
    plt.legend()
    plt.savefig(os.path.join(args.out, "infer_latency_percall.png"))
    plt.close()

    print("\nbatch |        rustwood |       XGBoost |      LightGBM |        baseline   (ns/row)")
    for i, b in enumerate(BATCHES):
        row = [f"{b:6d}"]
        for name in COLORS:
            r = series.get(name, [])
            row.append(f"{r[i][1]/r[i][0]:13.1f}" if i < len(r) else f"{'-':>13}")
        print(" |".join(row))
    print("wrote infer_latency_perrow.png / infer_latency_percall.png")


if __name__ == "__main__":
    main()
