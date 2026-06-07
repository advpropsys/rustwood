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


def build_series(data_dir, rustwood_bin, extra=None):
    """Measure all libraries' inference latency. `extra(Xtr, ytr, Xte)` may return
    {name: sweep-result} to append additional series."""
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    nf = meta["n_features"]
    Xtr = np.fromfile(os.path.join(data_dir, "x_train.bin"), np.float32).reshape(-1, nf)
    ytr = np.fromfile(os.path.join(data_dir, "y_train.bin"), np.float32)
    Xte = np.fromfile(os.path.join(data_dir, "x_test.bin"), np.float32).reshape(-1, nf)

    series = {}
    print("rustwood ...", flush=True)
    gpu, fast, cpu = run_rustwood(rustwood_bin, data_dir)
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

    if extra:
        series.update(extra(Xtr, ytr, Xte))
    return series


def _lines(ax, series, colors, scale):
    for name, res in series.items():
        if res:
            xs = [b for b, _ in res]
            ys = [c / (b if scale == "row" else 1e3) for b, c in res]
            ax.loglog(xs, ys, "o-", color=colors.get(name, style.color(name)),
                      label=name, lw=2.2, ms=7)


def plot_series(series, out, colors):
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    _lines(ax, series, colors, "row")
    ax.axhline(1000, color=style.FAINT, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.text(1.15, 1150, "1 µs / row", fontsize=8.5, color=style.MUTED)
    ax.set_xlabel("batch size (rows scored per call)")
    ax.set_ylabel("amortized latency (ns / row)")
    style.title(ax, "Inference latency vs batch size",
                "true end-to-end · CPU path serves single rows in <1 µs · GPU wins at large batch")
    style.loggrid(ax)
    ax.legend()
    fig.savefig(os.path.join(out, "infer_latency_perrow.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    _lines(ax, series, colors, "call")
    ax.set_xlabel("batch size (rows scored per call)")
    ax.set_ylabel("per-call latency (µs)")
    style.title(ax, "Inference latency per call vs batch size", "true end-to-end · lower is better")
    style.loggrid(ax)
    ax.legend()
    fig.savefig(os.path.join(out, "infer_latency_percall.png"))
    plt.close(fig)

    print("\nbatch |" + "|".join(f"{n.split()[0][:14]:>15}" for n in series) + "   (ns/row)")
    for i, b in enumerate(BATCHES):
        row = [f"{b:6d}"]
        for res in series.values():
            row.append(f"{res[i][1]/res[i][0]:13.1f}" if i < len(res) else f"{'-':>13}")
        print(" |".join(row))
    print(f"wrote {out}/infer_latency_perrow.png / infer_latency_percall.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/rustwood_bench/Synthetic_500K")
    ap.add_argument("--rustwood-bin", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    plot_series(build_series(args.data_dir, args.rustwood_bin), args.out, COLORS)


if __name__ == "__main__":
    main()
