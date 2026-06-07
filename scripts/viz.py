#!/usr/bin/env python3
"""Visualize rustwood benchmarks.

Produces, into --out:
  * flamegraph_<size>.svg  -- hierarchical GPU-kernel flamegraph (nanosecond counts)
    from a rustwood `--profile-out` JSON, rendered with Brendan Gregg's flamegraph.pl.
  * train_time_loglog.png  -- train wall-clock vs dataset size (log-log), all libs.
  * infer_throughput.png   -- inference us/row vs dataset size (log-log), all libs.
  * speedup_bars.png       -- rustwood training speedup vs each baseline, per size.
  * accuracy_bars.png      -- R2 per library per size.
  * iso_accuracy_1M.png    -- R2 vs train time frontier at 1M (rustwood depth/trees sweep).

Usage:
  python viz.py --results /tmp/rustwood_bench/results_final.json \
      --profiles 100K=/tmp/rustwood_bench/profile_100K.json,1000K=.../profile_1000K.json \
      --out crates/rustc-codegen-cuda/examples/rustwood/results
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

style.apply()

# Group flat kernel categories into pipeline phases for a readable flame.
PHASE = {
    "gradients": "gradient",
    "zero_hist": "histogram",
    "build_hist": "histogram",
    "split_gain": "split_find",
    "argmax_split": "split_find",
    "apply_split": "split_find",
    "leaf_hist": "leaf_reduce",
    "leaf_values": "leaf_reduce",
    "update_pred": "leaf_reduce",
}
COLORS = {n: style.color(n) for n in
          ["rustwood (B300)", "XGBoost-GPU", "LightGBM-gpu", "baseline (CPU)"]}
FLAMEGRAPH_PL = "flamegraph.pl"


def make_flamegraph(profile_path, out_svg, label):
    prof = json.load(open(profile_path))
    cats = prof["categories"]
    total_ns = sum(v for v in cats.values() if v > 0)
    folded = []
    for k, ns in cats.items():
        if ns > 0:
            folded.append(f"rustwood_train;{PHASE.get(k, 'other')};{k} {ns}")
    title = f"rustwood GPU training kernels - {label} (total {total_ns/1e6:.1f} ms, ns-resolution)"
    try:
        svg = subprocess.run(
            [FLAMEGRAPH_PL, "--title", title, "--countname", "ns",
             "--width", "1400", "--height", "28", "--fontsize", "12",
             "--colors", "hot"],
            input="\n".join(folded), capture_output=True, text=True, check=True,
        ).stdout
        with open(out_svg, "w") as fh:
            fh.write(svg)
        print("wrote", out_svg)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[flamegraph.pl unavailable: {e}] writing folded stacks instead")
        with open(out_svg.replace(".svg", ".folded"), "w") as fh:
            fh.write("\n".join(folded))


def synthetic_rows(results):
    """Return {dataset_name: {n_train, results:{lib:row}}} for synthetic sizes only."""
    out = {}
    for ds in results:
        if ds["dataset"].startswith("Synthetic"):
            byname = {r["name"]: r for r in ds["results"]}
            out[ds["n_train"]] = byname
    return dict(sorted(out.items()))


def plot_train_time(results, out_png):
    data = synthetic_rows(results)
    sizes = list(data.keys())
    libs = list(COLORS)
    plt.figure(figsize=(8, 5.2))
    for lib in libs:
        ys = [data[s][lib]["train_time"] for s in sizes if lib in data[s]]
        xs = [s for s in sizes if lib in data[s]]
        if xs:
            plt.loglog(xs, ys, "o-", color=COLORS[lib], label=lib, lw=2.2, ms=7)
    plt.xlabel("training rows")
    plt.ylabel("training wall-clock (s)")
    plt.title("Training time vs dataset size (log-log) — 100 trees, depth 6")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("wrote", out_png)


def plot_infer(results, out_png):
    data = synthetic_rows(results)
    sizes = list(data.keys())
    plt.figure(figsize=(8, 5.2))
    for lib in COLORS:
        xs, ys = [], []
        for s in sizes:
            if lib in data[s]:
                # n_test == n_train // 5 for the synthetic sizes (see bench.py).
                ys.append(data[s][lib]["pred_time"] * 1e9 / (s // 5))
                xs.append(s)
        if xs:
            plt.loglog(xs, ys, "o-", color=COLORS[lib], label=lib, lw=2.2, ms=7)
    plt.xlabel("training rows")
    plt.ylabel("inference time (ns / row)")
    plt.title("Inference latency vs dataset size (log-log, ns/row)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("wrote", out_png)


def plot_speedup(results, out_png):
    data = synthetic_rows(results)
    sizes = list(data.keys())
    others = ["XGBoost-GPU", "LightGBM-gpu", "baseline (CPU)"]
    import numpy as np
    x = np.arange(len(sizes))
    w = 0.25
    plt.figure(figsize=(9, 5.2))
    for i, lib in enumerate(others):
        sp = [data[s][lib]["train_time"] / data[s]["rustwood (B300)"]["train_time"]
              if lib in data[s] else 0 for s in sizes]
        plt.bar(x + (i - 1) * w, sp, w, color=COLORS[lib], label=f"vs {lib}")
    plt.axhline(1.0, color="k", ls="--", lw=1)
    plt.xticks(x, [f"{s//1000}K" for s in sizes])
    plt.ylabel("rustwood training speedup (×)")
    plt.title("How many times faster rustwood trains (same hyperparameters)")
    plt.legend()
    plt.grid(True, axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("wrote", out_png)


def plot_accuracy(results, out_png):
    data = synthetic_rows(results)
    sizes = list(data.keys())
    import numpy as np
    x = np.arange(len(sizes))
    libs = list(COLORS)
    w = 0.2
    plt.figure(figsize=(9, 5.2))
    for i, lib in enumerate(libs):
        r2 = [data[s][lib]["r2"] if lib in data[s] else 0 for s in sizes]
        plt.bar(x + (i - 1.5) * w, r2, w, color=COLORS[lib], label=lib)
    plt.xticks(x, [f"{s//1000}K" for s in sizes])
    plt.ylabel("test R²")
    plt.ylim(0.5, 1.0)
    plt.title("Accuracy (R²) — oblivious rustwood vs asymmetric XGBoost/LightGBM")
    plt.legend()
    plt.grid(True, axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("wrote", out_png)


def plot_iso_accuracy(results, rustwood_bin, data_dir, out_png):
    """R2 vs train time at 1M: rustwood depth/trees sweep + baseline points."""
    data = synthetic_rows(results)
    one_m = next((data[s] for s in data if 900_000 <= s <= 1_100_000), None)
    if one_m is None:
        print("no ~1M dataset in results; skipping iso-accuracy")
        return
    sweep = [(6, 100), (6, 200), (6, 300), (7, 150), (8, 100), (8, 150), (8, 250)]
    pts = []
    for depth, trees in sweep:
        out = subprocess.run(
            [rustwood_bin, "--data", data_dir, "--objective", "l2", "--trees", str(trees),
             "--depth", str(depth), "--lr", "0.1", "--replicas", "64"],
            capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
        if line:
            kv = dict(t.split("=") for t in line.split()[1:])
            pts.append((float(kv["train_s"]), float(kv["r2"]), f"d{depth}/{trees}"))
    plt.figure(figsize=(8.2, 5.4))
    if pts:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        plt.plot(xs, ys, "o-", color=COLORS["rustwood (B300)"], label="rustwood (depth/trees sweep)", lw=2.2, ms=8)
        for x, y, t in pts:
            plt.annotate(t, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8)
    for lib in ("XGBoost-GPU", "LightGBM-gpu", "baseline (CPU)"):
        if lib in one_m:
            plt.scatter([one_m[lib]["train_time"]], [one_m[lib]["r2"]],
                        color=COLORS[lib], s=130, marker="*", zorder=5, label=lib)
    plt.xlabel("training wall-clock (s)")
    plt.ylabel("test R²")
    plt.title("Accuracy vs training time @ 1M rows — the iso-accuracy frontier")
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/rustwood_bench/results_final.json")
    ap.add_argument("--profiles", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--rustwood-bin", default="")
    ap.add_argument("--data-dir-1m", default="/tmp/rustwood_bench/Synthetic_1000K")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = json.load(open(args.results))

    for pair in filter(None, args.profiles.split(",")):
        label, path = pair.split("=", 1)
        make_flamegraph(path, os.path.join(args.out, f"flamegraph_{label}.svg"), f"{label} rows")

    plot_train_time(results, os.path.join(args.out, "train_time_loglog.png"))
    plot_infer(results, os.path.join(args.out, "infer_throughput.png"))
    plot_speedup(results, os.path.join(args.out, "speedup_bars.png"))
    plot_accuracy(results, os.path.join(args.out, "accuracy_bars.png"))
    if args.rustwood_bin:
        plot_iso_accuracy(results, args.rustwood_bin, args.data_dir_1m,
                          os.path.join(args.out, "iso_accuracy_1M.png"))


if __name__ == "__main__":
    main()
