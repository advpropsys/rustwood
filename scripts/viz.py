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


PHASE_ORDER = ["histogram", "split_find", "leaf_reduce", "gradient", "other"]
PHASE_COLOR = {
    "histogram": "#E8613C",    # rustwood terracotta — the hotspot
    "split_find": "#33506B",   # slate
    "leaf_reduce": "#5FA08C",  # teal
    "gradient": "#B58DB6",     # mauve
    "other": "#9AA3AF",
}


def _tint(hexc, f):
    """Blend a hex color toward white by fraction f (0=orig, 1=white)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def make_flamegraph(profile_path, out_png, label):
    """Native icicle/flame chart (matplotlib) matching the editorial style.

    Three rows top-down: root → pipeline phases → kernels, widths proportional to GPU
    nanoseconds. Phases keep the project palette; kernels are tints of their phase.
    """
    from matplotlib.patches import FancyBboxPatch

    prof = json.load(open(profile_path))
    cats = {k: v for k, v in prof["categories"].items() if v > 0}
    total = sum(cats.values())
    phases = {}
    for k, v in cats.items():
        phases.setdefault(PHASE.get(k, "other"), []).append((k, v))

    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    h, gap = 1.0, 0.05

    def cell(x, y, w, color, label_main, label_sub, dark_text):
        if w <= 0:
            return
        ax.add_patch(FancyBboxPatch(
            (x + total * 0.0008, y + gap), w - total * 0.0016, h - 2 * gap,
            boxstyle="round,pad=0,rounding_size=" + str(total * 0.004),
            linewidth=0, facecolor=color))
        if w / total > 0.045:  # only label cells wide enough to read
            tc = "#15202B" if dark_text else "white"
            ax.text(x + w / 2, y + h * 0.60, label_main, ha="center", va="center",
                    fontsize=9, color=tc, fontweight="semibold")
            if label_sub and w / total > 0.07:
                ax.text(x + w / 2, y + h * 0.30, label_sub, ha="center", va="center",
                        fontsize=7.5, color=tc, alpha=0.85)

    cell(0, 2 * (h), total, "#EDF0F3", "rustwood training", f"{total/1e6:.0f} ms", True)
    x = 0
    for ph in PHASE_ORDER:
        if ph not in phases:
            continue
        items = sorted(phases[ph], key=lambda t: -t[1])
        pw = sum(v for _, v in items)
        cell(x, h, pw, PHASE_COLOR[ph], ph.replace("_", "-"),
             f"{100*pw/total:.0f}%", ph in ("leaf_reduce", "other"))
        kx = x
        for i, (k, v) in enumerate(items):
            cell(kx, 0, v, _tint(PHASE_COLOR[ph], 0.30 + 0.12 * (i % 3)), k,
                 f"{100*v/total:.0f}%", True)
            kx += v
        x += pw

    ax.set_xlim(-total * 0.01, total * 1.01)
    ax.set_ylim(-0.1, 3 * h + 0.15)
    ax.axis("off")
    style.title(ax, f"GPU training kernels — {label}",
                f"nanosecond-resolution profile · bar width $\\propto$ GPU time · total {total/1e6:.0f} ms")
    fig.savefig(out_png)
    plt.close(fig)
    print("wrote", out_png)


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
    style.title(plt.gca(), "Training time vs dataset size", "synthetic regression · 100 trees, depth 6 · log–log · lower is better")
    plt.legend()
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
    style.title(plt.gca(), "Inference throughput vs dataset size", "amortized per-row latency · log–log · lower is better")
    plt.legend()
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
    style.title(plt.gca(), "rustwood training speedup", "× faster than each baseline at matched hyperparameters · higher is better")
    plt.legend()
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
    style.title(plt.gca(), "Accuracy vs dataset size", "synthetic regression · test R² · oblivious rustwood vs asymmetric baselines")
    plt.legend()
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
    style.title(plt.gca(), "Accuracy vs training time (1M rows)", "the iso-accuracy frontier · up and to the left is better")
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend()
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
        make_flamegraph(path, os.path.join(args.out, f"flamegraph_{label}.png"), f"{label} rows")

    plot_train_time(results, os.path.join(args.out, "train_time_loglog.png"))
    plot_infer(results, os.path.join(args.out, "infer_throughput.png"))
    plot_speedup(results, os.path.join(args.out, "speedup_bars.png"))
    plot_accuracy(results, os.path.join(args.out, "accuracy_bars.png"))
    if args.rustwood_bin:
        plot_iso_accuracy(results, args.rustwood_bin, args.data_dir_1m,
                          os.path.join(args.out, "iso_accuracy_1M.png"))


if __name__ == "__main__":
    main()
