#!/usr/bin/env python3
"""Plot the real-data benchmark results (test AUC + train time) — high DPI, labeled.

Numbers are from scripts/real_bench.py on OpenML/sklearn datasets (300 trees, depth 6).
rustwood shows its best of {ordinal, +target-encoding}. Regenerate the raw numbers with
real_bench.py; this script renders the summary figures committed to results/.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DATASETS = ["Titanic", "Credit-G", "Bank-Mkt", "Adult", "Covertype\n581k"]
# (rustwood-best, XGBoost-GPU, LightGBM-GPU, baseline-CPU)
AUC = {
    "rustwood (B300)": [0.8956, 0.7974, 0.9346, 0.9294, 0.9250],
    "XGBoost-GPU":     [0.8609, 0.7344, 0.9305, 0.9301, 0.9481],
    "LightGBM-GPU":    [0.8816, 0.7432, 0.9301, 0.9294, 0.9492],
    "baseline (CPU)":    [0.9005, 0.7365, 0.9235, 0.9165, 0.9091],
}
TRAIN = {
    "rustwood (B300)": [0.246, 0.262, 0.260, 0.266, 0.571],
    "XGBoost-GPU":     [0.610, 0.356, 0.476, 0.435, 0.552],
    "LightGBM-GPU":    [2.749, 1.440, 6.188, 5.164, 5.574],
    "baseline (CPU)":    [0.147, 0.259, 0.440, 0.420, 3.968],
}
COLORS = {"rustwood (B300)": "#d81e5b", "XGBoost-GPU": "#1f77b4",
          "LightGBM-GPU": "#2ca02c", "baseline (CPU)": "#ff7f0e"}


def grouped(ax, data, ylabel, title, ylim=None, logy=False):
    libs = list(data)
    x = np.arange(len(DATASETS))
    w = 0.2
    for i, lib in enumerate(libs):
        ax.bar(x + (i - 1.5) * w, data[lib], w, color=COLORS[lib], label=lib)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)


def main():
    os.makedirs(OUT, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    grouped(ax, AUC, "test ROC-AUC", "Accuracy on real datasets (test AUC, 300 trees / depth 6)",
            ylim=(0.70, 0.97))
    # mark rustwood wins
    for j, d in enumerate(DATASETS):
        best = max(v[j] for v in AUC.values())
        if AUC["rustwood (B300)"][j] == best:
            ax.annotate("win", (j - 0.30, best + 0.004), fontsize=8, color="#d81e5b",
                        fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "real_auc.png"), dpi=200)
    plt.close(fig)
    print("wrote results/real_auc.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    grouped(ax, TRAIN, "training wall-clock (s)", "Training time on real datasets (log scale)",
            logy=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "real_train_time.png"), dpi=200)
    plt.close(fig)
    print("wrote results/real_train_time.png")


if __name__ == "__main__":
    main()
