#!/usr/bin/env python3
"""Fair real-data benchmark: rustwood vs XGBoost-GPU / LightGBM-GPU / baseline.

Fairness controls
-----------------
* **Warm timing** — a throwaway GPU fit pays the one-time CUDA/context cost before any
  measurement, so training times exclude cold-start.
* **Per-library learning-rate tuning** — each library's lr is chosen on a held-out
  validation split (grid 0.05/0.1/0.2/0.3); the model is then refit on the full train
  set at its best lr and scored once on the untouched test set.
* **Native categorical handling for every library** — XGBoost (`enable_categorical`),
  LightGBM (pandas `category` dtype), rustwood (out-of-fold target encoding). baseline
  uses ordinal codes (no categorical API). Missing values imputed identically.

Datasets: Titanic, Credit-G, Bank-Marketing, Adult, Covertype (OpenML / sklearn).
Reports test ROC-AUC, warm train time, inference time; writes results/real_results.json
and the publication-style figures results/real_auc.png and results/real_train_time.png.
"""
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RUSTWOOD_BIN = os.path.join(PROJECT, "target", "release", "rustwood")
RESULTS = os.path.join(PROJECT, "results")
TMP = "/tmp/rustwood_real"
TREES, DEPTH = 200, 6
LRS = [0.05, 0.1, 0.2, 0.3]
LIBS = ["rustwood (B300)", "XGBoost-GPU", "LightGBM-GPU", "baseline (CPU)"]


# ------------------------------------------------------------------ data loaders
def load_titanic():
    d = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    X = d.data.drop(columns=["name", "ticket", "cabin", "boat", "body", "home.dest"],
                    errors="ignore")
    return X, d.target.astype(int).values, "Titanic"


def load_creditg():
    d = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    return d.data, (d.target.values == "good").astype(int), "Credit-G"


def load_bank():
    d = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
    return d.data, (pd.Series(d.target).astype(str) == "2").astype(int).values, "Bank-Marketing"


def load_adult():
    d = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    y = pd.Series(d.target).astype(str).str.contains(">50K").astype(int).values
    return d.data, y, "Adult"


def load_covtype():
    d = fetch_covtype()
    return pd.DataFrame(d.data), (d.target == 2).astype(int), "Covertype-581k"


# ----------------------------------------------------------------- preprocessing
def preprocess(X: pd.DataFrame, y):
    """Return (X_num float32 ordinal, X_cat DataFrame w/ category dtype, y, cat_idx)."""
    X = X.copy()
    cat_idx, num = [], X.copy()
    for j, col in enumerate(X.columns):
        s = X[col]
        if s.dtype.kind in "Obc" or str(s.dtype) == "category":
            codes = pd.Series(s.astype("object")).fillna("__NA__").astype("category")
            num[col] = codes.cat.codes.astype(np.float32)  # ordinal for rustwood/baseline
            X[col] = codes                                 # category dtype for xgb/lgbm
            cat_idx.append(j)
        else:
            v = pd.to_numeric(s, errors="coerce").astype(np.float32)
            med = np.nanmedian(v.values)
            num[col] = v.fillna(0.0 if np.isnan(med) else med)
            X[col] = num[col]
    return num.values.astype(np.float32), X, np.asarray(y).astype(np.float32), cat_idx


def save_blobs(d, Xtr, ytr, Xte, yte):
    os.makedirs(d, exist_ok=True)
    np.ascontiguousarray(Xtr, np.float32).tofile(os.path.join(d, "x_train.bin"))
    np.ascontiguousarray(ytr, np.float32).tofile(os.path.join(d, "y_train.bin"))
    np.ascontiguousarray(Xte, np.float32).tofile(os.path.join(d, "x_test.bin"))
    np.ascontiguousarray(yte, np.float32).tofile(os.path.join(d, "y_test.bin"))
    json.dump({"n_train": len(Xtr), "n_test": len(Xte), "n_features": Xtr.shape[1]},
              open(os.path.join(d, "meta.json"), "w"))


# --------------------------------------------------------------------- runners
def rustwood_auc(data_dir, lr, cat_idx):
    args = [RUSTWOOD_BIN, "--data", data_dir, "--objective", "logistic",
            "--trees", str(TREES), "--depth", str(DEPTH), "--lr", str(lr),
            "--bins", "256", "--lambda", "1.0", "--replicas", "64"]
    if cat_idx:
        args += ["--categorical", ",".join(map(str, cat_idx)), "--unique-bins", "1"]
    out = subprocess.run(args, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
    if not line:
        print(out.stdout[-600:], out.stderr[-400:]); raise RuntimeError("rustwood failed")
    kv = dict(t.split("=") for t in line.split()[1:])
    return float(kv["auc"]), float(kv["train_s"]), float(kv["pred_ms"]) / 1e3


def run_rustwood(tune_dir, final_dir, cat_idx):
    best_lr = max(LRS, key=lambda lr: rustwood_auc(tune_dir, lr, cat_idx)[0])
    auc, tr, pred = rustwood_auc(final_dir, best_lr, cat_idx)
    return {"auc": auc, "train": tr, "pred": pred, "lr": best_lr}


def run_xgb(Xi, yi, Xv, yv, Xf, yf, Xte, yte):
    import xgboost as xgb

    def fit(X, y, lr):
        m = xgb.XGBClassifier(n_estimators=TREES, max_depth=DEPTH, learning_rate=lr,
                              reg_lambda=1.0, tree_method="hist", device="cuda",
                              max_bin=256, enable_categorical=True)
        m.fit(X, y)
        return m
    best_lr = max(LRS, key=lambda lr: roc_auc_score(yv, fit(Xi, yi, lr).predict_proba(Xv)[:, 1]))
    t = time.perf_counter(); m = fit(Xf, yf, best_lr); tr = time.perf_counter() - t
    t = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; pt = time.perf_counter() - t
    return {"auc": roc_auc_score(yte, p), "train": tr, "pred": pt, "lr": best_lr}


def run_lgbm(Xi, yi, Xv, yv, Xf, yf, Xte, yte):
    import lightgbm as lgb

    def fit(X, y, lr, dev):
        m = lgb.LGBMClassifier(n_estimators=TREES, max_depth=DEPTH, num_leaves=2 ** DEPTH,
                               learning_rate=lr, reg_lambda=1.0, max_bin=255,
                               device_type=dev, verbose=-1)
        m.fit(X, y)
        return m
    dev = "gpu"
    try:
        fit(Xi.iloc[:128], yi[:128], 0.1, dev)
    except Exception:
        dev = "cpu"
    best_lr = max(LRS, key=lambda lr: roc_auc_score(yv, fit(Xi, yi, lr, dev).predict_proba(Xv)[:, 1]))
    t = time.perf_counter(); m = fit(Xf, yf, best_lr, dev); tr = time.perf_counter() - t
    t = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; pt = time.perf_counter() - t
    return {"auc": roc_auc_score(yte, p), "train": tr, "pred": pt, "lr": best_lr}


def run_baseline(Xi, yi, Xv, yv, Xf, yf, Xte, yte):
    from baseline import BaselineRegressor

    def fit(X, y, lr):
        m = BaselineRegressor(n_estimators=TREES, max_depth=DEPTH, learning_rate=lr,
                            n_bins=256, l2=1.0, backend="rust", random_state=42)
        m.fit(X, y)
        return m
    best_lr = max(LRS, key=lambda lr: roc_auc_score(yv, np.asarray(fit(Xi, yi, lr).predict(Xv)).ravel()))
    t = time.perf_counter(); m = fit(Xf, yf, best_lr); tr = time.perf_counter() - t
    t = time.perf_counter(); p = np.asarray(m.predict(Xte)).ravel(); pt = time.perf_counter() - t
    return {"auc": roc_auc_score(yte, p), "train": tr, "pred": pt, "lr": best_lr}


# --------------------------------------------------------------------- plotting
SHORT = {"rustwood (B300)": "rustwood", "XGBoost-GPU": "XGBoost",
         "LightGBM-GPU": "LightGBM", "baseline (CPU)": "baseline"}


def grouped_bar(results, key, ylabel, main, sub, fname, fmt, logy=False, ylim=None,
                legend_loc="below", label_hero=True):
    names = [d["name"].replace("\n", " ") for d in results]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(names))
    w = 0.19
    for i, lib in enumerate(LIBS):
        vals = [d["by"].get(lib, {}).get(key, np.nan) for d in results]
        ax.bar(x + (i - 1.5) * w, vals, w, color=style.color(lib), label=SHORT[lib],
               zorder=3, edgecolor="white", linewidth=0.6)
    if label_hero:  # direct value labels on the rustwood (hero) bars
        for j, d in enumerate(results):
            v = d["by"].get("rustwood (B300)", {}).get(key, np.nan)
            if not np.isnan(v):
                won = v == max((r[key] for r in d["by"].values()), default=v)
                ax.annotate(fmt(v), (x[j] - 1.5 * w, v), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=8.6,
                            color=style.color("rustwood"),
                            fontweight="bold" if won else "semibold")
    style.title(ax, main, sub)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.margins(y=0.12)
    if legend_loc == "below":
        style.legend(ax, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    else:
        style.legend(ax, ncol=2, loc=legend_loc)
    fig.savefig(os.path.join(RESULTS, fname))
    plt.close(fig)
    print("wrote", fname)


def make_plots(results):
    grouped_bar(results, "auc", "test ROC-AUC",
                "Accuracy on real tabular datasets",
                "depth 6, 200 trees · each library learning-rate-tuned on validation · test ROC-AUC",
                "real_auc.png", fmt=lambda v: f"{v:.3f}", ylim=(0.70, 1.0), legend_loc="below")
    grouped_bar(results, "train", "training wall-clock (s)",
                "Training time on real datasets",
                "depth 6, 200 trees · warm (cold-start excluded) · lower is better",
                "real_train_time.png", fmt=lambda v: f"{v:.2f}s", logy=True,
                legend_loc="upper left", label_hero=True)


def main():
    style.apply()
    os.makedirs(RESULTS, exist_ok=True)
    if "--plot-only" in sys.argv:  # re-render figures from saved results
        make_plots(json.load(open(os.path.join(RESULTS, "real_results.json"))))
        return
    if not os.path.exists(RUSTWOOD_BIN):
        sys.exit(f"build rustwood first: {RUSTWOOD_BIN}")

    # Warm the GPU (pay CUDA/context cold-start once, before any timing).
    try:
        import xgboost as xgb
        xgb.XGBClassifier(n_estimators=8, device="cuda", tree_method="hist").fit(
            np.random.rand(256, 4).astype(np.float32), (np.random.rand(256) > 0.5).astype(int))
        print("GPU warmed up")
    except Exception as e:  # noqa: BLE001
        print("warmup skipped:", e)

    results = []
    for loader in (load_titanic, load_creditg, load_bank, load_adult, load_covtype):
        try:
            Xdf, y, name = loader()
            Xnum, Xcat, y, cat_idx = preprocess(Xdf, y)
        except Exception as e:  # noqa: BLE001
            print(f"[skip {loader.__name__}] {e}"); continue

        idx = np.arange(len(y))
        i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        i_ti, i_va = train_test_split(i_tr, test_size=0.2, random_state=42, stratify=y[i_tr])
        print(f"\n{'='*70}\n{name}: train={len(i_tr):,} test={len(i_te):,} "
              f"feat={Xnum.shape[1]} cat={len(cat_idx)} pos={y.mean():.3f}\n{'='*70}")

        save_blobs(os.path.join(TMP, name, "tune"), Xnum[i_ti], y[i_ti], Xnum[i_va], y[i_va])
        save_blobs(os.path.join(TMP, name, "final"), Xnum[i_tr], y[i_tr], Xnum[i_te], y[i_te])

        by = {}
        runners = {
            "rustwood (B300)": lambda: run_rustwood(os.path.join(TMP, name, "tune"),
                                                    os.path.join(TMP, name, "final"), cat_idx),
            "XGBoost-GPU": lambda: run_xgb(Xcat.iloc[i_ti], y[i_ti], Xcat.iloc[i_va], y[i_va],
                                           Xcat.iloc[i_tr], y[i_tr], Xcat.iloc[i_te], y[i_te]),
            "LightGBM-GPU": lambda: run_lgbm(Xcat.iloc[i_ti], y[i_ti], Xcat.iloc[i_va], y[i_va],
                                             Xcat.iloc[i_tr], y[i_tr], Xcat.iloc[i_te], y[i_te]),
            "baseline (CPU)": lambda: run_baseline(Xnum[i_ti], y[i_ti], Xnum[i_va], y[i_va],
                                               Xnum[i_tr], y[i_tr], Xnum[i_te], y[i_te]),
        }
        for lib, fn in runners.items():
            try:
                by[lib] = fn()
            except Exception as e:  # noqa: BLE001
                print(f"  [skip {lib}] {type(e).__name__}: {str(e)[:80]}")
        best = max((r["auc"] for r in by.values()), default=0)
        print(f"  {'model':<16}{'AUC':>8}{'lr':>6}{'train_s':>10}{'pred_ms':>10}")
        for lib in LIBS:
            if lib in by:
                r = by[lib]
                star = " *" if r["auc"] == best else ""
                print(f"  {lib:<16}{r['auc']:>8.4f}{r['lr']:>6}{r['train']:>10.3f}{r['pred']*1e3:>10.2f}{star}")
        results.append({"name": name.replace("-581k", "\n581k"), "by": by})

    out = [{"name": r["name"].replace("\n", " "), "by": r["by"]} for r in results]
    json.dump(out, open(os.path.join(RESULTS, "real_results.json"), "w"), indent=2)
    make_plots(out)


if __name__ == "__main__":
    main()
