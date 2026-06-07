#!/usr/bin/env python3
"""Benchmark rustwood (GPU oblivious-tree GBDT on B300) vs XGBoost-GPU and
LightGBM-GPU, using baseline's benchmark methodology.

- Datasets: California Housing (real) + baseline's scaling synthetic regression
  (20 numeric + 5 categorical features, 10 informative, nonlinear interaction).
- Hyperparameters (identical for every library): 100 trees, depth 6, lr 0.1,
  L2=1.0, max_bin 256.
- All features treated as numeric for every library (apples-to-apples; no library
  gets categorical special-casing).
- Metrics: RMSE / MAE / R2, train time (+ ms/tree), inference time (+ us/sample).

Run from the cuda-oxide repo root after `cargo oxide build rustwood --arch sm_103`.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(HERE)
RUSTWOOD_BIN = os.path.join(EXAMPLE_DIR, "target", "release", "rustwood")

N_TREES = 100
MAX_DEPTH = 6
LR = 0.1
L2 = 1.0
MAX_BIN = 256


# --------------------------------------------------------------------------- data
def gen_synthetic(n_total, seed=42, n_numeric=20, n_categorical=5, n_informative=10,
                  noise_std=0.5):
    """Port of baseline benchmarks/baseline_vs_catboost_FIXED.py:generate_regression_data."""
    rng = np.random.RandomState(seed)
    X_numeric = rng.randn(n_total, n_numeric)
    X_categorical = np.zeros((n_total, n_categorical), dtype=int)
    cardinalities = [5, 10, 20, 50, 100][:n_categorical]
    for i, card in enumerate(cardinalities):
        X_categorical[:, i] = rng.randint(0, card, size=n_total)
    X = np.concatenate([X_numeric, X_categorical], axis=1)

    y = np.zeros(n_total)
    n_num_inf = min(n_informative // 2, n_numeric)
    for i in range(n_num_inf):
        y += X_numeric[:, i] * (rng.randn() * 2.0)
    n_cat_inf = min(n_informative - n_num_inf, n_categorical)
    for i in range(n_cat_inf):
        n_cats = int(X_categorical[:, i].max() + 1)
        eff = rng.randn(n_cats) * 1.5
        for cv in range(n_cats):
            y[X_categorical[:, i] == cv] += eff[cv]
    if n_numeric >= 2:
        y += X_numeric[:, 0] * X_numeric[:, 1] * 0.5
    y += rng.randn(n_total) * noise_std
    return X.astype(np.float32), y.astype(np.float32)


def load_california():
    d = fetch_california_housing()
    return d.data.astype(np.float32), d.target.astype(np.float32)


def save_blobs(out_dir, Xtr, ytr, Xte, yte):
    os.makedirs(out_dir, exist_ok=True)
    np.ascontiguousarray(Xtr, np.float32).tofile(os.path.join(out_dir, "x_train.bin"))
    np.ascontiguousarray(ytr, np.float32).tofile(os.path.join(out_dir, "y_train.bin"))
    np.ascontiguousarray(Xte, np.float32).tofile(os.path.join(out_dir, "x_test.bin"))
    np.ascontiguousarray(yte, np.float32).tofile(os.path.join(out_dir, "y_test.bin"))
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump({"n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                   "n_features": int(Xtr.shape[1])}, fh)


def metrics(yte, pred):
    return (float(np.sqrt(mean_squared_error(yte, pred))),
            float(mean_absolute_error(yte, pred)),
            float(r2_score(yte, pred)))


# ---------------------------------------------------------------------- backends
def run_rustwood(data_dir, gpu=0):
    args = [RUSTWOOD_BIN, "--data", data_dir, "--objective", "l2",
            "--trees", str(N_TREES), "--depth", str(MAX_DEPTH), "--lr", str(LR),
            "--bins", str(MAX_BIN), "--lambda", str(L2), "--gpu", str(gpu)]
    t0 = time.perf_counter()
    out = subprocess.run(args, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if out.returncode != 0:
        print(out.stdout[-2000:]); print(out.stderr[-2000:])
        raise RuntimeError("rustwood failed")
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))
    kv = dict(tok.split("=") for tok in line.split()[1:])
    return {"name": "rustwood (B300)", "train_time": float(kv["train_s"]),
            "pred_time": float(kv["pred_ms"]) / 1e3, "rmse": float(kv["rmse"]),
            "mae": float(kv["mae"]), "r2": float(kv["r2"]), "wall": wall}


def run_xgb(Xtr, ytr, Xte, yte):
    import xrustwood as xgb
    m = xgb.XGBRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=LR,
                         reg_lambda=L2, tree_method="hist", device="cuda",
                         max_bin=MAX_BIN, objective="reg:squarederror")
    t0 = time.perf_counter(); m.fit(Xtr, ytr); train = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = m.predict(Xte); pt = time.perf_counter() - t0
    rmse, mae, r2 = metrics(yte, pred)
    return {"name": "XGBoost-GPU", "train_time": train, "pred_time": pt,
            "rmse": rmse, "mae": mae, "r2": r2, "wall": train}


def run_baseline(Xtr, ytr, Xte, yte):
    """baseline with its Rust-accelerated CPU backend (the reference oblivious-tree lib)."""
    from baseline import BaselineRegressor
    m = BaselineRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=LR,
                        n_bins=MAX_BIN, l2=L2, backend="rust", random_state=42)
    t0 = time.perf_counter(); m.fit(Xtr, ytr); train = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = m.predict(Xte); pt = time.perf_counter() - t0
    rmse, mae, r2 = metrics(yte, np.asarray(pred).ravel())
    return {"name": "baseline (CPU)", "train_time": train, "pred_time": pt,
            "rmse": rmse, "mae": mae, "r2": r2, "wall": train}


def run_lgbm(Xtr, ytr, Xte, yte):
    import lightgbm as lgb
    last = None
    # LightGBM's OpenCL GPU learner caps at 255 bins; drop 'cuda' (not compiled here).
    for device in ("gpu", "cpu"):
        try:
            m = lgb.LGBMRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                  num_leaves=2 ** MAX_DEPTH, learning_rate=LR,
                                  reg_lambda=L2, max_bin=255, device_type=device,
                                  verbose=-1)
            t0 = time.perf_counter(); m.fit(Xtr, ytr); train = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = m.predict(Xte); pt = time.perf_counter() - t0
            rmse, mae, r2 = metrics(yte, pred)
            return {"name": f"LightGBM-{device}", "train_time": train, "pred_time": pt,
                    "rmse": rmse, "mae": mae, "r2": r2, "wall": train}
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"LightGBM failed on all devices: {last}")


def run_catboost(Xtr, ytr, Xte, yte):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=N_TREES, depth=MAX_DEPTH, learning_rate=LR,
                          l2_leaf_reg=L2, task_type="GPU", devices="0",
                          logging_level="Silent", border_count=min(MAX_BIN - 1, 255))
    t0 = time.perf_counter(); m.fit(Xtr, ytr); train = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = m.predict(Xte); pt = time.perf_counter() - t0
    rmse, mae, r2 = metrics(yte, pred)
    return {"name": "CatBoost-GPU", "train_time": train, "pred_time": pt,
            "rmse": rmse, "mae": mae, "r2": r2, "wall": train}


# --------------------------------------------------------------------------- main
def bench_dataset(name, X, y, test_size, seed, gpu, data_root, want_catboost):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    data_dir = os.path.join(data_root, name.replace(" ", "_"))
    save_blobs(data_dir, Xtr, ytr, Xte, yte)

    print(f"\n{'=' * 78}\nDATASET: {name}  | train={len(Xtr):,} test={len(Xte):,} "
          f"features={X.shape[1]}\n{'=' * 78}")

    runners = [("rustwood", lambda: run_rustwood(data_dir, gpu)),
               ("xrustwood", lambda: run_xgb(Xtr, ytr, Xte, yte)),
               ("lightgbm", lambda: run_lgbm(Xtr, ytr, Xte, yte)),
               ("baseline", lambda: run_baseline(Xtr, ytr, Xte, yte))]
    if want_catboost:
        runners.append(("catboost", lambda: run_catboost(Xtr, ytr, Xte, yte)))

    rows = []
    for key, fn in runners:
        try:
            rows.append(fn())
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {key}] {e}")

    n_test = len(Xte)
    hdr = f"{'model':<16}{'train_s':>10}{'ms/tree':>10}{'pred_ms':>10}{'us/row':>9}{'RMSE':>11}{'R2':>9}"
    print(hdr); print("-" * len(hdr))
    base = next((r for r in rows if r["name"].startswith("rustwood")), None)
    for r in rows:
        print(f"{r['name']:<16}{r['train_time']:>10.3f}"
              f"{r['train_time'] / N_TREES * 1e3:>10.2f}"
              f"{r['pred_time'] * 1e3:>10.2f}{r['pred_time'] / n_test * 1e6:>9.3f}"
              f"{r['rmse']:>11.4f}{r['r2']:>9.4f}")
    if base:
        print("speedup vs rustwood (train):", ", ".join(
            f"{r['name']}={r['train_time'] / base['train_time']:.2f}x"
            for r in rows if r is not base))
    return {"dataset": name, "n_train": len(Xtr), "n_test": n_test,
            "n_features": int(X.shape[1]), "results": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="california,100000,500000,1000000",
                    help="comma list: 'california' and/or synthetic n_train ints")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-root", default="/tmp/rustwood_bench")
    ap.add_argument("--catboost", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not os.path.exists(RUSTWOOD_BIN):
        sys.exit(f"rustwood binary not found at {RUSTWOOD_BIN}; build it first")

    all_results = []
    for tok in args.sizes.split(","):
        tok = tok.strip()
        if tok == "california":
            X, y = load_california()
            all_results.append(bench_dataset("California Housing", X, y, 0.2,
                                             args.seed, args.gpu, args.data_root,
                                             args.catboost))
        else:
            n_train = int(tok)
            n_test = max(2000, n_train // 5)
            X, y = gen_synthetic(n_train + n_test, seed=args.seed)
            all_results.append(bench_dataset(f"Synthetic {n_train // 1000}K", X, y,
                                             n_test, args.seed, args.gpu,
                                             args.data_root, args.catboost))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
