#!/usr/bin/env python3
"""Benchmark rustwood vs XGBoost-GPU / LightGBM-GPU / baseline on REAL tabular datasets
(no synthetics): titanic, credit-g, bank-marketing, adult, covertype.

Each dataset is loaded from OpenML / sklearn, missing values imputed (numeric:
median; categorical: an explicit MISSING level), categoricals ordinal-integer-encoded,
and binarized to a 0/1 target. Every library trains on the *same* float32 features
(apples-to-apples). rustwood is also run with its out-of-fold target encoding on the
categorical columns (`rustwood+TE`) to show its native categorical handling.

Metric: test ROC-AUC. Also reports train wall-clock and inference time.
"""
import json
import os
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(HERE)
RUSTWOOD_BIN = os.path.join(EXAMPLE_DIR, "target", "release", "rustwood")
DATA_ROOT = "/tmp/rustwood_real"

TREES, DEPTH, LR, L2, MAXBIN = 300, 6, 0.1, 1.0, 256


# ----------------------------------------------------------------- dataset loaders
def load_titanic():
    d = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    X = d.data.drop(columns=["name", "ticket", "cabin", "boat", "body", "home.dest"],
                    errors="ignore")
    return X, (d.target.astype(int).values), "Titanic"


def load_creditg():
    d = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    return d.data, (d.target.values == "good").astype(int), "Credit-G"


def load_bank():
    d = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
    # classes are "1" (no, majority) and "2" (yes, subscribed = positive, minority).
    yb = (pd.Series(d.target).astype(str) == "2").astype(int).values
    return d.data, yb, "Bank-Marketing"


def load_adult():
    d = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    y = (pd.Series(d.target.values).astype(str).str.contains(">50K")).astype(int).values
    return d.data, y, "Adult"


def load_covtype():
    d = fetch_covtype()
    X = pd.DataFrame(d.data)
    y = (d.target == 2).astype(int)  # most frequent class vs rest (numpy array)
    return X, y, "Covertype-581k"


# ---------------------------------------------------------------- preprocessing
def preprocess(X: pd.DataFrame, y):
    """Impute + ordinal-encode categoricals -> float32. Returns (X32, y, cat_idx)."""
    X = X.copy()
    cat_idx = []
    for j, col in enumerate(X.columns):
        s = X[col]
        if s.dtype.kind in "Obc" or str(s.dtype) == "category":
            codes = pd.Series(s.astype("object")).fillna("__MISSING__").astype("category").cat.codes
            X[col] = codes.astype(np.float32)
            cat_idx.append(j)
        else:
            X[col] = pd.to_numeric(s, errors="coerce").astype(np.float32)
            med = np.nanmedian(X[col].values)
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)
    return X.values.astype(np.float32), np.asarray(y).astype(np.float32), cat_idx


def save_blobs(d, Xtr, ytr, Xte, yte):
    os.makedirs(d, exist_ok=True)
    np.ascontiguousarray(Xtr, np.float32).tofile(os.path.join(d, "x_train.bin"))
    np.ascontiguousarray(ytr, np.float32).tofile(os.path.join(d, "y_train.bin"))
    np.ascontiguousarray(Xte, np.float32).tofile(os.path.join(d, "x_test.bin"))
    np.ascontiguousarray(yte, np.float32).tofile(os.path.join(d, "y_test.bin"))
    json.dump({"n_train": len(Xtr), "n_test": len(Xte), "n_features": Xtr.shape[1]},
              open(os.path.join(d, "meta.json"), "w"))


# ----------------------------------------------------------------------- runners
def run_rustwood(data_dir, cat_idx=None):
    args = [RUSTWOOD_BIN, "--data", data_dir, "--objective", "logistic",
            "--trees", str(TREES), "--depth", str(DEPTH), "--lr", str(LR),
            "--bins", str(MAXBIN), "--lambda", str(L2), "--replicas", "64"]
    if cat_idx:
        args += ["--categorical", ",".join(map(str, cat_idx)), "--unique-bins", "1"]
    t0 = time.perf_counter()
    out = subprocess.run(args, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
    if not line:
        print(out.stdout[-800:], out.stderr[-800:]); raise RuntimeError("rustwood failed")
    kv = dict(t.split("=") for t in line.split()[1:])
    return {"train": float(kv["train_s"]), "pred": float(kv["pred_ms"]) / 1e3,
            "auc": float(kv["auc"]), "wall": wall}


def run_xgb(Xtr, ytr, Xte, yte):
    import xrustwood as xgb
    m = xgb.XGBClassifier(n_estimators=TREES, max_depth=DEPTH, learning_rate=LR,
                          reg_lambda=L2, tree_method="hist", device="cuda",
                          max_bin=MAXBIN, eval_metric="auc")
    t = time.perf_counter(); m.fit(Xtr, ytr); tr = time.perf_counter() - t
    t = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; pt = time.perf_counter() - t
    return {"train": tr, "pred": pt, "auc": roc_auc_score(yte, p)}


def run_lgbm(Xtr, ytr, Xte, yte):
    import lightgbm as lgb
    for dev in ("gpu", "cpu"):
        try:
            m = lgb.LGBMClassifier(n_estimators=TREES, max_depth=DEPTH, num_leaves=2 ** DEPTH,
                                   learning_rate=LR, reg_lambda=L2, max_bin=255,
                                   device_type=dev, verbose=-1)
            t = time.perf_counter(); m.fit(Xtr, ytr); tr = time.perf_counter() - t
            t = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; pt = time.perf_counter() - t
            return {"train": tr, "pred": pt, "auc": roc_auc_score(yte, p), "dev": dev}
        except Exception:
            continue
    raise RuntimeError("lightgbm failed")


def run_baseline(Xtr, ytr, Xte, yte):
    from baseline import BaselineRegressor
    m = BaselineRegressor(n_estimators=TREES, max_depth=DEPTH, learning_rate=LR,
                        n_bins=MAXBIN, l2=L2, backend="rust", random_state=42)
    t = time.perf_counter(); m.fit(Xtr, ytr); tr = time.perf_counter() - t
    t = time.perf_counter(); p = np.asarray(m.predict(Xte)).ravel(); pt = time.perf_counter() - t
    return {"train": tr, "pred": pt, "auc": roc_auc_score(yte, p)}


def main():
    if not os.path.exists(RUSTWOOD_BIN):
        raise SystemExit(f"build rustwood first: {RUSTWOOD_BIN}")
    datasets = [load_titanic, load_creditg, load_bank, load_adult, load_covtype]
    summary = []
    for loader in datasets:
        try:
            Xdf, y, name = loader()
            X, y, cat_idx = preprocess(Xdf, y)
        except Exception as e:  # noqa: BLE001
            print(f"[skip dataset {loader.__name__}] {type(e).__name__}: {str(e)[:90]}")
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        ddir = os.path.join(DATA_ROOT, name)
        save_blobs(ddir, Xtr, ytr, Xte, yte)
        print(f"\n{'='*76}\n{name}: train={len(Xtr):,} test={len(Xte):,} "
              f"features={X.shape[1]} categoricals={len(cat_idx)} pos_rate={y.mean():.3f}\n{'='*76}")

        rows = []
        for label, fn in [
            ("rustwood", lambda: run_rustwood(ddir)),
            ("rustwood+TE", lambda: run_rustwood(ddir, cat_idx) if cat_idx else None),
            ("XGBoost-GPU", lambda: run_xgb(Xtr, ytr, Xte, yte)),
            ("LightGBM-GPU", lambda: run_lgbm(Xtr, ytr, Xte, yte)),
            ("baseline(CPU)", lambda: run_baseline(Xtr, ytr, Xte, yte)),
        ]:
            try:
                r = fn()
                if r is None:
                    continue
                r["name"] = label
                rows.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip {label}] {type(e).__name__}: {str(e)[:90]}")

        hdr = f"{'model':<14}{'AUC':>9}{'train_s':>10}{'pred_ms':>10}"
        print(hdr); print("-" * len(hdr))
        best_auc = max(r["auc"] for r in rows)
        for r in rows:
            star = " *" if r["auc"] == best_auc else ""
            print(f"{r['name']:<14}{r['auc']:>9.4f}{r['train']:>10.3f}{r['pred']*1e3:>10.2f}{star}")
        summary.append((name, rows))

    print(f"\n\n{'#'*76}\nSUMMARY (test AUC)\n{'#'*76}")
    libs = ["rustwood", "rustwood+TE", "XGBoost-GPU", "LightGBM-GPU", "baseline(CPU)"]
    print(f"{'dataset':<18}" + "".join(f"{l:>13}" for l in libs))
    for name, rows in summary:
        by = {r["name"]: r for r in rows}
        print(f"{name:<18}" + "".join(
            f"{by[l]['auc']:>13.4f}" if l in by else f"{'-':>13}" for l in libs))


if __name__ == "__main__":
    main()
