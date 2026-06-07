#!/usr/bin/env python3
"""Generate a binary-classification dataset and dump flat f32 blobs for rustwood.

Layout (little-endian f32, row-major features):
  x_train.bin  [n_train * n_features]
  y_train.bin  [n_train]
  x_test.bin   [n_test  * n_features]
  y_test.bin   [n_test]
  meta.json    {n_train, n_test, n_features, ...}

Usage:
  python gen_data.py --out data --n 1000000 --test 200000 --features 50 \
      --informative 24 --seed 0
"""
import argparse
import json
import os

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=1_000_000, help="total rows before split")
    ap.add_argument("--test", type=int, default=200_000)
    ap.add_argument("--features", type=int, default=50)
    ap.add_argument("--informative", type=int, default=24)
    ap.add_argument("--redundant", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"generating make_classification n={args.n} f={args.features} ...", flush=True)
    X, y = make_classification(
        n_samples=args.n,
        n_features=args.features,
        n_informative=args.informative,
        n_redundant=args.redundant,
        n_clusters_per_class=3,
        flip_y=0.02,
        class_sep=0.9,
        random_state=args.seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test, random_state=args.seed
    )
    # Contiguous row-major for both Rust (flat read) and the Python baselines.
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
    Xte = np.ascontiguousarray(Xte, dtype=np.float32)
    ytr = np.ascontiguousarray(ytr, dtype=np.float32)
    yte = np.ascontiguousarray(yte, dtype=np.float32)

    os.makedirs(args.out, exist_ok=True)
    Xtr.tofile(os.path.join(args.out, "x_train.bin"))
    ytr.tofile(os.path.join(args.out, "y_train.bin"))
    Xte.tofile(os.path.join(args.out, "x_test.bin"))
    yte.tofile(os.path.join(args.out, "y_test.bin"))

    meta = {
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "positive_rate_train": float(ytr.mean()),
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote", args.out, meta, flush=True)


if __name__ == "__main__":
    main()
