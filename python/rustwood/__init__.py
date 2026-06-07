"""Thin Python wrapper around the rustwood binary (GPU + CPU).

A sklearn-style API that shells out to ``target/release/rustwood``. Training runs on the
GPU or CPU (``device=``); prediction loads the saved ``.rwood`` model and runs the host
scorer. Data is exchanged via temporary little-endian f32 blobs (negligible vs training).

    from rustwood import RustwoodRegressor, RustwoodClassifier, load
    m = RustwoodRegressor(n_trees=500, device="gpu").fit(X, y)
    p = m.predict(Xte)
    m.save("model.rwood")
    m2 = load("model.rwood")            # predicts on CPU, no GPU needed
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import tempfile

import numpy as np

__all__ = ["RustwoodRegressor", "RustwoodClassifier", "load", "find_binary"]


def find_binary() -> str:
    """Locate the rustwood executable (``RUSTWOOD_BIN`` env, bundled, repo build, or PATH)."""
    env = os.environ.get("RUSTWOOD_BIN")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "rustwood"),
              os.path.join(here, "..", "..", "target", "release", "rustwood")):
        c = os.path.abspath(c)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    w = shutil.which("rustwood")
    if w:
        return w
    raise RuntimeError("rustwood binary not found; set RUSTWOOD_BIN or run ./build.sh")


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _write_blobs(d, xtr, ytr, xte, yte):
    _f32(xtr).tofile(os.path.join(d, "x_train.bin"))
    _f32(ytr).ravel().tofile(os.path.join(d, "y_train.bin"))
    _f32(xte).tofile(os.path.join(d, "x_test.bin"))
    _f32(yte).ravel().tofile(os.path.join(d, "y_test.bin"))
    json.dump({"n_train": len(ytr), "n_test": len(yte), "n_features": int(xtr.shape[1])},
              open(os.path.join(d, "meta.json"), "w"))


def _run(args):
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"rustwood failed:\n{out.stdout[-800:]}\n{out.stderr[-800:]}")
    return out.stdout


class _Server:
    """A persistent ``rustwood --serve`` worker. Holds one resident CUDA context, so the
    ~400 ms init is paid once per process instead of per fit. Shared across all models in
    the session; set ``RUSTWOOD_NO_SERVER=1`` to fall back to one-shot subprocesses."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None or cls._instance.p.poll() is not None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.p = subprocess.Popen([find_binary(), "--serve", "1"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def run(self, args):
        self.p.stdin.write("\t".join(args) + "\n")
        self.p.stdin.flush()
        out = []
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if line == "__END__":
                break
            out.append(line)
        resp = out[-1] if out else ""
        if not resp.startswith("OK"):
            raise RuntimeError("rustwood serve error: " + "\n".join(out[-5:]))
        return resp


def _exec(args):
    """Run an arg list through the persistent worker (or one-shot if disabled). Returns the
    worker's ``OK ...`` line (or the one-shot stdout)."""
    if os.environ.get("RUSTWOOD_NO_SERVER"):
        return _run([find_binary()] + args)
    return _Server.get().run(args)


import atexit


@atexit.register
def _shutdown_server():
    s = _Server._instance
    if s is not None and s.p.poll() is None:
        try:
            s.p.stdin.write("QUIT\n")
            s.p.stdin.flush()
            s.p.wait(timeout=2)
        except Exception:
            s.p.terminate()


class _Model:
    _objective = "l2"

    def __init__(self, n_trees=500, depth=6, learning_rate=0.1, n_bins=256, l2=1.0,
                 subsample=1.0, goss_top=0.0, goss_other=0.1, min_child=1e-3,
                 categorical=None, device="gpu"):
        self.n_trees, self.depth, self.learning_rate = n_trees, depth, learning_rate
        self.n_bins, self.l2, self.subsample = n_bins, l2, subsample
        self.goss_top, self.goss_other, self.min_child = goss_top, goss_other, min_child
        self.categorical = list(categorical) if categorical else []
        self.device = device
        self._bin = find_binary()
        self._model_path = None
        self._owns_model = False
        self.n_features_ = None

    def _train_args(self):
        a = ["--trees", str(self.n_trees), "--depth", str(self.depth),
             "--lr", str(self.learning_rate), "--bins", str(self.n_bins),
             "--lambda", str(self.l2), "--subsample", str(self.subsample),
             "--goss-top", str(self.goss_top), "--goss-other", str(self.goss_other),
             "--min-child", str(self.min_child), "--device", self.device,
             "--objective", self._objective]
        if self.categorical:
            a += ["--categorical", ",".join(map(str, self.categorical)), "--unique-bins", "1"]
        return a

    def fit(self, X, y):
        X, y = _f32(X), _f32(y)
        self.n_features_ = X.shape[1]
        fd, self._model_path = tempfile.mkstemp(suffix=".rwood")
        os.close(fd)
        self._owns_model = True
        with tempfile.TemporaryDirectory() as d:
            _write_blobs(d, X, y, X[:2], y[:2])  # dummy test set to satisfy the loader
            resp = _exec(["--data", d] + self._train_args() + ["--save-model", self._model_path])
        m = re.search(r"train_s=([0-9.]+)", resp)
        self.train_time_ = float(m.group(1)) if m else None  # GPU/CPU compute time (s)
        return self

    def _raw_predict(self, X):
        if self._model_path is None:
            raise RuntimeError("model is not fitted / loaded")
        X = _f32(X)
        with tempfile.TemporaryDirectory() as d:
            preds = os.path.join(d, "p.bin")
            z = np.zeros(max(2, len(X)), np.float32)
            _write_blobs(d, X[:2], z[:2], X, z[:len(X)])  # X as the test set
            _exec(["--data", d, "--load-model", self._model_path, "--dump-pred", preds])
            return np.fromfile(preds, np.float32)

    def save(self, path):
        if self._model_path is None:
            raise RuntimeError("model is not fitted")
        shutil.copyfile(self._model_path, path)
        return self

    def __del__(self):
        if getattr(self, "_owns_model", False) and self._model_path and os.path.exists(self._model_path):
            try:
                os.remove(self._model_path)
            except OSError:
                pass


class RustwoodRegressor(_Model):
    _objective = "l2"

    def predict(self, X):
        return self._raw_predict(X)


class RustwoodClassifier(_Model):
    _objective = "logistic"

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self._raw_predict(X)))  # logistic link on the raw margins
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


def load(path):
    """Load a ``.rwood`` model. Returns a regressor or classifier per the stored objective.
    Prediction runs on the host (CPU), so no GPU is required."""
    with open(path, "rb") as f:
        if f.read(4) != b"RWD1":
            raise ValueError("not a .rwood file")
        _ver, n_features, _n_trees, _depth, obj = struct.unpack("<5I", f.read(20))
    cls = RustwoodClassifier if obj == 1 else RustwoodRegressor
    m = cls.__new__(cls)
    m._bin = find_binary()
    m._model_path = path
    m._owns_model = False
    m.n_features_ = n_features
    return m
