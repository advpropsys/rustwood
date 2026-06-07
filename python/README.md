# rustwood (Python)

Thin wrapper around the `rustwood` GPU/CPU oblivious-tree gradient booster. Requires the
`rustwood` binary (build it with `./build.sh`, or set `RUSTWOOD_BIN`).

```python
from rustwood import RustwoodRegressor, RustwoodClassifier, load
m = RustwoodRegressor(n_trees=500, device="gpu").fit(X, y)
p = m.predict(Xte)
m.save("model.rwood"); m2 = load("model.rwood")   # predicts on CPU
```
