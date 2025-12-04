# tl-tensor-examples

Examples and benchmarks for [tl-tensor](https://github.com/your-username/tl-tensor), a library for computing the Jones polynomial via tensor network contraction.

## Requirements

- `tl-tensor` (install from the main repo with `maturin develop --release`)
- `cotengra` for contraction path optimization
- `optuna` for hyperparameter tuning (recommended)
- `snappy` for knot/link manipulation (optional, for SnaPPy examples)

```bash
pip install cotengra optuna
conda install -c conda-forge snappy  # or pip install snappy
```

## Examples

### Torus Knots

`torus_knots.py` - Compute Jones polynomials for T(p,q) torus knots and benchmark performance.

### Rational Tangles (requires SnaPPy)

`rational_tangles.py` - Use SnaPPy to create rational tangle closures (2-bridge knots) and compute their Jones polynomials.

### Knot Census (requires SnaPPy)

`knot_census.py` - Compute Jones polynomials for knots from SnaPPy's knot tables.

## Benchmarks

On an ARM64 system (GH200), T(k,k) torus knots scale as follows:

| k  | Crossings | Tensors | log10(cost) | Opt(s) | Contract(s) |
|----|-----------|---------|-------------|--------|-------------|
| 3  | 6         | 7       | 2.5         | 0.35   | 0.000       |
| 4  | 12        | 13      | 3.2         | 0.15   | 0.000       |
| 5  | 20        | 21      | 4.0         | 0.17   | 0.001       |
| 6  | 30        | 31      | 4.6         | 0.21   | 0.004       |
| 7  | 42        | 43      | 5.3         | 0.29   | 0.018       |
| 8  | 56        | 57      | 6.0         | 0.37   | 0.091       |
| 9  | 72        | 73      | 6.6         | 0.50   | 0.470       |
| 10 | 90        | 91      | 7.3         | 0.65   | 2.696       |
| 11 | 110       | 111     | 7.9         | 0.86   | 12.484      |
| 12 | 132       | 133     | 8.6         | 1.09   | 74.819      |

Note: `kahypar` is not available on ARM64, so these benchmarks use the `greedy` method only.
