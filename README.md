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

### Random Braids

`random_braids.py` - Generate random braid words and compute their Jones polynomials.

```bash
# Single random braid
python random_braids.py -n 3 -l 30 --seed 42

# Generate only knots (single component)
python random_braids.py -n 3 -l 30 --knot

# Non-reduced words (allows consecutive inverses)
python random_braids.py -n 3 -l 30 --general

# Component statistics
python random_braids.py -n 4 -l 20 --stats --count 1000
```

**Note on components:** The number of components in a braid closure is determined by the permutation's cycle structure. On n strands, the parity constrains which component counts are possible:

| Strands | Possible components |
|---------|---------------------|
| 3       | 1, 3                |
| 4       | 2, 4                |
| 5       | 1, 3, 5             |
| n       | same parity as n    |

### LaTeX Utilities

`latex_utils.py` - Format braids and Jones polynomials as LaTeX.

```python
from latex_utils import braid_to_latex, jones_to_latex, jones_to_sympy

# Braid word to LaTeX
braid_to_latex([1, 1, 1])           # '\sigma_1 \sigma_1 \sigma_1'
braid_to_latex([1, 1, 1], compact=True)  # '\sigma_1^{3}'

# Jones polynomial to LaTeX (t = x^4)
jones_to_latex([(1, -8), (-1, -4), (1, 0)], 't')  # 't^{-2} - t^{-1} + 1'

# Get sympy expression for further manipulation
poly = jones_to_sympy(jones, 't')
```

Use `--latex` flag with random_braids.py:
```bash
python random_braids.py -n 3 -l 20 --knot --latex
```

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

## B_3 Experiments

`b3_experiments.py` - Investigate knot types in random B_3 braids.

```bash
python b3_experiments.py -l 20 -n 1000           # Single length
python b3_experiments.py --sweep-range 10 60 5   # Length sweep
```

Key findings:
- **Odd length braids**: Always knots (100%)
- **Even length braids**: ~70% knots, ~30% 3-component links
- **Unknot is rare**: <1% even at short lengths
- **Most knots are "unknown"**: Jones polynomials don't match knots ≤12 crossings
- **Nearly all distinct**: At length 50+, almost every braid has a unique Jones polynomial

### Knot Identification

`knot_identification.py` - Match braids against known knots via Jones polynomial.

```bash
python knot_identification.py --build --max-crossings 12  # Build database
python knot_identification.py --braid "1,1,1"             # Identify a braid
python knot_identification.py --stats                     # Show statistics
```

**Caveat**: The Jones polynomial from tl-tensor depends on the braid presentation, not just the knot type. Different braid representatives of the same knot may give different polynomials. This affects unknot detection especially.

### Variable Conventions

tl-tensor and SnaPPy use different variables for the Jones polynomial:
- **tl-tensor**: Uses x where t = x⁴ (so x = t^{1/4})
- **SnaPPy**: Uses q where q = t^{1/2}

The relationship is: **q = x²** or equivalently **q² = t**

After conversion, tl-tensor and SnaPPy give identical results:
```
K4a1 (figure-8):
  SnaPPy (q): q⁻⁴ - q⁻² + 1 - q² + q⁴
  tl-tensor:  t² - t + 1 - 1/t + 1/t²  (same!)
```

### Alexander Polynomial (requires Sage)

SnaPPy can compute Alexander polynomials when run inside Sage. Install snappy in the sage environment:

```bash
# In sage conda environment
pip install --no-deps snappy spherogram FXrays plink snappy_manifolds low_index
# Create cypari shim (see ~/devel/jones/Dockerfile for details)
```

Then use it to distinguish knots with the same Jones polynomial:
```python
import snappy
K4a1 = snappy.Link('K4a1')
K11n19 = snappy.Link('K11n19')

# Same Jones polynomial
print(K4a1.jones_polynomial())   # q^-4 - q^-2 + 1 - q^2 + q^4
print(K11n19.jones_polynomial()) # q^-4 - q^-2 + 1 - q^2 + q^4

# Different Alexander polynomials!
print(K4a1.alexander_polynomial())   # t^2 - 3*t + 1
print(K11n19.alexander_polynomial()) # t^6 - 2*t^5 + t^3 - 2*t + 1
```
