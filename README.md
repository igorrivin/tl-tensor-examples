# tl-tensor-examples

Fast knot identification and Jones polynomial computation using tensor networks.

This package combines [tl-tensor](https://github.com/igorrivin/tl-tensor) for tensor network computation with [SnaPPy](https://snappy.math.uic.edu/) for additional invariants, providing:

- **Hybrid Jones computation**: Auto-selects the fastest method based on braid width
- **Knot identification**: Match braids against known knots (≤12 crossings)
- **Multiple invariants**: Jones, Alexander, signature, and hyperbolic volume
- **Knot notation converters**: PD code, DT code, named knots to braid words
- **Braid utilities**: Random braid generation, component counting, LaTeX formatting
- **CLI tools**: `knot-identify` and `jones-compute` commands

## Installation

### Basic Installation (tl-tensor only)

```bash
pip install tl-tensor cotengra sympy
pip install -e .  # Install this package
```

### Full Installation with SnaPPy

SnaPPy provides additional invariants (Alexander polynomial, hyperbolic volume, signature) and is faster for narrow braids (≤5 strands).

#### Apple Silicon (M1/M2/M3)

```bash
# Create conda environment
conda create -n knots python=3.11
conda activate knots

# Install SnaPPy via conda (no ARM wheel on PyPI)
conda install -c conda-forge snappy

# Install tl-tensor and this package
pip install tl-tensor cotengra sympy
pip install -e .
```

#### Linux ARM64 (GH200, Graviton, etc.)

```bash
# SnaPPy via conda
conda install -c conda-forge snappy

# Or build from source if needed
pip install --no-deps snappy spherogram FXrays plink snappy_manifolds low_index
```

#### x86 (Intel/AMD)

```bash
pip install snappy  # PyPI wheel available
pip install tl-tensor cotengra sympy
pip install -e .
```

### Recommended: Topology Environment (Best Performance)

For optimal performance with both narrow and wide braids, create a unified "topology" environment that combines Sage+SnaPPy with tl-tensor:

```bash
# 1. Create base Sage environment
conda create -n sage sage snappy -c conda-forge

# 2. Clone to topology environment and add tl-tensor
conda create -n topology --clone sage
conda activate topology

# 3. Install tl-tensor (requires maturin and Rust)
pip install maturin cotengra sympy opt_einsum
cd /path/to/tl-tensor
maturin develop --release

# 4. Install tl-tensor-examples
pip install -e /path/to/tl-tensor-examples
```

This setup gives you:
- **Direct SnaPPy+Sage access**: No subprocess overhead for narrow braids
- **Full tl-tensor performance**: Tensor network computation for wide braids
- **Hybrid mode**: Automatically selects the fastest method

### Optional: KaHyPar Optimizer

[KaHyPar](https://github.com/kahypar/kahypar) provides an alternative hypergraph partitioning optimizer for tensor contraction. While greedy is typically faster for TL-algebra tensor networks, kahypar may be beneficial for other tensor network structures.

```bash
# Build from source (required for ARM64)
git clone --recursive https://github.com/kahypar/kahypar.git
cd kahypar
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKAHYPAR_PYTHON_INTERFACE=ON
make -j$(nproc)

# Copy the built module to your environment
cp python/kahypar*.so $(python -c "import site; print(site.getsitepackages()[0])")/
```

### Optional: Optuna for hyperparameter optimization

```bash
pip install optuna
```

## Quick Start

```python
from tl_examples import compute_jones, identify_braid, random_braid, has_kahypar

# Compute Jones polynomial
braid = [1, 1, 1]  # Trefoil (σ₁³)
jones = compute_jones(braid)
print(jones)  # [(-1, -16), (1, -12), (1, -4)]
# In t-variable: -t⁻⁴ + t⁻³ + t⁻¹

# Use different optimizers for tensor contraction
jones = compute_jones(braid, optimizer="greedy")  # Default, fastest for most cases
if has_kahypar():
    jones = compute_jones(braid, optimizer="kahypar")  # Hypergraph partitioning

# Identify a knot
matches = identify_braid(braid)
print(matches)  # ['K3a1']

# Generate random braids
braid = random_braid(n_strands=4, length=20, reduced=True)
```

## Knot Notation Converters

Convert from various knot notations to braid words (requires SnaPPy):

```python
from tl_examples import to_braid, from_name, from_pd_code, compute_jones

# From named knots (Rolfsen, Hoste-Thistlethwaite, torus)
braid = to_braid("4_1")      # Figure-8: [1, -2, 1, -2]
braid = to_braid("K8n3")     # HT notation
braid = to_braid("T(3,5)")   # Torus knot T(3,5)

# From PD code (Planar Diagram)
pd_code = [(5, 2, 0, 3), (3, 0, 4, 1), (1, 4, 2, 5)]  # Trefoil
braid = from_pd_code(pd_code)  # [-1, -1, -1]

# From DT code (Dowker-Thistlethwaite)
from tl_examples import from_dt_code
braid = from_dt_code("DT: [(4,6,2)]")  # Trefoil

# End-to-end: any notation -> Jones polynomial
jones = compute_jones(to_braid("5_1"))
```

The universal `to_braid()` function auto-detects notation type:
- Braid words: `[1, 1, 1]`
- Named knots: `"3_1"`, `"K4a1"`, `"T(2,7)"`
- PD codes: `[(a, b, c, d), ...]`
- DT codes: `"DT: [(4,6,2)]"`

## Performance Comparison: tl-tensor vs SnaPPy

The key factor is **network width (strand count)**, not braid length:

| Test Case | Strands | Length | tl-tensor | SnaPPy | Winner |
|-----------|---------|--------|-----------|--------|--------|
| Trefoil | 2 | 3 | 0.060s | 0.008s | SnaPPy (7x) |
| Figure-8 | 3 | 4 | 0.051s | 0.002s | SnaPPy (25x) |
| T(6,6) | 6 | 30 | 0.095s | 0.186s | **tl-tensor (2x)** |
| T(7,7) | 7 | 42 | 0.152s | 2.30s | **tl-tensor (15x)** |
| T(8,8) | 8 | 56 | 0.269s | 4.15s | **tl-tensor (15x)** |
| B₃ len=1000 | 3 | 1000 | 2.4s | 1.2s | SnaPPy (2x) |
| B₃ len=5000 | 3 | 5000 | 51s | 23s | SnaPPy (2x) |

**Rule of thumb**:
- **≤5 strands**: SnaPPy is ~2-25x faster
- **≥6 strands**: tl-tensor is 2-15x faster

The hybrid mode (`compute_jones_hybrid`) auto-selects the best method.

### Note: Sage Requirement for SnaPPy Jones/Alexander

SnaPPy's `jones_polynomial()` and `alexander_polynomial()` methods require **SageMath**. Without Sage:
- The hybrid mode automatically falls back to tl-tensor (which is always available)
- Knot notation converters (`to_braid`, `from_pd_code`, etc.) work fine
- Other invariants (signature, hyperbolic volume) work fine

### Using a Separate Sage Environment (Recommended)

For HPC/ML environments where installing Sage causes conflicts, the hybrid mode can **call out to a separate Sage conda environment**:

```bash
# 1. Create a separate Sage environment (one-time setup)
conda create -n sage sage snappy -c conda-forge
```

```python
# 2. Use hybrid mode - it auto-detects the Sage environment
from tl_examples import compute_jones_hybrid, set_sage_env, check_sage_env

# Check if Sage environment is available
if check_sage_env("sage"):
    print("Sage environment ready!")

# Optionally specify environment name
set_sage_env("my-sage-env")  # or set TL_SAGE_ENV env var

# Hybrid mode tries in order:
# 1. SnaPPy in current env (if Sage available)
# 2. SnaPPy via separate Sage environment
# 3. tl-tensor (always available)
result = compute_jones_hybrid([1, 1, 1])
print(result['method'])  # 'sage-env' if using separate environment
```

This approach gives you the best of both worlds:
- **Clean HPC environment**: No Sage dependency conflicts with PyTorch/CUDA
- **Fast narrow braids**: SnaPPy+Sage performance when beneficial
- **Graceful fallback**: tl-tensor handles everything if Sage unavailable

## Knot Identification

### Using Jones Polynomial

```python
from tl_examples import identify_braid

# Trefoil
matches = identify_braid([1, 1, 1])
print(matches)  # ['K3a1']

# Unknot (Jones = 1)
matches = identify_braid([1, 2, -1, -2])
print(matches)  # ['Unknot']
```

### Using Multiple Invariants

When Jones polynomial has collisions, use additional invariants:

```python
from tl_examples import identify_with_invariants

result = identify_with_invariants([1, -2, 1, -2])  # Figure-8
print(result['jones_matches'])  # ['K4a1', 'K11n19'] - same Jones!
print(result['volume'])         # 2.029... (hyperbolic volume)
print(result['signature'])      # 0
```

The invariant cascade for disambiguation:
```
Jones → Alexander → Signature → Volume
```

- **Jones**: Fast (tl-tensor), catches most knots
- **Alexander**: Resolves ~10% of Jones collisions
- **Signature**: Works for all knots including torus knots
- **Volume**: Resolves ~95% of remaining collisions (hyperbolic knots only)

## Variable Conventions

tl-tensor and SnaPPy use different variables:

| System | Variable | Relationship |
|--------|----------|--------------|
| tl-tensor | x | t = x⁴ |
| SnaPPy | q | q² = t |
| Standard | t | — |

Conversion: **q = x²**

### Chirality Convention

tl-tensor and SnaPPy use **opposite chirality** for braid generators:
- `[1, 1, 1]` in tl-tensor = left-hand trefoil
- `[1, 1, 1]` in SnaPPy = right-hand trefoil

For achiral knots (like figure-8), both systems agree exactly.

## Braid Utilities

```python
from tl_examples import random_braid, num_components, writhe, torus_braid
from tl_examples import braid_to_latex, jones_to_latex

# Random braid generation
braid = random_braid(n_strands=4, length=30, reduced=True)

# Component counting
nc = num_components([1, -1])  # 2 (unlink, not unknot!)
nc = num_components([1, 2, -1, -2])  # 1 (unknot)

# Torus knots
braid = torus_braid(3, 5)  # T(3,5) torus knot

# LaTeX formatting
latex = braid_to_latex([1, 1, 1], compact=True)  # '\sigma_1^{3}'
latex = jones_to_latex(jones, 't')  # '-t^{-4} + t^{-3} + t^{-1}'
```

## Important Notes

### The Unknot Trap

**Warning**: `[1, -1]` is NOT the unknot! It's a 2-component unlink.

```python
num_components([1, -1])      # 2 - two separate circles!
num_components([1, 2, -1, -2])  # 1 - true unknot

# Jones polynomial
compute_jones([1, -1])      # [(-1, -2), (-1, 2)] - NOT 1!
compute_jones([1, 2, -1, -2])  # [(1, 0)] - equals 1 ✓
```

The unknot has Jones polynomial = 1 (i.e., `[(1, 0)]` in tl-tensor format).

### Component Parity

For B_n braids, the number of components in the closure has the same parity as n:

| Strands | Possible components |
|---------|---------------------|
| 3 | 1, 3 |
| 4 | 2, 4 |
| 5 | 1, 3, 5 |
| n | same parity as n |

## Examples

### Torus Knots Benchmark

```bash
python examples/torus_knots.py
```

### Random Braid Experiments

```bash
# Component statistics
python examples/random_braids.py -n 3 -l 30 --stats --count 1000

# Generate only knots
python examples/random_braids.py -n 3 -l 30 --knot
```

### B₃ Knot Distribution

```bash
python examples/b3_experiments.py --sweep-range 10 60 5 -n 1000
```

## Command Line Interface

After installation, two CLI commands are available:

```bash
# Identify a knot from its braid word
knot-identify 1 1 1
# Output: K3a1

knot-identify 1 1 1 --verbose
# Output: Braid: [1, 1, 1]
#         Jones polynomial: [(-1, -16), (1, -12), (1, -4)]
#         Identified as: ['K3a1']

# Use multiple invariants for disambiguation
knot-identify 1 -2 1 -2 --invariants

# Compute Jones polynomial
jones-compute 1 1 1
# Output: [(-1, -16), (1, -12), (1, -4)]

jones-compute 1 1 1 --latex --variable t
# Output: \frac{1}{t} + \frac{1}{t^{3}} - \frac{1}{t^{4}}

# Use hybrid mode (auto-selects fastest method)
jones-compute 1 2 3 4 5 6 1 2 3 4 5 6 --hybrid

# Choose optimizer for tensor contraction
jones-compute 1 1 1 --optimizer greedy   # Default, fastest for TL-algebra
jones-compute 1 1 1 --optimizer kahypar  # Hypergraph partitioning (if installed)
jones-compute 1 1 1 --optimizer auto     # Try kahypar, fall back to greedy
```

## Building Databases

To rebuild the knot databases:

```python
from tl_examples.identification import build_jones_database, save_database

# Requires SnaPPy
db = build_jones_database(max_crossings=12)
save_database(db)
```

## License

MIT

## Citation

If you use this software, please cite:
- tl-tensor: [citation]
- SnaPPy: Culler, Dunfield, Goerner, Weeks, et al.
