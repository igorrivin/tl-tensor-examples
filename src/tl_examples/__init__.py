"""
tl-tensor-examples: Knot identification and Jones polynomial tools.

This package provides:
- Fast Jones polynomial computation via tl-tensor (tensor networks)
- Hybrid computation (auto-selects fastest method based on braid width)
- Knot identification using Jones, Alexander, signature, and volume
- Knot notation converters (PD code, DT code, named knots to braid)
- Braid utilities and LaTeX formatting

Installation:
    pip install tl-tensor-examples

    # For SnaPPy support (converters, Alexander polynomial, volume, signature):
    # On Apple Silicon / ARM64:
    conda install -c conda-forge snappy
    # On x86:
    pip install snappy

Quick Start:
    from tl_examples import compute_jones, identify_braid, to_braid

    # Compute Jones polynomial from braid
    braid = [1, 1, 1]  # Trefoil
    jones = compute_jones(braid)

    # Or from any knot notation (requires SnaPPy)
    braid = to_braid("4_1")  # Figure-8 by name
    braid = to_braid("T(3,5)")  # Torus knot

    # Identify a knot
    matches = identify_braid(braid)
"""

__version__ = "0.1.0"

from .jones import (
    compute_jones,
    compute_jones_hybrid,
    compute_jones_sage_env,
    check_sage_env,
    set_sage_env,
    has_kahypar,
)
from .braids import random_braid, num_components, writhe, torus_braid, n_strands
from .identification import identify_braid, identify_with_invariants
from .latex import braid_to_latex, jones_to_latex

# Converters require SnaPPy - import conditionally
try:
    from .converters import (
        to_braid,
        from_pd_code,
        from_dt_code,
        from_name,
        get_link_info,
    )

    _HAS_CONVERTERS = True
except ImportError:
    _HAS_CONVERTERS = False

    def _converter_not_available(*args, **kwargs):
        raise ImportError(
            "Knot notation converters require SnaPPy. "
            "Install via: conda install -c conda-forge snappy"
        )

    to_braid = from_pd_code = from_dt_code = from_name = get_link_info = (
        _converter_not_available
    )

__all__ = [
    # Jones computation
    "compute_jones",
    "compute_jones_hybrid",
    "compute_jones_sage_env",
    "check_sage_env",
    "set_sage_env",
    "has_kahypar",
    # Braid utilities
    "random_braid",
    "num_components",
    "writhe",
    "torus_braid",
    "n_strands",
    # Identification
    "identify_braid",
    "identify_with_invariants",
    # LaTeX
    "braid_to_latex",
    "jones_to_latex",
    # Converters (require SnaPPy)
    "to_braid",
    "from_pd_code",
    "from_dt_code",
    "from_name",
    "get_link_info",
]
