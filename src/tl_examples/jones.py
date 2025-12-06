"""
Jones polynomial computation via tl-tensor and hybrid methods.
"""

import json
import os
import subprocess
from typing import Literal, Optional
import warnings

warnings.filterwarnings("ignore")

# tl-tensor is required
from tl_tensor import TLTensorNetwork, CotengraOptimizer

# SnaPPy is optional (for narrow braids where it's faster)
try:
    import snappy

    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False

# KaHyPar is optional (alternative optimizer for tensor contraction)
try:
    import kahypar

    HAS_KAHYPAR = True
except ImportError:
    HAS_KAHYPAR = False


# Crossover point: use SnaPPy for braids with <= this many strands
DEFAULT_STRAND_THRESHOLD = 5

# Valid optimizer choices
OptimizerType = Literal["greedy", "kahypar", "auto"]


def has_kahypar() -> bool:
    """Check if kahypar optimizer is available."""
    return HAS_KAHYPAR

# Sage environment configuration
# Can be set via environment variable or programmatically
SAGE_ENV_NAME = os.environ.get("TL_SAGE_ENV", "sage")
CONDA_PREFIX = os.environ.get("CONDA_PREFIX_1", os.environ.get("CONDA_PREFIX", ""))

# Cache for Sage environment availability
_sage_env_available: Optional[bool] = None


def compute_jones(
    braid: list[int],
    max_repeats: int = 32,
    methods: list[str] = None,
    optimizer: OptimizerType = "greedy",
) -> list[tuple[int, int]]:
    """
    Compute Jones polynomial using tl-tensor.

    Args:
        braid: Braid word as list of generators (e.g., [1, 1, 1] for trefoil)
        max_repeats: Number of optimization repeats for cotengra
        methods: Optimization methods (deprecated, use optimizer instead)
        optimizer: Optimization strategy:
            - "greedy": Fast greedy algorithm (default, recommended for most cases)
            - "kahypar": Hypergraph partitioning (requires kahypar package)
            - "auto": Try kahypar if available, fall back to greedy

    Returns:
        Jones polynomial as list of (coefficient, exponent) tuples in x-variable
        where t = x^4 (standard Jones variable t satisfies t = x^4)

    Example:
        >>> compute_jones([1, 1, 1])  # Trefoil
        [(-1, -16), (1, -12), (1, -4)]
        # This is -x^{-16} + x^{-12} + x^{-4} = -t^{-4} + t^{-3} + t^{-1}

        >>> compute_jones([1, 1, 1], optimizer="kahypar")  # Use kahypar
    """
    # Handle methods parameter for backwards compatibility
    if methods is not None:
        opt_methods = methods
    elif optimizer == "kahypar":
        if not HAS_KAHYPAR:
            raise ImportError(
                "kahypar optimizer requested but kahypar is not installed. "
                "Install from source: https://github.com/kahypar/kahypar"
            )
        opt_methods = ["kahypar"]
    elif optimizer == "auto":
        opt_methods = ["kahypar"] if HAS_KAHYPAR else ["greedy"]
    else:
        opt_methods = ["greedy"]

    network = TLTensorNetwork.from_word(braid)
    opt = CotengraOptimizer(max_repeats=max_repeats, methods=opt_methods)
    info = network.contract_info(optimize=opt)
    contracted = network.contract(optimize=info.path)
    return contracted.tensors[0].terms[0][1]


def check_sage_env(env_name: Optional[str] = None) -> bool:
    """
    Check if a Sage conda environment is available.

    Args:
        env_name: Name of the conda environment (default: from TL_SAGE_ENV or "sage")

    Returns:
        True if the environment exists and has Sage+SnaPPy
    """
    global _sage_env_available

    if env_name is None:
        env_name = SAGE_ENV_NAME

    # Use cached result if checking default env
    if env_name == SAGE_ENV_NAME and _sage_env_available is not None:
        return _sage_env_available

    try:
        # Check if conda env exists and has snappy with sage
        result = subprocess.run(
            f'conda run -n {env_name} python -c "import snappy; snappy.Link(\'3_1\').jones_polynomial()"',
            shell=True,
            capture_output=True,
            timeout=30,
        )
        available = result.returncode == 0

        if env_name == SAGE_ENV_NAME:
            _sage_env_available = available

        return available
    except Exception:
        if env_name == SAGE_ENV_NAME:
            _sage_env_available = False
        return False


def set_sage_env(env_name: str) -> None:
    """
    Set the Sage environment name to use for hybrid computation.

    Args:
        env_name: Name of the conda environment with Sage+SnaPPy

    Example:
        >>> set_sage_env("my-sage-env")
        >>> result = compute_jones_hybrid([1,1,1])  # Will try my-sage-env
    """
    global SAGE_ENV_NAME, _sage_env_available
    SAGE_ENV_NAME = env_name
    _sage_env_available = None  # Reset cache


def compute_jones_sage_env(
    braid: list[int],
    env_name: Optional[str] = None,
) -> Optional[list[tuple[int, int]]]:
    """
    Compute Jones polynomial by calling SnaPPy in a separate Sage environment.

    This allows using SnaPPy's fast Jones computation without installing
    Sage in the current environment (which often causes dependency conflicts).

    Args:
        braid: Braid word as list of generators
        env_name: Conda environment name (default: from TL_SAGE_ENV or "sage")

    Returns:
        Jones polynomial as list of (coefficient, exponent) tuples in x-variable,
        or None if the Sage environment is not available

    Note:
        The result is converted to tl-tensor's x-variable format (t = x^4).
        Set the environment name via set_sage_env() or TL_SAGE_ENV env var.
    """
    if env_name is None:
        env_name = SAGE_ENV_NAME

    # Build a single-line Python command that's shell-safe
    braid_str = json.dumps(braid)
    # Single-line Python code (uses semicolons instead of newlines)
    # SnaPPy returns in q-variable where q = x^2 (and t = q^2 = x^4)
    # j.dict() gives {exponent: coefficient} in q-variable
    # We convert q^n -> x^(2n), then negate exponents to match tl-tensor chirality
    # (tl-tensor and SnaPPy use opposite chirality conventions)
    python_code = (
        f"import snappy, json; "
        f"K = snappy.Link(braid_closure={braid_str}); "
        f"j = K.jones_polynomial(); "
        f"d = j.dict(); "
        f"print(json.dumps([(int(c), -int(e)*2) for e, c in d.items()]))"
    )

    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", python_code],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return None

        # Parse the JSON output
        output = result.stdout.strip()
        if not output:
            return None
        jones_list = json.loads(output)
        return [(c, e) for c, e in jones_list]

    except Exception:
        return None


def compute_jones_snappy(braid: list[int]) -> Optional[str]:
    """
    Compute Jones polynomial using SnaPPy.

    Args:
        braid: Braid word as list of generators

    Returns:
        Jones polynomial as string in SnaPPy's q-variable format,
        or None if SnaPPy not available

    Note:
        SnaPPy uses q where q = t^{1/2}, so q^2 = t.
        tl-tensor uses x where t = x^4, so q = x^2.
    """
    if not HAS_SNAPPY:
        return None

    try:
        K = snappy.Link(braid_closure=braid)
        return str(K.jones_polynomial())
    except Exception:
        return None


def compute_jones_hybrid(
    braid: list[int],
    strand_threshold: int = DEFAULT_STRAND_THRESHOLD,
    prefer_snappy: bool = True,
    use_sage_env: bool = True,
    sage_env_name: Optional[str] = None,
    max_repeats: int = 32,
    optimizer: OptimizerType = "greedy",
) -> dict:
    """
    Compute Jones polynomial using the optimal method.

    Automatically selects between tl-tensor and SnaPPy based on braid width:
    - Narrow braids (few strands): SnaPPy+Sage is ~2-25x faster
    - Wide braids (many strands): tl-tensor is 2-15x faster

    For narrow braids, tries in order:
    1. SnaPPy in current environment (if Sage available)
    2. SnaPPy via separate Sage conda environment (if use_sage_env=True)
    3. tl-tensor (always available)

    Args:
        braid: Braid word as list of generators
        strand_threshold: Use SnaPPy for braids with <= this many strands
        prefer_snappy: If True, try SnaPPy first for narrow braids
        use_sage_env: If True, try calling a separate Sage environment
        sage_env_name: Conda env name for Sage (default: TL_SAGE_ENV or "sage")
        max_repeats: Cotengra optimization repeats (for tl-tensor)
        optimizer: Optimization strategy for tl-tensor:
            - "greedy": Fast greedy algorithm (default, recommended for most cases)
            - "kahypar": Hypergraph partitioning (requires kahypar package)
            - "auto": Try kahypar if available, fall back to greedy

    Returns:
        dict with:
        - 'result': Jones polynomial in x-variable [(coeff, exp), ...]
        - 'method': 'tl-tensor', 'snappy', or 'sage-env'
        - 'n_strands': Number of strands in braid
        - 'optimizer': Optimizer used (if method is 'tl-tensor')

    Example:
        >>> result = compute_jones_hybrid([1, 1, 1])
        >>> result['method']
        'sage-env'  # 2-strand braid, used Sage environment
        >>> result = compute_jones_hybrid([1,2,3,4,5,6] * 3)
        >>> result['method']
        'tl-tensor'  # 7-strand braid, tl-tensor is faster

    Environment Configuration:
        Set the Sage environment name via:
        - TL_SAGE_ENV environment variable
        - set_sage_env("my-sage-env") function
        - sage_env_name parameter
    """
    n_strands = max(abs(g) for g in braid) + 1

    # For narrow braids, try SnaPPy options
    if prefer_snappy and n_strands <= strand_threshold:
        # Option 1: SnaPPy in current environment (requires Sage)
        if HAS_SNAPPY:
            result = compute_jones_snappy(braid)
            if result is not None:
                return {"result": result, "method": "snappy", "n_strands": n_strands}

        # Option 2: Call separate Sage environment
        if use_sage_env:
            result = compute_jones_sage_env(braid, env_name=sage_env_name)
            if result is not None:
                return {"result": result, "method": "sage-env", "n_strands": n_strands}

    # Option 3: Use tl-tensor (always available, optimal for wide braids)
    # Determine actual optimizer used
    if optimizer == "auto":
        actual_optimizer = "kahypar" if HAS_KAHYPAR else "greedy"
    else:
        actual_optimizer = optimizer

    result = compute_jones(braid, max_repeats=max_repeats, optimizer=optimizer)
    return {
        "result": result,
        "method": "tl-tensor",
        "n_strands": n_strands,
        "optimizer": actual_optimizer,
    }


def jones_to_t_variable(jones: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Convert Jones polynomial from x-variable to t-variable.

    tl-tensor outputs in x where t = x^4, so exponents divide by 4.

    Args:
        jones: Jones polynomial in x-variable [(coeff, x_exp), ...]

    Returns:
        Jones polynomial in t-variable [(coeff, t_exp), ...]

    Note:
        Only works if all x-exponents are divisible by 4.
    """
    result = []
    for coeff, x_exp in jones:
        if x_exp % 4 != 0:
            raise ValueError(f"Exponent {x_exp} not divisible by 4")
        result.append((coeff, x_exp // 4))
    return result


def jones_to_q_variable(jones: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Convert Jones polynomial from x-variable to q-variable (SnaPPy convention).

    tl-tensor uses x where t = x^4, SnaPPy uses q where q = t^{1/2} = x^2.

    Args:
        jones: Jones polynomial in x-variable [(coeff, x_exp), ...]

    Returns:
        Jones polynomial in q-variable [(coeff, q_exp), ...]

    Note:
        Only works if all x-exponents are divisible by 2.
    """
    result = []
    for coeff, x_exp in jones:
        if x_exp % 2 != 0:
            raise ValueError(f"Exponent {x_exp} not divisible by 2")
        result.append((coeff, x_exp // 2))
    return result


def compute_jones_fast(
    braid: list[int],
    crossing_threshold: Optional[int] = None,
) -> dict:
    """
    Compute Jones polynomial using the fastest available method.

    This is the recommended function for optimal performance. It automatically
    selects between:
    - Numba-optimized Kauffman bracket state sum (for small diagrams)
    - tl-tensor network contraction (for larger/wider braids)

    The crossover point is ~16 crossings with Numba, ~12 without.

    Performance characteristics:
    - Small diagrams (≤16 crossings): Numba state sum ~0.3-7ms
    - Large/wide braids: tl-tensor ~2-20ms (polynomial in width)
    - tl-tensor scales with braid width, not crossing count
    - State sum is exponential in crossings but has low overhead

    Args:
        braid: Braid word as list of generators (e.g., [1, 1, 1] for trefoil)
        crossing_threshold: Override automatic threshold (default: 16 with Numba, 12 without)

    Returns:
        dict with:
        - 'jones': Jones polynomial as [(coeff, exp), ...] in x-variable (t = x^4)
        - 'method': 'numba-statesum', 'statesum', or 'tl-tensor'
        - 'crossings': Number of crossings in the diagram
        - 'strands': Number of strands in the braid

    Example:
        >>> result = compute_jones_fast([1, 1, 1])  # Trefoil - uses state sum
        >>> result['jones']
        [(-1, -16), (1, -12), (1, -4)]
        >>> result['method']
        'numba-statesum'

        >>> result = compute_jones_fast(torus_braid(10, 10))  # 90 crossings
        >>> result['method']
        'tl-tensor'  # State sum would be exponential, tensor is fast
    """
    from .kauffman import (
        jones_from_pd_code_numba,
        jones_from_pd_code,
        STATESUM_CROSSING_THRESHOLD,
        _HAS_NUMBA,
    )

    if crossing_threshold is None:
        crossing_threshold = STATESUM_CROSSING_THRESHOLD

    n_strands = max(abs(g) for g in braid) + 1

    # Get PD code and crossing count via snappy
    if HAS_SNAPPY:
        K = snappy.Link(braid_closure=braid)
        pd_code = K.PD_code()
        writhe_val = K.writhe()
        n_crossings = len(pd_code)
    else:
        # Without snappy, we can't get PD code, so always use tl-tensor
        n_crossings = float('inf')
        pd_code = None
        writhe_val = None

    # Decision: state sum for small diagrams, tl-tensor for large/wide
    if n_crossings <= crossing_threshold and pd_code is not None:
        if _HAS_NUMBA:
            jones = jones_from_pd_code_numba(pd_code, writhe_val)
            method = 'numba-statesum'
        else:
            jones = jones_from_pd_code(pd_code, writhe_val)
            method = 'statesum'
    else:
        # Use tl-tensor with minimal path optimization (max_repeats=1)
        # This avoids the ~70ms cotengra overhead while still being fast
        network = TLTensorNetwork.from_word(braid)
        opt = CotengraOptimizer(max_repeats=1, methods=['greedy'], progbar=False)
        info = network.contract_info(optimize=opt)
        contracted = network.contract(optimize=info.path)
        jones = contracted.tensors[0].terms[0][1]
        method = 'tl-tensor'
        n_crossings = len(braid)  # For braid, crossings = length

    return {
        'jones': jones,
        'method': method,
        'crossings': n_crossings,
        'strands': n_strands,
    }
