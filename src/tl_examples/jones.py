"""
Jones polynomial computation via tl-tensor and hybrid methods.
"""

import json
import os
import subprocess
from typing import Optional
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


# Crossover point: use SnaPPy for braids with <= this many strands
DEFAULT_STRAND_THRESHOLD = 5

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
) -> list[tuple[int, int]]:
    """
    Compute Jones polynomial using tl-tensor.

    Args:
        braid: Braid word as list of generators (e.g., [1, 1, 1] for trefoil)
        max_repeats: Number of optimization repeats for cotengra
        methods: Optimization methods (default: ['greedy'])

    Returns:
        Jones polynomial as list of (coefficient, exponent) tuples in x-variable
        where t = x^4 (standard Jones variable t satisfies t = x^4)

    Example:
        >>> compute_jones([1, 1, 1])  # Trefoil
        [(-1, -16), (1, -12), (1, -4)]
        # This is -x^{-16} + x^{-12} + x^{-4} = -t^{-4} + t^{-3} + t^{-1}
    """
    if methods is None:
        methods = ["greedy"]

    network = TLTensorNetwork.from_word(braid)
    opt = CotengraOptimizer(max_repeats=max_repeats, methods=methods)
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

    Returns:
        dict with:
        - 'result': Jones polynomial in x-variable [(coeff, exp), ...]
        - 'method': 'tl-tensor', 'snappy', or 'sage-env'
        - 'n_strands': Number of strands in braid

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
    result = compute_jones(braid, max_repeats=max_repeats)
    return {"result": result, "method": "tl-tensor", "n_strands": n_strands}


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
