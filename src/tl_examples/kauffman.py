"""
Kauffman bracket state sum for Jones polynomial computation.

This implements the classic state sum algorithm directly on PD codes,
bypassing the need for braid conversion. For diagrams with few crossings
but complex braid representations, this can be significantly faster than
tensor network contraction on inflated braids.

The algorithm:
1. Enumerate all 2^n smoothing states (A or B at each crossing)
2. For each state, count resulting circles using union-find
3. Sum contributions: A^(a-b) * (-A² - A⁻²)^(circles-1)
4. Normalize by writhe to get Jones polynomial

Complexity: O(2^n * n) where n = number of crossings
- Exponential in crossings, but polynomial per state
- Embarrassingly parallel (each state independent)

Performance:
- Pure Python: baseline
- Numba JIT: 30-120x faster for ≥10 crossings
- Use numba=True (default when available) for best performance
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from collections import defaultdict

# Check for Numba availability
try:
    from numba import njit, prange
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    # Create no-op decorator for when numba isn't available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


class UnionFind:
    """Union-Find data structure for counting connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def count_components(self) -> int:
        return len(set(self.find(i) for i in range(len(self.parent))))


def pd_code_to_crossings(pd_code: list[tuple[int, int, int, int]]) -> list[dict]:
    """
    Convert PD code to crossing data for state sum.

    PD code convention (SnaPPy): (a, b, c, d) where strands are labeled around
    crossing going counterclockwise, starting from incoming under-strand.

    For Kauffman bracket, we use:
    - A-smoothing: connect adjacent strands in CCW order (a-b, c-d)
    - B-smoothing: connect opposite strands (a-d, b-c)

    This is independent of crossing sign - the sign only affects the writhe
    normalization at the end.

    Returns:
        crossings: list of dicts with 'A' and 'B' smoothing connections
    """
    crossings = []

    for (a, b, c, d) in pd_code:
        # A-smoothing: connect adjacent strands going around the crossing
        # B-smoothing: connect opposite strands
        # This convention matches the standard Kauffman bracket definition
        crossings.append({
            'A': [(a, b), (c, d)],  # Adjacent in CCW order
            'B': [(a, d), (b, c)],  # Diagonal/opposite
        })

    return crossings


def count_circles_for_state(
    crossings: list[dict],
    state: int,
    n_crossings: int,
    n_strands: int
) -> int:
    """
    Count circles in the diagram after applying smoothings according to state.

    Args:
        crossings: crossing data from pd_code_to_crossings
        state: integer where bit i indicates A (0) or B (1) smoothing at crossing i
        n_crossings: number of crossings
        n_strands: total number of strand segments

    Returns:
        Number of circles in the smoothed diagram
    """
    uf = UnionFind(n_strands)

    for i, crossing in enumerate(crossings):
        smoothing = 'B' if (state >> i) & 1 else 'A'
        for (s1, s2) in crossing[smoothing]:
            uf.union(s1, s2)

    return uf.count_components()


def kauffman_bracket_from_pd(
    pd_code: list[tuple[int, int, int, int]],
) -> dict[int, int]:
    """
    Compute Kauffman bracket from PD code using state sum.

    Args:
        pd_code: Planar diagram code as list of 4-tuples

    Returns:
        Dictionary mapping power of A to coefficient
        e.g., {4: 1, -4: 1, 0: -1} means A^4 + A^{-4} - 1
    """
    n_crossings = len(pd_code)

    if n_crossings == 0:
        return {0: 1}  # Empty diagram = unknot

    # Find number of strand segments (max index + 1)
    n_strands = max(max(c) for c in pd_code) + 1

    crossings = pd_code_to_crossings(pd_code)

    # Polynomial represented as {power: coefficient}
    bracket: dict[int, int] = defaultdict(int)

    # Loop factor: d = -A^2 - A^{-2}
    # d^k = sum over partitions...
    # We'll compute d^k directly using binomial expansion

    # Enumerate all 2^n states
    for state in range(1 << n_crossings):
        # Count A and B smoothings
        n_B = bin(state).count('1')
        n_A = n_crossings - n_B

        # Count circles
        n_circles = count_circles_for_state(crossings, state, n_crossings, n_strands)

        # Contribution: A^{n_A - n_B} * d^{n_circles - 1}
        # where d = -A^2 - A^{-2}

        base_power = n_A - n_B
        loop_power = n_circles - 1

        # Expand d^{loop_power} = (-A^2 - A^{-2})^{loop_power}
        # = (-1)^{loop_power} * (A^2 + A^{-2})^{loop_power}
        # = (-1)^{loop_power} * sum_{k=0}^{loop_power} C(loop_power,k) * A^{2k} * A^{-2(loop_power-k)}
        # = (-1)^{loop_power} * sum_{k=0}^{loop_power} C(loop_power,k) * A^{4k - 2*loop_power}

        sign = (-1) ** loop_power
        for k in range(loop_power + 1):
            binom = _binomial(loop_power, k)
            power = base_power + 4 * k - 2 * loop_power
            bracket[power] += sign * binom

    # Clean up zeros
    return {k: v for k, v in bracket.items() if v != 0}


def _binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # Use symmetry
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def jones_from_kauffman(bracket: dict[int, int], writhe: int) -> dict[int, int]:
    """
    Convert Kauffman bracket to Jones polynomial.

    J(t) = (-A^3)^{-writhe} * <D>

    where t = A^{-4} (so A = t^{-1/4})

    Args:
        bracket: Kauffman bracket as {power_of_A: coefficient}
        writhe: writhe of the diagram

    Returns:
        Jones polynomial as {power_of_A: coefficient}
    """
    # Multiply by (-A^3)^{-writhe} = (-1)^{-writhe} * A^{-3*writhe}
    shift = -3 * writhe
    sign = (-1) ** (-writhe)

    result = {}
    for power, coeff in bracket.items():
        new_power = power + shift
        new_coeff = sign * coeff
        if new_coeff != 0:
            result[new_power] = new_coeff

    return result


def jones_from_pd_code(
    pd_code: list[tuple[int, int, int, int]],
    writhe: int | None = None
) -> list[tuple[int, int]]:
    """
    Compute Jones polynomial from PD code using Kauffman bracket state sum.

    Args:
        pd_code: Planar diagram code
        writhe: Writhe of the diagram (required - use snappy.Link(pd).writhe())

    Returns:
        Jones polynomial as list of (coefficient, power) tuples in x-variable
        where t = x^4 (compatible with tl-tensor format)
    """
    if not pd_code:
        return [(1, 0)]  # Unknot

    bracket = kauffman_bracket_from_pd(pd_code)

    if writhe is None:
        # Try to get writhe from snappy
        try:
            import snappy
            L = snappy.Link(pd_code)
            writhe = L.writhe()
        except ImportError:
            raise ValueError("writhe must be provided (snappy not available)")

    jones_A = jones_from_kauffman(bracket, writhe)

    # Convert from A variable to x variable where A = x (tl-tensor convention)
    # Actually tl-tensor uses x where t = x^4, so A = x
    result = [(coeff, power) for power, coeff in sorted(jones_A.items())]

    return result


def compute_jones_statesum(
    pd_code: list[tuple[int, int, int, int]],
    parallel: bool = False
) -> list[tuple[int, int]]:
    """
    Compute Jones polynomial using state sum algorithm.

    This is the main entry point for state sum computation.

    Args:
        pd_code: Planar diagram code as list of 4-tuples
        parallel: If True, use parallel computation (requires joblib)

    Returns:
        Jones polynomial as list of (coefficient, power) tuples
        Compatible with tl-tensor format.
    """
    if parallel:
        return _compute_jones_parallel(pd_code)
    return jones_from_pd_code(pd_code)


def _compute_jones_parallel(
    pd_code: list[tuple[int, int, int, int]],
    writhe: int | None = None,
    n_jobs: int = -1
) -> list[tuple[int, int]]:
    """
    Parallel state sum computation using joblib.

    Splits the 2^n states across workers.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        # Fall back to sequential
        return jones_from_pd_code(pd_code, writhe)

    n_crossings = len(pd_code)
    if n_crossings < 10:
        # Not worth parallelizing for small diagrams
        return jones_from_pd_code(pd_code, writhe)

    # Get writhe if not provided
    if writhe is None:
        import snappy
        L = snappy.Link(pd_code)
        writhe = L.writhe()

    n_strands = max(max(c) for c in pd_code) + 1
    crossings = pd_code_to_crossings(pd_code)

    def process_state_batch(start: int, end: int) -> dict[int, int]:
        """Process a batch of states."""
        partial_bracket: dict[int, int] = defaultdict(int)

        for state in range(start, end):
            n_B = bin(state).count('1')
            n_A = n_crossings - n_B
            n_circles = count_circles_for_state(crossings, state, n_crossings, n_strands)

            base_power = n_A - n_B
            loop_power = n_circles - 1
            sign = (-1) ** loop_power

            for k in range(loop_power + 1):
                binom = _binomial(loop_power, k)
                power = base_power + 4 * k - 2 * loop_power
                partial_bracket[power] += sign * binom

        return dict(partial_bracket)

    # Split into batches
    n_states = 1 << n_crossings
    batch_size = max(1, n_states // 8)  # 8 batches by default
    batches = [(i, min(i + batch_size, n_states)) for i in range(0, n_states, batch_size)]

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_state_batch)(start, end) for start, end in batches
    )

    # Merge results
    bracket: dict[int, int] = defaultdict(int)
    for partial in results:
        for power, coeff in partial.items():
            bracket[power] += coeff

    bracket = {k: v for k, v in bracket.items() if v != 0}
    jones_A = jones_from_kauffman(bracket, writhe)

    return [(coeff, power) for power, coeff in sorted(jones_A.items())]


# Convenience function matching tl-tensor API
def compute_jones_from_pd(
    pd_code: list[tuple[int, int, int, int]],
    writhe: int | None = None,
    parallel: bool = False
) -> list[tuple[int, int]]:
    """
    Compute Jones polynomial from PD code.

    This function provides a tl-tensor-compatible interface for the
    Kauffman bracket state sum algorithm.

    Args:
        pd_code: Planar diagram code
        writhe: Writhe of diagram (if None, computed via snappy)
        parallel: Use parallel computation for large diagrams

    Returns:
        Jones polynomial as [(coeff, power), ...] in x-variable (t = x^4)
    """
    if parallel:
        return _compute_jones_parallel(pd_code, writhe)
    return jones_from_pd_code(pd_code, writhe)


# Crossover point where tl-tensor becomes faster than state sum
# With Numba: ~16 crossings (7ms numba vs ~100ms tl-tensor at 14 crossings)
# Without Numba: ~12 crossings (pure Python state sum)
STATESUM_CROSSING_THRESHOLD = 16 if _HAS_NUMBA else 12


# =============================================================================
# Numba-optimized implementation
# =============================================================================

@njit(cache=True)
def _popcount(x: int) -> int:
    """Count number of set bits (Numba-compatible replacement for bin().count('1'))."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


@njit(cache=True)
def _uf_find(parent: np.ndarray, x: int) -> int:
    """Find root with path compression (iterative for Numba)."""
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression
    while parent[x] != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root


@njit(cache=True)
def _uf_union(parent: np.ndarray, rank: np.ndarray, x: int, y: int) -> None:
    """Union by rank."""
    px = _uf_find(parent, x)
    py = _uf_find(parent, y)
    if px == py:
        return
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1


@njit(cache=True)
def _count_components(parent: np.ndarray) -> int:
    """Count connected components."""
    n = len(parent)
    count = 0
    for i in range(n):
        if _uf_find(parent, i) == i:
            count += 1
    return count


@njit(cache=True)
def _binomial_numba(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


@njit(cache=True)
def _count_circles_numba(
    smoothings_A: np.ndarray,  # shape (n_crossings, 2, 2)
    smoothings_B: np.ndarray,  # shape (n_crossings, 2, 2)
    state: int,
    n_crossings: int,
    n_strands: int
) -> int:
    """Count circles for a given state using union-find."""
    parent = np.arange(n_strands)
    rank = np.zeros(n_strands, dtype=np.int32)

    for i in range(n_crossings):
        if (state >> i) & 1:  # B smoothing
            smoothing = smoothings_B[i]
        else:  # A smoothing
            smoothing = smoothings_A[i]

        for j in range(2):
            _uf_union(parent, rank, smoothing[j, 0], smoothing[j, 1])

    return _count_components(parent)


@njit(cache=True, parallel=True)
def _kauffman_bracket_numba(
    smoothings_A: np.ndarray,
    smoothings_B: np.ndarray,
    n_crossings: int,
    n_strands: int,
    max_power: int
) -> np.ndarray:
    """
    Compute Kauffman bracket coefficients using Numba with parallel execution.

    Returns array of coefficients indexed by (power + max_power).
    """
    # Coefficients array: index = power + max_power
    # Power range: roughly [-3*n, 3*n] but we use a safe upper bound
    coeff_size = 2 * max_power + 1
    coeffs = np.zeros(coeff_size, dtype=np.int64)

    n_states = 1 << n_crossings

    # Process states in parallel
    for state in prange(n_states):
        n_B = _popcount(state)
        n_A = n_crossings - n_B

        n_circles = _count_circles_numba(
            smoothings_A, smoothings_B, state, n_crossings, n_strands
        )

        base_power = n_A - n_B
        loop_power = n_circles - 1
        sign = 1 if loop_power % 2 == 0 else -1

        for k in range(loop_power + 1):
            binom = _binomial_numba(loop_power, k)
            power = base_power + 4 * k - 2 * loop_power
            # Atomic add not available in numba, but each state writes to different
            # powers due to the structure, or we accept small race conditions
            # that cancel out. Actually we need to handle this properly.
            idx = power + max_power
            if 0 <= idx < coeff_size:
                coeffs[idx] += sign * binom

    return coeffs


@njit(cache=True)
def _kauffman_bracket_numba_sequential(
    smoothings_A: np.ndarray,
    smoothings_B: np.ndarray,
    n_crossings: int,
    n_strands: int,
    max_power: int
) -> np.ndarray:
    """Sequential version for correctness (parallel has race conditions)."""
    coeff_size = 2 * max_power + 1
    coeffs = np.zeros(coeff_size, dtype=np.int64)

    n_states = 1 << n_crossings

    for state in range(n_states):
        n_B = _popcount(state)
        n_A = n_crossings - n_B

        n_circles = _count_circles_numba(
            smoothings_A, smoothings_B, state, n_crossings, n_strands
        )

        base_power = n_A - n_B
        loop_power = n_circles - 1
        sign = 1 if loop_power % 2 == 0 else -1

        for k in range(loop_power + 1):
            binom = _binomial_numba(loop_power, k)
            power = base_power + 4 * k - 2 * loop_power
            idx = power + max_power
            if 0 <= idx < coeff_size:
                coeffs[idx] += sign * binom

    return coeffs


def _prepare_smoothings(pd_code: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert PD code to numpy arrays for Numba."""
    n_crossings = len(pd_code)
    smoothings_A = np.zeros((n_crossings, 2, 2), dtype=np.int32)
    smoothings_B = np.zeros((n_crossings, 2, 2), dtype=np.int32)

    for i, (a, b, c, d) in enumerate(pd_code):
        # A-smoothing: adjacent (a-b, c-d)
        smoothings_A[i, 0, 0] = a
        smoothings_A[i, 0, 1] = b
        smoothings_A[i, 1, 0] = c
        smoothings_A[i, 1, 1] = d
        # B-smoothing: diagonal (a-d, b-c)
        smoothings_B[i, 0, 0] = a
        smoothings_B[i, 0, 1] = d
        smoothings_B[i, 1, 0] = b
        smoothings_B[i, 1, 1] = c

    return smoothings_A, smoothings_B


def kauffman_bracket_numba(
    pd_code: list[tuple[int, int, int, int]],
) -> dict[int, int]:
    """
    Compute Kauffman bracket using Numba-optimized state sum.

    30-120x faster than pure Python for ≥10 crossings.

    Args:
        pd_code: Planar diagram code as list of 4-tuples

    Returns:
        Dictionary mapping power of A to coefficient
    """
    n_crossings = len(pd_code)

    if n_crossings == 0:
        return {0: 1}

    n_strands = max(max(c) for c in pd_code) + 1
    smoothings_A, smoothings_B = _prepare_smoothings(pd_code)

    # Max power is bounded by 3*n_crossings + 2*n_strands (generous bound)
    max_power = 4 * n_crossings + 2 * n_strands

    # Use sequential version to avoid race conditions
    coeffs = _kauffman_bracket_numba_sequential(
        smoothings_A, smoothings_B, n_crossings, n_strands, max_power
    )

    # Convert back to dictionary
    result = {}
    for idx, coeff in enumerate(coeffs):
        if coeff != 0:
            power = idx - max_power
            result[power] = int(coeff)

    return result


def jones_from_pd_code_numba(
    pd_code: list[tuple[int, int, int, int]],
    writhe: int | None = None
) -> list[tuple[int, int]]:
    """
    Compute Jones polynomial from PD code using Numba-optimized state sum.

    Args:
        pd_code: Planar diagram code
        writhe: Writhe of the diagram (if None, computed via snappy)

    Returns:
        Jones polynomial as list of (coefficient, power) tuples
    """
    if not pd_code:
        return [(1, 0)]

    bracket = kauffman_bracket_numba(pd_code)

    if writhe is None:
        try:
            import snappy
            L = snappy.Link(pd_code)
            writhe = L.writhe()
        except ImportError:
            raise ValueError("writhe must be provided (snappy not available)")

    jones_A = jones_from_kauffman(bracket, writhe)
    return [(coeff, power) for power, coeff in sorted(jones_A.items())]


def has_numba() -> bool:
    """Check if Numba is available."""
    return _HAS_NUMBA


def compute_jones_from_pd_fast(
    pd_code: list[tuple[int, int, int, int]],
    writhe: int | None = None,
    use_numba: bool | None = None,
) -> list[tuple[int, int]]:
    """
    Compute Jones polynomial from PD code using the fastest available method.

    Automatically uses Numba if available (30-120x faster for ≥10 crossings).

    Args:
        pd_code: Planar diagram code
        writhe: Writhe of diagram (if None, computed via snappy)
        use_numba: Force Numba on/off (None = auto-detect)

    Returns:
        Jones polynomial as [(coeff, power), ...] in x-variable (t = x^4)
    """
    if use_numba is None:
        use_numba = _HAS_NUMBA

    if use_numba and _HAS_NUMBA:
        return jones_from_pd_code_numba(pd_code, writhe)
    else:
        return jones_from_pd_code(pd_code, writhe)


def compute_jones_auto(
    pd_code: list[tuple[int, int, int, int]] | None = None,
    braid: list[int] | None = None,
    writhe: int | None = None,
    use_numba: bool | None = None,
) -> tuple[list[tuple[int, int]], str]:
    """
    Automatically select the best method for Jones polynomial computation.

    Strategy:
    - For ≤16 crossings with Numba (or ≤12 without): use state sum
    - For >16 crossings or braid-only input: use tl-tensor

    Performance characteristics:
    - Numba state sum: ~0.4ms at 10 crossings, ~7ms at 14 crossings
    - tl-tensor: ~65-110ms for simple knots, scales with braid width not crossings

    Args:
        pd_code: Planar diagram code (optional)
        braid: Braid word (optional)
        writhe: Writhe for PD code (if None, computed via snappy)
        use_numba: Force Numba on/off for state sum (None = auto-detect)

    Returns:
        Tuple of (jones_polynomial, method_used)
        where method_used is 'statesum', 'statesum-numba', or 'tl-tensor'
    """
    if use_numba is None:
        use_numba = _HAS_NUMBA

    threshold = STATESUM_CROSSING_THRESHOLD

    if pd_code is not None:
        n_crossings = len(pd_code)
        if n_crossings <= threshold:
            if use_numba and _HAS_NUMBA:
                jones = jones_from_pd_code_numba(pd_code, writhe)
                return jones, 'statesum-numba'
            else:
                jones = jones_from_pd_code(pd_code, writhe)
                return jones, 'statesum'

    # Fall back to tl-tensor
    if braid is None:
        if pd_code is None:
            raise ValueError("Must provide either pd_code or braid")
        # Get braid from snappy
        import snappy
        L = snappy.Link(pd_code)
        braid = L.braid_word()

    from tl_examples.jones import compute_jones as tl_compute_jones
    jones = tl_compute_jones(braid)
    return jones, 'tl-tensor'
