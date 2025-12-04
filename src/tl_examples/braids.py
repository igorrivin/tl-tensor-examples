"""
Braid utilities: generation, properties, and component counting.
"""

import random
from typing import Optional


def writhe(braid: list[int]) -> int:
    """
    Compute the writhe (sum of crossing signs) of a braid.

    Args:
        braid: Braid word as list of generators

    Returns:
        Writhe (positive crossings minus negative crossings)

    Example:
        >>> writhe([1, 1, 1])  # Trefoil
        3
        >>> writhe([1, -1, 2, -2])
        0
    """
    return sum(1 if g > 0 else -1 for g in braid)


def n_strands(braid: list[int]) -> int:
    """
    Get the number of strands for a braid word.

    Args:
        braid: Braid word as list of generators

    Returns:
        Number of strands (max generator index + 1)

    Example:
        >>> n_strands([1, 1, 1])  # B_2 braid
        2
        >>> n_strands([1, 2, 3])  # B_4 braid
        4
    """
    if not braid:
        return 1
    return max(abs(g) for g in braid) + 1


def braid_permutation(braid: list[int], num_strands: int) -> list[int]:
    """
    Compute the permutation induced by a braid.

    Args:
        braid: Braid word as list of generators
        num_strands: Number of strands

    Returns:
        Permutation as list where perm[i] = j means strand i goes to position j
    """
    perm = list(range(num_strands))
    for g in braid:
        i = abs(g) - 1
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    return perm


def count_cycles(perm: list[int]) -> int:
    """Count the number of cycles in a permutation."""
    n = len(perm)
    visited = [False] * n
    cycles = 0

    for start in range(n):
        if visited[start]:
            continue
        cycles += 1
        i = start
        while not visited[i]:
            visited[i] = True
            i = perm[i]

    return cycles


def num_components(braid: list[int], num_strands: Optional[int] = None) -> int:
    """
    Compute the number of link components in the closure of a braid.

    Args:
        braid: Braid word as list of generators
        num_strands: Number of strands (auto-detected if None)

    Returns:
        Number of components in the braid closure

    Example:
        >>> num_components([1, 1, 1])  # Trefoil (knot)
        1
        >>> num_components([1, -1])  # 2-component unlink
        2
        >>> num_components([1, 2, -1, -2])  # Unknot
        1

    Note:
        The number of components equals the number of cycles in the
        permutation induced by the braid. For B_n braids, components
        have the same parity as n.
    """
    if num_strands is None:
        num_strands = n_strands(braid)
    perm = braid_permutation(braid, num_strands)
    return count_cycles(perm)


def random_braid(
    num_strands: int,
    length: int,
    reduced: bool = True,
    seed: Optional[int] = None,
) -> list[int]:
    """
    Generate a random braid word.

    Args:
        num_strands: Number of strands (generators are 1 to num_strands-1)
        length: Length of the braid word
        reduced: If True, avoid consecutive inverses (σᵢσᵢ⁻¹)
        seed: Random seed for reproducibility

    Returns:
        Random braid word as list of generators

    Example:
        >>> random_braid(3, 10, seed=42)
        [1, -2, 1, 2, -1, -2, 2, 1, -1, 2]

    Note:
        For reduced words, consecutive inverses are forbidden but non-adjacent
        inverses are allowed. This doesn't produce a geodesic in general.
    """
    if seed is not None:
        random.seed(seed)

    if num_strands < 2:
        raise ValueError("Need at least 2 strands")

    generators = []
    for i in range(1, num_strands):
        generators.extend([i, -i])

    word = []
    for _ in range(length):
        if reduced and word:
            # Avoid consecutive inverses
            while True:
                g = random.choice(generators)
                if g != -word[-1]:
                    word.append(g)
                    break
        else:
            word.append(random.choice(generators))

    return word


def random_knot_braid(
    num_strands: int,
    length: int,
    reduced: bool = True,
    seed: Optional[int] = None,
    max_attempts: int = 1000,
) -> Optional[list[int]]:
    """
    Generate a random braid whose closure is a knot (single component).

    Args:
        num_strands: Number of strands
        length: Length of the braid word
        reduced: If True, avoid consecutive inverses
        seed: Random seed
        max_attempts: Maximum attempts before giving up

    Returns:
        Braid word, or None if no knot found within max_attempts

    Note:
        For odd-strand braids with odd length, the closure is always a knot.
        For even strands or even length, some closures are multi-component links.
    """
    if seed is not None:
        random.seed(seed)

    for _ in range(max_attempts):
        braid = random_braid(num_strands, length, reduced=reduced)
        if num_components(braid, num_strands) == 1:
            return braid

    return None


def torus_braid(p: int, q: int) -> list[int]:
    """
    Generate the braid word for the T(p,q) torus knot/link.

    The T(p,q) torus knot is the closure of (σ₁σ₂...σₚ₋₁)^q on p strands.

    Args:
        p: Number of strands
        q: Number of full twists

    Returns:
        Braid word for T(p,q)

    Example:
        >>> torus_braid(2, 3)  # Trefoil T(2,3)
        [1, 1, 1]
        >>> torus_braid(3, 3)  # T(3,3)
        [1, 2, 1, 2, 1, 2]
    """
    if p < 2:
        raise ValueError("Need p >= 2")
    if q == 0:
        return []

    # One full twist is σ₁σ₂...σₚ₋₁
    full_twist = list(range(1, p))

    if q > 0:
        return full_twist * q
    else:
        # Negative twists use inverse generators
        return [-g for g in full_twist] * (-q)
