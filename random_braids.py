#!/usr/bin/env python3
"""
Generate random braids and compute their Jones polynomials.

Includes utilities for:
- Generating random braid words (general or reduced)
- Computing the number of link components from a braid word
"""

from tl_tensor import TLTensorNetwork, CotengraOptimizer
import random
import time
import warnings
import argparse

warnings.filterwarnings('ignore')


def braid_permutation(word: list[int], n_strands: int) -> list[int]:
    """
    Compute the permutation induced by a braid word.

    Args:
        word: List of generators (positive or negative integers)
        n_strands: Number of strands in the braid

    Returns:
        Permutation as a list where perm[i] = j means strand i maps to position j
    """
    perm = list(range(n_strands))
    for g in word:
        i = abs(g) - 1  # Convert to 0-indexed
        # σ_i swaps strands at positions i and i+1
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    return perm


def count_cycles(perm: list[int]) -> int:
    """Count the number of cycles in a permutation."""
    n = len(perm)
    visited = [False] * n
    cycles = 0

    for start in range(n):
        if not visited[start]:
            cycles += 1
            i = start
            while not visited[i]:
                visited[i] = True
                i = perm[i]

    return cycles


def num_components(word: list[int], n_strands: int) -> int:
    """
    Compute the number of link components in the closure of a braid.

    When a braid is closed (top connected to bottom), each cycle
    in the induced permutation becomes a link component.
    """
    perm = braid_permutation(word, n_strands)
    return count_cycles(perm)


def random_braid(n_strands: int, length: int, reduced: bool = True, seed: int = None) -> list[int]:
    """
    Generate a random braid word.

    Args:
        n_strands: Number of strands (generators are ±1 to ±(n_strands-1))
        length: Desired length of the braid word
        reduced: If True, avoid consecutive inverse pairs (σ_i σ_i^{-1})
        seed: Random seed for reproducibility

    Returns:
        List of generators (positive for σ_i, negative for σ_i^{-1})
    """
    if seed is not None:
        random.seed(seed)

    if n_strands < 2:
        raise ValueError("Need at least 2 strands")

    generators = list(range(1, n_strands)) + list(range(-(n_strands - 1), 0))
    word = []

    for _ in range(length):
        if reduced and word:
            # Exclude the inverse of the last generator
            last = word[-1]
            valid = [g for g in generators if g != -last]
            word.append(random.choice(valid))
        else:
            word.append(random.choice(generators))

    return word


def random_knot_braid(n_strands: int, length: int, reduced: bool = True,
                      max_attempts: int = 1000, seed: int = None) -> list[int]:
    """
    Generate a random braid whose closure is a knot (single component).

    Args:
        n_strands: Number of strands
        length: Desired length of the braid word
        reduced: If True, generate reduced words
        max_attempts: Maximum attempts to find a knot
        seed: Random seed

    Returns:
        Braid word whose closure is a knot

    Raises:
        ValueError: If no knot found within max_attempts
    """
    if seed is not None:
        random.seed(seed)

    for attempt in range(max_attempts):
        word = random_braid(n_strands, length, reduced)
        if num_components(word, n_strands) == 1:
            return word

    raise ValueError(f"Could not find a knot in {max_attempts} attempts")


def compute_jones(braid: list[int], max_repeats: int = 64) -> tuple:
    """Compute the Jones polynomial for a braid closure."""
    network = TLTensorNetwork.from_word(braid)

    start = time.time()
    opt = CotengraOptimizer(max_repeats=max_repeats, methods=['greedy'])
    info = network.contract_info(optimize=opt)
    opt_time = time.time() - start

    start = time.time()
    contracted = network.contract(optimize=info.path)
    contract_time = time.time() - start

    jones = contracted.tensors[0].terms[0][1]
    return jones, info, opt_time, contract_time


def main():
    parser = argparse.ArgumentParser(description='Generate random braids and compute Jones polynomials')
    parser.add_argument('-n', '--strands', type=int, required=True, help='Number of strands')
    parser.add_argument('-l', '--length', type=int, required=True, help='Braid word length')
    parser.add_argument('--general', action='store_true', help='Generate general (non-reduced) words')
    parser.add_argument('--knot', action='store_true', help='Only generate knots (single component)')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--count', type=int, default=1, help='Number of braids to generate')
    parser.add_argument('--repeats', type=int, default=64, help='Optimization repeats')
    parser.add_argument('--stats', action='store_true', help='Show component statistics')
    args = parser.parse_args()

    reduced = not args.general

    if args.stats:
        # Generate many braids and show component distribution
        component_counts = {}
        for i in range(args.count):
            word = random_braid(args.strands, args.length, reduced,
                               seed=args.seed + i if args.seed else None)
            nc = num_components(word, args.strands)
            component_counts[nc] = component_counts.get(nc, 0) + 1

        print(f"Component distribution for {args.count} random {'reduced' if reduced else 'general'} braids")
        print(f"({args.strands} strands, length {args.length}):\n")
        for nc in sorted(component_counts.keys()):
            pct = 100 * component_counts[nc] / args.count
            print(f"  {nc} component(s): {component_counts[nc]:>5} ({pct:5.1f}%)")
        return

    for i in range(args.count):
        seed = args.seed + i if args.seed else None

        if args.knot:
            try:
                word = random_knot_braid(args.strands, args.length, reduced, seed=seed)
            except ValueError as e:
                print(f"#{i+1}: {e}")
                continue
        else:
            word = random_braid(args.strands, args.length, reduced, seed=seed)

        nc = num_components(word, args.strands)

        print(f"\n{'='*60}")
        print(f"Random braid #{i+1}")
        print(f"Strands: {args.strands}, Length: {args.length}, {'Reduced' if reduced else 'General'}")
        print(f"Components: {nc} ({'knot' if nc == 1 else 'link'})")
        print(f"Word: {word}")

        jones, info, opt_time, contract_time = compute_jones(word, args.repeats)

        print(f"\nlog10(cost): {info.cost:.1f}")
        print(f"Optimization: {opt_time:.2f}s, Contraction: {contract_time:.3f}s")
        print(f"Jones polynomial ({len(jones)} terms):")
        print(jones)


if __name__ == '__main__':
    main()
