#!/usr/bin/env python3
"""
Experiments with random braids in B_3.

Investigates:
- How many distinct knot types appear in reduced words of length N?
- What is the distribution of known vs unknown knots?
- How does the distribution change with N?
"""

import random
from collections import defaultdict
import warnings
import argparse
import time

warnings.filterwarnings('ignore')

from tl_tensor import TLTensorNetwork, CotengraOptimizer
from random_braids import random_braid, num_components
from knot_identification import (
    get_database, compute_jones, jones_to_key, identify_knot
)
from latex_utils import jones_to_latex


def run_experiment(
    length: int,
    n_samples: int = 1000,
    database: dict = None,
    seed: int = None,
    verbose: bool = True
) -> dict:
    """
    Run experiment: generate random B_3 braids of given length and classify them.

    Returns dictionary with:
    - 'knot_counts': dict mapping knot name to count
    - 'jones_counts': dict mapping Jones polynomial to count
    - 'unknown_jones': list of Jones polynomials not in database
    - 'link_count': number of 3-component links (not knots)
    """
    if seed is not None:
        random.seed(seed)

    if database is None:
        database = get_database(verbose=False)

    knot_counts = defaultdict(int)
    jones_counts = defaultdict(int)
    unknown_jones = []
    link_count = 0

    start_time = time.time()

    for i in range(n_samples):
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i+1}/{n_samples} ({rate:.1f}/s)", end="\r")

        word = random_braid(3, length, reduced=True)

        # Check if it's a knot or 3-component link
        nc = num_components(word, 3)
        if nc == 3:
            link_count += 1
            knot_counts["3-component link"] += 1
            continue

        # Compute Jones polynomial
        jones = compute_jones(word)
        key = jones_to_key(jones)
        jones_counts[key] += 1

        # Identify the knot
        matches = identify_knot(jones, database)
        if matches == ["Unknown"]:
            unknown_jones.append(jones)
            knot_counts["Unknown"] += 1
        elif len(matches) == 1:
            knot_counts[matches[0]] += 1
        else:
            # Multiple knots share this Jones polynomial
            knot_counts[f"{matches[0]} (or {len(matches)-1} others)"] += 1

    if verbose:
        print()  # Clear the progress line

    return {
        'knot_counts': dict(knot_counts),
        'jones_counts': dict(jones_counts),
        'unknown_jones': unknown_jones,
        'link_count': link_count,
        'n_samples': n_samples,
        'length': length
    }


def print_results(results: dict, show_unknown: bool = False):
    """Print experiment results in a nice format."""
    length = results['length']
    n_samples = results['n_samples']
    knot_counts = results['knot_counts']
    jones_counts = results['jones_counts']
    unknown_jones = results['unknown_jones']

    print(f"\n{'='*60}")
    print(f"Results for B_3 reduced words of length {length}")
    print(f"Samples: {n_samples}")
    print(f"{'='*60}")

    # Summary statistics
    n_knots = n_samples - results['link_count']
    n_distinct_jones = len(jones_counts)
    n_unknown = knot_counts.get('Unknown', 0)

    print(f"\nSummary:")
    print(f"  Knots (1 component): {n_knots} ({100*n_knots/n_samples:.1f}%)")
    print(f"  Links (3 components): {results['link_count']} ({100*results['link_count']/n_samples:.1f}%)")
    print(f"  Distinct Jones polynomials: {n_distinct_jones}")
    print(f"  Unknown knots: {n_unknown} ({100*n_unknown/n_knots:.1f}% of knots)" if n_knots > 0 else "")

    # Top knot types
    print(f"\nKnot type distribution:")
    sorted_counts = sorted(knot_counts.items(), key=lambda x: -x[1])
    for name, count in sorted_counts[:15]:
        pct = 100 * count / n_samples
        bar = "█" * int(pct / 2)
        print(f"  {name:30s} {count:5d} ({pct:5.1f}%) {bar}")

    if len(sorted_counts) > 15:
        print(f"  ... and {len(sorted_counts) - 15} more types")

    # Show some unknown Jones polynomials
    if show_unknown and unknown_jones:
        print(f"\nSample unknown Jones polynomials:")
        seen = set()
        for jones in unknown_jones[:5]:
            key = str(jones)
            if key not in seen:
                seen.add(key)
                print(f"  {jones_to_latex(jones, 't')}")


def length_sweep(
    lengths: list[int],
    n_samples: int = 1000,
    database: dict = None,
    seed: int = None
):
    """Run experiments across multiple lengths and show how distribution changes."""
    if database is None:
        database = get_database(verbose=True)

    print(f"\n{'Length':>6} | {'Knots':>6} | {'Links':>6} | {'Distinct':>8} | {'Unknot':>8} | {'Trefoil':>8} | {'Fig-8':>8} | {'Unknown':>8}")
    print("-" * 85)

    for length in lengths:
        results = run_experiment(length, n_samples, database, seed, verbose=False)

        n_knots = n_samples - results['link_count']
        n_distinct = len(results['jones_counts'])
        unknot = results['knot_counts'].get('Unknot', 0)
        trefoil = results['knot_counts'].get('K3a1', 0)
        fig8 = results['knot_counts'].get('K4a1', 0)
        unknown = results['knot_counts'].get('Unknown', 0)

        print(f"{length:>6} | {n_knots:>6} | {results['link_count']:>6} | {n_distinct:>8} | "
              f"{unknot:>8} | {trefoil:>8} | {fig8:>8} | {unknown:>8}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiments with random B_3 braids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python b3_experiments.py -l 20 -n 1000      # Single length experiment
  python b3_experiments.py --sweep 10,20,30   # Multiple lengths
  python b3_experiments.py --sweep-range 5 50 5  # Range from 5 to 50, step 5
        """
    )
    parser.add_argument('-l', '--length', type=int, help='Braid word length')
    parser.add_argument('-n', '--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sweep', type=str, help='Comma-separated lengths for sweep')
    parser.add_argument('--sweep-range', type=int, nargs=3, metavar=('START', 'END', 'STEP'),
                        help='Length range for sweep')
    parser.add_argument('--show-unknown', action='store_true', help='Show unknown Jones polynomials')
    parser.add_argument('--rebuild-db', action='store_true', help='Rebuild the knot database')
    parser.add_argument('--max-crossings', type=int, default=12, help='Max crossings for database')
    args = parser.parse_args()

    # Get or build database
    database = get_database(args.max_crossings, rebuild=args.rebuild_db)

    if args.sweep:
        lengths = [int(x.strip()) for x in args.sweep.split(',')]
        length_sweep(lengths, args.samples, database, args.seed)

    elif args.sweep_range:
        start, end, step = args.sweep_range
        lengths = list(range(start, end + 1, step))
        length_sweep(lengths, args.samples, database, args.seed)

    elif args.length:
        results = run_experiment(args.length, args.samples, database, args.seed)
        print_results(results, args.show_unknown)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
