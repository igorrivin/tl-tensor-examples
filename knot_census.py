#!/usr/bin/env python3
"""
Compute Jones polynomials for knots from SnaPPy's knot census.

SnaPPy includes tables of knots up to a certain crossing number.
Knots are named like 'K3a1' (trefoil), 'K4a1' (figure-8), etc.
"""

from tl_tensor import TLTensorNetwork, CotengraOptimizer
import time
import warnings
import argparse

warnings.filterwarnings('ignore')

try:
    import snappy
except ImportError:
    print("Error: SnaPPy is required for this example.")
    print("Install with: conda install -c conda-forge snappy")
    exit(1)


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
    parser = argparse.ArgumentParser(description='Compute Jones polynomials from knot census')
    parser.add_argument('name', nargs='?', help='Knot name (e.g., K3a1, K10a1)')
    parser.add_argument('--crossings', type=int, help='Compute for all knots with this crossing number')
    parser.add_argument('--range', type=str, help='Range of crossing numbers (e.g., "3-10")')
    parser.add_argument('--max-knots', type=int, default=10, help='Max knots per crossing number')
    parser.add_argument('--repeats', type=int, default=64, help='Optimization repeats')
    args = parser.parse_args()

    if args.name:
        # Single knot
        try:
            K = snappy.Link(args.name)
            braid = list(K.braid_word())

            print(f"Knot: {args.name}")
            print(f"Crossings: {len(K.crossings)}")
            print(f"Braid word: {braid}")

            jones, info, opt_time, contract_time = compute_jones(braid, args.repeats)

            print(f"\nlog10(cost): {info.cost:.1f}")
            print(f"Optimization time: {opt_time:.2f}s")
            print(f"Contraction time: {contract_time:.3f}s")
            print(f"\nJones polynomial (t = x^4):")
            print(jones)

        except Exception as e:
            print(f"Error: {e}")

    elif args.crossings or args.range:
        if args.range:
            start, end = map(int, args.range.split('-'))
            crossing_range = range(start, end + 1)
        else:
            crossing_range = [args.crossings]

        print(f"{'Name':>8} | {'Crossings':>9} | {'Braid len':>9} | {'log10(cost)':>11} | {'Contract(s)':>11} | {'Terms':>6}")
        print("-" * 75)

        for n in crossing_range:
            count = 0
            for suffix in ['a', 'n']:  # alternating and non-alternating
                i = 1
                while count < args.max_knots:
                    name = f"K{n}{suffix}{i}"
                    try:
                        K = snappy.Link(name)
                        braid = list(K.braid_word())
                        crossings = len(K.crossings)

                        jones, info, opt_time, contract_time = compute_jones(braid, args.repeats)

                        print(f"{name:>8} | {crossings:>9} | {len(braid):>9} | {info.cost:>11.1f} | {contract_time:>11.3f} | {len(jones):>6}")
                        count += 1
                        i += 1
                    except:
                        break

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python knot_census.py K3a1           # Trefoil knot")
        print("  python knot_census.py K4a1           # Figure-8 knot")
        print("  python knot_census.py --crossings 10 # All 10-crossing knots")
        print("  python knot_census.py --range 3-8    # Knots from 3 to 8 crossings")


if __name__ == '__main__':
    main()
