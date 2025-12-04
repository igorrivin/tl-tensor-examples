#!/usr/bin/env python3
"""
Compute Jones polynomials for rational tangle closures using SnaPPy and tl-tensor.

A p/q rational tangle closure gives a 2-bridge knot/link.
The continued fraction expansion of p/q determines the crossing pattern.
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


def rational_tangle_braid(p: int, q: int) -> list[int]:
    """
    Get the braid word for the closure of a p/q rational tangle.

    Uses SnaPPy to construct the link and extract the braid representation.
    """
    K = snappy.RationalTangle(p, q).numerator_closure()
    return list(K.braid_word())


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
    parser = argparse.ArgumentParser(description='Compute Jones polynomials for rational tangles')
    parser.add_argument('p', type=int, nargs='?', help='Numerator of p/q')
    parser.add_argument('q', type=int, nargs='?', help='Denominator of p/q')
    parser.add_argument('--fibonacci', action='store_true', help='Run Fibonacci sequence benchmark')
    parser.add_argument('--max-n', type=int, default=20, help='Maximum Fibonacci index')
    parser.add_argument('--repeats', type=int, default=64, help='Optimization repeats')
    args = parser.parse_args()

    if args.fibonacci:
        # Fibonacci sequence: consecutive Fibonacci numbers are coprime
        fibs = [1, 1]
        while len(fibs) < args.max_n:
            fibs.append(fibs[-1] + fibs[-2])

        print(f"{'p':>6} | {'q':>6} | {'Crossings':>9} | {'Braid len':>9} | {'log10(cost)':>11} | {'Contract(s)':>11}")
        print("-" * 70)

        for i in range(2, len(fibs)):
            p, q = fibs[i], fibs[i + 1] if i + 1 < len(fibs) else fibs[i] + 1

            try:
                K = snappy.RationalTangle(p, q).numerator_closure()
                braid = list(K.braid_word())
                crossings = len(K.crossings)

                jones, info, opt_time, contract_time = compute_jones(braid, args.repeats)

                print(f"{p:>6} | {q:>6} | {crossings:>9} | {len(braid):>9} | {info.cost:>11.1f} | {contract_time:>11.3f}")
            except Exception as e:
                print(f"{p:>6} | {q:>6} | Error: {e}")

    elif args.p is not None and args.q is not None:
        K = snappy.RationalTangle(args.p, args.q).numerator_closure()
        braid = list(K.braid_word())

        print(f"{args.p}/{args.q} rational tangle closure")
        print(f"Link: {K}")
        print(f"Crossings: {len(K.crossings)}")
        print(f"Braid word: {braid}")

        jones, info, opt_time, contract_time = compute_jones(braid, args.repeats)

        print(f"\nlog10(cost): {info.cost:.1f}")
        print(f"Optimization time: {opt_time:.2f}s")
        print(f"Contraction time: {contract_time:.3f}s")
        print(f"\nJones polynomial (t = x^4):")
        print(jones)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
