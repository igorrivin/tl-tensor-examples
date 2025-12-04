#!/usr/bin/env python3
"""
Compute Jones polynomials for T(p,q) torus knots using tl-tensor.

The T(p,q) torus knot has braid word (σ₁ σ₂ ... σ_{p-1})^q,
giving (p-1)*q crossings.
"""

from tl_tensor import TLTensorNetwork, CotengraOptimizer
import time
import warnings
import argparse

warnings.filterwarnings('ignore')


def torus_braid(p: int, q: int) -> list[int]:
    """Generate the braid word for the T(p,q) torus knot."""
    return list(range(1, p)) * q


def compute_jones(braid: list[int], max_repeats: int = 64) -> tuple:
    """
    Compute the Jones polynomial for a braid closure.

    Returns (jones_polynomial, info, opt_time, contract_time)
    """
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
    parser = argparse.ArgumentParser(description='Compute Jones polynomials for torus knots')
    parser.add_argument('-p', type=int, help='First parameter of T(p,q)')
    parser.add_argument('-q', type=int, help='Second parameter of T(p,q)')
    parser.add_argument('--benchmark', action='store_true', help='Run T(k,k) benchmark')
    parser.add_argument('--max-k', type=int, default=12, help='Maximum k for benchmark')
    parser.add_argument('--max-time', type=float, default=60.0, help='Max contraction time before stopping')
    parser.add_argument('--repeats', type=int, default=64, help='Optimization repeats')
    args = parser.parse_args()

    if args.benchmark:
        print(f"{'k':>3} | {'Crossings':>9} | {'Tensors':>7} | {'log10(cost)':>11} | {'Opt(s)':>7} | {'Contract(s)':>11} | {'Terms':>6}")
        print("-" * 75)

        for k in range(3, args.max_k + 1):
            braid = torus_braid(k, k)
            crossings = (k - 1) * k

            jones, info, opt_time, contract_time = compute_jones(braid, args.repeats)

            print(f"{k:>3} | {crossings:>9} | {len(braid)+1:>7} | {info.cost:>11.1f} | {opt_time:>7.2f} | {contract_time:>11.3f} | {len(jones):>6}")

            if contract_time > args.max_time:
                print(f"\nStopping - contraction time exceeded {args.max_time}s")
                break

    elif args.p and args.q:
        braid = torus_braid(args.p, args.q)
        crossings = (args.p - 1) * args.q

        print(f"T({args.p},{args.q}) torus knot")
        print(f"Crossings: {crossings}")
        print(f"Braid length: {len(braid)}")

        jones, info, opt_time, contract_time = compute_jones(braid, args.repeats)

        print(f"log10(cost): {info.cost:.1f}")
        print(f"Optimization time: {opt_time:.2f}s")
        print(f"Contraction time: {contract_time:.3f}s")
        print(f"\nJones polynomial (t = x^4):")
        print(jones)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
