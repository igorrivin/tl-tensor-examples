#!/usr/bin/env python3
"""
Benchmark tl-tensor vs SnaPPy for Jones polynomial computation.
"""

import time
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import snappy
    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False
    print("Warning: SnaPPy not available, will only benchmark tl-tensor")

from tl_tensor import TLTensorNetwork, CotengraOptimizer


def tl_jones(braid, max_repeats=32):
    """Compute Jones via tl-tensor (includes optimization time)"""
    network = TLTensorNetwork.from_word(braid)
    opt = CotengraOptimizer(max_repeats=max_repeats, methods=['greedy'])
    info = network.contract_info(optimize=opt)
    contracted = network.contract(optimize=info.path)
    return contracted.tensors[0].terms[0][1]


def tl_jones_split(braid, max_repeats=32):
    """Compute Jones via tl-tensor, returning (opt_time, contract_time, result)"""
    network = TLTensorNetwork.from_word(braid)
    opt = CotengraOptimizer(max_repeats=max_repeats, methods=['greedy'])

    start = time.perf_counter()
    info = network.contract_info(optimize=opt)
    opt_time = time.perf_counter() - start

    start = time.perf_counter()
    contracted = network.contract(optimize=info.path)
    contract_time = time.perf_counter() - start

    result = contracted.tensors[0].terms[0][1]
    return opt_time, contract_time, result


def snappy_jones(braid):
    """Compute Jones via SnaPPy"""
    K = snappy.Link(braid_closure=braid)
    return K.jones_polynomial()


def benchmark_knots():
    """Benchmark on knots from SnaPPy's database"""
    if not HAS_SNAPPY:
        print("SnaPPy required for this benchmark")
        return

    print("=" * 80)
    print("BENCHMARK: tl-tensor vs SnaPPy jones_polynomial (knots from database)")
    print("=" * 80)
    print()

    test_knots = ['K3a1', 'K4a1', 'K5a1', 'K6a1', 'K7a1', 'K8a1', 'K9a1', 'K10a1', 'K11a1', 'K12a1']

    print(f"{'Knot':<10} {'Len':<6} {'tl-opt':<10} {'tl-cont':<10} {'tl-total':<10} {'SnaPPy':<10} {'Speedup':<10}")
    print("-" * 80)

    for name in test_knots:
        try:
            K = snappy.Link(name)
            braid = list(K.braid_word())

            # tl-tensor with split timing
            opt_time, contract_time, tl_result = tl_jones_split(braid)
            tl_total = opt_time + contract_time

            # SnaPPy
            start = time.perf_counter()
            snappy_result = snappy_jones(braid)
            snappy_time = time.perf_counter() - start

            speedup = snappy_time / tl_total if tl_total > 0 else float('inf')
            winner = "tl" if speedup > 1 else "snappy"

            print(f"{name:<10} {len(braid):<6} {opt_time:<10.4f} {contract_time:<10.4f} {tl_total:<10.4f} {snappy_time:<10.4f} {speedup:>6.2f}x ({winner})")
        except Exception as e:
            print(f"{name:<10} ERROR: {e}")


def benchmark_torus():
    """Benchmark on T(k,k) torus knots"""
    print()
    print("=" * 80)
    print("BENCHMARK: T(k,k) torus knots (increasing complexity)")
    print("=" * 80)
    print()

    print(f"{'Torus':<10} {'Len':<6} {'tl-opt':<10} {'tl-cont':<10} {'tl-total':<10}", end="")
    if HAS_SNAPPY:
        print(f" {'SnaPPy':<10} {'Speedup':<10}")
    else:
        print()
    print("-" * 80)

    for k in range(3, 12):
        braid = list(range(1, k)) * k
        name = f"T({k},{k})"

        try:
            opt_time, contract_time, tl_result = tl_jones_split(braid)
            tl_total = opt_time + contract_time

            print(f"{name:<10} {len(braid):<6} {opt_time:<10.4f} {contract_time:<10.4f} {tl_total:<10.4f}", end="")

            if HAS_SNAPPY:
                start = time.perf_counter()
                snappy_result = snappy_jones(braid)
                snappy_time = time.perf_counter() - start

                speedup = snappy_time / tl_total if tl_total > 0 else float('inf')
                winner = "tl" if speedup > 1 else "snappy"
                print(f" {snappy_time:<10.4f} {speedup:>6.2f}x ({winner})")
            else:
                print()

        except Exception as e:
            print(f"{name:<10} ERROR: {e}")


def benchmark_random_braids():
    """Benchmark on random braids of increasing length"""
    import random
    random.seed(42)

    print()
    print("=" * 80)
    print("BENCHMARK: Random B_3 braids (increasing length)")
    print("=" * 80)
    print()

    print(f"{'Length':<10} {'tl-opt':<10} {'tl-cont':<10} {'tl-total':<10}", end="")
    if HAS_SNAPPY:
        print(f" {'SnaPPy':<10} {'Speedup':<10}")
    else:
        print()
    print("-" * 80)

    for length in [10, 20, 30, 40, 50, 60, 80, 100]:
        # Generate random reduced B_3 braid
        generators = [1, -1, 2, -2]
        braid = []
        for _ in range(length):
            while True:
                g = random.choice(generators)
                if not braid or g != -braid[-1]:
                    braid.append(g)
                    break

        try:
            opt_time, contract_time, tl_result = tl_jones_split(braid)
            tl_total = opt_time + contract_time

            print(f"{length:<10} {opt_time:<10.4f} {contract_time:<10.4f} {tl_total:<10.4f}", end="")

            if HAS_SNAPPY:
                start = time.perf_counter()
                snappy_result = snappy_jones(braid)
                snappy_time = time.perf_counter() - start

                speedup = snappy_time / tl_total if tl_total > 0 else float('inf')
                winner = "tl" if speedup > 1 else "snappy"
                print(f" {snappy_time:<10.4f} {speedup:>6.2f}x ({winner})")
            else:
                print()

        except Exception as e:
            print(f"{length:<10} ERROR: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark tl-tensor vs SnaPPy')
    parser.add_argument('--knots', action='store_true', help='Benchmark on database knots')
    parser.add_argument('--torus', action='store_true', help='Benchmark on torus knots')
    parser.add_argument('--random', action='store_true', help='Benchmark on random braids')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    args = parser.parse_args()

    if args.all or (not args.knots and not args.torus and not args.random):
        args.knots = args.torus = args.random = True

    if args.knots:
        benchmark_knots()
    if args.torus:
        benchmark_torus()
    if args.random:
        benchmark_random_braids()
