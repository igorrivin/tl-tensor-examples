"""
Command-line interface for tl-tensor-examples.
"""

import argparse
import sys


def identify_main():
    """Entry point for knot-identify command."""
    parser = argparse.ArgumentParser(
        description="Identify knots from braid words using Jones polynomial"
    )
    parser.add_argument(
        "braid",
        nargs="+",
        type=int,
        help="Braid word as space-separated integers (e.g., 1 1 1 for trefoil)",
    )
    parser.add_argument(
        "--invariants",
        "-i",
        action="store_true",
        help="Use multiple invariants for disambiguation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    from .identification import identify_braid, identify_with_invariants
    from .jones import compute_jones

    braid = args.braid

    if args.invariants:
        result = identify_with_invariants(braid)
        print(f"Braid: {braid}")
        print(f"Jones polynomial: {result['jones']}")
        print(f"Jones matches: {result['jones_matches']}")
        if result["alexander"]:
            print(f"Alexander polynomial: {result['alexander']}")
        if result["signature"] is not None:
            print(f"Signature: {result['signature']}")
        if result["volume"] is not None:
            print(f"Hyperbolic volume: {result['volume']:.6f}")
        print(f"Best matches: {result['best_matches']}")
    else:
        matches = identify_braid(braid)
        if args.verbose:
            jones = compute_jones(braid)
            print(f"Braid: {braid}")
            print(f"Jones polynomial: {jones}")
            print(f"Identified as: {matches}")
        else:
            print(", ".join(matches))


def jones_main():
    """Entry point for jones-compute command."""
    parser = argparse.ArgumentParser(
        description="Compute Jones polynomial from braid word"
    )
    parser.add_argument(
        "braid",
        nargs="+",
        type=int,
        help="Braid word as space-separated integers",
    )
    parser.add_argument(
        "--variable",
        "-var",
        choices=["x", "t", "q"],
        default="x",
        help="Output variable (x=tl-tensor, t=standard, q=SnaPPy)",
    )
    parser.add_argument(
        "--latex",
        "-l",
        action="store_true",
        help="Output as LaTeX",
    )
    parser.add_argument(
        "--hybrid",
        "-H",
        action="store_true",
        help="Use hybrid mode (auto-select fastest method)",
    )

    args = parser.parse_args()

    from .jones import compute_jones, compute_jones_hybrid
    from .latex import jones_to_latex

    braid = args.braid

    if args.hybrid:
        result = compute_jones_hybrid(braid)
        print(f"Method: {result['method']} ({result['n_strands']} strands)")
        print(f"Result: {result['result']}")
    else:
        jones = compute_jones(braid)

        if args.latex:
            output = jones_to_latex(jones, variable=args.variable)
        else:
            output = str(jones)

        print(output)


if __name__ == "__main__":
    # Allow running as python -m tl_examples.cli
    if len(sys.argv) > 1 and sys.argv[1] == "identify":
        sys.argv = sys.argv[1:]
        identify_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "jones":
        sys.argv = sys.argv[1:]
        jones_main()
    else:
        print("Usage: python -m tl_examples.cli [identify|jones] ...")
