#!/usr/bin/env python3
"""
LaTeX formatting utilities for braids and Jones polynomials.

Provides functions to convert braid words and Jones polynomials
to nicely formatted LaTeX strings.
"""

from sympy import Symbol, Rational, latex, Add, Mul, Pow, Integer


def braid_to_latex(word: list[int], compact: bool = False) -> str:
    """
    Convert a braid word to LaTeX notation.

    Args:
        word: List of generators (positive for σ_i, negative for σ_i^{-1})
        compact: If True, group consecutive identical generators as powers

    Returns:
        LaTeX string representation

    Examples:
        >>> braid_to_latex([1, 2, -1, -1])
        '\\sigma_1 \\sigma_2 \\sigma_1^{-1} \\sigma_1^{-1}'

        >>> braid_to_latex([1, 2, -1, -1], compact=True)
        '\\sigma_1 \\sigma_2 \\sigma_1^{-2}'
    """
    if not word:
        return "1"

    if compact:
        # Group consecutive identical generators
        groups = []
        current = word[0]
        count = 1
        for g in word[1:]:
            if g == current:
                count += 1
            else:
                groups.append((current, count))
                current = g
                count = 1
        groups.append((current, count))

        parts = []
        for g, count in groups:
            i = abs(g)
            if g > 0:
                if count == 1:
                    parts.append(f"\\sigma_{i}")
                else:
                    parts.append(f"\\sigma_{i}^{{{count}}}")
            else:
                if count == 1:
                    parts.append(f"\\sigma_{i}^{{-1}}")
                else:
                    parts.append(f"\\sigma_{i}^{{{-count}}}")
        return " ".join(parts)
    else:
        parts = []
        for g in word:
            i = abs(g)
            if g > 0:
                parts.append(f"\\sigma_{i}")
            else:
                parts.append(f"\\sigma_{i}^{{-1}}")
        return " ".join(parts)


def jones_to_latex(jones: list[tuple[int, int]], variable: str = "t") -> str:
    """
    Convert a Jones polynomial to LaTeX using sympy.

    The Jones polynomial from tl-tensor is given as a list of (coefficient, exponent)
    pairs, where the variable is x and t = x^4.

    Args:
        jones: List of (coefficient, exponent) tuples
        variable: Variable name to use ('t' for standard, 'x' for tl-tensor native)

    Returns:
        LaTeX string representation

    Examples:
        >>> jones_to_latex([(1, -8), (-1, -4), (1, 0)])  # Figure-8 in x
        't^{-2} - t^{-1} + 1'
    """
    if not jones:
        return "0"

    if variable == "t":
        # Convert from x (where t = x^4) to t
        t = Symbol('t')
        terms = []
        for coeff, exp in jones:
            if exp % 4 != 0:
                # Exponent not divisible by 4, use fractional powers
                terms.append(coeff * t ** Rational(exp, 4))
            else:
                terms.append(coeff * t ** (exp // 4))
        poly = Add(*terms)
    else:
        # Use x directly
        x = Symbol(variable)
        terms = []
        for coeff, exp in jones:
            terms.append(coeff * x ** exp)
        poly = Add(*terms)

    return latex(poly)


def jones_to_sympy(jones: list[tuple[int, int]], variable: str = "t"):
    """
    Convert a Jones polynomial to a sympy expression.

    Args:
        jones: List of (coefficient, exponent) tuples
        variable: Variable name to use ('t' for standard, 'x' for tl-tensor native)

    Returns:
        Sympy expression
    """
    if not jones:
        return Integer(0)

    var = Symbol(variable)

    if variable == "t":
        # Convert from x (where t = x^4) to t
        terms = []
        for coeff, exp in jones:
            if exp % 4 != 0:
                terms.append(coeff * var ** Rational(exp, 4))
            else:
                terms.append(coeff * var ** (exp // 4))
    else:
        terms = [coeff * var ** exp for coeff, exp in jones]

    return Add(*terms)


def format_result(word: list[int], jones: list[tuple[int, int]],
                  n_strands: int = None, compact_braid: bool = True) -> str:
    """
    Format a complete result with braid word and Jones polynomial in LaTeX.

    Args:
        word: Braid word
        jones: Jones polynomial as (coeff, exp) pairs
        n_strands: Number of strands (optional, for display)
        compact_braid: Use compact notation for braid

    Returns:
        Multi-line LaTeX string suitable for display
    """
    lines = []

    if n_strands:
        lines.append(f"% Braid on {n_strands} strands, {len(word)} crossings")

    lines.append(f"\\beta = {braid_to_latex(word, compact=compact_braid)}")
    lines.append(f"V_{{\\hat{{\\beta}}}}(t) = {jones_to_latex(jones, 't')}")

    return "\n".join(lines)


# Command-line interface
if __name__ == '__main__':
    import argparse
    import sys

    # Add parent for imports if needed
    sys.path.insert(0, '.')

    parser = argparse.ArgumentParser(description='Format braids and Jones polynomials as LaTeX')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--braid', type=str, help='Braid word as comma-separated integers')
    parser.add_argument('--compact', action='store_true', help='Use compact braid notation')
    args = parser.parse_args()

    if args.demo:
        print("=== LaTeX Formatting Demo ===\n")

        # Example: Figure-8 knot
        word = [1, -2, 1, -2]
        jones = [(1, -8), (-1, -4), (1, 0), (-1, 4), (1, 8)]

        print("Figure-8 knot (K4a1):")
        print(f"  Braid word: {word}")
        print(f"  LaTeX: {braid_to_latex(word)}")
        print(f"  Compact: {braid_to_latex(word, compact=True)}")
        print(f"  Jones (x): {jones_to_latex(jones, 'x')}")
        print(f"  Jones (t): {jones_to_latex(jones, 't')}")
        print()

        # Example: Trefoil
        word = [1, 1, 1]
        jones = [(-1, -16), (1, -12), (1, -4)]

        print("Trefoil knot (K3a1):")
        print(f"  Braid word: {word}")
        print(f"  LaTeX: {braid_to_latex(word)}")
        print(f"  Compact: {braid_to_latex(word, compact=True)}")
        print(f"  Jones (t): {jones_to_latex(jones, 't')}")
        print()

        # Example with mixed generators
        word = [1, 2, -1, -1, 2, 3, 3, -2]
        print("Mixed example:")
        print(f"  Braid word: {word}")
        print(f"  LaTeX: {braid_to_latex(word)}")
        print(f"  Compact: {braid_to_latex(word, compact=True)}")
        print()

        print("=== Full Result Format ===\n")
        word = [1, -2, 1, -2]
        jones = [(1, -8), (-1, -4), (1, 0), (-1, 4), (1, 8)]
        print(format_result(word, jones, n_strands=3))

    elif args.braid:
        word = [int(x.strip()) for x in args.braid.split(',')]
        print(f"Braid: {braid_to_latex(word, compact=args.compact)}")
